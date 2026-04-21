"""
FastAPI backend for the multimodal RAG pipeline.

Key architecture note:
- Summaries are embedded and stored in Chroma for retrieval.
- Original content (full text/table HTML/base64 images) is stored in docstore.pkl.
- Retrieval uses doc_id metadata to map summary hits back to originals.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import DOCSTORE_PATH, INDEXED_PDFS_PATH, validate
from ingestion import partition_pdf_document, separate_elements
from rag_chain import build_rag_chain_with_sources
from summarizer import summarise_images, summarise_tables, summarise_texts
from utils import load_docstore, load_indexed_pdfs, save_indexed_pdf, truncate
from vectorstore import get_embeddings, get_retriever, get_vectorstore, index_all

logger = logging.getLogger("multimodal_rag.api")
logging.basicConfig(level=logging.INFO)

# Global state initialized once during startup.
_vectorstore = None
_docstore = None
_retriever = None


class HealthResponse(BaseModel):
    ok: bool
    ready: bool


class IndexedResponse(BaseModel):
    indexed: list[str]


class UploadResponse(BaseModel):
    filename: str
    texts: int = 0
    tables: int = 0
    images: int = 0
    status: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] | None = None
    image_count: int


def _is_ready() -> bool:
    return _vectorstore is not None and _docstore is not None and _retriever is not None


def _boot_stores() -> None:
    """Load or create persistent stores and build retriever once."""
    global _vectorstore, _docstore, _retriever

    logger.info("Loading vectorstore and docstore")
    embeddings = get_embeddings()
    _vectorstore = get_vectorstore(embeddings)
    _docstore = load_docstore(DOCSTORE_PATH)
    _retriever = get_retriever(_vectorstore, _docstore)
    logger.info("Stores loaded successfully")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Validate config and initialize retriever stack at startup."""
    try:
        validate()
        await run_in_threadpool(_boot_stores)
    except Exception:
        # Keep app running so /health can report not-ready and surface startup issues.
        logger.exception("Startup initialization failed")
    yield


app = FastAPI(title="Multimodal RAG API", lifespan=lifespan)

# Permissive CORS for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, ready=_is_ready())


@app.get("/indexed", response_model=IndexedResponse)
async def indexed() -> IndexedResponse:
    try:
        indexed_files = sorted(load_indexed_pdfs(INDEXED_PDFS_PATH))
        return IndexedResponse(indexed=indexed_files)
    except Exception as exc:
        logger.exception("Failed to read indexed PDF registry")
        raise HTTPException(status_code=500, detail=f"Failed to load indexed PDFs: {exc}")


def _run_ingestion_pipeline(pdf_path: str) -> tuple[int, int, int]:
    """Partition, summarize, and index one PDF into vectorstore + docstore."""
    if not _is_ready():
        raise RuntimeError("Stores are not initialized")

    elements = partition_pdf_document(pdf_path)
    texts, tables, images = separate_elements(elements)

    text_summaries = summarise_texts(texts)
    table_summaries = summarise_tables(tables)
    image_summaries = summarise_images(images)

    # Critical retrieval design:
    # - summaries -> Chroma vectors
    # - originals -> docstore.pkl
    # MultiVectorRetriever uses doc_id to map summary hit -> original content.
    index_all(
        _vectorstore,
        _docstore,
        text_summaries=text_summaries,
        texts=list(texts),
        table_summaries=table_summaries,
        tables=list(tables),
        image_summaries=image_summaries,
        images_b64=list(images),
        docstore_path=DOCSTORE_PATH,
    )

    return len(texts), len(tables), len(images)


@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    reindex: bool = Query(False, description="Reindex if filename already exists"),
) -> UploadResponse:
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Service not ready. Check /health")

    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        indexed_files = load_indexed_pdfs(INDEXED_PDFS_PATH)
    except Exception as exc:
        logger.exception("Failed to read indexed PDF registry")
        raise HTTPException(status_code=500, detail=f"Failed to check indexed PDFs: {exc}")

    if filename in indexed_files and not reindex:
        return UploadResponse(filename=filename, status="already_indexed")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        texts_count, tables_count, images_count = await run_in_threadpool(
            _run_ingestion_pipeline,
            str(temp_path),
        )

        save_indexed_pdf(filename, INDEXED_PDFS_PATH)

        return UploadResponse(
            filename=filename,
            texts=texts_count,
            tables=tables_count,
            images=images_count,
            status="indexed",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion/indexing failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}")
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not delete temp file: %s", temp_path)


def _run_rag(question: str) -> dict:
    """Invoke RAG chain and normalize response shape."""
    if not _is_ready():
        raise RuntimeError("Retriever is not initialized")

    chain = build_rag_chain_with_sources(_retriever)
    result = chain.invoke(question)

    answer = str(result.get("answer", ""))
    raw_sources = result.get("sources") or []
    sources = [truncate(str(src), 600) for src in raw_sources]
    image_count = int(result.get("image_count", 0))

    payload: dict = {
        "answer": answer,
        "image_count": image_count,
    }
    if sources:
        payload["sources"] = sources
    return payload


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Service not ready. Check /health")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must be non-empty")

    try:
        response_payload = await run_in_threadpool(_run_rag, question)
        return AskResponse(**response_payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}")
