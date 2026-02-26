"""
Vector store & retriever — ChromaDB + InMemoryStore + MultiVectorRetriever.

Summaries are embedded for retrieval; original elements are returned as
context to the LLM.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY
from utils import save_docstore

console = Console()

# The metadata key that links a summary document to its original in the
# docstore.  Must be the same everywhere.
_ID_KEY = "doc_id"


# ── Bootstrap / load existing stores ──────────────────────────────────────

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a configured Gemini embedding function."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_vectorstore(
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> Chroma:
    """
    Return a persistent ChromaDB vector store.  Creates the directory if
    it does not yet exist.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name="multimodal_rag",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_retriever(
    vectorstore: Chroma,
    docstore: InMemoryStore,
    *,
    search_kwargs: dict | None = None,
) -> MultiVectorRetriever:
    """Build a ``MultiVectorRetriever`` that links summaries → originals."""
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=_ID_KEY,
        search_kwargs=search_kwargs or {"k": 6},
    )


# ── Indexing helpers ────────────────────────────────────────────────────────

def add_texts_to_store(
    vectorstore: Chroma,
    docstore: InMemoryStore,
    summaries: Sequence[str],
    originals: Sequence[str],
    *,
    docstore_path: str,
) -> None:
    """
    Index *summaries* into ChromaDB and store *originals* in the docstore,
    linked by UUID.
    """
    if not summaries:
        return

    doc_ids = [str(uuid.uuid4()) for _ in summaries]

    # Summary documents carry the doc_id so the retriever can look up the
    # original in the docstore.
    summary_docs = [
        Document(page_content=s, metadata={_ID_KEY: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    vectorstore.add_documents(summary_docs)

    # Store originals keyed by the same UUID.
    docstore.mset(list(zip(doc_ids, originals)))

    # Persist docstore to disk after every batch.
    save_docstore(docstore, docstore_path)


def add_images_to_store(
    vectorstore: Chroma,
    docstore: InMemoryStore,
    image_summaries: Sequence[str],
    images_b64: Sequence[str],
    *,
    docstore_path: str,
) -> None:
    """
    Index image *summaries* into ChromaDB and store the original base64
    image strings in the docstore.
    """
    if not image_summaries:
        return

    doc_ids = [str(uuid.uuid4()) for _ in image_summaries]

    summary_docs = [
        Document(page_content=s, metadata={_ID_KEY: doc_ids[i]})
        for i, s in enumerate(image_summaries)
    ]

    vectorstore.add_documents(summary_docs)
    docstore.mset(list(zip(doc_ids, images_b64)))

    save_docstore(docstore, docstore_path)


# ── Convenience: full indexing pipeline ─────────────────────────────────────

def index_all(
    vectorstore: Chroma,
    docstore: InMemoryStore,
    *,
    text_summaries: Sequence[str],
    texts: Sequence[str],
    table_summaries: Sequence[str],
    tables: Sequence[str],
    image_summaries: Sequence[str],
    images_b64: Sequence[str],
    docstore_path: str,
) -> None:
    """Index texts, tables, and images in one call."""
    console.print("[cyan]Indexing text summaries …[/]")
    add_texts_to_store(
        vectorstore, docstore, text_summaries, texts,
        docstore_path=docstore_path,
    )

    console.print("[cyan]Indexing table summaries …[/]")
    add_texts_to_store(
        vectorstore, docstore, table_summaries, tables,
        docstore_path=docstore_path,
    )

    console.print("[cyan]Indexing image summaries …[/]")
    add_images_to_store(
        vectorstore, docstore, image_summaries, images_b64,
        docstore_path=docstore_path,
    )

    console.print("[green]✓ All elements indexed.[/]")
