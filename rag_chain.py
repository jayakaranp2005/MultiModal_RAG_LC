"""
RAG query chain — multimodal retrieval + Gemini answer generation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY
from utils import is_base64_image


# ── Document parsing ───────────────────────────────────────────────────────

def parse_docs(docs: list[Any]) -> dict[str, list[str]]:
    """
    Split retrieved documents into ``texts`` and ``images``.

    Documents coming from the docstore are either:
      - plain-text / HTML table strings  →  ``texts``
      - base64-encoded image strings      →  ``images``
    """
    texts: list[str] = []
    images: list[str] = []

    for doc in docs:
        content = doc if isinstance(doc, str) else getattr(doc, "page_content", str(doc))
        if is_base64_image(content):
            images.append(content)
        else:
            texts.append(content)

    return {"texts": texts, "images": images}


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(parsed: dict[str, list[str]], question: str) -> list[HumanMessage]:
    """
    Build a multimodal Gemini prompt.

    Text context and the user question go as plain text; images are inlined
    as base64 ``image_url`` parts.
    """
    content: list[dict[str, Any]] = []

    # System-style instruction
    context_text = "\n\n---\n\n".join(parsed["texts"]) if parsed["texts"] else ""
    instruction = (
        "You are a knowledgeable assistant. Answer the user's question using ONLY "
        "the provided context (text, tables, and images). If the context does not "
        "contain the answer, say so honestly.\n\n"
    )
    if context_text:
        instruction += f"CONTEXT:\n{context_text}\n\n"

    instruction += f"QUESTION:\n{question}"

    content.append({"type": "text", "text": instruction})

    # Inline images
    for img_b64 in parsed.get("images", []):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    return [HumanMessage(content=content)]


# ── Chain construction ─────────────────────────────────────────────────────

def build_rag_chain(retriever: Any):
    """
    Return a LangChain ``Runnable`` chain:

        question → retriever → parse → prompt → Gemini → answer string
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    chain = (
        {
            "docs": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(
            lambda x: build_prompt(x["docs"], x["question"])
        )
        | llm
        | StrOutputParser()
    )

    return chain


def build_rag_chain_with_sources(retriever: Any):
    """
    Return a chain variant that yields **both** the answer and the raw
    retrieved source documents (useful for showing page numbers or
    source context to the user).
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    def _answer_and_sources(inputs: Any) -> dict[str, Any]:
        question = inputs if isinstance(inputs, str) else inputs["question"]
        # Retrieve
        raw_docs = retriever.invoke(question)
        parsed = parse_docs(raw_docs)
        prompt = build_prompt(parsed, question)
        response = llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": parsed["texts"],  # text sources for display
            "image_count": len(parsed["images"]),
        }

    return RunnableLambda(_answer_and_sources)
