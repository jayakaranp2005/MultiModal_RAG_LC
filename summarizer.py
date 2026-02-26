"""
Summariser — generate concise summaries for text, tables, and images
using Google Gemini via LangChain.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.progress import track

from config import GEMINI_MODEL, GOOGLE_API_KEY, SUMMARISE_CONCURRENCY, SUMMARISE_TEMPERATURE

console = Console()

# ── Prompt templates ────────────────────────────────────────────────────────

_TEXT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "You are a document analysis assistant. "
    "Provide a concise, factual summary of the following text or table element. "
    "Do NOT add any preamble — go straight to the summary.\n\n"
    "{element}"
)

_IMAGE_SUMMARY_INSTRUCTION = (
    "You are a document analysis assistant. "
    "Describe this image in detail. If it contains a chart, diagram, or data visual, "
    "explain the structure, axes, labels, trends, and any key data points. "
    "Be factual and concise."
)


# ── Model factory ──────────────────────────────────────────────────────────

def _llm(temperature: float = SUMMARISE_TEMPERATURE) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )


# ── Text / table summarisation ─────────────────────────────────────────────

def summarise_texts(
    texts: Sequence[str],
    *,
    concurrency: int = SUMMARISE_CONCURRENCY,
) -> list[str]:
    """
    Summarise a list of text chunks (plain text or table HTML).

    Uses ``RunnableParallel``-style batching with *concurrency* to avoid
    rate-limit issues while still being faster than sequential calls.
    """
    if not texts:
        return []

    chain = _TEXT_SUMMARY_PROMPT | _llm() | StrOutputParser()

    console.print(f"[cyan]Summarising {len(texts)} text/table elements …[/]")
    summaries: list[str] = []

    # Process in batches of `concurrency`
    for i in range(0, len(texts), concurrency):
        batch = texts[i : i + concurrency]
        inputs = [{"element": t} for t in batch]
        try:
            results = chain.batch(inputs, config={"max_concurrency": concurrency})
            summaries.extend(results)
        except Exception as exc:
            console.print(f"[yellow]⚠ Batch summarisation error: {exc}[/]")
            # Fallback: use truncated originals
            summaries.extend([t[:500] for t in batch])

        console.print(f"  [dim]Processed {min(i + concurrency, len(texts))}/{len(texts)}[/]")

    return summaries


def summarise_tables(
    tables: Sequence[str],
    *,
    concurrency: int = SUMMARISE_CONCURRENCY,
) -> list[str]:
    """Summarise HTML table elements — delegates to ``summarise_texts``."""
    return summarise_texts(tables, concurrency=concurrency)


# ── Image summarisation ────────────────────────────────────────────────────

def summarise_images(
    images_b64: Sequence[str],
) -> list[str]:
    """
    Summarise a list of base64-encoded images using Gemini's multimodal
    input.  Each image is sent individually to avoid overwhelming the
    context window.
    """
    if not images_b64:
        return []

    llm = _llm(temperature=0.3)
    summaries: list[str] = []

    console.print(f"[cyan]Summarising {len(images_b64)} image(s) …[/]")

    for idx, img_b64 in enumerate(images_b64, 1):
        message = HumanMessage(
            content=[
                {"type": "text", "text": _IMAGE_SUMMARY_INSTRUCTION},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    },
                },
            ]
        )
        try:
            response = llm.invoke([message])
            summaries.append(response.content)
        except Exception as exc:
            console.print(f"[yellow]⚠ Image {idx} summarisation failed: {exc}[/]")
            summaries.append("[Image — summary unavailable]")

        console.print(f"  [dim]Image {idx}/{len(images_b64)} done[/]")

    return summaries
