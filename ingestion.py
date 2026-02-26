"""
PDF ingestion — partition PDFs with ``unstructured`` and separate elements.
"""

from __future__ import annotations

import base64
from typing import Any

from rich.console import Console
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement, Table

from config import (
    CHUNK_COMBINE_UNDER,
    CHUNK_MAX_CHARS,
    CHUNK_NEW_AFTER,
    OUTPUT_PATH,
)

console = Console()


# ── Main partitioning entry-point ───────────────────────────────────────────

def partition_pdf_document(pdf_path: str) -> list[Any]:
    """
    Partition a PDF using the ``hi_res`` strategy.

    Returns the raw list of unstructured ``Element`` objects.
    """
    console.print(f"[bold cyan]Partitioning:[/] {pdf_path}")
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=CHUNK_MAX_CHARS,
            combine_text_under_n_chars=CHUNK_COMBINE_UNDER,
            new_after_n_chars=CHUNK_NEW_AFTER,
        )
        console.print(
            f"[green]✓[/] Partitioned into [bold]{len(elements)}[/] elements"
        )
        return elements
    except Exception as exc:
        _handle_partition_error(exc)
        raise


# ── Element separation ──────────────────────────────────────────────────────

def separate_elements(
    elements: list[Any],
) -> tuple[list[str], list[str], list[str]]:
    """
    Separate partitioned elements into three lists:
      - **texts**: string content of ``CompositeElement``s
      - **tables**: HTML string content of ``Table``s
      - **images**: base64-encoded image strings extracted from metadata

    Returns ``(texts, tables, images)``.
    """
    texts: list[str] = []
    tables: list[str] = []

    for el in elements:
        if isinstance(el, Table):
            tables.append(str(el.metadata.text_as_html) if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html else str(el))
        elif isinstance(el, CompositeElement):
            texts.append(str(el))

    images = get_images_base64(elements)

    console.print(
        f"  [dim]texts={len(texts)}  tables={len(tables)}  images={len(images)}[/]"
    )
    return texts, tables, images


# ── Image extraction helper ────────────────────────────────────────────────

def get_images_base64(chunks: list[Any]) -> list[str]:
    """
    Walk through ``CompositeElement`` chunks and pull out base64 image
    payloads stored in ``metadata.orig_elements``.
    """
    images: list[str] = []

    for chunk in chunks:
        if not isinstance(chunk, CompositeElement):
            continue
        orig_elements = getattr(chunk.metadata, "orig_elements", None)
        if not orig_elements:
            continue
        for orig in orig_elements:
            if hasattr(orig, "metadata"):
                # Check for image_base64 in metadata
                image_b64 = getattr(orig.metadata, "image_base64", None)
                if image_b64 and isinstance(image_b64, str):
                    images.append(image_b64)
                    continue
                # Fallback: check image_payload
                payload = getattr(orig.metadata, "image_payload", None)
                if payload and isinstance(payload, (str, bytes)):
                    if isinstance(payload, bytes):
                        images.append(base64.b64encode(payload).decode("utf-8"))
                    else:
                        images.append(payload)

    return images


# ── Error diagnostics ──────────────────────────────────────────────────────

def _handle_partition_error(exc: Exception) -> None:
    """Print helpful diagnostics for common partition_pdf failures."""
    msg = str(exc).lower()
    if "poppler" in msg or "pdftoppm" in msg:
        console.print(
            "[bold red]Error:[/] poppler is not installed or not on PATH.\n"
            "  • Windows: download from https://github.com/oschwartz10612/poppler-windows/releases\n"
            "    and add the bin/ folder to your PATH.\n"
            "  • macOS:   brew install poppler\n"
            "  • Linux:   sudo apt-get install poppler-utils"
        )
    elif "tesseract" in msg:
        console.print(
            "[bold red]Error:[/] tesseract-ocr is not installed or not on PATH.\n"
            "  • Windows: download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  • macOS:   brew install tesseract\n"
            "  • Linux:   sudo apt-get install tesseract-ocr"
        )
    else:
        console.print(f"[bold red]Partition failed:[/] {exc}")
