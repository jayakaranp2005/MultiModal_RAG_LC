"""
Multimodal RAG — CLI entry point.

Run with:  python main.py
"""

from __future__ import annotations

from itertools import chain
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Ensure the project package is importable when executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CHROMA_PERSIST_DIR,
    DOCSTORE_PATH,
    GOOGLE_API_KEY,
    INDEXED_PDFS_PATH,
    validate,
)
from ingestion import partition_pdf_document, separate_elements
from summarizer import summarise_images, summarise_tables, summarise_texts
from utils import load_docstore, load_indexed_pdfs, save_indexed_pdf, truncate
from vectorstore import get_embeddings, get_retriever, get_vectorstore, index_all
from rag_chain import build_rag_chain_with_sources

console = Console()

# ── Global state (initialised once at startup) ─────────────────────────────
_vectorstore = None
_docstore = None
_retriever = None


def _boot() -> None:
    """Load or create persistent stores and build the retriever."""
    global _vectorstore, _docstore, _retriever

    console.print("[dim]Loading stores …[/]")
    embeddings = get_embeddings()
    _vectorstore = get_vectorstore(embeddings)
    _docstore = load_docstore(DOCSTORE_PATH)
    _retriever = get_retriever(_vectorstore, _docstore)
    console.print("[green]✓ Stores loaded.[/]")


# ── Menu actions ────────────────────────────────────────────────────────────

def _upload_pdf() -> None:
    """Ingest a PDF: partition → summarise → index."""
    global _vectorstore, _docstore, _retriever

    pdf_path = Prompt.ask("[bold]Enter PDF file path[/]").strip().strip('"').strip("'")
    if not os.path.isfile(pdf_path):
        console.print(f"[red]File not found:[/] {pdf_path}")
        return

    filename = os.path.basename(pdf_path)
    already_indexed = load_indexed_pdfs(INDEXED_PDFS_PATH)
    if filename in already_indexed:
        re_index = Prompt.ask(
            f"[yellow]{filename}[/] is already indexed. Re-index?",
            choices=["y", "n"],
            default="n",
        )
        if re_index != "y":
            return

    # 1. Partition
    console.rule("[bold cyan]Step 1/3 — Partitioning PDF[/]")
    elements = partition_pdf_document(pdf_path)
    texts, tables, images = separate_elements(elements)

    # 2. Summarise
    console.rule("[bold cyan]Step 2/3 — Summarising elements[/]")
    text_summaries = summarise_texts(texts)
    table_summaries = summarise_tables(tables)
    image_summaries = summarise_images(images)

    # 3. Index
    console.rule("[bold cyan]Step 3/3 — Indexing into vector store[/]")
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

    save_indexed_pdf(filename, INDEXED_PDFS_PATH)
    console.print(
        Panel(
            f"[bold green]✓ {filename} ingested successfully![/]\n"
            f"  texts: {len(texts)}  |  tables: {len(tables)}  |  images: {len(images)}",
            title="Done",
        )
    )


def _ask_question() -> None:
    """Run a RAG query and display the answer + sources."""
    global _retriever

    if _retriever is None:
        console.print("[red]No retriever available — upload a PDF first.[/]")
        return

    question = Prompt.ask("[bold]Your question[/]").strip()
    if not question:
        return

    console.print("[dim]Thinking …[/]")

    chain = build_rag_chain_with_sources(_retriever)
    result = chain.invoke(question)

    console.print()
    console.print(Panel(result["answer"], title="[bold green]Answer[/]", expand=False))

    # Optionally show sources
    show_src = Prompt.ask(
        "Show source context?", choices=["y", "n"], default="n"
    )
    if show_src == "y":
        if result["sources"]:
            for idx, src in enumerate(result["sources"], 1):
                console.print(
                    Panel(
                        truncate(src, 600),
                        title=f"[dim]Source {idx}[/]",
                        expand=False,
                    )
                )
        if result["image_count"]:
            console.print(
                f"[dim]{result['image_count']} image(s) were also used as context.[/]"
            )


def _show_indexed() -> None:
    """List PDFs that have already been indexed."""
    indexed = load_indexed_pdfs(INDEXED_PDFS_PATH)
    if not indexed:
        console.print("[yellow]No PDFs indexed yet.[/]")
        return
    console.print("[bold]Indexed PDFs:[/]")
    for name in sorted(indexed):
        console.print(f"  • {name}")


# ── Main loop ──────────────────────────────────────────────────────────────

_MENU = """
[bold cyan]═══ Multimodal RAG System ═══[/]

  [1] Upload PDF
  [2] Ask a question
  [3] Show indexed PDFs
  [4] Exit
"""


def main() -> None:
    console.print(
        Panel(
            "[bold]Multimodal RAG — PDF Question-Answering System[/]\n"
            "Powered by Google Gemini + ChromaDB + Unstructured",
            expand=False,
        )
    )

    # Validate config
    try:
        validate()
    except EnvironmentError as exc:
        console.print(f"[bold red]{exc}[/]")
        sys.exit(1)

    # Boot persistent stores
    _boot()

    while True:
        console.print(_MENU)
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4"], default="2")

        if choice == "1":
            _upload_pdf()
        elif choice == "2":
            _ask_question()
        elif choice == "3":
            _show_indexed()
        elif choice == "4":
            console.print("[bold]Goodbye![/]")
            break


if __name__ == "__main__":
    main()
