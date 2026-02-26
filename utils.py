"""
Utility helpers — base64 handling, display formatting, persistence helpers.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any
from langchain_classic.storage import InMemoryStore


# ── Base64 helpers ──────────────────────────────────────────────────────────

def is_base64_image(text: str) -> bool:
    """Return True if *text* looks like a base64-encoded image string."""
    if not isinstance(text, str):
        return False
    # Quick heuristic: long string with only base64 chars
    if len(text) < 200:
        return False
    b64_pattern = re.compile(r"^[A-Za-z0-9+/\n\r]+=*$")
    # Check a prefix to avoid scanning megabytes
    return bool(b64_pattern.match(text[:500]))


def encode_image_to_base64(image_path: str) -> str:
    """Read a local image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64(b64_string: str) -> bytes:
    """Decode a base64 string to raw bytes."""
    return base64.b64decode(b64_string)


# ── Docstore persistence ───────────────────────────────────────────────────

def save_docstore(store: InMemoryStore, path: str) -> None:
    """Pickle the InMemoryStore's internal dict to *path*."""
    internal: dict[str, Any] = {}
    for key in store.yield_keys():
        vals = store.mget([key])
        if vals and vals[0] is not None:
            internal[key] = vals[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(internal, f)


def load_docstore(path: str) -> InMemoryStore:
    """Load a pickled InMemoryStore from *path*, or return a fresh one."""
    store = InMemoryStore()
    if os.path.exists(path):
        with open(path, "rb") as f:
            internal: dict[str, Any] = pickle.load(f)  # noqa: S301
        if internal:
            store.mset(list(internal.items()))
    return store


# ── Indexed-PDF registry ───────────────────────────────────────────────────

def load_indexed_pdfs(json_path: str) -> set[str]:
    """Return set of already-indexed PDF filenames."""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_indexed_pdf(filename: str, json_path: str) -> None:
    """Append *filename* to the indexed-PDF registry."""
    existing = load_indexed_pdfs(json_path)
    existing.add(filename)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted(existing), f, indent=2)


# ── Display helpers ─────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 300) -> str:
    """Return *text* truncated to *max_len* chars with an ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + " …"
