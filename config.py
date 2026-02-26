"""
Configuration module — loads environment variables and defines constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# ── API keys ────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")
)
DOCSTORE_PATH: str = os.getenv(
    "DOCSTORE_PATH", str(PROJECT_ROOT / "docstore.pkl")
)
OUTPUT_PATH: str = os.getenv(
    "OUTPUT_PATH", str(PROJECT_ROOT / "content")
)
INDEXED_PDFS_PATH: str = str(PROJECT_ROOT / "indexed_pdfs.json")

# ── Model names ─────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.0-flash"
EMBEDDING_MODEL: str = "models/gemini-embedding-001"

# ── Summarisation settings ──────────────────────────────────────────────────
SUMMARISE_TEMPERATURE: float = 0.3
SUMMARISE_CONCURRENCY: int = 3

# ── Ingestion / chunking defaults ───────────────────────────────────────────
CHUNK_MAX_CHARS: int = 10_000
CHUNK_COMBINE_UNDER: int = 2_000
CHUNK_NEW_AFTER: int = 6_000


def validate() -> None:
    """Raise early if critical config is missing."""
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Create a .env file in the multimodal_rag/ directory with:\n"
            "  GOOGLE_API_KEY=your_key_here"
        )
