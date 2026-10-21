"""Loading lecture material and splitting it into sentences."""

from __future__ import annotations

import re
from pathlib import Path

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def load_text(path: str | Path) -> str:
    """Read a .txt, .md or .pdf file and return its text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace/punctuation for fuzzy comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple rule-based splitter)."""
    out: list[str] = []
    for line in text.splitlines():  # headings and paragraphs are separate units
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return out


def truncate(text: str, max_chars: int = 12000) -> str:
    """Keep the prompt within a small model's context window."""
    return text if len(text) <= max_chars else text[:max_chars] + " …"
