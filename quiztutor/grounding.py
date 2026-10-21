"""Grounding check: does each question's quoted evidence really occur in the lecture?

LLMs can produce plausible questions that the source text does not support.
Every generated question must quote the sentence it is based on; we verify that
quote against the lecture with a fuzzy match and flag questions that fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .text import normalise, split_sentences


@dataclass
class GroundingResult:
    grounded: bool
    score: float          # best similarity in [0, 1]
    matched_sentence: str  # closest lecture sentence


def _similarity(a: str, b: str) -> float:
    a, b = normalise(a), normalise(b)
    if not a or not b:
        return 0.0
    if a in b:  # exact containment (quote shorter than the sentence)
        return 1.0
    best = SequenceMatcher(None, a, b).ratio()
    # partial match: compare the quote with every window of the sentence of similar length
    aw, bw = a.split(), b.split()
    if len(aw) < len(bw):
        for size in {len(aw) - 1, len(aw), len(aw) + 1} - {0}:
            for i in range(0, len(bw) - size + 1):
                best = max(best, SequenceMatcher(None, a, " ".join(bw[i:i + size])).ratio())
    return best


def check_quote(quote: str, lecture: str, threshold: float = 0.8) -> GroundingResult:
    """Compare a quote with every lecture sentence (and adjacent pairs, for quotes spanning two)."""
    sents = split_sentences(lecture)
    candidates = sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    best, best_s = 0.0, ""
    for s in candidates:
        score = _similarity(quote, s)
        if score > best:
            best, best_s = score, s
    return GroundingResult(grounded=best >= threshold, score=round(best, 3), matched_sentence=best_s)
