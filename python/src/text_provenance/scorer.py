"""Core scoring logic: match a query against candidate sentences."""

from dataclasses import dataclass
from typing import Any
from .metrics import token_containment, sentence_containment, char_ngram_jaccard, quote_bonus, lcs_ratio


DEFAULT_WEIGHTS = {
    "token_containment": 0.40,
    "char_ngram_jaccard": 0.50,
    "quote_bonus": 0.10,
    "sentence_containment": 0.00,
    "lcs_ratio": 0.00,
}

DEFAULT_THRESHOLD = 0.04
DEFAULT_TOP_K = 3


@dataclass
class MetricScores:
    """Per-metric score breakdown."""
    token_containment: float
    char_ngram_jaccard: float
    quote_bonus: float
    sentence_containment: float
    lcs_ratio: float


@dataclass
class MatchResult:
    """A single match result with score and per-metric breakdown."""
    sentence: str
    index: int
    score: float
    metrics: MetricScores


def match(
    query: str,
    candidates: list[str],
    *,
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[MatchResult]:
    """Find the candidate sentences that best match a query text.

    Scores each candidate using a weighted combination of lightweight text
    similarity metrics and returns the top matches above a threshold.

    Args:
        query: The text to trace back to its source.
        candidates: List of candidate source sentences.
        weights: Metric weights dict. Keys: token_containment, char_ngram_jaccard,
            quote_bonus, sentence_containment, lcs_ratio. Defaults to optimized weights.
        threshold: Minimum score threshold. Default: 0.04.
        top_k: Maximum number of results. Default: 3.

    Returns:
        Ranked list of MatchResult with scores and per-metric breakdowns.
    """
    if not query or not candidates:
        return []

    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    scored: list[MatchResult] = []
    for index, candidate in enumerate(candidates):
        metrics = MetricScores(
            token_containment=token_containment(query, candidate),
            char_ngram_jaccard=char_ngram_jaccard(query, candidate),
            quote_bonus=quote_bonus(query, candidate),
            sentence_containment=sentence_containment(query, candidate),
            lcs_ratio=lcs_ratio(query, candidate),
        )

        score = (
            w["token_containment"] * metrics.token_containment
            + w["char_ngram_jaccard"] * metrics.char_ngram_jaccard
            + w["quote_bonus"] * metrics.quote_bonus
            + w["sentence_containment"] * metrics.sentence_containment
            + w["lcs_ratio"] * metrics.lcs_ratio
        )

        scored.append(MatchResult(
            sentence=candidate,
            index=index,
            score=score,
            metrics=metrics,
        ))

    scored.sort(key=lambda r: -r.score)

    return [r for r in scored[:top_k] if r.score >= threshold]
