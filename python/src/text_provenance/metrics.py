"""Individual text similarity metrics."""

import re
from .tokenizer import tokenize, DEFAULT_STOPWORDS


def token_containment(query: str, candidate: str, stopwords: frozenset[str] | set[str] = DEFAULT_STOPWORDS) -> float:
    """Fraction of query's tokens found in candidate.

    Measures how well the candidate "explains" the query.
    """
    query_tokens = tokenize(query, stopwords)
    candidate_tokens = tokenize(candidate, stopwords)
    if not query_tokens:
        return 0.0
    hits = sum(1 for t in query_tokens if t in candidate_tokens)
    return hits / len(query_tokens)


def sentence_containment(query: str, candidate: str, stopwords: frozenset[str] | set[str] = DEFAULT_STOPWORDS) -> float:
    """Fraction of candidate's tokens found in query.

    The reverse direction of token_containment.
    """
    query_tokens = tokenize(query, stopwords)
    candidate_tokens = tokenize(candidate, stopwords)
    if not candidate_tokens:
        return 0.0
    hits = sum(1 for t in candidate_tokens if t in query_tokens)
    return hits / len(candidate_tokens)


def char_ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    """Character n-gram Jaccard similarity.

    |ngrams(a) ∩ ngrams(b)| / |ngrams(a) ∪ ngrams(b)|

    Captures morphological overlap (e.g., "criminality" vs "criminal").

    Args:
        a: First text.
        b: Second text.
        n: N-gram size. Default: 4.
    """
    la = a.lower()
    lb = b.lower()
    if len(la) < n and len(lb) < n:
        return 0.0

    set_a = {la[i:i + n] for i in range(len(la) - n + 1)}
    set_b = {lb[i:i + n] for i in range(len(lb) - n + 1)}

    inter = len(set_a & set_b)
    union = len(set_a | set_b)

    return inter / union if union > 0 else 0.0


def quote_bonus(query: str, candidate: str, min_length: int = 15) -> float:
    """1.0 if candidate contains a quoted substring from query of at least
    min_length characters, else 0.0.

    Handles ellipsis in quotes: "text A... text B" splits on ellipsis and
    checks if all fragments (>= min_length) appear in the candidate.
    """
    quotes = _extract_quotes(query, min_length)
    if not quotes:
        return 0.0
    lower = candidate.lower()
    for q in quotes:
        ql = q.lower()
        if ql in lower:
            return 1.0
        # Handle ellipsis: "text A... text B"
        if "..." in ql or "\u2026" in ql:
            fragments = [f.strip() for f in re.split(r"\.{3}|\u2026", ql)]
            fragments = [f for f in fragments if len(f) >= min_length]
            if len(fragments) >= 2 and all(f in lower for f in fragments):
                return 1.0
    return 0.0


_QUOTE_RE_CACHE: dict[int, re.Pattern[str]] = {}


def _extract_quotes(text: str, min_length: int) -> list[str]:
    """Extract quoted strings (>= min_length chars) from text."""
    if min_length not in _QUOTE_RE_CACHE:
        _QUOTE_RE_CACHE[min_length] = re.compile(
            rf"""["\u201C\u201D\u2018\u2019']([^"\u201C\u201D\u2018\u2019']{{{min_length},}}?)["\u201C\u201D\u2018\u2019']"""
        )
    return [m.group(1).strip() for m in _QUOTE_RE_CACHE[min_length].finditer(text)]


def lcs_ratio(a: str, b: str) -> float:
    """Word-level Longest Common Subsequence ratio.

    LCS length / max(|wordsA|, |wordsB|).
    Uses space-optimized two-row DP.
    """
    if not a or not b:
        return 0.0

    wa = a.lower().split()
    wb = b.lower().split()

    m = len(wa)
    n = len(wb)
    if m == 0 or n == 0:
        return 0.0

    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if wa[i - 1] == wb[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    return prev[n] / max(m, n)
