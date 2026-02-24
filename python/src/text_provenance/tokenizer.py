"""Tokenization with stopword removal."""

import re

DEFAULT_STOPWORDS = frozenset([
    "the", "and", "that", "this", "with", "from", "which", "have", "has",
    "had", "been", "were", "was", "are", "for", "not", "but", "its",
    "also", "would", "could", "should", "may", "can", "will", "shall",
    "does", "did", "into", "than", "such", "each", "any", "all", "own",
    "other", "more", "most", "very", "only", "just", "about", "between",
    "through", "under", "over", "after", "before", "when", "where", "while",
    "both", "being", "their", "there", "these", "those", "they", "them",
])

_TOKEN_RE = re.compile(r"[a-z0-9\u00C0-\u024F]{2,}")


def tokenize(text: str, stopwords: frozenset[str] | set[str] = DEFAULT_STOPWORDS) -> set[str]:
    """Tokenize text into a set of lowercase alphanumeric strings (>= 2 chars),
    with stopword removal.

    Args:
        text: Input text to tokenize.
        stopwords: Set of words to filter out. Defaults to DEFAULT_STOPWORDS.

    Returns:
        Set of filtered tokens.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in stopwords}
