"""Tests for text_provenance using shared fixtures."""

import json
from pathlib import Path

import pytest

from text_provenance import match, token_containment, sentence_containment, char_ngram_jaccard, quote_bonus, lcs_ratio, tokenize

FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "test-cases.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text())


class TestMatchWithFixtures:
    """Test match() against shared fixture test cases."""

    @pytest.mark.parametrize(
        "tc",
        [tc for tc in FIXTURES if tc["expected_top_index"] is not None],
        ids=[tc["name"] for tc in FIXTURES if tc["expected_top_index"] is not None],
    )
    def test_expected_match(self, tc):
        results = match(tc["query"], tc["candidates"])
        assert len(results) > 0, f"No results for {tc['name']}"
        assert results[0].index == tc["expected_top_index"], (
            f"{tc['name']}: expected index {tc['expected_top_index']}, got {results[0].index}"
        )

    @pytest.mark.parametrize(
        "tc",
        [tc for tc in FIXTURES if tc["expected_top_index"] is None],
        ids=[tc["name"] for tc in FIXTURES if tc["expected_top_index"] is None],
    )
    def test_null_match(self, tc):
        # For null expected_top_index, either no results or no dominant winner
        results = match(tc["query"], tc["candidates"])
        # Just verify it doesn't crash; these test ambiguous/no-match cases


class TestMatchOptions:
    def test_top_k(self):
        query = "The court held that the defendant was liable."
        candidates = [
            "The court determined that the defendant bore liability for the damages caused.",
            "Weather forecasts predict rain throughout the weekend across the region.",
            "The appellant challenged the lower court's finding of liability.",
        ]
        results = match(query, candidates, top_k=1)
        assert len(results) <= 1

    def test_threshold(self):
        query = "The court held that the defendant was liable."
        candidates = [
            "The court determined that the defendant bore liability for the damages caused.",
        ]
        results = match(query, candidates, threshold=0.99)
        assert len(results) == 0

    def test_custom_weights(self):
        query = "The court held that the defendant was liable."
        candidates = [
            "The court determined that the defendant bore liability for the damages caused.",
        ]
        results = match(query, candidates, weights={"token_containment": 1.0, "char_ngram_jaccard": 0.0, "quote_bonus": 0.0})
        assert len(results) > 0

    def test_empty_query(self):
        assert match("", ["candidate"]) == []

    def test_empty_candidates(self):
        assert match("query", []) == []

    def test_metric_breakdown(self):
        results = match("the court held", ["the court determined the matter"])
        assert len(results) > 0
        m = results[0].metrics
        assert isinstance(m.token_containment, float)
        assert isinstance(m.char_ngram_jaccard, float)
        assert isinstance(m.quote_bonus, float)
        assert isinstance(m.sentence_containment, float)
        assert isinstance(m.lcs_ratio, float)


class TestIndividualMetrics:
    def test_token_containment_identical(self):
        assert token_containment("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)

    def test_token_containment_partial(self):
        score = token_containment("cat sat mat", "the cat was sleeping on the mat yesterday")
        assert score == pytest.approx(2 / 3, abs=0.1)

    def test_sentence_containment(self):
        score = sentence_containment(
            "long query about cats dogs birds and many other animals",
            "cats dogs",
        )
        assert score == pytest.approx(1.0)

    def test_char_ngram_jaccard_identical(self):
        assert char_ngram_jaccard("hello world", "hello world") == pytest.approx(1.0)

    def test_char_ngram_jaccard_zero(self):
        assert char_ngram_jaccard("abc", "xyz") == 0

    def test_quote_bonus_match(self):
        assert quote_bonus(
            'He said "compliance with treaty obligations is required" in the ruling.',
            "compliance with treaty obligations is required by all member states",
        ) == 1.0

    def test_quote_bonus_ellipsis(self):
        assert quote_bonus(
            'The text states "all parties must reduce emissions... in accordance with their capabilities" as a binding rule.',
            "all parties must reduce emissions of greenhouse gases in a manner that reflects equity and in accordance with their capabilities and national circumstances",
        ) == 1.0

    def test_quote_bonus_no_match(self):
        assert quote_bonus("no quotes here", "nothing to find") == 0.0

    def test_lcs_ratio_identical(self):
        assert lcs_ratio("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_lcs_ratio_empty(self):
        assert lcs_ratio("", "hello") == 0.0


class TestTokenize:
    def test_stopword_removal(self):
        tokens = tokenize("The Quick Brown Fox")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens

    def test_empty_string(self):
        assert len(tokenize("")) == 0
