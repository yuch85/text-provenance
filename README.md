# text-provenance

Identify which source sentence a piece of text most likely originated from, using lightweight text similarity metrics. No embeddings or API calls required at runtime — just fast, deterministic string matching that works in the browser or on the server.

Designed for RAG (Retrieval-Augmented Generation) systems that need to trace LLM answers back to their source sentences for citation highlighting, but works for any text provenance task.

## Install

```bash
# TypeScript
npm install text-provenance

# Python
pip install text-provenance
```

## Quick Start

### TypeScript

```typescript
import { match } from "text-provenance"

const query = "The court found that commanders who knew about crimes but failed to act could be held responsible."

const candidates = [
  "A superior who has knowledge of offences committed by persons under his command and fails to take necessary measures shall be criminally liable.",
  "The jurisdiction of the court extends to natural persons pursuant to the provisions of the present statute.",
  "Witness testimony shall be given orally unless the court permits a written statement.",
]

const results = match(query, candidates)
// results[0].sentence -> candidates[0]
// results[0].score -> 0.31
// results[0].index -> 0
// results[0].metrics -> { tokenContainment: 0.33, charNgramJaccard: 0.28, ... }
```

### Python

```python
from text_provenance import match

query = "The court found that commanders who knew about crimes but failed to act could be held responsible."

candidates = [
    "A superior who has knowledge of offences committed by persons under his command and fails to take necessary measures shall be criminally liable.",
    "The jurisdiction of the court extends to natural persons pursuant to the provisions of the present statute.",
    "Witness testimony shall be given orally unless the court permits a written statement.",
]

results = match(query, candidates)
# results[0].sentence -> candidates[0]
# results[0].score -> 0.31
# results[0].index -> 0
# results[0].metrics.token_containment -> 0.33
```

## API Reference

### `match(query, candidates, options?)`

Find the candidate sentences that best match a query text.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | — | The text to trace back to its source |
| `candidates` | `string[]` | — | Array of candidate source sentences |
| `options.weights` | `Weights` | See below | Metric weights (should sum to 1.0) |
| `options.threshold` | `number` | `0.04` | Minimum score to include in results |
| `options.topK` | `number` | `3` | Maximum number of results |

**Returns:** Array of `MatchResult` objects, sorted by score descending.

```typescript
interface MatchResult {
  sentence: string      // The matched candidate text
  index: number         // Index in the original candidates array
  score: number         // Combined weighted score
  metrics: MetricScores // Per-metric breakdown
}
```

### Default Weights

| Metric | Weight | Role |
|--------|--------|------|
| `charNgramJaccard` | **0.50** | Strongest signal — captures character-level overlap including morphological variants |
| `tokenContainment` | **0.40** | What fraction of the query's vocabulary appears in the candidate |
| `quoteBonus` | **0.10** | Tie-breaker when the query directly quotes the source |
| `sentenceContainment` | **0.00** | Available but zeroed — redundant with tokenContainment |
| `lcsRatio` | **0.00** | Available but zeroed — captured by the combination of other metrics |

These weights were optimized via grid search over 651 weight combinations on 350 real LLM citation pairs from a legal corpus, validated against embedding cosine similarity (Qwen3-Embedding-4B). F1=0.940, accuracy=89.9%, coverage=98.6%.

### Individual Metrics

All metrics are independently importable:

```typescript
import { tokenContainment, charNgramJaccard, quoteBonus, sentenceContainment, lcsRatio } from "text-provenance"
```

```python
from text_provenance import token_containment, char_ngram_jaccard, quote_bonus, sentence_containment, lcs_ratio
```

#### `tokenContainment(query, candidate)` / `token_containment(query, candidate)`

Fraction of the query's tokens found in the candidate. Uses tokenizer with stopword removal. Range: [0, 1].

#### `charNgramJaccard(a, b, n=4)` / `char_ngram_jaccard(a, b, n=4)`

Jaccard similarity over character n-gram sets. Captures morphological overlap (e.g., "criminality" vs "criminal"). Range: [0, 1].

#### `quoteBonus(query, candidate, minLength=15)` / `quote_bonus(query, candidate, min_length=15)`

1.0 if the candidate contains a quoted substring from the query of at least `minLength` characters, else 0.0. Handles ellipsis splitting (`...` and `…`).

#### `sentenceContainment(query, candidate)` / `sentence_containment(query, candidate)`

Fraction of the candidate's tokens found in the query. Reverse direction of tokenContainment. Range: [0, 1].

#### `lcsRatio(a, b)` / `lcs_ratio(a, b)`

Word-level Longest Common Subsequence divided by max word count. Space-optimized two-row DP. Range: [0, 1].

### `tokenize(text, stopwords?)` / `tokenize(text, stopwords=DEFAULT_STOPWORDS)`

Lowercase, extract alphanumeric strings >= 2 characters, filter stopwords. Returns a `Set<string>` (TypeScript) or `set[str]` (Python).

### `optimize_weights()` (Python only)

Grid search for optimal metric weights using an embedding model as an independent validation function.

```python
from text_provenance.optimizer import optimize_weights

results = optimize_weights(
    pairs=[
        {"query": "LLM said this", "candidates": ["source A", "source B", "source C"]},
        # ... more pairs
    ],
    embedding_endpoint="http://localhost:8080/v1/embeddings",
    embedding_model="your-model-name",
    weight_step=0.10,
    max_weight=0.50,
    thresholds=(0.04, 0.21, 0.01),
    batch_size=200,
)

print(results.best_weights)    # {'token_containment': 0.4, 'char_ngram_jaccard': 0.5, ...}
print(results.best_threshold)  # 0.04
print(results.f1)              # 0.94
print(results.accuracy)        # 0.899
print(results.coverage)        # 0.986
```

Requires the optimizer extras: `pip install text-provenance[optimizer]`

## Tuning for Your Domain

The default weights were tuned on international law text (ICTY/ICJ tribunal opinions). Other domains with different paraphrasing patterns may benefit from different weights. To tune:

1. Collect query-candidates pairs from your domain (50+ pairs recommended, 350+ for reliable results)
2. Run an OpenAI-compatible embedding model (any model works — it's only used for validation, not production)
3. Use `optimize_weights()` to search the weight space
4. Pass the resulting weights to `match()` via the `weights` option

## How It Works

Given a query and candidates, each candidate is scored using a weighted sum of metrics:

```
score = w1 * tokenContainment + w2 * charNgramJaccard + w3 * quoteBonus
      + w4 * sentenceContainment + w5 * lcsRatio
```

1. **Character N-gram Jaccard** (default: 0.50) — Builds sets of 4-character substrings from both texts, computes Jaccard similarity. This captures morphological overlap that word-level metrics miss (e.g., "obligations" matches "obligated").

2. **Token Containment** (default: 0.40) — Tokenizes both texts (lowercase alphanumeric, stopwords removed), computes what fraction of the query's tokens appear in the candidate. High values mean the candidate "explains" most of the query's distinctive vocabulary.

3. **Quote Bonus** (default: 0.10) — Checks if any quoted substring from the query (>= 15 chars) appears in the candidate. Handles LLM-style ellipsis truncation. Acts as a tie-breaker when direct quotes are present.

Candidates are ranked by score, and the top-K above a threshold are returned.

## License

MIT
