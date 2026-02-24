"""Grid search weight optimizer with embedding cross-metric validation.

Requires the 'optimizer' optional dependency group:
    pip install text-provenance[optimizer]

This module is not imported by default — only when explicitly used.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

try:
    import numpy as np
    import requests
except ImportError:
    raise ImportError(
        "The optimizer requires numpy and requests. Install with: "
        "pip install text-provenance[optimizer]"
    )

from .metrics import token_containment, sentence_containment, char_ngram_jaccard, quote_bonus, lcs_ratio


@dataclass
class OptimizationResult:
    """Result of a weight optimization run."""
    best_weights: dict[str, float]
    best_threshold: float
    f1: float
    accuracy: float
    coverage: float
    all_results: list[dict]


def _get_embeddings(
    texts: list[str],
    endpoint: str,
    model: str,
    batch_size: int = 200,
) -> list[np.ndarray]:
    """Batch embed texts via an OpenAI-compatible endpoint."""
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        r = requests.post(endpoint, json={"model": model, "input": batch}, timeout=120)
        r.raise_for_status()
        data = r.json()
        for item in data["data"]:
            all_embs.append(np.array(item["embedding"]))
    return all_embs


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _score_sentence(
    query: str,
    sentence: str,
    w_tc: float,
    w_sc: float,
    w_jac: float,
    w_qb: float,
    w_lcs: float,
) -> float:
    """Score a single sentence with given weights."""
    tc = token_containment(query, sentence)
    sc = sentence_containment(query, sentence)
    jac = char_ngram_jaccard(query, sentence)
    qb = quote_bonus(query, sentence)
    lr = lcs_ratio(query, sentence)
    return w_tc * tc + w_sc * sc + w_jac * jac + w_qb * qb + w_lcs * lr


def optimize_weights(
    pairs: list[dict],
    embedding_endpoint: str,
    embedding_model: str,
    *,
    weight_step: float = 0.10,
    max_weight: float = 0.50,
    thresholds: tuple[float, float, float] = (0.04, 0.21, 0.01),
    batch_size: int = 200,
) -> OptimizationResult:
    """Find optimal metric weights using embedding cosine similarity as cross-metric.

    Args:
        pairs: List of dicts with "query" (str) and "candidates" (list[str]).
        embedding_endpoint: URL of an OpenAI-compatible embedding API.
        embedding_model: Model name to pass to the embedding API.
        weight_step: Grid search step size for each weight. Default: 0.10.
        max_weight: Maximum value for any single weight. Default: 0.50.
        thresholds: (min, max, step) for threshold sweep. Default: (0.04, 0.21, 0.01).
        batch_size: Batch size for embedding API calls. Default: 200.

    Returns:
        OptimizationResult with best weights, threshold, and all evaluated combos.
    """
    # Build steps
    steps = []
    v = 0.0
    while v <= max_weight + 1e-9:
        steps.append(round(v, 4))
        v += weight_step

    # Precompute embeddings
    print(f"Collecting unique texts...")
    text_to_idx: dict[str, int] = {}
    unique_texts: list[str] = []
    for pair in pairs:
        for text in [pair["query"]] + pair["candidates"]:
            if text not in text_to_idx:
                text_to_idx[text] = len(unique_texts)
                unique_texts.append(text)

    print(f"Embedding {len(unique_texts)} unique texts...")
    t0 = time.time()
    embeddings = _get_embeddings(unique_texts, embedding_endpoint, embedding_model, batch_size)
    elapsed = time.time() - t0
    print(f"Embedded {len(unique_texts)} texts in {elapsed:.1f}s")

    # Precompute cosines for each pair
    pair_cosines: list[list[float]] = []
    for pair in pairs:
        query_emb = embeddings[text_to_idx[pair["query"]]]
        cosines = [
            _cosine_sim(query_emb, embeddings[text_to_idx[c]])
            for c in pair["candidates"]
        ]
        pair_cosines.append(cosines)

    # Build threshold range
    t_min, t_max, t_step = thresholds
    threshold_values: list[float] = []
    t = t_min
    while t <= t_max + 1e-9:
        threshold_values.append(round(t, 4))
        t += t_step

    # Grid search over 5 weights
    best_f1 = 0.0
    best_params: dict | None = None
    all_results: list[dict] = []
    combos = 0

    for w_tc in steps:
        for w_sc in steps:
            for w_lcs in steps:
                for w_qb in steps:
                    w_jac = round(1.0 - w_tc - w_sc - w_lcs - w_qb, 4)
                    if w_jac < -1e-9 or w_jac > max_weight + 1e-9:
                        continue
                    w_jac = max(0.0, w_jac)
                    combos += 1

                    # Evaluate all pairs
                    eval_results: list[dict] = []
                    for idx, pair in enumerate(pairs):
                        candidates = pair["candidates"]
                        if len(candidates) < 2:
                            continue
                        scores = [
                            _score_sentence(pair["query"], c, w_tc, w_sc, w_jac, w_qb, w_lcs)
                            for c in candidates
                        ]
                        top_idx = max(range(len(scores)), key=lambda i: scores[i])
                        top_score = scores[top_idx]

                        cross_sel = pair_cosines[idx][top_idx]
                        others = [pair_cosines[idx][i] for i in range(len(candidates)) if i != top_idx]
                        median_other = statistics.median(others) if others else 0.0

                        eval_results.append({
                            "prod_score": top_score,
                            "cross_accurate": cross_sel > median_other,
                        })

                    if len(eval_results) < 10:
                        continue

                    for t in threshold_values:
                        above = [r for r in eval_results if r["prod_score"] >= t]
                        if not above:
                            continue
                        acc = sum(1 for r in above if r["cross_accurate"]) / len(above)
                        cov = len(above) / len(eval_results)
                        f1 = 2 * acc * cov / (acc + cov) if (acc + cov) > 0 else 0.0

                        result_entry = {
                            "weights": {
                                "token_containment": w_tc,
                                "sentence_containment": w_sc,
                                "char_ngram_jaccard": w_jac,
                                "quote_bonus": w_qb,
                                "lcs_ratio": w_lcs,
                            },
                            "threshold": t,
                            "f1": f1,
                            "accuracy": acc,
                            "coverage": cov,
                            "n": len(above),
                        }
                        all_results.append(result_entry)

                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = result_entry

    print(f"Searched {combos} weight combinations x {len(threshold_values)} thresholds")

    if best_params:
        print(f"Best: {best_params['weights']}")
        print(f"  threshold={best_params['threshold']:.2f}  F1={best_params['f1']:.3f}  "
              f"accuracy={100*best_params['accuracy']:.1f}%  coverage={100*best_params['coverage']:.1f}%")
        return OptimizationResult(
            best_weights=best_params["weights"],
            best_threshold=best_params["threshold"],
            f1=best_params["f1"],
            accuracy=best_params["accuracy"],
            coverage=best_params["coverage"],
            all_results=all_results,
        )
    else:
        return OptimizationResult(
            best_weights=dict(token_containment=0.40, char_ngram_jaccard=0.50, quote_bonus=0.10,
                              sentence_containment=0.00, lcs_ratio=0.00),
            best_threshold=0.04,
            f1=0.0,
            accuracy=0.0,
            coverage=0.0,
            all_results=all_results,
        )
