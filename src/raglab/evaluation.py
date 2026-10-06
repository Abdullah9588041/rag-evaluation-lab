"""Retrieval metrics and statistical comparison.

Metrics (binary relevance, per query):
  recall@k, MRR@k, nDCG@k -- standard definitions, see docs/math_notes.md.

Comparison:
  Paired bootstrap confidence intervals for the mean difference of a metric
  between two strategies (resample queries with replacement), plus a two-sided
  bootstrap p-value. Queries are the experimental units, so resampling queries
  is the right level.
"""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & positives) / len(positives)


def mrr_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in positives:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    return sum(
        (1.0 / math.log2(rank + 1)) if cid in positives else 0.0
        for rank, cid in enumerate(ranked_ids[:k], start=1)
    )


def ideal_dcg(n_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_relevant, k) + 1))


def ndcg_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    ideal = ideal_dcg(len(positives), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, positives, k) / ideal


def per_query_metrics(
    ranked_ids: list[str], positives: set[str], k_values: tuple[int, ...] = (1, 5, 10, 20)
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranked_ids, positives, k)
        out[f"mrr@{k}"] = mrr_at_k(ranked_ids, positives, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, positives, k)
    return out


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap CI for mean(a) - mean(b).

    Returns dict with mean_diff, ci_lo, ci_hi, p_value (two-sided).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    alpha = 1.0 - ci
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    mean_diff = float(a.mean() - b.mean())
    # two-sided p: fraction of bootstrap diffs at least as extreme, mirrored
    p_value = float(2.0 * min(np.mean(diffs >= 0), np.mean(diffs <= 0)))
    return {"mean_diff": mean_diff, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": min(p_value, 1.0), "n_boot": n_boot}


def bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI for the mean of x (resample queries with replacement)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, n)].mean()
    alpha = 1.0 - ci
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
