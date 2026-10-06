"""Figures for the analysis: recall@k curves, metric bars with CIs, latency trade-offs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_recall_curves(results: dict[str, dict], k_values: list[int], path: str) -> None:
    """results[strategy] -> {'recall@k': mean ...}; line per strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, agg in sorted(results.items()):
        ys = [agg[f"recall@{k}"] for k in k_values]
        ax.plot(k_values, ys, marker="o", label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("Mean recall@k")
    ax.set_title("Recall@k by retrieval strategy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ndcg_bars(results: dict[str, dict], ci: dict[str, tuple[float, float]], path: str) -> None:
    """Mean nDCG@10 per strategy with 95% bootstrap CI error bars (vs. baseline)."""
    names = sorted(results)
    means = [results[n]["ndcg@10"] for n in names]
    lo = [results[n]["ndcg@10"] - ci[n][0] for n in names]
    hi = [ci[n][1] - results[n]["ndcg@10"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=[lo, hi], capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean nDCG@10")
    ax.set_title("nDCG@10 with 95% bootstrap CIs (per-query)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency_tradeoff(results: dict[str, dict], path: str) -> None:
    """Scatter: mean latency per query (ms) vs mean nDCG@10."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, agg in sorted(results.items()):
        ax.scatter(agg["latency_ms"], agg["ndcg@10"], s=80, label=name)
        ax.annotate(name, (agg["latency_ms"], agg["ndcg@10"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean latency per query (ms, log scale)")
    ax.set_ylabel("Mean nDCG@10")
    ax.set_xscale("log")
    ax.set_title("Quality vs. latency trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_win_loss(deltas: np.ndarray, label_a: str, label_b: str, path: str) -> None:
    """Histogram of per-query nDCG@10 differences (A - B)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=30, color="slateblue", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Per-query nDCG@10 difference ({label_a} − {label_b})")
    ax.set_ylabel("Number of queries")
    ax.set_title(f"Per-query win/loss: {label_a} vs {label_b}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
