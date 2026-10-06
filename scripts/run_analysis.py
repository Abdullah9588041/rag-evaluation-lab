"""Run the full retrieval bake-off and write results/metrics.json + figures.

Strategies (chunking x retrieval [+ rerank]):
  1. fixed / bm25
  2. fixed / dense
  3. fixed / hybrid
  4. overlapping / dense
  5. sentence / dense
  6. sentence / hybrid
  7. sentence / hybrid + cross-encoder rerank (top-20 -> top-10)

Usage: PYTHONPATH=src .venv/bin/python scripts/run_analysis.py [--max-queries N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from raglab.chunking import chunk_document, split_sentences
from raglab.corpus import Document, load_corpus
from raglab.evalset import EvalSet, build_eval_set
from raglab.evaluation import bootstrap_mean_diff, per_query_metrics
from raglab.rerank import CrossEncoderReranker
from raglab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from raglab.visualization import (
    plot_latency_tradeoff,
    plot_ndcg_bars,
    plot_recall_curves,
    plot_win_loss,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
K_VALUES = (1, 5, 10, 20)
EVAL_K = 10  # ranking depth used for metrics

# Module-level holder so evaluate_config can reach the shared cross-encoder
# used for the support (faithfulness-proxy) scoring in every config.
reranker_holder: list = []


def load_documents() -> list[Document]:
    corpus_path = DATA / "corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} missing -- run: PYTHONPATH=src .venv/bin/python -m raglab.corpus"
        )
    return load_corpus(corpus_path)


def build_chunks(docs: list[Document], strategy: str):
    chunks = []
    for doc in docs:
        sections = [(s.heading, s.text) for s in doc.sections]
        chunks.extend(chunk_document(doc.doc_id, sections, strategy))
    return chunks


def evaluate_config(name, chunks, retriever, evalset, reranker=None,
                    rerank_depth: int = 20, max_queries: int | None = None,
                    k_values=K_VALUES):
    """Evaluate one strategy. `retriever` must already be fit on `chunks`.

    The rerank stage is batched across queries (one big cross-encoder forward
    pass) instead of one predict() call per query -- ~4x faster on CPU.
    """
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    queries = evalset.queries[:max_queries] if max_queries else evalset.queries
    if max_queries and len(evalset.queries) > max_queries:
        # strided sample: spreads evenly across documents and query types
        stride = max(1, len(evalset.queries) // max_queries)
        queries = evalset.queries[::stride][:max_queries]

    # ---- stage 1: first-stage retrieval for all queries ----
    first_stage: list[list[tuple[int, float]]] = []
    lat1: list[float] = []
    for q in queries:
        t1 = time.perf_counter()
        hits = retriever.search(q.text, EVAL_K if reranker is None else rerank_depth)
        lat1.append((time.perf_counter() - t1) * 1000.0)
        first_stage.append(hits)

    # ---- stage 2 (optional): batched cross-encoder rerank ----
    rerank_ms_per_query = 0.0
    if reranker is not None:
        all_pairs: list[tuple[str, str]] = []
        all_locs: list[tuple[int, int]] = []  # (query_pos, chunk_idx)
        for qi, (q, hits) in enumerate(zip(queries, first_stage)):
            for i, _ in hits:
                all_pairs.append((q.text, texts[i]))
                all_locs.append((qi, i))
        t2 = time.perf_counter()
        all_scores = reranker.score_pairs(all_pairs, batch_size=128)
        rerank_ms_per_query = (time.perf_counter() - t2) * 1000.0 / max(1, len(queries))
        per_q_scores: list[list[tuple[int, float]]] = [[] for _ in queries]
        for (qi, i), s in zip(all_locs, all_scores):
            per_q_scores[qi].append((i, s))
        first_stage = [
            sorted(sc, key=lambda kv: (-kv[1], kv[0]))[:EVAL_K]
            for sc in per_q_scores
        ]

    # ---- metrics + batched support scoring ----
    per_q: list[dict[str, float]] = []
    support_pairs: list[tuple[str, str]] = []
    support_locs: list[int] = []  # query position per pair
    for qi, q in enumerate(queries):
        positives = set(evalset.positives(q, chunks))
        ranked = [chunk_ids[i] for i, _ in first_stage[qi]]
        per_q.append(per_query_metrics(ranked, positives, k_values))
        if first_stage[qi]:
            for sent in split_sentences(texts[first_stage[qi][0][0]])[:3]:
                support_pairs.append((q.text, sent))
                support_locs.append(qi)

    support_scores: list[float] = []
    if support_pairs:
        all_s = reranker_holder[0].score_pairs(support_pairs, batch_size=128)
        best: dict[int, float] = {}
        for qi, s in zip(support_locs, all_s):
            best[qi] = max(best.get(qi, float("-inf")), s)
        support_scores = [best[qi] for qi in sorted(best)]

    latencies = [a + rerank_ms_per_query for a in lat1]
    agg = {m: float(np.mean([p[m] for p in per_q])) for m in per_q[0]}
    return {
        "strategy": name,
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "latency_ms_rerank_stage": float(rerank_ms_per_query),
        "support_score_mean": float(np.mean(support_scores)) if support_scores else None,
        "metrics_mean": agg,
        "per_query_ndcg10": [p["ndcg@10"] for p in per_q],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=300,
                        help="strided subsample of the eval set (default 300 for sane runtime)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    evalset: EvalSet = build_eval_set(docs)
    print(f"Eval set: {len(evalset.queries)} queries "
          f"({sum(1 for q in evalset.queries if q.query_type=='heading')} heading, "
          f"{sum(1 for q in evalset.queries if q.query_type=='question')} question)")

    print("Loading cross-encoder (for rerank config + support scoring)...")
    reranker = CrossEncoderReranker()
    reranker._load()
    reranker_holder.append(reranker)

    chunk_cache: dict[str, list] = {}
    # Fit each chunking's BM25 + dense retrievers ONCE and share across configs.
    retriever_cache: dict[str, tuple] = {}

    def get_retrievers(chunk_strategy: str, chunks: list):
        if chunk_strategy not in retriever_cache:
            print(f"  fitting BM25 + dense retrievers ({len(chunks)} chunks)...")
            bm25 = BM25Retriever()
            bm25.fit(chunks)
            dense = DenseRetriever()
            dense.fit(chunks, cache_path=str(DATA / f"embeddings_{chunk_strategy}.npy"))
            retriever_cache[chunk_strategy] = (bm25, dense)
        return retriever_cache[chunk_strategy]

    configs = [
        ("fixed+bm25", "fixed", lambda bm25, dense: bm25, None),
        ("fixed+dense", "fixed", lambda bm25, dense: dense, None),
        ("fixed+hybrid", "fixed", lambda bm25, dense: HybridRetriever(bm25=bm25, dense=dense), None),
        ("overlapping+dense", "overlapping", lambda bm25, dense: dense, None),
        ("sentence+dense", "sentence", lambda bm25, dense: dense, None),
        ("sentence+hybrid", "sentence", lambda bm25, dense: HybridRetriever(bm25=bm25, dense=dense), None),
        ("sentence+hybrid+rerank", "sentence", lambda bm25, dense: HybridRetriever(bm25=bm25, dense=dense), reranker),
    ]

    results: list[dict] = []
    for name, chunk_strategy, make_retriever, rr in configs:
        if chunk_strategy not in chunk_cache:
            print(f"Chunking corpus with strategy '{chunk_strategy}'...")
            chunk_cache[chunk_strategy] = build_chunks(docs, chunk_strategy)
            print(f"  -> {len(chunk_cache[chunk_strategy])} chunks")
        chunks = chunk_cache[chunk_strategy]
        bm25, dense = get_retrievers(chunk_strategy, chunks)
        print(f"Evaluating {name}...")
        res = evaluate_config(name, chunks, make_retriever(bm25, dense),
                              evalset, reranker=rr, max_queries=args.max_queries)
        m = res["metrics_mean"]
        print(f"  recall@10={m['recall@10']:.3f} mrr@10={m['mrr@10']:.3f} "
              f"ndcg@10={m['ndcg@10']:.3f} latency={res['latency_ms_mean']:.1f}ms")
        results.append(res)

    # --- statistical comparisons: best vs rest on nDCG@10 (paired bootstrap) ---
    order = sorted(range(len(results)), key=lambda i: -results[i]["metrics_mean"]["ndcg@10"])
    best = results[order[0]]
    comparisons = {}
    for i in order[1:]:
        other = results[i]
        comp = bootstrap_mean_diff(
            np.array(best["per_query_ndcg10"]), np.array(other["per_query_ndcg10"]))
        comparisons[f"{best['strategy']}_vs_{other['strategy']}"] = comp
        print(f"{best['strategy']} vs {other['strategy']}: "
              f"diff={comp['mean_diff']:+.4f} 95%CI=[{comp['ci_lo']:+.4f},{comp['ci_hi']:+.4f}] "
              f"p={comp['p_value']:.4f}")

    # --- figures ---
    k_list = list(K_VALUES)
    agg_for_plots = {r["strategy"]: r["metrics_mean"] | {"latency_ms": r["latency_ms_mean"]}
                     for r in results}
    plot_recall_curves(agg_for_plots, k_list, str(RESULTS / "figures" / "recall_curves.png"))
    # per-strategy bootstrap CIs of mean nDCG@10 for the bar plot
    from raglab.evaluation import bootstrap_mean_ci
    ci_bars = {r["strategy"]: bootstrap_mean_ci(np.array(r["per_query_ndcg10"]))
               for r in results}
    plot_ndcg_bars(agg_for_plots, ci_bars, str(RESULTS / "figures" / "ndcg_bars.png"))
    plot_latency_tradeoff(agg_for_plots, str(RESULTS / "figures" / "latency_tradeoff.png"))
    # win/loss: best vs runner-up
    runner = results[order[1]]
    deltas = np.array(best["per_query_ndcg10"]) - np.array(runner["per_query_ndcg10"])
    plot_win_loss(deltas, best["strategy"], runner["strategy"],
                  str(RESULTS / "figures" / "win_loss.png"))

    # --- metrics.json (drop per-query arrays to keep it readable; keep aggregates) ---
    serializable = []
    for r in results:
        serializable.append({k: v for k, v in r.items() if k != "per_query_ndcg10"})
    payload = {
        "k_values": list(K_VALUES),
        "eval_k": EVAL_K,
        "n_documents": len(docs),
        "n_queries_total": len(evalset.queries),
        "n_queries_evaluated": results[0]["n_queries"],
        "strategies": serializable,
        "comparisons_ndcg10": comparisons,
        "best_strategy": best["strategy"],
        "notes": "Metrics computed on structure-derived synthetic eval set; see README.",
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2))
    print("Wrote results/metrics.json and figures.")


if __name__ == "__main__":
    main()
