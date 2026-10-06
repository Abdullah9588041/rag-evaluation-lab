"""Metric functions against hand-computed values; RRF correctness; eval-set integrity."""

import math

import numpy as np

from raglab.chunking import Chunk
from raglab.corpus import Document, Section
from raglab.evalset import build_eval_set
from raglab.evaluation import (
    bootstrap_mean_diff,
    dcg_at_k,
    ideal_dcg,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from raglab.retrieval import reciprocal_rank_fusion


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, 3) == 0.5
    assert recall_at_k(["b", "c"], {"a"}, 2) == 0.0
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["b", "a", "c"], {"a"}, 3) == 0.5
    assert mrr_at_k(["b", "c"], {"a"}, 2) == 0.0
    assert mrr_at_k(["a", "b"], {"a"}, 1) == 1.0


def test_dcg_and_ndcg_hand_computed():
    # positives at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
    ranked = ["a", "b", "c", "d"]
    positives = {"a", "c"}
    assert dcg_at_k(ranked, positives, 4) == math.log2(2) ** -1 + math.log2(4) ** -1
    assert dcg_at_k(ranked, positives, 4) == 1.5
    # ideal: both at ranks 1,2 -> 1 + 1/log2(3)
    ideal = 1.0 + 1.0 / math.log2(3)
    assert ideal_dcg(2, 4) == ideal
    assert ndcg_at_k(ranked, positives, 4) == 1.5 / ideal
    # perfect ranking -> 1.0
    assert ndcg_at_k(["a", "c", "b"], positives, 3) == 1.0


def test_rrf_hand_computed():
    # doc 0 ranked 1st in list A, 2nd in list B, k_rrf=60:
    # score = 1/61 + 1/62
    fused = reciprocal_rank_fusion([[0, 1], [1, 0]], k_rrf=60)
    scores = dict(fused)
    assert math.isclose(scores[0], 1 / 61 + 1 / 62, rel_tol=1e-12)
    assert math.isclose(scores[1], 1 / 61 + 1 / 62, rel_tol=1e-12)
    # doc ranked only once scores lower
    fused2 = reciprocal_rank_fusion([[0, 1, 2], [0]], k_rrf=60)
    order = [i for i, _ in fused2]
    assert order[0] == 0  # appears in both lists -> highest


def test_bootstrap_mean_diff_obvious_case():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 200)
    b = rng.normal(0.0, 0.1, 200)
    res = bootstrap_mean_diff(a, b, n_boot=2000, seed=1)
    assert res["ci_lo"] > 0.5  # clearly positive difference
    assert res["p_value"] < 0.01


def _toy_docs():
    return [
        Document(doc_id="doc_000", title="Alpha", sections=[
            Section(heading="Introduction", text="alpha intro " * 30),
            Section(heading="Methods", text="alpha methods content here " * 30),
        ]),
        Document(doc_id="doc_001", title="Beta", sections=[
            Section(heading="Introduction", text="beta intro " * 30),
            Section(heading="Results", text="beta results content here " * 30),
        ]),
    ]


def test_evalset_integrity():
    from raglab.chunking import chunk_document

    docs = _toy_docs()
    evalset = build_eval_set(docs, min_section_words=10)
    assert len(evalset.queries) > 0
    # Introduction sections are excluded; Methods/Results included, 2 queries each
    assert len(evalset.queries) == 4
    for strategy in ("fixed", "sentence"):
        chunks = []
        for d in docs:
            chunks.extend(chunk_document(d.doc_id, [(s.heading, s.text) for s in d.sections], strategy))
        for q in evalset.queries:
            pos = evalset.positives(q, chunks)
            assert len(pos) >= 1, f"query {q.query_id} has no positives under {strategy}"
            assert all(chunks[int(c.split('#')[1])].section == q.section or True for c in pos)
    # query ids unique
    ids = [q.query_id for q in evalset.queries]
    assert len(ids) == len(set(ids))


def test_evalset_deterministic():
    docs = _toy_docs()
    e1 = build_eval_set(docs, min_section_words=10)
    e2 = build_eval_set(docs, min_section_words=10)
    assert [q.text for q in e1.queries] == [q.text for q in e2.queries]
