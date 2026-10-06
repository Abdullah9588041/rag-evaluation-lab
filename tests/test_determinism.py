"""Determinism of the pure (model-free) pipeline components."""

from raglab.chunking import chunk_document, word_tokens
from raglab.corpus import parse_sections
from raglab.evalset import build_eval_set
from raglab.evaluation import bootstrap_mean_diff
from raglab.retrieval import reciprocal_rank_fusion
import numpy as np

RAW = "== Intro ==\nMachine learning is great. It learns from data.\n== Methods ==\nWe optimize everything always with gradient descent methods here.\n"


def test_parse_sections_deterministic():
    assert parse_sections(RAW) == parse_sections(RAW)


def test_chunking_deterministic():
    secs = [("Intro", "word " * 500)]
    a = chunk_document("d", secs, "sentence")
    b = chunk_document("d", secs, "sentence")
    assert [(c.chunk_id, c.text, c.start_word, c.end_word) for c in a] == \
           [(c.chunk_id, c.text, c.start_word, c.end_word) for c in b]


def test_rrf_deterministic():
    r = [[3, 1, 2], [1, 2, 0]]
    assert reciprocal_rank_fusion(r) == reciprocal_rank_fusion(r)


def test_bootstrap_deterministic():
    a = np.linspace(0, 1, 50)
    b = np.linspace(0, 0.5, 50)
    r1 = bootstrap_mean_diff(a, b, n_boot=500, seed=7)
    r2 = bootstrap_mean_diff(a, b, n_boot=500, seed=7)
    assert r1 == r2


def test_word_tokens_deterministic():
    assert word_tokens("Hello, World! 123") == word_tokens("Hello, World! 123") == ["hello", "world", "123"]
