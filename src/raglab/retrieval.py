"""Retrieval methods with a common interface: BM25, dense (FAISS), hybrid (RRF).

All retrievers implement:
    fit(chunks: list[Chunk]) -> None
    search(query: str, k: int) -> list[(chunk_index, score)]

Hybrid retrieval fuses rankings with Reciprocal Rank Fusion, which needs only
ranks (not calibrated scores) -- the standard way to combine lexical and dense
signals that live on incomparable scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk, word_tokens

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache: one loaded encoder shared by all retriever instances in a
# process (model init reads ~100 weight files; no need to repeat it per config).
_MODEL_CACHE: dict[str, object] = {}


@dataclass
class ScoredChunk:
    index: int
    score: float


class BM25Retriever:
    name = "bm25"

    def fit(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [word_tokens(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(word_tokens(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._index = None

    def _load(self):
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        self._model = _MODEL_CACHE[self.model_name]
        try:
            self._faiss
        except AttributeError:
            import faiss

            self._faiss = faiss
        return self._model

    def fit(self, chunks: list[Chunk], cache_path: str | None = None) -> None:
        """Encode chunks (or load cached embeddings) and build the FAISS index.

        cache_path: optional .npy file. Embeddings are cached with a SHA-256 of
        the chunk texts, so a stale cache (different corpus/chunking) is never
        silently reused.
        """
        import hashlib
        from pathlib import Path

        model = self._load()
        self.chunks = chunks
        texts = [c.text for c in chunks]
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]

        emb = None
        if cache_path and Path(cache_path).exists():
            try:
                payload = np.load(cache_path, allow_pickle=True).item()
                if payload.get("digest") == digest and payload.get("model") == self.model_name:
                    emb = np.asarray(payload["embeddings"], dtype=np.float32)
            except Exception:
                emb = None
        if emb is None:
            emb = np.asarray(
                model.encode(texts, batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True),
                dtype=np.float32,
            )
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, {"digest": digest, "model": self.model_name,
                                     "embeddings": emb}, allow_pickle=True)
        index = self._faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine on normalized vectors
        index.add(emb)
        self._index = index

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        model = self._load()
        q = np.asarray(
            model.encode([query], show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )
        scores, idx = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]


def reciprocal_rank_fusion(
    rankings: list[list[int]], k_rrf: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked lists of chunk indices via RRF: score(d) = sum 1/(k_rrf + rank).

    Ranks are 1-based. Returns (index, fused_score) sorted by score desc,
    ties broken by index for determinism.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """RRF fusion of BM25 and dense rankings (retrieves top_n from each first)."""

    name = "hybrid"

    def __init__(self, k_rrf: int = 60, fuse_depth: int = 100,
                 model_name: str = EMBEDDING_MODEL,
                 bm25: "BM25Retriever | None" = None,
                 dense: "DenseRetriever | None" = None):
        self.k_rrf = k_rrf
        self.fuse_depth = fuse_depth
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever(model_name)

    def fit(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25.fit(chunks)
        self.dense.fit(chunks)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        bm25_ranking = [i for i, _ in self.bm25.search(query, self.fuse_depth)]
        dense_ranking = [i for i, _ in self.dense.search(query, self.fuse_depth)]
        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k_rrf=self.k_rrf)
        return fused[:k]
