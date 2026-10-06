"""Cross-encoder reranking of a candidate list.

A cross-encoder scores (query, document) pairs jointly, which is more accurate
than bi-encoder cosine similarity but too slow for full-corpus search -- hence
the standard two-stage pipeline: cheap first-stage retrieval, expensive rerank
of the top candidates.
"""

from __future__ import annotations

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level cache: share one loaded cross-encoder per process.
_RERANK_MODEL_CACHE: dict[str, object] = {}


class CrossEncoderReranker:
    name = "cross-encoder-rerank"

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self.model_name not in _RERANK_MODEL_CACHE:
            from sentence_transformers import CrossEncoder

            _RERANK_MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
        self._model = _RERANK_MODEL_CACHE[self.model_name]
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        """Score (query, text) pairs in one batched forward pass."""
        model = self._load()
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[int, str]],
        top_k: int,
        batch_size: int = 32,
    ) -> list[tuple[int, float]]:
        """Rerank (chunk_index, chunk_text) candidates; return top_k (index, score)."""
        model = self._load()
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(
            ((idx, float(s)) for (idx, _), s in zip(candidates, scores)),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return ranked[:top_k]
