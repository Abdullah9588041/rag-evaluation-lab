"""rag-evaluation-lab: retrieval + evaluation half of RAG, done rigorously.

Compares chunking strategies, retrieval methods (BM25 / dense / hybrid RRF),
and cross-encoder reranking on a real Wikipedia corpus, with statistical
evaluation (recall@k, MRR, nDCG@k, bootstrap CIs) rather than vibes.
"""

__version__ = "0.1.0"
