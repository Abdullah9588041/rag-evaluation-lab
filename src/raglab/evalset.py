"""Synthetic-but-honest evaluation set, derived from corpus structure.

Construction (documented, reproducible, deterministic):
  For every section with >= MIN_WORDS words, two queries are generated:
    1. "heading"   -- the raw section heading (a keyword-style query, like real users type)
    2. "question"  -- a templated natural question: "What does the article say about {heading}?"
  The relevance judgment is the (doc_id, section) pair: a chunk is relevant iff
  it belongs to that section of that document.

This is BEIR-style structure-derived evaluation: it measures "can the retriever
find the passage a query refers to". It is NOT human-labeled QA, and it is never
presented as such. Known limits (documented in README):
  - queries are synthetic and correlated with section vocabulary (favors BM25 slightly)
  - binary relevance at section granularity, no graded judgments
  - one information need per query, no multi-hop
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .corpus import Document

MIN_SECTION_WORDS = 40


@dataclass
class EvalQuery:
    query_id: str
    text: str
    query_type: str  # "heading" | "question"
    doc_id: str
    section: str


@dataclass
class EvalSet:
    queries: list[EvalQuery] = field(default_factory=list)

    def positives(self, query: EvalQuery, chunks: list[Chunk]) -> list[str]:
        """Chunk ids relevant to a query under a given chunking scheme."""
        return [
            c.chunk_id for c in chunks
            if c.doc_id == query.doc_id and c.section == query.section
        ]


def build_eval_set(
    docs: list[Document],
    min_section_words: int = MIN_SECTION_WORDS,
) -> EvalSet:
    queries: list[EvalQuery] = []
    qid = 0
    for doc in docs:
        for sec in doc.sections:
            if sec.word_count < min_section_words:
                continue
            heading = sec.heading.strip()
            if not heading or heading.lower() == "introduction":
                continue
            queries.append(EvalQuery(
                query_id=f"q{qid:04d}", text=heading, query_type="heading",
                doc_id=doc.doc_id, section=sec.heading,
            ))
            qid += 1
            queries.append(EvalQuery(
                query_id=f"q{qid:04d}",
                text=f"What does the article on {doc.title} say about {heading}?",
                query_type="question",
                doc_id=doc.doc_id, section=sec.heading,
            ))
            qid += 1
    return EvalSet(queries=queries)
