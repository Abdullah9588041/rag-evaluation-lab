"""Chunking strategies under comparison.

All chunkers operate on the same word-tokenization (regex \\w+, lowercased only
at retrieval time) so differences in results come from segmentation, not tokenization.
Each chunker returns Chunk objects with word offsets, which makes the
segmentation testable (reconstruction invariants).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\"\'])")
_WS_RE = re.compile(r"\s+")


def word_tokens(text: str) -> list[str]:
    """Deterministic word tokenization shared by chunking and BM25."""
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (regex-based, no model download needed)."""
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    start_word: int  # inclusive, in section word coordinates
    end_word: int    # exclusive

    @property
    def n_words(self) -> int:
        return self.end_word - self.start_word


def _words_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words, spans = [], []
    for m in _WORD_RE.finditer(text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans


def chunk_fixed(
    doc_id: str,
    section: str,
    text: str,
    chunk_tokens: int = 200,
    overlap: int = 0,
) -> list[Chunk]:
    """Fixed-size sliding window over words. overlap=0 -> exact partition."""
    assert 0 <= overlap < chunk_tokens
    words, spans = _words_with_spans(text)
    chunks: list[Chunk] = []
    i, n, idx = 0, len(words), 0
    step = chunk_tokens - overlap
    while i < n:
        j = min(i + chunk_tokens, n)
        start_char, end_char = spans[i][0], spans[j - 1][1]
        chunks.append(Chunk(
            chunk_id=f"{doc_id}#{idx:04d}",
            doc_id=doc_id,
            section=section,
            text=text[start_char:end_char],
            start_word=i,
            end_word=j,
        ))
        idx += 1
        if j == n:
            break
        i += step
    return chunks


def chunk_sentence_aware(
    doc_id: str,
    section: str,
    text: str,
    max_tokens: int = 200,
) -> list[Chunk]:
    """Pack whole sentences greedily up to max_tokens words; never split a sentence."""
    words, spans = _words_with_spans(text)
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    idx = 0
    cur_sents: list[str] = []
    cur_words = 0
    word_ptr = 0  # word index where the current chunk starts

    def flush() -> None:
        nonlocal idx, cur_sents, cur_words, word_ptr
        if not cur_sents:
            return
        start_char = text.find(cur_sents[0])
        # find offsets robustly by scanning from an approximate position
        joined = " ".join(cur_sents)
        pos = text.find(joined[: min(40, len(joined))])
        if pos == -1:
            pos = 0
        end = pos + len(joined)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}#{idx:04d}",
            doc_id=doc_id,
            section=section,
            text=joined,
            start_word=word_ptr,
            end_word=word_ptr + cur_words,
        ))
        idx += 1
        word_ptr += cur_words
        cur_sents, cur_words = [], 0

    for sent in sentences:
        sw = len(_WORD_RE.findall(sent))
        if cur_words + sw > max_tokens and cur_sents:
            flush()
        if sw > max_tokens:
            # A single pathological sentence: fall back to fixed chunking for it.
            for c in chunk_fixed(doc_id, section, sent, chunk_tokens=max_tokens):
                c.start_word += word_ptr
                c.end_word += word_ptr
                c.chunk_id = f"{doc_id}#{idx:04d}"
                chunks.append(c)
                idx += 1
            word_ptr += sw
        else:
            cur_sents.append(sent)
            cur_words += sw
    flush()
    return chunks


CHUNKERS = {
    "fixed": lambda doc_id, section, text: chunk_fixed(doc_id, section, text, chunk_tokens=200, overlap=0),
    "overlapping": lambda doc_id, section, text: chunk_fixed(doc_id, section, text, chunk_tokens=200, overlap=50),
    "sentence": lambda doc_id, section, text: chunk_sentence_aware(doc_id, section, text, max_tokens=200),
}


def chunk_document(doc_id: str, sections: list[tuple[str, str]], strategy: str) -> list[Chunk]:
    """Chunk every section of a document with the named strategy."""
    chunker = CHUNKERS[strategy]
    chunks: list[Chunk] = []
    for heading, text in sections:
        chunks.extend(chunker(doc_id, heading, text))
    # re-number chunk ids sequentially across the document
    for i, c in enumerate(chunks):
        c.chunk_id = f"{doc_id}#{i:04d}"
    return chunks
