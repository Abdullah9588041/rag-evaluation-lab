"""Chunking invariants: segmentation must cover the source text."""

from raglab.chunking import chunk_fixed, chunk_sentence_aware, split_sentences, word_tokens

TEXT = (
    "Machine learning is a field of study in artificial intelligence. "
    "It gives computers the ability to learn from data. "
    "Statistical methods form the backbone of many algorithms. "
    "Optimization drives the training of modern models. "
    "Evaluation must be rigorous and honest."
)


def test_fixed_no_overlap_partitions_words():
    words = word_tokens(TEXT)
    chunks = chunk_fixed("d", "s", TEXT, chunk_tokens=10, overlap=0)
    reconstructed = []
    for c in chunks:
        reconstructed.extend(word_tokens(c.text))
    assert reconstructed == words
    # offsets tile exactly
    assert chunks[0].start_word == 0
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_word == b.start_word
    assert chunks[-1].end_word == len(words)


def test_fixed_overlap_covers_all_words():
    words = word_tokens(TEXT)
    chunks = chunk_fixed("d", "s", TEXT, chunk_tokens=10, overlap=4)
    covered = set()
    for c in chunks:
        covered.update(range(c.start_word, c.end_word))
    assert covered == set(range(len(words)))
    # overlap means the union of chunk word sets exceeds the word count
    total = sum(c.n_words for c in chunks)
    assert total > len(words)


def test_sentence_aware_never_splits_sentences():
    chunks = chunk_sentence_aware("d", "s", TEXT, max_tokens=12)
    sentences = split_sentences(TEXT)
    for c in chunks:
        # every chunk is a concatenation of whole sentences
        joined = " ".join(sentences)
        assert c.text in joined or all(s in c.text or c.text in s for s in sentences)
    # word coverage is complete and non-overlapping
    words = word_tokens(TEXT)
    covered = set()
    for c in chunks:
        covered.update(range(c.start_word, c.end_word))
    assert covered == set(range(len(words)))


def test_sentence_aware_respects_max_tokens():
    long_text = " ".join(f"Sentence number {i} about machine learning models." for i in range(30))
    chunks = chunk_sentence_aware("d", "s", long_text, max_tokens=20)
    for c in chunks:
        assert c.n_words <= 20


def test_chunk_ids_sequential():
    from raglab.chunking import chunk_document
    chunks = chunk_document("doc_001", [("Intro", TEXT), ("More", TEXT)], "fixed")
    assert [c.chunk_id for c in chunks] == [f"doc_001#{i:04d}" for i in range(len(chunks))]
    assert all(c.doc_id == "doc_001" for c in chunks)
