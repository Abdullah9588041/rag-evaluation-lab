# Data

## Source (REAL data — fetched live, no synthetic documents)

`corpus.json` / `corpus_meta.json` are built by `python -m raglab.corpus` from the
English Wikipedia via the public MediaWiki action API (`action=query&prop=extracts`,
`explaintext`, `exsectionformat=wiki`). No authentication required.

- **Fetch date:** 2026-10-06
- **Articles requested:** 49 machine-learning / statistics / probability topics (see `src/raglab/corpus.py::TOPICS`)
- **Documents kept:** 49 (all 49 topics returned non-empty extracts)
- **Sections:** 1,173 (boilerplate sections like "See also"/"References" removed)
- **Total words:** 268,127

Each document is stored as `{doc_id, title, sections: [{heading, text}]}`.
Section headings are preserved from the wiki markup (`== Heading ==`), which is
what the synthetic eval set uses to derive relevance judgments.

## Regenerating

```bash
PYTHONPATH=src .venv/bin/python -m raglab.corpus --out data/corpus.json --meta data/corpus_meta.json
```

One title per request (the API returns empty extracts for long articles when
batched), ~0.5s pause between requests, exponential-backoff retries.
Total runtime is a few minutes. The files are git-ignored (re-fetchable);
only this README is committed.

## License note

Wikipedia text is CC BY-SA 4.0. This corpus is used for non-commercial
research/education (retrieval evaluation). Article titles are listed in
`corpus_meta.json`.
