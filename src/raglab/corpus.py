"""Build a real document corpus from Wikipedia (no auth required).

Fetches ~40 machine-learning / statistics articles via the MediaWiki action API,
parses them into (section heading, text) structure, and caches the result so the
pipeline is reproducible without re-hitting the network.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-evaluation-lab/0.1.0 (portfolio research project; contact: portfolio)"

# Fixed, documented topic list: ML + statistics + probability articles.
TOPICS: list[str] = [
    "Machine learning",
    "Statistical classification",
    "Regression analysis",
    "Linear regression",
    "Logistic regression",
    "Decision tree",
    "Random forest",
    "Support vector machine",
    "Naive Bayes classifier",
    "K-means clustering",
    "Hierarchical clustering",
    "Principal component analysis",
    "Dimensionality reduction",
    "Neural network",
    "Deep learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (deep learning)",
    "Word embedding",
    "Natural language processing",
    "Gradient descent",
    "Stochastic gradient descent",
    "Backpropagation",
    "Overfitting",
    "Cross-validation (statistics)",
    "Bias–variance tradeoff",
    "Hyperparameter optimization",
    "Ensemble learning",
    "Boosting (machine learning)",
    "Bayesian inference",
    "Bayes' theorem",
    "Maximum likelihood estimation",
    "Expectation–maximization algorithm",
    "Markov chain Monte Carlo",
    "Hidden Markov model",
    "Kalman filter",
    "Time series",
    "Autoregressive integrated moving average",
    "Hypothesis testing",
    "P-value",
    "Confidence interval",
    "Central limit theorem",
    "Bootstrapping (statistics)",
    "Causal inference",
    "A/B testing",
    "Recommender system",
    "Anomaly detection",
    "Reinforcement learning",
    "Q-learning",
]

_HEADING_RE = re.compile(r"^==+\s*(.+?)\s*==+\s*$")


@dataclass
class Section:
    heading: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Document:
    doc_id: str
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sections)


def _fetch_via_curl(params: dict, max_retries: int = 5) -> dict:
    """Fetch one API response using curl (reliable through the egress proxy).

    Python's requests/urllib3 intermittently gets its connections dropped by the
    proxy; curl has been 100% reliable in this environment, so the corpus fetch
    shells out to it. The returned parsed JSON is identical either way.
    """
    import subprocess
    import urllib.parse

    query = urllib.parse.urlencode(params)
    url = f"{API_URL}?{query}"
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            proc = subprocess.run(
                ["curl", "-s", "-m", "90", "-A", USER_AGENT, url],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                raise RuntimeError(f"curl failed (rc={proc.returncode}): {proc.stderr[:200]}")
            return json.loads(proc.stdout)
        except Exception as exc:  # noqa: BLE001 - retry on any transient failure
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Wikipedia API request failed after {max_retries} attempts") from last_exc


def fetch_extracts(
    titles: list[str],
    batch_size: int = 1,
    pause: float = 1.0,
    partial_path: str | Path | None = None,
) -> dict[str, str]:
    """Fetch plain-text extracts for Wikipedia titles via the MediaWiki API.

    NOTE: batch_size=1 (one title per request). Batching multiple titles makes
    the API return empty extracts for long articles; single-title requests are
    reliable.

    If partial_path is given, per-title results are checkpointed to it after
    each fetch, so interrupted runs resume instead of restarting.
    """
    out: dict[str, str] = {}
    if partial_path and Path(partial_path).exists():
        out = json.loads(Path(partial_path).read_text(encoding="utf-8"))
    remaining = [t for t in titles if t not in out]
    for i, title in enumerate(remaining):
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "wiki",  # keeps '== Heading ==' markers in plain text
            "exlimit": "max",
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        data = _fetch_via_curl(params)
        pages = data["query"]["pages"]
        for page in pages.values():
            text = page.get("extract", "")
            if text:
                # key by the REQUESTED title so resume logic stays aligned
                out[title] = text
        if partial_path:
            Path(partial_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        time.sleep(pause)
    return out


def parse_sections(raw_text: str) -> list[Section]:
    """Split a wiki-format plain-text extract into (heading, text) sections."""
    sections: list[Section] = []
    current_heading = "Introduction"
    buf: list[str] = []
    for line in raw_text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            text = "\n".join(buf).strip()
            if text:
                sections.append(Section(heading=current_heading, text=text))
            current_heading = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    text = "\n".join(buf).strip()
    if text:
        sections.append(Section(heading=current_heading, text=text))
    # Drop boilerplate tail sections that add noise, not signal.
    sections = [
        s for s in sections
        if s.heading.lower() not in {"see also", "references", "external links", "further reading", "notes"}
    ]
    return sections


def build_corpus(topics: list[str] | None = None,
                partial_path: str | Path | None = None) -> list[Document]:
    """Fetch and parse all topics into a list of Documents (deterministic order)."""
    topics = topics or TOPICS
    extracts = fetch_extracts(topics, partial_path=partial_path)
    docs: list[Document] = []
    for i, title in enumerate(topics):
        raw = extracts.get(title)
        if not raw:
            continue  # redirect target already captured under its own title
        sections = parse_sections(raw)
        if not sections:
            continue
        docs.append(Document(doc_id=f"doc_{i:03d}", title=title, sections=sections))
    return docs


def save_corpus(docs: list[Document], path: str | Path) -> None:
    payload = [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "sections": [{"heading": s.heading, "text": s.text} for s in d.sections],
        }
        for d in docs
    ]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_corpus(path: str | Path) -> list[Document]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(
            doc_id=p["doc_id"],
            title=p["title"],
            sections=[Section(heading=s["heading"], text=s["text"]) for s in p["sections"]],
        )
        for p in payload
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch the Wikipedia corpus.")
    parser.add_argument("--out", default="data/corpus.json")
    parser.add_argument("--meta", default="data/corpus_meta.json")
    parser.add_argument("--partial", default="data/_partial_extracts.json")
    args = parser.parse_args()

    docs = build_corpus(partial_path=args.partial)
    save_corpus(docs, args.out)
    meta = {
        "source": "English Wikipedia via MediaWiki action API (no authentication)",
        "fetch_date": time.strftime("%Y-%m-%d"),
        "n_topics_requested": len(TOPICS),
        "n_docs": len(docs),
        "n_sections": sum(len(d.sections) for d in docs),
        "total_words": sum(d.word_count for d in docs),
        "titles": [d.title for d in docs],
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(docs)} documents, {meta['n_sections']} sections, "
          f"{meta['total_words']} words -> {args.out}")
