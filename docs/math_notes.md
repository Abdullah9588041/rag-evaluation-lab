# Mathematical notes

Concise derivations for the methods used in this lab. Notation is standard;
no result here is novel — the point is to show the machinery is understood.

## 1. Cosine similarity geometry

For vectors $a, b \in \mathbb{R}^d$, cosine similarity is

$$\cos(a,b) = \frac{\langle a, b \rangle}{\|a\|_2 \, \|b\|_2}.$$

Geometrically this is the cosine of the angle between the vectors: 1 means
identical direction, 0 means orthogonal, −1 means opposite. Dense retrieval
encodes the query and each chunk as vectors and ranks by this quantity. With
$\ell_2$-normalized embeddings, cosine similarity equals the inner product, so
FAISS `IndexFlatIP` performs exact cosine search.

Why it works at all: the embedding model is trained (contrastively) so that
semantically similar texts land in nearby directions. Nothing in the formula
itself understands language — all the semantics live in the learned encoder.

## 2. BM25 (sketch)

BM25 scores a document $D$ against query terms $q_1, \dots, q_n$ as

$$\mathrm{BM25}(D, Q) = \sum_{i=1}^{n} \underbrace{\log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1}_{\text{IDF-like weight}} \cdot \underbrace{\frac{f(q_i, D)\,(k_1 + 1)}{f(q_i, D) + k_1\left(1 - b + b\,\frac{|D|}{\mathrm{avgdl}}\right)}}_{\text{saturated TF, length-normalized}}$$

with $f$ the term frequency, $N$ the corpus size, $n(q_i)$ the document
frequency, and typical $k_1 = 1.2$, $b = 0.75$. Two ideas matter:

- **Term-frequency saturation**: the 10th occurrence of a word adds less than
  the 1st (unlike raw TF-IDF).
- **Length normalization**: long documents are penalized so they don't win
  just by containing more words.

BM25 is purely lexical: it cannot match "car" to "automobile". That is exactly
the failure mode dense retrieval is meant to cover — and the reason hybrid
methods exist.

## 3. Reciprocal Rank Fusion (RRF)

Given rankings $r_1, \dots, r_m$ (each a permutation of candidates), RRF scores
candidate $d$ as

$$\mathrm{RRF}(d) = \sum_{j=1}^{m} \frac{1}{k + \mathrm{rank}_j(d)}, \qquad k = 60\ \text{(standard)}.$$

Why ranks and not scores? BM25 scores and cosine similarities live on
incomparable scales; any linear combination would need careful calibration.
Ranks are scale-free. The constant $k$ dampens the influence of very top ranks
so that a document ranked #1 by one system and #500 by another doesn't
automatically dominate. RRF has no learned parameters, which is both its
strength (robust, no tuning) and its weakness (can't learn that dense > BM25
on your data).

## 4. Retrieval metrics (binary relevance)

For a query with relevant set $P$ and ranked list $R$:

- **Recall@k** $= |R_{1:k} \cap P| / |P|$: fraction of relevant chunks found.
- **MRR@k** $= 1 / \mathrm{rank}(\text{first relevant})$ (0 if none in top $k$):
  rewards putting *one* good answer first — the right metric when a single
  chunk suffices.
- **DCG@k** $= \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i+1)}$,
  **nDCG@k** $= \mathrm{DCG@k} / \mathrm{IDCG@k}$ where IDCG is the DCG of the
  ideal ranking (all relevant chunks first). The $\log$ discount encodes that
  users care more about the top of the list; normalization to $[0,1]$ makes
  scores comparable across queries with different numbers of relevant chunks.

## 5. Bootstrap confidence intervals for strategy comparison

Queries are the experimental units. For two strategies A, B with per-query
scores $a_i, b_i$ ($i = 1..n$), the paired bootstrap:

1. Repeat $B = 10{,}000$ times: sample $n$ query indices *with replacement*,
   compute $\bar{a}^* - \bar{b}^*$.
2. The 95% CI is the $[2.5\%, 97.5\%]$ quantiles of the bootstrap distribution.

If the interval excludes 0, the difference is significant at $\alpha = 0.05$
without any normality assumption. The two-sided p-value is
$2 \cdot \min(P(\Delta^* \ge 0), P(\Delta^* \le 0))$. Pairing (same resampled
queries for both strategies) removes query-difficulty variance — the same
reason paired t-tests beat unpaired ones.

## 6. What "faithfulness" means (and what we actually measure)

Formally, an answer $A$ with claims $c_1, \dots, c_m$ is faithful to context $C$
if $C \models c_j$ (entailment) for every $j$. True faithfulness checking needs
(1) claim extraction and (2) an NLI model for entailment — both out of scope
without a local LLM budget. This lab instead reports a **support score**: the
cross-encoder relevance of the query against the top-3 sentences of the
retrieved chunk. High relevance $\neq$ entailment, and the README says so
explicitly. It answers a weaker but still useful question: "does the top chunk
actually talk about what was asked?" A proper MNLI-based check is listed as
future work.
