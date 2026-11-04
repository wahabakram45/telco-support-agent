# Lesson 03 — Embeddings & vector search

> **Goal:** understand what embeddings are and what they miss, how similarity search works (exact
> and approximate), and what a vector database adds. Build and probe a real index of your corpus.
> **Time:** ~3–4 hours (1.5 h reading, 30 min building, 1 h probing, 30 min quiz).

Pipeline so far: chunk ✅ → **embed → store → retrieve** → generate.

---

## Part A — Concepts

### 1. What is an embedding?

An **embedding** is a fixed-length vector of numbers (1,536 floats for our model) that represents the
*meaning* of a text. Texts with similar meaning land close together in that space, even when they share
no words: "using my phone internet on holiday in Dubai" lands near "Roaming Policy (UAE)".

**How the models learn it** (you know this from ML): a transformer encoder, trained with
**contrastive learning** on huge sets of (query, relevant passage) pairs. The loss pulls matching pairs
together and pushes other passages in the batch (in-batch negatives) apart. Then mean pooling or a
[CLS] token turns the token vectors into one vector.

This setup is called a **bi-encoder**: query and document are encoded *separately*, so document
vectors can be computed once, ahead of time. That's what makes search over millions of documents fast.
The price: the query never "sees" the document. A **cross-encoder** reads both together and is much
more accurate, but far too slow to run against every document. It's used as a *reranker* on the top
results instead (Lesson 08).

Two rules that follow:
- **Query and documents must use the same model.** Vectors from different models live in unrelated
  spaces; comparing them gives garbage, with no error. Our `Retriever` refuses to run if the index
  and the query embedder differ.
- **Changing models means re-embedding everything.** Store the model name with the index (we do,
  in `data/index/meta.json`).

### 2. Measuring similarity

| Measure | Formula | Notes |
|---|---|---|
| **Cosine similarity** | `a·b / (‖a‖‖b‖)` | Angle only; ignores length. Range −1..1. The standard for text. |
| **Dot product** | `a·b` | Also rewards length. Same ranking as cosine *if vectors are unit length*. |
| **Euclidean (L2) distance** | `‖a − b‖` | Smaller = closer. For unit vectors, `‖a−b‖² = 2 − 2·cos`, so again the same ranking. |

The practical trick: **normalize once at indexing time**, and cosine becomes a plain dot product. A
whole search is then one matrix–vector multiply: `scores = matrix @ query`. OpenAI already returns unit
vectors, but normalizing yourself makes the code correct for any model.

**Absolute scores mean little.** With our model, an unrelated pair scores about 0.04 and a good
paraphrase only about 0.5. Each model has its own score range. Use scores to *rank*, and be very
careful using them as a threshold (Task 2 shows why).

### 3. Choosing an embedding model

| Factor | Questions to ask |
|---|---|
| **Quality on *your* data** | Public benchmark: **MTEB** leaderboard (retrieval tab). But always check on your own eval set; leaderboard rank often doesn't transfer. |
| **Languages** | Our users write Arabic too. Is the model multilingual? |
| **Dimensions** | More dimensions = more storage and slower search. Many models (OpenAI v3, Nomic…) support **Matryoshka** truncation: request 512 dims instead of 1,536 for a small quality loss. |
| **Max input** | 8,191 tokens for OpenAI; 512 for many open models. Longer text is truncated. |
| **Cost & latency** | API: pay per token plus a network call per query. Self-hosted: GPU/CPU cost, no per-call fee. |
| **Privacy** | Sending documents to an API may be forbidden for sensitive data → self-host. |

| Option | Examples | Good for |
|---|---|---|
| API | OpenAI `text-embedding-3-small` / `-large`, Cohere embed, Voyage, Google | Fast start, strong quality, no infrastructure |
| Open-source | BGE, E5 (multilingual-e5 for Arabic), GTE, Nomic, via `sentence-transformers` | Privacy, no per-call cost, fine-tuning on your domain |

### 4. Exact vs approximate search

**Exact (brute-force) search** compares the query with *every* vector: `n × d` multiply-adds. For our
201 chunks that's about 300k operations, well under a millisecond. It stays fine up to roughly
**100k vectors**. At 10M vectors it's too slow per query, so we use **approximate nearest neighbour
(ANN)** indexes, which trade a little recall for speed.

| Index | Idea | Knobs |
|---|---|---|
| **HNSW** (most common) | A layered graph: each vector links to its nearest neighbours. Search starts at the sparse top layer, greedily walks toward the query, then drops down to denser layers, like taking motorways and then local streets. | `M` (links per node: memory vs recall), `ef_construction` (build quality), `ef_search` (query-time recall vs latency) |
| **IVF** | Cluster vectors with k-means; search only the few clusters nearest the query. | `nlist` (clusters), `nprobe` (clusters searched) |
| **PQ** (product quantization) | Compress vectors to a few bytes each; often combined with IVF. | Code size: memory vs accuracy |

How to measure an ANN index: **recall@k against exact search** ("of the true top 10, how many did
HNSW return?"), plus latency and memory. Our brute-force index is the ground truth for exactly that
comparison.

### 5. Vector databases

A vector *library* (numpy, FAISS) does the math. A vector *database* adds what a production system
needs: persistence, **metadata filtering**, inserts, updates and deletes without a rebuild,
scaling, backups, access control, and often **hybrid search** (dense + keyword, Lesson 07).

| Option | Type | When |
|---|---|---|
| numpy / **FAISS** | Library, in process | Learning, prototypes, batch jobs; FAISS for fast ANN with no server |
| **Chroma** | Embedded DB (runs inside your Python process) | Local development, small apps. Our Lesson 04 store. |
| **pgvector** | Postgres extension | You already run Postgres: one database, SQL joins with your business data, transactions. Often the best default for a real product. |
| **Qdrant**, Weaviate, Milvus | Dedicated vector DB servers | Large scale, heavy filtering, hybrid search built in |
| **Pinecone** | Fully managed service | No operations work; pay for convenience |
| Elasticsearch / OpenSearch | Search engine with vector support | You already run them for keyword search |

### 6. Metadata filtering: pre vs post

"Roaming prices for KSA customers" vs "… UAE customers" scores **0.755**: to an embedding the two
markets are nearly the same. So the market must come from **metadata**, not from similarity.

- **Pre-filtering:** filter first (market ∈ {KSA, ALL}, not superseded), then rank only those. Always
  returns k valid results. With ANN graphs, heavy filters can disconnect the graph; good vector DBs
  handle this internally.
- **Post-filtering:** take the top-k, then drop the ones that don't match. Simple, but it can return
  fewer than k results, or none at all.

Our `VectorIndex.search(where=...)` pre-filters. Lesson 09 adds *deciding* the filters from the
question and the user's profile.

### 7. Our decisions

- **Model: `text-embedding-3-small`.** It costs $0.02 per 1M tokens: our whole corpus is about 13.5k
  tokens, so **$0.0003** for a full index. It returns 1,536 dims, handles Arabic reasonably, and needs
  no infrastructure. `-large` costs 6.5× more; we'll only try it if Phase 3 evals say retrieval is
  the bottleneck.
- **Embed `context + text`** (Lesson 02's contextual headers).
- **Cache every vector on disk** by (model, text). Re-running or re-chunking never pays twice:
  essential once Phase 3 compares several chunking strategies.
- **A brute-force numpy index now:** you see every step, and it's the exact ground truth. **Chroma in
  Lesson 04** for persistence and filtering through a real DB API; pgvector noted as the production
  path.
- **Pre-filtering** on market and freshness.

---

## Part B — Build

**You write one function**, the heart of vector search. I wrote the rest: `Embedder` (batching,
ordering, disk cache, cost), `VectorIndex` (normalize, filter, save/load), `Retriever`, and three
scripts.

### Task 1 — `cosine_top_k()` in `src/telco_support_agent/vector_index.py` (~20 min)
Rules are in the docstring; `tests/test_vector_index.py` checks them. Hints:
- `normalize()` is already written. After normalizing both, cosine is `matrix @ query`: one line,
  shape `(n,)`.
- `np.argsort(-scores, kind="stable")` sorts best-first and keeps ties in index order. Take `[:k]`.
- Return Python `int`/`float`, not numpy types: `int(i)`, `float(scores[i])`.
- (Interview bonus: for huge `n`, `np.argpartition` finds the top-k in O(n) without a full sort.)

`uv run pytest tests/test_vector_index.py -v` until green.

### Task 2 — Build the index and probe it (~1 h)
1. Open OpenAI's official pricing page, find the embedding price, and add it to your `.env`:
   `PRICE_EMBEDDING_PER_M=0.02` (check the number). The model is already set as a default in
   `config.py`.
2. Build the index (about 5 seconds, a fraction of a cent). Run it twice: the second run is free.
   ```bash
   uv run python scripts/build_index.py
   ```
3. **Predict, then run.** Write down a guess (high / medium / low) for each of the 8 sentence pairs in
   `scripts/probe_embeddings.py` before running:
   ```bash
   uv run python scripts/probe_embeddings.py pairs
   ```
4. Predict whether the right chunk comes first for each of the 6 probe queries, then run:
   ```bash
   uv run python scripts/probe_embeddings.py queries
   uv run python scripts/search.py "how much is the GCC day pass" --market KSA --current --k 3
   ```

<details>
<summary>What I measured and how to read it (open after running)</summary>

**Pairs**
| pair | score | lesson |
|---|---|---|
| paraphrase | 0.505 | Captures meaning without shared words, but the absolute number is low for this model |
| same topic | 0.428 | Related but not the same question: close to the paraphrase score! |
| unrelated | 0.043 | |
| **negation** | **0.823** | "want to cancel" vs "do **not** want to cancel": nearly identical. Embeddings are weak at negation. |
| **numbers** | **0.930** | 55 vs 95 SAR: almost the same vector. **Never trust embeddings to distinguish numbers.** Our number checker and structured tools exist for this. |
| **market** | **0.755** | KSA vs UAE: why we filter on metadata |
| arabic | 0.448 | Cross-lingual works, weaker than same-language |
| typo | 0.625 | Fairly robust to typos |

**Queries**
- **Synonyms ✅:** "phone internet on holiday in Dubai" → Roaming Policy (UAE), with no shared keywords.
  This is what dense retrieval is for.
- **Version trap:** the FAQ and current policies rank above the superseded 2025 page here, but the
  2025 page *is* in the index with nearly the same score as its replacement. Only the `--current` filter
  guarantees it never reaches the LLM.
- **Buried fact ✅:** "elderly customers" finds "Customers aged 65 and above get a 20% discount" in the
  T&C's *Charges and payments* section. Section-level chunking made it findable; a whole-document
  vector would have diluted it away.
- **Exact identifier:** right answer, but by a margin of 0.02 (0.523 vs 0.503 for the current version).
  Dense search is shaky on IDs, codes and rare terms → **BM25 / hybrid search, Lesson 07**.
- **Arabic:** finds roaming pages, but pay-as-you-go instead of the GCC Day Pass, and ranks a
  superseded document second. Usable, not great → candidate for the eval set.
- **Out of domain:** "weather in Riyadh" still returns 3 results (it always does: top-k never says "no").
  Top score **0.403**, about the same as correct answers to the Arabic (0.458) and elderly (0.479)
  queries. **A similarity threshold can't separate "no answer" from "hard question" here.** The
  generation step must handle "I don't know" (Lesson 04).
</details>

When you're done, commit and say "lesson 03 done". I'll clean up, then quiz you.

---

## Part C — Interview drill

Short-format quiz on "review lesson 03". Full model answers:
[interview-bank §03](../interview-bank.md#03--embeddings--vector-search).

1. What is an embedding, and how are embedding models trained?
2. Cosine vs dot product vs Euclidean distance: when do they agree?
3. How do you choose an embedding model?
4. What happens when you switch embedding models in production?
5. Exact vs approximate search; how does HNSW work, and what are its knobs?
6. Which vector database would you choose, and why?
7. Pre-filtering vs post-filtering on metadata?
8. Where does dense retrieval fail, and what do you do about it?
