# The same pipeline, written with LangChain

`zotero_rag_lc.py` is a point-for-point port of `../zotero_rag.py`. Same Zotero
library, same embedding model, same Claude model, same `k`, same system prompt,
same CLI. The only thing that changes is *who writes the plumbing*.

```bash
python langchain_version/zotero_rag_lc.py --reindex
python langchain_version/zotero_rag_lc.py -q "your question"
```

Everything below was **measured on this repo**, not assumed.

---

## Stage by stage

| # | Stage | From scratch | LangChain |
|---|-------|--------------|-----------|
| 1 | Load corpus | `load_corpus()` — raw SQL over Zotero's SQLite | **the identical function**, wrapped in `Document` |
| 2 | Embed | `embed_batch()` — `requests.post` to Ollama | `OllamaEmbeddings` |
| 3 | Index | `np.save` / `np.load` | `FAISS.save_local` / `load_local` |
| 4 | Retrieve | `matrix @ q`, `np.argsort` | `store.as_retriever(search_kwargs={"k": 6})` |
| 5 | Generate | `anthropic.messages.create` | `ChatPromptTemplate \| ChatAnthropic \| StrOutputParser` |

### Stage 1 — the framework adds nothing

LangChain ships ~200 document loaders. **None reads Zotero's SQLite.** The
LangChain version literally calls the original `load_corpus()` and repackages
each dict as a `Document` (`{page_content, metadata}`). All the real work —
the four-table pivot, the temp-file copy, the abstract filter — is the same
hand-written SQL either way.

Generalisation: a framework helps where your problem is *standard*. The moment
your data source is even slightly unusual, you write it yourself.

### Stage 2 — same vectors, same speed

Both produce **bit-identical vectors** (cosine between them = `1.000000`), and
embedding is equally fast:

| | 128 documents | extrapolated to 1317 |
|---|---:|---:|
| from scratch (batches of 64) | 13.33 s | ~137 s |
| `OllamaEmbeddings` | 13.31 s | ~137 s (measured: 136 s) |

`OllamaEmbeddings` batches just as well as the hand-rolled version. The two
minutes are Ollama's throughput, not framework overhead.

*(Aside discovered while testing: Ollama's `nomic-embed-text` already returns
unit vectors, so the normalisation in `zotero_rag.py:160` is a safety net. Keep
it — other providers don't normalise.)*

### Stage 3 — FAISS is the same two files

This is the most instructive finding. What FAISS writes to disk:

| from scratch | | LangChain |  |
|---|---:|---|---:|
| `vectors.npy` | 4 036 736 B | `index.faiss` | 4 045 869 B |
| `corpus.json` | 3 133 569 B | `index.pkl` | 3 206 994 B |

`index.faiss` holds 1317 × 768 float32 = **4 045 824 B** of raw vectors. The
file is 4 045 869 B — a **45-byte header**. It is the matrix, nothing more.

And the index type is **`IndexFlatL2`**. "Flat" means *no acceleration
structure at all*: no HNSW graph, no IVF clusters. **FAISS's default is the
same brute-force scan as `matrix @ q`.**

FAISS's clever approximate-search structures exist and are worth it — at
millions of vectors. At 1314 they would be pure overhead. Measured brute-force
cost on this machine:

| N vectors | search time |
|---:|---:|
| 1 314 (this library) | 0.02 ms |
| 100 000 | 7.7 ms |
| 1 000 000 | 64 ms |

For context, embedding the *question* takes ~390 ms. The search is not the
bottleneck and never will be at this scale.

FAISS does add one real thing: `index.pkl` is a **pickle**, so `load_local`
requires `allow_dangerous_deserialization=True`. Loading a downloaded index
executes arbitrary code. `corpus.json` has no such problem.

### Stage 4 — identical results

Same questions through both:

| question | same 6 docs | same order |
|---|---|---|
| phase transitions in ML | ✅ 6/6 | ✅ |
| VAEs and latent space topology | ✅ 6/6 | ✅ |
| neural network potentials for MD | ✅ 6/6 | ⚠️ positions 3–4 swapped |

The swap is **an exact score tie** — both documents score `0.7481085658`,
difference `0.00e+00`. Order between equals is arbitrary. There is no
behavioural difference.

### Stage 5 — where the styles actually differ

The LCEL chain (`prompt | llm | parser`) is the one place LangChain genuinely
changes the shape of the code. It's cleaner to compose and easy to swap models.

But note what does **not** change: `format_docs()` — the `[1] [2]` labelling
that makes citation possible — is hand-written in *both* versions. The
framework has no opinion about how you present sources, which is the part that
most affects answer quality.

---

## The tally

| | from scratch | LangChain |
|---|---:|---:|
| lines of code (excl. comments) | ~183 | ~123 |
| direct dependencies | 4 | 16 |
| retrieval results | — | identical |
| index on disk | 2 files, 7.2 MB | 2 files, 7.3 MB |
| search algorithm | brute force | brute force |

## What LangChain actually buys you

**Real wins**
- Swapping components is a one-line change: `OllamaEmbeddings` → `OpenAIEmbeddings`,
  `FAISS` → `Chroma`/`Pinecone`, `ChatAnthropic` → any other chat model.
- Free features you'd otherwise write: metadata filtering, MMR (diversity)
  retrieval, async, streaming, batching, callbacks/tracing.
- Fewer lines once you're past the parts it doesn't cover.

**Real costs**
- 16 dependencies instead of 4, and they move fast. `langchain-community`
  (where `FAISS` lives) already prints a **sunset warning** on import.
- The abstractions hide what's happening. You would not learn from this file
  that "the vector store" is a flat matrix and a brute-force dot product.
- Version churn: LangChain 1.x moved most import paths. Most tutorials online
  are written for 0.x and simply don't run.
- One ecosystem hazard found while setting this up: the PyPI package
  **`langchain-faiss` is an empty stub** by an unrelated author — a plausible
  name that isn't the official integration. The real one is
  `langchain_community.vectorstores.FAISS`.

## The honest conclusion

For **this** project — 1300 abstracts, one embedding provider, one LLM — the
from-scratch version is the better engineering. It has 4 dependencies, no
hidden behaviour, and its "naive" numpy search is provably the same algorithm
FAISS runs by default.

LangChain earns its keep when you need to *swap* things, when you want the
extras (filtering, MMR, tracing, async) without writing them, or when a team
benefits from a shared vocabulary. It does not make the pipeline faster, and
here it did not make it better — it made it *shorter*, at the price of no
longer being able to see what it does.

Read `../demos/` first. Then this file makes a lot more sense.
