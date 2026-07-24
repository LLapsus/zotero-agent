# Zotero RAG — from scratch vs. LangChain

A RAG (retrieval-augmented generation) pipeline over a personal Zotero library,
written twice: once with nothing but `numpy` and `requests`, and once with
LangChain. The point is to see exactly what a RAG system does — and what a
framework does and doesn't do for you.

```
Zotero SQLite  ──▶  corpus        title + abstract per item
corpus         ──▶  embeddings    one vector per item (local, via Ollama)
embeddings     ──▶  vector index  a matrix on disk
question       ──▶  top-k items   cosine similarity
top-k + Q      ──▶  answer        Anthropic Claude, with [n] citations
```

## Layout

| Path | What it is |
|---|---|
| `zotero_rag.py` | The from-scratch pipeline. No framework. Every stage explicit. |
| `demos/` | Five runnable scripts that walk through the pipeline one stage at a time. **Start here.** |
| `langchain_version/` | The same pipeline in LangChain, plus a measured comparison. |

## Quick start

```bash
conda env create -f environment.yml && conda activate anthropic-rag
# or: pip install -r requirements.txt

ollama pull nomic-embed-text          # local embeddings, free
cp .env.example .env                  # then put your ANTHROPIC_API_KEY in it

python zotero_rag.py --reindex        # build the index (~2 min for 1300 items)
python zotero_rag.py -q "your question"
```

Requires a Zotero library at `~/Zotero/zotero.sqlite` (change `ZOTERO_DB` in
the config block if yours lives elsewhere), Ollama running locally, and an
Anthropic API key.

## Learning path

Read the demos in order — each one runs against your real library and prints
what the stage actually produced:

```bash
python demos/01_load_corpus.py    # what the corpus looks like
python demos/02_embed.py          # text -> a 768-dim unit vector
python demos/03_build_index.py    # the "vector store" is two files
python demos/04_retrieve.py       # cosine search, and where it fails
python demos/05_generate.py       # the prompt, its token cost, the answer
```

Demos 1–4 are free and local. Demo 5 only calls the API if you pass `--run`
(~$0.009 per question); by default it shows the prompt and counts tokens.

Then read [`langchain_version/README.md`](langchain_version/README.md) for the
side-by-side comparison.

## What the comparison found

Measured, not assumed — details and numbers in the LangChain README:

- **Identical results.** Same retrieved documents, same order (one exact score
  tie aside).
- **FAISS's default index is `IndexFlatL2`** — a flat matrix and a brute-force
  scan, i.e. the same algorithm as the hand-written `matrix @ q`. Its on-disk
  index is the raw float32 vectors plus a 45-byte header.
- **Same speed.** Embedding 128 documents: 13.33 s hand-rolled, 13.31 s via
  `OllamaEmbeddings`.
- **~183 lines and 4 dependencies vs. ~123 lines and 16.**
- LangChain has **no Zotero loader**, so stage 1 is the same hand-written SQL
  in both versions.

The conclusion isn't "frameworks are bad" — it's that at this scale the
from-scratch version is the better engineering, and that LangChain's value is
in *swapping* components and in the extras (metadata filtering, MMR, async,
tracing), not in the core pipeline.

## Notes

- Embeddings run **locally** via Ollama and cost nothing. Only the final answer
  calls a paid API.
- The index is a cache with no invalidation — add papers to Zotero and neither
  version notices until you re-run `--reindex`.
- Abstracts are short, so one item = one chunk. No chunking logic is needed.
