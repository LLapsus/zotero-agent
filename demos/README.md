# Demos — a guided walk through the RAG pipeline

These scripts exist to **show you what each stage of `zotero_rag.py` does**, one
at a time. They don't reimplement anything: each demo imports the real functions
from `zotero_rag.py`, runs them, and prints the result in a readable way.

The idea is that you can read the pipeline top-to-bottom by running the demos in
order, seeing the actual data flow through your own Zotero library:

```
Zotero SQLite  ──▶  corpus        (01_load_corpus)   text per item
corpus         ──▶  embeddings    (02_embed)         one vector per item
embeddings     ──▶  vector index  (03_build_index)   an (N × dim) matrix on disk
question       ──▶  top-k items   (04_retrieve)      cosine similarity
top-k + Q      ──▶  answer        (05_generate)      Anthropic Claude
```

## How to run

From the **repo root** (not from inside `demos/`):

```bash
python demos/01_load_corpus.py
```

Each demo prints a short "Next stage" pointer at the end, so you always know
where to go next.

## The demos

| # | Script | Shows | Needs |
|---|--------|-------|-------|
| 1 | `01_load_corpus.py` | the corpus dicts read from Zotero's SQLite | Zotero DB |
| 2 | `02_embed.py` | text → vector, via local Ollama | Ollama running |
| 3 | `03_build_index.py` | the on-disk vector store | an index |
| 4 | `04_retrieve.py` | cosine search for top-k items | index + Ollama |
| 5 | `05_generate.py` | grounded answer from Claude | `ANTHROPIC_API_KEY` |
| 6 | `06_conversational.py` | adding memory: follow-up questions that work | `ANTHROPIC_API_KEY` |

Demo 5 is free by default (it only builds the prompt and counts tokens). Pass
`--run` when you want it to actually call Claude.

Demo 6 is a step *beyond* the base pipeline: `zotero_rag.py` answers each
question from a blank slate, so a follow-up like "explain that more simply"
fails — the retriever re-searches on those few words and Claude never saw the
previous answer. Demo 6 fixes both, from scratch, by keeping a message history
(so Claude remembers) and rewriting each follow-up into a standalone search
query (so retrieval follows the thread). It always calls the API — a
conversation is inherently interactive — so run `--scripted` for a fixed 3-turn
showcase. This is exactly the "conversational memory" a framework like LangChain
gives you prebuilt (`RunnableWithMessageHistory`,
`create_history_aware_retriever`); here you can see the machinery.

`_common.py` is a tiny shared helper that lets the demos `import zotero_rag`
from one directory up. It contains no pipeline logic.

## Cost & side effects

- `01`–`04` are **free and local**: they read files, hit your local Ollama, and
  do arithmetic. Nothing leaves your machine.
- `05` calls the Anthropic API, which **costs a small amount of money** per
  question (it's one short Claude request).
- `06` always calls the API and costs a little more: one answer call per turn,
  plus a cheap rewrite call from the second turn on. The `--scripted` run is a
  few cents total.

## Where this is heading

The end goal of this repo is a side-by-side comparison:

- **this** from-scratch pipeline (`zotero_rag.py` + `demos/`), where every step
  is a visible dot product and some string formatting, versus
- the **same** pipeline written with **LangChain**, to see exactly what the
  framework abstracts away — loaders, embeddings, vector stores, retrievers, and
  the prompt/LLM chain.

Reading these demos first means that when you later look at the LangChain
version, you'll recognise which library object stands in for which stage here.
