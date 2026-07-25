# Zotero Agent

Query a personal Zotero library in natural language. The project is built up in
layers — from a RAG pipeline written with nothing but `numpy` and `requests`, to
an agent that decides *how* to retrieve for each question — so you can see
exactly what each layer does and where its limits are.

## RAG vs. agent — the idea

**RAG (retrieval-augmented generation)** answers a question by first *fetching*
the most relevant documents and pasting them into the prompt. Here that means:
embed the question, cosine-search the paper abstracts, hand the top matches to
Claude. The retrieval strategy is fixed, and *we* choose it.

```
question ─▶ embed ─▶ cosine search over abstracts ─▶ top-k ─▶ Claude ─▶ answer
```

That's perfect for *"what papers are about X?"* — but wrong for *"what papers are
by person Y?"* Author lookup is an exact database query, not a fuzzy similarity
search (and author names aren't even in the embedded text). One fixed retrieval
strategy can't do both.

**An agent** fixes that by not choosing the strategy up front. We give Claude a
tool and let it decide, per question, how to retrieve:

```
question ─▶ Claude fills tool inputs ─▶ search_library(author?, topic?) ─▶ answer
                                          ├─ topic  → cosine search  (RAG)
                                          ├─ author → exact SQL lookup
                                          └─ both   → filter in SQL, then rank
```

Retrieval becomes a tool the model calls, not context we stuff into the prompt.
*"Novak's papers about VAEs"* becomes a real **filter + rank** — the thing
neither vectors nor SQL can do alone.

## The pieces

Read them in order — each builds on the one before:

| Path | What it is |
|---|---|
| `zotero_rag.py` | The from-scratch **RAG** pipeline. No framework; every stage explicit (SQLite → embeddings → cosine → Claude). **Start here.** |
| `demos/` | Five runnable scripts that walk through each RAG stage on your real library, plus one on conversational memory. |
| `zotero_agent_v0.py` | First **agent**: two separate tools (topic search + author lookup). Teaches tool *routing* — and shows where two tools fail to compose. |
| `zotero_agent.py` | The **agent**: one unified `search_library(author?, topic?)` tool — semantic, exact, or a real filter+rank join. |
| `langchain_version/` | The same RAG pipeline written with **LangChain**, plus a measured comparison. |

## Quick start

```bash
conda env create -f environment.yml && conda activate anthropic-rag
# or: pip install -r requirements.txt

ollama pull nomic-embed-text          # local embeddings, free
cp .env.example .env                  # then put your ANTHROPIC_API_KEY in it

python zotero_rag.py --reindex        # build the index (~2 min for ~1300 items)

python zotero_rag.py -q "your question"                 # plain RAG
python zotero_agent.py -q "Novak's papers about VAEs"   # the agent
```

Requires a Zotero library at `~/Zotero/zotero.sqlite` (change `ZOTERO_DB` in the
config block if yours lives elsewhere), Ollama running locally, and an Anthropic
API key. Indexing and retrieval are local and free; only the final answer (and
the agent's tool loop) calls the paid API.

## Learning path

1. **Read `demos/` in order** — each runs a single RAG stage on your library and
   prints what it produced (`01_load_corpus` … `05_generate`, plus
   `06_conversational` for memory). Demos 1–4 are free and local.
2. **Read `zotero_rag.py`** — the whole RAG pipeline in ~200 lines; the demos
   just call its functions. Its interactive mode also remembers the conversation
   (`/reset` starts over); `-q` is one-shot.
3. **Read `zotero_agent_v0.py`, then `zotero_agent.py`** — how retrieval becomes
   a tool the model calls, and why one unified tool composes better than two.
4. **Read `langchain_version/`** — the same RAG pipeline in a framework, measured
   side by side.

## What the LangChain comparison found

Measured, not assumed — numbers in
[`langchain_version/README.md`](langchain_version/README.md):

- **Identical results.** Same retrieved documents, same order (one exact score
  tie aside).
- **FAISS's default index is `IndexFlatL2`** — a flat matrix and a brute-force
  scan, i.e. the same algorithm as the hand-written `matrix @ q`.
- **Same speed.** ~183 lines / 4 dependencies vs. ~123 lines / 16.
- LangChain has **no Zotero loader**, so stage 1 is the same hand-written SQL in
  both versions.

At this scale the from-scratch version is the better engineering; LangChain's
value is in *swapping* components and the extras (metadata filtering, MMR, async,
tracing), not in the core pipeline.

## Notes

- Embeddings run **locally** via Ollama and cost nothing. Only the final answer
  (and the agent's tool loop) calls a paid API.
- The index is a cache with no invalidation — add papers to Zotero and nothing
  notices until you re-run `--reindex`.
- Author lookup reads Zotero's SQLite **directly** (always current, all authors),
  which is exactly why it belongs in a tool separate from the vector index.
- Abstracts are short, so one item = one chunk. No chunking logic is needed.
