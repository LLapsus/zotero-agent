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

## Installation & setup

This runs against **your own Zotero library** — there is no bundled dataset. You
point it at your `zotero.sqlite`, build a local index once, then ask questions
about your own papers.

### Prerequisites

- **Python 3.12**
- **[Ollama](https://ollama.com)** — a local server that produces the embeddings
  (free, runs on your machine)
- **An [Anthropic API key](https://console.anthropic.com)** — only the final
  answer calls the paid API
- **A Zotero library** — the desktop app keeps its data in a `zotero.sqlite`
  file (default `~/Zotero/zotero.sqlite`)

### 1. Get the code

```bash
git clone <your-fork-url> zotero_agent && cd zotero_agent
```

### 2. Create the environment

Conda:

```bash
conda env create -f environment.yml && conda activate anthropic-rag
```

or plain venv + pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`environment.yml` installs the **core** pipeline. The LangChain comparison in
`langchain_version/` needs extra packages — they're in `requirements.txt`, or
`pip install langchain langchain-community langchain-ollama langchain-anthropic faiss-cpu`
into the conda env.

### 3. Start Ollama and pull the embedding model

```bash
ollama pull nomic-embed-text     # 768-dim local embeddings
```

Ollama serves on `http://localhost:11434` (the default in `zotero_rag.py`'s
`OLLAMA_URL`). Make sure `ollama serve` is running — the desktop app starts it
for you; on a headless box run it yourself.

### 4. Point it at your Zotero library

The default is `~/Zotero/zotero.sqlite`. If yours lives elsewhere, edit the
`ZOTERO_DB` line in the **config block at the top of `zotero_rag.py`**.

You can leave **Zotero open** while running this — `load_corpus()` copies the
database to a temp file and reads it **read-only**, so it never locks or touches
your live library.

### 5. Add your Anthropic API key

```bash
cp .env.example .env         # then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

The code loads `.env` automatically (via `python-dotenv`); `.env` is gitignored.

### 6. Build the index

```bash
python zotero_rag.py --reindex
```

This embeds every title + abstract via Ollama (~2 min for ~1300 items) and
caches the result to `~/.zotero_rag/`. Re-run it whenever you add papers — the
index is a cache with no automatic invalidation.

## Usage

```bash
python zotero_rag.py                          # interactive RAG chat (/reset to clear)
python zotero_rag.py -q "your question"       # one-shot RAG

python zotero_agent.py                        # the agent, interactive
python zotero_agent.py -q "Novak's papers about VAEs"
python zotero_agent.py --scripted             # a portable demo (author/topic/both)
```

In the interactive modes, type `/exit` (or Ctrl-D) to quit; the RAG chat also
takes `/reset` to start a fresh conversation. Indexing and retrieval are **local
and free** — only the final answer (and the agent's tool loop) calls the paid
API (~$0.01 per question).

**Trouble?**
- *`No index found`* → run `python zotero_rag.py --reindex` first.
- *`Zotero DB not found`* → fix `ZOTERO_DB` in `zotero_rag.py`.
- *Ollama connection errors* → is `ollama serve` running, and did you
  `ollama pull nomic-embed-text`?
- *Auth errors* → is `ANTHROPIC_API_KEY` set in `.env`?

## Example — the agent routing a question

```
$ python zotero_agent.py
zotero_agent -- one unified search_library(author?, topic?) tool (COSTS MONEY)
Ask about your library. Commands:  /exit (or Ctrl-D)  quit

> Which of Medbouhi's papers is about variational autoencoders?
  [tool call] search_library({"author": "Medbouhi", "topic": "variational autoencoders"})
  [tool result] Papers by 'Medbouhi' ranked by relevance to 'variational autoenc...

Answer:

Medbouhi has one paper on that topic: "Towards topology-aware Variational
Auto-Encoders: from InvMap-VAE to Witness Simplicial VAE" [1]. It argues that
standard VAEs may not preserve the topology of the data between the input and
the latent space, and proposes topology-aware variants to address it.

> /exit
```

Notice the model filled **both** `author` and `topic`, so the tool filtered to
Medbouhi's papers in SQL first and *then* ranked that subset by topic — the
filter+rank join that plain RAG can't do. (In a real terminal the `[tool call]`
markers and `[n]` citations are colour-coded.)

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
