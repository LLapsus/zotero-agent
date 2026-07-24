#!/usr/bin/env python3
"""
zotero_rag_lc.py  -  the SAME RAG pipeline as ../zotero_rag.py, via LangChain.

Point-for-point equivalent, so you can diff the two files and see exactly what
the framework does and doesn't do for you:

    stage                 from scratch (../zotero_rag.py)   LangChain (here)
    -------------------   ------------------------------   ----------------------
    1. load corpus        load_corpus()  -- raw SQL         load_corpus() + Document
                                                            (identical: no Zotero loader exists)
    2. embed              embed_batch()  -- requests.post   OllamaEmbeddings
    3. index              np.save / np.load                 FAISS.save_local / load_local
    4. retrieve           matrix @ q, np.argsort            vectorstore.as_retriever()
    5. generate           anthropic.messages.create         ChatPromptTemplate | ChatAnthropic

Deps:  pip install langchain langchain-community langchain-ollama \
                   langchain-anthropic faiss-cpu
Needs: same as the original -- Ollama running, ANTHROPIC_API_KEY set.

Usage: python langchain_version/zotero_rag_lc.py --reindex
       python langchain_version/zotero_rag_lc.py -q "your question"
       python langchain_version/zotero_rag_lc.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the from-scratch module importable: we reuse its config and its SQLite
# reader (see stage 1 below for why).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zotero_rag  # noqa: E402  -- also runs load_dotenv() for ANTHROPIC_API_KEY

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.vectorstores.utils import DistanceStrategy  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

# ----------------------------------------------------------------------------
# 0. CONFIG  -  read from the original so both versions stay in lockstep.
# ----------------------------------------------------------------------------

INDEX_DIR = Path.home() / ".zotero_rag_lc"   # separate dir; don't clobber the original
EMBED_MODEL = zotero_rag.EMBED_MODEL
CLAUDE_MODEL = zotero_rag.CLAUDE_MODEL
TOP_K = zotero_rag.TOP_K

SYSTEM = (
    "You answer questions using ONLY the provided paper abstracts from the "
    "user's Zotero library. Cite sources inline as [1], [2], etc. If the "
    "abstracts do not contain the answer, say so plainly -- do not use outside "
    "knowledge or guess."
)


# ----------------------------------------------------------------------------
# 1. LOAD CORPUS  ->  list[Document]
#
# LangChain has ~200 document loaders. None of them reads Zotero's SQLite, so
# this stage is EXACTLY the same work as the original -- we call the very same
# load_corpus(). The only framework-shaped part is repackaging each dict into a
# Document, which is just {page_content: str, metadata: dict}.
#
# Takeaway: the framework saved us nothing here.
# ----------------------------------------------------------------------------

def load_documents() -> list[Document]:
    corpus = zotero_rag.load_corpus(zotero_rag.ZOTERO_DB)
    return [
        Document(
            page_content=item["text"],       # same string the original embeds
            metadata={
                "key": item["key"],
                "title": item["title"],
                "authors": item["authors"],
                "year": item["year"],
                "abstract": item["abstract"],
            },
        )
        for item in corpus
    ]


# ----------------------------------------------------------------------------
# 2 + 3. EMBED AND INDEX  ->  FAISS
#
# OllamaEmbeddings replaces the hand-written requests.post, and FAISS replaces
# the numpy matrix. FAISS.from_documents does BOTH stages in one call: it
# embeds every document (batched internally) and builds the index.
#
# distance_strategy=COSINE matches the original's cosine similarity. It happens
# not to matter here -- nomic-embed-text returns unit vectors, and for unit
# vectors L2 ranking and cosine ranking are identical -- but being explicit
# means this still behaves if you swap in an embedding model that doesn't
# normalize.
#
# What FAISS writes to disk mirrors the original exactly:
#     index.faiss  = the vectors        (~ vectors.npy)
#     index.pkl    = the documents      (~ corpus.json)
# ----------------------------------------------------------------------------

def embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_index() -> FAISS:
    docs = load_documents()
    print(f"Embedding {len(docs)} items via Ollama/{EMBED_MODEL} ...")
    store = FAISS.from_documents(
        docs,
        embeddings(),
        distance_strategy=DistanceStrategy.COSINE,
    )
    INDEX_DIR.mkdir(exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"Index saved to {INDEX_DIR}")
    return store


def load_index() -> FAISS:
    if not (INDEX_DIR / "index.faiss").exists():
        sys.exit("No index found. Run once with --reindex first.")
    # index.pkl is a pickle, so loading executes arbitrary code if the file was
    # tampered with. LangChain forces you to acknowledge that. It's your own
    # file here, so it's fine -- but never point this at a downloaded index.
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings(),
        allow_dangerous_deserialization=True,
    )


# ----------------------------------------------------------------------------
# 4 + 5. RETRIEVE AND GENERATE  ->  an LCEL chain
#
# `retriever | format | prompt | llm | parser` is LangChain Expression Language:
# the `|` operator wires callables together, each one's output feeding the next.
# It replaces the original's explicit retrieve() + answer() calls.
#
# format_docs is the interesting bit -- it's the [1] [2] labelling from the
# original, verbatim. The framework has no opinion about how you present
# sources, so this stays hand-written either way.
# ----------------------------------------------------------------------------

def format_docs(docs: list[Document]) -> str:
    blocks = []
    for n, d in enumerate(docs, 1):
        m = d.metadata
        tag = " ".join(x for x in [m.get("authors"), m.get("year")] if x)
        blocks.append(f"[{n}] {m['title']} ({tag})\n{m['abstract']}")
    return "\n\n".join(blocks)


def build_chain(store: FAISS):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Abstracts:\n\n{context}\n\n---\n\nQuestion: {question}"),
    ])
    llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)
    return prompt | llm | StrOutputParser()


# ----------------------------------------------------------------------------
# 6. CLI  -  same interface as the original, so you can compare answers directly
# ----------------------------------------------------------------------------

def run_query(question: str, store: FAISS, chain) -> None:
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})  # retrieve the top K docs from the index
    docs = retriever.invoke(question)  # returns a list of retrieved Document objects

    # print the retrieved documents' titles for context
    print("\nRetrieved:")
    for n, d in enumerate(docs, 1):
        print(f"  [{n}] {d.metadata['title'][:80]}")

    # print the answer generated by the chain
    print("\nAnswer:\n")
    print(chain.invoke({"context": format_docs(docs), "question": question}))  # invoke the chain with the formatted context and question
    print()


def main() -> None:
    # parse command-line arguments
    ap = argparse.ArgumentParser(description="LangChain RAG over a Zotero library.")
    ap.add_argument("--reindex", action="store_true", help="(re)build the vector index")
    ap.add_argument("-q", "--query", help="ask one question and exit")
    args = ap.parse_args()

    # build or load the index, then build the chain
    store = build_index() if args.reindex else load_index()
    
    # build the chain that wires retrieval and generation together
    chain = build_chain(store)

    # run a single query if provided, otherwise enter interactive mode
    if args.query:
        run_query(args.query, store, chain)
        return

    # enter interactive mode
    print("Ask about your library (empty line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        run_query(q, store, chain)


if __name__ == "__main__":
    main()
