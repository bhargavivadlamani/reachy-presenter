# RAG Pipeline

This module implements the **Data Ingestion** and **Retrieval** layers of the production RAG pipeline used by reachy-presenter.

![Production RAG Pipeline — Architecture & Key Components](architecture.png)

---

## Overview

| Stage | File | What it does |
|---|---|---|
| Ingest | `ingest.py` | Parse → chunk → embed (dense + sparse) → store in Qdrant |
| Retrieve | `retrieve.py` | Hybrid search → rerank → build context string for LLM |

---

## Prerequisites

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Verify: `curl http://localhost:6333/healthz`

### 3. Configure `.env`

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | If using `--provider openai` | — | OpenAI API key |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `CROSS_ENCODER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `COHERE_API_KEY` | If using `--reranker cohere` | — | Cohere API key |

### 4. (Optional) Start Ollama for local embeddings

```bash
# Install from https://ollama.com, then:
ollama pull nomic-embed-text
```

---

## Data Ingestion

Parses a PDF or PPTX file, splits it into 400-token chunks, embeds each chunk as both a **dense vector** (semantic) and a **sparse vector** (BM25), and stores everything in Qdrant.

Re-ingesting the same file is **idempotent** — existing chunks for that file are deleted before re-inserting.

### Usage

```bash
python -m app.rag.ingest <file> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `file` | required | Path to PDF or PPTX file |
| `--provider` | `ollama` | Embedding provider: `openai` or `ollama` |
| `--model` | `nomic-embed-text` | Dense embedding model name |
| `--collection` | `reachy_collection` | Qdrant collection name |
| `--parser` | `pdfplumber` | Parser to use: `pdfplumber` or `python-pptx` |
| `--sparse-model` | `Qdrant/bm25` | Sparse (BM25) model for hybrid search |

### Examples

```bash
# Local embeddings (free, no API key needed)
python -m app.rag.ingest slides.pdf

# OpenAI embeddings
python -m app.rag.ingest slides.pdf --provider openai

# PPTX file with python-pptx parser
python -m app.rag.ingest deck.pptx --parser python-pptx

# Custom collection
python -m app.rag.ingest slides.pdf --collection my_collection
```

### Embedding providers

| Provider | Model | Dims | Notes |
|---|---|---|---|
| `ollama` (default) | `nomic-embed-text` | 768 | Local, free, no API key |
| `openai` | `text-embedding-3-small` | 1536 | Hosted, requires `OPENAI_API_KEY` |

> **Important:** Dense vector dimensions are fixed at collection creation time. Switching providers without changing `--collection` will fail. Use separate collection names per provider (e.g. `--collection slides_openai`).

### Parsers

| Parser | File types | Notes |
|---|---|---|
| `pdfplumber` (default) | `.pdf` | Extracts raw text per page |
| `python-pptx` | `.pptx`, `.ppt` | Extracts text from all shapes per slide |

---

## Retrieval

Given a query string, runs **hybrid search** (BM25 + dense vector) over the Qdrant collection to fetch top-20 candidate chunks, then **reranks** them to return the top-N most relevant chunks with source metadata.

### Usage

```bash
python -m app.rag.retrieve "<query>" [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `query` | required | Natural language question or search string |
| `--collection` | `reachy_collection` | Qdrant collection to search |
| `--provider` | `ollama` | Embedding provider: `openai` or `ollama` |
| `--model` | `nomic-embed-text` | Dense embedding model (must match what was used at ingest) |
| `--sparse-model` | `Qdrant/bm25` | Sparse model (must match ingest) |
| `--reranker` | `cross-encoder` | Reranker to use: `cross-encoder` or `cohere` |
| `--top-n` | `5` | Number of final chunks to return after reranking |

### Examples

```bash
# Basic query (local embeddings + cross-encoder reranker)
python -m app.rag.retrieve "what does slide 3 cover"

# With OpenAI embeddings
python -m app.rag.retrieve "key takeaways" --provider openai

# Cohere reranker (requires COHERE_API_KEY in .env)
python -m app.rag.retrieve "main findings" --reranker cohere

# Return top 3 chunks
python -m app.rag.retrieve "introduction" --top-n 3
```

### Output format

Each retrieved chunk is printed with its source file and page number:

```
[1] Source: /path/to/slides.pdf, Page: 4
<chunk text...>

[2] Source: /path/to/slides.pdf, Page: 1
<chunk text...>
```

### Rerankers

| Reranker | Model | Notes |
|---|---|---|
| `cross-encoder` (default) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, free, ~100–300ms |
| `cohere` | `rerank-english-v3.0` | Hosted, best quality, requires `COHERE_API_KEY` |

Override the cross-encoder model via `CROSS_ENCODER_MODEL` in `.env`.

---

## Python API

Both modules expose a clean function interface for use in other code:

```python
from app.rag.ingest import ingest
from app.rag.retrieve import retrieve, build_context

# Ingest
n = ingest("slides.pdf", provider="ollama")
print(f"Stored {n} chunks")

# Retrieve
chunks = retrieve("what is the main thesis?")
context = build_context(chunks)  # formatted string ready for LLM prompt
```

---

## Retrieval pipeline internals

```
User query
    │
    ▼
Hybrid Search (Qdrant)
  ├── Dense: nomic-embed-text / text-embedding-3-small
  └── Sparse: BM25 (Qdrant/bm25)
    │   top-20 candidates
    ▼
Reranker
  ├── cross-encoder/ms-marco-MiniLM-L-6-v2  (local)
  └── cohere rerank-english-v3.0             (API)
    │   top-N chunks
    ▼
build_context()  →  [1] Source: ...\n<text>\n\n[2] ...
```

---

## Verify end-to-end

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Ingest
python -m app.rag.ingest slides.pdf

# 3. Retrieve
python -m app.rag.retrieve "your question here"

# 4. Check Qdrant dashboard
# Open http://localhost:6333/dashboard
# Verify collection "reachy_collection" has points with metadata fields:
#   source, doc_id, page, ingested_at, embedding_model
```
