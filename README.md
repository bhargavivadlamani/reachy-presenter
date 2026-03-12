# reachy-presenter

Animates a Reachy Mini robot to deliver presentations from PDF/PPTX files.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
```

## Running a presentation

```bash
python -m app.main path/to/file.pdf
```

---

## RAG Ingestion

Documents can be ingested into a Qdrant vector store for retrieval-augmented generation.

### Basic usage

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Ingest a file
python -m app.rag.ingest path/to/file.pdf
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--provider` | `ollama` | Embedding provider (`openai` or `ollama`) |
| `--model` | `nomic-embed-text` | Embedding model name |
| `--collection` | `reachy_collection` | Qdrant collection name |
| `--parser` | `pdfplumber` | Parser backend (`pdfplumber` or `python-pptx`) |
| `--metrics` | off | Print ingestion metrics report after ingest |
| `--encoding` | `cl100k_base` | tiktoken encoding used for chunking and metrics |

### Ingestion metrics

Pass `--metrics` to print a three-section report after ingestion:

```bash
python -m app.rag.ingest path/to/file.pdf --metrics
```

**Parse Quality** — measures how much of the source document was usable:

| Field | How it is measured |
|---|---|
| Total pages | Number of pages/slides returned by the parser |
| Empty pages | Pages whose text is blank after stripping whitespace |
| Error rate | `empty_pages / total_pages × 100` |

**Chunk Size Distribution (tokens)** — measures whether the splitter is filling chunks effectively:

| Statistic | Description |
|---|---|
| P50 | Median chunk size in tokens |
| P90 | 90th-percentile chunk size |
| P95 | 95th-percentile chunk size |
| Min / Max | Smallest and largest chunk |

Token counts use the same tiktoken encoding as the splitter (`--encoding`), so the numbers reflect what was actually embedded. A healthy distribution has P90/P95 close to the 400-token chunk ceiling and a Min well above 1.

**End-to-End Latency** — times each stage individually:

| Stage | What is timed |
|---|---|
| Parse | `parse()` call — file I/O and text extraction |
| Chunk | `TokenTextSplitter.split_documents()` |
| Embed + store | `QdrantVectorStore.from_documents()` — embedding API calls and Qdrant upsert |
| Total | Sum of the three stages |

#### Example output

```
=== Ingestion Metrics ===
File: /data/slides.pdf   Chunks: 42

Parse Quality
  Total pages  : 10
  Empty pages  : 1
  Error rate   : 10.0%

Chunk Size Distribution (tokens, encoding=cl100k_base)
  P50  : 312
  P90  : 387
  P95  : 398
  Min  : 45   Max : 400

End-to-End Latency
  Parse          :   120 ms
  Chunk          :    18 ms
  Embed + store  :  4320 ms
  Total          :  4458 ms
```

#### Changing the tokenizer encoding

`--encoding` controls the tiktoken encoding for **both** chunking and measurement, so they always stay in sync:

```bash
python -m app.rag.ingest path/to/file.pdf --metrics --encoding p50k_base
```

Using `ingest()` programmatically:

```python
from app.rag.ingest import ingest

n = ingest("file.pdf", metrics=True, encoding_name="cl100k_base")
```