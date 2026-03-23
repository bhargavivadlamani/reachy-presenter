# Generation Evaluation README

Offline evaluation of the RAG generation stage using [Ragas](https://docs.ragas.io) as an LLM-as-a-judge framework.

---

## What this evaluates

The retrieval eval (`eval_retrieval.py`) tells you whether the right chunks are being surfaced. The generation eval (`eval_generation.py`) tells you what the LLM does with those chunks — whether it stays faithful to them, answers the question, and produces correct responses.

---

## Metrics

### Faithfulness
**Does the answer only use information from the retrieved context?**

The judge LLM decomposes the generated answer into individual claims, then checks each claim against the retrieved chunks. Score = fraction of claims that are supported.

- 1.0 = fully grounded, no hallucinations
- 0.0 = answer ignores context entirely

This is the most important metric for RAG. A low score means the LLM is hallucinating beyond what was retrieved.

### Answer Relevancy
**Does the answer actually address the question?**

The judge generates several reverse-questions from the answer, then measures semantic similarity between those and the original question (using embeddings). A relevant answer should imply questions similar to the one asked.

- High score = answer stays on topic
- Low score = answer is off-topic, evasive, or padded

Requires embeddings — defaults to `text-embedding-3-small` for the judge regardless of retrieval embedding model. The two embedding spaces are independent.

### Answer Correctness
**How close is the answer to the reference (gold) answer?**

Combines factual overlap (F1 score on claims) and semantic similarity between generated and reference answer. Only computed when the testset includes `reference_answer` fields.

- Requires a reference answer per question — generated during testset creation or hand-annotated
- A `nan` result means scoring jobs timed out during evaluation, not that reference answers are missing

---

## Two LLM roles

The eval uses two separate LLMs that serve different purposes:

| Parameter | Purpose | Default |
|---|---|---|
| `--gen-provider` / `--gen-model` | Generates answers from retrieved chunks — the system under test | `openai/gpt-4o-mini` |
| `--judge-provider` / `--judge-model` | Scores the generated answers (LLM-as-a-judge) | `openai/gpt-4o-mini` |

Keeping them separate lets you evaluate a cheap or local model (e.g. `--gen-provider ollama --gen-model llama3`) while using a stronger judge (e.g. `--judge-model gpt-4o`).

---

## Testsets

Testsets are JSON files with the following schema:

```json
[
  {
    "question": "What is the role of context in multi-agent systems?",
    "reference_contexts": ["chunk text from the document ..."],
    "reference_answer": "Context is critical for ..."
  }
]
```

- `reference_contexts` — required, used by Ragas to assess faithfulness
- `reference_answer` — optional, enables Answer Correctness scoring

**Freshly generated testsets** are saved automatically to `app/data/<source_file_stem>_testset.json`. Use `--save-testset` to override the path.

**Testset generation** uses Ragas's `TestsetGenerator` with:
- 35% single-hop questions (answerable from one chunk)
- 65% multi-hop questions (require connecting facts across multiple chunks)

Reference answers are generated separately using a plain prompt (no citation format) to avoid `[N]`-style markers polluting the gold answers.

---

## Usage

### Run with existing testset
```bash
python -m app.rag.eval_generation path/to/file.pdf \
  --testset app/data/my_testset.json \
  --collection my_collection
```

### Generate fresh testset and run eval
```bash
python -m app.rag.eval_generation path/to/file.pdf \
  --collection my_collection \
  --testset-size 20
# Testset auto-saved to app/data/<file_stem>_testset.json
```

### Use a stronger judge model
```bash
python -m app.rag.eval_generation path/to/file.pdf \
  --testset app/data/my_testset.json \
  --collection my_collection \
  --gen-provider ollama --gen-model llama3 \
  --judge-provider openai --judge-model gpt-4o
```

---

## All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `file` | required | Path to PDF/PPTX to evaluate against |
| `--collection` | `reachy_collection` | Qdrant collection name |
| `--provider` | `ollama` | Embedding provider for retrieval |
| `--embedding-model` | `nomic-embed-text` | Embedding model for retrieval |
| `--sparse-model` | `Qdrant/bm25` | Sparse model for hybrid search |
| `--reranker` | `cross-encoder` | Reranker: `cross-encoder` or `cohere` |
| `--top-n` | `5` | Final chunks passed to LLM after reranking |
| `--retriever-k` | `20` | Candidate pool size before reranking |
| `--gen-provider` | `openai` | LLM provider for answer generation |
| `--gen-model` | `gpt-4o-mini` | LLM model for answer generation |
| `--judge-provider` | `openai` | LLM provider for scoring |
| `--judge-model` | `gpt-4o-mini` | LLM model for scoring |
| `--testset` | `None` | Path to existing testset JSON |
| `--testset-size` | `20` | Number of questions to generate if no testset |
| `--save-testset` | auto | Override auto-save path for generated testset |
| `--parser` | `pdfplumber` | Document parser |
| `--chunk-size` | `400` | Tokens per chunk |
| `--chunk-overlap` | `60` | Token overlap between chunks |

---

## Logs

Each run writes to `app/rag/eval_logs/eval_gen_<timestamp>.json`:

```json
{
  "timestamp": "2026-03-23T20:16:19Z",
  "file": "/path/to/file.pdf",
  "collection": "eval-gen-collection",
  "gen_provider": "openai",
  "gen_model": "gpt-4o-mini",
  "n_questions": 100,
  "metrics_computed": ["faithfulness", "answer_relevancy", "answer_correctness"],
  "faithfulness": 0.83,
  "answer_relevancy": 0.80,
  "answer_correctness": null
}
```

`answer_correctness` is `null` when no reference answers exist in the testset, and `NaN` when scoring timed out.

---

## Ragas version notes (0.4.x)

These are hard-won lessons from integrating Ragas 0.4.3:

**Use `ragas.metrics` (not `ragas.metrics.collections`) with `ragas.evaluate`**

`ragas.metrics.collections` metrics are incompatible with `ragas.evaluate` in 0.4.3 — they pass an `isinstance` check that `ragas.evaluate` rejects. Stick with `ragas.metrics` imports (deprecation warnings are acceptable until v1.0).

**The judge LLM must use `LangchainLLMWrapper`**

`ragas.metrics` (non-collections) accepts `LangchainLLMWrapper(ChatOpenAI(...))`. Do not use `.bind()` or `.with_structured_output()` — these return `RunnableBinding` objects that Ragas accesses `.temperature` on directly, causing `AttributeError`.

**Ragas testset generator needs JSON mode**

When using OpenAI for testset generation, pass `model_kwargs={"response_format": {"type": "json_object"}}` directly to `ChatOpenAI`. This must be set on a separate LLM instance from the one used for plain-text reference answer generation — `json_object` mode requires the prompt to contain the word "json", which the reference answer prompt does not.

**`MultiHopAbstractQuerySynthesizer` has a persona naming bug**

In Ragas 0.4.x, persona names get a `_1` suffix during knowledge graph construction but are looked up by the original name, causing a `KeyError`. Use `MultiHopSpecificQuerySynthesizer` instead.

**Judge embeddings are independent of retrieval embeddings**

`AnswerRelevancy` and `AnswerCorrectness` use the judge embeddings only to compare text strings at scoring time. They have no connection to the Qdrant collection or retrieval embedding space — no dimension mismatch is possible.
