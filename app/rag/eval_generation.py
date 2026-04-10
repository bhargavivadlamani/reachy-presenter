# python -m app.rag.eval_generation path/to/file.pdf [options]
# See app/rag/README.md for full documentation.

import argparse, json, os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.rate_limiters import InMemoryRateLimiter
from app.rag.eval_retrieval import _get_generator_llm, _load_or_generate_testset, _log_run, _generate_testset
from app.rag.retrieve import retrieve
from app.rag.generate import generate
from app.rag.ingest import get_embeddings
from app.parsers.parsers import parse

load_dotenv(override=True)


def eval_generation(
    file_path: str,
    collection: str = "reachy_collection",
    provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    sparse_model: str = "Qdrant/bm25",
    reranker: str = "cross-encoder",
    top_n: int = 5,
    retriever_k: int = 20,
    gen_provider: str = "openai",
    gen_model: str = "gpt-4o-mini",
    testset_size: int = 20,
    testset: str = None,
    save_testset: str = None,
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o-mini",
    parser: str = "pdfplumber",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    file_path = os.path.abspath(file_path)

    # 1. Parse file into chunks (needed for testset generation)
    raw_pages = parse(file_path, parser=parser)
    docs = [Document(page_content=text, metadata={"source": file_path, "page": i})
            for i, text in enumerate(raw_pages) if text.strip()]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings(provider, embedding_model)

    # Namespace args for _load_or_generate_testset
    class _Args:
        pass
    _args = _Args()
    _args.testset = testset
    _args.save_testset = None  # handled below after reference answers are added
    _args.testset_size = testset_size
    _args.model_provider = judge_provider
    _args.model = judge_model

    # 2. Load or generate testset (with reference answers if generating fresh)
    if testset:
        _args.save_testset = save_testset
        testset_data = _load_or_generate_testset(file_path, chunks, _args, embeddings)
    else:
        testset_data = _generate_testset(chunks, testset_size, judge_provider, judge_model, embeddings, with_reference_answers=True)
        if not save_testset:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            stem = os.path.splitext(os.path.basename(file_path))[0]
            save_testset = os.path.join(data_dir, f"{stem}_testset.json")
        with open(save_testset, "w") as f:
            json.dump(testset_data, f, indent=2)
        print(f"Testset saved to {save_testset}")

    # 3. Ingest if collection doesn't exist
    from qdrant_client import QdrantClient
    from app.rag.ingest import QDRANT_URL, ingest
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(collection):
        print(f"Collection '{collection}' not found. Ingesting {file_path}...")
        n = ingest(file_path, provider=provider, model=embedding_model, collection=collection,
                   parser=parser, sparse_model=sparse_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Ingested {n} chunks.")

    # 4. Run retrieve + generate for each question
    results = []
    for item in testset_data:
        retrieved_docs = retrieve(
            item["question"], collection, provider, embedding_model,
            sparse_model, reranker, top_n, retriever_k,
        )
        answer = generate(item["question"], retrieved_docs, gen_provider, gen_model, stream=False)
        results.append((item, retrieved_docs, answer))

    # 5. Build Ragas dataset
    has_reference = all(item.get("reference_answer") for item, _, _ in results)
    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=[d.page_content for d in retrieved_docs],
            reference=item.get("reference_answer"),
        )
        for item, retrieved_docs, answer in results
    ]

    _JUDGE_EMBED_MODELS = {"openai": "text-embedding-3-small", "ollama": "nomic-embed-text", "gemini": "models/text-embedding-004", "vultr": "nomic-embed-text"}
    _JUDGE_EMBED_PROVIDERS = {"openai": "openai", "ollama": "ollama", "gemini": "gemini", "vultr": "ollama"}
    judge_llm = LangchainLLMWrapper(_get_generator_llm(judge_provider, judge_model))
    judge_emb = LangchainEmbeddingsWrapper(get_embeddings(_JUDGE_EMBED_PROVIDERS[judge_provider], _JUDGE_EMBED_MODELS[judge_provider]))

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_emb),
    ]
    if has_reference:
        metrics.append(AnswerCorrectness(llm=judge_llm, embeddings=judge_emb))

    # 6. Score
    _RATE_LIMITED_PROVIDERS = {"gemini"}
    run_cfg = RunConfig(max_workers=1, timeout=180) if judge_provider in _RATE_LIMITED_PROVIDERS else RunConfig()
    scores = evaluate(EvaluationDataset(samples=samples), metrics=metrics, run_config=run_cfg)
    scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

    # 7. Print report
    print("\n=== Generation Evaluation Metrics ===")
    print(f"File: {file_path}   Questions: {len(results)}   Gen model: {gen_provider}/{gen_model}\n")
    print(f"  Faithfulness       :  {scores_dict.get('faithfulness', 0.0):.2f}")
    print(f"  Answer Relevancy   :  {scores_dict.get('answer_relevancy', 0.0):.2f}")
    if has_reference:
        print(f"  Answer Correctness :  {scores_dict.get('answer_correctness', 0.0):.2f}")
    else:
        print("  Answer Correctness :  (skipped — no reference answers in testset)")

    # 8. Log run
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "eval_logs")
    log_path = os.path.join(log_dir, f"eval_gen_{ts}.json")
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file": file_path,
            "collection": collection,
            "gen_provider": gen_provider,
            "gen_model": gen_model,
            "n_questions": len(results),
            "metrics_computed": [m.name for m in metrics],
            "faithfulness": scores_dict.get("faithfulness"),
            "answer_relevancy": scores_dict.get("answer_relevancy"),
            "answer_correctness": scores_dict.get("answer_correctness") if has_reference else None,
        }, f, indent=2)
    print(f"\nRun logged to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--embedding-model", default="nomic-embed-text", dest="embedding_model")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--sparse-model", default="Qdrant/bm25", dest="sparse_model")
    parser.add_argument("--reranker", choices=["cross-encoder", "cohere"], default="cross-encoder")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    parser.add_argument("--retriever-k", type=int, default=20, dest="retriever_k")
    parser.add_argument("--gen-provider", choices=["openai", "ollama", "gemini", "vultr"], default="openai", dest="gen_provider")
    parser.add_argument("--gen-model", default="gpt-4o-mini", dest="gen_model")
    parser.add_argument("--testset-size", type=int, default=20, dest="testset_size")
    parser.add_argument("--testset", default=None)
    parser.add_argument("--save-testset", default=None, dest="save_testset")
    parser.add_argument("--judge-provider", choices=["ollama", "openai", "gemini", "vultr"], default="openai", dest="judge_provider")
    parser.add_argument("--judge-model", default="gpt-4o-mini", dest="judge_model")
    parser.add_argument("--parser", default="pdfplumber")
    parser.add_argument("--chunk-size", type=int, default=700, dest="chunk_size")
    parser.add_argument("--chunk-overlap", type=int, default=100, dest="chunk_overlap")
    args = parser.parse_args()

    eval_generation(
        file_path=args.file,
        collection=args.collection,
        provider=args.provider,
        embedding_model=args.embedding_model,
        sparse_model=args.sparse_model,
        reranker=args.reranker,
        top_n=args.top_n,
        retriever_k=args.retriever_k,
        gen_provider=args.gen_provider,
        gen_model=args.gen_model,
        testset_size=args.testset_size,
        testset=args.testset,
        save_testset=args.save_testset,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        parser=args.parser,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
