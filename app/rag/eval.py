# python -m app.rag.eval path/to/file.pdf [options]
# See app/rag/README.md for full documentation.

import argparse, json, math, os
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from app.rag.ingest import QDRANT_URL, get_embeddings
from app.rag.retrieve import rerank
from app.parsers.parsers import parse
from ragas.testset import TestsetGenerator
from langchain_ollama import ChatOllama

load_dotenv()


def _get_generator_llm(provider: str, model: str):
    if provider == "ollama":
        return ChatOllama(model=model)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.environ["GOOGLE_API_KEY"], convert_system_message_to_human=True, model_kwargs={"response_mime_type": "application/json"})
    raise ValueError(f"Unknown generator provider: {provider}")


def _generate_testset(docs: list[Document], size: int, generator_provider: str, generator_model: str, embeddings) -> list[dict]:
    llm = _get_generator_llm(generator_provider, generator_model)
    generator = TestsetGenerator.from_langchain(llm, embeddings)
    result = generator.generate_with_langchain_docs(docs, testset_size=size)
    testset = []
    for row in result.to_pandas().itertuples():
        testset.append({
            "question": row.user_input,
            "reference_contexts": list(row.reference_contexts),
        })
    return testset


def _load_or_generate_testset(file_path: str, docs: list[Document], args, embeddings) -> list[dict]:
    if args.testset:
        with open(args.testset) as f:
            data = json.load(f)
        for item in data:
            if "question" not in item or "reference_contexts" not in item:
                raise ValueError("Each testset entry must have 'question' and 'reference_contexts' keys")
        return data

    testset = _generate_testset(docs, args.testset_size, args.generator_provider, args.generator_model, embeddings)

    if args.save_testset:
        with open(args.save_testset, "w") as f:
            json.dump(testset, f, indent=2)
        print(f"Testset saved to {args.save_testset}")

    return testset


def _is_relevant(chunk_text: str, reference_contexts: list[str], threshold: float = 0.3) -> bool:
    """
    Checks if a chunk is relevant to the reference contexts using Jaccard overlap.
    Jaccard overlap (or Jaccard similarity) is a measure of similarity between two sets:
    J(A, B) = |A ∩ B| / |A ∪ B|
    It's the size of the intersection divided by the size of the union. Result is always between 0 and 1
    """
    chunk_words = set(chunk_text.lower().split())
    for ctx in reference_contexts:
        ctx_words = set(ctx.lower().split())
        if not chunk_words and not ctx_words:
            continue
        intersection = chunk_words & ctx_words
        union = chunk_words | ctx_words
        if union and len(intersection) / len(union) >= threshold:
            return True
    return False


def _compute_metrics(ranked_chunks: list[Document], reference_contexts: list[str], k: int) -> dict:
    top_k = ranked_chunks[:k]
    relevance = [1 if _is_relevant(c.page_content, reference_contexts) else 0 for c in top_k]

    relevant_count = sum(relevance)
    recall = min(relevant_count / len(reference_contexts), 1.0) if reference_contexts else 0.0
    precision = relevant_count / k if k else 0.0
    hit = 1.0 if relevant_count > 0 else 0.0

    mrr = 0.0
    for i, r in enumerate(relevance):
        if r:
            mrr = 1.0 / (i + 1)
            break

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0

    return {"recall": recall, "precision": precision, "hit": hit, "mrr": mrr, "ndcg": ndcg}


def _chunk_diversity(chunks: list[Document]) -> float:
    """Mean pairwise cosine similarity of TF vectors (0 = diverse, 1 = identical)."""
    if len(chunks) < 2:
        return 0.0
    texts = [c.page_content.lower().split() for c in chunks]
    vocab = sorted({w for t in texts for w in t})
    idx = {w: i for i, w in enumerate(vocab)}
    vecs = np.zeros((len(texts), len(vocab)))
    for i, tokens in enumerate(texts):
        for w in tokens:
            vecs[i, idx[w]] += 1
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vecs / norms
    sim = normed @ normed.T
    n = len(chunks)
    pairs = [sim[i, j] for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(pairs))


def _avg(metrics_list: list[dict]) -> dict:
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def _print_report(file_path: str, n_questions: int, reranker: str, top_n: int,
                  stage1_avg: dict, stage2_avg: dict, delta: dict, retriever_k: int = 20,
                  avg_diversity: dict = None):
    print("\n=== Retrieval Evaluation Metrics ===")
    print(f"File: {file_path}   Test questions: {n_questions}   Reranker: {reranker}   top-n: {top_n}\n")

    print(f"Stage 1: Retriever (k={retriever_k})")
    print(f"  Recall@{retriever_k:<5}   :  {stage1_avg['recall']:.2f}")
    print(f"  Precision@{retriever_k:<3}   :  {stage1_avg['precision']:.2f}")
    print(f"  Hit@{retriever_k:<7}   :  {stage1_avg['hit']:.2f}")
    print(f"  MRR           :  {stage1_avg['mrr']:.2f}")
    print(f"  NDCG@{retriever_k:<6}   :  {stage1_avg['ndcg']:.2f}")

    print(f"\nStage 2: Reranker (k={top_n})")
    print(f"  Recall@{top_n:<5}   :  {stage2_avg['recall']:.2f}")
    print(f"  Precision@{top_n:<3}   :  {stage2_avg['precision']:.2f}")
    print(f"  Hit@{top_n:<7}   :  {stage2_avg['hit']:.2f}")
    print(f"  MRR           :  {stage2_avg['mrr']:.2f}")
    print(f"  NDCG@{top_n:<6}   :  {stage2_avg['ndcg']:.2f}")

    print(f"\nReranker Delta (Stage 2 @ k={top_n}  vs  Stage 1 @ k={top_n})")
    for key in ("recall", "precision", "hit", "mrr", "ndcg"):
        sign = "+" if delta[key] >= 0 else ""
        print(f"  {key.capitalize():<13} :  {sign}{delta[key]:.2f}")

    if avg_diversity:
        print(f"\nChunk Diversity (mean pairwise cosine sim; lower = more diverse)")
        print(f"  Stage 1 (k={retriever_k})  :  {avg_diversity['stage1']:.3f}")
        print(f"  Stage 2 (k={top_n})    :  {avg_diversity['stage2']:.3f}")


def _log_run(record: dict):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "offline_eval_logs")
    log_path = os.path.join(log_dir, f"eval_{ts}.json")
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRun logged to {log_path}")


def eval_retrieval(
    file_path: str,
    provider: str = "ollama",
    model: str = "nomic-embed-text",
    collection: str = "reachy_collection",
    sparse_model: str = "Qdrant/bm25",
    reranker: str = "cross-encoder",
    top_n: int = 5,
    testset_size: int = 20,
    testset: str = None,
    save_testset: str = None,
    generator_provider: str = "openai",
    generator_model: str = "mistral",
    retriever_k: int = 20,
):
    file_path = os.path.abspath(file_path)

    # 1. Re-parse file for Ragas document objects
    raw_pages = parse(file_path)
    docs = [Document(page_content=text, metadata={"source": file_path, "page": i})
            for i, text in enumerate(raw_pages) if text.strip()]

    embeddings = get_embeddings(provider, model)

    # Namespace args for _load_or_generate_testset
    class _Args:
        pass
    _args = _Args()
    _args.testset = testset
    _args.save_testset = save_testset
    _args.testset_size = testset_size
    _args.generator_provider = generator_provider
    _args.generator_model = generator_model

    # 2. Load or generate testset
    testset_data = _load_or_generate_testset(file_path, docs, _args, embeddings)

    # 3. Build vector store
    client = QdrantClient(url=QDRANT_URL)
    sparse = FastEmbedSparse(model_name=sparse_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    # 4. Evaluate each question
    stage1_metrics = []
    stage2_metrics = []
    stage1_at_topn_metrics = []
    stage1_diversity = []
    stage2_diversity = []

    for item in testset_data:
        q = item["question"]
        ref_ctx = item["reference_contexts"]

        candidates = store.similarity_search(q, k=retriever_k)
        final = rerank(q, candidates, reranker=reranker, top_k=top_n)

        stage1_metrics.append(_compute_metrics(candidates, ref_ctx, retriever_k))
        stage1_at_topn_metrics.append(_compute_metrics(candidates, ref_ctx, top_n))
        stage2_metrics.append(_compute_metrics(final, ref_ctx, top_n))
        stage1_diversity.append(_chunk_diversity(candidates))
        stage2_diversity.append(_chunk_diversity(final))

    # 5. Average and compute delta
    stage1_avg = _avg(stage1_metrics)
    stage2_avg = _avg(stage2_metrics)
    stage1_at_topn_avg = _avg(stage1_at_topn_metrics)
    delta = {k: stage2_avg[k] - stage1_at_topn_avg[k] for k in stage2_avg}
    avg_diversity = {
        "stage1": float(np.mean(stage1_diversity)),
        "stage2": float(np.mean(stage2_diversity)),
    }

    # 6. Print report
    _print_report(file_path, len(testset_data), reranker, top_n,
                  stage1_avg, stage2_avg, delta, retriever_k, avg_diversity)

    # 7. Log run
    _log_run({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": file_path,
        "collection": collection,
        "provider": provider,
        "model": model,
        "reranker": reranker,
        "top_n": top_n,
        "retriever_k": retriever_k,
        "n_questions": len(testset_data),
        "stage1": stage1_avg,
        "stage2": stage2_avg,
        "delta": delta,
        "diversity": avg_diversity,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--sparse-model", default="Qdrant/bm25", dest="sparse_model")
    parser.add_argument("--reranker", choices=["cross-encoder", "cohere"], default="cross-encoder")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    parser.add_argument("--testset-size", type=int, default=20, dest="testset_size")
    parser.add_argument("--testset", default=None)
    parser.add_argument("--save-testset", default=None, dest="save_testset")
    parser.add_argument("--generator-provider", choices=["ollama", "openai", "gemini"], default="ollama", dest="generator_provider")
    parser.add_argument("--generator-model", default="mistral", dest="generator_model")
    args = parser.parse_args()

    eval_retrieval(
        file_path=args.file,
        provider=args.provider,
        model=args.model,
        collection=args.collection,
        sparse_model=args.sparse_model,
        reranker=args.reranker,
        top_n=args.top_n,
        testset_size=args.testset_size,
        testset=args.testset,
        save_testset=args.save_testset,
        generator_provider=args.generator_provider,
        generator_model=args.generator_model,
    )
