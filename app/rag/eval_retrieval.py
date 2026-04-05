# python -m app.rag.eval_retrieval path/to/file.pdf [options]
# See app/rag/README.md for full documentation.

import argparse, json, math, os
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from langchain_text_splitters import TokenTextSplitter
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from app.rag.ingest import QDRANT_URL, get_embeddings
from app.rag.retrieve import rerank
from app.parsers.parsers import parse
from ragas.testset import TestsetGenerator
from langchain_ollama import ChatOllama

_REFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based solely on the provided context. Be concise."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

load_dotenv()


def _get_generator_llm(provider: str, model: str, json_mode: bool = False):
    if provider == "ollama":
        return ChatOllama(model=model)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        from langchain_core.rate_limiters import InMemoryRateLimiter
        rate_limiter = InMemoryRateLimiter(requests_per_second=1/6)
        kwargs = {"model_kwargs": {"response_format": {"type": "json_object"}}} if json_mode else {}
        return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"], rate_limiter=rate_limiter, **kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.environ["GOOGLE_API_KEY"], convert_system_message_to_human=True, response_mime_type="application/json", model_kwargs={"thinking_config": {"thinking_budget": 0}})
    elif provider == "vultr":
        from langchain_openai import ChatOpenAI
        kwargs = {"model_kwargs": {"response_format": {"type": "json_object"}}} if json_mode else {}
        return ChatOpenAI(
            model=model,
            api_key=os.environ["VULTR_API_KEY"],
            base_url="https://api.vultrinference.com/v1",
            **kwargs,
        )
    raise ValueError(f"Unknown model provider: {provider}")


def _generate_testset(chunks: list[Document], size: int, model_provider: str, model: str, embeddings, with_reference_answers: bool = False) -> list[dict]:
    llm = _get_generator_llm(model_provider, model)
    ragas_llm = _get_generator_llm(model_provider, model, json_mode=True)
    # LangChain's structured output methods (.with_structured_output(), .bind()) wrap the LLM in a RunnableSequence or RunnableBinding — they return a chain, not a ChatOpenAI instance.
    # Ragas's TestsetGenerator expects a raw LLM object and calls it internally using its own prompting/parsing logic. It accesses attributes like .temperature directly on the object, which breaks the moment you wrap it in any Runnable.
    generator = TestsetGenerator.from_langchain(ragas_llm, embeddings)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.35),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.65)
    ]
    result = generator.generate_with_chunks(chunks, testset_size=size, query_distribution=query_distribution)
    testset = []
    for row in result.to_pandas().itertuples():
        testset.append({
            "question": row.user_input,
            "reference_contexts": list(row.reference_contexts),
        })
    if with_reference_answers:
        chain = _REFERENCE_PROMPT | llm | StrOutputParser()
        for entry in testset:
            ctx = "\n\n".join(entry["reference_contexts"])
            entry["reference_answer"] = chain.invoke({"context": ctx, "question": entry["question"]})
    return testset


def _load_or_generate_testset(file_path: str, docs: list[Document], args, embeddings) -> list[dict]:
    if args.testset:
        with open(args.testset) as f:
            data = json.load(f)
        for item in data:
            if "question" not in item or "reference_contexts" not in item:
                raise ValueError("Each testset entry must have 'question' and 'reference_contexts' keys")
            # 'reference_answer' is optional — present enables Answer Correctness in generation eval
        return data

    testset = _generate_testset(docs, args.testset_size, args.model_provider, args.model, embeddings)

    save_path = args.save_testset
    if not save_path:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        stem = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(data_dir, f"{stem}_testset.json")
    with open(save_path, "w") as f:
        json.dump(testset, f, indent=2)
    print(f"Testset saved to {save_path}")

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
    log_dir = os.path.join(os.path.dirname(__file__), "eval_logs")
    log_path = os.path.join(log_dir, f"eval_{ts}.json")
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRun logged to {log_path}")


def eval_retrieval(
    file_path: str,
    embedding_provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    collection: str = "reachy_collection",
    sparse_model: str = "Qdrant/bm25",
    reranker: str = "cross-encoder",
    top_n: int = 5,
    testset_size: int = 20,
    testset: str = None,
    save_testset: str = None,
    model_provider: str = "openai",
    model: str = "mistral",
    retriever_k: int = 20,
    parser: str = "docling",
    chunk_size: int = 400,
    chunk_overlap: int = 60,
):
    file_path = os.path.abspath(file_path)

    # 1. Re-parse file for Ragas document objects
    raw_pages = parse(file_path, parser=parser)
    docs = [Document(page_content=text, metadata={"source": file_path, "page": i})
            for i, text in enumerate(raw_pages) if text.strip()]
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings(embedding_provider, embedding_model)

    # Namespace args for _load_or_generate_testset
    class _Args:
        pass
    _args = _Args()
    _args.testset = testset
    _args.save_testset = save_testset
    _args.testset_size = testset_size
    _args.model_provider = model_provider
    _args.model = model

    # 2. Load or generate testset
    testset_data = _load_or_generate_testset(file_path, chunks, _args, embeddings)

    # 3. Ingest if collection doesn't exist
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(collection):
        print(f"Collection '{collection}' not found. Ingesting {file_path}...")
        from app.rag.ingest import ingest
        n = ingest(file_path, provider=provider, model=embedding_model, collection=collection,
                   parser=parser, sparse_model=sparse_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Ingested {n} chunks.")

    # 4. Build vector store
    sparse = FastEmbedSparse(model_name=sparse_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    # 5. Evaluate each question
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

    # 6. Average and compute delta
    stage1_avg = _avg(stage1_metrics)
    stage2_avg = _avg(stage2_metrics)
    stage1_at_topn_avg = _avg(stage1_at_topn_metrics)
    delta = {k: stage2_avg[k] - stage1_at_topn_avg[k] for k in stage2_avg}
    avg_diversity = {
        "stage1": float(np.mean(stage1_diversity)),
        "stage2": float(np.mean(stage2_diversity)),
    }

    # 7. Print report
    _print_report(file_path, len(testset_data), reranker, top_n,
                  stage1_avg, stage2_avg, delta, retriever_k, avg_diversity)

    # 7. Log run
    _log_run({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": file_path,
        "collection": collection,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
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
    parser.add_argument("--embedding-provider", choices=["openai", "ollama"], default="ollama", dest="embedding_provider")
    parser.add_argument("--embedding-model", default="nomic-embed-text", dest="embedding_model")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--sparse-model", default="Qdrant/bm25", dest="sparse_model")
    parser.add_argument("--reranker", choices=["cross-encoder", "cohere"], default="cross-encoder")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    parser.add_argument("--testset-size", type=int, default=20, dest="testset_size")
    parser.add_argument("--testset", default=None)
    parser.add_argument("--save-testset", default=None, dest="save_testset")
    parser.add_argument("--model-provider", choices=["ollama", "openai", "gemini", "vultr"], default="ollama", dest="model_provider")
    parser.add_argument("--model", default="mistral", dest="model")
    parser.add_argument("--parser", default="pdfplumber")
    parser.add_argument("--chunk-size", type=int, default=400, dest="chunk_size")
    parser.add_argument("--chunk-overlap", type=int, default=60, dest="chunk_overlap")
    args = parser.parse_args()

    eval_retrieval(
        file_path=args.file,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        collection=args.collection,
        sparse_model=args.sparse_model,
        reranker=args.reranker,
        top_n=args.top_n,
        testset_size=args.testset_size,
        testset=args.testset,
        save_testset=args.save_testset,
        model_provider=args.model_provider,
        model=args.model,
        parser=args.parser,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
