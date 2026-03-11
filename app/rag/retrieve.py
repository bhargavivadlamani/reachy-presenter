# python -m app.rag.retrieve "your query" [--collection ...] [--provider ...] [--model ...] [--sparse-model ...] [--reranker cross-encoder|cohere] [--top-n 5]

import argparse, os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from app.rag.ingest import QDRANT_URL, get_embeddings

load_dotenv()

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list[Document], reranker: str = "cross-encoder", top_k: int = 5) -> list[Document]:
    if reranker == "cross-encoder":
        return _rerank_cross_encoder(query, chunks, top_k)
    elif reranker == "cohere":
        return _rerank_cohere(query, chunks, top_k)
    raise ValueError(f"Unknown reranker: {reranker}")


def _rerank_cross_encoder(query: str, chunks: list[Document], top_k: int) -> list[Document]:
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    scores = model.predict([(query, c.page_content) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


def _rerank_cohere(query: str, chunks: list[Document], top_k: int) -> list[Document]:
    import cohere  # optional dep — install separately if using this reranker
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    results = co.rerank(
        query=query,
        documents=[c.page_content for c in chunks],
        top_n=top_k,
        model="rerank-english-v3.0",
    )
    return [chunks[r.index] for r in results.results]


def build_context(chunks: list[Document]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        header = f"[{i}] Source: {meta.get('source', 'unknown')}, Page: {meta.get('page', '?')}"
        parts.append(f"{header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def retrieve(
    query: str,
    collection: str = "reachy_collection",
    provider: str = "ollama",
    model: str = "nomic-embed-text",
    sparse_model: str = "Qdrant/bm25",
    reranker: str = "cross-encoder",
    top_n: int = 3,
) -> list[Document]:
    embeddings = get_embeddings(provider, model)
    sparse = FastEmbedSparse(model_name=sparse_model)
    client = QdrantClient(url=QDRANT_URL)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    candidates = store.similarity_search(query, k=20)
    return rerank(query, candidates, reranker=reranker, top_k=top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--sparse-model", default="Qdrant/bm25", dest="sparse_model")
    parser.add_argument("--reranker", choices=["cross-encoder", "cohere"], default="cross-encoder")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    args = parser.parse_args()

    chunks = retrieve(args.query, args.collection, args.provider, args.model, args.sparse_model, args.reranker, args.top_n)
    print(build_context(chunks))
