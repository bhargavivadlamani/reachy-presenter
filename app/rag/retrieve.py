# python -m app.rag.retrieve "your query" [--collection ...] [--provider ...] [--model ...] [--sparse-model ...] [--reranker cross-encoder|cohere] [--top-n 5]

import argparse, os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langsmith import traceable
from qdrant_client import QdrantClient
from app.rag.ingest import QDRANT_URL, QDRANT_API_KEY, get_embeddings


def rerank(query: str, chunks: list[Document], reranker: str = "cross-encoder", top_k: int = 5) -> list[Document]:
    if reranker == "cross-encoder":
        return _rerank_cross_encoder(query, chunks, top_k)
    elif reranker == "cohere":
        return _rerank_cohere(query, chunks, top_k)
    raise ValueError(f"Unknown reranker: {reranker}")

@traceable(name="rerank using cross encoder")
def _rerank_cross_encoder(query: str, chunks: list[Document], top_k: int) -> list[Document]:
    from sentence_transformers import CrossEncoder
    model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = CrossEncoder(model).predict([(query, c.page_content) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]

@traceable(name="rerank using cohere")
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


@traceable(name="retrieval_run")
def retrieve(
    query: str,
    collection: str = "reachy_collection",
    provider: str = "ollama",
    model: str = "nomic-embed-text",
    sparse_model: str = "Qdrant/bm25",
    reranker: str = "cohere",
    top_n: int = 3,
    retriever_k: int = 20,
) -> list[Document]:

    """
    Retrieves relevant documents from the a vector store using hybrid search. 
    Use it to get the relevant context for the query and ground the answers.
    
    Args:
        query: The query to search for.
        collection: The name of the collection to search in.
        provider: The embedding provider to use.
        model: The embedding model to use.
        sparse_model: The sparse embedding model to use.
        reranker: The reranker to use.
        top_n: The number of documents to return.
        retriever_k: The number of documents to retrieve.
    
    Returns:
        A list of relevant documents.
    """
    embeddings = get_embeddings(provider, model)
    sparse = FastEmbedSparse(model_name=sparse_model)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    @traceable(name="hybrid_search")
    def hybrid_search(q: str) -> list[Document]:
        return store.similarity_search(q, k=retriever_k)

    candidates = hybrid_search(query)
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
    parser.add_argument("--retriever-k", type=int, default=20, dest="retriever_k")
    args = parser.parse_args()

    chunks = retrieve(args.query, args.collection, args.provider, args.model, args.sparse_model, args.reranker, args.top_n, args.retriever_k)
    print(build_context(chunks))
