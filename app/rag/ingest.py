# python -m app.rag.ingest path/to/file.pdf [--provider openai|ollama] [--model MODEL]

import argparse, hashlib, os
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.parsers.parsers import parse

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def get_embeddings(provider: str, model: str):
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    elif provider == "ollama":
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown provider: {provider}")

def ingest(file_path: str, provider: str = "ollama", model: str = "nomic-embed-text", collection: str = "reachy_collection", parser: str = "pdfplumber", sparse_model: str = "Qdrant/bm25") -> int:
    file_path = os.path.abspath(file_path)
    doc_id = hashlib.sha256(file_path.encode()).hexdigest()

    # 1. Load data via parsers module
    data = parse(file_path, parser=parser)

    # 2. Wrap each data instance as a LangChain Document with metadata
    docs = [
        Document(
            page_content=text,
            metadata={
                "source": file_path,
                "doc_id": doc_id,
                "page": i,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": model,
            },
        )
        for i, text in enumerate(data)
        if text.strip()
    ]

    # 3. Split into chunks (TokenTextSplitter uses tiktoken under the hood)
    splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    # 4. Delete existing chunks for this document (idempotency)
    qdrant = QdrantClient(url=QDRANT_URL)
    try:
        qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))]
            ),
        )
    except Exception:
        pass  # collection doesn't exist yet on first run

    # 5. Embed + store (LangChain handles collection creation, batching, upsert)
    embeddings = get_embeddings(provider, model)
    sparse = FastEmbedSparse(model_name=sparse_model)
    QdrantVectorStore.from_documents(
        chunks, embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        url=QDRANT_URL, collection_name=collection,
    )

    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--parser", default="pdfplumber")
    parser.add_argument("--sparse-model", default="Qdrant/bm25", dest="sparse_model")
    parser.add_argument("--eval", type=int, choices=[0, 1], default=1,
                        help="1 = run retrieval eval after ingestion")
    args = parser.parse_args()

    n = ingest(args.file, args.provider, args.model, args.collection, args.parser, args.sparse_model)
    print(f"Ingested {n} chunks from {args.file} using {args.provider}")

    if args.eval:
        from app.rag.eval import eval_retrieval
        eval_retrieval(
            file_path=args.file,
            provider=args.provider,
            model=args.model,
            collection=args.collection,
            sparse_model=args.sparse_model,
        )
