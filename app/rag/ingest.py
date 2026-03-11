# python -m app.rag.ingest path/to/file.pdf [--provider openai|ollama] [--model MODEL]

import argparse, hashlib, os
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = "http://localhost:6333"


def get_embeddings(provider: str, model: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown provider: {provider}")

def ingest(file_path: str, provider: str = "ollama", model: str = "nomic-embed-text", collection: str = "reachy_collection", parser: str = "pdfplumber") -> int:
    from app.parsers.parsers import parse
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
                "index": i,
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
    QdrantVectorStore.from_documents(
        chunks, embeddings, url=QDRANT_URL, collection_name=collection,
    )

    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--parser", default="pdfplumber")
    args = parser.parse_args()

    n = ingest(args.file, args.provider, args.model, args.collection, args.parser)
    print(f"Ingested {n} chunks from {args.file} using {args.provider}")
