# python -m app.rag.ingest path/to/file.pdf [--provider openai|ollama] [--model MODEL]

import argparse, hashlib, os
from datetime import datetime, timezone
from time import perf_counter
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

def ingest(file_path: str, provider: str = "ollama", model: str = "nomic-embed-text",
           collection: str = "reachy_collection", parser: str = "pdfplumber",
           metrics: bool = False, encoding_name: str = "cl100k_base") -> int:
    from app.parsers.parsers import parse
    file_path = os.path.abspath(file_path)
    doc_id = hashlib.sha256(file_path.encode()).hexdigest()

    # 1. Parse
    t0 = perf_counter()
    data = parse(file_path, parser=parser)
    parse_ms = (perf_counter() - t0) * 1000
    empty_pages = sum(1 for t in data if not t.strip()) if metrics else 0

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
    splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=60, encoding_name=encoding_name)
    t0 = perf_counter()
    chunks = splitter.split_documents(docs)
    chunk_ms = (perf_counter() - t0) * 1000

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
    t0 = perf_counter()
    QdrantVectorStore.from_documents(
        chunks, embeddings, url=QDRANT_URL, collection_name=collection,
    )
    store_ms = (perf_counter() - t0) * 1000

    if metrics:
        import tiktoken
        import numpy as np
        enc = tiktoken.get_encoding(encoding_name)
        sizes = [len(enc.encode(c.page_content)) for c in chunks]
        p50, p90, p95 = np.percentile(sizes, [50, 90, 95])
        total_pages = len(data)
        error_rate = (empty_pages / total_pages * 100) if total_pages else 0.0
        total_ms = parse_ms + chunk_ms + store_ms
        print(f"""
=== Ingestion Metrics ===
File: {file_path}   Chunks: {len(chunks)}

Parse Quality
  Total pages  : {total_pages}
  Empty pages  : {empty_pages}
  Error rate   : {error_rate:.1f}%

Chunk Size Distribution (tokens, encoding={encoding_name})
  P50  : {int(p50)}
  P90  : {int(p90)}
  P95  : {int(p95)}
  Min  : {min(sizes)}   Max : {max(sizes)}

End-to-End Latency
  Parse          : {parse_ms:6.0f} ms
  Chunk          : {chunk_ms:6.0f} ms
  Embed + store  : {store_ms:6.0f} ms
  Total          : {total_ms:6.0f} ms""")

    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--parser", default="pdfplumber")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--encoding", default="cl100k_base")
    args = parser.parse_args()

    n = ingest(args.file, args.provider, args.model, args.collection, args.parser,
               metrics=args.metrics, encoding_name=args.encoding)
    print(f"Ingested {n} chunks from {args.file} using {args.provider}")
