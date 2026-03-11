# python -m app.rag.ingest path/to/file.pdf [--provider openai|ollama] [--model MODEL]

import argparse, hashlib, os
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = "http://localhost:6333"

PROVIDERS = {
    "openai":  {"model": "text-embedding-3-small", "dims": 1536},
    "ollama":  {"model": "nomic-embed-text",        "dims": 768},
}

def get_embeddings(provider: str, model: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown provider: {provider}")

def load_slides(file_path: str) -> list[str]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from app.parsers.pdf_parser import extract_slides
    elif ext in (".pptx", ".ppt"):
        from app.parsers.pptx_parser import extract_slides
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return extract_slides(file_path)

def ingest(file_path: str, provider: str = "openai", model: str | None = None) -> int:
    file_path = os.path.abspath(file_path)
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    doc_id = hashlib.sha256(file_path.encode()).hexdigest()
    model = model or PROVIDERS[provider]["model"]
    collection = f"reachy_slides_{provider}"

    # 1. Load slides via existing parsers
    slides = load_slides(file_path)

    # 2. Wrap each slide as a LangChain Document with metadata
    docs = [
        Document(
            page_content=text,
            metadata={
                "source": file_path,
                "doc_id": doc_id,
                "slide_index": i,
                "file_type": ext,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": model,
            },
        )
        for i, text in enumerate(slides)
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
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    n = ingest(args.file, args.provider, args.model)
    print(f"Ingested {n} chunks from {args.file} using {args.provider}")
