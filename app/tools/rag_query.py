import os

from app.tools.load_presentation import get_collection_name
from app.rag.retrieve import retrieve, build_context

_RAG_EMBED_PROVIDER = os.getenv("RAG_EMBED_PROVIDER", "gemini")
_RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "models/text-embedding-004")
_RAG_RERANKER = os.getenv("RAG_RERANKER", "cohere")


_DEFAULT_COLLECTION = "testcollection2"


def rag_query(query: str, collection_name: str = "") -> str:
    """Search the Reachy Mini knowledge base to answer factual questions.

    The knowledge base contains official Reachy Mini documentation — hardware specs,
    SDK usage, sensors, motors, antennas, audio, camera, head tracking, and setup guides.
    Also contains any ingested presentation content.

    Call this whenever someone asks about:
    - Reachy Mini hardware, capabilities, or specifications
    - How to use the SDK or control the robot
    - Questions related to any loaded presentation content
    - Any factual topic that might be covered in the ingested documents

    Args:
        query: The question in natural language.
        collection_name: Qdrant collection to search. Leave empty to auto-select:
            uses the loaded presentation's collection if available, otherwise
            falls back to the default Reachy Mini knowledge base.

    Returns:
        Formatted context chunks with source citations, or an error message.
    """
    collection = collection_name or get_collection_name() or _DEFAULT_COLLECTION

    print(f"[rag_query] collection={collection!r} query={query!r}")
    try:
        docs = retrieve(
            query=query,
            collection=collection,
            provider=_RAG_EMBED_PROVIDER,
            model=_RAG_EMBED_MODEL,
            reranker=_RAG_RERANKER,
        )
        if not docs:
            print("[rag_query] No results returned.")
            return "No relevant content found for that query."
        print(f"[rag_query] Returning {len(docs)} chunks.")
        return build_context(docs)
    except Exception as e:
        print(f"[rag_query] Error: {e}")
        return f"Retrieval failed: {e}"
