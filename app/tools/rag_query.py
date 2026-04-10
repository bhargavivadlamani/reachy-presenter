import os

from app.tools.load_presentation import get_collection_name
from app.rag.retrieve import retrieve, build_context

_RAG_EMBED_PROVIDER = os.getenv("RAG_EMBED_PROVIDER", "ollama")
_RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")


def rag_query(query: str, collection_name: str = "") -> str:
    """Retrieve relevant document chunks from the knowledge base to help answer a question.

    Call this whenever a student asks something that needs factual grounding —
    questions about slide content, deeper topic explanations, or anything from
    ingested reference documents (textbooks, papers, notes). The returned
    context contains numbered source citations; use them in your answer.

    Args:
        query: The student's question in natural language.
        collection_name: Qdrant collection to search. Leave empty to use the
            currently loaded presentation's collection automatically. Pass an
            explicit name to search a different ingested document.

    Returns:
        Formatted context chunks with source citations, or an error message.
    """
    collection = collection_name or get_collection_name()
    if not collection:
        return "No collection available. Load a presentation first or specify a collection_name."

    try:
        docs = retrieve(
            query=query,
            collection=collection,
            provider=_RAG_EMBED_PROVIDER,
            model=_RAG_EMBED_MODEL,
        )
        if not docs:
            return "No relevant content found for that query."
        return build_context(docs)
    except Exception as e:
        return f"Retrieval failed: {e}"
