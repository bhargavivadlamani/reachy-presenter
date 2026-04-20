# python -m app.rag.generate "<query>" [--gen-provider ...] [--gen-model ...] [--stream] ...

import argparse, os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from app.rag.retrieve import retrieve, build_context

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using only the provided context. "
     "Cite sources as [N] where N is the number shown before each context block. "
     "If the context does not contain the answer, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {query}"),
])


def _get_llm(provider: str, model: str):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        from langchain_core.rate_limiters import InMemoryRateLimiter
        rate_limiter = InMemoryRateLimiter(requests_per_second=1/6)
        return ChatOpenAI(
            model=model, 
            api_key=os.environ["OPENAI_API_KEY"], 
            base_url=os.environ["OPENAI_BASE_URL"], 
            rate_limiter=rate_limiter 
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model)
    elif provider == "vultr":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ["VULTR_API_KEY"],
            base_url="https://api.vultrinference.com/v1",
        )
    raise ValueError(f"Unknown provider: {provider}")


# @traceable(name="rag_generate")
def generate(query: str, docs, provider: str = "openai", model: str = "gpt-4o-mini", stream: bool = False):
    context = build_context(docs)
    chain = _PROMPT | _get_llm(provider, model) | StrOutputParser()
    if stream:
        return chain.stream({"context": context, "query": query})
    return chain.invoke({"context": context, "query": query})


# @traceable(name="rag_retrieve_generate")
def retrieve_generate(
    query: str,
    collection: str = "reachy_collection",
    provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    sparse_model: str = "Qdrant/bm25",
    reranker: str = "cross-encoder",
    top_n: int = 5,
    retriever_k: int = 20,
    gen_provider: str = "openai",
    gen_model: str = "gpt-4o-mini",
    stream: bool = False,
):
    docs = retrieve(query, collection, provider, embedding_model, sparse_model, reranker, top_n, retriever_k)
    return generate(query, docs, gen_provider, gen_model, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--collection", default="reachy_collection")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--embedding-model", default="nomic-embed-text", dest="embedding_model")
    parser.add_argument("--sparse-model", default="Qdrant/bm25", dest="sparse_model")
    parser.add_argument("--reranker", choices=["cross-encoder", "cohere"], default="cross-encoder")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    parser.add_argument("--retriever-k", type=int, default=20, dest="retriever_k")
    parser.add_argument("--gen-provider", choices=["openai", "ollama", "gemini", "vultr"], default="openai", dest="gen_provider")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    result = retrieve_generate(
        args.query,
        collection=args.collection,
        provider=args.provider,
        embedding_model=args.embedding_model,
        sparse_model=args.sparse_model,
        reranker=args.reranker,
        top_n=args.top_n,
        retriever_k=args.retriever_k,
        gen_provider=args.gen_provider,
        gen_model=args.model,
        stream=args.stream,
    )

    if args.stream:
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result)
