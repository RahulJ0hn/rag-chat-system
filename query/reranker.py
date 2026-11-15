"""
Reranks Qdrant search results using Cohere Rerank API.
Called between retriever.py and prompt_builder.py.
"""

import os
from typing import Any

import cohere

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))


def rerank_results(query: str, results: list[Any]) -> list[Any]:
    """
    Takes top-10 Qdrant results, reranks with Cohere,
    returns top RERANKER_TOP_N results in new order.
    If Cohere is unavailable, return original results as fallback.
    """
    if not results:
        return results

    documents = [r.payload["text"] for r in results]

    try:
        response = co.rerank(
            model=os.getenv("RERANKER_MODEL", "rerank-english-v3.0"),
            query=query,
            documents=documents,
            top_n=int(os.getenv("RERANKER_TOP_N", 5)),
        )

        # Reorder original Qdrant results by Cohere's ranking
        reranked = [results[hit.index] for hit in response.results]

        # Attach Cohere relevance score to each result's payload
        for i, hit in enumerate(response.results):
            reranked[i].payload["rerank_score"] = round(hit.relevance_score, 4)

        return reranked

    except Exception as exc:
        # Graceful fallback — log warning, return original results
        import structlog

        logger = structlog.get_logger()
        logger.warning(f"RERANKER_FAILED | {exc} | falling back to cosine order")
        return results[: int(os.getenv("RERANKER_TOP_N", 5))]


# Feature: Cohere Reranker implemented
