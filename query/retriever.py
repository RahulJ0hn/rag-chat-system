"""
Qdrant retrieval with access-tag filtering, and context formatting
that tells the LLM what kind of content (table/code/list/narrative)
it is reading.
"""

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
)

from query.sparse_encoder import compute_sparse_vector

# Module-level client — never instantiate per call.
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))

RETRIEVAL_LIMIT = 10  # Feature: Cohere Reranker — fetch more candidates to rerank
HYBRID_PREFETCH_LIMIT = 20  # Feature: Hybrid Search — candidates per prefetch leg

_TYPE_LABELS = {
    "table": "The following is a table:",
    "code": "The following is a code block:",
    "list": "The following is a list:",
}


def search(query_embedding: list[float], user_role: str, query_text: str = "") -> list[Any]:
    """
    Search Qdrant for the top matching chunks visible to user_role.

    Feature: Hybrid Search (Dense + Sparse BM25) — when USE_HYBRID_SEARCH=true,
    fuses dense semantic search with sparse BM25 keyword search via
    Reciprocal Rank Fusion so exact-term queries (e.g. "ISO 27001",
    "clause 4.2.1") aren't missed by cosine similarity alone.
    """
    access_filter = Filter(
        must=[FieldCondition(key="access_tags", match=MatchAny(any=[user_role]))]
    )

    if os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true":
        response = qdrant_client.query_points(
            collection_name=os.getenv("QDRANT_COLLECTION"),
            prefetch=[
                # Dense semantic search
                Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=HYBRID_PREFETCH_LIMIT,
                ),
                # Sparse BM25 keyword search
                Prefetch(
                    query=compute_sparse_vector(query_text),
                    using="sparse",
                    limit=HYBRID_PREFETCH_LIMIT,
                ),
            ],
            # Fuse with Reciprocal Rank Fusion
            query=FusionQuery(fusion=Fusion.RRF),
            limit=RETRIEVAL_LIMIT,
            query_filter=access_filter,
            with_payload=True,
        )
        return response.points

    # Fallback: existing dense-only search (named "dense" vector config).
    return qdrant_client.search(
        collection_name=os.getenv("QDRANT_COLLECTION"),
        query_vector=("dense", query_embedding),
        limit=RETRIEVAL_LIMIT,
        query_filter=access_filter,
        with_payload=True,
    )


def format_context_chunk(payload: dict[str, Any]) -> str:
    """Prepend element-type and section context so the LLM reads chunks correctly."""
    element_type = payload["element_type"]
    text = payload["text"]

    type_prefix = _TYPE_LABELS.get(element_type)
    body = f"{type_prefix}\n{text}" if type_prefix else text

    return f"Section: {payload['section']}\n{body}"
