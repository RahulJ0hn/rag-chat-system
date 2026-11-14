"""
Qdrant retrieval with access-tag filtering, and context formatting
that tells the LLM what kind of content (table/code/list/narrative)
it is reading.
"""

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

# Module-level client — never instantiate per call.
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))

RETRIEVAL_LIMIT = 5

_TYPE_LABELS = {
    "table": "The following is a table:",
    "code": "The following is a code block:",
    "list": "The following is a list:",
}


def search(query_embedding: list[float], user_role: str) -> list[Any]:
    """Search Qdrant for the top matching chunks visible to user_role."""
    return qdrant_client.search(
        collection_name=os.getenv("QDRANT_COLLECTION"),
        query_vector=query_embedding,
        limit=RETRIEVAL_LIMIT,
        query_filter=Filter(
            must=[FieldCondition(key="access_tags", match=MatchAny(any=[user_role]))]
        ),
        with_payload=True,
    )


def format_context_chunk(payload: dict[str, Any]) -> str:
    """Prepend element-type and section context so the LLM reads chunks correctly."""
    element_type = payload["element_type"]
    text = payload["text"]

    type_prefix = _TYPE_LABELS.get(element_type)
    body = f"{type_prefix}\n{text}" if type_prefix else text

    return f"Section: {payload['section']}\n{body}"
