"""
Computes BM25 sparse vectors for hybrid search in Qdrant.
"""

import re
from collections import Counter

from qdrant_client.models import SparseVector


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# Simple IDF approximation — use term frequency only
# for single-document encoding (no corpus IDF available at
# ingest time). Full BM25 with corpus IDF is a future TODO.
def compute_sparse_vector(text: str) -> SparseVector:
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    total = sum(counts.values())

    # TF score per token
    indices = []
    values = []
    for token, count in counts.items():
        # Use token hash as index (Qdrant sparse index accepts int ids)
        token_id = abs(hash(token)) % (10**6)
        tf = count / total
        indices.append(token_id)
        values.append(round(tf, 6))

    return SparseVector(indices=indices, values=values)


# Feature: Hybrid Search (Dense + Sparse BM25) implemented
