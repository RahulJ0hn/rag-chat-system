"""
Shared OpenAI embedding function used by both ingestion and query paths.

Using one function for both ensures the same model is always used,
so document and query vectors stay comparable in Qdrant.
"""

import os

import openai

# Module-level client — never instantiate per call.
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATCH_SIZE = 100  # rate limit safety


def get_embedding(text: str) -> list[float]:
    """Embed a single piece of text using the configured embedding model."""
    response = client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL"),
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches of BATCH_SIZE to stay within rate limits."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            input=batch,
            model=os.getenv("EMBEDDING_MODEL"),
        )
        embeddings.extend(item.embedding for item in response.data)
    return embeddings
