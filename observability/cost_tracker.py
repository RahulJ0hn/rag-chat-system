"""
Token -> USD cost calculation for OpenAI chat completion and
embedding calls, used to annotate Langfuse query traces.

Prices are per 1M tokens, matching OpenAI's published pricing for
the models configured in .env. Update PRICING_PER_MILLION_TOKENS if
the configured models or their prices change.
"""

import os

PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
}


def calculate_chat_cost(tokens_prompt: int, tokens_completion: int) -> float:
    """Calculate the USD cost of a chat completion call."""
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    rates = PRICING_PER_MILLION_TOKENS.get(model, PRICING_PER_MILLION_TOKENS["gpt-4o-mini"])
    cost = (tokens_prompt / 1_000_000) * rates["prompt"]
    cost += (tokens_completion / 1_000_000) * rates["completion"]
    return round(cost, 8)


def calculate_embedding_cost(tokens: int) -> float:
    """Calculate the USD cost of an embedding call."""
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    rates = PRICING_PER_MILLION_TOKENS.get(
        model, PRICING_PER_MILLION_TOKENS["text-embedding-3-small"]
    )
    return round((tokens / 1_000_000) * rates["prompt"], 8)
