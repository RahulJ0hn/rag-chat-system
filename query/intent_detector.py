"""
Classifies user query intent using a lightweight OpenAI call.
Uses gpt-4o-mini with structured JSON output.
Falls back to 'document_lookup' on any failure.
"""

import hashlib
import json
import os

import openai
import redis

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
r = redis.from_url(os.getenv("REDIS_URL"))

INTENT_PROMPT = """
Classify the user's query into exactly one of these intents:
- document_lookup: looking for specific information in documents
- summarization: wants a summary of a document or section
- comparison: wants to compare information across documents
- greeting: hello, hi, how are you, etc.
- out_of_scope: unrelated to documents (weather, news, etc.)

Respond ONLY with valid JSON:
{"intent": "<intent_name>", "confidence": 0.0-1.0}
"""


def detect_intent(query: str) -> str:
    """
    Classifies query intent using gpt-4o-mini with structured
    JSON output. Result is cached in Redis for 1 hour so
    repeated similar queries skip the LLM call entirely.
    Falls back to 'document_lookup' on any failure.
    """
    # Check Redis cache first — avoids extra API call on repeat queries
    cache_key = f"intent:{hashlib.sha256(query.encode()).hexdigest()}"
    try:
        cached = r.get(cache_key)
        if cached:
            return cached.decode()
    except Exception:
        pass  # Redis unavailable — proceed to LLM classification

    # LLM-based classification
    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=50,
        )
        result = json.loads(response.choices[0].message.content.strip())
        intent = result.get("intent", "document_lookup")

        # Cache result for 1 hour
        try:
            r.setex(cache_key, 3600, intent)
        except Exception:
            pass  # Redis unavailable — still return the intent

        return intent

    except Exception:
        return "document_lookup"


# Feature: Intent Detection implemented
