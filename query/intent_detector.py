"""
Classifies user query intent using a lightweight OpenAI call.
Uses gpt-4o-mini with structured JSON output.
Falls back to 'document_lookup' on any failure.
"""

import hashlib
import json
import os

import redis

# langfuse.openai.OpenAI is a drop-in subclass of openai.OpenAI that
# auto-records every call as a Langfuse generation (model, tokens,
# latency, $ cost) — plain openai.OpenAI would be invisible to it.
from langfuse.openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
r = redis.from_url(os.getenv("REDIS_URL"))

INTENT_PROMPT = """
You are the intent classifier for a RAG chat system where the user is
always chatting about documents they have uploaded. Every question —
even a short, vague, or context-free one like "what is the highest
cost" or "who is responsible for this" — is implicitly about those
documents unless it is unmistakably about something else entirely.

Classify the user's query into exactly one of these intents:
- document_lookup: looking for specific information (the default —
  use this for any question that could plausibly be answered by a
  document, including short or vague ones)
- summarization: explicitly asks for a summary of a document/section
- comparison: explicitly asks to compare information across documents
- greeting: a bare greeting with no question — hi, hello, how are you
- out_of_scope: ONLY for queries clearly unrelated to any document,
  e.g. real-world trivia, weather, sports scores, news, small talk
  with no document angle

When uncertain, choose document_lookup — never guess out_of_scope or
greeting unless you are confident.

Examples:
"what is the highest cost" -> document_lookup
"summarize the findings" -> summarization
"compare q1 and q2 revenue" -> comparison
"hi" -> greeting
"hello there" -> greeting
"what's the weather today" -> out_of_scope
"who won the cricket match" -> out_of_scope

Respond ONLY with valid JSON:
{"intent": "<intent_name>", "confidence": 0.0-1.0}
"""

# Below this confidence, a greeting/out_of_scope call is treated as
# document_lookup instead — a wrong "document_lookup" costs one extra
# retrieval; a wrong "out_of_scope" silently drops the user's question.
MIN_CONFIDENCE_TO_SKIP_RETRIEVAL = 0.75


def detect_intent(query: str) -> str:
    """
    Classifies query intent using gpt-4o-mini with structured
    JSON output. Result is cached in Redis for 1 hour so
    repeated similar queries skip the LLM call entirely.
    Falls back to 'document_lookup' on any failure.
    """
    cache_key = f"intent:{hashlib.sha256(query.encode()).hexdigest()}"

    # Fast path — keyword check (free, instant, no LLM call)
    tokens = set(query.lower().split())
    GREETING_TOKENS = {"hi", "hello", "hey", "howdy", "greetings", "sup", "hiya"}
    OUT_OF_SCOPE_TOKENS = {"weather", "news", "sports", "stock", "price", "cricket"}

    if tokens & GREETING_TOKENS and len(tokens) <= 5:
        try:
            r.setex(cache_key, 3600, "greeting")
        except Exception:
            pass
        return "greeting"

    if tokens & OUT_OF_SCOPE_TOKENS:
        try:
            r.setex(cache_key, 3600, "out_of_scope")
        except Exception:
            pass
        return "out_of_scope"

    # Check Redis cache first — avoids extra API call on repeat queries
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
        confidence = float(result.get("confidence", 1.0))

        # A low-confidence greeting/out_of_scope call is treated as
        # document_lookup — being wrong in that direction costs one
        # extra retrieval; being wrong the other way silently drops
        # the user's actual question.
        if intent in ("greeting", "out_of_scope") and confidence < MIN_CONFIDENCE_TO_SKIP_RETRIEVAL:
            intent = "document_lookup"

        # Cache result for 1 hour
        try:
            r.setex(cache_key, 3600, intent)
        except Exception:
            pass  # Redis unavailable — still return the intent

        return intent

    except Exception:
        return "document_lookup"


# Feature: Intent Detection implemented
