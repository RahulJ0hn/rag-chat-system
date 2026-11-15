"""
Query orchestration: normalization, embedding, cache lookup, Qdrant
retrieval, prompt assembly, the OpenAI chat completion call, source
building with presigned download URLs, cache writes, and async eval.
"""

import os
import time
from typing import Any

import openai
from fastapi import HTTPException

from cache import manager as cache_manager
from evals.score_logger import evaluate_and_log
from observability.cost_tracker import calculate_chat_cost
from observability.langfuse_client import langfuse_context, log_error_trace, observe
from query.embedder import get_embedding
from query.intent_detector import detect_intent
from query.memory import append_turn, get_history
from query.prompt_builder import build_prompt
from query.reranker import rerank_results
from query.retriever import search
from storage.object_store import get_presigned_url

# Module-level client — never instantiate per call.
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NULL_EVAL_SCORES = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

GREETING_ANSWER = (
    "Hello! I can answer questions about your uploaded documents. "
    "What would you like to know?"
)
OUT_OF_SCOPE_ANSWER = (
    "I can only answer questions about the documents that have been uploaded. "
    "Could you ask something related to those?"
)


def normalize_query(query: str) -> str:
    """Strip, normalize whitespace, and lowercase the incoming query."""
    return " ".join(query.strip().lower().split())


def _call_llm(messages: list[dict]) -> Any:
    """
    Call the chat completion endpoint with a single retry on rate limit
    and a translated 504 on timeout, per the error-handling spec.

    Feature: Conversation Memory — messages is the full message list
    (system + prior turns + current user turn) built by build_prompt().
    """
    try:
        return client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE")),
            stream=False,
        )
    except openai.RateLimitError:
        time.sleep(2)
        try:
            return client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                temperature=float(os.getenv("LLM_TEMPERATURE")),
                stream=False,
            )
        except openai.RateLimitError as exc:
            raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded") from exc
    except openai.APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="OpenAI request timed out") from exc


def _build_sources(results: list[Any]) -> list[dict[str, Any]]:
    """Build the sources list, attaching a presigned S3 download URL per chunk."""
    return [
        {
            "doc_name": result.payload["doc_name"],
            "section": result.payload["section"],
            "page_number": result.payload["page_number"],
            "element_type": result.payload["element_type"],
            "score": round(result.score, 4),
            "rerank_score": result.payload.get("rerank_score", None),
            "download_url": get_presigned_url(result.payload["s3_key"]),
        }
        for result in results
    ]


@observe()
def answer_query(query: str, user_role: str, session_id: str | None = None) -> dict[str, Any]:
    """
    Run the full query workflow and return the API response payload.

    Feature: Conversation Memory — when session_id is provided, prior
    turns are loaded from Redis and appended to the prompt, and this
    turn is appended back to the session's history once answered.
    """
    normalized_query = normalize_query(query)

    # Feature: Intent Detection — classify before doing any retrieval work
    # so greetings/out-of-scope queries skip embedding, cache, and search.
    intent = detect_intent(normalized_query)
    langfuse_context.update_current_observation(metadata={"intent": intent})

    if intent == "greeting":
        return {
            "answer": GREETING_ANSWER,
            "sources": [],
            "cache_hit": False,
            "cache_type": "none",
            "intent": intent,
            "eval_scores": NULL_EVAL_SCORES,
        }

    if intent == "out_of_scope":
        return {
            "answer": OUT_OF_SCOPE_ANSWER,
            "sources": [],
            "cache_hit": False,
            "cache_type": "none",
            "intent": intent,
            "eval_scores": NULL_EVAL_SCORES,
        }

    try:
        query_embedding = get_embedding(normalized_query)
    except Exception as exc:
        log_error_trace(exc, {"stage": "embedding", "query": query})
        raise

    # --- Cache check: exact, then semantic ---
    cached = cache_manager.get_exact(normalized_query)
    if cached is not None:
        if session_id:
            append_turn(session_id, query, cached["answer"])
        return {
            **cached,
            "cache_hit": True,
            "cache_type": "exact",
            "intent": intent,
            "eval_scores": NULL_EVAL_SCORES,
        }

    cached = cache_manager.get_semantic(normalized_query, query_embedding)
    if cached is not None:
        if session_id:
            append_turn(session_id, query, cached["answer"])
        return {
            **cached,
            "cache_hit": True,
            "cache_type": "semantic",
            "intent": intent,
            "eval_scores": NULL_EVAL_SCORES,
        }

    cache_manager.record_miss()

    # --- Cache miss: retrieve, prompt, generate ---
    try:
        results = search(query_embedding, user_role, query_text=normalized_query)
    except Exception as exc:
        log_error_trace(exc, {"stage": "retrieval", "query": query})
        raise HTTPException(status_code=503, detail="Qdrant unavailable") from exc

    # Feature: Cohere Reranker — re-score the cosine-similarity candidates
    # by actual relevance to the question, keeping only the top N.
    results = rerank_results(normalized_query, results)
    reranker_used = any("rerank_score" in r.payload for r in results)

    history = get_history(session_id) if session_id else []
    system_prompt, messages = build_prompt(query, results, conversation_history=history)
    response = _call_llm(messages)

    answer = response.choices[0].message.content
    tokens_prompt = response.usage.prompt_tokens
    tokens_completion = response.usage.completion_tokens
    cost_usd = calculate_chat_cost(tokens_prompt, tokens_completion)

    sources = _build_sources(results)

    result_payload = {
        "answer": answer,
        "sources": sources,
        "cache_hit": False,
        "cache_type": "none",
        "intent": intent,
    }

    # --- Cache write (cache the base payload, without hit/type flags) ---
    cacheable = {"answer": answer, "sources": sources}
    cache_manager.set_exact(normalized_query, cacheable)
    cache_manager.set_semantic(normalized_query, query_embedding, cacheable)

    # Feature: Conversation Memory — store this turn for future context.
    if session_id:
        append_turn(session_id, query, answer)

    langfuse_context.update_current_observation(
        metadata={
            "query": query,
            "normalized_query": normalized_query,
            "intent": intent,
            "cache_hit": False,
            "cache_type": "none",
            "chunks_retrieved": len(results),
            "element_types": [r.payload["element_type"] for r in results],
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "cost_usd": cost_usd,
            "reranker_used": reranker_used,
        }
    )

    eval_scores = evaluate_and_log(
        trace_id=langfuse_context.get_current_trace_id(),
        query=query,
        answer=answer,
        retrieved_chunks=[r.payload["text"] for r in results],
    )
    result_payload["eval_scores"] = eval_scores or NULL_EVAL_SCORES

    return result_payload
