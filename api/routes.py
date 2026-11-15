"""
FastAPI route definitions for ingestion, query, document management,
cache administration, and health checks.
"""

import json
import os
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client.models import PointIdsList
from sse_starlette.sse import EventSourceResponse

from cache import manager as cache_manager
from ingestion.pipeline import ingest_document, qdrant_client
from observability.langfuse_client import log_error_trace
from query.embedder import get_embedding
from query.inference import _build_sources, answer_query, normalize_query
from query.memory import clear_session, create_session, get_history
from query.prompt_builder import build_prompt
from query.reranker import rerank_results
from query.retriever import search
from query.stream import stream_llm_response
from storage import document_registry, object_store

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    user_role: str = "public"
    session_id: str | None = None  # None = stateless query


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, Any]:
    """Ingest an uploaded document into the RAG pipeline."""
    file_bytes = await file.read()
    try:
        return ingest_document(file_bytes, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        log_error_trace(exc, {"stage": "ingest", "filename": file.filename})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
    """Answer a user query using retrieval-augmented generation."""
    try:
        return answer_query(request.query, request.user_role, session_id=request.session_id)
    except HTTPException:
        raise
    except Exception as exc:
        log_error_trace(exc, {"stage": "query", "query": request.query})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query/stream")
async def query_stream(request: QueryRequest) -> EventSourceResponse:
    """
    Streaming version of /query.
    Returns tokens via SSE as they generate.

    Pipeline:
    1. Intent detection — greetings/out-of-scope handled immediately
    2. Cache check — streams cached answer token by token if hit
    3. Qdrant retrieval + Cohere reranking
    4. Prompt assembly with conversation history
    5. OpenAI streaming — tokens yielded as SSE events
    6. After stream: cache write + memory append + RAGAS evals
    """
    from query.intent_detector import detect_intent
    from query.memory import append_turn
    from evals.score_logger import evaluate_and_log

    normalized = normalize_query(request.query)

    # Step 1: Intent detection — same as /query
    intent = detect_intent(normalized)

    if intent == "greeting":
        async def greeting_stream():
            answer = (
                "Hello! I can answer questions about your "
                "uploaded documents. What would you like to know?"
            )
            yield f"data: {json.dumps({'token': answer})}\n\n"
            yield "data: [DONE]\n\n"
            yield f"data: {json.dumps({'sources': [], 'cache_hit': False, 'intent': intent})}\n\n"
        return EventSourceResponse(greeting_stream())

    if intent == "out_of_scope":
        async def out_of_scope_stream():
            answer = (
                "I can only answer questions about the documents "
                "that have been uploaded. Could you ask something "
                "related to those?"
            )
            yield f"data: {json.dumps({'token': answer})}\n\n"
            yield "data: [DONE]\n\n"
            yield f"data: {json.dumps({'sources': [], 'cache_hit': False, 'intent': intent})}\n\n"
        return EventSourceResponse(out_of_scope_stream())

    query_embedding = get_embedding(normalized)

    # Step 2: Cache check
    cached = cache_manager.get_exact(normalized)
    if cached is None:
        cached = cache_manager.get_semantic(normalized, query_embedding)

    if cached is not None:
        async def cached_stream():
            for token in cached["answer"].split():
                yield (
                    f"data: {json.dumps({'token': token + ' '})}\n\n"
                )
            yield "data: [DONE]\n\n"
            yield f"data: {json.dumps({'sources': cached['sources'], 'cache_hit': True, 'intent': intent})}\n\n"

            # Append cached answer to memory
            if request.session_id:
                append_turn(
                    request.session_id,
                    request.query,
                    cached["answer"]
                )
        return EventSourceResponse(cached_stream())

    cache_manager.record_miss()

    # Step 3: Retrieve + rerank
    results = search(
        query_embedding,
        request.user_role,
        query_text=normalized
    )
    results = rerank_results(normalized, results)

    # Step 4: Build prompt with conversation history
    history = (
        get_history(request.session_id)
        if request.session_id else []
    )
    system_prompt, messages = build_prompt(
        request.query, results, conversation_history=history
    )
    # stream_llm_response takes a flat system+user pair;
    # pass the current turn's content as user_message
    user_message = messages[-1]["content"]
    sources = _build_sources(results)

    # Step 5: Stream tokens, collect full answer for post-stream work
    async def response_stream():
        collected_tokens: list[str] = []

        async for chunk in stream_llm_response(
            system_prompt, user_message
        ):
            # Collect tokens to reconstruct full answer
            if chunk.startswith("data: ") and "[DONE]" not in chunk:
                try:
                    data = json.loads(
                        chunk.removeprefix("data: ").strip()
                    )
                    if "token" in data:
                        collected_tokens.append(data["token"])
                except Exception:
                    pass
            yield chunk

        full_answer = "".join(collected_tokens)

        # Step 6a: Write to cache
        cacheable = {"answer": full_answer, "sources": sources}
        cache_manager.set_exact(normalized, cacheable)
        cache_manager.set_semantic(
            normalized, query_embedding, cacheable
        )

        # Step 6b: Append to conversation memory
        if request.session_id:
            append_turn(
                request.session_id,
                request.query,
                full_answer
            )

        # Step 6c: Run RAGAS evals async
        evaluate_and_log(
            trace_id=None,  # no Langfuse trace_id in stream context
            query=request.query,
            answer=full_answer,
            retrieved_chunks=[r.payload["text"] for r in results],
        )

        # Final metadata event
        yield f"data: {json.dumps({'sources': sources, 'cache_hit': False, 'intent': intent})}\n\n"

    return EventSourceResponse(response_stream())


@router.post("/session")
async def create_new_session() -> dict[str, Any]:
    """Create a new conversation session for multi-turn chat."""
    return {"session_id": create_session()}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """Clear a session's conversation memory."""
    clear_session(session_id)
    return {"status": "cleared"}


@router.delete("/document/{doc_hash}")
async def delete_document(doc_hash: str) -> dict[str, Any]:
    """Delete a document's chunks, S3 object, and registry entries."""
    record = document_registry.get_record_by_hash(doc_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        qdrant_client.delete(
            collection_name=os.getenv("QDRANT_COLLECTION"),
            points_selector=PointIdsList(points=record["chunk_ids"]),
        )
    except Exception:
        pass

    object_store.delete_document(record["s3_key"])
    document_registry.delete_registry_entries(doc_hash)

    # A deleted/replaced document can no longer back any cached answer,
    # so drop all cached query responses rather than leaving stale ones.
    cache_manager.flush_cache()

    return {"status": "deleted", "doc_hash": doc_hash}


@router.get("/document/{doc_hash}")
async def get_document(doc_hash: str) -> dict[str, Any]:
    """Return the registry record for a single document."""
    record = document_registry.get_record_by_hash(doc_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "hash": record["hash"],
        "filenames": record["filenames"],
        "s3_key": record["s3_key"],
        "version": record["version"],
        "ingested_at": record["ingested_at"],
        "chunks_stored": len(record["chunk_ids"]),
        "element_types": record["element_types"],
    }


@router.get("/documents")
async def list_documents() -> dict[str, Any]:
    """List every document currently in the registry."""
    return {"documents": document_registry.list_all_records()}


@router.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Return exact/semantic cache hit and miss counters."""
    return cache_manager.get_cache_stats()


@router.delete("/cache/flush")
async def flush_cache() -> dict[str, Any]:
    """Clear all cached query responses."""
    keys_deleted = cache_manager.flush_cache()
    return {"status": "cleared", "keys_deleted": keys_deleted}


@router.get("/health")
async def health() -> dict[str, Any]:
    """Report connectivity to Qdrant, Redis, and S3."""
    try:
        qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "unreachable"

    redis_status = (
        "connected" if cache_manager.check_redis_health()
        else "unreachable"
    )

    s3_status = "connected" if object_store.check_s3_health() else "unreachable"

    overall = (
        "ok"
        if qdrant_status == "connected"
        and redis_status == "connected"
        and s3_status == "connected"
        else "degraded"
    )

    return {
        "status": overall,
        "qdrant": qdrant_status,
        "redis": redis_status,
        "s3": s3_status,
    }
