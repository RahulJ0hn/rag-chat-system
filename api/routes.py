"""
FastAPI route definitions for ingestion, query, document management,
cache administration, and health checks.
"""

import os
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client.models import PointIdsList

from cache import manager as cache_manager
from ingestion.pipeline import ingest_document, qdrant_client
from observability.langfuse_client import log_error_trace
from query.inference import answer_query
from storage import document_registry, object_store

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    user_role: str = "public"


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
        return answer_query(request.query, request.user_role)
    except HTTPException:
        raise
    except Exception as exc:
        log_error_trace(exc, {"stage": "query", "query": request.query})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
