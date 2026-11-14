"""
Full document ingestion pipeline orchestration.

Validates -> checks the registry for duplicates/updates -> uploads the
original to S3 -> parses + cleans + chunks -> embeds -> upserts into
Qdrant -> writes the registry record. Cleans up stale Qdrant/S3/registry
state first when a document is being replaced.
"""

import os
import tempfile
import time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from ingestion.chunker import chunk_elements, count_tokens
from ingestion.cleaner import clean_elements
from ingestion.extractor import extract_elements
from ingestion.validator import validate_file
from observability.langfuse_client import log_metadata, observe
from query.embedder import get_embeddings_batch
from storage import document_registry, object_store

# Module-level client — never instantiate per call.
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))


def ensure_collection() -> None:
    """Create the Qdrant collection on startup if it does not already exist."""
    collection_name = os.getenv("QDRANT_COLLECTION")
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if collection_name not in existing:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=int(os.getenv("EMBEDDING_DIM")),
                distance=Distance.COSINE,
            ),
        )


@observe()
def ingest_document(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Run the full ingestion pipeline for one uploaded file."""
    start_time = time.monotonic()
    file_type = os.path.splitext(filename)[1]

    log_metadata(
        doc_name=filename,
        file_size=len(file_bytes),
        file_type=file_type,
    )

    validate_file(filename, file_bytes)

    decision = document_registry.check_upload(file_bytes, filename)
    log_metadata(doc_hash=decision["hash"], registry_status=decision["status"])

    # Duplicate content — nothing further to do.
    if decision["status"] == "duplicate_content":
        return {
            "status": "duplicate_content",
            "doc_name": filename,
            "doc_hash": decision["hash"],
            "s3_key": None,
            "chunks_stored": 0,
            "element_counts": {"narrative": 0, "table": 0, "list": 0, "code": 0},
            "ingestion_ms": round((time.monotonic() - start_time) * 1000, 2),
            "skipped": True,
        }

    # Replacing an existing document — tear down its old chunks/object first.
    old_record = decision.get("old_record")
    if old_record is not None:
        try:
            qdrant_client.delete(
                collection_name=os.getenv("QDRANT_COLLECTION"),
                points_selector=PointIdsList(points=old_record["chunk_ids"]),
            )
        except Exception:
            pass  # old points may already be gone; proceed with replacement
        object_store.delete_document(old_record["s3_key"])

    doc_hash = decision["hash"]
    version = (old_record["version"] + 1) if old_record else 1

    try:
        s3_key = object_store.upload_document(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {exc}") from exc
    log_metadata(s3_key=s3_key)

    # Unstructured needs a real file path, so write the upload to a temp file.
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file.flush()
        raw_elements = extract_elements(tmp_file.name)

    if not raw_elements:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_document", "detail": "No extractable content found."},
        )

    cleaned_elements = clean_elements(raw_elements)
    chunks = chunk_elements(cleaned_elements, doc_name=filename, doc_hash=doc_hash)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_document", "detail": "No chunks produced after cleaning."},
        )

    embedding_start = time.monotonic()
    embeddings = get_embeddings_batch([chunk["text"] for chunk in chunks])
    embedding_time_ms = round((time.monotonic() - embedding_start) * 1000, 2)

    element_counts = {"narrative": 0, "table": 0, "list": 0, "code": 0, "image_caption": 0}
    points: list[PointStruct] = []
    chunk_ids: list[str] = []

    for chunk, embedding in zip(chunks, embeddings):
        element_counts[chunk["element_type"]] = (
            element_counts.get(chunk["element_type"], 0) + 1
        )
        point_id = str(uuid4())
        chunk_ids.append(point_id)
        payload = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "element_type": chunk["element_type"],
            "section": chunk["section"],
            "page_number": chunk["page_number"],
            "doc_name": chunk["doc_name"],
            "doc_hash": chunk["doc_hash"],
            "s3_key": s3_key,
            "source_department": "general",
            "access_tags": ["public"],
            "token_count": chunk["token_count"],
        }
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    upsert_start = time.monotonic()
    try:
        qdrant_client.upsert(collection_name=os.getenv("QDRANT_COLLECTION"), points=points)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    upsert_time_ms = round((time.monotonic() - upsert_start) * 1000, 2)

    total_tokens = sum(chunk["token_count"] for chunk in chunks)
    element_types_present = sorted({chunk["element_type"] for chunk in chunks})

    log_metadata(
        element_counts={
            key: element_counts.get(key, 0)
            for key in ("narrative", "table", "list", "code", "image_caption")
        },
        chunks_created=len(chunks),
        token_count=total_tokens,
        embedding_time_ms=embedding_time_ms,
        upsert_time_ms=upsert_time_ms,
    )

    record = document_registry.build_record(
        doc_hash=doc_hash,
        filename=filename,
        s3_key=s3_key,
        chunk_ids=chunk_ids,
        version=version,
        token_count=total_tokens,
        element_types=element_types_present,
    )
    document_registry.save_record(record)

    return {
        "status": decision["status"],
        "doc_name": filename,
        "doc_hash": doc_hash,
        "s3_key": s3_key,
        "chunks_stored": len(chunks),
        "element_counts": {
            key: element_counts.get(key, 0) for key in ("narrative", "table", "list", "code")
        },
        "ingestion_ms": round((time.monotonic() - start_time) * 1000, 2),
        "skipped": False,
    }
