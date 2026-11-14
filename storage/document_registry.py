"""
Document registry backed by Redis.

Maintains two indexes:
  - "doc_registry:hash:{sha256}"     -> full document record (JSON)
  - "doc_registry:filename:{name}"   -> sha256 hash string

Used to detect duplicate content, renamed files, and content updates
on every upload, and to look up chunk_ids / s3_key for cleanup when a
document is deleted or replaced.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import redis

logger = logging.getLogger(__name__)

# Module-level client — never instantiate per call.
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
)

HASH_KEY_PREFIX = "doc_registry:hash:"
FILENAME_KEY_PREFIX = "doc_registry:filename:"


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute the SHA256 content hash used as the registry's primary key."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_record_by_hash(doc_hash: str) -> dict[str, Any] | None:
    """
    Fetch the full registry record for a given content hash, if present.
    A Redis outage degrades to "not found" rather than raising, so an
    upload is treated as a new document instead of failing the request.
    """
    try:
        raw = redis_client.get(f"{HASH_KEY_PREFIX}{doc_hash}")
    except redis.RedisError as exc:
        logger.warning(f"REGISTRY_READ_FAILED | hash={doc_hash} | {exc}")
        return None
    if raw is None:
        return None
    return json.loads(raw)


def get_hash_by_filename(filename: str) -> str | None:
    """
    Look up the content hash currently associated with a filename.
    A Redis outage degrades to "not found" rather than raising.
    """
    try:
        return redis_client.get(f"{FILENAME_KEY_PREFIX}{filename}")
    except redis.RedisError as exc:
        logger.warning(f"REGISTRY_READ_FAILED | filename={filename} | {exc}")
        return None


def save_record(record: dict[str, Any]) -> None:
    """
    Persist a registry record under its hash key, and point every alias
    filename in record["filenames"] at that hash.
    """
    doc_hash = record["hash"]
    redis_client.set(f"{HASH_KEY_PREFIX}{doc_hash}", json.dumps(record))
    for filename in record["filenames"]:
        redis_client.set(f"{FILENAME_KEY_PREFIX}{filename}", doc_hash)


def add_filename_alias(doc_hash: str, filename: str) -> None:
    """Attach a new filename alias to an existing hash record, if not already present."""
    record = get_record_by_hash(doc_hash)
    if record is None:
        return
    if filename not in record["filenames"]:
        record["filenames"].append(filename)
        save_record(record)


def delete_registry_entries(doc_hash: str) -> dict[str, Any] | None:
    """
    Remove both the hash record and all of its filename alias entries.
    Returns the deleted record so the caller can clean up S3/Qdrant
    using its s3_key / chunk_ids before the record is gone.
    """
    record = get_record_by_hash(doc_hash)
    if record is None:
        return None
    for filename in record["filenames"]:
        redis_client.delete(f"{FILENAME_KEY_PREFIX}{filename}")
    redis_client.delete(f"{HASH_KEY_PREFIX}{doc_hash}")
    return record


def check_upload(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Apply the registry decision logic for an incoming upload.

    Returns one of:
      {"status": "duplicate_content", "message": ..., "skipped": True, "hash": ...}
      {"status": "updated", "previous_version": int, "hash": ..., "old_record": dict}
      {"status": "created", "hash": ...}
    """
    new_hash = compute_file_hash(file_bytes)

    # Case 1: identical content already ingested, possibly under another name.
    existing_by_hash = get_record_by_hash(new_hash)
    if existing_by_hash is not None:
        add_filename_alias(new_hash, filename)
        original_filename = existing_by_hash["filenames"][0]
        return {
            "status": "duplicate_content",
            "message": f"Already ingested as {original_filename}",
            "skipped": True,
            "hash": new_hash,
        }

    # Case 2: same filename, different content — treat as an update.
    old_hash = get_hash_by_filename(filename)
    if old_hash is not None and old_hash != new_hash:
        old_record = delete_registry_entries(old_hash)
        previous_version = old_record["version"] if old_record else 0
        return {
            "status": "updated",
            "previous_version": previous_version,
            "hash": new_hash,
            "old_record": old_record,
        }

    # Case 3: brand new document.
    return {"status": "created", "hash": new_hash}


def build_record(
    doc_hash: str,
    filename: str,
    s3_key: str,
    chunk_ids: list[str],
    version: int,
    token_count: int,
    element_types: list[str],
) -> dict[str, Any]:
    """Assemble a fresh registry record to be written after ingestion completes."""
    return {
        "hash": doc_hash,
        "filenames": [filename],
        "s3_key": s3_key,
        "chunk_ids": chunk_ids,
        "version": version,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "token_count": token_count,
        "element_types": element_types,
    }


def list_all_records() -> list[dict[str, Any]]:
    """Return every document record currently in the registry."""
    records = []
    try:
        for key in redis_client.scan_iter(match=f"{HASH_KEY_PREFIX}*"):
            raw = redis_client.get(key)
            if raw:
                records.append(json.loads(raw))
    except redis.RedisError as exc:
        logger.warning(f"REGISTRY_READ_FAILED | list_all_records | {exc}")
    return records


# TODO: page-level hash comparison for large documents
#       to avoid re-ingesting unchanged pages
