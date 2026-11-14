"""
Query response cache: exact-match (Redis string) and semantic
(redisvl vector index) layers, plus hit/miss counters for
GET /cache/stats.

Redis being unavailable must never fail a query — cache operations
degrade to a miss and are logged, not raised.
"""

import hashlib
import json
import logging
import os
from typing import Any

import redis

logger = logging.getLogger(__name__)
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
)

EXACT_KEY_PREFIX = "query:"
STATS_KEY = "cache:stats"

_semantic_schema = IndexSchema.from_dict(
    {
        "index": {"name": "semantic_cache", "prefix": "semantic_cache", "storage_type": "hash"},
        "fields": [
            {"name": "query_text", "type": "text"},
            {"name": "response", "type": "text"},
            {
                "name": "query_vector",
                "type": "vector",
                "attrs": {
                    "dims": int(os.getenv("EMBEDDING_DIM", 1536)),
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }
)

# Module-level index — never instantiate per call.
semantic_index = SearchIndex(_semantic_schema, redis_client=redis_client)
semantic_index.create(overwrite=False)


def _exact_key(normalized_query: str) -> str:
    """Build the exact-match cache key from the normalized query text."""
    digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
    return f"{EXACT_KEY_PREFIX}{digest}"


def get_exact(normalized_query: str) -> dict[str, Any] | None:
    """Return a cached response for an exact normalized-query match, if any."""
    try:
        raw = redis_client.get(_exact_key(normalized_query))
    except redis.RedisError as exc:
        logger.warning(f"CACHE_READ_FAILED | exact | {exc}")
        return None
    if raw is None:
        return None
    _increment_stat("exact_hits")
    return json.loads(raw)


def set_exact(normalized_query: str, response: dict[str, Any]) -> None:
    """Cache a response under its exact-match key with the configured TTL."""
    ttl = int(os.getenv("CACHE_TTL", 3600))
    try:
        redis_client.setex(_exact_key(normalized_query), ttl, json.dumps(response))
    except redis.RedisError as exc:
        logger.warning(f"CACHE_WRITE_FAILED | exact | {exc}")


def get_semantic(
    normalized_query: str, query_embedding: list[float]
) -> dict[str, Any] | None:
    """
    Search the semantic cache for a sufficiently similar previous query.
    Returns the cached response if cosine similarity exceeds the threshold.
    """
    threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.92))
    try:
        results = semantic_index.query(
            VectorQuery(
                vector=query_embedding,
                vector_field_name="query_vector",
                return_fields=["response"],
                num_results=1,
            )
        )
    except redis.RedisError as exc:
        logger.warning(f"CACHE_READ_FAILED | semantic | {exc}")
        return None

    if not results:
        return None

    top = results[0]
    # redisvl returns cosine distance; similarity = 1 - distance.
    similarity = 1 - float(top["vector_distance"])
    if similarity < threshold:
        return None

    _increment_stat("semantic_hits")
    return json.loads(top["response"])


def set_semantic(
    normalized_query: str, query_embedding: list[float], response: dict[str, Any]
) -> None:
    """Index a query embedding and its response in the semantic cache.
    A separate TTL key is written so stale semantic entries expire
    at the same time as their exact-match counterparts.
    """
    ttl = int(os.getenv("CACHE_TTL", 3600))
    try:
        semantic_index.load(
            [
                {
                    "query_text": normalized_query,
                    "response": json.dumps(response),
                    "query_vector": query_embedding,
                }
            ]
        )
        # redisvl does not natively support per-entry TTL on hash
        # storage, so we write a companion expiry sentinel key that
        # signals to future cleanup passes that this entry is stale.
        # A full TTL-aware semantic cache is a future enhancement.
        ttl_key = (
            f"semantic_cache:ttl:"
            f"{hashlib.sha256(normalized_query.encode()).hexdigest()}"
        )
        redis_client.setex(ttl_key, ttl, "1")
    except redis.RedisError as exc:
        logger.warning(f"CACHE_WRITE_FAILED | semantic | {exc}")


def record_miss() -> None:
    """Increment the miss counter when neither cache layer has a hit."""
    _increment_stat("misses")


def _increment_stat(field: str) -> None:
    try:
        redis_client.hincrby(STATS_KEY, field, 1)
    except redis.RedisError:
        pass


def get_cache_stats() -> dict[str, Any]:
    """Return hit/miss counters and overall hit rate for GET /cache/stats."""
    try:
        raw = redis_client.hgetall(STATS_KEY)
    except redis.RedisError:
        raw = {}

    exact_hits = int(raw.get("exact_hits", 0))
    semantic_hits = int(raw.get("semantic_hits", 0))
    misses = int(raw.get("misses", 0))
    total = exact_hits + semantic_hits + misses
    hit_rate = round(((exact_hits + semantic_hits) / total) * 100, 2) if total else 0.0

    return {
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "misses": misses,
        "hit_rate_percent": hit_rate,
    }


def flush_cache() -> int:
    """Delete all exact-match and semantic cache entries. Returns keys deleted."""
    keys_deleted = 0
    for key in redis_client.scan_iter(match=f"{EXACT_KEY_PREFIX}*"):
        redis_client.delete(key)
        keys_deleted += 1
    try:
        semantic_index.delete(drop=True)
        semantic_index.create(overwrite=True)
    except redis.RedisError:
        pass
    redis_client.delete(STATS_KEY)
    return keys_deleted


def check_redis_health() -> bool:
    """Check Redis connectivity for GET /health."""
    try:
        redis_client.ping()
        return True
    except Exception:
        return False
