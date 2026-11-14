"""
S3 object storage for original uploaded documents.

Handles upload of raw file bytes, presigned URL generation for
downloads, deletion, and a health check used by GET /health.
"""

import os
from uuid import uuid4

import boto3

# Module-level client — never instantiate per call.
s3_client = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def upload_document(file_bytes: bytes, filename: str) -> str:
    """Upload raw file bytes to S3 under a UUID-prefixed key and return that key."""
    key = f"documents/{uuid4()}_{filename}"
    s3_client.put_object(
        Bucket=os.getenv("S3_BUCKET_NAME"),
        Key=key,
        Body=file_bytes,
        ContentType="application/octet-stream",
    )
    return key


def get_presigned_url(s3_key: str) -> str:
    """Generate a time-limited presigned GET URL for a stored document."""
    return s3_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": os.getenv("S3_BUCKET_NAME"),
            "Key": s3_key,
        },
        ExpiresIn=int(os.getenv("S3_PRESIGN_EXPIRY", 3600)),
    )


def delete_document(s3_key: str) -> None:
    """Delete a stored document object from S3."""
    s3_client.delete_object(
        Bucket=os.getenv("S3_BUCKET_NAME"),
        Key=s3_key,
    )


def check_s3_health() -> bool:
    """Verify the configured bucket is reachable, for GET /health."""
    try:
        s3_client.head_bucket(Bucket=os.getenv("S3_BUCKET_NAME"))
        return True
    except Exception:
        return False
