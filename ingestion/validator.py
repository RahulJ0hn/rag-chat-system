"""
Upload validation: file extension and size checks performed before
any parsing, hashing, or storage work begins.
"""

import os

from fastapi import HTTPException

ALLOWED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20MB


def validate_file(filename: str, file_bytes: bytes) -> None:
    """
    Validate file type and size.

    Raises HTTPException(400) for an unsupported extension and
    HTTPException(413) when the file exceeds the size limit.
    """
    _, extension = os.path.splitext(filename.lower())

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "detail": f"Unsupported file type '{extension}'. "
                f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}",
            },
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "validation_failed",
                "detail": f"File exceeds max size of "
                f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
            },
        )
