"""
Langfuse setup. Provides the module-level client plus the @observe
decorator used across ingestion and query pipeline functions, and a
helper for logging error traces.

Targets Langfuse Python SDK v3+ (OpenTelemetry-based). The pre-v3
`langfuse.decorators.langfuse_context` API no longer exists; this
module wraps the current `get_client()` singleton instead so the
rest of the app can keep calling log_metadata()/log_error_trace()
without caring which SDK generation is installed.
"""

from langfuse import get_client, observe

# Module-level singleton — reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
# LANGFUSE_HOST from the environment itself; never instantiate per call.
langfuse_client = get_client()


def log_metadata(**fields: object) -> None:
    """Attach arbitrary key/value metadata to the current Langfuse span."""
    langfuse_client.update_current_span(metadata=fields)


def log_error_trace(error: Exception, context: dict[str, object]) -> None:
    """Record an error as a Langfuse event for later debugging."""
    langfuse_client.create_event(
        name="error",
        level="ERROR",
        status_message=str(error),
        metadata={"error": str(error), **context},
    )


__all__ = ["langfuse_client", "observe", "log_metadata", "log_error_trace"]
