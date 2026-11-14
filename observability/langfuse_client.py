"""
Langfuse setup. Provides the module-level client plus the @observe
decorator used across ingestion and query pipeline functions, and a
helper for logging error traces.
"""

import os

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

# Module-level client — never instantiate per call.
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


def log_metadata(**fields: object) -> None:
    """Attach arbitrary key/value metadata to the current Langfuse trace."""
    langfuse_context.update_current_observation(metadata=fields)


def log_error_trace(error: Exception, context: dict[str, object]) -> None:
    """Record an error as a Langfuse trace for later debugging."""
    trace = langfuse_client.trace(name="error", metadata={"error": str(error), **context})
    # level is an Event/Span attribute, not a Trace attribute — attach it via a child event.
    trace.event(name="error", level="ERROR", status_message=str(error))


__all__ = ["langfuse_client", "observe", "langfuse_context", "log_metadata", "log_error_trace"]
