"""
Sends RAGAS eval scores to Langfuse, and dispatches evaluation on a
background thread when EVAL_ASYNC=true so it never blocks the query
response.
"""

import os
import threading

from evals.ragas_runner import run_ragas_eval
from observability.langfuse_client import langfuse_client


def log_scores_to_langfuse(trace_id: str, scores: dict[str, float | None]) -> None:
    """Attach each RAGAS metric as a Langfuse score on the given trace."""
    for metric_name, value in scores.items():
        if value is None:
            continue
        langfuse_client.score(trace_id=trace_id, name=metric_name, value=value)


def _run_and_log(
    trace_id: str, query: str, answer: str, retrieved_chunks: list[str]
) -> None:
    scores = run_ragas_eval(query, answer, retrieved_chunks)
    log_scores_to_langfuse(trace_id, scores)


def evaluate_and_log(
    trace_id: str, query: str, answer: str, retrieved_chunks: list[str]
) -> dict[str, float | None] | None:
    """
    Run the eval + score logging. If RUN_EVALS is disabled, this is a
    no-op. If EVAL_ASYNC is enabled, runs on a background thread and
    returns None immediately; otherwise runs synchronously and returns
    the scores.
    """
    if os.getenv("RUN_EVALS", "true").lower() != "true":
        return None

    if os.getenv("EVAL_ASYNC", "true").lower() == "true":
        thread = threading.Thread(
            target=_run_and_log,
            args=(trace_id, query, answer, retrieved_chunks),
            daemon=True,
        )
        thread.start()
        return None

    scores = run_ragas_eval(query, answer, retrieved_chunks)
    log_scores_to_langfuse(trace_id, scores)
    return scores
