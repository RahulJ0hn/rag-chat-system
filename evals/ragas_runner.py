"""
RAGAS evaluation of each response: faithfulness, answer relevancy,
and context precision. Runs after every query; EVAL_ASYNC=true keeps
it off the response's critical path via a background thread.

RAGAS failures must never break the query path — they are logged and
null scores are returned instead.
"""

import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

logger = logging.getLogger(__name__)

LOW_SCORE_THRESHOLD = 0.7


def run_ragas_eval(
    query: str, answer: str, retrieved_chunks: list[str]
) -> dict[str, float | None]:
    """Run the RAGAS metric suite for one query/answer/context triple."""
    try:
        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [retrieved_chunks],
            }
        )

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        scores = {
            "faithfulness": round(result["faithfulness"], 4),
            "answer_relevancy": round(result["answer_relevancy"], 4),
            "context_precision": round(result["context_precision"], 4),
        }
    except Exception as exc:
        logger.warning(f"RAGAS_EVAL_FAILED | {query} | {exc}")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    if scores["faithfulness"] < LOW_SCORE_THRESHOLD:
        logger.warning(f"LOW_FAITHFULNESS | {query}")
    if scores["answer_relevancy"] < LOW_SCORE_THRESHOLD:
        logger.warning(f"LOW_RELEVANCE | {query}")

    return scores
