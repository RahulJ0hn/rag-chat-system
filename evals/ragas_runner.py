"""
RAGAS evaluation of each response: faithfulness, answer relevancy,
and context precision. Runs after every query; EVAL_ASYNC=true keeps
it off the response's critical path via a background thread.

RAGAS failures must never break the query path — they are logged and
null scores are returned instead.
"""

import logging
import os

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import _LLMContextPrecisionWithoutReference, answer_relevancy, faithfulness

logger = logging.getLogger(__name__)

LOW_SCORE_THRESHOLD = 0.7

# The plain `context_precision` singleton requires a human-written
# 'reference' answer, which doesn't exist for live production queries —
# only the reference-free variant works here. Re-aliased to the name
# the rest of this module (and callers) expect.
context_precision = _LLMContextPrecisionWithoutReference(name="context_precision")


def run_ragas_eval(
    query: str, answer: str, retrieved_chunks: list[str]
) -> dict[str, float | None]:
    """Run the RAGAS metric suite for one query/answer/context triple."""
    try:
        # Column names match ragas' current SingleTurnSample schema —
        # the old question/answer/contexts naming no longer validates.
        dataset = Dataset.from_dict(
            {
                "user_input": [query],
                "response": [answer],
                "retrieved_contexts": [retrieved_chunks],
            }
        )

        llm = LangchainLLMWrapper(ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ))

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
        )

        # evaluate() returns one score per dataset row as a list;
        # we always evaluate exactly one query/answer pair at a time.
        scores = {
            "faithfulness": round(result["faithfulness"][0], 4),
            "answer_relevancy": round(result["answer_relevancy"][0], 4),
            "context_precision": round(result["context_precision"][0], 4),
        }
    except Exception as exc:
        logger.warning(f"RAGAS_EVAL_FAILED | {query} | {exc}")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    if scores["faithfulness"] < LOW_SCORE_THRESHOLD:
        logger.warning(f"LOW_FAITHFULNESS | {query}")
    if scores["answer_relevancy"] < LOW_SCORE_THRESHOLD:
        logger.warning(f"LOW_RELEVANCE | {query}")

    return scores
