"""
System + user prompt assembly for the RAG chat completion call.
"""

from typing import Any

from query.retriever import format_context_chunk

SYSTEM_PROMPT = """You are a helpful enterprise document assistant.
Answer strictly using the provided context.
The context may include tables (in markdown), lists,
code blocks, and narrative text.
Read all context types carefully before answering.
If the answer is not in the context, say exactly:
"I don't have enough information to answer this
from the provided documents."
Cite sources at the end:
[Source: {doc_name}, Section: {section}, Page {page}]
"""


def build_user_message(results: list[Any], user_query: str) -> str:
    """Assemble the user message from retrieved chunks and the question."""
    context_blocks = []
    for result in results:
        payload = result.payload
        block = (
            f"Context:\n---\n"
            f"{format_context_chunk(payload)}\n"
            f"[Source: {payload['doc_name']}, Page {payload['page_number']}]\n"
            f"---"
        )
        context_blocks.append(block)

    context_section = "\n".join(context_blocks)
    return f"{context_section}\nQuestion: {user_query}"


def build_prompt(results: list[Any], user_query: str) -> tuple[str, str]:
    """Return the (system_prompt, user_message) pair for the LLM call."""
    return SYSTEM_PROMPT, build_user_message(results, user_query)
