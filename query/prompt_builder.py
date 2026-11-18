"""
System + user prompt assembly for the RAG chat completion call.
"""

from typing import Any

from query.retriever import format_context_chunk

SYSTEM_PROMPT = """You are a helpful enterprise document assistant.
Answer using only facts, figures, and statements found in the
provided context — never invent information that isn't there.
The context may include tables (in markdown), lists,
code blocks, and narrative text.
Read all context types carefully before answering.

You MAY reason over the facts in the context: comparisons (e.g. is
a listed figure above or below a threshold), arithmetic (sums,
differences, totals), and straightforward inferences — the answer
does not need to appear verbatim. For example, if the context lists
costs of $74.52 and $90.76 and the question asks whether anything
exceeds $1000, correctly answer "no" using those figures rather than
refusing.

Only say the context is insufficient — exactly:
"I don't have enough information to answer this
from the provided documents."
— when the context genuinely contains no facts relevant to the
question at all, not merely because the exact phrasing differs.
Cite sources at the end:
[Source: {doc_name}, Section: {section}, Page {page}]
"""


def build_context_block(retrieved_chunks: list[Any]) -> str:
    """Assemble the context block from retrieved chunks, with source citations."""
    context_blocks = []
    for result in retrieved_chunks:
        payload = result.payload
        block = (
            f"{format_context_chunk(payload)}\n"
            f"[Source: {payload['doc_name']}, Page {payload['page_number']}]"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_prompt(
    query: str,
    retrieved_chunks: list[Any],
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Returns (system_prompt, messages_list).

    Feature: Conversation Memory (Multi-turn Chat) — messages_list
    includes prior conversation turns so the LLM has multi-turn
    context when a session_id is provided.
    """
    history = conversation_history or []
    context_block = build_context_block(retrieved_chunks)

    # Build messages array with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append prior turns (gives LLM conversation context)
    messages.extend(history)

    # Append current user message with context
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n---\n{context_block}\n---\n\nQuestion: {query}",
        }
    )

    return SYSTEM_PROMPT, messages
