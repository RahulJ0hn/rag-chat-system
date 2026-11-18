"""
Streaming LLM response via Server-Sent Events (SSE).
"""

import json
import os
from typing import AsyncGenerator

# langfuse.openai.AsyncOpenAI is a drop-in subclass of openai.AsyncOpenAI
# that auto-records every call as a Langfuse generation (model, tokens,
# latency, $ cost) — plain openai.AsyncOpenAI would be invisible to it.
from langfuse.openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def stream_llm_response(
    system_prompt: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Yields raw SSE payload strings (no "data: " prefix, no trailing
    newlines) as tokens arrive from OpenAI. EventSourceResponse applies
    the actual "data: ...\\n\\n" wire framing itself — pre-formatting it
    here would double-wrap the payload and break the client's parser.
    Yields a final [DONE] sentinel once the stream ends.
    """
    stream = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.2)),
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield json.dumps({"token": delta})

    yield "[DONE]"


# Feature: Streaming Response implemented
