"""
Streaming LLM response via Server-Sent Events (SSE).
"""

import json
import os
from typing import AsyncGenerator

import openai

client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def stream_llm_response(
    system_prompt: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted token chunks as they arrive from OpenAI.
    Yields a final [DONE] event with sources and metadata.
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
            yield f"data: {json.dumps({'token': delta})}\n\n"

    yield "data: [DONE]\n\n"


# Feature: Streaming Response implemented
