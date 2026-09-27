"""Generate answers with OpenAI API or free local Ollama (see llm_client)."""
from __future__ import annotations

import os
from typing import List

from llm_client import get_chat_client


READER_SYSTEM = (
    "You are a precise assistant. Answer the question using only the information "
    "in the provided chat history when possible. If the history does not contain "
    "enough information, say so briefly."
)


def sessions_to_block(sessions_text: List[str], max_chars: int = 100_000) -> str:
    parts = []
    total = 0
    for i, st in enumerate(sessions_text):
        block = f"--- Session {i+1} ---\n{st}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def generate_answer(
    question: str,
    retrieved_session_texts: List[str],
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """
    Uses OPENAI_API_KEY if set; otherwise forces Ollama (set USE_OLLAMA=1 or omit key).
    """
    client, default_model = get_chat_client()
    m = model or default_model
    history = sessions_to_block(retrieved_session_texts)
    user_msg = f"Chat history:\n{history}\n\nQuestion: {question}\n\nAnswer concisely."
    rsp = client.chat.completions.create(
        model=m,
        temperature=temperature,
        messages=[
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    return (rsp.choices[0].message.content or "").strip()


def full_history_answer(
    question: str,
    all_session_texts: List[str],
    model: str | None = None,
) -> str:
    return generate_answer(question, all_session_texts, model=model)
