"""Route to OpenAI API or free local Ollama (OpenAI-compatible)."""
from __future__ import annotations

import os
from openai import OpenAI


def get_chat_client() -> tuple[OpenAI, str]:
    """
    Returns (client, model_name).

    Priority:
      1) USE_OLLAMA=1  -> Ollama at OLLAMA_BASE_URL (default http://localhost:11434/v1)
      2) OPENAI_API_KEY -> OpenAI cloud
    """
    use_ollama = os.environ.get("USE_OLLAMA", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if use_ollama or not os.environ.get("OPENAI_API_KEY"):
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        client = OpenAI(base_url=base, api_key="ollama")
        return client, model

    model = os.environ.get("OPENAI_READER_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client, model
