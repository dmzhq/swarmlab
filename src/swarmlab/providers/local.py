"""Local LLM provider — speaks the OpenAI-compatible chat completions API.

Supports vLLM and Ollama out of the box.  Point ``LOCAL_LLM_ENDPOINT`` at
any server that serves ``POST /v1/chat/completions`` with an OpenAI-shaped
request/response body.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from swarmlab.providers import CompletionResult, Message, _register

_DEFAULT_ENDPOINT = "http://localhost:11434"
_TIMEOUT = 120.0


@_register("local")
class LocalProvider:
    """Sends requests to a local OpenAI-compatible inference server."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", _DEFAULT_ENDPOINT).rstrip("/")
        url = f"{endpoint}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return CompletionResult(
            content=choice["message"]["content"] or "",
            model=data.get("model", model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason", "stop") or "stop",
            raw_response=data,
        )
