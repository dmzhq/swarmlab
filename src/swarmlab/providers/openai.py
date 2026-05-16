"""OpenAI provider adapter backed by litellm."""

from __future__ import annotations

import os
from typing import Any

import litellm  # type: ignore[import-untyped]

from swarmlab.providers import CompletionResult, Message, _register


@_register("openai")
class OpenAIProvider:
    """Wraps litellm.completion for OpenAI models with seed wiring."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        raw: Any = litellm.completion(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            api_key=api_key or None,
        )
        choice = raw.choices[0]
        usage = raw.usage
        return CompletionResult(
            content=choice.message.content or "",
            model=raw.model or model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw_response=raw.model_dump() if hasattr(raw, "model_dump") else None,
        )
