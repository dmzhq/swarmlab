"""Anthropic provider adapter backed by litellm.

Anthropic's API does not accept a seed parameter for random-number control.
Instead, determinism is achieved via input identity: the same (model,
temperature, messages) tuple always produces the same cache key in the
content-addressed store.  Callers that need replay should use the store's
``lookup_output`` with ``hash_messages`` rather than relying on temperature=0
being truly deterministic.
"""

from __future__ import annotations

import os
from typing import Any

import litellm  # type: ignore[import-untyped]

from swarmlab.providers import CompletionResult, Message, _register


@_register("anthropic")
class AnthropicProvider:
    """Wraps litellm.completion for Anthropic Claude models."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,  # recorded for cache-key derivation; not forwarded to API
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        # litellm expects the "anthropic/" prefix for Anthropic models.
        litellm_model = model if model.startswith("anthropic/") else f"anthropic/{model}"
        raw: Any = litellm.completion(
            model=litellm_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or None,
        )
        choice = raw.choices[0]
        usage = raw.usage
        return CompletionResult(
            content=choice.message.content or "",
            model=raw.model or model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "end_turn",
            raw_response=raw.model_dump() if hasattr(raw, "model_dump") else None,
        )
