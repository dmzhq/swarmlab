"""Provider registry and Protocol for LLM completion adapters."""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable


@dataclasses.dataclass(frozen=True)
class Message:
    """A single chat message with a role and text content."""

    role: str
    content: str


@dataclasses.dataclass(frozen=True)
class CompletionResult:
    """The result of a single LLM completion call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw_response: dict[str, Any] | None = None


@runtime_checkable
class Provider(Protocol):
    """Structural protocol that all provider adapters must satisfy."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        """Execute a chat completion and return a structured result."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Provider]] = {}


def _register(name: str) -> Any:
    """Class decorator that registers a provider under *name*."""

    def _decorator(cls: type[Provider]) -> type[Provider]:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_provider(name: str) -> Provider:
    """Return an instantiated provider by registry name.

    Raises KeyError for unknown names — intentional so callers can catch it.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown provider: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


# ---------------------------------------------------------------------------
# Lazy registrations — import side-effects wire the registry.
# ---------------------------------------------------------------------------


def _bootstrap_registry() -> None:
    """Import each adapter module so their @_register decorators fire."""
    import importlib

    # Side-effect imports: each module calls @_register on load.
    _adapters = [
        "swarmlab.providers.anthropic",
        "swarmlab.providers.deterministic",
        "swarmlab.providers.local",
        "swarmlab.providers.openai",
    ]
    for _mod in _adapters:
        importlib.import_module(_mod)


_bootstrap_registry()

__all__ = [
    "CompletionResult",
    "Message",
    "Provider",
    "_register",
    "get_provider",
]
