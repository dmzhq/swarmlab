"""Tests for the provider registry and DeterministicMockProvider."""

from __future__ import annotations

import pytest

from swarmlab.providers import CompletionResult, Message, Provider, get_provider
from swarmlab.providers.deterministic import DeterministicMockProvider
from swarmlab.seed import hash_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msgs(*pairs: tuple[str, str]) -> list[Message]:
    return [Message(role=r, content=c) for r, c in pairs]


def _call(provider: Provider, seed: int = 0, model: str = "test-model") -> CompletionResult:
    return provider.complete(
        _msgs(("user", "hello")),
        seed=seed,
        temperature=0.0,
        max_tokens=64,
        model=model,
    )


# ---------------------------------------------------------------------------
# Protocol structural test
# ---------------------------------------------------------------------------


class _MinimalProvider:
    """Minimal class that structurally satisfies the Provider protocol."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        return CompletionResult(
            content="ok",
            model=model,
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )


def test_provider_protocol_structural_typing() -> None:
    """A class matching the protocol is recognised as a Provider instance."""
    p = _MinimalProvider()
    assert isinstance(p, Provider)


# ---------------------------------------------------------------------------
# DeterministicMockProvider
# ---------------------------------------------------------------------------


def test_deterministic_mock_same_input_same_output() -> None:
    """Same (messages, seed, temperature, model) always yields identical content."""
    p = DeterministicMockProvider()
    msgs = _msgs(("user", "reproduce this"))
    kwargs = {"seed": 42, "temperature": 0.5, "max_tokens": 100, "model": "mock"}
    r1 = p.complete(msgs, **kwargs)
    r2 = p.complete(msgs, **kwargs)
    assert r1.content == r2.content
    assert r1.model == r2.model


def test_deterministic_mock_different_seeds_differ() -> None:
    """Different seeds produce different outputs."""
    p = DeterministicMockProvider()
    msgs = _msgs(("user", "vary this"))
    r1 = p.complete(msgs, seed=1, temperature=0.0, max_tokens=100, model="mock")
    r2 = p.complete(msgs, seed=2, temperature=0.0, max_tokens=100, model="mock")
    assert r1.content != r2.content


def test_deterministic_mock_different_messages_differ() -> None:
    """Different message content yields different outputs."""
    p = DeterministicMockProvider()
    kwargs = {"seed": 0, "temperature": 0.0, "max_tokens": 100, "model": "mock"}
    r1 = p.complete(_msgs(("user", "message A")), **kwargs)
    r2 = p.complete(_msgs(("user", "message B")), **kwargs)
    assert r1.content != r2.content


def test_deterministic_mock_result_shape() -> None:
    """Result fields are populated and non-empty."""
    p = DeterministicMockProvider()
    result = _call(p)
    assert result.content
    assert result.model == "test-model"
    assert result.prompt_tokens >= 0
    assert result.completion_tokens > 0
    assert result.finish_reason == "stop"
    assert result.raw_response is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_provider_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown provider"):
        get_provider("no_such_provider")


@pytest.mark.parametrize("name", ["openai", "anthropic", "local", "deterministic_mock"])
def test_get_provider_known_returns_instance(name: str) -> None:
    """Known provider names return an object satisfying the Provider protocol."""
    provider = get_provider(name)
    assert isinstance(provider, Provider)


# ---------------------------------------------------------------------------
# hash_messages
# ---------------------------------------------------------------------------


def test_hash_messages_deterministic() -> None:
    msgs = _msgs(("user", "hello"), ("assistant", "world"))
    assert hash_messages(msgs) == hash_messages(msgs)


def test_hash_messages_same_content_same_hash() -> None:
    a = _msgs(("user", "same"))
    b = _msgs(("user", "same"))
    assert hash_messages(a) == hash_messages(b)


def test_hash_messages_different_content_different_hash() -> None:
    a = _msgs(("user", "hello"))
    b = _msgs(("user", "goodbye"))
    assert hash_messages(a) != hash_messages(b)


# ---------------------------------------------------------------------------
# compute_invocation_seed
# ---------------------------------------------------------------------------


def test_compute_invocation_seed_deterministic() -> None:
    from swarmlab.seed import compute_invocation_seed

    assert compute_invocation_seed("run-1", 0) == compute_invocation_seed("run-1", 0)


def test_compute_invocation_seed_different_pairs_differ() -> None:
    from swarmlab.seed import compute_invocation_seed

    seeds = {compute_invocation_seed(f"run-{i}", j) for i in range(5) for j in range(5)}
    # 25 pairs — expect at most a handful of collisions on 31-bit space.
    assert len(seeds) >= 20


def test_compute_invocation_seed_in_valid_range() -> None:
    from swarmlab.seed import compute_invocation_seed

    for i in range(10):
        s = compute_invocation_seed("run-range-test", i)
        assert 0 <= s < 2**31 - 1
