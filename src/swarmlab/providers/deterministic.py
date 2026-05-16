"""Deterministic mock provider for hermetic, network-free tests.

Output is keyed on (model, seed, messages_hash, temperature).  The same
inputs always produce the same pseudo-response; different inputs produce
different pseudo-responses with high probability.
"""

from __future__ import annotations

import hashlib
import random

from swarmlab.providers import CompletionResult, Message, _register
from swarmlab.seed import hash_messages

# Small wordlist used to assemble pseudo-responses.
_WORDS = [
    "agent",
    "batch",
    "cache",
    "delta",
    "epoch",
    "flow",
    "graph",
    "hash",
    "infer",
    "join",
    "kernel",
    "layer",
    "model",
    "node",
    "output",
    "plan",
    "queue",
    "replay",
    "seed",
    "trace",
    "unit",
    "vector",
    "weight",
    "xor",
    "yield",
    "zero",
]

_RESPONSE_WORD_COUNT = 12


def _pseudo_response(prng: random.Random) -> str:
    """Generate a readable fixed-length pseudo-response from *prng*."""
    words = [prng.choice(_WORDS) for _ in range(_RESPONSE_WORD_COUNT)]
    # Capitalise first word and append a period for realism.
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _derive_prng_seed(model: str, seed: int, messages_hash: str, temperature: float) -> int:
    """Combine inputs into a single integer suitable for ``random.Random``."""
    key = f"{model}|{seed}|{messages_hash}|{temperature:.6f}"
    digest = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
    return int(digest, 16)


@_register("deterministic_mock")
class DeterministicMockProvider:
    """Zero-network provider that returns deterministic pseudo-responses."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        msg_hash = hash_messages(messages)
        prng_seed = _derive_prng_seed(model, seed, msg_hash, temperature)
        prng = random.Random(prng_seed)  # noqa: S311 — not security-sensitive
        content = _pseudo_response(prng)
        # Simulate token counts proportional to word count.
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = _RESPONSE_WORD_COUNT
        return CompletionResult(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
            raw_response=None,
        )
