"""Seed wiring utilities for deterministic per-step sampling.

All functions are pure and rely solely on BLAKE2b, matching the hash
family used by the content-addressed store.
"""

from __future__ import annotations

import hashlib
import json

from swarmlab.providers import Message

# OpenAI seed field accepts integers in [0, 2**31 - 1].
_MAX_SEED = 2**31 - 1


def hash_messages(messages: list[Message]) -> str:
    """Return a stable hex digest that canonically identifies *messages*.

    The digest is stable across Python sessions because it is derived from
    a deterministic serialisation (sorted-key JSON) fed into BLAKE2b.
    """
    canonical = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def compute_invocation_seed(run_id: str, step_index: int) -> int:
    """Return a stable, non-negative seed for *run_id* at *step_index*.

    The seed lies in ``[0, 2**31 - 1]`` for compatibility with the OpenAI
    API's ``seed`` parameter.  Different (run_id, step_index) pairs produce
    different seeds with high probability (birthday bound on 31 bits).
    """
    key = f"{run_id}:{step_index}".encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    # Interpret 4 bytes as unsigned big-endian integer, then clamp.
    value = int.from_bytes(digest, "big")
    return value % _MAX_SEED
