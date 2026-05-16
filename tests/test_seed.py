"""Focused tests for the swarmlab.seed module."""

from __future__ import annotations

from swarmlab.providers import Message
from swarmlab.seed import compute_invocation_seed, hash_messages

_MAX_SEED = 2**31 - 1


# ---------------------------------------------------------------------------
# hash_messages
# ---------------------------------------------------------------------------


def test_hash_messages_empty_list() -> None:
    result = hash_messages([])
    assert isinstance(result, str)
    assert len(result) == 32  # blake2b digest_size=16 → 32 hex chars


def test_hash_messages_single_message() -> None:
    msgs = [Message(role="user", content="hello")]
    result = hash_messages(msgs)
    assert isinstance(result, str)
    assert len(result) == 32


def test_hash_messages_multi_role_conversation() -> None:
    msgs = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="And 3+3?"),
    ]
    result = hash_messages(msgs)
    assert isinstance(result, str)
    assert len(result) == 32


def test_hash_messages_stable_across_calls() -> None:
    """Same input always returns the same hash, independent of call order."""
    msgs = [
        Message(role="user", content="reproduce me"),
        Message(role="assistant", content="sure"),
    ]
    results = [hash_messages(msgs) for _ in range(10)]
    assert len(set(results)) == 1


def test_hash_messages_order_matters() -> None:
    """Reversing message order changes the hash."""
    a = [Message(role="user", content="first"), Message(role="assistant", content="second")]
    b = [Message(role="assistant", content="second"), Message(role="user", content="first")]
    assert hash_messages(a) != hash_messages(b)


def test_hash_messages_role_matters() -> None:
    """Same content but different role produces a different hash."""
    user_msg = [Message(role="user", content="ping")]
    sys_msg = [Message(role="system", content="ping")]
    assert hash_messages(user_msg) != hash_messages(sys_msg)


def test_hash_messages_content_matters() -> None:
    a = [Message(role="user", content="alpha")]
    b = [Message(role="user", content="beta")]
    assert hash_messages(a) != hash_messages(b)


# ---------------------------------------------------------------------------
# compute_invocation_seed
# ---------------------------------------------------------------------------


def test_compute_invocation_seed_in_valid_range() -> None:
    for step in range(20):
        seed = compute_invocation_seed("run-abc", step)
        assert 0 <= seed < _MAX_SEED


def test_compute_invocation_seed_deterministic() -> None:
    run_id = "deterministic-run"
    for step in range(10):
        assert compute_invocation_seed(run_id, step) == compute_invocation_seed(run_id, step)


def test_compute_invocation_seed_step_variation() -> None:
    """Adjacent steps within the same run produce different seeds."""
    run_id = "step-vary"
    seeds = [compute_invocation_seed(run_id, i) for i in range(10)]
    assert len(set(seeds)) == len(seeds), "Adjacent step seeds collided"


def test_compute_invocation_seed_run_variation() -> None:
    """Different run_ids at the same step produce different seeds."""
    step = 0
    seeds = [compute_invocation_seed(f"run-{i}", step) for i in range(10)]
    assert len(set(seeds)) == len(seeds), "Run seeds collided at the same step"


def test_compute_invocation_seed_is_int() -> None:
    result = compute_invocation_seed("type-check-run", 0)
    assert isinstance(result, int)
