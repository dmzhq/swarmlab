"""Tests for swarmlab.replay — hermetic, network-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarmlab.dag import AgentSpec, DAGSpec, EdgeSpec
from swarmlab.providers import CompletionResult, Message, Provider, get_provider
from swarmlab.replay import ReplayEngine, ReplayProvider, ReplayResult
from swarmlab.scheduler import Scheduler
from swarmlab.store import ContentAddressedStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_agent_dag() -> DAGSpec:
    """Simple linear DAG: alpha -> beta."""
    return DAGSpec(
        name="test-dag",
        version="0.1",
        agents=[
            AgentSpec(name="alpha", model_provider="deterministic_mock", system_prompt="You are alpha."),
            AgentSpec(name="beta", model_provider="deterministic_mock", system_prompt="You are beta."),
        ],
        tools=[],
        edges=[EdgeSpec(from_agent="alpha", to_agent="beta")],
        entry_point="alpha",
    )


def _mock_provider() -> Provider:
    return get_provider("deterministic_mock")


def _run_dag(dag: DAGSpec, store: ContentAddressedStore, run_id: str, state: dict[str, object] | None = None) -> None:
    """Execute *dag* against *store* — caller owns store lifecycle."""
    scheduler = Scheduler(dag, _mock_provider(), store, run_id=run_id)
    scheduler.execute(state or {})
    # Do NOT close store here; callers share the store and close it themselves.


# ---------------------------------------------------------------------------
# ReplayProvider tests
# ---------------------------------------------------------------------------


def test_replay_provider_serves_from_store_on_cache_hit(tmp_path: Path) -> None:
    """ReplayProvider returns replay result when the blob is in the store."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    # Populate the store with a real run.
    _run_dag(dag, store, "original-run", {"task": "hello"})

    call_count = 0

    class _CountingLive:
        def complete(self, messages: list[Message], *, seed: int, temperature: float, max_tokens: int, model: str) -> CompletionResult:
            nonlocal call_count
            call_count += 1
            return _mock_provider().complete(messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model)

    replay_prov = ReplayProvider(store, _CountingLive())

    # Drive alpha manually — same messages as the original run.
    from swarmlab.scheduler import _build_messages
    alpha_spec = next(a for a in dag.agents if a.name == "alpha")
    messages = _build_messages(alpha_spec, {"task": "hello"}, {})
    result = replay_prov.complete(messages, seed=0, temperature=0.0, max_tokens=512, model="deterministic_mock")

    assert result.finish_reason == "replay"
    assert replay_prov.replayed == 1
    assert replay_prov.live_calls == 0
    assert call_count == 0
    store.close()


def test_replay_provider_falls_back_to_live_on_miss(tmp_path: Path) -> None:
    """ReplayProvider falls back to live when no matching blob exists."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    call_count = 0

    class _CountingLive:
        def complete(self, messages: list[Message], *, seed: int, temperature: float, max_tokens: int, model: str) -> CompletionResult:
            nonlocal call_count
            call_count += 1
            return _mock_provider().complete(messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model)

    replay_prov = ReplayProvider(store, _CountingLive())

    # Messages that were never stored.
    messages = [Message(role="system", content="brand new"), Message(role="user", content="hi")]
    replay_prov.complete(messages, seed=1, temperature=0.0, max_tokens=256, model="deterministic_mock")

    assert replay_prov.replayed == 0
    assert replay_prov.live_calls == 1
    assert call_count == 1
    store.close()


# ---------------------------------------------------------------------------
# ReplayEngine tests
# ---------------------------------------------------------------------------


def test_replay_full_run_zero_live_calls(tmp_path: Path) -> None:
    """Full replay of a two-agent run should require zero live LLM calls."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    _run_dag(dag, store, "original-run", {"x": 42})

    live_calls = 0

    class _ForbidLive:
        def complete(self, messages: list[Message], *, seed: int, temperature: float, max_tokens: int, model: str) -> CompletionResult:
            nonlocal live_calls
            live_calls += 1
            return _mock_provider().complete(messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model)

    engine = ReplayEngine(store, _ForbidLive())
    result = engine.replay(dag, "original-run", "replay-001", initial_state={"x": 42})

    assert result.status == "ok"
    assert result.source_run_id == "original-run"
    assert result.run_id == "replay-001"
    assert result.replayed_steps == 2
    assert result.live_steps == 0
    assert live_calls == 0
    store.close()


def test_replay_outputs_match_original(tmp_path: Path) -> None:
    """Replayed agent outputs must be identical to the original run's outputs."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    scheduler = Scheduler(dag, _mock_provider(), store, run_id="original-run")
    original = scheduler.execute({"q": "test"})

    engine = ReplayEngine(store, _mock_provider())
    replayed = engine.replay(dag, "original-run", "replay-002", initial_state={"q": "test"})

    assert replayed.agent_outputs["alpha"] == original.agent_outputs["alpha"]
    assert replayed.agent_outputs["beta"] == original.agent_outputs["beta"]
    store.close()


def test_replay_missing_source_run_raises(tmp_path: Path) -> None:
    """ReplayEngine raises ValueError when the source run is not in the store."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    engine = ReplayEngine(store, _mock_provider())
    with pytest.raises(ValueError, match="not found in store"):
        engine.replay(dag, "does-not-exist", "replay-003")
    store.close()


def test_replay_same_run_id_raises(tmp_path: Path) -> None:
    """ReplayEngine raises ValueError when source and new run IDs are equal."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    _run_dag(dag, store, "original-run")

    engine = ReplayEngine(store, _mock_provider())
    with pytest.raises(ValueError, match="must differ"):
        engine.replay(dag, "original-run", "original-run")
    store.close()


def test_replay_steps_executed_matches_agent_count(tmp_path: Path) -> None:
    """Steps executed in a replay equals the number of agents run."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    _run_dag(dag, store, "original-run", {})

    engine = ReplayEngine(store, _mock_provider())
    result = engine.replay(dag, "original-run", "replay-004")

    assert result.steps_executed == len(dag.agents)
    store.close()


def test_replay_engine_no_live_provider_raises_on_miss(tmp_path: Path) -> None:
    """Without a live provider, cache miss raises RuntimeError."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    # Populate run with state "A".
    _run_dag(dag, store, "original-run", {"state": "A"})

    # Try to replay with state "B" — different hash, guaranteed miss.
    engine = ReplayEngine(store)  # no live provider
    with pytest.raises(RuntimeError, match="no live provider was configured"):
        engine.replay(dag, "original-run", "replay-005", initial_state={"state": "B"})
    store.close()


def test_replay_result_fields_populated(tmp_path: Path) -> None:
    """ReplayResult contains all expected fields after a successful replay."""
    dag = _two_agent_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    _run_dag(dag, store, "src-run", {})

    engine = ReplayEngine(store, _mock_provider())
    result = engine.replay(dag, "src-run", "new-run")

    assert isinstance(result, ReplayResult)
    assert result.run_id == "new-run"
    assert result.source_run_id == "src-run"
    assert result.status == "ok"
    assert "alpha" in result.agent_outputs
    assert "beta" in result.agent_outputs
    assert result.steps_executed >= 0
    assert result.replayed_steps + result.live_steps == result.steps_executed
    store.close()
