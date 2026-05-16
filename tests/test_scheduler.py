"""Tests for swarmlab.scheduler — deterministic, hermetic, network-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarmlab.dag import AgentSpec, DAGSpec, EdgeSpec, ToolSpec
from swarmlab.providers import CompletionResult, Message, Provider
from swarmlab.scheduler import RetryPolicy, Scheduler
from swarmlab.store import ContentAddressedStore
from swarmlab.tool import FunctionTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_dag(*, extra_tools: list[ToolSpec] | None = None) -> DAGSpec:
    """Two-agent linear DAG: alpha -> beta."""
    tools = extra_tools or []
    tool_names = [t.name for t in tools]
    return DAGSpec(
        name="test-dag",
        version="0.1",
        agents=[
            AgentSpec(
                name="alpha",
                model_provider="deterministic_mock",
                system_prompt="You are agent alpha.",
                tools=[t for t in tool_names if t == "counter"],
            ),
            AgentSpec(
                name="beta",
                model_provider="deterministic_mock",
                system_prompt="You are agent beta.",
            ),
        ],
        tools=tools,
        edges=[EdgeSpec(from_agent="alpha", to_agent="beta")],
        entry_point="alpha",
    )


def _mock_provider() -> Provider:
    from swarmlab.providers import get_provider

    return get_provider("deterministic_mock")


class _FailOnce:
    """Provider that raises on the first N calls then succeeds."""

    def __init__(self, fail_count: int = 1, *, delegate: Provider) -> None:
        self._remaining_failures = fail_count
        self._delegate = delegate
        self.call_count = 0

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        self.call_count += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("transient error")
        return self._delegate.complete(
            messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model
        )


class _AlwaysFail:
    """Provider that always raises."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        self.call_count += 1
        raise RuntimeError("permanent error")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_linear_dag_executes_both_agents(tmp_path: Path) -> None:
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    provider = _mock_provider()

    sched = Scheduler(dag, provider, store, run_id="run-001")
    result = sched.execute({"task": "hello"})

    assert result.status == "ok"
    assert "alpha" in result.agent_outputs
    assert "beta" in result.agent_outputs
    assert result.steps_executed == 2
    store.close()


def test_provider_called_once_per_agent(tmp_path: Path) -> None:
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))

    class _CountingProvider:
        def __init__(self, delegate: Provider) -> None:
            self._delegate = delegate
            self.calls: list[str] = []

        def complete(
            self,
            messages: list[Message],
            *,
            seed: int,
            temperature: float,
            max_tokens: int,
            model: str,
        ) -> CompletionResult:
            self.calls.append(model)
            return self._delegate.complete(
                messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model
            )

    counting = _CountingProvider(_mock_provider())
    sched = Scheduler(dag, counting, store, run_id="run-002")
    sched.execute({})

    assert len(counting.calls) == 2  # alpha and beta, once each
    store.close()


def test_retry_on_transient_failure(tmp_path: Path) -> None:
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    base_provider = _mock_provider()
    failing = _FailOnce(fail_count=1, delegate=base_provider)

    # No sleep in tests — override backoff to zero.
    policy = RetryPolicy(max_attempts=3, base_delay=0.0)
    sched = Scheduler(dag, failing, store, run_id="run-003", default_retry_policy=policy)
    result = sched.execute({})

    assert result.status == "ok"
    assert failing.call_count >= 2  # at least one retry happened
    store.close()


def test_permanent_failure_raises_after_retries_exhausted(tmp_path: Path) -> None:
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    always_fail = _AlwaysFail()

    policy = RetryPolicy(max_attempts=2, base_delay=0.0)
    sched = Scheduler(dag, always_fail, store, run_id="run-004", default_retry_policy=policy)

    with pytest.raises(RuntimeError, match="permanent error"):
        sched.execute({})

    assert always_fail.call_count == 2
    store.close()


def test_cache_hit_skips_provider_call(tmp_path: Path) -> None:
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    base_provider = _mock_provider()

    class _CountingProvider:
        def __init__(self, delegate: Provider) -> None:
            self._delegate = delegate
            self.call_count = 0

        def complete(
            self,
            messages: list[Message],
            *,
            seed: int,
            temperature: float,
            max_tokens: int,
            model: str,
        ) -> CompletionResult:
            self.call_count += 1
            return self._delegate.complete(
                messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model
            )

    counting = _CountingProvider(base_provider)

    # First execution — populates cache.
    sched1 = Scheduler(dag, counting, store, run_id="run-005")
    result1 = sched1.execute({"x": 1})
    assert result1.cache_hits == 0
    calls_after_first = counting.call_count

    # Second execution with same state — should hit cache for all agents.
    sched2 = Scheduler(dag, counting, store, run_id="run-006")
    result2 = sched2.execute({"x": 1})

    assert result2.cache_hits == 2
    # Provider should not have been called again.
    assert counting.call_count == calls_after_first
    store.close()


def test_cache_miss_tracked_on_different_state(tmp_path: Path) -> None:
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    provider = _mock_provider()

    sched1 = Scheduler(dag, provider, store, run_id="run-007")
    result1 = sched1.execute({"state": "A"})

    sched2 = Scheduler(dag, provider, store, run_id="run-008")
    result2 = sched2.execute({"state": "B"})

    assert result1.cache_hits == 0
    assert result2.cache_hits == 0
    store.close()


def test_fanout_routes_to_multiple_downstream_agents(tmp_path: Path) -> None:
    """Fan-out: entry agent emits NEXT_TARGETS pointing to two agents."""

    class _FanOutProvider:
        """Entry agent outputs NEXT_TARGETS; others return normally."""

        def __init__(self, delegate: Provider) -> None:
            self._delegate = delegate
            self.visited: list[str] = []

        def complete(
            self,
            messages: list[Message],
            *,
            seed: int,
            temperature: float,
            max_tokens: int,
            model: str,
        ) -> CompletionResult:
            system_content = next(m.content for m in messages if m.role == "system")
            if "planner" in system_content:
                self.visited.append("planner")
                return CompletionResult(
                    content='NEXT_TARGETS:["worker_a","worker_b"]',
                    model=model,
                    prompt_tokens=5,
                    completion_tokens=5,
                    finish_reason="stop",
                )
            name = "worker_a" if "worker_a" in system_content else "worker_b"
            self.visited.append(name)
            return self._delegate.complete(
                messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model
            )

    dag = DAGSpec(
        name="fanout-dag",
        version="0.1",
        agents=[
            AgentSpec(name="planner", model_provider="x", system_prompt="You are planner."),
            AgentSpec(name="worker_a", model_provider="x", system_prompt="You are worker_a."),
            AgentSpec(name="worker_b", model_provider="x", system_prompt="You are worker_b."),
        ],
        tools=[],
        edges=[
            EdgeSpec(from_agent="planner", to_agent="worker_a"),
            EdgeSpec(from_agent="planner", to_agent="worker_b"),
        ],
        entry_point="planner",
    )
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    fan_out_prov = _FanOutProvider(_mock_provider())
    sched = Scheduler(dag, fan_out_prov, store, run_id="run-009")
    result = sched.execute({})

    assert result.status == "ok"
    assert "planner" in result.agent_outputs
    assert "worker_a" in result.agent_outputs
    assert "worker_b" in result.agent_outputs
    assert set(fan_out_prov.visited) == {"planner", "worker_a", "worker_b"}
    store.close()


def test_tool_routing_delivers_result(tmp_path: Path) -> None:
    """Agent emits TOOL_CALL; scheduler dispatches to registered FunctionTool."""
    call_log: list[dict[str, object]] = []

    def _counter_fn(**kwargs: object) -> dict[str, object]:
        call_log.append(dict(kwargs))
        return {"count": 42}

    tool_spec = ToolSpec(name="counter", kind="function", parameters={})
    dag = _simple_dag(extra_tools=[tool_spec])

    class _ToolCallProvider:
        def __init__(self, delegate: Provider) -> None:
            self._delegate = delegate
            self._first = True

        def complete(
            self,
            messages: list[Message],
            *,
            seed: int,
            temperature: float,
            max_tokens: int,
            model: str,
        ) -> CompletionResult:
            if self._first:
                self._first = False
                return CompletionResult(
                    content='TOOL_CALL:counter:{"value": 7}',
                    model=model,
                    prompt_tokens=3,
                    completion_tokens=3,
                    finish_reason="stop",
                )
            return self._delegate.complete(
                messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model
            )

    store = ContentAddressedStore(str(tmp_path / "r.db"))
    prov = _ToolCallProvider(_mock_provider())
    sched = Scheduler(dag, prov, store, run_id="run-010")
    sched.register_tool("counter", FunctionTool(_counter_fn))
    result = sched.execute({})

    assert result.status == "ok"
    assert len(call_log) == 1
    assert call_log[0] == {"value": 7}
    store.close()


def test_topological_order_respected(tmp_path: Path) -> None:
    """Three agents A -> B -> C; verify execution order matches topological sort."""
    execution_order: list[str] = []

    class _TrackingProvider:
        def __init__(self, delegate: Provider) -> None:
            self._delegate = delegate

        def complete(
            self,
            messages: list[Message],
            *,
            seed: int,
            temperature: float,
            max_tokens: int,
            model: str,
        ) -> CompletionResult:
            system = next(m.content for m in messages if m.role == "system")
            for name in ("agent_a", "agent_b", "agent_c"):
                if name in system:
                    execution_order.append(name)
                    break
            return self._delegate.complete(
                messages, seed=seed, temperature=temperature, max_tokens=max_tokens, model=model
            )

    dag = DAGSpec(
        name="chain-dag",
        version="0.1",
        agents=[
            AgentSpec(name="agent_a", model_provider="x", system_prompt="You are agent_a."),
            AgentSpec(name="agent_b", model_provider="x", system_prompt="You are agent_b."),
            AgentSpec(name="agent_c", model_provider="x", system_prompt="You are agent_c."),
        ],
        tools=[],
        edges=[
            EdgeSpec(from_agent="agent_a", to_agent="agent_b"),
            EdgeSpec(from_agent="agent_b", to_agent="agent_c"),
        ],
        entry_point="agent_a",
    )
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    prov = _TrackingProvider(_mock_provider())
    sched = Scheduler(dag, prov, store, run_id="run-011")
    sched.execute({})

    assert execution_order == ["agent_a", "agent_b", "agent_c"]
    store.close()


def test_run_id_stable_across_same_input(tmp_path: Path) -> None:
    """Same input -> same output (deterministic provider wired correctly)."""
    dag = _simple_dag()
    store = ContentAddressedStore(str(tmp_path / "r.db"))
    provider = _mock_provider()

    sched1 = Scheduler(dag, provider, store, run_id="stable-run-1")
    result1 = sched1.execute({"q": "test"})

    sched2 = Scheduler(dag, provider, store, run_id="stable-run-2")
    result2 = sched2.execute({"q": "test"})

    # Outputs should be identical — same provider inputs produce same content.
    assert result1.agent_outputs["alpha"] == result2.agent_outputs["alpha"]
    assert result1.agent_outputs["beta"] == result2.agent_outputs["beta"]
    store.close()
