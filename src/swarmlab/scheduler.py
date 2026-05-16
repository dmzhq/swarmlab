"""Scheduler — executes a DAGSpec against a provider with retries, fan-out, and tool routing."""

from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from typing import Any, cast

from swarmlab.dag import AgentSpec, DAGSpec
from swarmlab.providers import Message, Provider
from swarmlab.seed import compute_invocation_seed, hash_messages
from swarmlab.store import ContentAddressedStore
from swarmlab.tool import Tool, get_tool


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """Per-agent retry configuration."""

    max_attempts: int = 3  # 1 initial + 2 retries
    base_delay: float = 1.0  # seconds; doubles on each retry


@dataclasses.dataclass
class RunResult:
    """Outcome of a single DAG execution."""

    run_id: str
    status: str  # "ok" | "error"
    agent_outputs: dict[str, str]
    steps_executed: int
    cache_hits: int


def _default_retry_policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, base_delay=1.0)


def _kahn_order(dag: DAGSpec) -> list[str]:
    """Return agent names in topological order using Kahn's algorithm."""
    agent_names = [a.name for a in dag.agents]
    in_degree: dict[str, int] = dict.fromkeys(agent_names, 0)
    adjacency: dict[str, list[str]] = {n: [] for n in agent_names}

    for edge in dag.edges:
        adjacency[edge.from_agent].append(edge.to_agent)
        in_degree[edge.to_agent] += 1

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    return order


def _build_messages(
    agent: AgentSpec,
    state: dict[str, Any],
    upstream_outputs: dict[str, str],
) -> list[Message]:
    """Assemble the message list for *agent* from system prompt and current context."""
    context_parts: list[str] = []
    if state:
        context_parts.append("Current state:\n" + json.dumps(state, sort_keys=True, indent=2))
    if upstream_outputs:
        for upstream_name, output in upstream_outputs.items():
            context_parts.append(f"Output from {upstream_name}:\n{output}")

    messages: list[Message] = [Message(role="system", content=agent.system_prompt)]
    if context_parts:
        messages.append(Message(role="user", content="\n\n".join(context_parts)))
    else:
        messages.append(Message(role="user", content="Begin."))
    return messages


def _extract_tool_calls(content: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse tool-call directives from agent output.

    Convention: a line of the form ``TOOL_CALL:<name>:<json_args>`` triggers a
    tool invocation.  This is a lightweight, parseable convention that avoids
    requiring a fully-featured function-calling schema at this stage.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("TOOL_CALL:"):
            parts = stripped[len("TOOL_CALL:") :].split(":", 1)
            if len(parts) == 2:  # split(":", 1) → at most 2 parts
                tool_name = parts[0].strip()
                try:
                    tool_args: dict[str, Any] = json.loads(parts[1])
                    calls.append((tool_name, tool_args))
                except json.JSONDecodeError:
                    pass  # malformed line — skip silently
    return calls


def _extract_next_targets(content: str, dag_agent_names: set[str]) -> list[str] | None:
    """Parse fan-out targets from agent output.

    Convention: a line ``NEXT_TARGETS:<json_list>`` overrides the default
    single-path DAG edge traversal with an explicit fan-out target list.
    Only names that appear in the DAG are honoured; others are discarded.
    Returns ``None`` when no directive is present (use DAG edges instead).
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("NEXT_TARGETS:"):
            payload = stripped[len("NEXT_TARGETS:") :]
            try:
                raw: Any = json.loads(payload)
                if isinstance(raw, list):
                    raw_list = cast("list[object]", raw)
                    return [s for s in raw_list if isinstance(s, str) and s in dag_agent_names]
            except json.JSONDecodeError:
                pass
    return None


class Scheduler:
    """Executes a DAGSpec in topological order with provider dispatch.

    Features:
    - Cache-first: before calling the provider, checks the content-addressed
      store for a prior output with the same input hash.
    - Retry with exponential backoff: configurable per-agent via RetryPolicy.
    - Fan-out: agents may emit ``NEXT_TARGETS`` to branch to multiple
      downstream agents simultaneously.
    - Tool routing: agents may emit ``TOOL_CALL`` directives; the scheduler
      dispatches to registered tool implementations.
    """

    def __init__(
        self,
        dag: DAGSpec,
        provider: Provider,
        store: ContentAddressedStore,
        *,
        run_id: str,
        default_retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._dag = dag
        self._provider = provider
        self._store = store
        self._run_id = run_id
        self._default_retry = default_retry_policy or _default_retry_policy()
        # Per-name retry overrides (populated by register_retry_policy).
        self._retry_overrides: dict[str, RetryPolicy] = {}
        # Per-name tool registry (separate from module-level registry).
        self._tools: dict[str, Tool] = {}
        # Pre-compute adjacency for edge traversal.
        self._adjacency: dict[str, list[str]] = {a.name: [] for a in dag.agents}
        for edge in dag.edges:
            self._adjacency[edge.from_agent].append(edge.to_agent)
        # Topological execution order.
        self._topo_order: list[str] = _kahn_order(dag)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def register_tool(self, name: str, impl: Tool) -> None:
        """Register *impl* under *name* for use by agents in this scheduler."""
        self._tools[name] = impl

    def register_retry_policy(self, agent_name: str, policy: RetryPolicy) -> None:
        """Override the default retry policy for a specific agent."""
        self._retry_overrides[agent_name] = policy

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, initial_state: dict[str, Any]) -> RunResult:
        """Run the DAG from the entry point and return a ``RunResult``."""
        self._store.create_run(self._run_id, self._dag.name, self._dag.version)

        agent_outputs: dict[str, str] = {}
        steps_executed = 0
        cache_hits = 0
        agent_names = {a.name for a in self._dag.agents}
        # Track which agents have been scheduled for execution.
        scheduled: set[str] = set()
        # Queue of agent names to execute (respects topological order).
        run_queue: list[str] = [self._dag.entry_point]
        scheduled.add(self._dag.entry_point)

        # We iterate the topo_order to drain the queue in correct order.
        # The queue may grow (fan-out) as we proceed.
        topo_index: dict[str, int] = {name: i for i, name in enumerate(self._topo_order)}

        # Process agents following topological ordering.
        # run_queue acts as a frontier; we always pull the earliest in topo order.
        while run_queue:
            # Pick the agent with the smallest topo index from the queue.
            run_queue.sort(key=lambda n: topo_index.get(n, len(self._topo_order)))
            current_name = run_queue.pop(0)

            agent_spec = next(a for a in self._dag.agents if a.name == current_name)
            retry_policy = self._retry_overrides.get(current_name, self._default_retry)

            # Gather outputs from all upstream agents.
            upstream_names = [e.from_agent for e in self._dag.edges if e.to_agent == current_name]
            upstream_outputs = {n: agent_outputs[n] for n in upstream_names if n in agent_outputs}

            messages = _build_messages(agent_spec, initial_state, upstream_outputs)
            input_hash = hash_messages(messages)

            # Cache lookup — try to reuse a prior output for the same input hash.
            content: str | None = None
            cached_output_hash = self._store.lookup_output(input_hash)
            if cached_output_hash is not None:
                try:
                    content = self._store.get(cached_output_hash).decode("utf-8")
                    agent_outputs[current_name] = content
                    cache_hits += 1
                    steps_executed += 1
                    self._store.record_entry(
                        self._run_id,
                        steps_executed,
                        current_name,
                        "llm_cache_hit",
                        input_hash,
                        cached_output_hash,
                    )
                except KeyError:
                    # Blob evicted — fall through to live call.
                    content = None

            if content is None:
                # Live provider call with retry.
                seed = compute_invocation_seed(self._run_id, steps_executed)
                live_content, _ = self._call_with_retry(agent_spec, messages, seed, retry_policy)
                content = live_content
                agent_outputs[current_name] = content
                steps_executed += 1

                blob_hash = self._store.put(content.encode("utf-8"), kind="llm_output")
                self._store.record_entry(
                    self._run_id,
                    steps_executed,
                    current_name,
                    "llm_call",
                    input_hash,
                    blob_hash,
                )

            # Tool routing: dispatch any TOOL_CALL directives in the output.
            self._dispatch_tool_calls(content, current_name, steps_executed)

            # Fan-out: determine next agents.
            next_targets = _extract_next_targets(content, agent_names)
            if next_targets is not None:
                # Explicit fan-out from the agent's output.
                candidates = next_targets
            else:
                # Default: follow DAG edges from this agent.
                candidates = self._adjacency.get(current_name, [])

            for target in candidates:
                if target not in scheduled:
                    run_queue.append(target)
                    scheduled.add(target)

        self._store.finish_run(self._run_id, status="ok")
        return RunResult(
            run_id=self._run_id,
            status="ok",
            agent_outputs=agent_outputs,
            steps_executed=steps_executed,
            cache_hits=cache_hits,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        agent: AgentSpec,
        messages: list[Message],
        seed: int,
        policy: RetryPolicy,
    ) -> tuple[str, str]:
        """Call the provider, retrying up to ``policy.max_attempts`` times.

        Returns ``(content, "")`` on success.  Raises the last exception when
        all attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(policy.max_attempts):
            try:
                result = self._provider.complete(
                    messages,
                    seed=seed,
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens,
                    model=agent.model_provider,
                )
                return result.content, ""
            except Exception as exc:  # broad catch is intentional — bubble after retries
                last_exc = exc
                if attempt < policy.max_attempts - 1:
                    delay = policy.base_delay * (2**attempt)
                    time.sleep(delay)

        if last_exc is None:
            raise RuntimeError("retry loop exited without result or exception")
        raise last_exc

    def _dispatch_tool_calls(
        self,
        content: str,
        agent_name: str,
        step_offset: int,
    ) -> None:
        """Execute any TOOL_CALL directives found in *content*."""
        calls = _extract_tool_calls(content)
        for i, (tool_name, tool_args) in enumerate(calls):
            # Look up in scheduler-local registry first, then module-level.
            if tool_name in self._tools:
                tool_impl = self._tools[tool_name]
            else:
                try:
                    tool_impl = get_tool(tool_name)
                except KeyError:
                    continue  # unknown tool — skip; do not crash the run

            tool_result = tool_impl.execute(tool_args)
            # Record tool execution as a separate entry in the store.
            tool_call_payload = json.dumps({"name": tool_name, "args": tool_args}, sort_keys=True)
            input_hash = hash_messages([Message(role="tool_call", content=tool_call_payload)])
            output_bytes = json.dumps(tool_result, sort_keys=True).encode("utf-8")
            output_hash = self._store.put(output_bytes, kind="tool_output")
            self._store.record_entry(
                self._run_id,
                step_offset * 100 + i + 1,  # unique sub-step index
                agent_name,
                "tool_call",
                input_hash,
                output_hash,
            )


__all__ = [
    "RetryPolicy",
    "RunResult",
    "Scheduler",
]
