"""DAG schema — Pydantic v2 models for agent workflow definitions."""

from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ToolSpec(BaseModel):
    """Describes a tool that agents can invoke."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: Literal["function", "http", "shell"]
    parameters: dict[str, object]


class AgentSpec(BaseModel):
    """Describes a single agent node in the DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    model_provider: str
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = 2048
    tools: list[str] = []

    @field_validator("temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            msg = f"temperature must be in [0.0, 2.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_positive(cls, v: int) -> int:
        if v < 1:
            msg = f"max_tokens must be >= 1, got {v}"
            raise ValueError(msg)
        return v


class EdgeSpec(BaseModel):
    """A directed edge between two agent nodes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_agent: str
    to_agent: str
    condition: str | None = None


class DAGSpec(BaseModel):
    """Top-level DAG definition: agents, tools, edges, and entry point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    agents: list[AgentSpec]
    tools: list[ToolSpec]
    edges: list[EdgeSpec]
    entry_point: str

    @model_validator(mode="after")
    def validate_dag(self) -> DAGSpec:
        agent_names = {a.name for a in self.agents}
        tool_names = {t.name for t in self.tools}

        # Duplicate agent names
        if len(agent_names) != len(self.agents):
            seen: set[str] = set()
            for a in self.agents:
                if a.name in seen:
                    msg = f"duplicate agent name: '{a.name}'"
                    raise ValueError(msg)
                seen.add(a.name)

        # Duplicate tool names
        if len(tool_names) != len(self.tools):
            seen_t: set[str] = set()
            for t in self.tools:
                if t.name in seen_t:
                    msg = f"duplicate tool name: '{t.name}'"
                    raise ValueError(msg)
                seen_t.add(t.name)

        # entry_point must be a known agent
        if self.entry_point not in agent_names:
            msg = f"entry_point '{self.entry_point}' is not a defined agent name"
            raise ValueError(msg)

        # Agents referencing unknown tools
        for agent in self.agents:
            for tool_ref in agent.tools:
                if tool_ref not in tool_names:
                    msg = f"agent '{agent.name}' references unknown tool '{tool_ref}'"
                    raise ValueError(msg)

        # Edges referencing unknown agents
        for edge in self.edges:
            if edge.from_agent not in agent_names:
                msg = f"edge from_agent '{edge.from_agent}' is not a defined agent name"
                raise ValueError(msg)
            if edge.to_agent not in agent_names:
                msg = f"edge to_agent '{edge.to_agent}' is not a defined agent name"
                raise ValueError(msg)

        # Cycle detection via Kahn's algorithm (topological sort)
        _check_for_cycles(agent_names, self.edges)

        return self


def _check_for_cycles(agent_names: set[str], edges: list[EdgeSpec]) -> None:
    """Raise ValueError if the edge set contains a cycle."""
    in_degree: dict[str, int] = dict.fromkeys(agent_names, 0)
    adjacency: dict[str, list[str]] = {name: [] for name in agent_names}

    for edge in edges:
        adjacency[edge.from_agent].append(edge.to_agent)
        in_degree[edge.to_agent] += 1

    queue: deque[str] = deque(n for n, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited != len(agent_names):
        msg = "DAG contains a cycle; v0.1 supports acyclic graphs only"
        raise ValueError(msg)
