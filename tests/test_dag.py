"""Tests for DAGSpec validation and loader happy paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarmlab import DAGSpec, DAGValidationError, load_python, load_yaml
from swarmlab.dag import AgentSpec, EdgeSpec, ToolSpec

EXAMPLES = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dag(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "test-dag",
        "version": "0.1.0",
        "agents": [
            {
                "name": "alpha",
                "model_provider": "openai/gpt-4o-mini",
                "system_prompt": "You are alpha.",
            },
            {
                "name": "beta",
                "model_provider": "openai/gpt-4o-mini",
                "system_prompt": "You are beta.",
            },
        ],
        "tools": [],
        "edges": [{"from_agent": "alpha", "to_agent": "beta"}],
        "entry_point": "alpha",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid DAGSpec construction
# ---------------------------------------------------------------------------


def test_valid_dag_passes_validation() -> None:
    dag = DAGSpec.model_validate(_minimal_dag())
    assert dag.name == "test-dag"
    assert len(dag.agents) == 2
    assert dag.entry_point == "alpha"


def test_defaults_are_applied() -> None:
    dag = DAGSpec.model_validate(_minimal_dag())
    agent = dag.agents[0]
    assert agent.temperature == 0.0
    assert agent.max_tokens == 2048
    assert agent.tools == []


# ---------------------------------------------------------------------------
# Duplicate name validation
# ---------------------------------------------------------------------------


def test_duplicate_agent_names_raise() -> None:
    data = _minimal_dag()
    agents = list(data["agents"])  # type: ignore[arg-type]
    agents.append(
        {
            "name": "alpha",  # duplicate
            "model_provider": "openai/gpt-4o-mini",
            "system_prompt": "Duplicate.",
        }
    )
    data["agents"] = agents

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    assert "alpha" in str(exc_info.value)


# ---------------------------------------------------------------------------
# entry_point validation
# ---------------------------------------------------------------------------


def test_missing_entry_point_raises() -> None:
    data = _minimal_dag(entry_point="nonexistent")

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    assert "entry_point" in str(exc_info.value) or "nonexistent" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------


def test_edge_to_nonexistent_agent_raises() -> None:
    data = _minimal_dag()
    data["edges"] = [{"from_agent": "alpha", "to_agent": "ghost"}]

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    assert "ghost" in str(exc_info.value)


def test_edge_from_nonexistent_agent_raises() -> None:
    data = _minimal_dag()
    data["edges"] = [{"from_agent": "ghost", "to_agent": "beta"}]

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    assert "ghost" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def test_cycle_raises() -> None:
    data: dict[str, object] = {
        "name": "cyclic-dag",
        "version": "0.1.0",
        "agents": [
            {"name": "a", "model_provider": "openai/gpt-4o-mini", "system_prompt": "A"},
            {"name": "b", "model_provider": "openai/gpt-4o-mini", "system_prompt": "B"},
            {"name": "c", "model_provider": "openai/gpt-4o-mini", "system_prompt": "C"},
        ],
        "tools": [],
        "edges": [
            {"from_agent": "a", "to_agent": "b"},
            {"from_agent": "b", "to_agent": "c"},
            {"from_agent": "c", "to_agent": "a"},  # closes the cycle
        ],
        "entry_point": "a",
    }

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    assert "cycle" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Agent referencing unknown tool
# ---------------------------------------------------------------------------


def test_agent_referencing_unknown_tool_raises() -> None:
    data = _minimal_dag()
    agents = list(data["agents"])  # type: ignore[arg-type]
    agents[0] = {
        "name": "alpha",
        "model_provider": "openai/gpt-4o-mini",
        "system_prompt": "Alpha.",
        "tools": ["nonexistent_tool"],
    }
    data["agents"] = agents

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    assert "nonexistent_tool" in str(exc_info.value)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_yaml_loader_happy_path() -> None:
    yaml_file = EXAMPLES / "01_minimal_two_agents.yaml"
    dag = load_yaml(yaml_file)
    assert dag.name == "minimal-two-agents"
    assert dag.entry_point == "planner"
    assert len(dag.agents) == 2
    assert len(dag.edges) == 1


def test_yaml_loader_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "this_file_does_not_exist_swarmlab.yaml"
    with pytest.raises(DAGValidationError) as exc_info:
        load_yaml(missing)

    assert "not found" in str(exc_info.value).lower()


def test_yaml_loader_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: test\nagents: [{{{{", encoding="utf-8")

    with pytest.raises(DAGValidationError) as exc_info:
        load_yaml(bad)

    assert "yaml" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Python loader
# ---------------------------------------------------------------------------


def test_python_loader_happy_path() -> None:
    py_file = EXAMPLES / "02_retrieval_then_synth.py"
    dag = load_python(py_file)
    assert dag.name == "retrieval-then-synth"
    assert dag.entry_point == "planner"
    assert len(dag.agents) == 3
    assert len(dag.edges) == 2


def test_python_loader_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "this_file_does_not_exist_swarmlab.py"
    with pytest.raises(DAGValidationError) as exc_info:
        load_python(missing)

    assert "not found" in str(exc_info.value).lower()


def test_python_loader_missing_dag_variable_raises(tmp_path: Path) -> None:
    mod = tmp_path / "no_dag.py"
    mod.write_text("# no 'dag' variable here\nx = 42\n", encoding="utf-8")

    with pytest.raises(DAGValidationError) as exc_info:
        load_python(mod)

    assert "dag" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Error messages contain field context
# ---------------------------------------------------------------------------


def test_error_message_names_the_bad_field() -> None:
    data = _minimal_dag()
    data["edges"] = [{"from_agent": "alpha", "to_agent": "phantom_node"}]

    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        DAGSpec.model_validate(data)

    # The error should identify the unknown agent name
    assert "phantom_node" in str(exc_info.value)


def test_temperature_out_of_range_raises() -> None:
    with pytest.raises((ValueError, DAGValidationError)) as exc_info:
        AgentSpec(
            name="bad",
            model_provider="openai/gpt-4o-mini",
            system_prompt="Temp test.",
            temperature=3.5,
        )

    assert "temperature" in str(exc_info.value)


def test_tool_kind_must_be_literal() -> None:
    with pytest.raises((ValueError, DAGValidationError)):
        ToolSpec(name="t", kind="websocket", parameters={})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EdgeSpec optional condition
# ---------------------------------------------------------------------------


def test_edge_condition_is_optional() -> None:
    edge_no_cond = EdgeSpec(from_agent="a", to_agent="b")
    assert edge_no_cond.condition is None

    edge_with_cond = EdgeSpec(from_agent="a", to_agent="b", condition="output.success == true")
    assert edge_with_cond.condition == "output.success == true"
