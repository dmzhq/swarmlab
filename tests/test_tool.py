"""Tests for swarmlab.tool — FunctionTool, registry, ShellTool."""

from __future__ import annotations

import pytest

from swarmlab.tool import FunctionTool, ShellTool, get_tool, register_tool

# ---------------------------------------------------------------------------
# FunctionTool
# ---------------------------------------------------------------------------


def test_function_tool_calls_underlying_callable() -> None:
    called_with: list[dict[str, object]] = []

    def _fn(**kwargs: object) -> dict[str, object]:
        called_with.append(dict(kwargs))
        return {"result": "ok"}

    tool = FunctionTool(_fn)
    result = tool.execute({"x": 1, "y": 2})

    assert result == {"result": "ok"}
    assert called_with == [{"x": 1, "y": 2}]


def test_function_tool_forwards_return_value() -> None:
    def _add(**kwargs: object) -> dict[str, object]:
        return {"sum": int(kwargs["a"]) + int(kwargs["b"])}  # type: ignore[arg-type]

    tool = FunctionTool(_add)
    assert tool.execute({"a": 3, "b": 4}) == {"sum": 7}


def test_function_tool_propagates_exceptions() -> None:
    def _boom(**_: object) -> dict[str, object]:
        raise ValueError("boom")

    tool = FunctionTool(_boom)
    with pytest.raises(ValueError, match="boom"):
        tool.execute({})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_returns_registered_tool() -> None:
    def _greet(**kwargs: object) -> dict[str, object]:
        return {"greeting": f"hello {kwargs.get('name', 'world')}"}

    register_tool("test_greet", FunctionTool(_greet))
    tool = get_tool("test_greet")
    assert tool.execute({"name": "alice"}) == {"greeting": "hello alice"}


def test_registry_raises_key_error_on_unknown() -> None:
    with pytest.raises(KeyError, match="Unknown tool"):
        get_tool("__nonexistent_tool_xyz__")


def test_registry_overwrites_on_re_register() -> None:
    def _v1(**_: object) -> dict[str, object]:
        return {"v": 1}

    def _v2(**_: object) -> dict[str, object]:
        return {"v": 2}

    register_tool("overwrite_me", FunctionTool(_v1))
    register_tool("overwrite_me", FunctionTool(_v2))
    assert get_tool("overwrite_me").execute({}) == {"v": 2}


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------


def test_shell_tool_timeout_fails_gracefully() -> None:
    """A shell command that sleeps past the timeout returns a timed-out result."""
    tool = ShellTool("sleep 60", timeout=0.05)
    result = tool.execute({})

    assert result["returncode"] == -1
    assert "timed out" in result["stderr"]


def test_shell_tool_format_map_substitution() -> None:
    """Template placeholders are substituted from args."""
    tool = ShellTool("echo {msg}", timeout=5.0)
    result = tool.execute({"msg": "hello"})

    assert result["returncode"] == 0
    assert "hello" in result["stdout"]
