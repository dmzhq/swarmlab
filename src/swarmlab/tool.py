"""Tool execution layer — abstract protocol plus concrete implementations."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import httpx


@runtime_checkable
class Tool(Protocol):
    """Structural protocol that all tool implementations must satisfy."""

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with the given arguments and return a result dict."""
        ...


class FunctionTool:
    """Wraps a plain Python callable as a Tool."""

    def __init__(self, fn: Callable[..., dict[str, Any]]) -> None:
        self._fn = fn

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._fn(**args)


class HTTPTool:
    """Calls an HTTP endpoint and returns the parsed JSON response body.

    The endpoint receives *args* as a JSON POST body.  The response must be
    JSON-decodable; non-2xx status codes raise ``httpx.HTTPStatusError``.
    """

    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        self._url = url
        self._timeout = timeout

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(self._url, json=args, timeout=self._timeout)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


class ShellTool:
    """Runs a shell command template with sandboxing constraints.

    *command_template* is a format string.  ``args`` keys are substituted
    via ``str.format_map``.  Execution is bounded by *timeout* seconds.
    The environment is stripped to a minimal safe set.
    """

    _SAFE_ENV_KEYS = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"})

    def __init__(self, command_template: str, *, timeout: float = 5.0) -> None:
        self._template = command_template
        self._timeout = timeout

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        import os

        cmd = self._template.format_map({k: str(v) for k, v in args.items()})
        env = {k: v for k, v in os.environ.items() if k in self._SAFE_ENV_KEYS}

        try:
            proc = subprocess.run(  # noqa: S602
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "timed out", "returncode": -1}

        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }


# ---------------------------------------------------------------------------
# Module-level registry (process-scoped singleton)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Tool] = {}


def register_tool(name: str, impl: Tool) -> None:
    """Register *impl* under *name* so agents can invoke it by name."""
    _REGISTRY[name] = impl


def get_tool(name: str) -> Tool:
    """Return the registered tool named *name*.

    Raises ``KeyError`` if the name is unknown — intentional so callers can
    distinguish missing tools from execution errors.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = [
    "FunctionTool",
    "HTTPTool",
    "ShellTool",
    "Tool",
    "get_tool",
    "register_tool",
]
