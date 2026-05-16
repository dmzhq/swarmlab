"""YAML and Python loaders for DAGSpec definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from swarmlab.dag import DAGSpec


class DAGValidationError(ValueError):
    """Raised when a DAG file fails schema validation or cannot be parsed."""


def load_yaml(path: Path | str) -> DAGSpec:
    """Load a DAGSpec from a YAML file.

    Raises DAGValidationError with the file path and validation details on failure.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        msg = f"DAG file not found: {resolved}"
        raise DAGValidationError(msg)

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Could not read {resolved}: {exc}"
        raise DAGValidationError(msg) from exc

    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"YAML parse error in {resolved}: {exc}"
        raise DAGValidationError(msg) from exc

    if not isinstance(data, dict):
        msg = f"{resolved}: expected a YAML mapping at the top level, got {type(data).__name__}"
        raise DAGValidationError(msg)

    return _validate(data, source=str(resolved))


def load_python(module_path: Path | str) -> DAGSpec:
    """Load a DAGSpec from a Python file that exposes a module-level `dag` variable.

    Raises DAGValidationError if the file is missing, does not expose `dag`,
    or `dag` is not a valid DAGSpec.
    """
    resolved = Path(module_path).resolve()
    if not resolved.exists():
        msg = f"DAG module not found: {resolved}"
        raise DAGValidationError(msg)

    module_name = f"_swarmlab_dag_{resolved.stem}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        msg = f"Could not create module spec for {resolved}"
        raise DAGValidationError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        sys.modules.pop(module_name, None)
        msg = f"Error executing {resolved}: {exc}"
        raise DAGValidationError(msg) from exc

    dag = getattr(module, "dag", None)
    if dag is None:
        msg = f"{resolved}: module must expose a top-level variable named 'dag'"
        raise DAGValidationError(msg)

    if isinstance(dag, DAGSpec):
        return dag

    if isinstance(dag, dict):
        return _validate(dag, source=str(resolved))

    msg = f"{resolved}: 'dag' must be a DAGSpec instance or a dict, got {type(dag).__name__}"
    raise DAGValidationError(msg)


def _validate(data: dict[str, Any], source: str) -> DAGSpec:
    """Wrap Pydantic validation and re-raise as DAGValidationError."""
    try:
        return DAGSpec.model_validate(data)
    except (ValidationError, ValueError) as exc:
        msg = f"DAG validation failed in {source}:\n{exc}"
        raise DAGValidationError(msg) from exc
