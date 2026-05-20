"""Tests for swarmlab.cli — integration tests using typer's test client."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal two-agent YAML DAG to a temp file."""
    content = """\
name: test-dag
version: "0.1"

agents:
  - name: alpha
    model_provider: deterministic_mock
    system_prompt: "You are alpha."
    temperature: 0.0
    max_tokens: 256
    tools: []
  - name: beta
    model_provider: deterministic_mock
    system_prompt: "You are beta."
    temperature: 0.0
    max_tokens: 256
    tools: []

tools: []

edges:
  - from_agent: alpha
    to_agent: beta

entry_point: alpha
"""
    p = tmp_path / "test_dag.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# swarmlab version
# ---------------------------------------------------------------------------


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "swarmlab" in result.output
    assert "0.1" in result.output


# ---------------------------------------------------------------------------
# swarmlab run
# ---------------------------------------------------------------------------


def test_run_with_yaml_dag(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    result = runner.invoke(
        app,
        ["run", str(minimal_yaml), "--store-path", str(store), "--run-id", "cli-test-001"],
    )
    assert result.exit_code == 0, result.output
    assert "cli-test-001" in result.output
    assert "status" in result.output
    assert "ok" in result.output
    assert store.exists()


def test_run_with_state_json(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    result = runner.invoke(
        app,
        [
            "run",
            str(minimal_yaml),
            "--store-path",
            str(store),
            "--run-id",
            "cli-test-002",
            "--state",
            '{"task": "hello"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ok" in result.output


def test_run_bad_state_json_exits_nonzero(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    result = runner.invoke(
        app,
        ["run", str(minimal_yaml), "--store-path", str(store), "--state", "not-json"],
    )
    assert result.exit_code != 0


def test_run_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "dag.txt"
    bad_file.write_text("name: x")
    store = tmp_path / "test.db"
    result = runner.invoke(app, ["run", str(bad_file), "--store-path", str(store)])
    assert result.exit_code != 0


def test_run_nonexistent_dag_file(tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    result = runner.invoke(app, ["run", str(tmp_path / "no_such.yaml"), "--store-path", str(store)])
    assert result.exit_code != 0


def test_run_unknown_provider_exits_nonzero(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    result = runner.invoke(
        app,
        ["run", str(minimal_yaml), "--store-path", str(store), "--provider", "does_not_exist"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# swarmlab replay
# ---------------------------------------------------------------------------


def test_replay_after_run_is_full_cache_hit(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    # First: run to populate the store.
    run_result = runner.invoke(
        app,
        ["run", str(minimal_yaml), "--store-path", str(store), "--run-id", "source-run"],
    )
    assert run_result.exit_code == 0, run_result.output

    # Then: replay.
    replay_result = runner.invoke(
        app,
        ["replay", "source-run", str(minimal_yaml), "--store-path", str(store), "--new-run-id", "replay-001"],
    )
    assert replay_result.exit_code == 0, replay_result.output
    assert "replayed_steps:  2" in replay_result.output
    assert "live_steps:      0" in replay_result.output


def test_replay_missing_source_run_exits_nonzero(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    # Create an empty store without any run.
    from swarmlab.store import ContentAddressedStore
    with ContentAddressedStore(str(store)):
        pass

    result = runner.invoke(
        app,
        ["replay", "does-not-exist", str(minimal_yaml), "--store-path", str(store)],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# swarmlab inspect
# ---------------------------------------------------------------------------


def test_inspect_run_shows_steps(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    runner.invoke(
        app,
        ["run", str(minimal_yaml), "--store-path", str(store), "--run-id", "inspect-run"],
    )
    result = runner.invoke(app, ["inspect", "inspect-run", "--store-path", str(store)])
    assert result.exit_code == 0, result.output
    assert "inspect-run" in result.output
    assert "Steps" in result.output


def test_inspect_missing_run_exits_nonzero(tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    from swarmlab.store import ContentAddressedStore
    with ContentAddressedStore(str(store)):
        pass

    result = runner.invoke(app, ["inspect", "ghost-run", "--store-path", str(store)])
    assert result.exit_code != 0


def test_inspect_blobs_flag_prints_blob_preview(minimal_yaml: Path, tmp_path: Path) -> None:
    store = tmp_path / "test.db"
    runner.invoke(
        app,
        ["run", str(minimal_yaml), "--store-path", str(store), "--run-id", "blob-run"],
    )
    result = runner.invoke(app, ["inspect", "blob-run", "--store-path", str(store), "--blobs"])
    assert result.exit_code == 0, result.output
    assert "blob:" in result.output
