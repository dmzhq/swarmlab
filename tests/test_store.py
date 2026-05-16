"""Tests for ContentAddressedStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarmlab.store import ContentAddressedStore, Entry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def open_store(tmp_path: Path, name: str = "store.db") -> ContentAddressedStore:
    return ContentAddressedStore(str(tmp_path / name))


# ---------------------------------------------------------------------------
# put / get round-trips
# ---------------------------------------------------------------------------


def test_put_get_bytes(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        h = s.put(b"hello world", kind="text", mime="text/plain")
        assert s.get(h) == b"hello world"


def test_put_get_str(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        h = s.put("hello", kind="text", mime="text/plain")
        assert s.get(h) == b"hello"


def test_put_get_dict(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        payload: dict[str, int] = {"b": 2, "a": 1}
        h = s.put(payload, kind="json")
        assert s.get_json(h) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_put_dedupes_identical_content(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        h1 = s.put(b"same", kind="text")
        h2 = s.put(b"same", kind="text")
        assert h1 == h2
        assert s.blob_count(h1) == 1


# ---------------------------------------------------------------------------
# Canonical JSON: key order does not affect hash
# ---------------------------------------------------------------------------


def test_canonical_json_key_order_stability(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        h1 = s.put({"z": 1, "a": 2}, kind="json")
        h2 = s.put({"a": 2, "z": 1}, kind="json")
        assert h1 == h2
        assert s.get_json(h1) == {"z": 1, "a": 2}


# ---------------------------------------------------------------------------
# lookup_output
# ---------------------------------------------------------------------------


def test_lookup_output_returns_none_when_no_entry(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        s.put(b"input", kind="raw")
        assert s.lookup_output("deadbeef" * 4) is None


def test_lookup_output_returns_hash_when_entry_exists(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        in_h = s.put(b"input", kind="raw")
        out_h = s.put(b"output", kind="raw")
        s.create_run("r1", "dag", "1.0")
        s.record_entry("r1", 0, "agent-a", "llm", in_h, out_h)
        assert s.lookup_output(in_h) == out_h


# ---------------------------------------------------------------------------
# record_entry + iter_run_entries order
# ---------------------------------------------------------------------------


def test_iter_run_entries_preserves_order(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        s.create_run("run-x", "dag", "1.0")
        hashes = [s.put(f"payload-{i}".encode(), kind="raw") for i in range(4)]
        for i in range(3):
            s.record_entry("run-x", i, f"agent-{i}", "llm", hashes[i], hashes[i + 1])

        entries = list(s.iter_run_entries("run-x"))
        assert len(entries) == 3
        for i, entry in enumerate(entries):
            assert entry.step_index == i
            assert entry.input_hash == hashes[i]
            assert entry.output_hash == hashes[i + 1]
            assert entry.agent_name == f"agent-{i}"
        assert all(isinstance(e, Entry) for e in entries)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


def test_run_lifecycle_status_transitions(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        s.create_run("run-1", "mydag", "0.1")
        info = s.get_run("run-1")
        assert info is not None
        assert info["status"] == "running"
        assert info["ended_at"] is None

        s.finish_run("run-1", status="ok")
        info = s.get_run("run-1")
        assert info is not None
        assert info["status"] == "ok"
        assert info["ended_at"] is not None


def test_finish_run_custom_status(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        s.create_run("run-2", "mydag", "0.1")
        s.finish_run("run-2", status="error")
        info = s.get_run("run-2")
        assert info is not None
        assert info["status"] == "error"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        s.put(b"data", kind="raw")
    # After __exit__ the connection is closed; any operation should raise.
    with pytest.raises(Exception):  # noqa: B017
        s.get("anyhash")


# ---------------------------------------------------------------------------
# Persistence across re-opens
# ---------------------------------------------------------------------------


def test_data_persists_across_reopens(tmp_path: Path) -> None:
    db_path = str(tmp_path / "persist.db")
    with ContentAddressedStore(db_path) as s:
        h = s.put({"key": "value"}, kind="json")

    with ContentAddressedStore(db_path) as s:
        assert s.get_json(h) == {"key": "value"}


# ---------------------------------------------------------------------------
# KeyError on missing hash
# ---------------------------------------------------------------------------


def test_get_missing_hash_raises_key_error(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        with pytest.raises(KeyError):
            s.get("0" * 32)


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


def test_hash_is_deterministic_same_session(tmp_path: Path) -> None:
    with open_store(tmp_path) as s:
        h1 = s.put(b"deterministic", kind="raw")
        h2 = s.put(b"deterministic", kind="raw")
        assert h1 == h2


def test_hash_is_deterministic_across_sessions(tmp_path: Path) -> None:
    db = str(tmp_path / "det.db")
    with ContentAddressedStore(db) as s:
        h1 = s.put(b"cross-session", kind="raw")
    with ContentAddressedStore(db) as s:
        h2 = s.put(b"cross-session", kind="raw")
    assert h1 == h2
