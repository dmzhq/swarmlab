"""Content-addressed SQLite store for deterministic run replay."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from hashlib import blake2b
from typing import Any, NamedTuple


class Entry(NamedTuple):
    """One recorded step within a run."""

    run_id: str
    step_index: int
    agent_name: str | None
    kind: str
    input_hash: str
    output_hash: str
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    hash       TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    mime       TEXT,
    payload    BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dag_name    TEXT NOT NULL,
    dag_version TEXT NOT NULL,
    started_at  TEXT DEFAULT (datetime('now')),
    ended_at    TEXT,
    status      TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS entries (
    run_id      TEXT    NOT NULL,
    step_index  INTEGER NOT NULL,
    agent_name  TEXT,
    kind        TEXT    NOT NULL,
    input_hash  TEXT    NOT NULL,
    output_hash TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, step_index)
);

CREATE INDEX IF NOT EXISTS entries_input_hash ON entries(input_hash);
CREATE INDEX IF NOT EXISTS entries_run_id     ON entries(run_id);
"""


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _hash(payload: bytes) -> str:
    return blake2b(payload, digest_size=16).hexdigest()


class ContentAddressedStore:
    """Persistent content-addressed store backed by a single SQLite file."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Blob API
    # ------------------------------------------------------------------

    def put(
        self,
        payload: bytes | str | dict[str, Any],
        kind: str,
        mime: str = "application/json",
    ) -> str:
        if isinstance(payload, dict):
            raw = _canonical_json(payload)
        elif isinstance(payload, str):
            raw = payload.encode("utf-8")
        else:
            raw = payload

        h = _hash(raw)
        self._conn.execute(
            "INSERT OR IGNORE INTO blobs(hash, kind, mime, payload) VALUES (?, ?, ?, ?)",
            (h, kind, mime, raw),
        )
        self._conn.commit()
        return h

    def get(self, content_hash: str) -> bytes:
        row = self._conn.execute(
            "SELECT payload FROM blobs WHERE hash = ?", (content_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(content_hash)
        return bytes(row["payload"])

    def get_json(self, content_hash: str) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.get(content_hash))
        return result

    # ------------------------------------------------------------------
    # Entry API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        run_id: str,
        step_index: int,
        agent_name: str | None,
        kind: str,
        input_hash: str,
        output_hash: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO entries(run_id, step_index, agent_name, kind, input_hash, output_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, step_index, agent_name, kind, input_hash, output_hash),
        )
        self._conn.commit()

    def lookup_output(self, input_hash: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT output_hash FROM entries
            WHERE input_hash = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (input_hash,),
        ).fetchone()
        return str(row["output_hash"]) if row else None

    def iter_run_entries(self, run_id: str) -> Iterator[Entry]:
        rows = self._conn.execute(
            """
            SELECT run_id, step_index, agent_name, kind, input_hash, output_hash, created_at
            FROM entries
            WHERE run_id = ?
            ORDER BY step_index
            """,
            (run_id,),
        ).fetchall()
        for row in rows:
            yield Entry(
                run_id=row["run_id"],
                step_index=row["step_index"],
                agent_name=row["agent_name"],
                kind=row["kind"],
                input_hash=row["input_hash"],
                output_hash=row["output_hash"],
                created_at=row["created_at"],
            )

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, dag_name: str, dag_version: str) -> None:
        self._conn.execute(
            "INSERT INTO runs(run_id, dag_name, dag_version) VALUES (?, ?, ?)",
            (run_id, dag_name, dag_version),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, status: str = "ok") -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, ended_at = datetime('now') WHERE run_id = ?",
            (status, run_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Introspection helpers (useful for tests and diagnostics)
    # ------------------------------------------------------------------

    def blob_count(self, content_hash: str) -> int:
        """Return how many rows exist for a given hash (0 or 1)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM blobs WHERE hash = ?", (content_hash,)
        ).fetchone()
        return int(row[0])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return run metadata or None if not found."""
        row = self._conn.execute(
            "SELECT run_id, dag_name, dag_version, started_at, ended_at, status "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "dag_name": row["dag_name"],
            "dag_version": row["dag_version"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "status": row["status"],
        }

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ContentAddressedStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
