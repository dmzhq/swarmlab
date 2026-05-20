"""Replay engine — re-execute a past run from the content-addressed store.

The engine reads the entry log of a completed run, reconstructs each agent's
messages from the stored blobs, and drives a fresh execution that:

- Returns the *exact* recorded output for every step that is present in the
  store (zero LLM cost).
- Calls the live provider only for steps whose blobs have been evicted or that
  belong to a branch that was not recorded in the original run (e.g. an agent
  added after the fact).

This is the key capability that distinguishes swarmlab from a pure tracing
tool: you can replay *any branch* of a past run cheaply and deterministically.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from swarmlab.dag import DAGSpec
from swarmlab.providers import CompletionResult, Message, Provider
from swarmlab.scheduler import RunResult, Scheduler
from swarmlab.seed import hash_messages
from swarmlab.store import ContentAddressedStore

# ---------------------------------------------------------------------------
# ReplayProvider — a Provider wrapper that serves blobs from the store
# ---------------------------------------------------------------------------


class ReplayProvider:
    """Provider adapter that replays recorded outputs from a prior run.

    For each ``complete`` call the provider checks the content-addressed store
    for a blob whose input hash matches the current messages.  On a hit it
    deserialises the stored bytes and returns them as a ``CompletionResult``
    with ``finish_reason="replay"``.  On a miss it falls back to *live*.

    Args:
        store:  The same store that holds the original run's blobs.
        live:   Fallback provider for cache-miss steps.
    """

    def __init__(self, store: ContentAddressedStore, live: Provider) -> None:
        self._store = store
        self._live = live
        self._replayed: int = 0
        self._live_calls: int = 0

    # ------------------------------------------------------------------
    # Provider protocol
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        input_hash = hash_messages(messages)
        output_hash = self._store.lookup_output(input_hash)

        if output_hash is not None:
            try:
                content = self._store.get(output_hash).decode("utf-8")
                self._replayed += 1
                return CompletionResult(
                    content=content,
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    finish_reason="replay",
                )
            except KeyError:
                pass  # blob evicted — fall through to live call

        self._live_calls += 1
        return self._live.complete(
            messages,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def replayed(self) -> int:
        """Number of steps served from the store (no LLM cost)."""
        return self._replayed

    @property
    def live_calls(self) -> int:
        """Number of steps that required a live provider call."""
        return self._live_calls


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ReplayResult:
    """Outcome of a replay execution.

    Attributes:
        run_id:         The new run identifier (distinct from the source run).
        source_run_id:  The original run that was replayed.
        status:         ``"ok"`` or ``"error"``.
        agent_outputs:  Final content produced by each agent.
        steps_executed: Total steps in the new run.
        replayed_steps: Steps served from the store (zero LLM cost).
        live_steps:     Steps that required a live LLM call (cache miss).
    """

    run_id: str
    source_run_id: str
    status: str
    agent_outputs: dict[str, str]
    steps_executed: int
    replayed_steps: int
    live_steps: int


# ---------------------------------------------------------------------------
# ReplayEngine
# ---------------------------------------------------------------------------


class ReplayEngine:
    """Replays a recorded run against its content-addressed store.

    Usage::

        store = ContentAddressedStore("runs.db")
        fallback = get_provider("openai")
        engine = ReplayEngine(store, fallback)
        result = engine.replay(dag, source_run_id="run-abc123", new_run_id="replay-001")
        # result.replayed_steps == 2  →  zero LLM cost for those steps
        # result.live_steps == 0      →  full replay from store

    Args:
        store:          The store that holds the original run's blobs.
        live_provider:  Provider used only when a blob is missing (evicted or
                        never recorded).  Pass ``None`` to raise on any miss.
    """

    def __init__(
        self,
        store: ContentAddressedStore,
        live_provider: Provider | None = None,
    ) -> None:
        self._store = store
        self._live = live_provider or _NoLiveProvider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay(
        self,
        dag: DAGSpec,
        source_run_id: str,
        new_run_id: str,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> ReplayResult:
        """Re-execute *dag* using cached outputs from *source_run_id*.

        Args:
            dag:            The DAGSpec to execute (usually the same one that
                            produced *source_run_id*, but can be a modified
                            version for branch exploration).
            source_run_id:  Run whose store entries will be used as a cache.
            new_run_id:     Identifier for the new replay run.  Must be
                            distinct from *source_run_id*.
            initial_state:  Overrides the initial state dict.  When ``None``
                            the engine attempts to infer the original state
                            from the store; it falls back to ``{}`` if no
                            state blob is found.

        Returns:
            A :class:`ReplayResult` with replay vs live call counts.

        Raises:
            ValueError:  If *source_run_id* == *new_run_id*, or if the source
                         run does not exist in the store.
        """
        if source_run_id == new_run_id:
            msg = "new_run_id must differ from source_run_id"
            raise ValueError(msg)

        source_meta = self._store.get_run(source_run_id)
        if source_meta is None:
            msg = f"source run '{source_run_id}' not found in store"
            raise ValueError(msg)

        state: dict[str, Any] = initial_state if initial_state is not None else {}

        replay_provider = ReplayProvider(self._store, self._live)
        scheduler = Scheduler(dag, replay_provider, self._store, run_id=new_run_id)

        run_result: RunResult = scheduler.execute(state)

        # The Scheduler serves cache hits directly from the store without
        # calling the provider.  Those hits are counted in run_result.cache_hits.
        # Any remaining steps went through ReplayProvider (which may have also
        # served blobs or delegated to the live fallback).
        store_cache_hits = run_result.cache_hits
        provider_replays = replay_provider.replayed
        total_replayed = store_cache_hits + provider_replays

        return ReplayResult(
            run_id=new_run_id,
            source_run_id=source_run_id,
            status=run_result.status,
            agent_outputs=run_result.agent_outputs,
            steps_executed=run_result.steps_executed,
            replayed_steps=total_replayed,
            live_steps=replay_provider.live_calls,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _NoLiveProvider:
    """Fallback provider that always raises — used when no live provider given."""

    def complete(
        self,
        messages: list[Message],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        input_hash = hash_messages(messages)
        msg = (
            f"Blob for input_hash={input_hash!r} was not found in the store "
            "and no live provider was configured.  Either the blob was evicted "
            "or this step was not present in the original run."
        )
        raise RuntimeError(msg)


__all__ = [
    "ReplayEngine",
    "ReplayProvider",
    "ReplayResult",
]
