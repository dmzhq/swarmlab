"""swarmlab CLI — run and replay multi-agent DAGs from the terminal."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="swarmlab",
    help="Deterministic, replayable multi-agent runs across any LLM provider.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# swarmlab run
# ---------------------------------------------------------------------------


_DAG_ARG = typer.Argument(help="Path to a YAML or Python DAG file.")
_PROVIDER_OPT = typer.Option(help="Provider name (e.g. openai, anthropic, deterministic_mock).")
_STORE_OPT = typer.Option(help="Path to the SQLite store file.")
_RUN_ID_OPT = typer.Option(help="Custom run identifier.  Defaults to a UUID.")
_STATE_OPT = typer.Option(help='Initial state as a JSON string, e.g. \'{"key": "value"}\'.')
_NEW_RUN_ID_OPT = typer.Option(help="Identifier for the replay run.  Defaults to a UUID.")
_FALLBACK_OPT = typer.Option(help="Fallback provider for cache-miss steps.")
_BLOBS_OPT = typer.Option("--blobs", help="Also print the content of each blob.")


@app.command()
def run(
    dag_path: Annotated[Path, _DAG_ARG],
    provider: Annotated[str, _PROVIDER_OPT] = "deterministic_mock",
    store_path: Annotated[Path, _STORE_OPT] = Path("swarmlab.db"),
    run_id: Annotated[str | None, _RUN_ID_OPT] = None,
    state: Annotated[str | None, _STATE_OPT] = None,
) -> None:
    """Execute a DAG and record every step to the content-addressed store."""
    import json

    from swarmlab.loaders import load_python, load_yaml
    from swarmlab.providers import get_provider
    from swarmlab.scheduler import Scheduler
    from swarmlab.store import ContentAddressedStore

    # Load DAG.
    suffix = dag_path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            dag = load_yaml(str(dag_path))
        elif suffix == ".py":
            dag = load_python(str(dag_path))
        else:
            typer.echo(
                f"Unsupported DAG file extension: {suffix!r}. Use .yaml, .yml, or .py.",
                err=True,
            )
            raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Failed to load DAG: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Resolve provider.
    try:
        prov = get_provider(provider)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    # Parse initial state.
    initial_state: dict[str, object] = {}
    if state is not None:
        try:
            parsed = json.loads(state)
            if not isinstance(parsed, dict):
                typer.echo("--state must be a JSON object (dict).", err=True)
                raise typer.Exit(code=1)
            initial_state = parsed
        except json.JSONDecodeError as exc:
            typer.echo(f"Invalid JSON in --state: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    effective_run_id = run_id or str(uuid.uuid4())

    with ContentAddressedStore(str(store_path)) as store:
        scheduler = Scheduler(dag, prov, store, run_id=effective_run_id)
        typer.echo(f"Running DAG '{dag.name}' (run_id={effective_run_id}) …")
        result = scheduler.execute(initial_state)

    typer.echo(f"status:          {result.status}")
    typer.echo(f"run_id:          {result.run_id}")
    typer.echo(f"steps_executed:  {result.steps_executed}")
    typer.echo(f"cache_hits:      {result.cache_hits}")
    typer.echo(f"store:           {store_path}")
    typer.echo("")
    for agent_name, output in result.agent_outputs.items():
        typer.echo(f"--- {agent_name} ---")
        typer.echo(output)
        typer.echo("")


# ---------------------------------------------------------------------------
# swarmlab replay
# ---------------------------------------------------------------------------


_SRC_RUN_ARG = typer.Argument(help="run_id of the original run to replay.")
_REPLAY_DAG_ARG = typer.Argument(help="Path to the DAG file to replay against.")


@app.command()
def replay(
    source_run_id: Annotated[str, _SRC_RUN_ARG],
    dag_path: Annotated[Path, _REPLAY_DAG_ARG],
    store_path: Annotated[Path, _STORE_OPT] = Path("swarmlab.db"),
    new_run_id: Annotated[str | None, _NEW_RUN_ID_OPT] = None,
    provider: Annotated[str, _FALLBACK_OPT] = "deterministic_mock",
) -> None:
    """Replay a recorded run from the content-addressed store (zero LLM cost on cache hit)."""
    from swarmlab.loaders import load_python, load_yaml
    from swarmlab.providers import get_provider
    from swarmlab.replay import ReplayEngine
    from swarmlab.store import ContentAddressedStore

    suffix = dag_path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            dag = load_yaml(str(dag_path))
        elif suffix == ".py":
            dag = load_python(str(dag_path))
        else:
            typer.echo(f"Unsupported DAG file extension: {suffix!r}.", err=True)
            raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Failed to load DAG: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        fallback = get_provider(provider)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    effective_new_run_id = new_run_id or str(uuid.uuid4())

    with ContentAddressedStore(str(store_path)) as store:
        engine = ReplayEngine(store, fallback)
        typer.echo(f"Replaying run '{source_run_id}' → new run '{effective_new_run_id}' …")
        try:
            result = engine.replay(dag, source_run_id, effective_new_run_id)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

    typer.echo(f"status:          {result.status}")
    typer.echo(f"run_id:          {result.run_id}")
    typer.echo(f"source_run_id:   {result.source_run_id}")
    typer.echo(f"steps_executed:  {result.steps_executed}")
    typer.echo(f"replayed_steps:  {result.replayed_steps}  (zero LLM cost)")
    typer.echo(f"live_steps:      {result.live_steps}")
    typer.echo("")
    for agent_name, output in result.agent_outputs.items():
        typer.echo(f"--- {agent_name} ---")
        typer.echo(output)
        typer.echo("")


# ---------------------------------------------------------------------------
# swarmlab inspect
# ---------------------------------------------------------------------------


_INSPECT_RUN_ARG = typer.Argument(help="run_id to inspect.")


@app.command()
def inspect(
    run_id: Annotated[str, _INSPECT_RUN_ARG],
    store_path: Annotated[Path, _STORE_OPT] = Path("swarmlab.db"),
    blobs: Annotated[bool, _BLOBS_OPT] = False,
) -> None:
    """Print the recorded steps and metadata for a run."""
    from swarmlab.store import ContentAddressedStore

    with ContentAddressedStore(str(store_path)) as store:
        meta = store.get_run(run_id)
        if meta is None:
            typer.echo(f"Run '{run_id}' not found in {store_path}.", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"run_id:      {meta['run_id']}")
        typer.echo(f"dag_name:    {meta['dag_name']}")
        typer.echo(f"dag_version: {meta['dag_version']}")
        typer.echo(f"started_at:  {meta['started_at']}")
        typer.echo(f"ended_at:    {meta['ended_at']}")
        typer.echo(f"status:      {meta['status']}")
        typer.echo("")

        entries = list(store.iter_run_entries(run_id))
        typer.echo(f"Steps ({len(entries)}):")
        for entry in entries:
            typer.echo(
                f"  [{entry.step_index:>4}] agent={entry.agent_name or '-':20s} "
                f"kind={entry.kind:15s} "
                f"in={entry.input_hash[:12]}… out={entry.output_hash[:12]}…"
            )
            if blobs:
                try:
                    blob_bytes = store.get(entry.output_hash)
                    preview = blob_bytes[:200].decode("utf-8", errors="replace")
                    typer.echo(f"         blob: {preview!r}")
                except KeyError:
                    typer.echo("         blob: <evicted>")


# ---------------------------------------------------------------------------
# swarmlab version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed swarmlab version."""
    from swarmlab import __version__

    typer.echo(f"swarmlab {__version__}")
    sys.exit(0)


__all__ = ["app"]
