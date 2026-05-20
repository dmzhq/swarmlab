# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-20

### Added

- **Replay engine** (`swarmlab.replay`): `ReplayEngine` re-executes any past
  run from the content-addressed store with zero LLM cost on full cache hit.
  `ReplayProvider` wraps any live provider and intercepts calls that have a
  matching blob in the store. `ReplayResult` reports replayed vs live step
  counts.
- **CLI** (`swarmlab.cli`): three commands wired to `pyproject.toml`'s
  `[project.scripts]` entry:
  - `swarmlab run <dag>` — execute a DAG and record every step.
  - `swarmlab replay <run-id> <dag>` — replay a recorded run.
  - `swarmlab inspect <run-id>` — print steps and metadata for a run.
  - `swarmlab version` — print installed version.
- **DAG schema + YAML/Python loaders** (`swarmlab.dag`, `swarmlab.loaders`):
  Pydantic v2 models for `DAGSpec`, `AgentSpec`, `EdgeSpec`, `ToolSpec` with
  cycle detection and full validation.
- **Content-addressed store** (`swarmlab.store`): SQLite-backed store keyed
  by BLAKE2b hashes. Stores blobs, run metadata, and per-step entries.
  Supports cache lookup by input hash for deterministic replay.
- **Provider adapter** (`swarmlab.providers`): structural `Provider` protocol
  with adapters for OpenAI, Anthropic, and locally-served models (Ollama /
  vLLM via litellm). Includes a `DeterministicMockProvider` for hermetic
  tests.
- **Scheduler** (`swarmlab.scheduler`): executes a `DAGSpec` in topological
  order with cache-first dispatch, exponential-backoff retries, fan-out via
  `NEXT_TARGETS` directives, and tool routing via `TOOL_CALL` directives.
- **Deterministic seed wiring** (`swarmlab.seed`): BLAKE2b-derived seeds
  ensure same `(run_id, step_index)` always maps to the same sampling seed.
- **Tool registry** (`swarmlab.tool`): `FunctionTool` wraps any Python
  callable; `register_tool` / `get_tool` provide a module-level registry.

### Changed

- `__version__` bumped from `0.0.1` to `0.1.0`.

---

## [0.0.1] — 2026-05-19

### Added

- Project scaffold: `uv`-based Python 3.13 package with `ruff`, `pyright`,
  `pytest`, and `pre-commit` wired up.

[0.1.0]: https://github.com/dmzhq/swarmlab/releases/tag/v0.1.0
[0.0.1]: https://github.com/dmzhq/swarmlab/releases/tag/v0.0.1
