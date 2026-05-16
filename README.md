# swarmlab

> Deterministic, replayable multi-agent runs across any LLM provider.

[![CI](https://github.com/dmzhq/swarmlab/actions/workflows/ci.yml/badge.svg)](https://github.com/dmzhq/swarmlab/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)

> [!NOTE]
> Early development. The 0.1 release will ship the deterministic replay engine
> and provider parity for OpenAI, Anthropic, and locally-served models via
> vLLM and Ollama. Star the repo to follow progress.

---

## Why

Multi-agent debugging in 2026 is mostly "run it again and hope it does the
same thing." Vendor tracing tools show you *what* happened — they do not let
you re-run a branch of a past trace without spending money again.

`swarmlab` makes multi-agent runs **deterministic by construction** (seeded
sampling, recorded tool calls, content-addressed cache) and **fully
replayable** from a single run-id. It compiles a YAML or Python DAG of agents
and tools into an execution plan, captures every tool call and LLM response
into a content-addressed store, and lets you replay any branch of a past run
with **zero LLM cost**.

It is **provider-agnostic** — OpenAI, Anthropic, Mistral, and vLLM-served
local models — and ships with a built-in eval harness so you can measure
quality drift across providers on the same trace.

## Status

| Component                          | Status        |
|------------------------------------|---------------|
| Toolchain bootstrap (uv, ruff, pyright, pytest) | ✅ shipped |
| DAG schema + loaders               | 🚧 in progress |
| Content-addressed store            | 🚧 in progress |
| Provider adapter + deterministic seeds | planned   |
| Scheduler with retries + fan-out   | planned       |
| Replay engine                      | planned       |
| Eval harness                       | planned       |
| OpenTelemetry export               | planned       |

## Install

```bash
# coming soon (will publish to PyPI with 0.1)
pip install swarmlab
```

## Quickstart

```bash
# planned for 0.1 — placeholder
swarmlab run examples/01_minimal_two_agents.yaml
swarmlab replay <run-id>
```

## Design principles

1. **Determinism over speed.** Same input + same seed = byte-identical output.
2. **Provider portability.** Swap OpenAI for Anthropic for a local Qwen run
   without rewriting your agent code.
3. **Content-addressed storage.** Every tool call and LLM response is keyed
   by the hash of its inputs. Replay reads from cache. Real LLM calls happen
   only on cache miss.
4. **Audit-ready by default.** Every run produces a single, complete trace
   you can hand to a reviewer, a compliance officer, or a future you.

## Comparison

| | `swarmlab` | LangGraph | CrewAI | Inngest / Temporal |
|---|---|---|---|---|
| Provider-agnostic | ✅ | ⚠️ (LangChain-coupled) | ✅ | n/a |
| Deterministic by construction | ✅ | ❌ | ❌ | ⚠️ (durable, not deterministic) |
| Replay branch with zero LLM cost | ✅ | ❌ | ❌ | ❌ |
| Built-in eval harness | ✅ | ❌ | ❌ | ❌ |
| Generic durable execution | partial | partial | ❌ | ✅ |

## Security

See [SECURITY.md](SECURITY.md) for responsible disclosure.

## License

Apache-2.0 — see [LICENSE](LICENSE).
