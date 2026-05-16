# Contributing to swarmlab

Thanks for your interest. This document covers the basics.

## Dev setup

Prerequisites:

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency and venv management

```bash
git clone https://github.com/dmzhq/swarmlab.git
cd swarmlab
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run pyright
```

## Style

- `ruff` for linting and formatting (configuration in `pyproject.toml`).
- `pyright` strict mode for type checking.
- `pytest` with `-ra` so warnings are surfaced and failures are detailed.
- Conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`, `chore:`, `perf:`, `ci:`, `build:`.

## Pull requests

1. Open an issue first for non-trivial changes so we can align on direction.
2. Keep PRs focused — one concern per PR.
3. Include tests for behavior changes.
4. Run the full local checklist before pushing:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run pytest
   ```

5. Update `README.md` or relevant docs if behavior changes.

## Code of conduct

By participating, you agree to abide by the project's code of conduct.
Be respectful, be specific, and assume good faith.
