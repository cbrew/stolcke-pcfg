# Development Guide

## Environment

- Create venv and install deps: `make setup` (or `uv venv && uv sync`).
- Install hooks: `make hooks`.

## Common Tasks

- Lint: `make lint`
- Format: `make fmt`
- Test: `make test`
- Coverage: `make coverage`
- Run CLI: `make run` (demo grammar)

Direct equivalents: `uv run ruff check .`, `uv run ruff format .`, `uv run pytest`.

## Conventions

- Style: Managed by Ruff (checking + formatting). Line length 100.
- Naming: modules `lower_snake_case`, classes `PascalCase`, functions/vars `lower_snake_case`.
- Imports: grouped/sorted; `stolcke_pcfg` is first-party.
- Commits: Conventional Commits (e.g., `feat(parser): add tokenizer`).

## Testing

- Framework: `pytest` with mirrors of `src/` structure in `tests/`.
- Add deterministic tests; prefer small, isolated fixtures under `tests/fixtures/`.
- For probability checks, compare in log-space; avoid fragile float equality.

## Notes

- The parser currently rejects epsilon and unit productions; plan tests accordingly.
- In restricted environments, running `make lint` via `uv` may fail; use the venv tools directly (e.g., `.venv/bin/ruff`).

