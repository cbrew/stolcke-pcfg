# Repository Guidelines

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://cbrew.github.io/stolcke-pcfg/)

This guide explains how to structure, build, test, and contribute to this repository. Keep changes small and focused; prefer automation via Make targets and scripts.

## Project Structure & Module Organization
- `src/`: Production code organized by feature/domain (e.g., `src/parser/`, `src/cli/`).
- `tests/`: Mirrors `src/` paths (e.g., `src/foo/bar.py` → `tests/foo/test_bar.py`). Fixtures in `tests/fixtures/`.
- `scripts/`: Dev utilities (setup, lint, release). Use portable shell/Node/Python.
- `docs/`: Design notes and user-facing docs.
- `assets/`: Static assets; keep test data under `tests/fixtures/`.

## Build, Test, and Development Commands
Prefer Make targets:
- `make setup`: Create `.venv` with `uv`, install deps, install `pre-commit` hooks.
- `make run`: Run the CLI entrypoint (`stolcke-parser`).
- `make test`: Run tests via `pytest`.
- `make lint` / `make fmt`: Lint and format with Ruff.
- `make precommit`: Run all hooks locally.
- `make coverage`: Generate a coverage report.
Direct equivalents: `uv venv`, `uv sync`, `uv run ruff check .`, `uv run pytest`.

## Coding Style & Naming Conventions
- Formatter/linter: Ruff controls checking and formatting; do not hand-format.
- Python: 4-space indentation, line length 100.
- Naming: packages/modules `lower_snake_case`; classes `PascalCase`; functions/vars `lower_snake_case`.
- Imports: group/sort; `stolcke_parser` is first-party.

## Testing Guidelines
- Framework: `pytest` with deterministic, isolated tests.
- Naming: `test_*.py`; mirror `src/` paths for structure and clarity.
- Fixtures: place under `tests/fixtures/`.
- Run: `make test`; for coverage use `make coverage`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(parser): add tokenizer`), reference issues (`Closes #123`).
- PRs: include summary, rationale, testing steps/output; keep focused and incremental.

## Security & Configuration Tips
- Do not commit secrets. Use `.env`; maintain `.env.example` with required variables.
- Configure via environment variables; document defaults/behavior in `docs/`.
