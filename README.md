# Stolcke PCFG Parser

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://cbrew.github.io/stolcke-pcfg/)

A compact Python implementation of the probabilistic Earley parser (Stolcke 1994/1995) with prefix probabilities and an adapter for constrained LLM decoding.

Note: Portions of this codebase and documentation were authored with AI assistance.
Please review changes with care and validate behavior in your environment before
production use.

## Features
- Normalized PCFG with log-space probabilities.
- Earley parser tracking forward (alpha) and inner (gamma) scores.
- Prefix probability for incremental decoding.
- Exact sentence log-probability via a span-based inside DP.
- Simple demo CLI and adapter to map allowed terminals to token IDs.

## Install & Run
- Create env and install: `make setup` (or `uv venv && uv sync`).
- Run demo CLI: `make run` or `uv run stolcke-parser a a a`.
  - Optionally disable unit elimination: `uv run stolcke-parser --no-unit-elim a a a`.
- Lint/format: `make lint` / `make fmt`. Tests: `make test`. Coverage: `make coverage`.

## Library Quickstart
```python
from stolcke_pcfg import PCFG, StolckeParser
G = PCFG([
    ("S", ["S", "a"], 0.4),
    ("S", ["a"], 0.6),
])
P = StolckeParser(G, "S")
P.step("a"); print(P.prefix_logprob(), P.accepted())
```

## Documentation
- Start here: docs/index.md
- Architecture: docs/architecture.md
- API: docs/api.md
- CLI: docs/cli.md
- Unit transform: docs/unit_transform.md
- Examples: docs/examples.md

Serve docs locally: `make docs-serve` (or `uv run mkdocs serve`). Build: `make docs-build`. Hosted docs: https://cbrew.github.io/stolcke-pcfg/

## Development
- Style via Ruff (line length 100). First-party: `stolcke_pcfg`.
- Conventional Commits, focused PRs. See docs/development.md.

## Notes
- Unit productions supported via matrix-based closure; epsilon productions are not supported.
- Left recursion supported; coordination (e.g., NP conjunction) yields correct Catalan counts.
