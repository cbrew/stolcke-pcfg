# Stolcke PCFG Parser

A compact implementation of the probabilistic Earley parser (Stolcke 1994/1995) with prefix probabilities, written in Python. It provides:

- A normalized PCFG model with log-space probabilities.
- An Earley-style parser that tracks forward (alpha) and inner (gamma) scores.
- Prefix probability for incremental decoding, plus an adapter for constrained LLM generation.
- A small demo CLI and a simple smoke test.

## Installation

- Local dev (preferred): `make setup && make sync`
- Direct with uv: `uv venv && uv sync`

Requires Python 3.12+.

## Quickstart (Library)

```python
from stolcke_pcfg import PCFG, StolckeParser

G = PCFG([
    ("S", ["S", "a"], 0.4),
    ("S", ["a"], 0.6),
])
P = StolckeParser(G, "S")

print(P.allowed_terminals())   # {"a"}
P.step("a")                    # True
print(P.prefix_logprob())      # log P(prefix)
print(P.accepted())            # True after ≥1 token
```

## Quickstart (CLI)

- Run default demo: `stolcke-parser`
- Provide tokens: `stolcke-parser a a a`

See docs/cli.md for details.

## Key Concepts

- PCFG: Rules `lhs -> rhs` with probabilities normalized per `lhs`.
- Earley items: dotted rules plus start index; scan/predict/complete.
- Prefix probability: sum of alpha over scanned states at each prefix.

For design and API details, see docs/architecture.md and docs/api.md.

