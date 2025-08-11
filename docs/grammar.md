# Grammar Model

The `PCFG` class represents a probabilistic context-free grammar with per-LHS normalization.

## Rules

Create a grammar from triplets `(lhs, rhs, p)` where:
- `lhs: str`
- `rhs: Iterable[str]` (use empty for epsilon only when supported; current parser forbids epsilon/unit)
- `p: float`, `p > 0`

Example:

```python
G = PCFG([
    ("S", ["NP", "VP"], 0.9),
    ("S", ["VP"], 0.1),
    ("NP", ["Det", "N"], 1.0),
    ("VP", ["V", "NP"], 0.5),
    ("VP", ["V"], 0.5),
    ("Det", ["the"], 0.6),
    ("Det", ["a"], 0.4),
    ("N", ["cat"], 0.7),
    ("N", ["mat"], 0.3),
    ("V", ["sleeps"], 0.5),
    ("V", ["likes"], 0.5),
])
```

Probabilities are normalized per LHS during construction.

## Terminals vs Nonterminals

- Nonterminals are exactly the symbols that appear as an LHS.
- Terminals are symbols that never appear as an LHS.
- Current parser limitations: no epsilon productions and no unit productions (e.g., `A -> B`). Left recursion is supported.

