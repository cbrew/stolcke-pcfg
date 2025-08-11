# Examples

This repository ships with a simple prefix-probability demo.

## demo_prefix_prob.py

Path: `examples/demo_prefix_prob.py`

Demonstrates a left-recursive grammar `S -> S 'a' | 'a'` and prints:
- Allowed terminals at the start.
- For each consumed token: whether the parser advanced and the current prefix log-probability.
- Whether the current prefix is accepted.

Run it with:

```
uv run python examples/demo_prefix_prob.py
```

or try the CLI:

```
uv run stolcke-parser a a a
```

You can adapt the example to different grammars by constructing `PCFG([...])` and `StolckeParser(G, "S")` manually.

