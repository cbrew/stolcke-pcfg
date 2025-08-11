# CLI

The package exposes a small demo CLI for experimentation.

## Command

- Entry point: `stolcke-parser`
- Arguments: a sequence of space-separated tokens (optional). If omitted, defaults to `a a a`.
- Flags: `--no-unit-elim` disables unit-production elimination (not recommended).

## Examples

```
stolcke-parser
stolcke-parser a a a
stolcke-parser a a b   # will stop when an unexpected token is encountered
```

The CLI reports:
- Allowed terminals at the start.
- For each token: whether it advanced, and the current prefix log-probability.
- Final acceptance status of the consumed prefix.

Note: The CLI uses a built-in left-recursive grammar `S -> S 'a' | 'a'`. For custom grammars and programmatic usage, use the library API.
