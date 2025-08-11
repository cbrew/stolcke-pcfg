# API Reference

High-level imports are available from `stolcke_pcfg`.

```python
from stolcke_pcfg import PCFG, Rule, StolckeParser, ConstrainedDecoderAdapter
```

## grammar.Rule
- Fields: `lhs: str`, `rhs: tuple[str, ...]`, `logp: float` (log probability).
- String form: `"A -> a b (0.123456)"` with linear-space probability for readability.

## grammar.PCFG
- `PCFG(rules: Iterable[tuple[str, Iterable[str], float]])`
  - Builds and normalizes rules per LHS; `float` probabilities must be `> 0`.
- `rules_for(lhs: str) -> list[Rule]`
- `nonterminals -> frozenset[str]`
- `is_terminal(sym: str) -> bool`

## stolcke_parser.StolckeParser
- `StolckeParser(grammar: PCFG, start_symbol: str)`
  - Validates unsupported productions; creates augmented start `S' -> S`.
- `reset() -> None`
- `allowed_terminals() -> set[str]`
- `step(terminal: str) -> bool`
  - Advances one token if allowed; updates prefix probability.
- `prefix_logprob() -> float`
- `accepted() -> bool`

## constrained_adapter.ConstrainedDecoderAdapter
- `ConstrainedDecoderAdapter(parser, token_id_to_str, str_to_token_id, next_token_filter=None)`
  - `allowed_token_ids() -> set[int]`: current mask of allowed next token IDs.
  - `step_with_token(token_id: int) -> bool`: advance via token ID.
  - Optional `next_token_filter(terms: set[str]) -> set[int]` can pre-filter allowed IDs.

## Notes
- All probabilities are accumulated in log-space. Use `math.exp(logp)` to inspect linear values.
- The internal `alpha`/`gamma` arrays live in `probabilities.ProbChart` and are not part of the public API surface.

