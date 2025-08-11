# Architecture

This package implements a probabilistic Earley parser with Stolcke-style forward (alpha) and inner (gamma) probabilities. Components are small and focused.

## Modules

- `grammar.py`
  - `Rule`: immutable production with `lhs`, `rhs: tuple[str, ...]`, and `logp`.
  - `PCFG`: normalizes rule probabilities per LHS; identifies terminals vs nonterminals.
- `earley_core.py`
  - `EarleyItem`: `(rule, dot, start)`; supports `next_symbol()`, `advance()`, `finished()`.
  - `BP`: minimal backpointer info for provenance.
  - `EarleyChart`: per-position maps of items → best Viterbi score and backpointers.
- `probabilities.py`
  - `ProbChart`: extends chart with `alpha` (forward) and `gamma` (inner) maps; log-sum accumulation.
- `stolcke_parser.py`
  - `StolckeParser`: orchestrates SCAN, PREDICT, COMPLETE, keeps prefix probability, exposes `allowed_terminals()`.
- `constrained_adapter.py`
  - `ConstrainedDecoderAdapter`: maps allowed terminals to token IDs for LLM decoding.

## Algorithm Highlights

- Items: dotted `Rule` with origin `start` index. State sets exist at each input position `k`.
- Predictor: when item expects a nonterminal `Y`, add `Y -> • γ` at position `k`; update alpha and initialize gamma with rule prob.
- Scanner: when item expects terminal `a`, consuming `a` at `k` yields `•` advanced item at `k+1`; propagate alpha/gamma.
- Completer: when an item finishes at `k`, combine with parents waiting for its LHS at their origin `i`; propagate alpha/gamma.
- Prefix probability: at each scan, sum alphas of scanned states at the new position.

## Constraints and Assumptions

- Epsilon and unit productions are supported; left recursion allowed.
- Probabilities are in log-space; `LOG_ZERO` sentinel avoids `-inf` arithmetic.
- Terminals are symbols that never appear on any LHS.

## Complexity

- Worst-case cubic in input length for general CFGs (Earley), with grammar-dependent constants. Alpha/gamma bookkeeping adds overhead but remains linear in number of state updates.
