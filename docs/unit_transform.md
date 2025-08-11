# Unit-Production Elimination

This project eliminates unit productions (rules of the form `A -> B`) before parsing.
The transform ensures the Earley closure does not chase unit cycles at parse time,
and it preserves sentence probabilities.

## Method (Matrix Series)

- Let `U[A,B] = P(A -> B)` be the total probability mass of unit rules from `A` to `B`.
- If the spectral radius of `U` is < 1, the Neumann series converges:
  `C = I + U + U^2 + ... = (I - U)^{-1}`.
- For each non-unit rule `B -> γ` with probability `p`, lift it to `A -> γ` with
  probability `C[A,B] * p`. Drop all unit rules. Finally, re-normalize per LHS.

This sums all possible unit chains `A ⇒* B` as a geometric series.

## Convergence and Safety

- The implementation sums `I + U + U^2 + ...` iteratively until changes are below a
  tolerance, or a maximum number of powers is reached. If convergence is not achieved,
  it raises an error. Grammars whose unit mass per LHS is near or above 1 will not converge.

## Parser Option

- By default, `StolckeParser` applies unit elimination automatically.
- You can disable it (not recommended) via:
  - Python: `StolckeParser(G, 'S', eliminate_units=False)`
  - CLI: `stolcke-parser --no-unit-elim ...`

Note: Epsilon (empty) productions are not supported; provide grammars without ε.

