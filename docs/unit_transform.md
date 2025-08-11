# Unit-Production Elimination

This project eliminates unit productions (rules of the form `A -> B`) before parsing.
The transform ensures the Earley closure does not chase unit cycles at parse time,
and it preserves sentence probabilities.

## Method (NumPy Solve)

- Let `U[A,B] = P(A -> B)` be the total probability mass of unit rules from `A` to `B`.
- If the spectral radius of `U` is < 1, the closure is the Neumann series
  `C = I + U + U^2 + ... = (I - U)^{-1}`.
- We compute `C` via `np.linalg.solve(I - U, I)` for stability (no explicit inverse).
- For each non-unit rule `B -> γ` with probability `p`, lift it to `A -> γ` with
  probability `C[A,B] * p`. Drop all unit rules. Finally, re-normalize per LHS.

This sums all possible unit chains `A ⇒* B` as a geometric series.

## Convergence and Safety

- The implementation checks the spectral radius of `U` and solves `(I - U) C = I`.
  If the radius is ≥ 1 (within tolerance), it raises an error. Grammars whose unit
  mass per LHS is near or above 1 will not converge.

## Parser Option

- By default, `StolckeParser` applies unit elimination automatically.
- You can disable it (not recommended) via:
  - Python: `StolckeParser(G, 'S', eliminate_units=False)`
  - CLI: `stolcke-parser --no-unit-elim ...`

Note: Epsilon (empty) productions are not supported; provide grammars without ε.
