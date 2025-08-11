# Changelog

All notable changes to this project are documented here. Dates use YYYY-MM-DD.

## v0.2.3 — Constrained decoding prototype (2025-08-11)
- feat(adapter): add `allowed_token_mask(vocab_size)` for fast logits masking.
- docs: new guides — `docs/constrained_decoding.md`, `docs/json_constraints.md`.
- examples: `examples/hf_constrained_decoding.py` demo with `distilgpt2` (optional deps).
- note: This code and documentation were produced with AI assistance. Please review
  carefully and validate in your environment before production use.

## v0.2.2 — Normalization + JSON example (2025-08-11)
- tests: ensure per-LHS normalization is preserved after unit elimination.
- docs/examples: add JSON constraints example and acceptance test.

## v0.2.1 — NumPy unit-closure + tests (2025-08-11)
- transform: switch unit-production closure to NumPy `solve((I-U), I)` with spectral check.
- deps: add `numpy` as a runtime dependency.
- tests: equivalence, chain, branching, divergence, mixed unit/non-unit.

## v0.2.0 — Solid PCFG core (2025-08-11)
- parser: unit-production elimination (pre-parse); epsilon productions rejected.
- algorithm: Stolcke-style event-driven α/γ; exact `sentence_logprob()` via span-based inside DP.
- tests: Catalan counts for coordination; NL prefix probability monotonicity.
