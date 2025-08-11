# Constrained Decoding

This guide outlines how to use PCFG grammars to constrain token generation.

Core idea
- At each decoding step, compute the set of allowed terminals from the parser.
- Map allowed terminals to token IDs (respecting tokenizer boundaries).
- Mask logits for all disallowed token IDs before sampling/selecting the next token.
- After choosing a token, advance the parser via `step_with_token`.

Adapter
- `ConstrainedDecoderAdapter(parser, id2s, s2id, next_token_filter=None)` bridges between string terminals and token IDs.
- Key methods:
  - `allowed_token_ids() -> set[int]`
  - `allowed_token_mask(vocab_size: int) -> list[bool]` (True for allowed IDs)
  - `step_with_token(token_id: int) -> bool` (advance the parser)

Tokenizer alignment
- Terminals must correspond to complete tokens. For BPE/SentencePiece tokenizers, a literal like `"name"` may be multi-token.
- Use `next_token_filter` to restrict allowed IDs to those mapping from single tokens, or implement custom logic to compose multi-token terminals.

Prototype with Transformers
- See `examples/hf_constrained_decoding.py` for a greedy loop that masks logits using the adapter.
- It demonstrates constraining to a small JSON schema (see `docs/json_constraints.md`).

Performance tips
- Keep grammars epsilon-free and eliminate unit productions (done by default).
- For large vocabularies, materialize the boolean mask and reuse its buffer when possible.

