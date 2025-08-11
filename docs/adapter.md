# Constrained Decoding Adapter

The `ConstrainedDecoderAdapter` bridges the parser and a tokenizer-based decoder by translating allowed terminals into token IDs.

## Usage

```python
from stolcke_pcfg import ConstrainedDecoderAdapter, PCFG, StolckeParser

# Build grammar and parser
G = PCFG([...])
P = StolckeParser(G, "S")

# Tokenizer bridges
id2s = lambda i: your_tokenizer.decode([i])
s2id = lambda s: your_tokenizer.encode(s)[0] if s in your_vocab else None

adapter = ConstrainedDecoderAdapter(P, id2s, s2id)
mask = adapter.allowed_token_ids()  # set[int]
```

- Call `adapter.allowed_token_ids()` at each decoding step to mask logits.
- After sampling/choosing a token ID, call `adapter.step_with_token(token_id)` to advance the parser.
- You may provide `next_token_filter(terms: set[str]) -> set[int]` to customize which tokens are allowed (e.g., subword boundary handling).

## Notes

- Terminals must correspond to full tokens in your vocabulary; otherwise, insert a filter/mapper that enforces only vocabulary-aligned terminals or composes multi-token terminals.
- Parser provides `allowed_terminals()` directly if you do not need IDs.

