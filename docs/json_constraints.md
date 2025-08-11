# JSON Constraints Example

This example shows how to constrain generation to a simple JSON schema using a PCFG.

Schema (informal)
- Object with exactly these keys: `"name"`, `"age"`, `"tags"`.
- `name`: one of `"Alice"`, `"Bob"`.
- `age`: one of `0`, `1`, `2`.
- `tags`: array of one or more items, each `"x"` or `"y"`.

Grammar (terminals are literal tokens like `{`, `}`, `:`, `,`, and quoted strings)
- `S -> Obj`
- `Obj -> { Members }`
- `Members -> Pair | Pair , Members` (comma-separated pairs)
- `Pair -> "name" : Name | "age" : Age | "tags" : Tags`
- `Name -> "Alice" | "Bob"`
- `Age -> 0 | 1 | 2`
- `Tags -> [ TagList ]`
- `TagList -> NameTag | NameTag , TagList`
- `NameTag -> "x" | "y"`

Usage
- See `examples/json_constraints_demo.py` for a runnable demo.
- Integrate with the `ConstrainedDecoderAdapter` by mapping terminals to token IDs.
  - For subword tokenizers, add a `next_token_filter` to ensure only complete JSON tokens are allowed at boundaries.

Notes
- This is a minimal schema to demonstrate the approach. You can extend it to more keys,
  nested objects, or richer value sets by adding productions and terminals.
- The parser enforces structure; probabilities can be tuned to favor values or structures.

