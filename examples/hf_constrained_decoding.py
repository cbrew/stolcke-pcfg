"""
Prototype: Grammar-constrained decoding with Hugging Face Transformers.

This demo shows a simple greedy decoding loop that masks logits according to
the allowed terminals from the grammar at each step. It uses the JSON schema
example grammar from docs/json_constraints.md.

Requirements: transformers, torch (not installed by default in this repo).
Run at your own environment if available.
"""
from __future__ import annotations

from typing import Any  # noqa: F401

from stolcke_pcfg import PCFG, ConstrainedDecoderAdapter, StolckeParser

try:  # pragma: no cover - optional example
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise SystemExit("This example requires transformers + torch") from e


def build_json_grammar() -> PCFG:
    return PCFG(
        [
            ("S", ["Obj"], 1.0),
            ("Obj", ["{", "Members", "}"], 1.0),
            ("Members", ["Pair"], 0.5),
            ("Members", ["Pair", ",", "Members"], 0.5),
            ("Pair", ['"name"', ":", "Name"], 1 / 3),
            ("Pair", ['"age"', ":", "Age"], 1 / 3),
            ("Pair", ['"tags"', ":", "Tags"], 1 / 3),
            ("Name", ['"Alice"'], 0.5),
            ("Name", ['"Bob"'], 0.5),
            ("Age", ["0"], 1 / 3),
            ("Age", ["1"], 1 / 3),
            ("Age", ["2"], 1 / 3),
            ("Tags", ["[", "TagList", "]"], 1.0),
            ("TagList", ["NameTag"], 0.5),
            ("TagList", ["NameTag", ",", "TagList"], 0.5),
            ("NameTag", ['"x"'], 0.5),
            ("NameTag", ['"y"'], 0.5),
        ]
    )


def single_token_filter_factory(tokenizer) -> callable:
    def to_ids(terms: set[str]) -> set[int]:
        out: set[int] = set()
        for t in terms:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
        return out

    return to_ids


def greedy_constrained(model_name: str = "gpt2", max_new_tokens: int = 64) -> None:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Build grammar + parser
    G = build_json_grammar()
    P = StolckeParser(G, "S")

    # Bridges
    def id2s(idx: int) -> str:
        return tok.decode([idx], clean_up_tokenization_spaces=False)

    def s2id(s: str) -> int | None:
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    adapter = ConstrainedDecoderAdapter(
        P,
        id2s,
        s2id,
        next_token_filter=single_token_filter_factory(tok),
    )

    # Start with BOS token for proper model initialization
    # Then we'll mask the next token predictions based on grammar
    input_ids = torch.tensor([[tok.bos_token_id]], dtype=torch.long)
    generated = input_ids.clone()
    print(f"Starting generation with BOS token {tok.bos_token_id}")

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=generated)
            logits = out.logits[:, -1, :]

        # Mask disallowed tokens
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        disallowed = ~torch.tensor(mask, dtype=torch.bool, device=logits.device)
        logits[0, disallowed] = -1e30

        # Greedy pick
        next_id = int(torch.argmax(logits, dim=-1).item())

        progressed = adapter.step_with_token(next_id)
        if not progressed:
            print("Token not permitted by grammar; stopping.")
            break
        generated = torch.cat([generated, torch.tensor([[next_id]], device=generated.device)], dim=-1)

        token_str = tok.decode([next_id], clean_up_tokenization_spaces=False)
        print(f"@{step} next={next_id} {token_str!r}; allowed={len(adapter.allowed_token_ids())}")

        if P.accepted():
            print("Accepted by grammar; stopping.")
            break

    print("Generated:", tok.decode(generated[0], clean_up_tokenization_spaces=False))
    if P.accepted():
        print("sentence_logprob:", P.sentence_logprob())


if __name__ == "__main__":  # pragma: no cover
    greedy_constrained()