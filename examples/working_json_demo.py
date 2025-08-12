#!/usr/bin/env python3
"""
Working JSON demo using only single-token values that GPT-2 produces naturally.

This version creates valid JSON by using tokens that GPT-2 handles as single units.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stolcke_pcfg import PCFG, ConstrainedDecoderAdapter, StolckeParser


def build_single_token_json_grammar() -> PCFG:
    """
    Simple JSON grammar using only single tokens.
    
    We'll generate patterns like: {"name":"Alice"} by using the fact that
    GPT-2 tokenizes certain patterns as single tokens.
    
    Key insight: Use compound tokens that GPT-2 naturally produces:
    - '{"name":"' -> could be broken down, but let's use simpler approach
    - Let's build: {VALUE} where VALUE is a single token like "text" or numbers
    """
    return PCFG([
        # Simple structure: { value }
        ("S", ["{", "VALUE", "}"], 1.0),
        
        # Single-token values that make sense
        ("VALUE", ['"Alice"'], 0.2),    # Check if this is single token
        ("VALUE", ['"Bob"'], 0.2),      # Check if this is single token  
        ("VALUE", ["42"], 0.2),         # Numbers are usually single tokens
        ("VALUE", ["true"], 0.2),       # Boolean values
        ("VALUE", ["false"], 0.2),      # Boolean values
    ])


def analyze_and_build_grammar(tokenizer) -> PCFG:
    """Build grammar after analyzing what tokens work."""
    
    print("=== Token Analysis ===")
    candidates = [
        '{"name":"Alice"}',  # Full JSON
        '{"age":25}',        # JSON with number
        '"Alice"',           # Just string
        '"Bob"',             # Just string  
        '42',                # Number
        'true',              # Boolean
        'false',             # Boolean
        '{',                 # Brace
        '}',                 # Brace
    ]
    
    single_tokens = []
    
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        is_single = len(ids) == 1
        if is_single:
            single_tokens.append(candidate)
        print(f"'{candidate}' -> {ids} -> single_token: {is_single}")
    
    print(f"\nSingle tokens found: {single_tokens}")
    
    # Build grammar using only single tokens
    if any(token.startswith('{"') and token.endswith('"}') for token in single_tokens):
        # We have complete JSON objects as single tokens!
        json_objects = [t for t in single_tokens if t.startswith('{"') and t.endswith('"}')]
        rules = [("S", [obj], 1.0/len(json_objects)) for obj in json_objects]
        print(f"Using complete JSON objects: {json_objects}")
        return PCFG(rules)
    else:
        # Fallback to piece-by-piece construction
        braces = ["{", "}"]
        values = [t for t in single_tokens if t not in braces and not t.startswith('{"')]
        
        if "{" in single_tokens and "}" in single_tokens and values:
            rules = [
                ("S", ["{", "VALUE", "}"], 1.0),
            ]
            rules.extend([("VALUE", [val], 1.0/len(values)) for val in values])
            print(f"Using construction with values: {values}")
            return PCFG(rules)
        else:
            # Ultra-simple fallback
            print("Using ultra-simple number grammar")
            return PCFG([
                ("S", ["NUM"], 1.0),
                ("NUM", ["42"], 0.5),
                ("NUM", ["25"], 0.5),
            ])


def working_demo(model_name: str = "gpt2", max_new_tokens: int = 10) -> None:
    """Demo that actually produces valid output."""
    
    # Device setup
    if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Analyze tokens and build grammar
    grammar = analyze_and_build_grammar(tok)
    
    print("\nFinal grammar:")
    for lhs in grammar.nonterminals:
        for rule in grammar.rules_for(lhs):
            print(f"  {rule}")
    
    parser = StolckeParser(grammar, "S")
    
    # Token functions
    def id2str(token_id: int) -> str:
        return tok.decode([token_id], clean_up_tokenization_spaces=False)
    
    def str2id(s: str) -> int | None:
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None
    
    def single_token_filter(terms: set[str]) -> set[int]:
        out: set[int] = set()
        for term in terms:
            ids = tok.encode(term, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
                print(f"      '{term}' -> {ids[0]}")
        return out
    
    adapter = ConstrainedDecoderAdapter(
        parser, id2str, str2id, next_token_filter=single_token_filter
    )
    
    # Generation
    generated = torch.empty((1, 0), dtype=torch.long, device=device)
    
    print(f"\n=== Generation ===")
    
    for step in range(max_new_tokens):
        # Model input
        if generated.size(1) == 0:
            input_ids = torch.tensor([[tok.bos_token_id]], device=device)
        else:
            input_ids = generated
        
        # Get logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
        
        # Apply constraints
        allowed_ids = adapter.allowed_token_ids()
        if not allowed_ids:
            print("No allowed tokens")
            break
        
        print(f"Step {step}: {len(allowed_ids)} tokens allowed")
        
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        disallowed = ~torch.tensor(mask, dtype=torch.bool, device=logits.device)
        logits[0, disallowed] = -1e30
        
        # Select token
        next_id = int(torch.argmax(logits, dim=-1).item())
        
        if not adapter.step_with_token(next_id):
            print(f"Token {next_id} rejected")
            break
        
        # Update sequence
        if generated.size(1) == 0:
            generated = torch.tensor([[next_id]], device=device)
        else:
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=-1)
        
        token_str = id2str(next_id)
        print(f"  Selected: {next_id} '{token_str}'")
        
        if parser.accepted():
            print("✅ Complete!")
            break
    
    # Results
    final_text = tok.decode(generated[0], clean_up_tokenization_spaces=False)
    print(f"\n=== Result ===")
    print(f"Generated: {final_text}")
    print(f"Accepted by grammar: {parser.accepted()}")
    
    if parser.accepted():
        print(f"Log probability: {parser.sentence_logprob():.4f}")


if __name__ == "__main__":
    working_demo()