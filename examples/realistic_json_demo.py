#!/usr/bin/env python3
"""
Realistic JSON generation demo with proper key-value pairs.

This version generates valid JSON with actual key-value structure by using
GPT-2's natural multi-token patterns and proper grammar rules.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stolcke_pcfg import PCFG, ConstrainedDecoderAdapter, StolckeParser


def build_realistic_json_grammar() -> PCFG:
    """
    Grammar for realistic JSON using GPT-2's natural tokenization patterns.
    
    Key insight: Use the actual tokens GPT-2 produces for JSON patterns:
    - '{"name":' -> [4895, 3672, 1298] but we can use separate rules for each part
    - ' "Alice"' -> [366, 44484, 1] but ' "Alice' -> [366, 44484] (without closing quote)
    """
    return PCFG([
        # Main structure: {"key": "value"}
        ("S", ["OBJ_START", "KEY", "SEP", "VALUE", "OBJ_END"], 1.0),
        
        # JSON structural tokens (single tokens in GPT-2)
        ("OBJ_START", ['{"'], 1.0),      # 4895
        ("OBJ_END", ['"}'], 1.0),        # 20662 
        ("SEP", ['":'], 1.0),            # 1298 ":"
        
        # Keys (without quotes - they're part of structure tokens)
        ("KEY", ['name'], 0.33),         # 3672
        ("KEY", ['age'], 0.33),          # 496  
        ("KEY", ['city'], 0.34),         # 1748
        
        # Values with leading space (GPT-2 pattern)
        ("VALUE", [' "Alice'], 0.25),    # 366 + 44484
        ("VALUE", [' "Bob'], 0.25),      # 366 + 18861
        ("VALUE", [' "NYC'], 0.25),      # 366 + 17947 
        ("VALUE", [' 25'], 0.25),        # 1679
    ])


def multi_token_aware_filter(tokenizer) -> callable:
    """
    Filter that properly handles multi-token terminals.
    
    For multi-token terminals, we need to allow the sequence to start,
    so we allow the first token but track partial matches in the parser.
    """
    def to_ids(terms: set[str]) -> set[int]:
        out: set[int] = set()
        print(f"    Grammar expects terms: {sorted(terms)}")
        
        for term in terms:
            ids = tokenizer.encode(term, add_special_tokens=False)
            if len(ids) >= 1:
                out.add(ids[0])  # Always allow the first token
                token_repr = tokenizer.decode([ids[0]], clean_up_tokenization_spaces=False)
                print(f"      '{term}' -> {ids} -> allowing first token {ids[0]} '{token_repr}'")
        
        return out
    
    return to_ids


def realistic_json_demo(model_name: str = "gpt2", max_new_tokens: int = 20) -> None:
    """Generate realistic JSON with key-value pairs."""
    
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
    
    # Build grammar and parser
    grammar = build_realistic_json_grammar()
    print("Grammar rules:")
    for lhs in grammar.nonterminals:
        for rule in grammar.rules_for(lhs):
            print(f"  {rule}")
    
    parser = StolckeParser(grammar, "S")
    
    # Token conversion functions  
    def id2str(token_id: int) -> str:
        return tok.decode([token_id], clean_up_tokenization_spaces=False)
    
    def str2id(s: str) -> int | None:
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) >= 1 else None
    
    # Create adapter with multi-token aware filtering
    adapter = ConstrainedDecoderAdapter(
        parser, id2str, str2id,
        next_token_filter=multi_token_aware_filter(tok),
    )
    
    # Generation
    generated = torch.empty((1, 0), dtype=torch.long, device=device)
    
    print(f"\n=== Realistic JSON Generation ===")
    print(f"Target: {{\"key\": \"value\"}} structure")
    print(f"Expected patterns: {{\"name\": \"Alice\"}} or {{\"age\": 25}}\n")
    
    for step in range(max_new_tokens):
        # Prepare model input
        if generated.size(1) == 0:
            # Start with BOS for proper model state
            input_ids = torch.tensor([[tok.bos_token_id]], device=device)
        else:
            input_ids = generated
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
        
        # Get allowed tokens from grammar
        allowed_ids = adapter.allowed_token_ids()
        if not allowed_ids:
            print("❌ No tokens allowed by grammar")
            break
        
        print(f"Step {step:2d}: {len(allowed_ids)} allowed tokens")
        
        # Apply grammar mask
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        disallowed = ~torch.tensor(mask, dtype=torch.bool, device=logits.device)
        logits[0, disallowed] = -1e30
        
        # Select best allowed token
        next_id = int(torch.argmax(logits, dim=-1).item())
        
        # Advance parser
        if not adapter.step_with_token(next_id):
            print(f"❌ Parser rejected token {next_id} '{id2str(next_id)}'")
            break
        
        # Add to sequence
        if generated.size(1) == 0:
            generated = torch.tensor([[next_id]], device=device)
        else:
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=-1)
        
        # Show progress
        token_str = id2str(next_id)
        current_text = tok.decode(generated[0], clean_up_tokenization_spaces=False)
        print(f"         Selected {next_id:5d} '{token_str}' -> {current_text!r}")
        
        # Check completion
        if parser.accepted():
            print(f"🎉 JSON Complete!")
            break
        
    else:
        print("⚠️  Reached maximum tokens without completion")
    
    # Final results
    final_json = tok.decode(generated[0], clean_up_tokenization_spaces=False)
    
    print(f"\n=== Generated JSON ===")
    print(f"Token sequence: {generated[0].tolist()}")
    print(f"JSON text: {final_json}")
    print(f"Valid: {parser.accepted()}")
    
    if parser.accepted():
        log_prob = parser.sentence_logprob()
        print(f"Grammar log-probability: {log_prob:.4f}")
        print(f"Grammar probability: {torch.exp(torch.tensor(log_prob)):.6f}")
    
    # Verify it's valid JSON
    try:
        import json
        parsed = json.loads(final_json)
        print(f"✅ Valid JSON parsed as: {parsed}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")


if __name__ == "__main__":
    realistic_json_demo()