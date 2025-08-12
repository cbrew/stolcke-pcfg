#!/usr/bin/env python3
"""
Robust JSON generation demo using the new RobustConstrainedAdapter.

This demo shows how to generate valid JSON without being limited by
tokenization quirks. The RobustConstrainedAdapter handles multi-token
terminals seamlessly by tracking partial matches.

Key improvements:
1. Can use natural JSON strings like '"Alice"' even though GPT-2 tokenizes them as multiple tokens
2. Handles complex JSON structures with proper key-value pairs
3. Resistant to tokenizer variations across different models
"""
from __future__ import annotations

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stolcke_pcfg import PCFG, StolckeParser
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter


def build_natural_json_grammar() -> PCFG:
    """
    Build a JSON grammar using natural string literals.
    
    Unlike previous attempts, we can now use multi-token strings like '"Alice"'
    because the RobustConstrainedAdapter handles the tokenization properly.
    """
    return PCFG([
        # Main structure: {"key": "value"} 
        ("S", ["Object"], 1.0),
        
        # Object with one key-value pair
        ("Object", ["{", "Pair", "}"], 1.0),
        
        # Key-value pairs 
        ("Pair", ["Key", ":", "Value"], 1.0),
        
        # Keys (proper JSON strings)
        ("Key", ['"name"'], 0.4),
        ("Key", ['"age"'], 0.3),
        ("Key", ['"city"'], 0.3),
        
        # Values (mix of strings and numbers)
        ("Value", ['"Alice"'], 0.2),
        ("Value", ['"Bob"'], 0.2), 
        ("Value", ['"Charlie"'], 0.2),
        ("Value", ["25"], 0.1),
        ("Value", ["30"], 0.1),
        ("Value", ["35"], 0.1),
        ("Value", ["true"], 0.05),
        ("Value", ["false"], 0.05),
    ])


def demonstrate_tokenization_robustness(tokenizer):
    """Show how the robust adapter handles various tokenization patterns."""
    
    print("🔍 Tokenization Analysis")
    print("=" * 50)
    
    test_strings = [
        '"name"',     # Key
        '"Alice"',    # String value  
        '"Charlie"',  # Longer string
        '25',         # Number
        'true',       # Boolean
        '{',          # Structural
        '}',          # Structural
        ':',          # Structural
    ]
    
    for test_str in test_strings:
        token_ids = tokenizer.encode(test_str, add_special_tokens=False)
        tokens = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in token_ids]
        
        print(f"'{test_str}' -> {token_ids} -> {tokens}")
        if len(token_ids) > 1:
            print(f"  ⚠️  Multi-token: Would break naive approach")
        else:
            print(f"  ✅ Single-token: Works with any approach")
    
    print()


def robust_json_demo(model_name: str = "gpt2", max_new_tokens: int = 25) -> None:
    """Demonstrate robust JSON generation."""
    
    print("🚀 Robust JSON Generation Demo")
    print("🎯 Goal: Generate complete, valid JSON objects")
    print("💪 Using RobustConstrainedAdapter to handle multi-token terminals")
    print("=" * 70)
    
    # Device setup
    if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
        device = torch.device("mps")
        print(f"✅ Device: {device} (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print(f"✅ Device: {device}")
    
    # Load model and tokenizer
    print(f"\n📥 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Show tokenization challenges
    demonstrate_tokenization_robustness(tokenizer)
    
    # Build grammar
    grammar = build_natural_json_grammar()
    print("📝 JSON Grammar:")
    for nonterminal in grammar.nonterminals:
        for rule in grammar.rules_for(nonterminal):
            print(f"  {rule}")
    print()
    
    # Create parser and robust adapter
    parser = StolckeParser(grammar, "S")
    
    def id_to_string(token_id: int) -> str:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    
    def string_to_id(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None
    
    adapter = RobustConstrainedAdapter(
        parser=parser,
        token_id_to_str=id_to_string,
        str_to_token_id=string_to_id,
        tokenizer=tokenizer
    )
    
    # Generation
    generated_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
    
    print("🎲 Starting Generation")
    print("-" * 30)
    
    for step in range(max_new_tokens):
        # Prepare model input
        if generated_tokens.size(1) == 0:
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        else:
            input_ids = generated_tokens
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
        
        # Get constraints
        allowed_tokens = adapter.allowed_token_ids()
        state_info = adapter.get_current_state_info()
        
        print(f"Step {step:2d}: {len(allowed_tokens):2d} allowed tokens, "
              f"{state_info['partial_matches']} partial matches")
        
        if not allowed_tokens:
            print("  ❌ No allowed tokens - stopping")
            break
        
        # Apply constraints
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        constrained_logits = logits.clone()
        constrained_logits[0, ~torch.tensor(mask, dtype=torch.bool, device=device)] = -1e30
        
        # Select token
        next_token = int(torch.argmax(constrained_logits, dim=-1).item())
        
        # Advance adapter (this handles partial matches internally)
        if not adapter.step_with_token(next_token):
            print(f"  ❌ Adapter rejected token {next_token}")
            break
        
        # Update sequence
        if generated_tokens.size(1) == 0:
            generated_tokens = torch.tensor([[next_token]], device=device)
        else:
            generated_tokens = torch.cat([generated_tokens, torch.tensor([[next_token]], device=device)], dim=-1)
        
        # Show progress
        token_str = id_to_string(next_token)
        current_text = tokenizer.decode(generated_tokens[0], clean_up_tokenization_spaces=False)
        
        print(f"  → Token {next_token:5d} '{token_str}' → '{current_text}'")
        
        # Show state info for debugging
        if state_info['active_matches']:
            print(f"    Active matches: {len(state_info['active_matches'])}")
            for match in state_info['active_matches'][:2]:  # Show first 2
                print(f"      '{match['terminal']}' progress: {match['progress']}")
        
        # Check completion
        if parser.accepted():
            print("  🎉 Grammar accepted - JSON complete!")
            break
        
        print()
    else:
        print("  ⚠️  Maximum tokens reached")
    
    # Final results
    final_json = tokenizer.decode(generated_tokens[0], clean_up_tokenization_spaces=False)
    
    print(f"\n📊 Final Results")
    print("=" * 40)
    print(f"Token sequence: {generated_tokens[0].tolist()}")
    print(f"Generated JSON: '{final_json}'")
    print(f"Grammar accepted: {parser.accepted()}")
    
    if parser.accepted():
        log_prob = parser.sentence_logprob()
        print(f"Log probability: {log_prob:.4f}")
        print(f"Probability: {torch.exp(torch.tensor(log_prob)):.6f}")
    
    # Validate JSON
    print(f"\n🔍 JSON Validation")
    try:
        parsed = json.loads(final_json)
        print(f"✅ Valid JSON!")
        print(f"   Parsed as: {parsed}")
        print(f"   Type: {type(parsed)}")
        if isinstance(parsed, dict):
            print(f"   Keys: {list(parsed.keys())}")
            print(f"   Values: {list(parsed.values())}")
        return parsed
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        print(f"   This may be due to incomplete generation")
        return final_json


if __name__ == "__main__":
    print("Starting Robust JSON Generation Demo...")
    print("This demo shows how RobustConstrainedAdapter handles multi-token terminals.")
    print()
    
    result = robust_json_demo()
    
    print(f"\n🏆 Demo Complete!")
    print(f"Result: {result}")
    
    if isinstance(result, dict):
        print("🎉 Successfully generated valid JSON object!")