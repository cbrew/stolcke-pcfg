#!/usr/bin/env python3
"""
Simple JSON demo that works with GPT-2's natural tokenization.

This version uses a very simple grammar with single tokens that GPT-2 actually produces.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stolcke_pcfg import PCFG, ConstrainedDecoderAdapter, StolckeParser


def build_working_grammar() -> PCFG:
    """
    Grammar using only single tokens that GPT-2 naturally produces.
    Based on analysis of GPT-2 tokenization:
    - '{"' -> [4895] 
    - '"}' -> [20662]
    - '"Alice' -> [1, 44484] -> just use 'Alice' -> [44484] 
    - Numbers are single tokens: 25->25, 30->1270, etc.
    """
    return PCFG([
        # Top level: just a simple object
        ("S", ["START", "NAME", "END"], 1.0),
        
        # Tokens that GPT-2 actually produces as single tokens
        ("START", ['{"'], 1.0),  # Token 4895
        ("END", ['"}'], 1.0),    # Token 20662  
        
        # For names, use tokens that appear in the middle of JSON
        ("NAME", ['Alice'], 0.5),  # Token 44484 
        ("NAME", ['Bob'], 0.5),    # Token 18861
    ])


def simple_filter_factory(tokenizer) -> callable:
    """Filter that only accepts exact single-token matches."""
    def to_ids(terms: set[str]) -> set[int]:
        out: set[int] = set()
        for term in terms:
            ids = tokenizer.encode(term, add_special_tokens=False)
            if len(ids) == 1:  # Only single tokens
                out.add(ids[0])
                print(f"      Accepting '{term}' -> token {ids[0]}")
        return out
    return to_ids


def simple_json_demo(model_name: str = "gpt2", max_new_tokens: int = 16) -> None:
    """Generate JSON using only single-token terminals."""
    
    # Device setup
    if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Setup grammar and parser
    grammar = build_working_grammar()
    parser = StolckeParser(grammar, "S")
    
    def id2str(token_id: int) -> str:
        return tok.decode([token_id], clean_up_tokenization_spaces=False)
    
    def str2id(s: str) -> int | None:
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None
    
    adapter = ConstrainedDecoderAdapter(
        parser, id2str, str2id,
        next_token_filter=simple_filter_factory(tok),
    )
    
    # Start generation
    generated = torch.empty((1, 0), dtype=torch.long, device=device)
    
    print("\n=== JSON Generation with Single Tokens ===")
    print("Grammar: START -> NAME -> END")
    print("Expected: {\"Alice\"} or {\"Bob\"}\n")
    
    for step in range(max_new_tokens):
        # Prepare input for model
        if generated.size(1) == 0:
            input_ids = torch.tensor([[tok.bos_token_id]], device=device)
        else:
            input_ids = generated
        
        # Get logits
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :]
        
        # Check what's allowed
        allowed_ids = adapter.allowed_token_ids()
        print(f"Step {step}: Grammar allows {len(allowed_ids)} tokens:")
        for aid in sorted(allowed_ids)[:5]:  # Show first 5
            print(f"    {aid} -> {id2str(aid)!r}")
        
        if not allowed_ids:
            print("No allowed tokens!")
            break
        
        # Apply constraints
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        disallowed = ~torch.tensor(mask, dtype=torch.bool, device=logits.device)
        logits[0, disallowed] = -1e30
        
        # Pick token
        next_id = int(torch.argmax(logits, dim=-1).item())
        
        # Advance parser
        if not adapter.step_with_token(next_id):
            print(f"❌ Token {next_id} rejected!")
            break
        
        # Update generation
        if generated.size(1) == 0:
            generated = torch.tensor([[next_id]], device=device)
        else:
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=-1)
        
        token_str = id2str(next_id)
        print(f"✅ Selected: {next_id} -> {token_str!r}")
        
        if parser.accepted():
            print("🎉 Complete! Grammar accepted.")
            break
        print()
    
    # Results
    final_text = tok.decode(generated[0], clean_up_tokenization_spaces=False)
    print(f"\n=== Final Result ===")
    print(f"Tokens: {generated[0].tolist()}")
    print(f"Text: {final_text!r}")
    print(f"Output: {final_text}")
    
    if parser.accepted():
        print(f"Log probability: {parser.sentence_logprob():.4f}")
    else:
        print("⚠️ Incomplete generation")


if __name__ == "__main__":
    simple_json_demo()