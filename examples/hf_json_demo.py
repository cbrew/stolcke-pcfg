#!/usr/bin/env python3
"""
Improved JSON generation demo that produces valid JSON.

This version uses GPT-2's natural tokenization patterns and a simpler grammar
to successfully generate complete JSON objects.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stolcke_pcfg import PCFG, ConstrainedDecoderAdapter, StolckeParser


def build_simple_json_grammar() -> PCFG:
    """Build a grammar using GPT-2's natural tokenization patterns."""
    return PCFG([
        # Start with object
        ("S", ["Object"], 1.0),
        
        # Objects use GPT-2's natural tokens like '{"' and '"}'
        ("Object", ['{"', "KeyValue", '"}'], 1.0),
        
        # Key-value patterns matching GPT-2 tokenization
        ("KeyValue", ['":', "Value"], 0.25),
        ("KeyValue", ['name":', "NameValue"], 0.25),  
        ("KeyValue", ['age":', "AgeValue"], 0.25),
        ("KeyValue", ['city":', "CityValue"], 0.25),
        
        # Values that are single tokens
        ("NameValue", [' "Alice'], 0.5),
        ("NameValue", [' "Bob'], 0.5),
        
        ("AgeValue", [' 25'], 0.33),
        ("AgeValue", [' 30'], 0.33), 
        ("AgeValue", [' 35'], 0.34),
        
        ("CityValue", [' "NYC'], 0.5),
        ("CityValue", [' "LA'], 0.5),
        
        ("Value", [' "default'], 1.0),
    ])


def multi_token_filter_factory(tokenizer) -> callable:
    """Filter that handles both single and multi-token terminals."""
    def to_ids(terms: set[str]) -> set[int]:
        out: set[int] = set()
        for term in terms:
            ids = tokenizer.encode(term, add_special_tokens=False)
            # For multi-token terminals, we only allow the first token
            # This works because the parser tracks partial matches
            if len(ids) >= 1:
                out.add(ids[0])
        return out
    return to_ids


def greedy_json_demo(model_name: str = "gpt2", max_new_tokens: int = 32) -> None:
    """Generate a complete JSON object using constrained decoding."""
    
    # Device setup for Apple Silicon compatibility  
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
    grammar = build_simple_json_grammar()
    parser = StolckeParser(grammar, "S")
    
    # Token conversion functions
    def id2str(token_id: int) -> str:
        return tok.decode([token_id], clean_up_tokenization_spaces=False)
    
    def str2id(s: str) -> int | None:
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) >= 1 else None
    
    # Create adapter
    adapter = ConstrainedDecoderAdapter(
        parser,
        id2str, 
        str2id,
        next_token_filter=multi_token_filter_factory(tok),
    )
    
    # Start generation (no BOS token, start directly)
    generated = torch.empty((1, 0), dtype=torch.long, device=device)
    
    print("=== Starting JSON Generation ===")
    print("Expected pattern: {\"key\": value}")
    print()
    
    for step in range(max_new_tokens):
        # Get next token logits
        if generated.size(1) == 0:
            # First token: use a dummy input
            input_ids = torch.tensor([[tok.bos_token_id]], device=device)
        else:
            input_ids = generated
            
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :]
        
        # Apply grammar constraints
        allowed_ids = adapter.allowed_token_ids()
        if not allowed_ids:
            print("No allowed tokens - stopping")
            break
            
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        disallowed = ~torch.tensor(mask, dtype=torch.bool, device=logits.device)
        logits[0, disallowed] = -1e30
        
        # Greedy selection
        next_id = int(torch.argmax(logits, dim=-1).item())
        
        # Try to advance parser
        if not adapter.step_with_token(next_id):
            print(f"Token {next_id} rejected by grammar")
            break
            
        # Add to generated sequence
        if generated.size(1) == 0:
            generated = torch.tensor([[next_id]], device=device)
        else:
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=-1)
        
        # Decode and display progress
        token_str = id2str(next_id)
        allowed_count = len(allowed_ids)
        
        print(f"Step {step:2d}: token_id={next_id:5d} token={token_str!r:>8} allowed={allowed_count}")
        
        # Check if complete
        if parser.accepted():
            print()
            print("✅ Grammar accepted! Complete JSON generated.")
            break
    else:
        print()
        print("⚠️  Reached max tokens without completion.")
    
    # Show final result
    final_text = tok.decode(generated[0], clean_up_tokenization_spaces=False)
    print(f"\n=== Generated JSON ===")
    print(f"Raw tokens: {generated[0].tolist()}")
    print(f"Text: {final_text!r}")
    print(f"Formatted:")
    print(final_text)
    
    if parser.accepted():
        print(f"\n📊 Sentence log probability: {parser.sentence_logprob():.4f}")


if __name__ == "__main__":
    greedy_json_demo()