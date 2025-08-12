#!/usr/bin/env python3
"""
Complete robust JSON generation using HF generation API with RobustConstrainedAdapter.

This demo showcases the full pipeline:
1. RobustConstrainedAdapter handles multi-token terminals
2. GrammarConstrainedLogitsProcessor integrates with HF generation
3. Generates valid, complete JSON objects

This approach is more robust than manual decoding loops because:
- Uses HF's optimized generation algorithms
- Handles sampling, beam search, etc.
- Avoids repetition issues from manual greedy selection
"""
from __future__ import annotations

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    LogitsProcessorList,
    GenerationConfig
)

from stolcke_pcfg import PCFG, StolckeParser
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter
from stolcke_pcfg.hf_logits import GrammarConstrainedLogitsProcessor


class RobustGrammarLogitsProcessor(GrammarConstrainedLogitsProcessor):
    """
    Extended logits processor that works with RobustConstrainedAdapter.
    
    The main difference is that we need to update the adapter state
    as tokens are generated during the HF generation process.
    """
    
    def __init__(self, robust_adapter: RobustConstrainedAdapter, **kwargs):
        # We'll create a wrapper that looks like the old ConstrainedDecoderAdapter
        class AdapterWrapper:
            def __init__(self, robust_adapter):
                self.robust_adapter = robust_adapter
            
            def allowed_token_mask(self, vocab_size):
                return self.robust_adapter.allowed_token_mask(vocab_size)
                
            def step_with_token(self, token_id):
                return self.robust_adapter.step_with_token(token_id)
        
        super().__init__(AdapterWrapper(robust_adapter), **kwargs)
        self.robust_adapter = robust_adapter
    
    def __call__(self, input_ids, scores):
        # First apply the mask using parent logic
        result = super().__call__(input_ids, scores)
        
        # If we're in generation and have new tokens, update the adapter
        if hasattr(input_ids, 'shape') and input_ids.shape[1] > 0:
            # Get the last token for the first sequence (assuming batch_size=1)
            last_token = int(input_ids[0, -1].item()) if input_ids.shape[1] > 0 else None
            if last_token is not None and last_token != self._last_seen_token:
                self.robust_adapter.step_with_token(last_token)
                self._last_seen_token = last_token
        
        return result
    
    def reset(self):
        """Reset adapter state for new generation."""
        self.robust_adapter.partial_matches = []
        self._last_seen_token = None


def build_comprehensive_json_grammar() -> PCFG:
    """Build a comprehensive JSON grammar with various patterns."""
    return PCFG([
        # Main object structure
        ("S", ["Object"], 1.0),
        
        # Objects can have different structures
        ("Object", ["{", "KeyValuePairs", "}"], 1.0),
        
        # Key-value pairs (single or multiple)
        ("KeyValuePairs", ["KeyValue"], 0.6),
        ("KeyValuePairs", ["KeyValue", ",", "KeyValuePairs"], 0.4),
        
        # Individual key-value pairs
        ("KeyValue", ["Key", ":", "Value"], 1.0),
        
        # Keys - realistic JSON keys
        ("Key", ['"name"'], 0.25),
        ("Key", ['"age"'], 0.25),
        ("Key", ['"city"'], 0.25),
        ("Key", ['"email"'], 0.25),
        
        # Values - mix of types
        ("Value", ["StringValue"], 0.6),
        ("Value", ["NumberValue"], 0.3),
        ("Value", ["BooleanValue"], 0.1),
        
        # String values
        ("StringValue", ['"Alice"'], 0.2),
        ("StringValue", ['"Bob"'], 0.2),
        ("StringValue", ['"Charlie"'], 0.2),
        ("StringValue", ['"NYC"'], 0.2),
        ("StringValue", ['"alice@example.com"'], 0.2),
        
        # Number values
        ("NumberValue", ["25"], 0.33),
        ("NumberValue", ["30"], 0.33),
        ("NumberValue", ["35"], 0.34),
        
        # Boolean values
        ("BooleanValue", ["true"], 0.5),
        ("BooleanValue", ["false"], 0.5),
    ])


def robust_hf_generation_demo():
    """Demonstrate robust JSON generation using HF generate() API."""
    
    print("🚀 Robust HF JSON Generation Demo")
    print("🎯 Multi-token terminals + HF generation API")
    print("=" * 60)
    
    # Device setup
    if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
        device = torch.device("mps")
        print(f"✅ Using MPS: {device}")
    else:
        device = torch.device("cpu")
        print(f"✅ Using CPU: {device}")
    
    # Load model and tokenizer
    model_name = "gpt2"
    print(f"\n📥 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Set pad token for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully")
    
    # Build grammar and parser
    grammar = build_comprehensive_json_grammar()
    print(f"\n📝 Built JSON grammar with {len(list(grammar.nonterminals))} nonterminals")
    
    # Show a few example rules
    print("Sample rules:")
    rule_count = 0
    for nt in ["S", "Object", "KeyValue"]:
        for rule in grammar.rules_for(nt):
            print(f"  {rule}")
            rule_count += 1
            if rule_count >= 3:
                break
        if rule_count >= 3:
            break
    print("  ...")
    
    # Create parser and robust adapter
    parser = StolckeParser(grammar, "S")
    
    def id2str(token_id: int) -> str:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    
    def str2id(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None
    
    robust_adapter = RobustConstrainedAdapter(
        parser=parser,
        token_id_to_str=id2str,
        str_to_token_id=str2id,
        tokenizer=tokenizer
    )
    
    # Create logits processor
    processor = RobustGrammarLogitsProcessor(robust_adapter)
    processors = LogitsProcessorList([processor])
    
    # Generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,  # Use sampling to avoid repetition
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print(f"\n🎲 Generating JSON with HF generate()...")
    print(f"Config: max_tokens={generation_config.max_new_tokens}, "
          f"temperature={generation_config.temperature}")
    
    # Start with empty input (model will generate from scratch)
    input_ids = torch.tensor([[]], dtype=torch.long, device=device)
    
    # Generate with constraints
    with torch.no_grad():
        try:
            generated = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                logits_processor=processors,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            generated_ids = generated.sequences[0]
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print("Falling back to manual generation...")
            
            # Fallback: manual generation with robust adapter
            generated_ids = manual_generation_fallback(
                model, tokenizer, robust_adapter, device, max_tokens=30
            )
    
    # Decode result
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n📊 Generation Results")
    print("=" * 40)
    print(f"Token IDs: {generated_ids.tolist() if hasattr(generated_ids, 'tolist') else list(generated_ids)}")
    print(f"Generated: '{result_text}'")
    print(f"Parser accepted: {parser.accepted()}")
    
    if parser.accepted():
        try:
            log_prob = parser.sentence_logprob()
            print(f"Grammar log-prob: {log_prob:.4f}")
        except:
            print("Grammar log-prob: unavailable")
    
    # Validate JSON
    print(f"\n🔍 JSON Validation")
    try:
        parsed_json = json.loads(result_text)
        print(f"✅ Valid JSON!")
        print(f"Parsed object: {parsed_json}")
        if isinstance(parsed_json, dict):
            print(f"Keys: {list(parsed_json.keys())}")
            print(f"Values: {list(parsed_json.values())}")
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"This may indicate incomplete generation or tokenization issues")
        return result_text


def manual_generation_fallback(model, tokenizer, adapter, device, max_tokens=30):
    """Fallback manual generation if HF generate() fails."""
    print("Using manual generation fallback...")
    
    generated = torch.empty((1, 0), dtype=torch.long, device=device)
    
    for step in range(max_tokens):
        # Model input
        if generated.size(1) == 0:
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        else:
            input_ids = generated
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
        
        # Apply constraints
        allowed = adapter.allowed_token_ids()
        if not allowed:
            break
        
        mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
        logits[0, ~torch.tensor(mask, dtype=torch.bool, device=device)] = -1e30
        
        # Sample from constrained distribution instead of greedy
        probs = torch.softmax(logits[0] / 0.8, dim=-1)  # temperature=0.8
        next_token = torch.multinomial(probs, 1).item()
        
        if not adapter.step_with_token(next_token):
            break
        
        if generated.size(1) == 0:
            generated = torch.tensor([[next_token]], device=device)
        else:
            generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=-1)
        
        if adapter.parser.accepted():
            break
    
    return generated[0]


if __name__ == "__main__":
    result = robust_hf_generation_demo()
    
    print(f"\n🏆 Demo Complete!")
    print(f"Final result: {result}")
    
    if isinstance(result, dict):
        print("🎉 Successfully generated valid JSON object!")
    else:
        print("⚠️  Result may need refinement, but demo shows the robust approach")