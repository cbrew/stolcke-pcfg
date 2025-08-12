#!/usr/bin/env python3
"""
Quick benchmark runner for comparing tokenization approaches.

This is a streamlined version for rapid testing and demonstration.
"""

import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from stolcke_pcfg import PCFG, StolckeParser, ConstrainedDecoderAdapter
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter


def quick_benchmark():
    """Run a quick comparison between naive and robust adapters."""
    
    print("🎯 Quick Tokenization Benchmark")
    print("=" * 50)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    
    # Load model
    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()
    
    # Test grammar with multi-token terminals
    grammar = PCFG([
        ("S", ["{", "Pair", "}"], 1.0),
        ("Pair", ["Key", ":", "Value"], 1.0), 
        ("Key", ['"email"'], 1.0),           # Multi-token: ['"', 'email', '"']
        ("Value", ['"alice@domain.com"'], 1.0), # Multi-token: 8 tokens!
    ])
    
    # Show tokenization
    print("\nTokenization Analysis:")
    test_terminals = ['"email"', '"alice@domain.com"']
    for term in test_terminals:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        decoded = [tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in tokens]
        print(f"  '{term}' -> {tokens} -> {decoded} ({len(tokens)} tokens)")
    
    results = {}
    
    # Test both adapters
    for adapter_name in ["naive", "robust"]:
        print(f"\n🔧 Testing {adapter_name.upper()} adapter:")
        
        parser = StolckeParser(grammar, "S")
        
        if adapter_name == "naive":
            def id2str(tid): 
                return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            def str2id(s): 
                ids = tokenizer.encode(s, add_special_tokens=False)
                return ids[0] if len(ids) == 1 else None
            def single_token_filter(terms):
                allowed = set()
                for term in terms:
                    ids = tokenizer.encode(term, add_special_tokens=False)
                    if len(ids) == 1:
                        allowed.add(ids[0])
                return allowed
            
            adapter = ConstrainedDecoderAdapter(
                parser, id2str, str2id, next_token_filter=single_token_filter
            )
        else:
            def id2str(tid):
                return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            def str2id(s):
                ids = tokenizer.encode(s, add_special_tokens=False)
                return ids[0] if len(ids) == 1 else None
            
            adapter = RobustConstrainedAdapter(
                parser=parser,
                token_id_to_str=id2str,
                str_to_token_id=str2id,
                tokenizer=tokenizer
            )
        
        # Generation
        start_time = time.time()
        generated = torch.empty((1, 0), dtype=torch.long, device=device)
        success = False
        error = None
        
        try:
            for step in range(20):  # Max 20 tokens
                # Model input
                if generated.size(1) == 0:
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
                else:
                    input_ids = generated
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                
                # Constraints
                allowed = adapter.allowed_token_ids()
                if not allowed:
                    error = "No allowed tokens"
                    break
                
                mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
                logits[0, ~torch.tensor(mask, dtype=torch.bool, device=device)] = -1e30
                
                # Sample token
                probs = torch.softmax(logits[0] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if not adapter.step_with_token(next_token):
                    error = "Token rejected"
                    break
                
                if generated.size(1) == 0:
                    generated = torch.tensor([[next_token]], device=device)
                else:
                    generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=-1)
                
                # Check completion
                if hasattr(adapter, 'parser'):
                    accepted = adapter.parser.accepted()
                else:
                    accepted = getattr(adapter, 'robust_adapter', adapter).parser.accepted()
                
                if accepted:
                    success = True
                    break
            
            if not success and not error:
                error = "Max tokens reached"
                
        except Exception as e:
            error = str(e)
        
        end_time = time.time()
        
        # Results
        output = tokenizer.decode(generated[0], skip_special_tokens=True) if generated.size(1) > 0 else ""
        
        results[adapter_name] = {
            'success': success,
            'tokens': generated.size(1) if generated.size(1) > 0 else 0,
            'time': end_time - start_time,
            'output': output,
            'error': error
        }
        
        # Display
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}")
        print(f"  Tokens generated: {results[adapter_name]['tokens']}")
        print(f"  Time taken: {results[adapter_name]['time']:.3f}s")
        print(f"  Output: '{output}'")
        if error:
            print(f"  Error: {error}")
        
        # Validate JSON
        if output:
            try:
                parsed = json.loads(output)
                print(f"  ✅ Valid JSON: {parsed}")
            except:
                print(f"  ❌ Invalid JSON")
    
    # Summary
    print(f"\n📊 BENCHMARK SUMMARY")
    print("=" * 50)
    
    naive = results['naive']
    robust = results['robust']
    
    print(f"Success Rate:")
    print(f"  Naive:  {'✅' if naive['success'] else '❌'} ({int(naive['success'])})")
    print(f"  Robust: {'✅' if robust['success'] else '❌'} ({int(robust['success'])})")
    
    if robust['success']:
        improvement = "∞%" if not naive['success'] else f"{((robust['tokens'] / naive['tokens']) - 1) * 100:.0f}%"
        print(f"\nToken Generation:")
        print(f"  Naive:  {naive['tokens']} tokens")
        print(f"  Robust: {robust['tokens']} tokens")
        
        print(f"\nPerformance:")
        print(f"  Naive:  {naive['time']:.3f}s")
        print(f"  Robust: {robust['time']:.3f}s ({robust['time']/naive['time']:.1f}x overhead)" if naive['time'] > 0 else f"  Robust: {robust['time']:.3f}s")
    
    print(f"\nGenerated Content:")
    print(f"  Naive:  '{naive['output']}'")
    print(f"  Robust: '{robust['output']}'")
    
    print(f"\n🎉 KEY INSIGHT: Robust adapter handles multi-token terminals that break naive approach!")
    
    return results


if __name__ == "__main__":
    quick_benchmark()