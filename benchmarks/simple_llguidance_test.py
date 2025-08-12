#!/usr/bin/env python3
"""
Simple test to get llguidance working and compare basic functionality.
"""

import time
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Stolcke imports
from stolcke_pcfg import PCFG, StolckeParser
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter

# llguidance imports
import llguidance
from llguidance import LLTokenizer, LLInterpreter, JsonCompiler


def test_llguidance_basic():
    """Test basic llguidance functionality."""
    
    print("🔍 Testing llguidance Basic Functionality")
    print("-" * 40)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Create llguidance tokenizer - try LLTokenizer
        ll_tokenizer = LLTokenizer(tokenizer)
        print("✅ LLTokenizer created")
        
        # Simple JSON schema
        schema = {
            "type": "object", 
            "properties": {
                "name": {"type": "string", "enum": ["Alice", "Bob"]},
                "age": {"type": "integer", "enum": [25, 30]}
            },
            "required": ["name"],
            "additionalProperties": False
        }
        
        # Create JSON compiler
        compiler = JsonCompiler(schema)
        print("✅ JsonCompiler created")
        
        # Create interpreter - try different API patterns
        try:
            interpreter = LLInterpreter(
                ll_tokenizer,
                compiler
            )
            print("✅ LLInterpreter created with positional args")
        except Exception as e1:
            try:
                interpreter = LLInterpreter(
                    ll_tokenizer=ll_tokenizer,
                    grammar=compiler
                )
                print("✅ LLInterpreter created with grammar param")
            except Exception as e2:
                try:
                    interpreter = LLInterpreter(ll_tokenizer, compiler, limits=None)
                    print("✅ LLInterpreter created with limits param")
                except Exception as e3:
                    print(f"❌ All LLInterpreter creation attempts failed:")
                    print(f"   Attempt 1: {e1}")
                    print(f"   Attempt 2: {e2}")
                    print(f"   Attempt 3: {e3}")
                    raise e3
        
        # Test basic operations
        print("\n🧪 Testing Basic Operations:")
        
        # Test different API methods to understand what's available
        print(f"Available interpreter methods: {[m for m in dir(interpreter) if not m.startswith('_')]}")
        
        # Test getting allowed tokens initially
        try:
            # Try different method names for getting constraints
            if hasattr(interpreter, 'compute_mask'):
                result = interpreter.compute_mask()
                print(f"✅ compute_mask() works, got result type: {type(result)}")
                print(f"   Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                
                if hasattr(result, 'sample_mask'):
                    if result.sample_mask is not None:
                        allowed_count = sum(1 for x in result.sample_mask if x) if hasattr(result.sample_mask, '__iter__') else 0
                        print(f"   Allowed tokens: {allowed_count}")
                    else:
                        print("   sample_mask is None")
                else:
                    print("   No sample_mask attribute")
            else:
                print("❌ No compute_mask method available")
            
        except Exception as e:
            print(f"❌ compute_mask() failed: {e}")
            
            # Try alternative methods
            try:
                if hasattr(interpreter, 'get_mask'):
                    mask = interpreter.get_mask()
                    print(f"✅ get_mask() works: {type(mask)}")
                elif hasattr(interpreter, 'allowed_tokens'):
                    tokens = interpreter.allowed_tokens()
                    print(f"✅ allowed_tokens() works: {len(tokens) if tokens else 0} tokens")
                else:
                    print("❌ No alternative mask methods found")
            except Exception as e2:
                print(f"❌ Alternative methods failed: {e2}")
        
        return True
        
    except Exception as e:
        print(f"❌ llguidance basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def simple_generation_comparison():
    """Compare simple generation between Stolcke and llguidance."""
    
    print("\n🥊 Simple Generation Comparison")
    print("=" * 40)
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()
    
    # Test 1: Stolcke PCFG
    print("\n📊 Test 1: Stolcke PCFG Parser")
    print("-" * 30)
    
    start_time = time.perf_counter()
    
    try:
        # Simple grammar
        grammar = PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ['"name"', ":", '"Alice"'], 1.0),
        ])
        
        parser = StolckeParser(grammar, "S")
        
        def id2str(tid): return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        def str2id(s):
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        
        adapter = RobustConstrainedAdapter(
            parser=parser,
            token_id_to_str=id2str,
            str_to_token_id=str2id,
            tokenizer=tokenizer
        )
        
        # Simple generation loop
        generated = []
        for step in range(10):
            allowed = adapter.allowed_token_ids()
            if not allowed:
                break
            
            # Just pick first allowed token for deterministic test
            next_token = list(allowed)[0]
            
            if not adapter.step_with_token(next_token):
                break
            
            generated.append(next_token)
            
            if parser.accepted():
                break
        
        stolcke_time = time.perf_counter() - start_time
        stolcke_output = tokenizer.decode(generated, skip_special_tokens=True)
        stolcke_success = parser.accepted()
        
        print(f"Result: {'✅' if stolcke_success else '❌'} '{stolcke_output}' ({stolcke_time*1000:.1f}ms)")
        if stolcke_success:
            log_prob = parser.sentence_logprob()
            print(f"Log probability: {log_prob:.4f}")
        
    except Exception as e:
        stolcke_success = False
        stolcke_output = ""
        stolcke_time = time.perf_counter() - start_time
        print(f"❌ Stolcke failed: {e}")
    
    # Test 2: llguidance
    print("\n📊 Test 2: llguidance")
    print("-" * 20)
    
    start_time = time.perf_counter()
    
    try:
        # Create llguidance components
        ll_tokenizer = TokenizerWrapper(tokenizer)
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["Alice"]}
            },
            "required": ["name"]
        }
        
        compiler = JsonCompiler(schema=schema)
        interpreter = LLInterpreter(
            ll_tokenizer=ll_tokenizer,
            compiler=compiler
        )
        
        # Simple generation loop
        generated = []
        for step in range(10):
            try:
                result = interpreter.compute_mask()
                if not hasattr(result, 'sample_mask') or not result.sample_mask:
                    break
                
                # Find first allowed token
                allowed_tokens = [i for i, allowed in enumerate(result.sample_mask) if allowed]
                if not allowed_tokens:
                    break
                
                next_token = allowed_tokens[0]
                
                # Advance interpreter
                advance_result = interpreter.advance_parser(next_token)
                if hasattr(advance_result, 'stop') and advance_result.stop:
                    break
                
                generated.append(next_token)
                
                # Check if we should stop (this is a guess at the API)
                if hasattr(result, 'is_accepting') and result.is_accepting():
                    break
                
            except Exception as e:
                print(f"   Step {step} error: {e}")
                break
        
        llguidance_time = time.perf_counter() - start_time
        llguidance_output = tokenizer.decode(generated, skip_special_tokens=True)
        
        # Check if output looks valid
        llguidance_success = False
        if llguidance_output.strip():
            try:
                json.loads(llguidance_output)
                llguidance_success = True
            except:
                llguidance_success = llguidance_output.strip().startswith('{')
        
        print(f"Result: {'✅' if llguidance_success else '❌'} '{llguidance_output}' ({llguidance_time*1000:.1f}ms)")
        
    except Exception as e:
        llguidance_success = False
        llguidance_output = ""
        llguidance_time = time.perf_counter() - start_time
        print(f"❌ llguidance failed: {e}")
    
    # Comparison
    print(f"\n🏆 COMPARISON SUMMARY")
    print("-" * 25)
    print(f"Stolcke PCFG:  {'✅' if stolcke_success else '❌'} Success, {stolcke_time*1000:.1f}ms")
    print(f"llguidance:    {'✅' if llguidance_success else '❌'} Success, {llguidance_time*1000:.1f}ms")
    
    if stolcke_success and llguidance_success:
        if stolcke_time < llguidance_time:
            faster = "Stolcke"
            ratio = llguidance_time / stolcke_time
        else:
            faster = "llguidance"  
            ratio = stolcke_time / llguidance_time
        print(f"Speed winner: {faster} ({ratio:.1f}x faster)")
    
    print(f"\nOutputs:")
    print(f"  Stolcke:    '{stolcke_output}'")
    print(f"  llguidance: '{llguidance_output}'")


def main():
    """Run llguidance exploration and comparison."""
    
    print("🔬 llguidance Integration & Comparison")
    print("=" * 45)
    
    # Test basic llguidance functionality
    if test_llguidance_basic():
        # Run simple comparison
        simple_generation_comparison()
    else:
        print("❌ Cannot run comparison - llguidance basic test failed")
        
        # Show what we have available
        print(f"\nAvailable in llguidance module:")
        for item in sorted(dir(llguidance)):
            if not item.startswith('_'):
                print(f"  • {item}")


if __name__ == "__main__":
    main()