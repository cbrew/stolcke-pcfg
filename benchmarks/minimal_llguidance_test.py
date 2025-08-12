#!/usr/bin/env python3
"""
Minimal test to understand llguidance's tokenizer requirements.
"""

import json
from transformers import AutoTokenizer
import llguidance


def create_custom_tokenizer_wrapper(hf_tokenizer):
    """Create a custom tokenizer wrapper that llguidance can use."""
    
    class CustomTokenizerWrapper:
        def __init__(self, hf_tok):
            self.hf_tok = hf_tok
            self.bos_token_id = hf_tok.bos_token_id
            self.eos_token_id = hf_tok.eos_token_id
            
            # Create the tokens list that llguidance expects
            self.tokens = []
            vocab = hf_tok.get_vocab()
            
            # Sort by token ID to create ordered list
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            self.tokens = [token for token, _ in sorted_vocab]
            
            print(f"Created tokenizer wrapper with {len(self.tokens)} tokens")
        
        def encode(self, text):
            return self.hf_tok.encode(text, add_special_tokens=False)
        
        def decode(self, tokens):
            return self.hf_tok.decode(tokens)
        
        def __call__(self, text):
            return self.encode(text)
    
    return CustomTokenizerWrapper(hf_tokenizer)


def test_minimal_llguidance():
    """Test minimal llguidance functionality."""
    
    print("🔬 Minimal llguidance Test")
    print("=" * 30)
    
    try:
        # Load tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✅ HuggingFace tokenizer loaded")
        
        # Create custom wrapper
        custom_tokenizer = create_custom_tokenizer_wrapper(hf_tokenizer)
        print("✅ Custom tokenizer wrapper created")
        
        # Try to create llguidance TokenizerWrapper and LLTokenizer
        tokenizer_wrapper = llguidance.TokenizerWrapper(custom_tokenizer)
        print("✅ llguidance TokenizerWrapper created")
        
        ll_tokenizer = llguidance.LLTokenizer(tokenizer_wrapper)
        print("✅ llguidance LLTokenizer created")
        
        # Try creating a simple grammar using EBNF/Lark format
        try:
            # JsonCompiler signature: (separators=None, whitespace_flexible=False, coerce_one_of=False, whitespace_pattern=None)
            # It seems JsonCompiler is for creating JSON grammar, not taking a schema
            compiler = llguidance.JsonCompiler()
            print("✅ JsonCompiler created (no args)")
        except Exception as e1:
            try:
                # Try Lark grammar format
                grammar_text = '''
                start: "{\\"name\\":\\"Alice\\"}"
                '''
                compiler = llguidance.LarkCompiler()
                grammar = compiler.compile(grammar_text)
                print("✅ Lark grammar compiled")
            except Exception as e2:
                print(f"❌ All compiler attempts failed:")
                print(f"   JsonCompiler: {e1}")
                print(f"   LarkCompiler: {e2}")
                raise e2
        
        # Create interpreter
        interpreter = llguidance.LLInterpreter(ll_tokenizer, compiler)
        print("✅ LLInterpreter created")
        
        # Test basic operations
        print("\n🧪 Testing compute_mask:")
        result = interpreter.compute_mask()
        print(f"✅ compute_mask() returned: {type(result)}")
        
        if hasattr(result, 'sample_mask') and result.sample_mask:
            allowed_count = sum(1 for x in result.sample_mask if x)
            print(f"   Allowed tokens: {allowed_count}")
            
            # Show some allowed tokens
            allowed_tokens = [i for i, allowed in enumerate(result.sample_mask) if allowed][:10]
            allowed_strings = [custom_tokenizer.tokens[i] for i in allowed_tokens]
            print(f"   First 10 allowed: {allowed_strings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_minimal_llguidance()