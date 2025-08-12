#!/usr/bin/env python3
"""
Benchmark comparing Stolcke PCFG parser against actual llguidance library.

This is the real competition - llguidance is a production-ready library
for grammar-constrained generation. Let's see how our approach compares.

Installation: pip install llguidance
"""

import time
import torch
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Stolcke imports
from stolcke_pcfg import PCFG, StolckeParser
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter

# Try to import llguidance - it's optional
try:
    import llguidance
    from llguidance import LLTokenizer, LLInterpreter
    LLGUIDANCE_AVAILABLE = True
except ImportError:
    LLGUIDANCE_AVAILABLE = False
    print("⚠️  llguidance not available. Install with: pip install llguidance")


@dataclass
class ComparisonResult:
    """Results from comparing Stolcke vs llguidance."""
    library: str
    success: bool
    steps: int
    total_time: float
    avg_step_time: float
    output_text: str
    has_probabilities: bool
    log_probability: Optional[float] = None
    memory_usage: Optional[int] = None
    error: Optional[str] = None


class LLGuidanceComparison:
    """Compare Stolcke parser against actual llguidance library."""
    
    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
    
    def create_json_grammar(self) -> Dict[str, str]:
        """Create matching grammars for both systems."""
        
        # Stolcke PCFG format
        stolcke_grammar = PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ["Key", ":", "Value"], 1.0),
            ("Key", ['"name"'], 0.4),
            ("Key", ['"email"'], 0.3),
            ("Key", ['"age"'], 0.3),
            ("Value", ['"Alice"'], 0.2),
            ("Value", ['"alice@domain.com"'], 0.2),
            ("Value", ['"Bob"'], 0.2),
            ("Value", ["25"], 0.2),
            ("Value", ["30"], 0.2),
        ])
        
        # llguidance EBNF format (approximate equivalent)
        llguidance_grammar = """
        start: "{" pair "}"
        pair: key ":" value
        key: "\\"name\\"" | "\\"email\\"" | "\\"age\\""
        value: "\\"Alice\\"" | "\\"alice@domain.com\\"" | "\\"Bob\\"" | "25" | "30"
        """
        
        return {
            'stolcke': stolcke_grammar,
            'llguidance': llguidance_grammar
        }
    
    def run_stolcke_generation(
        self,
        model,
        tokenizer,
        grammar,
        max_tokens: int = 20
    ) -> ComparisonResult:
        """Run generation with Stolcke parser."""
        
        start_time = time.perf_counter()
        
        try:
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
            
            generated = torch.empty((1, 0), dtype=torch.long, device=self.device)
            steps = 0
            
            for step in range(max_tokens):
                # Model input
                if generated.size(1) == 0:
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=self.device)
                else:
                    input_ids = generated
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                
                # Constraints
                allowed = adapter.allowed_token_ids()
                if not allowed:
                    break
                
                mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
                logits[0, ~torch.tensor(mask, dtype=torch.bool, device=self.device)] = -1e30
                
                # Sample
                probs = torch.softmax(logits[0] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if not adapter.step_with_token(next_token):
                    break
                
                if generated.size(1) == 0:
                    generated = torch.tensor([[next_token]], device=self.device)
                else:
                    generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=-1)
                
                steps += 1
                
                if parser.accepted():
                    break
            
            total_time = time.perf_counter() - start_time
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Get log probability
            log_prob = None
            if parser.accepted():
                try:
                    log_prob = parser.sentence_logprob()
                except:
                    pass
            
            return ComparisonResult(
                library="Stolcke",
                success=parser.accepted(),
                steps=steps,
                total_time=total_time,
                avg_step_time=total_time / steps if steps > 0 else 0,
                output_text=output_text,
                has_probabilities=True,
                log_probability=log_prob
            )
            
        except Exception as e:
            total_time = time.perf_counter() - start_time
            return ComparisonResult(
                library="Stolcke",
                success=False,
                steps=0,
                total_time=total_time,
                avg_step_time=0,
                output_text="",
                has_probabilities=True,
                error=str(e)
            )
    
    def run_llguidance_generation(
        self,
        model,
        tokenizer,
        grammar_str: str,
        max_tokens: int = 20
    ) -> ComparisonResult:
        """Run generation with llguidance library."""
        
        if not LLGUIDANCE_AVAILABLE:
            return ComparisonResult(
                library="llguidance",
                success=False,
                steps=0,
                total_time=0,
                avg_step_time=0,
                output_text="",
                has_probabilities=False,
                error="llguidance not available"
            )
        
        start_time = time.perf_counter()
        
        try:
            # Create llguidance components
            ll_tokenizer = LLTokenizer(tokenizer)
            
            # Create grammar - llguidance expects different format
            # Let's try a simpler JSON schema approach
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["Alice", "Bob"]},
                    "email": {"type": "string", "enum": ["alice@domain.com"]},
                    "age": {"type": "number", "enum": [25, 30]}
                },
                "oneOf": [
                    {"required": ["name"]},
                    {"required": ["email"]}, 
                    {"required": ["age"]}
                ]
            }
            
            # Initialize llguidance interpreter
            interpreter = LLInterpreter(
                ll_tokenizer=ll_tokenizer,
                schema=schema
            )
            
            generated = []
            steps = 0
            
            for step in range(max_tokens):
                # Get allowed tokens from llguidance
                allowed_tokens = interpreter.get_allowed_tokens()
                if not allowed_tokens:
                    break
                
                # Model forward pass
                if not generated:
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=self.device)
                else:
                    input_ids = torch.tensor([generated], device=self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                
                # Apply llguidance constraints
                mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=self.device)
                for token_id in allowed_tokens:
                    if 0 <= token_id < logits.size(-1):
                        mask[token_id] = True
                
                logits[0, ~mask] = -1e30
                
                # Sample
                probs = torch.softmax(logits[0] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Advance llguidance
                if not interpreter.advance(next_token):
                    break
                
                generated.append(next_token)
                steps += 1
                
                if interpreter.is_complete():
                    break
            
            total_time = time.perf_counter() - start_time
            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            
            return ComparisonResult(
                library="llguidance",
                success=interpreter.is_complete() if 'interpreter' in locals() else False,
                steps=steps,
                total_time=total_time,
                avg_step_time=total_time / steps if steps > 0 else 0,
                output_text=output_text,
                has_probabilities=False
            )
            
        except Exception as e:
            total_time = time.perf_counter() - start_time
            return ComparisonResult(
                library="llguidance",
                success=False,
                steps=0,
                total_time=total_time,
                avg_step_time=0,
                output_text="",
                has_probabilities=False,
                error=str(e)
            )
    
    def run_comparison(self, model_name: str = "gpt2", trials: int = 3):
        """Run comprehensive comparison."""
        
        print("🥊 Stolcke PCFG vs llguidance Comparison")
        print("🎯 Real-World Performance Battle")
        print("=" * 60)
        
        if not LLGUIDANCE_AVAILABLE:
            print("❌ llguidance not available. Install with:")
            print("   pip install llguidance")
            print("   or")
            print("   pip install guidance")
            return None
        
        # Load model
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        grammars = self.create_json_grammar()
        
        results = {
            'stolcke': [],
            'llguidance': []
        }
        
        for trial in range(trials):
            print(f"\n🔄 Trial {trial + 1}/{trials}")
            print("-" * 25)
            
            # Test Stolcke
            print("Testing Stolcke PCFG Parser...")
            stolcke_result = self.run_stolcke_generation(
                model, tokenizer, grammars['stolcke']
            )
            results['stolcke'].append(stolcke_result)
            
            status = "✅" if stolcke_result.success else "❌"
            print(f"  {status} {stolcke_result.steps} steps, "
                  f"{stolcke_result.total_time*1000:.1f}ms, "
                  f"'{stolcke_result.output_text}'")
            if stolcke_result.error:
                print(f"  Error: {stolcke_result.error}")
            
            # Test llguidance
            print("Testing llguidance...")
            llguidance_result = self.run_llguidance_generation(
                model, tokenizer, grammars['llguidance']
            )
            results['llguidance'].append(llguidance_result)
            
            status = "✅" if llguidance_result.success else "❌"
            print(f"  {status} {llguidance_result.steps} steps, "
                  f"{llguidance_result.total_time*1000:.1f}ms, "
                  f"'{llguidance_result.output_text}'")
            if llguidance_result.error:
                print(f"  Error: {llguidance_result.error}")
        
        self.analyze_results(results)
        return results
    
    def analyze_results(self, results: Dict[str, List[ComparisonResult]]):
        """Analyze comparison results."""
        
        print(f"\n📊 COMPARISON ANALYSIS")
        print("=" * 40)
        
        stolcke_results = results['stolcke']
        llguidance_results = results['llguidance']
        
        def avg_metric(results_list, metric):
            values = [getattr(r, metric) for r in results_list 
                     if getattr(r, metric) is not None and r.success]
            return sum(values) / len(values) if values else 0
        
        print("| Metric | Stolcke PCFG | llguidance | Winner |")
        print("|--------|--------------|------------|--------|")
        
        # Success rate
        stolcke_success = sum(r.success for r in stolcke_results) / len(stolcke_results)
        llguidance_success = sum(r.success for r in llguidance_results) / len(llguidance_results)
        success_winner = "Stolcke" if stolcke_success > llguidance_success else "llguidance" if llguidance_success > stolcke_success else "Tie"
        
        print(f"| Success Rate | {stolcke_success:>10.1%} | {llguidance_success:>8.1%} | {success_winner:>6} |")
        
        # Speed
        stolcke_speed = avg_metric(stolcke_results, 'avg_step_time') * 1000
        llguidance_speed = avg_metric(llguidance_results, 'avg_step_time') * 1000
        if stolcke_speed > 0 and llguidance_speed > 0:
            speed_winner = "llguidance" if llguidance_speed < stolcke_speed else "Stolcke"
            print(f"| Avg Step Time | {stolcke_speed:>7.1f}ms | {llguidance_speed:>6.1f}ms | {speed_winner:>6} |")
        
        # Probabilities
        stolcke_probs = all(r.has_probabilities for r in stolcke_results)
        llguidance_probs = all(r.has_probabilities for r in llguidance_results)
        prob_winner = "Stolcke" if stolcke_probs and not llguidance_probs else "Both" if stolcke_probs and llguidance_probs else "Neither"
        
        print(f"| Probabilities | {str(stolcke_probs):>10} | {str(llguidance_probs):>8} | {prob_winner:>6} |")
        
        # Example outputs
        print(f"\n📝 EXAMPLE OUTPUTS")
        print("-" * 20)
        
        if stolcke_results and stolcke_results[0].output_text:
            output = stolcke_results[0].output_text
            prob = stolcke_results[0].log_probability
            print(f"Stolcke:    '{output}'")
            if prob is not None:
                print(f"            Log-prob: {prob:.4f}")
        
        if llguidance_results and llguidance_results[0].output_text:
            output = llguidance_results[0].output_text
            print(f"llguidance: '{output}'")
            print(f"            Log-prob: N/A")
        
        # Performance analysis
        if stolcke_speed > 0 and llguidance_speed > 0:
            speed_ratio = stolcke_speed / llguidance_speed
            faster_lib = "llguidance" if speed_ratio > 1 else "Stolcke"
            
            print(f"\n🏆 PERFORMANCE VERDICT")
            print("-" * 25)
            print(f"Speed: {faster_lib} is {abs(speed_ratio):.1f}x faster")
            print(f"Stolcke advantage: Grammar log-probabilities")
            print(f"llguidance advantage: {'Speed' if speed_ratio > 1 else 'None identified'}")
        
        # Error analysis
        stolcke_errors = [r.error for r in stolcke_results if r.error]
        llguidance_errors = [r.error for r in llguidance_results if r.error]
        
        if stolcke_errors or llguidance_errors:
            print(f"\n🐛 ERROR ANALYSIS")
            print("-" * 17)
            if stolcke_errors:
                print(f"Stolcke errors: {len(stolcke_errors)}")
                for error in stolcke_errors[:2]:
                    print(f"  • {error}")
            if llguidance_errors:
                print(f"llguidance errors: {len(llguidance_errors)}")
                for error in llguidance_errors[:2]:
                    print(f"  • {error}")


def main():
    """Run llguidance comparison if available."""
    
    if not LLGUIDANCE_AVAILABLE:
        print("🔧 Setup Instructions:")
        print("To run this comparison, install llguidance:")
        print("  pip install llguidance")
        print("\nOr try guidance (the predecessor):")
        print("  pip install guidance")
        print("\nThen re-run this script.")
        return
    
    comparison = LLGuidanceComparison()
    results = comparison.run_comparison(trials=3)
    
    print(f"\n🎯 This benchmark shows how Stolcke PCFG compares to the")
    print(f"real-world production library llguidance for constrained generation.")


if __name__ == "__main__":
    main()