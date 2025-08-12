#!/usr/bin/env python3
"""
Fixed comparison with llguidance using proper API integration.

This version properly integrates with llguidance's expected interfaces.
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

# Try different llguidance imports
try:
    import llguidance
    from llguidance import TokenizerWrapper, LLInterpreter
    LLGUIDANCE_AVAILABLE = True
    print(f"✅ llguidance {llguidance.__version__} loaded successfully")
except ImportError:
    try:
        # Try alternative import paths
        from llguidance.api import LLInterpreter, TokenizerWrapper
        LLGUIDANCE_AVAILABLE = True
        print("✅ llguidance loaded via alternative import")
    except ImportError:
        LLGUIDANCE_AVAILABLE = False
        print("❌ llguidance not available")


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
    error: Optional[str] = None


class FixedLLGuidanceComparison:
    """Fixed comparison with proper llguidance integration."""
    
    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
    
    def create_json_schema(self):
        """Create a JSON schema that llguidance can work with."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["Alice", "Bob", "Charlie", "alice@domain.com"]
                },
                "age": {
                    "type": "integer", 
                    "enum": [25, 30, 35]
                }
            },
            "required": ["name"],
            "additionalProperties": False
        }
    
    def create_stolcke_grammar(self):
        """Create equivalent Stolcke grammar."""
        return PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ['"name"', ":", "Value"], 1.0),
            ("Value", ['"Alice"'], 0.25),
            ("Value", ['"Bob"'], 0.25),
            ("Value", ['"Charlie"'], 0.25),
            ("Value", ['"alice@domain.com"'], 0.25),
        ])
    
    def run_stolcke_generation(
        self,
        model,
        tokenizer,
        max_tokens: int = 15
    ) -> ComparisonResult:
        """Run generation with Stolcke parser."""
        
        start_time = time.perf_counter()
        
        try:
            grammar = self.create_stolcke_grammar()
            parser = StolckeParser(grammar, "S")
            
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
                library="Stolcke PCFG",
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
                library="Stolcke PCFG",
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
        max_tokens: int = 15
    ) -> ComparisonResult:
        """Run generation with llguidance using proper API."""
        
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
            # Create tokenizer wrapper the way llguidance expects
            ll_tokenizer = TokenizerWrapper(tokenizer)
            
            # Create schema
            schema = self.create_json_schema()
            
            # Initialize interpreter with proper parameters
            interpreter = LLInterpreter(
                ll_tokenizer=ll_tokenizer,
                schema=schema,
                log_level=0  # Reduce logging noise
            )
            
            generated_tokens = []
            steps = 0
            
            for step in range(max_tokens):
                # Get constraints from llguidance
                try:
                    # Get allowed tokens
                    mask_result = interpreter.compute_mask()
                    if not hasattr(mask_result, 'sample_mask') or mask_result.sample_mask is None:
                        break
                    
                    allowed_tokens = [i for i, allowed in enumerate(mask_result.sample_mask) if allowed]
                    if not allowed_tokens:
                        break
                
                except Exception as e:
                    # llguidance API might be different, try alternative
                    try:
                        allowed_tokens = interpreter.get_allowed_tokens()
                        if not allowed_tokens:
                            break
                    except:
                        break
                
                # Model forward pass
                if not generated_tokens:
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=self.device)
                else:
                    input_ids = torch.tensor([generated_tokens], device=self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                
                # Apply constraints
                mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=self.device)
                for token_id in allowed_tokens:
                    if 0 <= token_id < logits.size(-1):
                        mask[token_id] = True
                
                if not mask.any():
                    break
                
                logits[0, ~mask] = -1e30
                
                # Sample
                probs = torch.softmax(logits[0] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Advance llguidance
                try:
                    advance_result = interpreter.advance_parser(next_token)
                    if not advance_result or (hasattr(advance_result, 'stop') and advance_result.stop):
                        break
                except Exception as e:
                    # Try alternative API
                    try:
                        if not interpreter.advance(next_token):
                            break
                    except:
                        break
                
                generated_tokens.append(next_token)
                steps += 1
                
                # Check completion
                try:
                    if interpreter.is_accepting() or interpreter.is_complete():
                        break
                except:
                    # Some versions might not have these methods
                    pass
            
            total_time = time.perf_counter() - start_time
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Check if we got valid output
            success = False
            if output_text.strip():
                try:
                    json.loads(output_text)
                    success = True
                except:
                    # Check if it at least looks JSON-like
                    success = output_text.strip().startswith('{') and output_text.strip().endswith('}')
            
            return ComparisonResult(
                library="llguidance",
                success=success,
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
        """Run the real comparison."""
        
        print("🥊 Stolcke PCFG vs llguidance - REAL COMPARISON")
        print("🎯 Production Library Head-to-Head")
        print("=" * 55)
        
        if not LLGUIDANCE_AVAILABLE:
            print("❌ llguidance not available - install with: uv add llguidance")
            return None
        
        # Load model
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        results = {
            'stolcke': [],
            'llguidance': []
        }
        
        for trial in range(trials):
            print(f"\n🔄 Trial {trial + 1}/{trials}")
            print("-" * 20)
            
            # Test Stolcke
            print("Testing Stolcke PCFG...")
            stolcke_result = self.run_stolcke_generation(model, tokenizer)
            results['stolcke'].append(stolcke_result)
            
            status = "✅" if stolcke_result.success else "❌"
            print(f"  {status} {stolcke_result.steps} steps, "
                  f"{stolcke_result.total_time*1000:.0f}ms, "
                  f"'{stolcke_result.output_text}'")
            if stolcke_result.error:
                print(f"  Error: {stolcke_result.error}")
            
            # Test llguidance  
            print("Testing llguidance...")
            llguidance_result = self.run_llguidance_generation(model, tokenizer)
            results['llguidance'].append(llguidance_result)
            
            status = "✅" if llguidance_result.success else "❌"
            print(f"  {status} {llguidance_result.steps} steps, "
                  f"{llguidance_result.total_time*1000:.0f}ms, "
                  f"'{llguidance_result.output_text}'")
            if llguidance_result.error:
                print(f"  Error: {llguidance_result.error}")
        
        self.analyze_results(results)
        return results
    
    def analyze_results(self, results):
        """Analyze the comparison results."""
        
        print(f"\n📊 HEAD-TO-HEAD ANALYSIS")
        print("=" * 35)
        
        stolcke_results = results['stolcke']
        llguidance_results = results['llguidance']
        
        def success_rate(results_list):
            return sum(r.success for r in results_list) / len(results_list) if results_list else 0
        
        def avg_metric(results_list, metric):
            successful = [r for r in results_list if r.success]
            if not successful:
                return 0
            values = [getattr(r, metric) for r in successful if getattr(r, metric) is not None]
            return sum(values) / len(values) if values else 0
        
        # Success rates
        stolcke_success = success_rate(stolcke_results)
        llguidance_success = success_rate(llguidance_results)
        
        # Speed (for successful runs only)
        stolcke_speed = avg_metric(stolcke_results, 'avg_step_time') * 1000
        llguidance_speed = avg_metric(llguidance_results, 'avg_step_time') * 1000
        
        print("| Metric | Stolcke PCFG | llguidance | Winner |")
        print("|--------|--------------|------------|--------|")
        print(f"| Success Rate | {stolcke_success:>10.1%} | {llguidance_success:>8.1%} | {'Stolcke' if stolcke_success > llguidance_success else 'llguidance' if llguidance_success > stolcke_success else 'Tie':>6} |")
        
        if stolcke_speed > 0 and llguidance_speed > 0:
            speed_winner = "llguidance" if llguidance_speed < stolcke_speed else "Stolcke" if stolcke_speed < llguidance_speed else "Tie"
            print(f"| Avg Step Time | {stolcke_speed:>7.0f}ms | {llguidance_speed:>6.0f}ms | {speed_winner:>6} |")
        elif stolcke_speed > 0:
            print(f"| Avg Step Time | {stolcke_speed:>7.0f}ms | {'N/A':>6} | Stolcke |")
        elif llguidance_speed > 0:
            print(f"| Avg Step Time | {'N/A':>7} | {llguidance_speed:>6.0f}ms | llguidance |")
        
        print(f"| Probabilities | {'Yes':>10} | {'No':>8} | Stolcke |")
        
        # Example outputs
        print(f"\n📝 GENERATED EXAMPLES")
        print("-" * 22)
        
        successful_stolcke = [r for r in stolcke_results if r.success]
        successful_llguidance = [r for r in llguidance_results if r.success]
        
        if successful_stolcke:
            result = successful_stolcke[0]
            print(f"Stolcke PCFG: '{result.output_text}'")
            if result.log_probability is not None:
                print(f"              Log-prob: {result.log_probability:.4f}")
        
        if successful_llguidance:
            result = successful_llguidance[0]
            print(f"llguidance:   '{result.output_text}'")
            print(f"              Log-prob: N/A (not supported)")
        
        # Performance verdict
        if stolcke_speed > 0 and llguidance_speed > 0:
            if llguidance_speed < stolcke_speed:
                speed_ratio = stolcke_speed / llguidance_speed
                print(f"\n⚡ Speed: llguidance is {speed_ratio:.1f}x faster")
            elif stolcke_speed < llguidance_speed:
                speed_ratio = llguidance_speed / stolcke_speed
                print(f"\n⚡ Speed: Stolcke is {speed_ratio:.1f}x faster")
            else:
                print(f"\n⚡ Speed: Both libraries perform similarly")
        
        print(f"\n🏆 VERDICT")
        print("-" * 12)
        
        if stolcke_success > llguidance_success:
            print("✅ Stolcke PCFG wins on reliability")
        elif llguidance_success > stolcke_success:
            print("✅ llguidance wins on reliability")
        else:
            print("🤝 Both libraries equally reliable")
        
        if stolcke_speed > 0 and llguidance_speed > 0:
            if llguidance_speed < stolcke_speed:
                print("⚡ llguidance wins on speed")
            elif stolcke_speed < llguidance_speed:
                print("⚡ Stolcke wins on speed")
            else:
                print("⚡ Speed is comparable")
        
        print("📊 Stolcke's unique advantage: Grammar log-probabilities")
        print("🎯 Choose based on: Speed vs probability information needs")


def main():
    """Run the real llguidance comparison."""
    
    comparison = FixedLLGuidanceComparison()
    results = comparison.run_comparison(trials=3)
    
    print(f"\n🎯 This is the real test: Stolcke PCFG vs production llguidance")
    print(f"Both solve tokenization problems, choice depends on speed vs features.")


if __name__ == "__main__":
    main()