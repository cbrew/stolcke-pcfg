#!/usr/bin/env python3
"""
Detailed performance profiler to understand where the 6x overhead comes from.

This breaks down the generation pipeline into components to identify bottlenecks:
1. Model forward pass
2. Adapter constraint calculation  
3. Token sampling
4. State updates
"""

import time
import torch
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM
from stolcke_pcfg import PCFG, StolckeParser, ConstrainedDecoderAdapter
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter


@dataclass
class ProfileData:
    """Performance profiling data for one generation step."""
    step: int
    model_forward_time: float
    constraint_time: float
    sampling_time: float
    state_update_time: float
    total_step_time: float
    allowed_tokens_count: int
    partial_matches_count: int = 0


class PerformanceProfiler:
    """Detailed profiler for generation pipeline components."""
    
    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.profiles: Dict[str, List[ProfileData]] = {}
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing code blocks."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        setattr(self, f"_{name}_time", end - start)
    
    def profile_generation(
        self,
        adapter_type: str,
        model,
        tokenizer,
        adapter,
        grammar_name: str,
        max_tokens: int = 20
    ) -> List[ProfileData]:
        """Profile a complete generation with detailed timing."""
        
        print(f"🔍 Profiling {adapter_type} adapter on {grammar_name}")
        
        generated = torch.empty((1, 0), dtype=torch.long, device=self.device)
        profiles = []
        
        for step in range(max_tokens):
            step_start = time.perf_counter()
            
            # 1. Model Forward Pass
            with self.timer("model_forward"):
                if generated.size(1) == 0:
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=self.device)
                else:
                    input_ids = generated
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
            
            # 2. Constraint Calculation  
            with self.timer("constraint"):
                allowed_tokens = adapter.allowed_token_ids()
                if not allowed_tokens:
                    break
                mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
                logits[0, ~torch.tensor(mask, dtype=torch.bool, device=self.device)] = -1e30
            
            # Get partial matches count for robust adapter
            partial_matches_count = 0
            if hasattr(adapter, 'partial_matches'):
                partial_matches_count = len(adapter.partial_matches)
            elif hasattr(adapter, 'get_current_state_info'):
                state_info = adapter.get_current_state_info()
                partial_matches_count = state_info.get('partial_matches', 0)
            
            # 3. Token Sampling
            with self.timer("sampling"):
                probs = torch.softmax(logits[0] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # 4. State Update
            with self.timer("state_update"):
                if not adapter.step_with_token(next_token):
                    break
                
                if generated.size(1) == 0:
                    generated = torch.tensor([[next_token]], device=self.device)
                else:
                    generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=-1)
            
            step_end = time.perf_counter()
            
            # Record profile data
            profile = ProfileData(
                step=step,
                model_forward_time=self._model_forward_time,
                constraint_time=self._constraint_time,
                sampling_time=self._sampling_time,
                state_update_time=self._state_update_time,
                total_step_time=step_end - step_start,
                allowed_tokens_count=len(allowed_tokens),
                partial_matches_count=partial_matches_count
            )
            profiles.append(profile)
            
            # Check completion
            parser = getattr(adapter, 'parser', getattr(adapter, 'robust_adapter', {}).parser if hasattr(adapter, 'robust_adapter') else None)
            if parser and parser.accepted():
                break
        
        self.profiles[f"{adapter_type}_{grammar_name}"] = profiles
        return profiles
    
    def compare_adapters(self, model_name: str = "gpt2"):
        """Compare performance between naive and robust adapters."""
        
        print("🚀 Performance Profiling Comparison")
        print("=" * 60)
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        # Test grammar
        grammar = PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ["Key", ":", "Value"], 1.0),
            ("Key", ['"email"'], 1.0),
            ("Value", ['"alice@domain.com"'], 1.0),
        ])
        
        results = {}
        
        # Profile both adapters
        for adapter_type in ["naive", "robust"]:
            parser = StolckeParser(grammar, "S")
            
            if adapter_type == "naive":
                def id2str(tid): return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
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
            
            # Profile generation
            profiles = self.profile_generation(
                adapter_type, model, tokenizer, adapter, "email_json", max_tokens=15
            )
            
            if profiles:
                results[adapter_type] = profiles
        
        return results
    
    def analyze_bottlenecks(self, results: Dict[str, List[ProfileData]]):
        """Analyze where the performance differences come from."""
        
        print("\n🔬 PERFORMANCE BOTTLENECK ANALYSIS")
        print("=" * 50)
        
        if "naive" not in results or "robust" not in results:
            print("❌ Need both naive and robust results for comparison")
            return
        
        naive_profiles = results["naive"]
        robust_profiles = results["robust"]
        
        # Compare average times per component
        def avg_component_times(profiles):
            if not profiles:
                return {}
            return {
                'model_forward': sum(p.model_forward_time for p in profiles) / len(profiles),
                'constraint': sum(p.constraint_time for p in profiles) / len(profiles),
                'sampling': sum(p.sampling_time for p in profiles) / len(profiles),
                'state_update': sum(p.state_update_time for p in profiles) / len(profiles),
                'total_step': sum(p.total_step_time for p in profiles) / len(profiles),
            }
        
        naive_avg = avg_component_times(naive_profiles)
        robust_avg = avg_component_times(robust_profiles)
        
        # Print comparison table
        print("\n📊 AVERAGE TIME PER COMPONENT (milliseconds)")
        print("-" * 55)
        print("| Component     | Naive  | Robust | Overhead | % of Total |")
        print("|---------------|--------|--------|----------|------------|")
        
        components = ['model_forward', 'constraint', 'sampling', 'state_update']
        component_names = ['Model Forward', 'Constraints', 'Sampling', 'State Update']
        
        total_naive = naive_avg.get('total_step', 0)
        total_robust = robust_avg.get('total_step', 0)
        
        for comp, name in zip(components, component_names):
            naive_ms = naive_avg.get(comp, 0) * 1000
            robust_ms = robust_avg.get(comp, 0) * 1000
            overhead = (robust_ms / naive_ms) if naive_ms > 0 else float('inf')
            pct_total = (robust_ms / (total_robust * 1000)) * 100 if total_robust > 0 else 0
            
            print(f"| {name:<13} | {naive_ms:6.1f} | {robust_ms:6.1f} | {overhead:8.1f}x | {pct_total:8.1f}% |")
        
        print(f"| {'TOTAL':<13} | {total_naive*1000:6.1f} | {total_robust*1000:6.1f} | {total_robust/total_naive if total_naive > 0 else 0:8.1f}x | {'100.0':>8}% |")
        
        # Analyze constraint computation in detail
        print(f"\n🔍 CONSTRAINT COMPUTATION ANALYSIS")
        print("-" * 40)
        
        if robust_profiles:
            avg_allowed_tokens = sum(p.allowed_tokens_count for p in robust_profiles) / len(robust_profiles)
            avg_partial_matches = sum(p.partial_matches_count for p in robust_profiles) / len(robust_profiles)
            
            print(f"Average allowed tokens per step: {avg_allowed_tokens:.1f}")
            print(f"Average partial matches per step: {avg_partial_matches:.1f}")
            print(f"Constraint time per token: {robust_avg['constraint']*1000/avg_allowed_tokens:.2f}ms" if avg_allowed_tokens > 0 else "N/A")
        
        # Memory analysis
        print(f"\n💾 MEMORY & COMPLEXITY ANALYSIS")
        print("-" * 35)
        
        # Estimate memory usage
        naive_memory = "O(1) - no state tracking"
        robust_memory = f"O(T×L) - {avg_partial_matches:.1f} partial matches avg" if robust_profiles else "N/A"
        
        print(f"Naive adapter memory: {naive_memory}")
        print(f"Robust adapter memory: {robust_memory}")
        
        # Identify the biggest bottleneck
        print(f"\n🎯 BOTTLENECK IDENTIFICATION")
        print("-" * 30)
        
        bottlenecks = []
        for comp, name in zip(components, component_names):
            naive_time = naive_avg.get(comp, 0)
            robust_time = robust_avg.get(comp, 0)
            if naive_time > 0:
                overhead = robust_time / naive_time
                bottlenecks.append((name, overhead, robust_time))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        print("Top bottlenecks (by overhead multiplier):")
        for i, (name, overhead, time_ms) in enumerate(bottlenecks[:3]):
            print(f"{i+1}. {name}: {overhead:.1f}x slower ({time_ms*1000:.1f}ms)")
        
        # Optimization suggestions
        print(f"\n💡 OPTIMIZATION OPPORTUNITIES")
        print("-" * 32)
        
        constraint_overhead = robust_avg['constraint'] / naive_avg['constraint'] if naive_avg.get('constraint', 0) > 0 else 0
        
        if constraint_overhead > 5:
            print("🔴 HIGH: Constraint calculation is the main bottleneck")
            print("   • Cache tokenizations more aggressively")
            print("   • Optimize partial match data structures")
            print("   • Prune impossible matches earlier")
        elif constraint_overhead > 2:
            print("🟡 MEDIUM: Constraint calculation has room for improvement")
            print("   • Consider lazy initialization optimizations")
            print("   • Profile tokenizer encode() calls")
        else:
            print("🟢 LOW: Constraint calculation is reasonably efficient")
        
        model_overhead = robust_avg['model_forward'] / naive_avg['model_forward'] if naive_avg.get('model_forward', 0) > 0 else 1
        if model_overhead > 1.5:
            print("🔴 Unexpected: Model forward pass slower in robust case")
            print("   • May indicate measurement noise or device issues")
        
        return {
            'naive_avg': naive_avg,
            'robust_avg': robust_avg, 
            'bottlenecks': bottlenecks,
            'constraint_overhead': constraint_overhead
        }
    
    def generate_optimization_report(self, analysis_results):
        """Generate recommendations for performance optimization."""
        
        print(f"\n📋 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 40)
        
        constraint_overhead = analysis_results.get('constraint_overhead', 0)
        bottlenecks = analysis_results.get('bottlenecks', [])
        
        print("Immediate opportunities:")
        
        if constraint_overhead > 10:
            print("🔥 CRITICAL: Constraint calculation needs optimization")
            print("   1. Profile tokenizer.encode() calls - may be the real bottleneck")
            print("   2. Implement more aggressive caching of terminal tokenizations") 
            print("   3. Consider batch tokenization for multiple terminals")
            print("   4. Use more efficient partial match data structures")
        elif constraint_overhead > 5:
            print("⚠️  HIGH: Constraint calculation is main bottleneck")
            print("   1. Cache tokenizations at adapter creation time")
            print("   2. Optimize allowed_token_ids() with better data structures")
            print("   3. Early termination in partial match loops")
        elif constraint_overhead > 2:
            print("ℹ️  MEDIUM: Some optimization possible in constraints")
            print("   1. Minor caching improvements")
            print("   2. Algorithmic refinements")
        else:
            print("✅ GOOD: Constraint calculation is reasonably efficient")
        
        print(f"\nLong-term optimizations:")
        print("• Implement C++ backend for critical path operations")
        print("• Use more efficient tensor operations for masking")
        print("• Consider streaming/lazy evaluation for large vocabularies")
        print("• Batch processing multiple generation requests")
        
        print(f"\nReality check:")
        robust_total = analysis_results['robust_avg']['total_step']
        print(f"• Current robust time: {robust_total*1000:.1f}ms per step")
        print(f"• Target: <50ms per step for real-time applications")
        print(f"• Status: {'✅ Acceptable' if robust_total < 0.05 else '⚠️ Could be better' if robust_total < 0.1 else '❌ Too slow'}")


def main():
    """Run complete performance profiling analysis."""
    
    profiler = PerformanceProfiler()
    
    # Run comparison
    results = profiler.compare_adapters()
    
    # Analyze bottlenecks
    analysis = profiler.analyze_bottlenecks(results)
    
    # Generate optimization recommendations
    profiler.generate_optimization_report(analysis)
    
    print(f"\n🏁 Performance profiling complete!")
    print(f"Summary: The 6x overhead is {'primarily' if analysis['constraint_overhead'] > 5 else 'partially'} due to constraint computation")


if __name__ == "__main__":
    main()