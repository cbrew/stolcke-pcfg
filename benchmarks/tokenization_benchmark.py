#!/usr/bin/env python3
"""
Comprehensive benchmark comparing naive vs robust tokenization approaches.

This benchmark measures:
1. Success rates (how often generation completes successfully)
2. Convergence time (how many tokens to reach accepted state)  
3. Performance overhead (time per generation step)
4. Grammar coverage (how many terminals actually work)
5. Cross-tokenizer robustness (consistency across different tokenizers)
"""

import json
import time
import torch
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import statistics

from transformers import AutoTokenizer, AutoModelForCausalLM

from stolcke_pcfg import PCFG, StolckeParser, ConstrainedDecoderAdapter
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    adapter_type: str
    grammar_name: str
    tokenizer_name: str
    success: bool
    tokens_generated: int
    time_taken: float
    parser_accepted: bool
    valid_output: bool
    output_text: str
    error: Optional[str] = None


@dataclass  
class BenchmarkSummary:
    """Aggregated benchmark results."""
    adapter_type: str
    total_runs: int
    success_rate: float
    avg_tokens: float
    avg_time: float
    grammar_coverage: float
    tokenizer_robustness: float


class TokenizationBenchmark:
    """Main benchmark suite for comparing tokenization approaches."""
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        self.device = device
        self.results: List[BenchmarkResult] = []
        
    def create_test_grammars(self) -> Dict[str, PCFG]:
        """Create test grammars with varying tokenization complexity."""
        
        grammars = {}
        
        # 1. Simple single-token grammar (baseline)
        grammars["simple_numbers"] = PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ['"value"', ":", "Number"], 1.0),
            ("Number", ["42"], 0.33),
            ("Number", ["25"], 0.33),
            ("Number", ["100"], 0.34),
        ])
        
        # 2. Mixed complexity - some single, some multi-token
        grammars["mixed_complexity"] = PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ["Key", ":", "Value"], 1.0),
            ("Key", ['"id"'], 0.5),          # Multi-token: ['"', 'id', '"']
            ("Key", ['"name"'], 0.5),        # Multi-token: ['"', 'name', '"']
            ("Value", ["123"], 0.5),         # Single token
            ("Value", ['"Alice"'], 0.5),     # Multi-token: ['"', 'Alice', '"']
        ])
        
        # 3. High complexity - all multi-token strings
        grammars["high_complexity"] = PCFG([
            ("S", ["{", "Pair", "}"], 1.0),
            ("Pair", ["Key", ":", "Value"], 1.0),
            ("Key", ['"email_address"'], 0.5),     # Multi-token
            ("Key", ['"phone_number"'], 0.5),      # Multi-token  
            ("Value", ['"alice@domain.com"'], 0.33), # Very multi-token (8 tokens)
            ("Value", ['"bob@company.org"'], 0.33),  # Very multi-token
            ("Value", ['"555-123-4567"'], 0.34),     # Multi-token phone
        ])
        
        # 4. Extreme complexity - very long multi-token sequences
        grammars["extreme_complexity"] = PCFG([
            ("S", ["Sentence"], 1.0),
            ("Sentence", ['"', "Content", '"'], 1.0),
            ("Content", ['"Hello, this is a very long sentence with multiple words and punctuation!"'], 0.5),
            ("Content", ['"The quick brown fox jumps over the lazy dog in the countryside."'], 0.5),
        ])
        
        # 5. Realistic JSON schema
        grammars["realistic_json"] = PCFG([
            ("S", ["{", "PersonData", "}"], 1.0),
            ("PersonData", ["NameField", ",", "EmailField", ",", "AgeField"], 1.0),
            ("NameField", ['"full_name"', ":", '"Alice Johnson"'], 1.0),
            ("EmailField", ['"email_address"', ":", '"alice.johnson@company.com"'], 1.0),
            ("AgeField", ['"age_years"', ":", "28"], 1.0),
        ])
        
        return grammars
    
    def create_naive_adapter(self, parser, tokenizer) -> ConstrainedDecoderAdapter:
        """Create naive adapter that only works with single-token terminals."""
        
        def id2str(token_id: int) -> str:
            return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        
        def str2id(s: str) -> int | None:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        
        def single_token_filter(terms):
            """Only allow single-token terminals."""
            allowed = set()
            for term in terms:
                ids = tokenizer.encode(term, add_special_tokens=False)
                if len(ids) == 1:  # Only single tokens
                    allowed.add(ids[0])
            return allowed
        
        return ConstrainedDecoderAdapter(
            parser, id2str, str2id, next_token_filter=single_token_filter
        )
    
    def create_robust_adapter(self, parser, tokenizer) -> RobustConstrainedAdapter:
        """Create robust adapter that handles multi-token terminals."""
        
        def id2str(token_id: int) -> str:
            return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        
        def str2id(s: str) -> int | None:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        
        return RobustConstrainedAdapter(
            parser=parser,
            token_id_to_str=id2str,
            str_to_token_id=str2id,
            tokenizer=tokenizer
        )
    
    def run_single_generation(
        self,
        model,
        tokenizer,
        adapter,
        adapter_type: str,
        grammar_name: str,
        max_tokens: int = 50,
        timeout: float = 30.0
    ) -> BenchmarkResult:
        """Run a single constrained generation and measure results."""
        
        start_time = time.time()
        generated_tokens = []
        error = None
        
        try:
            generated = torch.empty((1, 0), dtype=torch.long, device=self.device)
            
            for step in range(max_tokens):
                # Check timeout
                if time.time() - start_time > timeout:
                    error = "Timeout"
                    break
                
                # Prepare input
                if generated.size(1) == 0:
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=self.device)
                else:
                    input_ids = generated
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                
                # Get constraints
                allowed_tokens = adapter.allowed_token_ids()
                if not allowed_tokens:
                    error = "No allowed tokens"
                    break
                
                # Apply constraints
                mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
                constrained_logits = logits.clone()
                constrained_logits[0, ~torch.tensor(mask, dtype=torch.bool, device=self.device)] = -1e30
                
                # Sample token (use sampling to avoid loops)
                probs = torch.softmax(constrained_logits[0] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Advance adapter
                if not adapter.step_with_token(next_token):
                    error = "Token rejected by adapter"
                    break
                
                # Update sequence
                if generated.size(1) == 0:
                    generated = torch.tensor([[next_token]], device=self.device)
                else:
                    generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=-1)
                
                generated_tokens.append(next_token)
                
                # Check completion
                if hasattr(adapter, 'parser'):
                    parser_accepted = adapter.parser.accepted()
                else:
                    parser_accepted = adapter.robust_adapter.parser.accepted() if hasattr(adapter, 'robust_adapter') else False
                
                if parser_accepted:
                    break
            else:
                error = "Max tokens reached"
                
        except Exception as e:
            error = str(e)
            
        end_time = time.time()
        
        # Generate output text
        if generated_tokens:
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = ""
        
        # Check parser acceptance
        try:
            if hasattr(adapter, 'parser'):
                parser_accepted = adapter.parser.accepted()
            else:
                parser_accepted = adapter.robust_adapter.parser.accepted() if hasattr(adapter, 'robust_adapter') else False
        except:
            parser_accepted = False
        
        # Validate output (try to parse as JSON for JSON grammars)
        valid_output = False
        if output_text and not error:
            if grammar_name.endswith('json') or '{' in output_text:
                try:
                    json.loads(output_text)
                    valid_output = True
                except:
                    pass
            else:
                valid_output = len(output_text.strip()) > 0
        
        return BenchmarkResult(
            adapter_type=adapter_type,
            grammar_name=grammar_name,
            tokenizer_name=tokenizer.name_or_path.split('/')[-1],
            success=(error is None and parser_accepted),
            tokens_generated=len(generated_tokens),
            time_taken=end_time - start_time,
            parser_accepted=parser_accepted,
            valid_output=valid_output,
            output_text=output_text,
            error=error
        )
    
    def run_benchmark_suite(
        self,
        model_name: str = "gpt2",
        tokenizer_names: List[str] = ["gpt2"],
        trials_per_config: int = 5
    ) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        
        print(f"🚀 Starting Tokenization Benchmark Suite")
        print(f"Model: {model_name}")
        print(f"Tokenizers: {tokenizer_names}")  
        print(f"Trials per config: {trials_per_config}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Load model once
        print(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        print("✅ Model loaded")
        
        grammars = self.create_test_grammars()
        results = []
        
        total_configs = len(tokenizer_names) * len(grammars) * 2 * trials_per_config
        config_count = 0
        
        for tokenizer_name in tokenizer_names:
            print(f"\n📝 Testing tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            for grammar_name, grammar in grammars.items():
                print(f"  📊 Grammar: {grammar_name}")
                
                # Test both adapters
                for adapter_type in ["naive", "robust"]:
                    print(f"    🔧 Adapter: {adapter_type}")
                    
                    for trial in range(trials_per_config):
                        config_count += 1
                        progress = (config_count / total_configs) * 100
                        print(f"      Trial {trial + 1}/{trials_per_config} ({progress:.1f}% overall)")
                        
                        # Create fresh parser and adapter for each trial
                        parser = StolckeParser(grammar, "S")
                        
                        if adapter_type == "naive":
                            adapter = self.create_naive_adapter(parser, tokenizer)
                        else:
                            adapter = self.create_robust_adapter(parser, tokenizer)
                        
                        # Run generation
                        result = self.run_single_generation(
                            model, tokenizer, adapter, adapter_type, grammar_name,
                            max_tokens=30, timeout=15.0
                        )
                        
                        results.append(result)
                        
                        # Quick status
                        status = "✅" if result.success else "❌"
                        print(f"        {status} {result.tokens_generated} tokens, "
                              f"{result.time_taken:.2f}s, '{result.output_text[:30]}...'")
        
        self.results.extend(results)
        print(f"\n🏁 Benchmark complete! {len(results)} total runs")
        return results
    
    def analyze_results(self) -> Dict[str, BenchmarkSummary]:
        """Analyze benchmark results and create summary statistics."""
        
        if not self.results:
            return {}
        
        # Group results by adapter type
        grouped = defaultdict(list)
        for result in self.results:
            grouped[result.adapter_type].append(result)
        
        summaries = {}
        
        for adapter_type, results in grouped.items():
            # Calculate metrics
            total_runs = len(results)
            successes = [r for r in results if r.success]
            success_rate = len(successes) / total_runs if total_runs > 0 else 0
            
            # Average metrics (only for successful runs)
            if successes:
                avg_tokens = statistics.mean(r.tokens_generated for r in successes)
                avg_time = statistics.mean(r.time_taken for r in successes)
            else:
                avg_tokens = 0
                avg_time = 0
            
            # Grammar coverage (how many different grammars succeeded)
            successful_grammars = set(r.grammar_name for r in successes)
            total_grammars = set(r.grammar_name for r in results)
            grammar_coverage = len(successful_grammars) / len(total_grammars) if total_grammars else 0
            
            # Tokenizer robustness (consistency across tokenizers)
            tokenizer_success_rates = defaultdict(list)
            for result in results:
                tokenizer_success_rates[result.tokenizer_name].append(result.success)
            
            tokenizer_rates = []
            for tokenizer, successes_list in tokenizer_success_rates.items():
                rate = sum(successes_list) / len(successes_list) if successes_list else 0
                tokenizer_rates.append(rate)
            
            tokenizer_robustness = 1.0 - statistics.stdev(tokenizer_rates) if len(tokenizer_rates) > 1 else 1.0
            
            summaries[adapter_type] = BenchmarkSummary(
                adapter_type=adapter_type,
                total_runs=total_runs,
                success_rate=success_rate,
                avg_tokens=avg_tokens,
                avg_time=avg_time,
                grammar_coverage=grammar_coverage,
                tokenizer_robustness=tokenizer_robustness
            )
        
        return summaries
    
    def generate_report(self, output_file: str = "benchmark_report.md"):
        """Generate comprehensive benchmark report."""
        
        summaries = self.analyze_results()
        
        if not summaries:
            print("No results to analyze!")
            return
        
        report_lines = [
            "# Tokenization Benchmark Report",
            "",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total runs**: {len(self.results)}",
            f"**Device**: {self.device}",
            "",
            "## Executive Summary",
            "",
        ]
        
        # High-level comparison
        if "naive" in summaries and "robust" in summaries:
            naive = summaries["naive"] 
            robust = summaries["robust"]
            
            improvement = {
                "success_rate": robust.success_rate - naive.success_rate,
                "grammar_coverage": robust.grammar_coverage - naive.grammar_coverage,
                "tokenizer_robustness": robust.tokenizer_robustness - naive.tokenizer_robustness,
            }
            
            report_lines.extend([
                f"- **Success Rate**: Robust {robust.success_rate:.1%} vs Naive {naive.success_rate:.1%} "
                f"({improvement['success_rate']:+.1%} improvement)",
                f"- **Grammar Coverage**: Robust {robust.grammar_coverage:.1%} vs Naive {naive.grammar_coverage:.1%} "
                f"({improvement['grammar_coverage']:+.1%} improvement)",
                f"- **Tokenizer Robustness**: Robust {robust.tokenizer_robustness:.3f} vs Naive {naive.tokenizer_robustness:.3f}",
                f"- **Performance Overhead**: Robust {robust.avg_time:.3f}s vs Naive {naive.avg_time:.3f}s per generation",
                "",
            ])
        
        # Detailed results by adapter
        report_lines.extend([
            "## Detailed Results",
            "",
            "| Adapter | Success Rate | Avg Tokens | Avg Time | Grammar Coverage | Tokenizer Robustness |",
            "|---------|-------------|------------|----------|------------------|---------------------|",
        ])
        
        for adapter_type, summary in summaries.items():
            report_lines.append(
                f"| {adapter_type.capitalize()} | {summary.success_rate:.1%} | "
                f"{summary.avg_tokens:.1f} | {summary.avg_time:.3f}s | "
                f"{summary.grammar_coverage:.1%} | {summary.tokenizer_robustness:.3f} |"
            )
        
        # Results by grammar complexity
        report_lines.extend([
            "",
            "## Results by Grammar Complexity",
            "",
        ])
        
        complexity_results = defaultdict(lambda: defaultdict(list))
        for result in self.results:
            complexity_results[result.grammar_name][result.adapter_type].append(result)
        
        for grammar_name in sorted(complexity_results.keys()):
            report_lines.extend([
                f"### {grammar_name.replace('_', ' ').title()}",
                "",
            ])
            
            for adapter_type in ["naive", "robust"]:
                if adapter_type in complexity_results[grammar_name]:
                    results = complexity_results[grammar_name][adapter_type]
                    successes = [r for r in results if r.success]
                    success_rate = len(successes) / len(results) if results else 0
                    
                    report_lines.append(f"- **{adapter_type.capitalize()}**: {success_rate:.1%} success rate ({len(successes)}/{len(results)})")
                    
                    if successes:
                        avg_tokens = statistics.mean(r.tokens_generated for r in successes)
                        example = successes[0].output_text[:50] + "..." if len(successes[0].output_text) > 50 else successes[0].output_text
                        report_lines.append(f"  - Avg tokens: {avg_tokens:.1f}")
                        report_lines.append(f"  - Example output: `{example}`")
            
            report_lines.append("")
        
        # Failed cases analysis
        failures = [r for r in self.results if not r.success]
        if failures:
            failure_reasons = defaultdict(int)
            for failure in failures:
                reason = failure.error or "Unknown"
                failure_reasons[reason] += 1
            
            report_lines.extend([
                "## Failure Analysis",
                "",
                "| Failure Reason | Count | Percentage |",
                "|---------------|-------|------------|",
            ])
            
            total_failures = len(failures)
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_failures) * 100
                report_lines.append(f"| {reason} | {count} | {pct:.1f}% |")
        
        # Write report
        report_content = "\n".join(report_lines)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"📊 Benchmark report written to: {output_file}")
        
        # Also print key metrics to console
        print("\n📈 Key Results:")
        for adapter_type, summary in summaries.items():
            print(f"  {adapter_type.capitalize()}: {summary.success_rate:.1%} success rate, "
                  f"{summary.avg_tokens:.1f} avg tokens, {summary.avg_time:.3f}s avg time")


def main():
    """Run the complete benchmark suite."""
    
    benchmark = TokenizationBenchmark()
    
    # Run benchmarks
    results = benchmark.run_benchmark_suite(
        model_name="gpt2",
        tokenizer_names=["gpt2"],  # Could add others: ["gpt2", "microsoft/DialoGPT-small"]
        trials_per_config=3  # Reduce for faster testing, increase for more reliable results
    )
    
    # Generate report
    benchmark.generate_report("benchmarks/tokenization_benchmark_report.md")
    
    print("\n🎉 Benchmark suite completed!")


if __name__ == "__main__":
    main()