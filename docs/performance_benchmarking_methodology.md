# Performance Benchmarking Methodology

## Statistical Requirements for Valid Speed Comparisons

### Executive Summary
To make valid performance claims about speed optimizations, we need statistically rigorous benchmarking with sufficient sample sizes, controlled conditions, and proper statistical analysis.

## Sample Size Requirements

### For Speed Measurements
**Minimum sample size**: 1,000 measurements per benchmark scenario
**Recommended**: 10,000 measurements for high-confidence claims
**Gold standard**: 100,000 measurements for publication-quality results

### Statistical Justification
```python
# Central Limit Theorem requirements
# For valid confidence intervals on mean latency:
n_min = 30   # Absolute minimum for CLT to apply
n_good = 100  # Good for basic comparisons  
n_robust = 1000  # Robust to outliers and distribution shape
n_rigorous = 10000  # High confidence, tight intervals
```

### Confidence Intervals
With 10,000 samples:
- **95% confidence**: Mean ± 1.96 × (std/√n)
- **99% confidence**: Mean ± 2.58 × (std/√n)
- **Margin of error**: Typically <1% of mean for large n

## Benchmark Categories & Required Samples

### 1. Micro-benchmarks (Individual Operations)
```python
operations = [
    "token_trie_lookup",
    "allowed_terminals_computation", 
    "chart_update_single_token",
    "probability_calculation",
    "mask_generation"
]

# Per operation: 10,000 measurements
# Total: 50,000 micro-benchmark measurements
```

### 2. Grammar Complexity Categories
```python
grammar_types = {
    "simple_json": {"properties": 1-3, "nesting": 0-1},
    "medium_json": {"properties": 4-10, "nesting": 1-3}, 
    "complex_json": {"properties": 10+, "nesting": 3+},
    "recursive_grammar": {"left_recursion": True},
    "regex_heavy": {"terminal_patterns": 20+}
}

# Per category: 5,000 full generation runs
# Total: 25,000 end-to-end measurements
```

### 3. Token Length Categories
```python
token_length_buckets = {
    "short": "1-5 tokens",
    "medium": "6-15 tokens", 
    "long": "16-50 tokens",
    "very_long": "50+ tokens"
}

# Per bucket: 2,500 measurements
# Total: 10,000 length-stratified measurements  
```

### 4. Comparison Baselines
```python
systems_to_compare = [
    "stolcke_baseline",     # Current implementation
    "stolcke_optimized",    # After each optimization phase
    "llguidance",          # External competition
    "naive_approach"       # Baseline for success rate
]

# Cross-product with all test categories
# Total comparative measurements: ~100,000
```

## Total Benchmark Suite Size

### Conservative Estimate
- **Micro-benchmarks**: 50,000 measurements
- **End-to-end tests**: 25,000 measurements  
- **Length stratification**: 10,000 measurements
- **Comparative baselines**: 100,000 measurements
- **Regression testing**: 15,000 measurements

**Total: ~200,000 performance measurements**

### Measurement Time Estimates
```python
# Current Stolcke performance: ~26ms per measurement
baseline_time = 200_000 * 26e-3  # 5,200 seconds = 87 minutes

# Optimized performance target: ~200μs per measurement  
optimized_time = 200_000 * 200e-6  # 40 seconds

# Development iteration time (during optimization):
# Mix of baseline + partially optimized
iteration_time = 200_000 * 5e-3  # ~17 minutes per full benchmark run
```

## Benchmark Infrastructure Requirements

### 1. Automated Test Suite
```python
class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking framework"""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.measurements = []
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def run_full_benchmark(self, system_under_test):
        """Run all 200K benchmark measurements"""
        results = {}
        
        # Micro-benchmarks
        results['micro'] = self._run_micro_benchmarks(system_under_test)
        
        # End-to-end tests by grammar complexity
        results['e2e'] = self._run_e2e_benchmarks(system_under_test)
        
        # Length-stratified tests
        results['length'] = self._run_length_benchmarks(system_under_test)
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results):
        """Statistical analysis with confidence intervals"""
        return {
            'mean_latency': self._compute_confidence_interval(results),
            'percentiles': self._compute_percentiles([50, 90, 95, 99]),
            'regression_analysis': self._detect_performance_regressions(),
            'comparative_analysis': self._compare_against_baselines()
        }
```

### 2. Test Case Generation
```python
def generate_benchmark_test_cases():
    """Generate statistically representative test cases"""
    
    # JSON Schema Bench style - real-world complexity distribution
    json_schemas = load_jsonschemabench_subset(n=10000)
    
    # Grammar complexity stratification
    simple_grammars = generate_simple_pcfg_grammars(n=5000)
    complex_grammars = generate_complex_pcfg_grammars(n=5000) 
    recursive_grammars = generate_recursive_grammars(n=2500)
    
    # Token sequence length distribution
    # Based on real-world JSON generation patterns
    length_distribution = {
        1: 0.05,   # Very short
        5: 0.25,   # Short  
        15: 0.40,  # Medium (most common)
        30: 0.25,  # Long
        50: 0.05   # Very long
    }
    
    return stratified_sample(all_cases, distribution=length_distribution)
```

### 3. Statistical Analysis Framework
```python
class StatisticalAnalyzer:
    """Rigorous statistical analysis of performance results"""
    
    def compute_confidence_intervals(self, measurements, confidence=0.95):
        """Compute confidence intervals for mean latency"""
        import scipy.stats as stats
        
        mean = np.mean(measurements)
        std_err = stats.sem(measurements)
        interval = stats.t.interval(
            confidence, 
            len(measurements)-1, 
            loc=mean, 
            scale=std_err
        )
        
        return {
            'mean': mean,
            'confidence_interval': interval,
            'margin_of_error': (interval[1] - interval[0]) / 2,
            'relative_error': ((interval[1] - interval[0]) / 2) / mean
        }
    
    def detect_significant_difference(self, baseline, optimized, alpha=0.05):
        """Statistical hypothesis testing for performance improvements"""
        from scipy.stats import ttest_ind
        
        # Two-sample t-test
        statistic, p_value = ttest_ind(baseline, optimized)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.var(baseline) + np.var(optimized)) / 2))
        cohens_d = (np.mean(optimized) - np.mean(baseline)) / pooled_std
        
        return {
            'statistically_significant': p_value < alpha,
            'p_value': p_value,
            'effect_size': cohens_d,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, d):
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small" 
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
```

## Benchmark Execution Strategy

### Phase-by-Phase Validation
```python
phases = {
    "baseline": {
        "measurements": 50000,
        "purpose": "Establish current performance characteristics",
        "time_estimate": "22 minutes"
    },
    "phase_1_optimizations": {
        "measurements": 50000, 
        "purpose": "Validate token trie, caching, batch ops improvements",
        "time_estimate": "4 minutes"
    },
    "phase_2_architecture": {
        "measurements": 50000,
        "purpose": "Validate lexer split, incremental charts", 
        "time_estimate": "50 seconds"
    },
    "phase_3_advanced": {
        "measurements": 50000,
        "purpose": "Validate slicing, probability pruning",
        "time_estimate": "10 seconds"
    }
}

total_development_measurements = 200000
total_development_time = "~27 minutes per complete validation cycle"
```

### Continuous Integration Benchmarks
```python
# Lighter benchmark suite for CI/CD
ci_benchmark_size = 10000  # 10K measurements
ci_runtime_target = "< 2 minutes"  # Even with current slow baseline

# Regression detection threshold
performance_regression_threshold = 1.10  # 10% slowdown triggers alert
```

## Quality Control Measures

### 1. Environmental Controls
- **Hardware consistency**: Same machine, same CPU cores, same memory
- **System load**: Run during low system activity, disable background processes
- **Temperature**: Monitor CPU temperature, pause if thermal throttling detected
- **JIT warmup**: Include warmup runs, exclude from measurements

### 2. Outlier Detection & Handling
```python
def remove_outliers(measurements, method='iqr'):
    """Remove statistical outliers from measurement set"""
    if method == 'iqr':
        Q1 = np.percentile(measurements, 25)
        Q3 = np.percentile(measurements, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return measurements[
            (measurements >= lower_bound) & 
            (measurements <= upper_bound)
        ]
```

### 3. Measurement Precision
- **Timing method**: Use `time.perf_counter()` for high precision
- **Resolution**: Nanosecond precision where available
- **Multiple runs**: Average over multiple runs per test case
- **Garbage collection**: Control GC timing to avoid measurement artifacts

## Success Criteria & Claims

### Statistically Valid Claims
With 200K measurements, we can make these claims with high confidence:

#### Speed Improvement Claims
```python
# Example: "5x speedup with 99% confidence"
baseline_mean = 26000  # μs
optimized_mean = 5200  # μs  
speedup_ratio = baseline_mean / optimized_mean  # 5.0x

# With n=50K per measurement, margin of error ~0.1%
# Can claim: "5.0x ± 0.1x speedup (99% confidence)"
```

#### Comparative Performance Claims
```python
# "Faster than llguidance on X% of test cases"
# "Within Y% of llguidance performance on average"
# "Z% faster than naive approach"

# All with statistical significance testing
```

### Publication-Quality Results
- **Sample size**: 200K measurements total
- **Statistical power**: >99% to detect 10% performance differences
- **Confidence intervals**: 99% confidence, <1% margin of error
- **Multiple comparisons**: Bonferroni correction for multiple testing
- **Effect sizes**: Cohen's d for practical significance

## Implementation Timeline

### Week 1: Benchmark Infrastructure
- Set up automated benchmark suite
- Implement statistical analysis framework  
- Generate 200K test cases
- Validate measurement precision

### Week 2: Baseline Measurements  
- Run comprehensive baseline benchmarks
- Establish performance characteristics
- Identify high-variance scenarios
- Tune measurement methodology

### Weeks 3-8: Optimization + Validation
- After each optimization: run targeted benchmarks
- Weekly: run subset of full benchmark suite
- End of each phase: run complete 200K measurement validation

### Final Validation
- Complete 200K measurement suite on final optimized system
- Comparative benchmarking against llguidance
- Statistical analysis and claims validation
- Performance regression testing setup

## Bottom Line: Required Sample Size

**Minimum for valid claims**: 10,000 measurements per comparison
**Recommended for robust claims**: 50,000 measurements per comparison  
**Gold standard for publication**: 200,000 total measurements across all test categories

**Runtime estimates**:
- Current system: ~87 minutes for full benchmark
- Optimized system: ~40 seconds for full benchmark
- Development iterations: ~17 minutes per validation cycle

This gives us the statistical power to make confident, defensible claims about performance improvements while maintaining reasonable development iteration times.