# Speed Optimization Roadmap: Making Stolcke PCFG Competitive

## Executive Summary

This document outlines a comprehensive plan to incrementally adopt speed optimizations from llguidance while preserving Stolcke PCFG's unique probability advantages. The goal is to achieve competitive performance (50-200μs per token) through strategic, low-risk improvements.

## Current Performance Gap

```
llguidance:    ~50μs per token  ⚡
Stolcke PCFG:  ~26ms per token  🐌 (520x slower!)
Target:        <200μs per token 🎯 (competitive + probabilistic)
```

## Key Insight: Incremental Hybrid Architecture

Instead of replacing our system, we adopt llguidance's speed techniques while keeping our probability advantages:

- **Keep**: Robust tokenization, 100% success rate, grammar log-probabilities
- **Add**: Token tries, lexer/parser split, slicing optimizations, caching
- **Leverage**: Probability-guided pruning (our unique advantage)

## Phase 1: Quick Wins (1-2 weeks) 🥉

### 1.1 Token Trie Optimization ⭐⭐⭐⭐⭐
**Target Impact**: 5-10x speedup in tokenizer operations
**Effort**: Low - self-contained optimization
**Time**: 3-5 days

```python
class TokenTrie:
    """Organize tokenizer vocabulary in prefix tree for O(log n) lookup"""
    def __init__(self, tokenizer):
        self.root = {}
        self._build_trie(tokenizer.get_vocab())
    
    def find_token_prefixes(self, byte_string):
        """Find all tokens that could start this string - O(log n)"""
        # Implementation details in Phase 1 section
```

**Integration Point**: `RobustConstrainedAdapter.allowed_token_ids()`

### 1.2 Caching & Memoization ⭐⭐⭐⭐
**Target Impact**: 2-5x speedup on repeated patterns  
**Effort**: Low - add caching layer
**Time**: 2-3 days

```python
class CachedStolckeAdapter:
    def __init__(self, base_adapter):
        self.base_adapter = base_adapter
        self._mask_cache = {}
        self._terminal_cache = {}
    
    @lru_cache(maxsize=1000)
    def allowed_token_ids(self):
        # Cache results by parser state hash
```

### 1.3 Batch Operations ⭐⭐⭐
**Target Impact**: 2-3x speedup in constraint checking
**Effort**: Medium - vectorize operations  
**Time**: 3-4 days

```python
def batch_check_terminals(self, terminals, tokens):
    """Vectorized terminal-token compatibility checking"""
    # Matrix operations instead of nested loops
```

**Expected Phase 1 Result**: 26ms → 3-5ms per token

## Phase 2: Architecture Changes (2-3 weeks) 🥈

### 2.1 Lexer/Parser Split ⭐⭐⭐⭐⭐
**Target Impact**: 10-50x speedup (llguidance's biggest win)
**Effort**: High - major architecture change
**Time**: 1-2 weeks

```python
class HybridStolckeLexer:
    """Handle 90% of constraints with fast regex, 10% with Earley"""
    
    def quick_token_check(self, token_str):
        """Fast regex-based checking for simple terminals"""
        # Implementation handles majority of cases
    
    def needs_earley_parser(self, current_state):
        """Only invoke expensive Earley for complex grammar rules"""
        # Detect when full parsing is necessary
```

**Key Innovation**: Preserve probability tracking in the 10% complex cases while accelerating the 90% simple cases.

### 2.2 Incremental Chart Updates ⭐⭐⭐⭐
**Target Impact**: 3-5x speedup in Earley operations
**Effort**: Medium - optimize existing code
**Time**: 1 week

```python
class IncrementalEarleyChart:
    """Update chart incrementally instead of rebuilding"""
    
    def add_token_incremental(self, token, position):
        """Only update chart items affected by new token"""
        # Preserve probability calculations for new items only
```

### 2.3 State Deduplication ⭐⭐⭐
**Target Impact**: 2-3x memory reduction, faster lookups
**Effort**: Medium - add hash consing
**Time**: 3-5 days

**Expected Phase 2 Result**: 3-5ms → 500μs-1ms per token

## Phase 3: Advanced Optimizations (3-4 weeks) 🥇

### 3.1 Slicer-Style Optimization ⭐⭐⭐⭐⭐
**Target Impact**: 10-100x speedup (llguidance's secret weapon)
**Effort**: High - novel technique adaptation
**Time**: 2-3 weeks

```python
class StolckeGrammarSlicer:
    """Pre-compute masks for grammar 'slices' - common sub-patterns"""
    
    def __init__(self, grammar):
        self.slices = self._identify_grammar_slices(grammar)
        self.slice_masks = self._precompute_slice_masks()
    
    def compute_mask_with_slicing(self, parser_state):
        """Try slicing first, fall back to full computation"""
        
        # Fast path: Pre-computed slice (~1μs lookup)
        precomputed = self.get_precomputed_mask(parser_state.context)
        if precomputed:
            return precomputed
        
        # Slow path: Full Earley computation with probabilities (~1ms)
        return self._full_earley_computation(parser_state)
```

### 3.2 Probability-Guided Pruning ⭐⭐⭐⭐⭐
**Target Impact**: 5-20x speedup (our unique advantage!)
**Effort**: Medium - leverage existing probabilities
**Time**: 1 week

```python
class ProbabilityGuidedParser:
    """Use probabilities to prune unlikely parse paths early"""
    
    def prune_unlikely_paths(self, parse_items, threshold=0.01):
        """Remove parse items with very low probability"""
        # Only explore high-probability grammar paths
    
    def smart_beam_search(self, beam_width=10):
        """Keep only top-K most probable parse paths"""
        # Focus computation on likely outcomes
```

**Unique Advantage**: No other system can do probability-guided pruning because they don't track grammar probabilities!

### 3.3 Lazy Grammar Expansion ⭐⭐⭐⭐
**Target Impact**: 5-10x speedup on complex grammars
**Effort**: High - fundamental algorithm change
**Time**: 1-2 weeks

**Expected Phase 3 Result**: 500μs-1ms → 50-200μs per token

## Implementation Priority Matrix

| Technique | Impact | Effort | Priority | Time | Risk |
|-----------|--------|--------|----------|------|------|
| **Token Trie** | ⭐⭐⭐⭐⭐ | Low | 🔥 FIRST | 3-5 days | Low |
| **Caching** | ⭐⭐⭐⭐ | Low | 🔥 FIRST | 2-3 days | Low |
| **Batch Ops** | ⭐⭐⭐ | Medium | 🔥 FIRST | 3-4 days | Low |
| **Lexer Split** | ⭐⭐⭐⭐⭐ | High | 🎯 CORE | 1-2 weeks | Medium |
| **Incremental Charts** | ⭐⭐⭐⭐ | Medium | 🎯 CORE | 1 week | Medium |
| **Slicer System** | ⭐⭐⭐⭐⭐ | High | 🚀 ADVANCED | 2-3 weeks | High |
| **Probability Pruning** | ⭐⭐⭐⭐⭐ | Medium | 🚀 UNIQUE | 1 week | Low |

## Success Metrics & Timeline

### Performance Targets
- **Week 2**: 26ms → 5ms (5x speedup) - Quick wins complete
- **Week 4**: 5ms → 1ms (5x additional) - Architecture changes complete  
- **Week 8**: 1ms → 200μs (5x additional) - Advanced optimizations complete

### Competitive Benchmarks
- **Match llguidance speed**: <200μs average
- **Exceed llguidance features**: + grammar log-probabilities
- **Beat naive approaches**: 100% success rate maintained
- **Unique advantages**: Probability-guided optimization

### Validation Tests
- **JSONSchemaBench**: Run on 10K real-world schemas
- **MaskBench**: Measure mask computation times
- **Tokenization Benchmark**: Maintain 100% success rate
- **Probability Accuracy**: Preserve log-probability correctness

## Risk Mitigation

### Low-Risk Phase 1
- Self-contained optimizations
- No changes to core algorithms
- Easy to rollback if issues arise
- Incremental performance validation

### Medium-Risk Phase 2  
- Major architecture changes
- Extensive testing required
- Staged rollout with fallback options
- Preserve existing API compatibility

### High-Risk Phase 3
- Novel optimization techniques
- Research-level implementations
- A/B testing against existing system
- Performance vs accuracy tradeoffs

## Competitive Positioning Strategy

### Phase 1: **"Competitive Performance"** 
- Fast enough for production use
- Matches Outlines/XGrammar speed class
- Unique: Still has probabilities

### Phase 2: **"Speed + Intelligence"**
- llguidance-class speed with Stolcke intelligence
- Probability-guided optimization unavailable elsewhere
- Production-ready with research capabilities

### Phase 3: **"Speed Leadership"**
- Potentially faster than llguidance due to probability pruning
- Unique hybrid lexer/probabilistic parser architecture
- New state-of-the-art for constrained generation

## Technical Architecture

### Current Stolcke Architecture
```
Input → StolckeParser → EarleyChart → ProbChart → RobustAdapter → TokenMask
        (26ms total)
```

### Target Hybrid Architecture
```
Input → FastLexer (90% cases, <50μs) → TokenMask
      ↘ StolckeParser (10% cases, <1ms) → TokenMask
        
# Best case: 50μs (lexer only)
# Worst case: 1ms (full probabilistic parsing)
# Average: ~200μs (mixed workload)
```

### Integration Points

1. **TokenTrie**: Plugs into `RobustConstrainedAdapter.allowed_token_ids()`
2. **Caching**: Wraps existing parser with `CachedStolckeAdapter`  
3. **Lexer Split**: New `HybridStolckeLexer` routes to existing `StolckeParser`
4. **Incremental**: Modify `EarleyChart` for incremental updates
5. **Slicing**: New `StolckeGrammarSlicer` with precomputed masks
6. **Pruning**: Enhance `ProbChart` with probability-guided beam search

## Development Guidelines

### Code Organization
```
src/stolcke_pcfg/
├── optimizations/
│   ├── token_trie.py          # Phase 1
│   ├── caching.py             # Phase 1  
│   ├── batch_operations.py    # Phase 1
│   ├── lexer_split.py         # Phase 2
│   ├── incremental_charts.py  # Phase 2
│   ├── grammar_slicer.py      # Phase 3
│   └── probability_pruning.py # Phase 3
├── fast_adapter.py            # New optimized adapter
└── hybrid_parser.py           # Orchestrates all optimizations
```

### Testing Strategy
- Unit tests for each optimization
- Integration tests preserving existing behavior
- Performance benchmarks at each phase
- Regression tests for probability accuracy
- Comparative benchmarks against llguidance

### Rollout Strategy
1. **Feature flags**: Enable/disable optimizations individually
2. **A/B testing**: Compare optimized vs original on subsets
3. **Performance monitoring**: Track latency, accuracy, memory usage
4. **Gradual adoption**: Start with simple grammars, expand complexity

## Next Steps

1. **Set up development branch**: `feature/speed-optimizations`
2. **Implement Phase 1 optimizations** in priority order
3. **Create benchmark suite** matching JSONSchemaBench
4. **Establish performance baselines** before and after each change
5. **Document integration points** for easy rollback if needed

## Phase 4: Rust Migration (Optional - If Python Hits Limits) 🦀

### When to Consider Rust Migration
**Trigger Conditions**:
- Python optimizations plateau before reaching <50μs target
- GIL becomes bottleneck in multi-threaded scenarios
- Memory allocations dominate performance profile
- Need to match/exceed llguidance's Rust-native performance

### Migration Strategy: Incremental, Not Big Bang

#### 4.1 Hot Path Extraction ⭐⭐⭐⭐⭐
**Target**: Move only the 10% of code responsible for 90% of runtime
**Effort**: High - but focused on critical bottlenecks
**Time**: 4-6 weeks

```python
# Identify hot paths through profiling
hot_paths = [
    "token_trie_traversal",      # Most frequent operation
    "grammar_slice_matching",    # Computationally intensive  
    "probability_calculations",  # Numerical heavy-lifting
    "chart_update_operations"    # Memory allocation intensive
]

# Rust implementation strategy
"""
1. Create Rust library with Python bindings (PyO3)
2. Implement hot paths in Rust
3. Maintain Python API compatibility
4. Gradual function-by-function replacement
"""
```

#### 4.2 Hybrid Architecture ⭐⭐⭐⭐
**Design**: Python orchestration + Rust hot paths
**Benefit**: Keep development velocity while gaining critical performance

```python
# Python orchestration layer
class HybridStolckeParser:
    def __init__(self, grammar, start_symbol):
        # Python: High-level logic, API, error handling
        self.grammar = grammar
        self.start_symbol = start_symbol
        
        # Rust: Performance-critical operations
        self.rust_engine = stolcke_rust.ParserEngine(grammar, start_symbol)
    
    def allowed_terminals(self):
        # Python: Input validation, result processing
        # Rust: Core computation
        return self.rust_engine.compute_allowed_terminals()
```

#### 4.3 Performance-Critical Components for Rust
```rust
// Priority order for Rust implementation
mod token_trie {
    // Fastest possible tokenizer vocabulary traversal
    // Target: <1μs per lookup vs Python's ~10μs
}

mod grammar_slicer {
    // Precomputed slice matching with SIMD optimizations
    // Target: <5μs slice identification vs Python's ~50μs  
}

mod probability_engine {
    // Vectorized probability calculations
    // Target: <10μs probability updates vs Python's ~100μs
}

mod earley_core {
    // Memory-efficient chart operations
    // Target: Zero-copy operations, custom allocators
}
```

### Rust Migration Phases

#### Phase 4A: Foundation (2 weeks)
```toml
# Cargo.toml - Rust library setup
[package]
name = "stolcke-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
pyo3 = "0.20"              # Python bindings
numpy = "0.20"             # NumPy integration
rayon = "1.8"              # Parallel processing
ahash = "0.8"              # Fast hashing
smallvec = "1.11"          # Stack-allocated vectors

[lib]
name = "stolcke_rust" 
crate-type = ["cdylib"]    # Python extension
```

```python
# Python integration layer
import stolcke_rust  # Rust extension

class RustAcceleratedAdapter(RobustConstrainedAdapter):
    def __init__(self, parser, token_id_to_str, str_to_token_id, tokenizer):
        super().__init__(parser, token_id_to_str, str_to_token_id, tokenizer)
        
        # Hot path: Rust implementation
        self.rust_trie = stolcke_rust.TokenTrie(tokenizer.get_vocab())
        
        # Cold path: Keep Python implementation
        self.python_fallback = True
    
    def allowed_token_ids(self):
        try:
            # Try Rust hot path first
            return self.rust_trie.find_allowed_tokens(self.current_state)
        except Exception:
            # Fallback to Python implementation
            return super().allowed_token_ids()
```

#### Phase 4B: Hot Path Implementation (3-4 weeks)
```rust
// Example: Token trie in Rust
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct TokenTrie {
    root: TrieNode,
    vocab: HashMap<String, u32>,
}

#[pymethods]
impl TokenTrie {
    #[new]
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        let mut trie = TokenTrie {
            root: TrieNode::default(),
            vocab,
        };
        trie.build_trie();
        trie
    }
    
    pub fn find_allowed_tokens(&self, state: &ParserState) -> PyResult<Vec<u32>> {
        // Rust implementation: ~10x faster than Python
        // SIMD optimizations, zero-copy operations
        Ok(self.traverse_with_simd(state))
    }
}
```

#### Phase 4C: Integration & Validation (1-2 weeks)
- Comprehensive benchmarking: Rust vs Python hot paths
- API compatibility testing
- Memory usage profiling 
- Error handling and edge cases

### Expected Rust Performance Gains

#### Conservative Estimates
```python
rust_speedups = {
    "token_trie_lookup": 10,        # 10μs → 1μs
    "grammar_slice_matching": 10,   # 50μs → 5μs  
    "probability_calculations": 5,   # 100μs → 20μs
    "memory_allocations": 3,        # GC pressure → stack allocation
}

# Combined effect on critical path
python_optimized_target = 200     # μs per token
rust_hybrid_target = 50          # μs per token  
rust_native_target = 20          # μs per token (if full migration)
```

#### Best-Case Scenario
- **Rust hybrid**: Match llguidance performance (~50μs) + probabilities
- **Full Rust**: Exceed llguidance performance (~20μs) + probabilities

### Migration Decision Framework

#### Stay with Python If:
- Python optimizations achieve <100μs per token
- Development velocity is more important than peak performance
- Team expertise is primarily Python-based
- Integration complexity outweighs performance gains

#### Migrate to Rust If:
- Python plateaus above 100μs per token after all optimizations
- Need to demonstrate clear technical leadership over llguidance
- Performance is critical for target applications
- Team has Rust expertise or willingness to learn

### Rust Migration Milestones
```python
decision_points = {
    "week_8_python_results": {
        "if_performance": ">100μs per token",
        "decision": "Begin Rust foundation work",
        "timeline": "4 weeks to hybrid system"
    },
    
    "week_12_hybrid_results": {
        "if_performance": "<50μs per token", 
        "decision": "Success - hybrid approach sufficient",
        "timeline": "Continue with hybrid architecture"
    },
    
    "week_16_evaluation": {
        "if_market_pressure": "High competitive pressure",
        "decision": "Consider full Rust migration",
        "timeline": "8-12 weeks for complete system"
    }
}
```

## Long-term Vision

**Goal**: Make Stolcke PCFG the fastest AND most capable grammar-constrained generation system available.

**Technology Strategy**: 
- **Python-first**: Maximize development velocity and maintainability
- **Rust-when-needed**: Adopt only for performance-critical bottlenecks
- **Hybrid advantage**: Best of both worlds - Python flexibility + Rust performance

**Unique Position**: The only system that combines:
- llguidance-level speed (20-50μs per token via Python+Rust hybrid)
- Full grammar log-probabilities
- 100% tokenization robustness  
- Probability-guided optimizations
- Maintainable Python API with Rust performance core

**Market Impact**: Enable new applications requiring both speed and probabilistic reasoning in constrained generation scenarios while maintaining the development agility that comes with a Python-based system.

### Final Architecture Vision
```python
# The ultimate hybrid system
class OptimalStolckeParser:
    """
    Python API and orchestration layer for developer productivity
    Rust performance engine for computational hot paths
    Best-in-class speed WITH unique probability features
    """
    
    def __init__(self, grammar, start_symbol):
        # Python: Grammar loading, validation, API
        self.python_layer = StolckeParserAPI(grammar, start_symbol)
        
        # Rust: Performance-critical operations
        self.rust_engine = stolcke_rust.PerformanceEngine(grammar)
        
        # Automatic hot path detection and routing
        self.profiler = AdaptiveProfiler()
    
    def parse(self, tokens):
        # Route to optimal implementation based on profiling data
        if self.profiler.should_use_rust(len(tokens), grammar_complexity):
            return self.rust_engine.parse_fast(tokens)
        else:
            return self.python_layer.parse_with_full_features(tokens)
```

---

*This roadmap provides a concrete, low-risk path from Python optimization through optional Rust acceleration, ensuring we can achieve competitive performance while preserving development velocity and our unique advantages in the grammar-constrained generation landscape.*