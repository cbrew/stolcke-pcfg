# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Stolcke PCFG Parser, a Python implementation of the probabilistic Earley parser with prefix probabilities and LLM constrained decoding adapter. The package provides normalized PCFG parsing, prefix probability computation for incremental decoding, and integration with Hugging Face transformers for constrained text generation.

## Development Commands

### Environment Setup
- `make setup` - Create venv and install dependencies with pre-commit hooks
- `uv venv && uv sync` - Direct alternative to setup
- `make hooks` - Install pre-commit hooks only

### Common Development Tasks
- `make test` - Run pytest test suite
- `make lint` - Run Ruff linting
- `make fmt` - Format code with Ruff
- `make coverage` - Run tests with coverage report
- `make precommit` - Run all pre-commit hooks locally
- `make run` - Run CLI demo with default grammar
- `uv run stolcke-parser a a a` - Run CLI with specific tokens
- `uv run stolcke-parser --no-unit-elim a a a` - Run without unit elimination

### Documentation
- `make docs-serve` - Serve docs locally with MkDocs
- `make docs-build` - Build static docs site
- `make docs-deploy` - Deploy to GitHub Pages

### Testing Specific Components
- `uv run pytest tests/test_smoke.py` - Run smoke tests
- `uv run pytest tests/test_hf_logits.py` - Test HuggingFace integration
- `uv run pytest tests/test_json_constraints.py` - Test JSON constraint functionality

## Architecture Overview

### Core Modules
- **grammar.py**: `PCFG` class and `Rule` data structures for probabilistic context-free grammars
- **earley_core.py**: `EarleyItem` and `EarleyChart` implementing the core Earley algorithm
- **probabilities.py**: `ProbChart` extending charts with forward (alpha) and inner (gamma) probabilities
- **stolcke_parser.py**: `StolckeParser` orchestrating SCAN/PREDICT/COMPLETE operations with prefix probability tracking
- **inside.py**: Span-based inside dynamic programming for exact sentence log-probability computation
- **constrained_adapter.py**: `ConstrainedDecoderAdapter` mapping allowed terminals to token IDs for LLM integration
- **hf_logits.py**: `GrammarConstrainedLogitsProcessor` for Hugging Face transformers integration
- **transform.py**: Unit production elimination via matrix-based closure

### Key Implementation Details
- Uses log-space probabilities throughout to avoid underflow
- Supports unit productions via elimination transform before parsing
- Does NOT support epsilon (empty) productions
- Handles left recursion correctly
- Provides exact prefix probabilities for incremental decoding

## Code Style and Conventions

- **Formatting**: Managed by Ruff with 100-character line length
- **Naming**: modules `lower_snake_case`, classes `PascalCase`, functions/variables `lower_snake_case`
- **Imports**: `stolcke_pcfg` is first-party package
- **Testing**: pytest with test files mirroring src/ structure
- **Commits**: Use Conventional Commits format (e.g., `feat(parser): add tokenizer`)

## Dependencies and Environment

- **Python**: Requires >=3.12
- **Package Manager**: uv for dependency management
- **Core Dependencies**: numpy>=1.26, torch>=2.8.0, transformers>=4.55.0
- **Dev Tools**: ruff, pytest, mkdocs, pre-commit

## Testing Notes

- Framework uses pytest with deterministic, isolated tests
- For probability checks, compare in log-space to avoid float equality issues
- Test fixtures should be placed in `tests/fixtures/`
- Parser rejects epsilon productions - design tests accordingly
- Coverage reports available via `make coverage`

## Performance Optimization Plan

**Status**: Ready for implementation (see `docs/speed_optimization_roadmap.md`)

**Goal**: Achieve competitive performance (50-200μs per token) while preserving probability advantages.

**Current Gap**: 520x slower than llguidance (~26ms vs ~50μs per token)

### Implementation Phases
1. **Phase 1 (Quick Wins)**: Token trie optimization, caching, batch operations
   - Target: 26ms → 3-5ms per token
   - Time: 1-2 weeks, low risk
   
2. **Phase 2 (Architecture Changes)**: Lexer/parser split, incremental charts, state deduplication
   - Target: 3-5ms → 500μs-1ms per token  
   - Time: 2-3 weeks, medium risk
   
3. **Phase 3 (Advanced)**: Grammar slicing, probability-guided pruning, lazy expansion
   - Target: 500μs-1ms → 50-200μs per token
   - Time: 3-4 weeks, high impact

### Key Strategy
- **Incremental adoption** of llguidance speed techniques
- **Preserve unique advantages**: grammar log-probabilities, 100% success rate
- **Add unique optimizations**: probability-guided pruning (unavailable in other systems)

### Priority Order
1. Token trie (5-10x speedup, 3-5 days, low risk)
2. Caching layer (2-5x speedup, 2-3 days, low risk)  
3. Lexer/parser split (10-50x speedup, 1-2 weeks, medium risk)
4. Grammar slicing (10-100x speedup, 2-3 weeks, high risk/reward)

### Optional Rust Migration (Phase 4)
**Trigger**: If Python optimizations plateau above 100μs per token
**Strategy**: Incremental hybrid approach - Python API + Rust hot paths
**Timeline**: 4-6 weeks for hybrid system, 8-12 weeks for full migration
**Target**: 20-50μs per token while preserving Python development velocity

**Decision Points**:
- Week 8: Evaluate Python-only results
- Week 12: Assess hybrid system performance  
- Week 16: Consider full migration based on competitive pressure

**Next Step**: Start with token trie implementation in `src/stolcke_pcfg/optimizations/token_trie.py`

### Benchmarking Requirements
**Statistical Rigor**: See `docs/performance_benchmarking_methodology.md` for complete methodology

- **Sample sizes**: 10K measurements minimum, 50K recommended per comparison
- **Total benchmark suite**: ~200K measurements across all test categories  
- **Confidence level**: 99% confidence intervals, <1% margin of error
- **Test categories**: Micro-benchmarks, grammar complexity, token lengths, comparative baselines

**Validation timeline**: ~17 minutes per development iteration, ~40 seconds final optimized system