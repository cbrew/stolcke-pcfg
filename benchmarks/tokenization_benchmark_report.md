# Tokenization Benchmark Report

**Generated**: 2025-08-11 18:27:31
**Total runs**: 30
**Device**: mps

## Executive Summary

- **Success Rate**: Robust 80.0% vs Naive 0.0% (+80.0% improvement)
- **Grammar Coverage**: Robust 80.0% vs Naive 0.0% (+80.0% improvement)
- **Tokenizer Robustness**: Robust 1.000 vs Naive 1.000
- **Performance Overhead**: Robust 0.215s vs Naive 0.000s per generation

## Detailed Results

| Adapter | Success Rate | Avg Tokens | Avg Time | Grammar Coverage | Tokenizer Robustness |
|---------|-------------|------------|----------|------------------|---------------------|
| Naive | 0.0% | 0.0 | 0.000s | 0.0% | 1.000 |
| Robust | 80.0% | 11.7 | 0.215s | 80.0% | 1.000 |

## Results by Grammar Complexity

### Extreme Complexity

- **Naive**: 0.0% success rate (0/3)
- **Robust**: 100.0% success rate (3/3)
  - Avg tokens: 16.0
  - Example output: `""The quick brown fox jumps over the lazy dog in t...`

### High Complexity

- **Naive**: 0.0% success rate (0/3)
- **Robust**: 100.0% success rate (3/3)
  - Avg tokens: 16.0
  - Example output: `{"email_address":"bob@company.org"}`

### Mixed Complexity

- **Naive**: 0.0% success rate (0/3)
- **Robust**: 100.0% success rate (3/3)
  - Avg tokens: 7.7
  - Example output: `{"id":123}`

### Realistic Json

- **Naive**: 0.0% success rate (0/3)
- **Robust**: 0.0% success rate (0/3)

### Simple Numbers

- **Naive**: 0.0% success rate (0/3)
- **Robust**: 100.0% success rate (3/3)
  - Avg tokens: 7.0
  - Example output: `{"value":42}`

## Failure Analysis

| Failure Reason | Count | Percentage |
|---------------|-------|------------|
| No allowed tokens | 15 | 83.3% |
| Max tokens reached | 3 | 16.7% |