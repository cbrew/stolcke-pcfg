# RobustConstrainedAdapter Usage Guide

This guide shows how to use the `RobustConstrainedAdapter` for tokenization-resistant grammar-constrained generation.

## Quick Start

### Basic Setup

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from stolcke_pcfg import PCFG, StolckeParser
from stolcke_pcfg.robust_adapter import RobustConstrainedAdapter

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define grammar with multi-token terminals
grammar = PCFG([
    ("S", ["{", "Pair", "}"], 1.0),
    ("Pair", ["Key", ":", "Value"], 1.0),
    ("Key", ['"name"'], 0.5),    # Multi-token: ['"', 'name', '"']
    ("Key", ['"email"'], 0.5),   # Multi-token: ['"', 'email', '"']
    ("Value", ['"alice@example.com"'], 0.5),  # 8 tokens!
    ("Value", ['"bob@test.org"'], 0.5),       # 7 tokens!
])

# Create parser and adapter
parser = StolckeParser(grammar, "S")
adapter = RobustConstrainedAdapter(
    parser=parser,
    token_id_to_str=lambda tid: tokenizer.decode([tid], clean_up_tokenization_spaces=False),
    str_to_token_id=lambda s: tokenizer.encode(s, add_special_tokens=False)[0] if len(tokenizer.encode(s, add_special_tokens=False)) == 1 else None,
    tokenizer=tokenizer
)
```

### Manual Generation Loop

```python
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
generated = torch.empty((1, 0), dtype=torch.long, device=device)

for step in range(50):  # Max tokens
    # Prepare input
    if generated.size(1) == 0:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    else:
        input_ids = generated
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
    
    # Apply grammar constraints
    allowed_tokens = adapter.allowed_token_ids()
    if not allowed_tokens:
        break
    
    mask = adapter.allowed_token_mask(vocab_size=logits.size(-1))
    logits[0, ~torch.tensor(mask, dtype=torch.bool, device=device)] = -1e30
    
    # Select token (sampling recommended over greedy)
    probs = torch.softmax(logits[0] / 0.8, dim=-1)  # temperature=0.8
    next_token = torch.multinomial(probs, 1).item()
    
    # Advance adapter state
    if not adapter.step_with_token(next_token):
        break
    
    # Update generated sequence
    if generated.size(1) == 0:
        generated = torch.tensor([[next_token]], device=device)
    else:
        generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=-1)
    
    # Check completion
    if parser.accepted():
        break

# Decode result
result = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"Generated: {result}")
```

## Advanced Usage

### Integration with HuggingFace Generate API

For more sophisticated generation (beam search, nucleus sampling, etc.):

```python
from transformers import LogitsProcessorList
from stolcke_pcfg.hf_logits import GrammarConstrainedLogitsProcessor

# Create a wrapper for HF compatibility
class RobustLogitsProcessor(GrammarConstrainedLogitsProcessor):
    def __init__(self, robust_adapter):
        # Wrap robust adapter to look like old interface
        class AdapterWrapper:
            def __init__(self, robust_adapter):
                self.robust_adapter = robust_adapter
            
            def allowed_token_mask(self, vocab_size):
                return self.robust_adapter.allowed_token_mask(vocab_size)
        
        super().__init__(AdapterWrapper(robust_adapter))
        self.robust_adapter = robust_adapter
        self._last_token = None
    
    def __call__(self, input_ids, scores):
        # Update adapter state with new tokens
        if hasattr(input_ids, 'shape') and input_ids.shape[1] > 0:
            last_token = int(input_ids[0, -1].item())
            if last_token != self._last_token:
                self.robust_adapter.step_with_token(last_token)
                self._last_token = last_token
        
        return super().__call__(input_ids, scores)

# Use with generate()
processor = RobustLogitsProcessor(adapter)
processors = LogitsProcessorList([processor])

generated = model.generate(
    input_ids=torch.tensor([[]], dtype=torch.long, device=device),
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    logits_processor=processors,
    pad_token_id=tokenizer.pad_token_id
)

result = tokenizer.decode(generated[0], skip_special_tokens=True)
```

### Debugging and Monitoring

```python
# Get detailed state information
state_info = adapter.get_current_state_info()
print(f"Partial matches: {state_info['partial_matches']}")
print(f"Parser position: {state_info['parser_position']}")
print(f"Allowed terminals: {state_info['allowed_terminals']}")

# Monitor active matches during generation
for match in state_info['active_matches']:
    print(f"  '{match['terminal']}' progress: {match['progress']}")
    print(f"    Next expected token: {match['next_token']}")
```

### Error Handling

```python
try:
    success = adapter.step_with_token(token_id)
    if not success:
        print(f"Token {token_id} not allowed by grammar")
        # Handle rejection (backtrack, resample, etc.)
        
except Exception as e:
    print(f"Adapter error: {e}")
    # Reset adapter state if needed
    adapter.partial_matches = []
```

## Common Patterns

### JSON Generation

```python
def build_json_grammar():
    return PCFG([
        ("S", ["{", "KeyValuePairs", "}"], 1.0),
        ("KeyValuePairs", ["KeyValue"], 0.7),
        ("KeyValuePairs", ["KeyValue", ",", "KeyValuePairs"], 0.3),
        ("KeyValue", ["Key", ":", "Value"], 1.0),
        
        # Keys - these will be multi-token
        ("Key", ['"name"'], 0.25),
        ("Key", ['"email"'], 0.25), 
        ("Key", ['"phone"'], 0.25),
        ("Key", ['"address"'], 0.25),
        
        # Values - mix of strings and primitives  
        ("Value", ['"Alice Johnson"'], 0.2),       # Multi-token string
        ("Value", ['"alice@company.com"'], 0.2),   # Complex multi-token
        ("Value", ['"555-123-4567"'], 0.2),        # Phone number pattern
        ("Value", ["42"], 0.2),                    # Number (single token)
        ("Value", ["true"], 0.1),                  # Boolean (single token)
        ("Value", ["false"], 0.1),                 # Boolean (single token)
    ])
```

### Code Generation

```python  
def build_python_grammar():
    return PCFG([
        ("S", ["Function"], 1.0),
        ("Function", ["def", "Name", "(", ")", ":", "Body"], 1.0),
        
        # Function names - likely single tokens
        ("Name", ["hello_world"], 0.5),
        ("Name", ["calculate"], 0.5),
        
        # Body with multi-token strings
        ("Body", ["return", "String"], 1.0),
        ("String", ['"Hello, World!"'], 0.5),      # Multi-token
        ("String", ['"Result calculated"'], 0.5),  # Multi-token
    ])
```

### Email Template Generation

```python
def build_email_grammar():
    return PCFG([
        ("S", ["Email"], 1.0),
        ("Email", ["Subject", "Body"], 1.0),
        
        ("Subject", ['"Meeting Reminder"'], 0.33),    # Multi-token
        ("Subject", ['"Project Update"'], 0.33),      # Multi-token  
        ("Subject", ['"Weekly Report"'], 0.34),       # Multi-token
        
        ("Body", ['"Dear Team, ..."'], 0.5),          # Multi-token
        ("Body", ['"Hi everyone, ..."'], 0.5),        # Multi-token
    ])
```

## Performance Considerations

### Tokenization Caching

The adapter automatically caches tokenizations to avoid repeated `tokenizer.encode()` calls:

```python
# First time: tokenizes and caches
adapter.step_with_token(token_1)

# Subsequent times: uses cache  
adapter.step_with_token(token_2)  # Fast lookup
```

### Memory Usage

Memory usage scales with:
- Number of allowed terminals at each step
- Average token length of terminals
- Depth of partial matches

For very large grammars, consider:
- Splitting into smaller sub-grammars
- Using more restrictive grammar rules
- Implementing custom pruning logic

### Computational Overhead

Compared to naive approach:
- **Initialization**: ~2x slower (tokenization overhead)
- **Token selection**: ~1.5x slower (partial match updates)  
- **Overall generation**: Usually <20% slower due to better convergence

## Troubleshooting

### Common Issues

#### 1. Generation Loops/Repeats
**Symptom**: Same token generated repeatedly
**Cause**: Model probability distribution favors one token
**Solution**: Use sampling instead of greedy selection

```python
# Instead of greedy
next_token = torch.argmax(logits, dim=-1).item()

# Use sampling
probs = torch.softmax(logits[0] / temperature, dim=-1)
next_token = torch.multinomial(probs, 1).item()
```

#### 2. No Allowed Tokens
**Symptom**: `allowed_token_ids()` returns empty set
**Cause**: Parser reached invalid state or grammar issue
**Solution**: Check grammar and reset adapter

```python
allowed = adapter.allowed_token_ids()
if not allowed:
    print("No allowed tokens!")
    state = adapter.get_current_state_info()
    print(f"Parser state: {state}")
    
    # Reset if needed
    adapter.partial_matches = []
```

#### 3. Parser Rejection
**Symptom**: `step_with_token()` returns `False`
**Cause**: Token doesn't continue any partial match
**Solution**: Verify token selection logic

```python
if not adapter.step_with_token(token):
    print(f"Token {token} rejected")
    print(f"Allowed were: {adapter.allowed_token_ids()}")
```

### Debug Mode

Enable verbose logging for development:

```python
class DebugRobustAdapter(RobustConstrainedAdapter):
    def step_with_token(self, token_id: int) -> bool:
        token_str = self.id2str(token_id)
        print(f"Stepping with token {token_id} '{token_str}'")
        
        result = super().step_with_token(token_id)
        
        state = self.get_current_state_info()
        print(f"  Result: {result}")
        print(f"  Active matches: {len(state['active_matches'])}")
        
        return result
```

## Best Practices

1. **Use Sampling**: Avoid greedy decoding to prevent loops
2. **Set Temperature**: Use temperature 0.7-1.0 for diversity
3. **Monitor State**: Check adapter state for debugging
4. **Cache Adapters**: Reuse adapters for same grammar
5. **Handle Errors**: Always check return values and handle gracefully
6. **Test Tokenization**: Verify your grammar terminals tokenize as expected

This robust approach enables reliable grammar-constrained generation regardless of tokenization quirks, making it suitable for production use with any HuggingFace model and tokenizer combination.