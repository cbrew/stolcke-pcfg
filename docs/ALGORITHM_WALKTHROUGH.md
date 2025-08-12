# Algorithm Walkthrough: Robust Tokenization Handling

This document provides a detailed step-by-step walkthrough of how the `RobustConstrainedAdapter` handles multi-token terminals during generation.

## Example Scenario

Let's trace through generating the JSON: `{"email":"alice@domain.com"}`

### Grammar Definition

```python
grammar = PCFG([
    ("S", ["{", "Pair", "}"], 1.0),
    ("Pair", ["Key", ":", "Value"], 1.0),
    ("Key", ['"email"'], 1.0),
    ("Value", ['"alice@domain.com"'], 1.0),
])
```

### Tokenization Analysis

First, let's see how GPT-2 tokenizes our terminals:

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

terminals = ['{', '"email"', ':', '"alice@domain.com"', '}']
for term in terminals:
    tokens = tokenizer.encode(term, add_special_tokens=False)
    decoded = [tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in tokens]
    print(f"'{term}' -> {tokens} -> {decoded}")
```

**Output**:
```
'{' -> [90] -> ['{']                                    # ✅ Single token
'"email"' -> [1, 12888, 1] -> ['"', 'email', '"']      # ❌ 3 tokens  
':' -> [25] -> [':']                                    # ✅ Single token
'"alice@domain.com"' -> [1, 282, 501, 31, 27830, 13, 785, 1] -> ['"', 'al', 'ice', '@', 'domain', '.', 'com', '"']  # ❌ 8 tokens!
'}' -> [92] -> ['}']                                    # ✅ Single token
```

## Step-by-Step Generation

### Initial State (Step 0)

**Parser State**:
- Position: 0  
- Allowed terminals: `['{']`

**Adapter Initialization**:
```python
# _initialize_partial_matches() called
allowed_terminals = parser.allowed_terminals()  # ['{']
partial_matches = [
    PartialMatch(terminal='{', token_sequence=[], remaining_tokens=[90])
]
```

**Token Allowance**:
```python
# allowed_token_ids() called
allowed_tokens = {90}  # Only '{' token allowed
```

**Model Generation**: Model generates token `90` ('{'')

**Token Consumption**:
```python
# step_with_token(90) called
continuing_matches = []
for match in partial_matches:
    if match.next_expected_token == 90:  # '{' match expects 90
        new_match = PartialMatch(
            terminal='{',
            token_sequence=[90],
            remaining_tokens=[]  # Complete!
        )
        continuing_matches.append(new_match)

# Check for completion
completed_matches = [new_match]  # Match is complete
parser.step('{')  # Advance parser with complete terminal ✅
partial_matches = []  # Reset for next terminal
```

**Result**: Token `90` accepted, parser advances to position 1

---

### After Step 0: Parser State Change

**Parser State**:
- Position: 1
- Allowed terminals: `['"email"']` (from Pair -> Key rule)

**Adapter Re-initialization**:
```python
# New parser state triggers re-initialization  
partial_matches = [
    PartialMatch(
        terminal='"email"',
        token_sequence=[],
        remaining_tokens=[1, 12888, 1]  # ['"', 'email', '"']
    )
]
```

---

### Step 1: First Token of "email"

**Token Allowance**:
```python
allowed_tokens = {1}  # First token of '"email"' sequence
```

**Model Generation**: Model generates token `1` ('"')

**Token Consumption**:
```python
# step_with_token(1) called
for match in partial_matches:
    if match.next_expected_token == 1:  # '"email"' expects 1
        new_match = PartialMatch(
            terminal='"email"',
            token_sequence=[1],           # Consumed '"'
            remaining_tokens=[12888, 1]   # Still need 'email', '"'
        )
        continuing_matches.append(new_match)

partial_matches = [new_match]
# Not complete yet, continue with partial match
```

**Result**: Token `1` accepted as partial match, generation continues

---

### Step 2: Second Token of "email"

**Token Allowance**:
```python
# Current partial match expects token 12888 ('email')
allowed_tokens = {12888}
```

**Model Generation**: Model generates token `12888` ('email')

**Token Consumption**:
```python
# step_with_token(12888) called
for match in partial_matches:
    if match.next_expected_token == 12888:  # Matches!
        new_match = PartialMatch(
            terminal='"email"',
            token_sequence=[1, 12888],    # Consumed '"', 'email'
            remaining_tokens=[1]          # Still need final '"'
        )
        continuing_matches.append(new_match)

partial_matches = [new_match]
# Still not complete
```

**Result**: Token `12888` accepted as partial match

---

### Step 3: Final Token of "email"

**Token Allowance**:
```python
# Partial match expects final token 1 ('"')
allowed_tokens = {1}
```

**Model Generation**: Model generates token `1` ('"')

**Token Consumption**:
```python
# step_with_token(1) called
for match in partial_matches:
    if match.next_expected_token == 1:  # Final '"'
        new_match = PartialMatch(
            terminal='"email"',
            token_sequence=[1, 12888, 1],  # Complete sequence!
            remaining_tokens=[]            # Nothing left
        )
        continuing_matches.append(new_match)

partial_matches = [new_match]

# Check for completion
completed_matches = [new_match]  # Complete!
parser.step('"email"')  # Advance parser with complete terminal ✅
partial_matches = []    # Reset
```

**Result**: Complete terminal `"email"` consumed, parser advances

---

### Step 4: Colon Separator

**Parser State**:
- Position: 2
- Allowed terminals: `[':']`

**Similar Process**:
- Single token `25` (':'') allowed
- Model generates `25`
- Immediately completes and advances parser

---

### Steps 5-12: Complex Multi-Token Value

**Parser State**:
- Position: 3
- Allowed terminals: `['"alice@domain.com"']`

**Tokenization**: `[1, 282, 501, 31, 27830, 13, 785, 1]`

**Process**: Each token is handled sequentially:

| Step | Token | Decoded | Partial State | Complete? |
|------|-------|---------|---------------|-----------|
| 5 | 1 | '"' | `[1]`, need `[282, 501, 31, 27830, 13, 785, 1]` | No |
| 6 | 282 | 'al' | `[1, 282]`, need `[501, 31, 27830, 13, 785, 1]` | No |
| 7 | 501 | 'ice' | `[1, 282, 501]`, need `[31, 27830, 13, 785, 1]` | No |
| 8 | 31 | '@' | `[1, 282, 501, 31]`, need `[27830, 13, 785, 1]` | No |
| 9 | 27830 | 'domain' | `[1, 282, 501, 31, 27830]`, need `[13, 785, 1]` | No |
| 10 | 13 | '.' | `[1, 282, 501, 31, 27830, 13]`, need `[785, 1]` | No |
| 11 | 785 | 'com' | `[1, 282, 501, 31, 27830, 13, 785]`, need `[1]` | No |
| 12 | 1 | '"' | `[1, 282, 501, 31, 27830, 13, 785, 1]`, need `[]` | **Yes!** |

At step 12, the complete terminal `"alice@domain.com"` is consumed and the parser advances.

---

### Step 13: Final Brace

**Parser State**:
- Position: 4  
- Allowed terminals: `['}']`

**Process**:
- Token `92` ('}') allowed and generated
- Completes immediately  
- Parser accepts final state

## Key Algorithmic Insights

### 1. **State Persistence**
Unlike naive approaches that reset after each token, the robust adapter maintains `PartialMatch` state across multiple generation steps:

```python
# Naive approach (broken):
def naive_step(token_id):
    # Can only handle single tokens
    token_str = decode(token_id) 
    return parser.step(token_str)  # Fails for multi-token terminals

# Robust approach:
def robust_step(token_id):
    # Maintains partial matches across calls
    for match in self.partial_matches:
        if match.expects(token_id):
            match.consume(token_id)
            if match.is_complete():
                return parser.step(match.terminal)  # Success!
```

### 2. **Lazy Evaluation**
Partial matches are only created when needed (when parser state changes):

```python
def allowed_token_ids(self):
    if not self.partial_matches:  # Only if empty
        self._initialize_partial_matches()  # Expensive operation
    return {m.next_expected_token for m in self.partial_matches}
```

### 3. **Pruning Strategy**
Invalid partial matches are naturally pruned as generation proceeds:

```python
# Start with multiple possible terminals
partial_matches = [
    PartialMatch(terminal='"Alice"', remaining=[1, 44484, 1]),
    PartialMatch(terminal='"Bob"', remaining=[1, 18861, 1])
]

# After token 44484 ('Alice'), only one match continues
continuing_matches = []
for match in partial_matches:
    if match.next_expected_token == 44484:  # Only "Alice" match
        continuing_matches.append(match.advance(44484))

partial_matches = continuing_matches  # Pruned to one match
```

### 4. **Parser Synchronization**
The adapter only advances the parser when complete terminals are consumed, maintaining grammar semantics:

```python
# Partial consumption - don't advance parser yet
if not any(m.is_complete() for m in partial_matches):
    return True  # Accept token but don't advance parser

# Complete consumption - advance parser  
for complete_match in completed_matches:
    success = parser.step(complete_match.terminal)
    if success:
        self.partial_matches = []  # Reset for next terminal
        return True
```

This synchronization ensures that:
- Grammar log-probabilities remain accurate
- Parser state invariants are maintained  
- Error recovery is possible

## Complexity Analysis

### Time Complexity Per Step

1. **Initialization**: O(T × L) where T = terminals, L = avg token length
2. **Token allowance**: O(M) where M = active partial matches  
3. **Token consumption**: O(M) for updating matches
4. **Parser advancement**: O(1) for single step

**Overall**: O(T × L + M) per generation step

### Space Complexity

1. **Partial matches**: O(T × L) in worst case (all terminals active)
2. **Token cache**: O(T × L) for memoization
3. **Parser state**: O(P) where P = parser complexity

**Overall**: O(T × L + P) total memory

### Scaling Characteristics

- **Linear** in grammar size and tokenization complexity
- **Constant** in generation length (state resets after each terminal)
- **Efficient** due to natural pruning and lazy evaluation

The algorithm scales well for practical grammars while providing complete tokenization independence.