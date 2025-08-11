from src.stolcke_pcfg import PCFG, StolckeParser

# Grammar: one or more 'a's (no epsilon, left-recursive permitted)
# S -> S 'a' (0.4) | 'a' (0.6)
G = PCFG([
    ("S", ["S", "a"], 0.4),
    ("S", ["a"], 0.6),
])
P = StolckeParser(G, "S")

print("Allowed at start:", P.allowed_terminals())
for ch in ["a", "a", "a"]:
    ok = P.step(ch)
    print(f"step('{ch}') ->", ok, " prefix_logprob=", P.prefix_logprob())
print("accepted?", P.accepted())
