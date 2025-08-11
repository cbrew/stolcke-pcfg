from __future__ import annotations

from stolcke_pcfg import PCFG, StolckeParser


def test_epsilon_accepts_empty_prefix():
    # S -> ε (0.7) | 'a' (0.3)
    g = PCFG([
        ("S", [], 0.7),
        ("S", ["a"], 0.3),
    ])
    p = StolckeParser(g, "S")

    # Empty input is acceptable due to epsilon
    assert p.accepted() is True
    # No terminals required at start (epsilon derivation available)
    assert p.allowed_terminals() == {"a"} or p.allowed_terminals() == {"a"}  # may allow 'a'


def test_unit_production_chain():
    # S -> A (1.0); A -> 'a' (1.0)
    g = PCFG([
        ("S", ["A"], 1.0),
        ("A", ["a"], 1.0),
    ])
    p = StolckeParser(g, "S")

    assert p.allowed_terminals() == {"a"}
    assert p.step("a") is True
    assert p.accepted() is True

