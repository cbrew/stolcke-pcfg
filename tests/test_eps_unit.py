import pytest

from stolcke_pcfg import PCFG, StolckeParser


def test_epsilon_rejected_on_construction():
    # S -> ε (0.7) | 'a' (0.3)
    g = PCFG([
        ("S", [], 0.7),
        ("S", ["a"], 0.3),
    ])
    with pytest.raises(ValueError):
        StolckeParser(g, "S")


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
