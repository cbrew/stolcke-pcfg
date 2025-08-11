from stolcke_pcfg import PCFG, StolckeParser


def test_left_recursive_one_or_more_a():
    g = PCFG([
        ("S", ["S", "a"], 0.4),
        ("S", ["a"], 0.6),
    ])
    p = StolckeParser(g, "S")

    # At start, only 'a' is allowed
    assert p.allowed_terminals() == {"a"}

    # Consume three 'a's; all steps should progress
    assert p.step("a") is True
    assert p.step("a") is True
    assert p.step("a") is True

    # After at least one 'a', the string can be accepted
    assert p.accepted() is True
