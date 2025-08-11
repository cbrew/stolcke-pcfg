from __future__ import annotations

from math import isfinite

from stolcke_pcfg import PCFG, StolckeParser


def build_nl_grammar() -> PCFG:
    # Simple NL-like grammar with NP/VP/PP and determiners
    # S -> NP VP
    # NP -> Det N | N
    # VP -> V | V NP | VP PP
    # PP -> P NP
    # Det -> 'the' | 'a'
    # N -> 'cat' | 'mat'
    # V -> 'sleeps' | 'likes'
    # P -> 'on'
    return PCFG(
        [
            ("S", ["NP", "VP"], 1.0),
            ("NP", ["Det", "N"], 0.6),
            ("NP", ["N"], 0.4),
            ("VP", ["V"], 0.4),
            ("VP", ["V", "NP"], 0.4),
            ("VP", ["VP", "PP"], 0.2),  # allow PP attachment to VP
            ("PP", ["P", "NP"], 1.0),
            ("Det", ["the"], 0.6),
            ("Det", ["a"], 0.4),
            ("N", ["cat"], 0.5),
            ("N", ["mat"], 0.5),
            ("V", ["sleeps"], 0.5),
            ("V", ["likes"], 0.5),
            ("P", ["on"], 1.0),
        ]
    )


def test_simple_sentence_accepts():
    g = build_nl_grammar()
    p = StolckeParser(g, "S")

    # At start, NP can begin with a determiner or a noun
    allowed0 = p.allowed_terminals()
    assert {"the", "a"}.issubset(allowed0) and {"cat", "mat"}.issubset(allowed0)

    for tok in ["the", "cat", "sleeps"]:
        assert p.step(tok) is True

    assert p.accepted() is True


def test_transitive_sentence_accepts():
    g = build_nl_grammar()
    p = StolckeParser(g, "S")

    sent = ["the", "cat", "likes", "the", "mat"]
    for t in sent:
        assert p.step(t) is True
    assert p.accepted() is True


def test_pp_attachment_sentence_accepts_and_prefix_monotonic():
    g = build_nl_grammar()
    p = StolckeParser(g, "S")

    sent = ["the", "cat", "sleeps", "on", "the", "mat"]
    last_lp = 0.0  # log P(empty prefix) = 0
    for _i, t in enumerate(sent):
        assert p.step(t) is True
        lp = p.prefix_logprob()
        assert isfinite(lp)
        assert lp <= last_lp + 1e-12  # non-increasing (probability shrinks with longer prefixes)
        last_lp = lp

        if t == "on":
            # After preposition, expect an NP next: determiner or noun
            allowed = p.allowed_terminals()
            assert {"the", "a"}.issubset(allowed) or {"cat", "mat"}.issubset(allowed)

    assert p.accepted() is True
