import math

import pytest

from stolcke_pcfg import PCFG, eliminate_unit_productions
from stolcke_pcfg.inside import sentence_inside_logprob


def logeq(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_unit_transform_equivalence_simple():
    # S -> A (0.7) | B (0.3); A -> 'a'; B -> 'a'
    G = PCFG([
        ("S", ["A"], 0.7),
        ("S", ["B"], 0.3),
        ("A", ["a"], 1.0),
        ("B", ["a"], 1.0),
    ])
    GT = eliminate_unit_productions(G)

    lp_orig = sentence_inside_logprob(G, ["a"], "S")
    lp_trans = sentence_inside_logprob(GT, ["a"], "S")

    assert logeq(lp_orig, lp_trans)


def test_unit_transform_chain_equivalence():
    # S -> A; A -> B; B -> 'b'
    G = PCFG([
        ("S", ["A"], 1.0),
        ("A", ["B"], 1.0),
        ("B", ["b"], 1.0),
    ])
    GT = eliminate_unit_productions(G)
    # Ensure no unit rules remain
    for lhs in GT.nonterminals:
        for r in GT.rules_for(lhs):
            assert not (len(r.rhs) == 1 and r.rhs[0] in GT.nonterminals)

    lp_orig = sentence_inside_logprob(G, ["b"], "S")
    lp_trans = sentence_inside_logprob(GT, ["b"], "S")
    assert logeq(lp_orig, lp_trans)


def test_unit_transform_branching_distribution():
    # S -> A; A -> B (0.4) | C (0.6); B -> 'x'; C -> 'y'
    G = PCFG([
        ("S", ["A"], 1.0),
        ("A", ["B"], 0.4),
        ("A", ["C"], 0.6),
        ("B", ["x"], 1.0),
        ("C", ["y"], 1.0),
    ])
    GT = eliminate_unit_productions(G)

    lp_x = sentence_inside_logprob(GT, ["x"], "S")
    lp_y = sentence_inside_logprob(GT, ["y"], "S")
    assert logeq(lp_x, math.log(0.4))
    assert logeq(lp_y, math.log(0.6))


def test_unit_transform_divergent_spectral_radius_raises():
    # U = [[0,1],[1,0]] -> spectral radius 1 (non-convergent)
    G = PCFG([
        ("A", ["B"], 1.0),
        ("B", ["A"], 1.0),
    ])
    with pytest.raises(ValueError):
        eliminate_unit_productions(G)


def test_mixed_unit_and_nonunit_on_same_lhs():
    # S -> A
    # A -> B (0.5) | 'x' (0.5)   # mix of unit and non-unit on same LHS
    # B -> C (0.4) | 'y' (0.6)   # mix again
    # C -> 'z'
    G = PCFG([
        ("S", ["A"], 1.0),
        ("A", ["B"], 0.5),
        ("A", ["x"], 0.5),
        ("B", ["C"], 0.4),
        ("B", ["y"], 0.6),
        ("C", ["z"], 1.0),
    ])
    GT = eliminate_unit_productions(G)

    # Check that sentence probabilities are preserved across transform
    cases = {
        "x": math.log(0.5),
        "y": math.log(0.5 * 0.6),
        "z": math.log(0.5 * 0.4 * 1.0),
    }
    for tok, expected in cases.items():
        lp_orig = sentence_inside_logprob(G, [tok], "S")
        lp_trans = sentence_inside_logprob(GT, [tok], "S")
        assert logeq(lp_orig, expected)
        assert logeq(lp_trans, expected)
