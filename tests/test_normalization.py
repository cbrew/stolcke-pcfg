import math

from stolcke_pcfg import PCFG
from stolcke_pcfg.transform import eliminate_unit_productions


def sum_probs_by_lhs(G: PCFG) -> dict[str, float]:
    sums: dict[str, float] = {}
    for lhs in G.nonterminals:
        total = 0.0
        for r in G.rules_for(lhs):
            total += math.exp(r.logp)
        sums[lhs] = total
    return sums


def test_large_normalization_after_unit_elimination():
    # A larger grammar with mixed unit/non-unit rules across several LHSs
    G = PCFG(
        [
            ("S", ["A"], 0.3),
            ("S", ["B"], 0.7),
            ("A", ["B"], 0.2),
            ("A", ["C"], 0.3),
            ("A", ["x"], 0.5),
            ("B", ["C"], 0.4),
            ("B", ["y"], 0.6),
            ("C", ["z"], 1.0),
        ]
    )

    # Baseline sums are 1.0 by PCFG construction
    sums_orig = sum_probs_by_lhs(G)
    for _lhs, s in sums_orig.items():
        assert abs(s - 1.0) < 1e-9

    GT = eliminate_unit_productions(G)
    sums = sum_probs_by_lhs(GT)
    # After elimination, each LHS should remain normalized
    for _lhs, s in sums.items():
        assert abs(s - 1.0) < 1e-9
