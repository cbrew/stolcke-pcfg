import math

from stolcke_pcfg import PCFG, StolckeParser


def build_coord_grammar() -> PCFG:
    # S -> NP
    # NP -> NP 'and' NP | N
    # N -> 'cats' | 'dogs' | 'foxes' | 'wolves'
    # Use equal probabilities so per-tree probability is constant across bracketings.
    return PCFG(
        [
            ("S", ["NP"], 1.0),
            ("NP", ["NP", "and", "NP"], 0.5),
            ("NP", ["N"], 0.5),
            ("N", ["cats"], 0.25),
            ("N", ["dogs"], 0.25),
            ("N", ["foxes"], 0.25),
            ("N", ["wolves"], 0.25),
        ]
    )


def parse_and_logp(tokens: list[str]) -> tuple[StolckeParser, float]:
    g = build_coord_grammar()
    p = StolckeParser(g, "S")
    for t in tokens:
        assert p.step(t)
    assert p.accepted()
    return p, p.sentence_logprob()


def expected_count_for(tokens: list[str], count_expected: int) -> None:
    p, logp = parse_and_logp(tokens)

    n = (len(tokens) + 1) // 2  # number of nouns
    # Per-tree log prob: (NP->NP and NP)^(n-1) * (NP->N)^n * prod N->token
    # All normalized to:
    #   P(NP->coord) = 0.5, P(NP->N) = 0.5, P(N->token) = 0.25
    log_per_tree = (n - 1) * math.log(0.5) + n * math.log(0.5) + n * math.log(0.25)
    count = round(math.exp(logp - log_per_tree))
    assert count == count_expected


def test_coordination_three_items_two_trees():
    expected_count_for(["cats", "and", "dogs", "and", "foxes"], 2)


def test_coordination_four_items_five_trees():
    expected_count_for(["cats", "and", "dogs", "and", "foxes", "and", "wolves"], 5)
