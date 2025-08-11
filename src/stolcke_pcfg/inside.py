from functools import cache

from .grammar import PCFG, Rule
from .util import LOG_ZERO, LogProb, logsumexp


def sentence_inside_logprob(grammar: PCFG, tokens: list[str], start_symbol: str) -> LogProb:
    """Exact inside log-probability for tokens under `grammar` from `start_symbol`.

    Assumes: no epsilon productions (each nonterminal expands to at least one token).
    Uses a memoized span-based dynamic program over rules of arbitrary arity.
    Returns LOG_ZERO if the string is not generated.
    """
    n = len(tokens)

    # Pre-index rules per LHS
    rules_for: dict[str, list[Rule]] = {nt: grammar.rules_for(nt) for nt in grammar.nonterminals}

    @cache
    def inside_nt(A: str, i: int, k: int) -> LogProb:
        if i >= k:
            return LOG_ZERO
        total: LogProb = LOG_ZERO
        for r in rules_for.get(A, []):
            total = logsumexp(total, rule_inside(r, i, k))
        return total

    @cache
    def rule_inside(r: Rule, i: int, k: int) -> LogProb:
        # Compute inside prob for rule r covering tokens[i:k]
        return r.logp + match_rhs(r.rhs, 0, i, k)

    @cache
    def match_rhs(rhs: tuple[str, ...], pos: int, i: int, k: int) -> LogProb:
        # Returns log prob for matching rhs[pos:] to tokens[i:k]
        if pos == len(rhs):
            return 0.0 if i == k else LOG_ZERO

        sym = rhs[pos]
        remaining = rhs[pos + 1 :]

        # Minimal tokens needed for remaining symbols
        min_needed_rem = 0
        for s in remaining:
            if s in grammar.nonterminals:
                min_needed_rem += 1  # assume at least 1 token per nonterminal (no eps)
            else:
                min_needed_rem += 1  # terminal consumes exactly 1

        if sym in grammar.nonterminals:
            # Allocate at least 1 token for this nonterminal
            best: LogProb = LOG_ZERO
            # Maximum end for this nonterminal so that remaining can still fit
            max_end = k - min_needed_rem
            for mid in range(i + 1, max_end + 1):
                left = inside_nt(sym, i, mid)
                if left == LOG_ZERO:
                    continue
                rest = match_rhs(rhs, pos + 1, mid, k)
                if rest == LOG_ZERO:
                    continue
                best = logsumexp(best, left + rest)
            return best
        else:
            # Terminal: must match one token exactly
            if i < k and tokens[i] == sym:
                return match_rhs(rhs, pos + 1, i + 1, k)
            return LOG_ZERO

    return inside_nt(start_symbol, 0, n)
