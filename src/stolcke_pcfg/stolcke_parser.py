from __future__ import annotations

from .earley_core import BP, EarleyItem
from .grammar import PCFG, Rule
from .probabilities import ProbChart
from .util import LOG_ZERO, LogProb, logsumexp


class StolckeParser:
    """Stolcke (1994/1995) probabilistic Earley with α/γ and prefix probs.

    Restrictions in this reference implementation:
      - No ε-productions and no unit productions. (Planned extension.)
    """

    def __init__(self, grammar: PCFG, start_symbol: str):
        self.G = grammar
        self.S = start_symbol
        self._start_rule = Rule("S'", (self.S,), 0.0)
        self.reset()

    def reset(self) -> None:
        self.chart = ProbChart()
        self.pos = 0
        self.chart.ensure_pos(0)
        start_item = EarleyItem(self._start_rule, 0, 0)
        # Initialize α and γ at position 0 for the augmented start
        self.chart.add_alpha(0, start_item, 0.0)
        self.chart.add_gamma(0, start_item, 0.0)
        # Seed item (no Viterbi score needed, but store for completeness)
        self.chart.items[0][start_item] = 0.0
        self.chart.bp[0][start_item] = BP("INIT", None)
        self._complete_predictor_loop(0)
        self._last_prefix_lp: LogProb = LOG_ZERO

    def allowed_terminals(self) -> set[str]:
        k = self.pos
        allowed: set[str] = set()
        for it in self.chart.items[k]:
            nxt = it.next_symbol()
            if nxt is not None and self.G.is_terminal(nxt):
                allowed.add(nxt)
        return allowed

    def step(self, terminal: str) -> bool:
        k = self.pos
        self.chart.ensure_pos(k + 1)
        progressed = False
        scanned_this_step: list[EarleyItem] = []

        # SCAN: move dot over matching terminal
        for it in list(self.chart.items[k].keys()):
            nxt = it.next_symbol()
            if nxt is not None and self.G.is_terminal(nxt) and nxt == terminal:
                nit = it.advance()
                self.chart.items[k + 1][nit] = 0.0  # keep set membership
                self.chart.bp[k + 1][nit] = BP("SCAN", it, terminal)
                progressed = True
                # Stolcke scanning: α' = α ; γ' = γ
                a_prev = self.chart.get_alpha(k, it)
                g_prev = self.chart.get_gamma(k, it)
                if a_prev != LOG_ZERO:
                    self.chart.add_alpha(k + 1, nit, a_prev)
                if g_prev != LOG_ZERO:
                    self.chart.add_gamma(k + 1, nit, g_prev)
                scanned_this_step.append(nit)

        if not progressed:
            return False

        # Prefix probability at position k+1: sum of α over scanned states
        lp = LOG_ZERO
        for it in scanned_this_step:
            a = self.chart.get_alpha(k + 1, it)
            lp = logsumexp(lp, a)
        self._last_prefix_lp = lp

        # COMPLETE+PREDICT closure at k+1
        self._complete_predictor_loop(k + 1)
        self.pos += 1
        return True

    def prefix_logprob(self) -> LogProb:
        return self._last_prefix_lp

    def accepted(self) -> bool:
        # Accept if S' -> S • spanning [0,pos] exists
        k = self.pos
        for it in self.chart.items[k]:
            if it.rule is self._start_rule and it.finished() and it.start == 0:
                return True
        return False

    # -------------------- internal closure --------------------
    def _complete_predictor_loop(self, k: int) -> None:
        agenda = list(self.chart.items[k].keys())
        seen: set[EarleyItem] = set()
        while agenda:
            it = agenda.pop()
            if it in seen:
                continue
            seen.add(it)
            nxt = it.next_symbol()
            if nxt is None:
                # COMPLETER: combine waiting parents at position i with child's γ
                i = it.start
                g_child = self.chart.get_gamma(k, it)
                for pit in list(self.chart.items[i].keys()):
                    if pit.next_symbol() == it.rule.lhs:
                        nit = pit.advance()
                        if nit not in self.chart.items[k]:
                            self.chart.items[k][nit] = 0.0
                            self.chart.bp[k][nit] = BP("COMP", pit, it)
                            agenda.append(nit)
                        # α' += α(parent) * γ(child)
                        a_parent = self.chart.get_alpha(i, pit)
                        if a_parent != LOG_ZERO and g_child != LOG_ZERO:
                            self.chart.add_alpha(k, nit, a_parent + g_child)
                        # γ' += γ(parent) * γ(child)
                        g_parent = self.chart.get_gamma(i, pit)
                        if g_parent != LOG_ZERO and g_child != LOG_ZERO:
                            self.chart.add_gamma(k, nit, g_parent + g_child)
            else:
                if not self.G.is_terminal(nxt):
                    # PREDICTOR: expand nonterminal
                    for r in self.G.rules_for(nxt):
                        nit = EarleyItem(r, 0, k)
                        is_new = False
                        if nit not in self.chart.items[k]:
                            self.chart.items[k][nit] = 0.0
                            self.chart.bp[k][nit] = BP("PRED", it)
                            agenda.append(nit)
                            is_new = True
                        # α' += α(current) * P(Y->γ)
                        a_cur = self.chart.get_alpha(k, it)
                        if a_cur != LOG_ZERO:
                            self.chart.add_alpha(k, nit, a_cur + r.logp)
                        # γ init: set rule prob once per predicted item
                        if is_new:
                            self.chart.add_gamma(k, nit, r.logp)
                # else: waiting for terminal scan
