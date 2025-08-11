from .earley_core import BP, EarleyItem
from .grammar import PCFG, Rule
from .inside import sentence_inside_logprob
from .probabilities import ProbChart
from .transform import eliminate_unit_productions
from .util import LOG_ZERO, LogProb, logsumexp


class StolckeParser:
    """Stolcke (1994/1995) probabilistic Earley with α/γ and prefix probs.

    Restrictions in this reference implementation:
      - No ε-productions and no unit productions. (Planned extension.)
    """

    def __init__(self, grammar: PCFG, start_symbol: str, *, eliminate_units: bool = True):
        # Reject epsilon productions (not supported by this implementation)
        for lhs in grammar.nonterminals:
            for r in grammar.rules_for(lhs):
                if len(r.rhs) == 0:
                    raise ValueError("Epsilon (empty) productions are not supported")
        # Eliminate unit productions to avoid unit-cycles at parse time
        self.G = eliminate_unit_productions(grammar) if eliminate_units else grammar
        self.S = start_symbol
        self._start_rule = Rule("S'", (self.S,), 0.0)
        self.reset()

    def reset(self) -> None:
        self.chart = ProbChart()
        self.pos = 0
        self._tokens: list[str] = []
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
        self._tokens.append(terminal)
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

    def sentence_logprob(self) -> LogProb:
        """Log P(tokens[0:pos]) if it's a complete sentence; LOG_ZERO otherwise.

        Returns the inside probability computed via a span-based DP for stability.
        """
        return sentence_inside_logprob(self.G, self._tokens, self.S)

    # -------------------- internal closure --------------------
    def _complete_predictor_loop(self, k: int) -> None:
        from collections import deque

        queue = deque(self.chart.items[k].keys())
        while queue:
            it = queue.popleft()
            nxt = it.next_symbol()
            if nxt is None:
                # COMPLETER: combine waiting parents at position i with child's γ
                i = it.start
                g_child = self.chart.get_gamma(k, it)
                if g_child == LOG_ZERO:
                    continue
                for pit in list(self.chart.items[i].keys()):
                    if pit.next_symbol() == it.rule.lhs:
                        nit = pit.advance()
                        new_struct = False
                        if nit not in self.chart.items[k]:
                            self.chart.items[k][nit] = 0.0
                            self.chart.bp[k][nit] = BP("COMP", pit, it)
                            new_struct = True
                        changed = False
                        # α' += α(parent) * γ(child) only if β is empty (no epsilons supported)
                        if nit.dot == len(nit.rule.rhs):
                            a_parent = self.chart.get_alpha(i, pit)
                            if a_parent != LOG_ZERO:
                                prev = self.chart.get_alpha(k, nit)
                                self.chart.add_alpha(k, nit, a_parent + g_child)
                                if self.chart.get_alpha(k, nit) != prev:
                                    changed = True
                        # γ' += γ(parent) * γ(child)
                        g_parent = self.chart.get_gamma(i, pit)
                        if g_parent != LOG_ZERO:
                            prevg = self.chart.get_gamma(k, nit)
                            self.chart.add_gamma(k, nit, g_parent + g_child)
                            if self.chart.get_gamma(k, nit) != prevg:
                                changed = True
                        if new_struct or changed:
                            queue.append(nit)
            else:
                if not self.G.is_terminal(nxt):
                    # PREDICTOR: expand nonterminal
                    beta_nonempty = (it.dot + 1) < len(it.rule.rhs)
                    for r in self.G.rules_for(nxt):
                        nit = EarleyItem(r, 0, k)
                        new_struct = False
                        if nit not in self.chart.items[k]:
                            self.chart.items[k][nit] = 0.0
                            self.chart.bp[k][nit] = BP("PRED", it)
                            new_struct = True
                        changed = False
                        # Predictor α only when suffix β is empty (no epsilons supported)
                        if not beta_nonempty:
                            a_cur = self.chart.get_alpha(k, it)
                            if a_cur != LOG_ZERO:
                                prev = self.chart.get_alpha(k, nit)
                                self.chart.add_alpha(k, nit, a_cur + r.logp)
                                if self.chart.get_alpha(k, nit) != prev:
                                    changed = True
                        # γ init: set rule prob once
                        prevg = self.chart.get_gamma(k, nit)
                        if prevg == LOG_ZERO:
                            self.chart.add_gamma(k, nit, r.logp)
                            changed = True
                        if new_struct or changed:
                            queue.append(nit)
