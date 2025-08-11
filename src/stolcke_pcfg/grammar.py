import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .util import LogProb


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: tuple[str, ...]
    logp: LogProb

    def __str__(self) -> str:
        rhs_s = " ".join(self.rhs) if self.rhs else "ε"
        return f"{self.lhs} -> {rhs_s} ({math.exp(self.logp):.6f})"


class PCFG:
    """A compact PCFG with per-LHS normalization (log-space probs).

    Terminals are symbols that never appear on the left-hand side.
    ε- and unit-productions are allowed when constructing the object, but
    parsers in this package may reject them depending on capabilities.
    """

    def __init__(self, rules: Iterable[tuple[str, Iterable[str], float]]):
        by_lhs: dict[str, list[Rule]] = defaultdict(list)
        for lhs, rhs, p in rules:
            if p <= 0.0:
                raise ValueError(f"Rule {lhs}->{tuple(rhs)} must have p>0, got {p}")
            by_lhs[lhs].append(Rule(lhs, tuple(rhs), math.log(p)))
        # normalize
        self._rules: dict[str, list[Rule]] = {}
        for lhs, rlist in by_lhs.items():
            total = sum(math.exp(r.logp) for r in rlist)
            self._rules[lhs] = [Rule(r.lhs, r.rhs, math.log(math.exp(r.logp)/total)) for r in rlist]
        self._lhs_set: frozenset[str] = frozenset(self._rules.keys())

    def rules_for(self, lhs: str) -> list[Rule]:
        return self._rules.get(lhs, [])

    @property
    def nonterminals(self) -> frozenset[str]:
        return self._lhs_set

    def is_terminal(self, sym: str) -> bool:
        return sym not in self._lhs_set
