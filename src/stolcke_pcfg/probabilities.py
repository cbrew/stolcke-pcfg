from __future__ import annotations

from .earley_core import EarleyChart, EarleyItem
from .util import LOG_ZERO, LogProb, logsumexp


class ProbChart(EarleyChart):
    """Extends the base chart with **forward (alpha)** and **inner (gamma)**."""
    def __init__(self) -> None:
        super().__init__()
        self.alpha: list[dict[EarleyItem, LogProb]] = []
        self.gamma: list[dict[EarleyItem, LogProb]] = []

    def ensure_pos(self, k: int) -> None:
        while len(self.items) <= k:
            self.items.append({})
            self.bp.append({})
            self.alpha.append({})
            self.gamma.append({})

    def get_alpha(self, k: int, it: EarleyItem) -> LogProb:
        return self.alpha[k].get(it, LOG_ZERO)

    def get_gamma(self, k: int, it: EarleyItem) -> LogProb:
        return self.gamma[k].get(it, LOG_ZERO)

    def add_alpha(self, k: int, it: EarleyItem, contrib: LogProb) -> None:
        cur = self.alpha[k].get(it, LOG_ZERO)
        self.alpha[k][it] = logsumexp(cur, contrib)

    def add_gamma(self, k: int, it: EarleyItem, contrib: LogProb) -> None:
        cur = self.gamma[k].get(it, LOG_ZERO)
        self.gamma[k][it] = logsumexp(cur, contrib)
