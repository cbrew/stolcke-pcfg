from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .grammar import Rule
from .util import LOG_ZERO, LogProb


@dataclass(frozen=True)
class EarleyItem:
    rule: Rule
    dot: int
    start: int

    def next_symbol(self) -> str | None:
        return self.rule.rhs[self.dot] if self.dot < len(self.rule.rhs) else None

    def finished(self) -> bool:
        return self.dot >= len(self.rule.rhs)

    def advance(self) -> EarleyItem:
        return EarleyItem(self.rule, self.dot + 1, self.start)


@dataclass
class BP:
    kind: str
    prev: EarleyItem | None
    aux: Any | None = None


class EarleyChart:
    """State sets per position, with best Viterbi scores and backpointers."""
    def __init__(self) -> None:
        self.items: list[dict[EarleyItem, LogProb]] = []
        self.bp: list[dict[EarleyItem, BP]] = []

    def ensure_pos(self, k: int) -> None:
        while len(self.items) <= k:
            self.items.append({})
            self.bp.append({})

    def best_score(self, k: int, it: EarleyItem) -> LogProb:
        return self.items[k].get(it, LOG_ZERO)

    def update(self, k: int, it: EarleyItem, score: LogProb, bp: BP | None) -> bool:
        cur = self.items[k].get(it, LOG_ZERO)
        if score > cur:
            self.items[k][it] = score
            if bp is not None:
                self.bp[k][it] = bp
            return True
        return False
