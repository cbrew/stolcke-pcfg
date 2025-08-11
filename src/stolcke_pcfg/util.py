from __future__ import annotations

import math
from typing import Final

LogProb = float
LOG_ZERO: Final[LogProb] = -1e100  # robust "-inf" surrogate

def logsumexp(a: LogProb, b: LogProb) -> LogProb:
    if a < b:
        a, b = b, a
    if a == LOG_ZERO:
        return b
    if b == LOG_ZERO:
        return a
    return a + math.log1p(math.exp(b - a))
