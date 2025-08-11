import math

from .grammar import PCFG


def _exp(logp: float) -> float:
    return math.exp(logp)


def eliminate_unit_productions(g: PCFG, tol: float = 1e-12, max_pow: int = 64) -> PCFG:
    """Return an equivalent grammar without unit productions using series closure.

    - Builds U[A,B] = sum P(A->B) over unit rules (RHS len==1 and symbol is nonterminal).
    - Computes closure C = I + U + U^2 + ... via Neumann series until convergence.
    - For every non-unit rule B->gamma with prob p, adds A->gamma with prob C[A,B]*p.
    - Drops all unit rules. Per-LHS renormalization is performed by PCFG constructor.

    If the series doesn't converge within `max_pow`, raises a ValueError.
    """
    nts = sorted(g.nonterminals)
    idx: dict[str, int] = {nt: i for i, nt in enumerate(nts)}
    n = len(nts)

    # Collect unit mass matrix U and non-unit rules per LHS
    U: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    base_rules: dict[str, list[tuple[tuple[str, ...], float]]] = {nt: [] for nt in nts}
    for A in nts:
        for r in g.rules_for(A):
            if len(r.rhs) == 1 and r.rhs[0] in g.nonterminals:
                B = r.rhs[0]
                U[idx[A]][idx[B]] += _exp(r.logp)
            else:
                base_rules[A].append((r.rhs, _exp(r.logp)))

    # Neumann series C = I + U + U^2 + ...
    C: list[list[float]] = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    T: list[list[float]] = [row[:] for row in U]  # current power term

    def add_in_place(X: list[list[float]], Y: list[list[float]]) -> None:
        for i in range(n):
            ri = X[i]
            yi = Y[i]
            for j in range(n):
                ri[j] += yi[j]

    def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        out = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            Ai = A[i]
            Oi = out[i]
            for k in range(n):
                aik = Ai[k]
                if aik == 0.0:
                    continue
                Bk = B[k]
                for j in range(n):
                    bj = Bk[j]
                    if bj != 0.0:
                        Oi[j] += aik * bj
        return out

    def max_abs(X: list[list[float]]) -> float:
        m = 0.0
        for i in range(n):
            for j in range(n):
                v = X[i][j]
                if v < 0:
                    v = -v
                if v > m:
                    m = v
        return m

    k = 1
    while max_abs(T) > tol and k < max_pow:
        add_in_place(C, T)
        T = matmul(T, U)
        k += 1
    # Add final small term if below tol but nonzero
    if max_abs(T) > tol and k >= max_pow:
        raise ValueError("Unit-production closure did not converge; check grammar unit cycles")

    # Produce lifted non-unit rules using C
    lifted: list[tuple[str, tuple[str, ...], float]] = []
    for Ai, A in enumerate(nts):
        for Bj, B in enumerate(nts):
            coef = C[Ai][Bj]
            if coef == 0.0:
                continue
            for rhs, p in base_rules[B]:
                lifted.append((A, rhs, coef * p))

    return PCFG(lifted)
