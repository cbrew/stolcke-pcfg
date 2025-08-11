import numpy as np

from .grammar import PCFG


def eliminate_unit_productions(g: PCFG, tol: float = 1e-12) -> PCFG:
    """Return an equivalent grammar without unit productions using a NumPy closure.

    - Builds U[A,B] = sum P(A->B) over unit rules (RHS len==1 and symbol is nonterminal).
    - Computes C = (I - U)^{-1} via `np.linalg.solve` after checking spectral radius.
    - For every non-unit rule B->gamma with prob p, adds A->gamma with prob C[A,B]*p.
    - Drops all unit rules. Per-LHS renormalization is performed by PCFG constructor.
    """
    nts = sorted(g.nonterminals)
    idx: dict[str, int] = {nt: i for i, nt in enumerate(nts)}
    n = len(nts)

    U = np.zeros((n, n), dtype=float)
    base_rules: dict[str, list[tuple[tuple[str, ...], float]]] = {nt: [] for nt in nts}
    for A in nts:
        for r in g.rules_for(A):
            if len(r.rhs) == 1 and r.rhs[0] in g.nonterminals:
                B = r.rhs[0]
                U[idx[A], idx[B]] += np.exp(r.logp)
            else:
                base_rules[A].append((r.rhs, float(np.exp(r.logp))))

    if n:
        rho = float(np.max(np.abs(np.linalg.eigvals(U))))
        if rho >= 1 - tol:
            raise ValueError(f"Unit closure diverges: spectral radius {rho:.6f} ≥ 1")
        eye = np.eye(n, dtype=float)
        C = np.linalg.solve(eye - U, eye)
        C[C < 0] = 0.0  # clamp tiny negatives
    else:
        C = np.zeros((0, 0), dtype=float)

    lifted: list[tuple[str, tuple[str, ...], float]] = []
    for Ai, A in enumerate(nts):
        for Bj, B in enumerate(nts):
            coef = float(C[Ai, Bj]) if n else 0.0
            if coef == 0.0:
                continue
            for rhs, p in base_rules[B]:
                lifted.append((A, rhs, coef * p))

    return PCFG(lifted)
