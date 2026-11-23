"""Selective compliance enrichment (go/no-go test of FINAL_RESEARCH_PLAN.md D13).

Full enriched space (baseline G): ``Z = [V, X^-1 B^T]`` (r + n_c columns). Selective space: ``Z_S = [V, X^-1 B_S^T]``
for a row set ``S`` (|S| << n_c); the Galerkin contact problem is solved on ``range Z_S`` with **all** n_c constraints:
    min_q 1/2 q^T H_S(mu) q - q^T Z_S^T f(mu)   s.t.   B Z_S q >= c(mu).
The columns are kept raw (not orthonormalised) so that any subset can be selected online from precomputed blocks
``Z^T K_q Z``, ``B Z`` and ``Z^T f_j``. Since H_S may be ill-conditioned when V nearly lies in the span of the selected
compliance columns, it is factorised by a symmetric eigendecomposition with a relative cut-off (``EIG_RTOL``).

The QP is solved by a working-set (constraint-generation) loop: the dual LCP is solved on a working set W of rows,
all n_c rows are then checked, the violated ones enter W, repeat. If the loop does not reach feasibility, the caller's
R0 clean-up restores it in constraint space (the certificate stays valid; the clean-up is reported).

Selectors (static ones are fixed offline; ``predicted`` is per query):
  random            k rows at random (seeded);
  training_force    the k rows most often strictly active in the training snapshots (ties: larger mean force);
  greedy            certificate-driven, offline, on held-out parameters (no full-order solves): starting from the
                    training-force rows of a quarter of the budget, add in batches the rows with the largest mean
                    contribution l*_i g^_i of the S1 multiplier (missed rows, paper Theorem 3.4), else the largest mean
                    residual sensitivity |(B X^-1 r)_i|;
  predicted         per query: the ROM gap rule {g~_i <= t} (paper Corollary 4.4), t = factor * tau_g * char_gap.
"""

import numpy as np

from .rom import solve_complementarity

EIG_RTOL = 1e-12            # eigenvalues of H_S below this fraction of the largest are dropped
MAX_PASSES = 30             # working-set passes
FEAS_RTOL = 1e-10           # constraint violation tolerance relative to max |c|
SELECTORS = ("random", "training_force", "greedy", "predicted")


class RawEnrichment:
    """Precomputed blocks of the raw enriched space ``Z = [V, X^-1 B^T]``; duck-types the ``Z``, ``H_terms``, ``C``
    attributes used by offline_online.OnlineCertifier."""

    def __init__(self, problem, V: np.ndarray, xinv_bt: np.ndarray, right_inverse):
        self.problem, self.right_inverse, self.r = problem, right_inverse, V.shape[1]
        self.Z = np.hstack([V, xinv_bt])
        self.H_terms = tuple(self.Z.T @ (Kq @ self.Z) for Kq in problem.stiffness_terms)
        self.C = np.asarray(problem.B @ self.Z)

    def columns(self, rows: np.ndarray) -> np.ndarray:
        """Column indices of ``Z_S`` in ``Z``: all of V plus the compliance columns of ``rows``."""
        return np.concatenate([np.arange(self.r), self.r + np.asarray(rows, dtype=int)])


def pseudo_inverse_solver(H: np.ndarray):
    """``x -> H^+ x`` on the numerically non-singular subspace of symmetric positive semi-definite ``H``."""
    w, Q = np.linalg.eigh(0.5 * (H + H.T))
    keep = w > EIG_RTOL * w.max()
    Qk, wk = Q[:, keep], w[keep]
    return lambda x: Qk @ ((Qk.T @ x) / (wk[:, None] if x.ndim == 2 else wk))


def solve_working_set(H: np.ndarray, b: np.ndarray, C: np.ndarray, c: np.ndarray,
                      w0: np.ndarray | None = None) -> tuple[np.ndarray, bool, int]:
    """``min 1/2 q^T H q - b^T q  s.t.  C q >= c`` by constraint generation. Returns ``(q, feasible, passes)``."""
    solve = pseudo_inverse_solver(H)
    q0 = solve(b)
    tol = FEAS_RTOL * max(float(np.abs(c).max(initial=0.0)), 1.0)
    work = np.flatnonzero(C @ q0 - c < -tol) if w0 is None else np.union1d(w0, np.flatnonzero(C @ q0 - c < -tol))
    q = q0
    for passes in range(1, MAX_PASSES + 1):
        if work.size == 0:
            return q, True, passes
        Cw = C[work]
        HiCt = solve(Cw.T)
        M = Cw @ HiCt
        nu, ok = solve_complementarity(0.5 * (M + M.T), Cw @ q0 - c[work])
        if not ok:
            return q, False, passes
        q = q0 + HiCt @ np.maximum(nu, 0.0)
        violated = np.flatnonzero(C @ q - c < -tol)
        new = np.setdiff1d(violated, work)
        if new.size == 0:
            return q, violated.size == 0, passes
        work = np.union1d(work, new)
    return q, False, MAX_PASSES


# ---------------------------------------------------------------------------------------------------- selectors
def select_random(n_c: int, k: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n_c, size=k, replace=False))


def select_training_force(L: np.ndarray, k: int) -> np.ndarray:
    """``L``: multiplier snapshots (n_c x n_train)."""
    freq = (L > 0).mean(axis=1)
    mean = L.mean(axis=1)
    order = np.lexsort((-mean, -freq))
    return np.sort(order[:k])


def select_greedy(certifier, mus_hold: np.ndarray, L: np.ndarray, k: int, tau: float, tau_gap: float,
                  n_batches: int = 4) -> np.ndarray:
    """Certificate-driven selection on held-out parameters (no full-order solves)."""
    S = select_training_force(L, max(1, k // 4))
    n_c = L.shape[0]
    batch = max(1, (k - S.size) // n_batches)
    while S.size < k:
        score_gap = np.zeros(n_c)
        score_res = np.zeros(n_c)
        for mu in mus_hold:
            res = certifier.evaluate(mu, tau, tau_gap, correction="GS", select=S)
            score_gap += res.multiplier * res.g_hat
            score_res += np.abs(res.bxr) if res.bxr is not None else 0.0
        score_gap[S], score_res[S] = -np.inf, -np.inf
        order = np.lexsort((-score_res, -score_gap))          # missed-row contribution first, then residual
        add = order[:min(batch, k - S.size)]
        S = np.sort(np.union1d(S, add))
    return S
