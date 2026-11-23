"""Offline-online evaluation of the reject-correct-certify pipeline for R0 / R0s with the S1 multiplier
(MATH_FORMULATION_v6c.md, Sec. 8; FINAL_RESEARCH_PLAN.md, weeks 7-8).

The corrected displacement is ``u^ = V u_r + P d`` with ``P = B^+`` (as an N x n_c matrix) and ``d`` non-zero only on the
repaired / snapped rows. Its residual for a multiplier ``l`` is a linear combination of fixed vectors,
    r = f + B^T l - K u^ = sum_j phi_j f_j - sum_q theta_q K_q V u_r - sum_q theta_q K_q P d + B^T l,
so with the matrix of all these columns ``Wc = [f_1..f_Qf, K_1 V..K_Q V, K_1 P..K_Q P, B^T]`` and its representers
``R = X^-1 Wc`` (offline), every quantity of the certificate is a small dense computation online:
    ||r||^2_{X^-1} = c^T G c,  G = Wc^T R;      B X^-1 r = (B R) c;       g^ = (B V) u_r + d - c(mu);
    ||u^||_K^2 = sum_q theta_q [u_r; d]^T E_q [u_r; d],  E_q = [V P]^T K_q [V P].
Only the columns with non-zero coefficients enter (the ``K_q P`` columns of the repaired rows), so the online cost is
independent of N whenever the load is affine (``problem.load_affine()``). For a non-affine load the load enters
through one solve ``X^-1 f(mu)`` with the stored factor and O(N n_w) products per query ("hybrid" mode).

Finite precision: ``c^T G c`` is evaluated as a quadratic form and loses relative accuracy when the residual is small
compared with the individual terms (cancellation, relative error ~ eps ||c||^2 ||G|| / rho^2). The value is clamped at 0
and its rounding is bounded rigorously only in X8 (week 13); tests check agreement with the full-order evaluation.
Correction Bs (week 8): ``u^ = V u_r + Z delta + P d_c`` with ``delta`` from the projection QP in coefficients
(``H(mu) = sum_q theta_q Z^T K_q Z``, ``C = B Z``; dual problem of size ``n_c`` with equality rows on ``A^``) and the
clean-up ``d_c`` of corrections.EnergyProjection; the representers of ``K_q Z`` are added offline.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from rom_engine import solve_reduced_contact

from .active_set import by_gap, support_by_gap
from .certificate import is_accepted, quadratic_root_bound
from .corrections import EnergyProjection, RightInverse, solve_mixed_lcp
from .rom import solve_complementarity

STAGES = ("rom", "correction", "multiplier", "certificate")


@dataclass
class OnlineResult:
    u_r: np.ndarray
    lam_r: np.ndarray
    d: np.ndarray                 # constraint-space correction (n_c), u^ = V u_r + P d
    multiplier: np.ndarray        # certificate multiplier (S1 or S1-restricted)
    g_hat: np.ndarray
    delta: float
    rho_bar: float
    eta: float
    u_norm: float                 # ||u^||_K(mu)
    accepted: bool
    rom_ok: bool
    lcp_ok: bool
    seconds: dict = field(default_factory=dict)
    delta_z: np.ndarray | None = None     # projection coefficients (Bs only)


class OnlineCertifier:
    """Offline stage in the constructor; ``evaluate(mu, ...)`` is the online stage."""

    def __init__(self, problem, V: np.ndarray, W: np.ndarray, inner, right_inverse: RightInverse | None = None,
                 projection: EnergyProjection | None = None):
        """``projection`` (an EnergyProjection built for the same ``V``) enables the correction Bs online."""
        start = time.perf_counter()
        self.problem, self.inner = problem, inner
        B = sp.csr_matrix(problem.B)
        self.right_inverse = right_inverse or RightInverse.from_constraints(B)
        self.n_c, self.r = B.shape[0], V.shape[1]
        P = np.column_stack([self.right_inverse.apply(e) for e in np.eye(self.n_c)])
        affine = problem.load_affine()
        self.load_terms, self.load_phi = (affine if affine is not None else ((), None))
        self.is_affine_load = affine is not None
        Q = len(problem.stiffness_terms)
        blocks = [np.column_stack(self.load_terms)] if self.is_affine_load else []
        blocks += [Kq @ V for Kq in problem.stiffness_terms]
        blocks += [np.asarray(Kq @ P) for Kq in problem.stiffness_terms]
        blocks.append(B.T.toarray())
        self.projection = projection
        Z = projection.Z if projection is not None else np.zeros((problem.n_dof, 0))
        self.m = Z.shape[1]
        blocks += [Kq @ Z for Kq in problem.stiffness_terms]
        Wc = np.hstack(blocks)
        R = np.column_stack([inner.solve(col) for col in Wc.T])
        G = Wc.T @ R
        self.G = 0.5 * (G + G.T)
        self.BR = B @ R
        self.R_full = R if not self.is_affine_load else None          # hybrid mode needs R^T f(mu)
        # column index sets of the blocks
        nf = len(self.load_terms)
        self.idx_f = np.arange(nf)
        self.idx_v = [nf + q * self.r + np.arange(self.r) for q in range(Q)]
        off = nf + Q * self.r
        self.idx_p = [off + q * self.n_c + np.arange(self.n_c) for q in range(Q)]
        self.idx_b = off + Q * self.n_c + np.arange(self.n_c)
        off_z = off + Q * self.n_c + self.n_c
        self.idx_z = [off_z + q * self.m + np.arange(self.m) for q in range(Q)]
        self.S = self.G[np.ix_(self.idx_b, self.idx_b)]                 # B X^-1 B^T
        # energy blocks of [V P Z] and constraint-space map of V
        VPZ = np.hstack([V, P, Z])
        self.E = tuple(VPZ.T @ (Kq @ VPZ) for Kq in problem.stiffness_terms)
        self.BV = np.asarray(B @ V)
        # reduced model (affine): A_q, V^T f_j, B_r = W^T B V
        self.V, self.W = V, W
        self.A_terms = tuple(V.T @ (Kq @ V) for Kq in problem.stiffness_terms)
        self.Vf = tuple(V.T @ fj for fj in self.load_terms)
        self.B_r = W.T @ self.BV
        self.offline_seconds = time.perf_counter() - start
        self.n_columns = Wc.shape[1]

    # ------------------------------------------------------------------ online
    def evaluate(self, mu: np.ndarray, tau: float, tau_gap: float, snap: bool = True,
                 support_factor: float | None = None, correction: str = "R0s") -> OnlineResult:
        """ROM solve, correction (``R0s``, or R0 with ``snap=False``, or ``Bs``) with the gap rule, S1 (or
        S1-restricted on ``S_t`` with ``t = support_factor * tau_gap * char_gap``), certificate and acceptance, all
        from reduced quantities."""
        if correction not in ("R0s", "Bs"):
            raise ValueError("correction must be 'R0s' or 'Bs'")
        if correction == "Bs" and self.projection is None:
            raise ValueError("construct OnlineCertifier with projection=EnergyProjection(...) to use Bs")
        p, sec = self.problem, {}
        t0 = time.perf_counter()
        theta = p.theta(mu)
        c = p.offset(mu)
        A = sum(t * Aq for t, Aq in zip(theta, self.A_terms))
        if self.is_affine_load:
            phi = self.load_phi(mu)
            b = sum(ph * vf for ph, vf in zip(phi, self.Vf))
            f_mu = None
        else:
            f_mu = p.load(mu)                                              # hybrid: O(N) load evaluation
            b = self.V.T @ f_mu
        u_r, lam_r, _, rom_ok = solve_reduced_contact(A, b, self.B_r, self.W.T @ c)
        t1 = time.perf_counter()
        sec["rom"] = t1 - t0
        g_tilde = self.BV @ u_r - c
        a_hat = by_gap(g_tilde, tau_gap, p.char_gap) if snap else np.zeros(0, dtype=int)
        delta_z = np.zeros(self.m)
        g_pre = g_tilde
        if correction == "Bs":
            H = sum(t * Hq for t, Hq in zip(theta, self.projection.H_terms))
            cf = np.linalg.cholesky(0.5 * (H + H.T))
            Ct = self.projection.C.T
            HiCt = np.linalg.solve(cf.T, np.linalg.solve(cf, Ct))
            M = self.projection.C @ HiCt
            nu, _ = solve_mixed_lcp(0.5 * (M + M.T), g_tilde, a_hat)
            delta_z = HiCt @ nu
            g_pre = g_tilde + self.projection.C @ delta_z
        d = np.maximum(-g_pre, 0.0)
        d[a_hat] = -g_pre[a_hat]
        g_hat = np.maximum(g_pre + d, 0.0)
        t2 = time.perf_counter()
        sec["correction"] = t2 - t1
        # coefficient vector of a = f - K u^ over the active columns
        rows = np.flatnonzero(d)
        idx = [] if not self.is_affine_load else [self.idx_f]
        coef = [] if not self.is_affine_load else [phi]
        for q, t in enumerate(theta):
            idx += [self.idx_v[q], self.idx_p[q][rows]]
            coef += [-t * u_r, -t * d[rows]]
            if correction == "Bs":
                idx.append(self.idx_z[q])
                coef.append(-t * delta_z)
        idx, coef = np.concatenate(idx), np.concatenate(coef)
        BXa = self.BR[:, idx] @ coef
        if f_mu is not None:
            xf = self.inner.solve(f_mu)                                     # hybrid: one solve with X
            BXa = BXa + p.B @ xf
        alpha = self.inner.alpha(mu)
        q_lcp = 2.0 * BXa / alpha + g_hat
        support = np.arange(self.n_c) if support_factor is None else \
            np.union1d(support_by_gap(g_hat, support_factor * tau_gap * p.char_gap), a_hat)
        lam = np.zeros(self.n_c)
        lcp_ok = True
        if support.size:
            z, lcp_ok = solve_complementarity((2.0 / alpha) * self.S[np.ix_(support, support)], q_lcp[support])
            lam[support] = np.maximum(z, 0.0)
        t3 = time.perf_counter()
        sec["multiplier"] = t3 - t2
        # rho^2 = ||a + B^T l||^2_{X^-1}: quadratic form over the active columns plus the multiplier block
        lam_rows = np.flatnonzero(lam)
        G_aa = self.G[np.ix_(idx, idx)]
        G_ba = self.G[np.ix_(self.idx_b[lam_rows], idx)]
        rho2 = coef @ (G_aa @ coef) + 2.0 * lam[lam_rows] @ (G_ba @ coef) + \
            lam[lam_rows] @ (self.S[np.ix_(lam_rows, lam_rows)] @ lam[lam_rows])
        if f_mu is not None:
            Rf = self.R_full[:, np.concatenate([idx, self.idx_b[lam_rows]])].T @ f_mu
            full_coef = np.concatenate([coef, lam[lam_rows]])
            rho2 += 2.0 * full_coef @ Rf + f_mu @ xf
        rho_bar = float(np.sqrt(max(rho2, 0.0) / alpha))
        eta = float(lam @ g_hat)
        delta = quadratic_root_bound(rho_bar, eta)
        x = np.concatenate([u_r, d, delta_z])
        keep = np.concatenate([np.arange(self.r), self.r + rows, self.r + self.n_c + np.arange(self.m)])             if correction == "Bs" else np.concatenate([np.arange(self.r), self.r + rows])
        u_norm2 = sum(t * (x[keep] @ (Eq[np.ix_(keep, keep)] @ x[keep])) for t, Eq in zip(theta, self.E))
        u_norm = float(np.sqrt(max(u_norm2, 0.0)))
        sec["certificate"] = time.perf_counter() - t3
        return OnlineResult(u_r, lam_r, d, lam, g_hat, delta, rho_bar, eta, u_norm, is_accepted(delta, u_norm, tau),
                            bool(rom_ok), bool(lcp_ok), sec, delta_z if correction == "Bs" else None)

    def displacement(self, result: OnlineResult) -> np.ndarray:
        """Full-size ``u^ = V u_r + Z delta + B^+ d`` (offline checks and output only; not needed online)."""
        u = self.V @ result.u_r + self.right_inverse.apply(result.d)
        if result.delta_z is not None and result.delta_z.size:
            u = u + self.projection.Z @ result.delta_z
        return u
