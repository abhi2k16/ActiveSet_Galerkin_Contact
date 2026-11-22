"""Rules for the predicted active set ``A^`` (plan Sec. 3, item 6), and the diagnostics of experiment X3.

  * ``multiplier`` : rows with ``lam~_i > tau_lambda ||lam~||_inf``.
  * ``gap``        : rows with ``(B u~ - c)_i <= tau_gap * char_gap`` (includes every penetrating row).
  * ``neighbours`` : union of the training active sets of the ``k`` nearest training parameters.
The certificate is valid for any ``A^``; only its sharpness and the correction's accuracy depend on the rule.
Status: all rules and the diagnostics done (``neighbours`` and the S1 support rule in week 5).
"""

import numpy as np

RULES = ("multiplier", "gap", "neighbours")


def by_multiplier(lam_tilde: np.ndarray, tau_lambda: float) -> np.ndarray:
    """Rows whose lifted multiplier is positive relative to its largest entry (empty if ``lam~ <= 0``)."""
    if tau_lambda < 0:
        raise ValueError("tau_lambda must be >= 0")
    peak = float(np.max(lam_tilde, initial=0.0))
    if peak <= 0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(lam_tilde > tau_lambda * peak)


def by_gap(gap_tilde: np.ndarray, tau_gap: float, char_gap: float) -> np.ndarray:
    """Rows in or near contact: ``g~_i <= tau_gap * char_gap``."""
    if tau_gap < 0 or char_gap <= 0:
        raise ValueError("need tau_gap >= 0 and char_gap > 0")
    return np.flatnonzero(gap_tilde <= tau_gap * char_gap)


def by_neighbours(mu: np.ndarray, train_mus: np.ndarray, train_active: list[np.ndarray], k: int,
                  lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Union of the training active sets of the ``k`` nearest training parameters, with distances measured after
    scaling the training box ``[lo, hi]`` to the unit cube (parameters have different units)."""
    if k < 1 or k > len(train_mus):
        raise ValueError(f"k must be in [1, {len(train_mus)}], got {k}")
    width = np.where(hi > lo, hi - lo, 1.0)
    dist = np.linalg.norm((train_mus - mu) / width, axis=1)
    nearest = np.argsort(dist)[:k]
    return np.unique(np.concatenate([np.asarray(train_active[i], dtype=int) for i in nearest]))


def support_by_gap(g_hat: np.ndarray, t: float) -> np.ndarray:
    """Support ``S_t = {i : g^_i <= t}`` of S1-restricted (paper Theorem 3.6); contains ``A^`` for a snapped
    correction, and ``A_+`` whenever ``||B e||_inf <= t``."""
    if t < 0:
        raise ValueError("t must be >= 0")
    return np.flatnonzero(g_hat <= t)


def recall_and_oversnap(active_hat: np.ndarray, strictly_active: np.ndarray,
                        contact_set: np.ndarray) -> tuple[float, float]:
    """X3 diagnostics against the full-order solution.

    recall   : share of ``A_+ = {lam_i > 0}`` covered by ``A^`` (1 if ``A_+`` is empty); recall 1 is the hypothesis
               of Proposition E3.
    oversnap : share of ``A^`` outside the true contact set ``{g_i = 0}`` (0 if ``A^`` is empty); these rows are pulled
               onto the obstacle although the true solution lifts off there (bound T2).
    """
    active_hat = np.asarray(active_hat, dtype=int)
    plus = np.asarray(strictly_active, dtype=int)
    recall = 1.0 if plus.size == 0 else float(np.isin(plus, active_hat).mean())
    oversnap = 0.0 if active_hat.size == 0 else float((~np.isin(active_hat, contact_set)).mean())
    return recall, oversnap
