"""Rules for the predicted active set ``A^`` (plan Sec. 3, item 6), and the diagnostics of experiment X3.

  * ``multiplier`` : rows with ``lam~_i > tau_lambda ||lam~||_inf``.
  * ``gap``        : rows with ``(B u~ - c)_i <= tau_gap * char_gap`` (includes every penetrating row).
  * ``neighbours`` : union of the training active sets of the ``k`` nearest training parameters.
The certificate is valid for any ``A^``; only its sharpness and the correction's accuracy depend on the rule.
Status: ``multiplier``, ``gap`` and the diagnostics done (week 1); ``neighbours`` in weeks 5-6.
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


def by_neighbours(mu: np.ndarray, train_mus: np.ndarray, train_active: list[np.ndarray], k: int) -> np.ndarray:
    raise NotImplementedError("weeks 5-6")


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
