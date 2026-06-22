"""markov_dynamics — pure linear algebra over the transition kernel.

This module is dependency-light (numpy/pandas only) and stateless: it takes a
row-stochastic transition matrix ``P`` (transient ranks 0..4 + absorbing rank 5)
and derives the Markov / Markov-Reward-Process quantities:

* ``n_step``               — ``P^t`` (with an empirical t-step Markov-property test)
* ``stationary_distribution`` — stationary law of the transient ergodic block
* canonical partition ``P = [[Q, R], [0, I]]`` and from it
  ``N = (I - Q)^-1``       — the fundamental matrix
  ``expected_lifetime``    — ``N @ 1`` (time to absorption)
  ``absorption_prob``      — ``N @ R`` (lifetime churn probability)
  ``expected_visits``      — expected visits to each transient state
* ``clv_mrp``              — discounted CLV ``(I - gamma * Q)^-1 @ r``

Conventions: the absorbing state occupies the last index; transient states are
the leading block. All functions operate on a single matrix; vectorised
per-customer use is provided by stacking in the scoring layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def n_step(P: np.ndarray, t: int) -> np.ndarray:
    """Return the t-step transition matrix ``P^t``.

    Args:
        P: Row-stochastic (n x n) transition matrix.
        t: Non-negative integer number of steps.

    Returns:
        ``P`` raised to the ``t``-th power (``I`` when ``t == 0``).
    """
    if t < 0:
        raise ValueError("n_step requires t >= 0")
    return np.linalg.matrix_power(P, t)


def markov_property_error(
    P: np.ndarray,
    empirical_t_step: np.ndarray,
    t: int,
) -> float:
    """Max absolute deviation between ``P^t`` and an empirical t-step matrix.

    A small error supports the (approximate) Markov assumption: composing the
    one-step kernel reproduces the directly-observed t-step transitions.

    Args:
        P: One-step transition matrix.
        empirical_t_step: Empirically-estimated t-step transition matrix.
        t: The horizon ``t``.

    Returns:
        ``max |P^t - empirical_t_step|``.
    """
    return float(np.max(np.abs(n_step(P, t) - empirical_t_step)))


def partition(P: np.ndarray, n_transient: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``P`` into the canonical ``Q`` (transient) and ``R`` (to-absorbing).

    Args:
        P: Row-stochastic matrix in canonical order (transient block first).
        n_transient: Number of transient states.

    Returns:
        Tuple ``(Q, R)`` where ``Q`` is (k x k) transient-to-transient and ``R``
        is (k x (n-k)) transient-to-absorbing.
    """
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]
    return Q, R


def fundamental_matrix(Q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute the fundamental matrix ``N = (I - Q)^-1``.

    Args:
        Q: Transient-to-transient sub-stochastic block.
        eps: Singularity guard; raises if ``(I - Q)`` is near-singular.

    Returns:
        The fundamental matrix ``N`` (expected visit counts).

    Raises:
        np.linalg.LinAlgError: If ``(I - Q)`` is (near-)singular.
    """
    k = Q.shape[0]
    M = np.eye(k) - Q
    if abs(np.linalg.det(M)) < eps:
        raise np.linalg.LinAlgError("(I - Q) is near-singular; check absorbing structure")
    return np.linalg.inv(M)


@dataclass
class AbsorptionResult:
    """Container for absorption / lifetime quantities.

    Attributes:
        fundamental: ``N = (I - Q)^-1``.
        expected_lifetime: ``N @ 1`` — expected steps to absorption per origin.
        absorption_prob: ``N @ R`` — lifetime probability of each absorbing state.
        expected_visits: Expected visits per transient state (rows of ``N``).
    """

    fundamental: np.ndarray
    expected_lifetime: np.ndarray
    absorption_prob: np.ndarray
    expected_visits: np.ndarray


def absorption_analysis(P: np.ndarray, n_transient: int, eps: float = 1e-9) -> AbsorptionResult:
    """Compute fundamental matrix, expected lifetime and absorption probability.

    Args:
        P: Row-stochastic matrix in canonical order.
        n_transient: Number of transient states.
        eps: Singularity guard passed to :func:`fundamental_matrix`.

    Returns:
        An :class:`AbsorptionResult`.
    """
    Q, R = partition(P, n_transient)
    N = fundamental_matrix(Q, eps)
    expected_lifetime = N @ np.ones(n_transient)
    absorption_prob = N @ R
    return AbsorptionResult(
        fundamental=N,
        expected_lifetime=expected_lifetime,
        absorption_prob=absorption_prob,
        expected_visits=N,
    )


def stationary_distribution(P: np.ndarray, n_transient: int) -> np.ndarray:
    """Stationary distribution of the transient block, renormalised on Q.

    For an absorbing chain the unique stationary law sits on the absorbing
    state; the quantity of business interest is the quasi-stationary law of the
    transient ergodic block, obtained from the normalised left eigenvector of
    ``Q`` for its dominant eigenvalue.

    Args:
        P: Row-stochastic matrix in canonical order.
        n_transient: Number of transient states.

    Returns:
        A length-``n_transient`` non-negative vector summing to 1.
    """
    Q, _ = partition(P, n_transient)
    eigvals, eigvecs = np.linalg.eig(Q.T)
    dominant = int(np.argmax(eigvals.real))
    vec = np.abs(eigvecs[:, dominant].real)
    total = vec.sum()
    if total == 0:
        return np.full(n_transient, 1.0 / n_transient)
    return vec / total


def clv_mrp(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    n_transient: int,
    eps: float = 1e-9,
) -> np.ndarray:
    """Expected discounted CLV via the Markov Reward Process closed form.

    ``V = (I - gamma * Q)^-1 @ r`` where ``Q`` is the transient block and ``r``
    is the per-period reward attached to each transient state. The absorbing
    state contributes zero future reward by construction.

    Args:
        P: Row-stochastic matrix in canonical order.
        r: Length-``n_transient`` per-period reward vector.
        gamma: Discount factor in (0, 1).
        n_transient: Number of transient states.
        eps: Singularity guard.

    Returns:
        Length-``n_transient`` expected discounted CLV per origin state.

    Raises:
        ValueError: If ``gamma`` is not in (0, 1).
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in the open interval (0, 1)")
    Q, _ = partition(P, n_transient)
    k = Q.shape[0]
    M = np.eye(k) - gamma * Q
    if abs(np.linalg.det(M)) < eps:
        raise np.linalg.LinAlgError("(I - gamma*Q) is near-singular")
    return np.linalg.solve(M, r[:n_transient])
