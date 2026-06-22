"""Layered guardrails (L1 / L2 / L3) for the Markov pipeline.

The guardrails are intentionally separated by concern so violations are easy to
triage and route:

* **L1 — Contract / schema**: the state space and stochastic structure of the
  transition matrix. Violations here are hard failures (the math is wrong).
* **L2 — Leakage / point-in-time**: no future information may enter a snapshot.
  Violations are hard failures (the model is invalid).
* **L3 — Business plausibility**: soft, advisory warnings (rare classes, CLV
  sign, finite lifetime). These warn and surface, they do not necessarily stop
  the run.

Every check accepts the loaded config so thresholds stay config-driven.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .action_logger import ActionLogger
from .state_contract import StateContract


class GuardrailViolation(Exception):
    """Raised when an L1/L2 (hard) guardrail is violated."""


# ===================================================================== #
# L1 — CONTRACT / SCHEMA
# ===================================================================== #
def l1_check_row_stochastic(
    matrix: np.ndarray,
    config: Mapping[str, object],
    logger: ActionLogger | None = None,
) -> None:
    """Assert that every row of ``matrix`` sums to 1 within tolerance.

    Args:
        matrix: An (n_states x n_states) transition matrix.
        config: Loaded config (reads ``guardrails.l1_contract.row_sum_atol``).
        logger: Optional :class:`ActionLogger`.

    Raises:
        GuardrailViolation: If any row does not sum to 1 within tolerance, or
            negative probabilities are present.
    """
    atol = float(config["guardrails"]["l1_contract"]["row_sum_atol"])  # type: ignore[index]
    if np.any(matrix < -atol):
        raise GuardrailViolation("L1: transition matrix contains negative probabilities")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=atol))[0].tolist()
        raise GuardrailViolation(f"L1: rows not stochastic (sum != 1) at indices {bad}")
    if logger:
        logger.info("l1.row_stochastic.ok", n_rows=int(matrix.shape[0]), atol=atol)


def l1_check_absorbing(
    matrix: np.ndarray,
    contract: StateContract,
    logger: ActionLogger | None = None,
) -> None:
    """Assert the absorbing (``churned``) row is a unit vector on itself.

    Raises:
        GuardrailViolation: If the absorbing row is not ``e_churned``.
    """
    rank = contract.absorbing_rank
    expected = np.zeros(matrix.shape[1])
    expected[rank] = 1.0
    if not np.allclose(matrix[rank], expected, atol=1e-8):
        raise GuardrailViolation(
            f"L1: absorbing state '{contract.absorbing_name}' (rank {rank}) "
            "is not absorbing (row != unit vector)"
        )
    if logger:
        logger.info("l1.absorbing.ok", absorbing_rank=rank)


def l1_check_state_contract(
    cluster_rank: Mapping[str, int],
    contract: StateContract,
    logger: ActionLogger | None = None,
) -> None:
    """Assert a supplied cluster_rank map matches the immutable contract.

    Raises:
        GuardrailViolation: If the ordinal map differs from the contract.
    """
    if dict(cluster_rank) != dict(contract.cluster_rank):
        raise GuardrailViolation(
            "L1: cluster_rank map does not match the state contract (reordering "
            "the ordinal hierarchy is forbidden)"
        )
    if logger:
        logger.info("l1.state_contract.ok")


# ===================================================================== #
# L2 — LEAKAGE / POINT-IN-TIME
# ===================================================================== #
def l2_check_no_future_columns(
    feature_columns: Sequence[str],
    forbidden_substrings: Sequence[str] = ("future", "next_", "_t1", "label", "target"),
    logger: ActionLogger | None = None,
) -> None:
    """Heuristic leakage guard: forbid obviously forward-looking feature names.

    Args:
        feature_columns: Names of columns used as model features.
        forbidden_substrings: Substrings that indicate a leaked future signal.
        logger: Optional :class:`ActionLogger`.

    Raises:
        GuardrailViolation: If any feature name looks forward-looking.
    """
    offending = [
        col
        for col in feature_columns
        if any(sub in col.lower() for sub in forbidden_substrings)
    ]
    if offending:
        raise GuardrailViolation(f"L2: potential leakage in feature columns {offending}")
    if logger:
        logger.info("l2.no_future_columns.ok", n_features=len(feature_columns))


def l2_check_label_embargo(
    snapshot_date_col: str,
    label_date_col: str,
    embargo_days: int,
    overlap_count: int,
    logger: ActionLogger | None = None,
) -> None:
    """Assert no train rows violate the label embargo window.

    Args:
        snapshot_date_col: Name of the feature snapshot date column (for logs).
        label_date_col: Name of the label observation date column (for logs).
        embargo_days: Required gap between snapshot and label observation.
        overlap_count: Number of rows found within the embargo window (computed
            by the caller against the actual frame).
        logger: Optional :class:`ActionLogger`.

    Raises:
        GuardrailViolation: If any rows fall inside the embargo window.
    """
    if overlap_count > 0:
        raise GuardrailViolation(
            f"L2: {overlap_count} rows violate the {embargo_days}-day label embargo "
            f"between {snapshot_date_col} and {label_date_col}"
        )
    if logger:
        logger.info("l2.label_embargo.ok", embargo_days=embargo_days)


# ===================================================================== #
# L3 — BUSINESS PLAUSIBILITY (advisory)
# ===================================================================== #
def l3_check_class_shares(
    shares: Mapping[str, float],
    origin_name: str,
    config: Mapping[str, object],
    logger: ActionLogger | None = None,
) -> list[str]:
    """Warn when any destination class share is below the configured band.

    Args:
        shares: Mapping of destination state name -> observed share for one origin.
        origin_name: The origin state these shares belong to.
        config: Loaded config (reads ``guardrails.l3_business`` thresholds).
        logger: Optional :class:`ActionLogger`.

    Returns:
        List of destination names that fell below the warn threshold.
    """
    warn_min = float(config["guardrails"]["l3_business"]["min_class_share_warn"])  # type: ignore[index]
    rare = [name for name, share in shares.items() if 0.0 < share < warn_min]
    if rare and logger:
        logger.warn(
            "l3.rare_destination_classes",
            origin=origin_name,
            rare_classes=rare,
            warn_threshold=warn_min,
        )
    return rare


def l3_check_clv_nonnegative(
    clv_values: np.ndarray,
    config: Mapping[str, object],
    logger: ActionLogger | None = None,
) -> bool:
    """Warn when CLV estimates are negative (implausible for a value model)."""
    if not config["guardrails"]["l3_business"].get("clv_non_negative", True):  # type: ignore[index]
        return True
    n_negative = int(np.sum(clv_values < 0))
    if n_negative and logger:
        logger.warn("l3.negative_clv", n_negative=n_negative)
    return n_negative == 0


def l3_check_finite_lifetime(
    lifetimes: np.ndarray,
    logger: ActionLogger | None = None,
) -> bool:
    """Warn when expected lifetimes are non-finite (degenerate fundamental matrix)."""
    n_bad = int(np.sum(~np.isfinite(lifetimes)))
    if n_bad and logger:
        logger.warn("l3.nonfinite_lifetime", n_bad=n_bad)
    return n_bad == 0
