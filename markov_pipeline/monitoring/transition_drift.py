"""transition_drift — month-over-month drift on the transition matrix.

Transition drift is the most business-relevant drift signal for this system and
typically *precedes* feature drift: the relationships between clusters shift
before any single feature distribution looks alarming. We therefore monitor:

* **Per-row PSI** of the transition matrix month-over-month. Each origin row is
  a distribution over destinations; PSI per row localises *where* dynamics moved.
* **n-step backtest error** — does ``P^t`` from an earlier month still reproduce
  the directly-observed t-step transitions? A breach means the Markov composition
  is no longer trustworthy.

A retraining trigger fires when either PSI or backtest error breaches the
configured threshold, subject to a quarterly floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.state_contract import StateContract
from ..composition.markov_dynamics import markov_property_error


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two discrete distributions.

    ``PSI = sum((a - e) * ln(a / e))`` over bins, with epsilon flooring to keep
    the logarithm finite for empty cells.

    Args:
        expected: Baseline distribution (sums to ~1).
        actual: Current distribution (sums to ~1).
        eps: Floor applied to both distributions.

    Returns:
        The PSI value (>= 0).
    """
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class DriftReport:
    """Per-row drift report for one month-over-month comparison.

    Attributes:
        per_row_psi: Mapping origin name -> PSI for that row.
        max_psi: Largest per-row PSI.
        breached_rows: Origin names whose PSI breached the threshold.
        warned_rows: Origin names whose PSI exceeded the warn threshold.
        backtest_error: n-step composition error (or NaN if not evaluated).
        retrain_triggered: Whether a retraining trigger fired.
    """

    per_row_psi: dict[str, float]
    max_psi: float
    breached_rows: list[str]
    warned_rows: list[str]
    backtest_error: float
    retrain_triggered: bool


def transition_matrix_psi(
    baseline_P: np.ndarray,
    current_P: np.ndarray,
    contract: StateContract,
    config: Mapping[str, object],
    *,
    backtest_t: int | None = None,
    empirical_t_step: np.ndarray | None = None,
    months_since_last_train: int | None = None,
    logger: ActionLogger | None = None,
) -> DriftReport:
    """Compute per-row PSI + backtest error and evaluate the retraining trigger.

    Args:
        baseline_P: Prior-month aggregate transition matrix.
        current_P: Current-month aggregate transition matrix.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads the ``monitoring`` block).
        backtest_t: Optional horizon for the n-step backtest.
        empirical_t_step: Optional empirical t-step matrix for the backtest.
        months_since_last_train: Optional age of the model for the quarterly floor.
        logger: Optional :class:`ActionLogger`.

    Returns:
        A :class:`DriftReport`.
    """
    logger = logger or ActionLogger("monitoring")
    mon = config["monitoring"]  # type: ignore[index]
    warn_t = float(mon["psi"]["warn_threshold"])
    breach_t = float(mon["psi"]["breach_threshold"])
    backtest_tol = float(mon["backtest"]["n_step_error_tolerance"])
    trigger = mon["retraining_trigger"]
    quarterly_floor = int(trigger["quarterly_floor_months"])

    names = contract.rank_to_name
    per_row: dict[str, float] = {}
    breached: list[str] = []
    warned: list[str] = []
    for origin in contract.transient_ranks:
        psi = population_stability_index(baseline_P[origin], current_P[origin])
        per_row[names[origin]] = psi
        if psi >= breach_t:
            breached.append(names[origin])
        elif psi >= warn_t:
            warned.append(names[origin])

    backtest_error = float("nan")
    if backtest_t is not None and empirical_t_step is not None:
        backtest_error = markov_property_error(current_P, empirical_t_step, backtest_t)

    psi_breach = bool(breached) and bool(trigger["on_psi_breach"])
    backtest_breach = (
        not np.isnan(backtest_error)
        and backtest_error > backtest_tol
        and bool(trigger["on_backtest_breach"])
    )
    floor_breach = (
        months_since_last_train is not None
        and months_since_last_train >= quarterly_floor
    )
    retrain = psi_breach or backtest_breach or floor_breach

    max_psi = max(per_row.values()) if per_row else 0.0
    logger.info(
        "transition_drift.evaluated",
        max_psi=round(max_psi, 4),
        breached_rows=breached,
        warned_rows=warned,
        backtest_error=None if np.isnan(backtest_error) else round(backtest_error, 4),
        retrain_triggered=retrain,
    )
    if retrain:
        logger.warn(
            "transition_drift.retrain_trigger",
            psi_breach=psi_breach,
            backtest_breach=backtest_breach,
            quarterly_floor_breach=floor_breach,
        )
    logger.flush_to_mlflow()

    return DriftReport(
        per_row_psi=per_row,
        max_psi=max_psi,
        breached_rows=breached,
        warned_rows=warned,
        backtest_error=backtest_error,
        retrain_triggered=retrain,
    )


def drift_report_to_frame(report: DriftReport, score_date: str) -> pd.DataFrame:
    """Flatten a :class:`DriftReport` into a tidy frame for the drift Delta table."""
    return pd.DataFrame(
        [
            {
                "score_date": score_date,
                "origin": origin,
                "psi": psi,
                "breached": origin in report.breached_rows,
                "warned": origin in report.warned_rows,
                "backtest_error": report.backtest_error,
                "retrain_triggered": report.retrain_triggered,
            }
            for origin, psi in report.per_row_psi.items()
        ]
    )
