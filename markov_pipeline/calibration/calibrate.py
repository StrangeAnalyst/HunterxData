"""calibrate — mandatory per-row probability calibration.

Composition multiplies the transition matrix many times (``P^t``, fundamental
matrix, CLV), so small miscalibrations compound. Calibration therefore matters
*more* than raw accuracy here. Each origin row gets its own calibrator
(isotonic by default, Platt/sigmoid optional), applied per destination column in
a one-vs-rest fashion and then renormalised.

Quality is validated with reliability diagrams and the Expected Calibration
Error (ECE) per transition; the per-row ECE must stay under the configured
acceptance threshold (a BAU gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.state_contract import StateContract


@dataclass
class RowCalibrator:
    """Per-row, per-destination-column calibrator (isotonic or Platt).

    Implements the kernel's ``RowCalibrator`` protocol via :meth:`transform`.

    Attributes:
        origin_rank: Origin state this calibrator belongs to.
        method: ``"isotonic"`` or ``"platt"``.
        n_states: Number of destination columns.
        calibrators_: Per-column fitted 1-D calibrators (or ``None``).
    """

    origin_rank: int
    method: str
    n_states: int
    calibrators_: list[Any] = field(default_factory=list)

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "RowCalibrator":
        """Fit one 1-D calibrator per destination column (one-vs-rest).

        Args:
            raw_proba: (n x n_states) uncalibrated row-model probabilities.
            y_true: (n,) true destination ranks.

        Returns:
            ``self``.
        """
        self.calibrators_ = []
        for col in range(self.n_states):
            target = (y_true == col).astype(int)
            scores = raw_proba[:, col]
            # Need both classes present to calibrate; otherwise pass through.
            if target.min() == target.max():
                self.calibrators_.append(None)
                continue
            self.calibrators_.append(self._fit_column(scores, target))
        return self

    def _fit_column(self, scores: np.ndarray, target: np.ndarray) -> Any:
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(max_iter=1000)
            lr.fit(scores.reshape(-1, 1), target)
            return lr
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores, target)
        return iso

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply per-column calibration and renormalise each row to sum to 1."""
        if not self.calibrators_:
            return probabilities
        out = np.zeros_like(probabilities, dtype=float)
        for col in range(self.n_states):
            cal = self.calibrators_[col]
            scores = probabilities[:, col]
            if cal is None:
                out[:, col] = scores
            elif self.method == "platt":
                out[:, col] = cal.predict_proba(scores.reshape(-1, 1))[:, 1]
            else:
                out[:, col] = cal.predict(scores)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums


def expected_calibration_error(
    proba: np.ndarray,
    y_true: np.ndarray,
    target_class: int,
    n_bins: int = 10,
) -> float:
    """Compute one-vs-rest ECE for a single destination class.

    Args:
        proba: (n x n_states) probabilities.
        y_true: (n,) true destination ranks.
        target_class: The destination column to evaluate.
        n_bins: Number of equal-width probability bins.

    Returns:
        The expected calibration error in [0, 1].
    """
    p = proba[:, target_class]
    y = (y_true == target_class).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_table(
    proba: np.ndarray,
    y_true: np.ndarray,
    target_class: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a reliability-diagram table (mean confidence vs empirical accuracy)."""
    p = proba[:, target_class]
    y = (y_true == target_class).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin": b,
                "bin_lower": bins[b],
                "bin_upper": bins[b + 1],
                "count": int(mask.sum()),
                "mean_confidence": float(p[mask].mean()) if mask.any() else np.nan,
                "empirical_accuracy": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibrate_rows(
    row_models: Mapping[int, Any],
    calib_df: pd.DataFrame,
    feature_columns: list[str],
    contract: StateContract,
    config: Mapping[str, object],
    *,
    col_origin: str = "origin_cluster",
    col_dest: str = "destination_rank",
    logger: ActionLogger | None = None,
) -> dict[int, RowCalibrator]:
    """Fit a per-row calibrator for every origin model on a calibration split.

    Also computes per-row ECE before and after calibration and logs whether the
    acceptance threshold is met (BAU gate).

    Args:
        row_models: Mapping origin_rank -> fitted row model (predict_proba).
        calib_df: Held-out calibration frame (not used for training).
        feature_columns: Ordered feature columns.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads the ``calibration`` block).
        col_origin: Origin rank column.
        col_dest: Destination rank column.
        logger: Optional :class:`ActionLogger`.

    Returns:
        Mapping origin_rank -> fitted :class:`RowCalibrator`.
    """
    logger = logger or ActionLogger("calibration")
    cal_cfg = config["calibration"]  # type: ignore[index]
    method = str(cal_cfg["method"])
    n_bins = int(cal_cfg["reliability"]["n_bins"])
    max_ece = float(cal_cfg["acceptance"]["max_ece"])

    calibrators: dict[int, RowCalibrator] = {}
    for origin_rank, model in row_models.items():
        origin_df = calib_df[calib_df[col_origin] == origin_rank]
        if origin_df.empty:
            logger.warn("calibrate.empty_origin", origin_rank=origin_rank)
            continue
        x = origin_df[feature_columns]
        y = origin_df[col_dest].astype(int).to_numpy()
        raw = model.predict_proba(x)

        calibrator = RowCalibrator(origin_rank, method, contract.n_states).fit(raw, y)
        calibrated = calibrator.transform(raw)

        ece_before = _mean_ece(raw, y, contract, n_bins)
        ece_after = _mean_ece(calibrated, y, contract, n_bins)
        logger.info(
            "calibrate.row",
            origin_rank=origin_rank,
            method=method,
            ece_before=round(ece_before, 5),
            ece_after=round(ece_after, 5),
            passes_acceptance=bool(ece_after <= max_ece),
        )
        if ece_after > max_ece:
            logger.warn(
                "calibrate.ece_above_threshold",
                origin_rank=origin_rank,
                ece_after=round(ece_after, 5),
                max_ece=max_ece,
            )
        calibrators[origin_rank] = calibrator
        _log_calibration_to_mlflow(origin_rank, method, ece_before, ece_after)

    logger.flush_to_mlflow()
    return calibrators


def _mean_ece(
    proba: np.ndarray, y: np.ndarray, contract: StateContract, n_bins: int
) -> float:
    eces = [
        expected_calibration_error(proba, y, c, n_bins) for c in contract.all_ranks
    ]
    return float(np.mean(eces))


def _log_calibration_to_mlflow(
    origin_rank: int, method: str, ece_before: float, ece_after: float
) -> None:  # pragma: no cover - MLflow side effect
    try:
        import mlflow

        mlflow.log_metrics(
            {
                f"ece_before_origin_{origin_rank}": ece_before,
                f"ece_after_origin_{origin_rank}": ece_after,
            }
        )
        mlflow.set_tag(f"calibration_method_origin_{origin_rank}", method)
    except Exception:
        pass
