"""feature_builder — point-in-time-correct feature snapshots at month-end.

Two responsibilities:

1. **Point-in-time correctness.** Every feature is observed strictly as-of the
   month-end ``snapshot_date``; the label is observed at ``snapshot_date +
   horizon`` with a configurable embargo. No future column is permitted (L2
   guardrail).

2. **Duration / path features.** A first-order Markov kernel is memoryless, but
   customer cluster dynamics are not. To mitigate the Markov violation we feed
   the kernel explicit history features that summarise *how long* and *how
   volatile* the customer's trajectory has been:

       * tenure_in_state    — months in the current origin cluster
       * previous_cluster   — origin cluster one period earlier (rank)
       * n_transitions_12m  — count of cluster changes in the trailing 12 months
       * mean_delta         — mean signed rank change over the trailing window
       * delta_variance     — variance of the signed rank change (volatility)

The pandas implementation below documents the exact logic; the Spark feature job
applies the same transformations per customer partition.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.config import HorizonSpec
from ..common.guardrails import l2_check_label_embargo, l2_check_no_future_columns


def build_path_features(
    history: pd.DataFrame,
    *,
    col_id: str = "customer_id",
    col_date: str = "snapshot_date",
    col_state_rank: str = "origin_cluster",
    trailing_months: int = 12,
) -> pd.DataFrame:
    """Compute duration/path features from a customer's monthly cluster history.

    Args:
        history: Long frame, one row per (customer, month-end), sorted-able by
            date, containing the customer's cluster rank at each month-end.
        col_id: Customer id column.
        col_date: Month-end snapshot date column.
        col_state_rank: Cluster rank column at each month-end.
        trailing_months: Trailing window (months) for transition/volatility
            features.

    Returns:
        One row per (customer, snapshot_date) with the five path features plus
        the keys, suitable for joining onto the base feature snapshot.
    """
    hist = history.sort_values([col_id, col_date]).copy()
    grp = hist.groupby(col_id, group_keys=False)

    # previous_cluster: rank at the prior month-end.
    hist["previous_cluster"] = grp[col_state_rank].shift(1)

    # tenure_in_state: consecutive months in the current cluster (inclusive).
    changed = grp[col_state_rank].transform(lambda s: s.ne(s.shift(1)).cumsum())
    hist["_run_id"] = changed
    hist["tenure_in_state"] = (
        hist.groupby([col_id, "_run_id"]).cumcount() + 1
    ).astype(int)

    # signed delta vs previous month-end.
    hist["_delta"] = grp[col_state_rank].diff()

    # Trailing-window aggregates computed via a rolling window over rows.
    def _trailing(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(col_date)
        deltas = frame["_delta"]
        # n_transitions: number of non-zero rank changes in the trailing window.
        is_change = (deltas.fillna(0) != 0).astype(int)
        frame["n_transitions_12m"] = (
            is_change.rolling(trailing_months, min_periods=1).sum().astype(int)
        )
        frame["mean_delta"] = deltas.rolling(trailing_months, min_periods=1).mean()
        frame["delta_variance"] = (
            deltas.rolling(trailing_months, min_periods=2).var().fillna(0.0)
        )
        return frame

    hist = hist.groupby(col_id, group_keys=False).apply(_trailing)

    out_cols = [
        col_id,
        col_date,
        "previous_cluster",
        "tenure_in_state",
        "n_transitions_12m",
        "mean_delta",
        "delta_variance",
    ]
    result = hist[out_cols].copy()
    # First-ever observation has no history: stabilise with neutral defaults.
    result["previous_cluster"] = result["previous_cluster"].fillna(
        hist[col_state_rank]
    )
    result["mean_delta"] = result["mean_delta"].fillna(0.0)
    return result


def build_snapshot(
    base_features: pd.DataFrame,
    path_features: pd.DataFrame,
    config: Mapping[str, object],
    *,
    logger: ActionLogger | None = None,
) -> pd.DataFrame:
    """Join base + path features into a single point-in-time snapshot frame.

    Args:
        base_features: Behavioural/value features observed as-of the snapshot.
        path_features: Output of :func:`build_path_features`.
        config: Loaded config (reads the ``features`` block + L2 guardrail keys).
        logger: Optional :class:`ActionLogger`.

    Returns:
        The merged snapshot frame, with the L2 no-future-column guardrail
        enforced over the assembled feature set.
    """
    feats = config["features"]  # type: ignore[index]
    id_col = feats["customer_id_column"]
    date_col = feats["snapshot_date_column"]

    snapshot = base_features.merge(path_features, on=[id_col, date_col], how="left")

    feature_cols = list(feats["base_features"]) + list(feats["path_features"])
    if config["guardrails"]["l2_leakage"].get("enforce_point_in_time", True):  # type: ignore[index]
        l2_check_no_future_columns(feature_cols, logger=logger)
    if logger:
        logger.info(
            "feature_snapshot.built",
            n_rows=len(snapshot),
            n_features=len(feature_cols),
        )
    return snapshot


def attach_label_with_embargo(
    snapshot: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: HorizonSpec,
    config: Mapping[str, object],
    *,
    logger: ActionLogger | None = None,
) -> pd.DataFrame:
    """Attach the horizon label and enforce the point-in-time embargo (L2).

    The label observation date must be at least ``label_embargo_days`` after the
    snapshot date; the join is on customer id and snapshot date and assumes the
    label frame already references the correct future window.

    Args:
        snapshot: Point-in-time feature snapshot.
        labels: Label frame keyed by customer id + snapshot date with a
            ``label_date`` column for embargo verification.
        horizon: The :class:`HorizonSpec` (drives the look-ahead months).
        config: Loaded config (reads training/PIT + guardrail keys).
        logger: Optional :class:`ActionLogger`.

    Returns:
        The labelled frame.
    """
    feats = config["features"]  # type: ignore[index]
    id_col = feats["customer_id_column"]
    date_col = feats["snapshot_date_column"]
    pit = config["training"]["point_in_time"]  # type: ignore[index]
    embargo_days = int(pit["label_embargo_days"])

    merged = snapshot.merge(labels, on=[id_col, date_col], how="inner")

    if config["guardrails"]["l2_leakage"].get("enforce_label_embargo", True):  # type: ignore[index]
        overlap = 0
        if "label_date" in merged.columns:
            gap_days = (
                pd.to_datetime(merged["label_date"]) - pd.to_datetime(merged[date_col])
            ).dt.days
            min_required = horizon.months_ahead * 28 - embargo_days
            overlap = int((gap_days < min_required).sum())
        l2_check_label_embargo(
            date_col, "label_date", embargo_days, overlap, logger=logger
        )
    return merged


def feature_matrix(
    df: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, Sequence[str]]:
    """Return the model feature matrix and the ordered feature column list.

    Args:
        df: A labelled or scoring frame containing all configured features.
        config: Loaded config (reads the ``features`` block).

    Returns:
        Tuple of (feature frame restricted to feature columns, feature names).
    """
    feats = config["features"]  # type: ignore[index]
    feature_cols = list(feats["base_features"]) + list(feats["path_features"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature_matrix: missing configured feature columns {missing}")
    return df[feature_cols].copy(), feature_cols
