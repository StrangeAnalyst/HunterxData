"""build_transition_target — destination-state labelling for the Markov chain.

The target is the **DESTINATION state conditioned on the ORIGIN** (a 6-class
categorical including the absorbing ``churned`` state), *not* a signed delta. The
delta is recoverable as ``dest_rank - origin_rank`` for reporting, but the model
target is the destination class itself.

For multi-month horizons the destination is determined by a **persistence rule**
(e.g. 6-of-9 for the 9m horizon): a customer is assigned to a destination cluster
only if they occupy it for at least ``required_months`` of the
``window_months`` look-ahead window. Churn (absorbing) takes precedence: if the
churn condition holds, the destination is ``churned`` regardless of cluster.

Works on either a pandas or Spark DataFrame; the heavy persistence aggregation is
expressed in pandas for clarity and reused by the Spark path via grouped
``applyInPandas`` in the scoring/feature jobs.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.config import HorizonSpec
from ..common.state_contract import StateContract


def build_destination_target(
    df: pd.DataFrame,
    col_origin: str,
    col_future: str,
    cluster_rank: Mapping[str, int],
    contract: StateContract,
    *,
    col_dest: str = "destination_state",
    col_dest_rank: str = "destination_rank",
    col_delta: str = "delta_rank",
) -> pd.DataFrame:
    """Build the destination-state target conditioned on origin.

    Args:
        df: Frame with an origin column and a (already persistence-resolved)
            future cluster column, both rank-encoded integers; the future column
            uses the absorbing rank for churned customers.
        col_origin: Name of the origin rank column (0..4).
        col_future: Name of the resolved future state rank column (0..5).
        cluster_rank: The ordinal cluster map (validated against the contract).
        contract: The immutable :class:`StateContract`.
        col_dest: Output destination state-name column.
        col_dest_rank: Output destination rank column.
        col_delta: Output signed delta column (dest_rank - origin_rank), with
            churn encoded as NaN since delta arithmetic does not apply to it.

    Returns:
        A copy of ``df`` with destination state name, destination rank and delta.

    Raises:
        ValueError: If the supplied ``cluster_rank`` violates the contract.
    """
    if dict(cluster_rank) != dict(contract.cluster_rank):
        raise ValueError("cluster_rank does not match the immutable state contract")

    out = df.copy()
    rank_to_name = contract.rank_to_name
    out[col_dest_rank] = out[col_future].astype(int)
    out[col_dest] = out[col_dest_rank].map(rank_to_name)

    absorbing = contract.absorbing_rank
    delta = out[col_dest_rank].to_numpy() - out[col_origin].to_numpy()
    delta = np.where(out[col_dest_rank].to_numpy() == absorbing, np.nan, delta)
    out[col_delta] = delta
    return out


def resolve_future_state_by_persistence(
    monthly_states: pd.DataFrame,
    horizon: HorizonSpec,
    contract: StateContract,
    *,
    col_id: str = "customer_id",
    col_month_offset: str = "month_offset",
    col_state_rank: str = "state_rank",
    col_is_churned: str = "is_churned",
) -> pd.DataFrame:
    """Resolve each customer's future destination via the persistence rule.

    The persistence rule: within the look-ahead window of ``window_months``, the
    destination is the modal cluster that the customer occupies for at least
    ``required_months`` months. Churn overrides: if the customer is churned at
    horizon end (or for ``min_consecutive_months``), the destination is the
    absorbing state. If no cluster meets the persistence bar, the most recent
    observed cluster in the window is used as a stable fallback.

    Args:
        monthly_states: Long frame with one row per (customer, month_offset) in
            the look-ahead window; columns include the customer id, an integer
            offset (1..window_months), the cluster rank, and a churn flag.
        horizon: The :class:`HorizonSpec` defining window/required months.
        contract: The immutable :class:`StateContract`.
        col_id: Customer id column.
        col_month_offset: Month offset column (1-based, into the future window).
        col_state_rank: Per-month cluster rank column.
        col_is_churned: Per-month churn boolean column.

    Returns:
        Frame with one row per customer and a resolved future state rank in a
        column named ``future_state_rank``.
    """
    window = monthly_states[monthly_states[col_month_offset] <= horizon.window_months]
    absorbing = contract.absorbing_rank

    resolved: list[dict[str, object]] = []
    for cust_id, grp in window.groupby(col_id):
        # Churn override.
        if bool(grp[col_is_churned].any()) and (
            int(grp[col_is_churned].sum()) >= 1
        ):
            future_rank = absorbing
        else:
            counts = grp[col_state_rank].value_counts()
            persistent = counts[counts >= horizon.required_months]
            if not persistent.empty:
                future_rank = int(persistent.idxmax())
            else:
                # Fallback: most recent month's cluster.
                latest = grp.sort_values(col_month_offset).iloc[-1]
                future_rank = int(latest[col_state_rank])
        resolved.append({col_id: cust_id, "future_state_rank": future_rank})

    return pd.DataFrame(resolved)


def diagnose_transition_distribution(
    df: pd.DataFrame,
    contract: StateContract,
    config: Mapping[str, object],
    *,
    col_origin: str = "origin_cluster",
    col_dest: str = "destination_rank",
    logger: ActionLogger | None = None,
) -> pd.DataFrame:
    """Diagnose per-origin destination distributions and warn on rare classes.

    Emits a warning (via the L3 guardrail thresholds) whenever any destination
    class accounts for less than the configured share (default 3%) within an
    origin row — these sparse cells are where calibration and partial pooling
    matter most.

    Args:
        df: Frame containing origin and destination rank columns.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads ``guardrails.l3_business`` thresholds).
        col_origin: Origin rank column.
        col_dest: Destination rank column.
        logger: Optional :class:`ActionLogger`.

    Returns:
        A tidy frame ``(origin, dest, count, share, is_rare)`` for every observed
        origin/destination pair.
    """
    warn_min = float(config["guardrails"]["l3_business"]["min_class_share_warn"])  # type: ignore[index]
    rank_to_name = contract.rank_to_name
    records: list[dict[str, object]] = []

    for origin_rank, grp in df.groupby(col_origin):
        total = len(grp)
        dist = grp[col_dest].value_counts(normalize=True)
        counts = grp[col_dest].value_counts()
        rare_for_origin: list[str] = []
        for dest_rank in contract.all_ranks:
            share = float(dist.get(dest_rank, 0.0))
            count = int(counts.get(dest_rank, 0))
            is_rare = 0.0 < share < warn_min
            if is_rare:
                rare_for_origin.append(rank_to_name[dest_rank])
            records.append(
                {
                    "origin": rank_to_name[int(origin_rank)],
                    "dest": rank_to_name[dest_rank],
                    "count": count,
                    "share": share,
                    "is_rare": is_rare,
                    "n_origin": total,
                }
            )
        if rare_for_origin and logger:
            logger.warn(
                "transition_distribution.rare_classes",
                origin=rank_to_name[int(origin_rank)],
                rare_classes=rare_for_origin,
                warn_threshold=warn_min,
                n_origin=total,
            )
    return pd.DataFrame(records)
