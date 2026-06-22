"""batch_score — monthly Spark batch scoring + trajectory derivation.

Two jobs run back to back each month:

1. **Transition scoring.** Load the registered :class:`MarkovKernel` pyfunc,
   score the point-in-time feature snapshot, and write a **long-format** Delta
   table: ``(customer_id, score_date, horizon, origin, dest, prob)``.

2. **Trajectory derivation.** Compose each customer's 6x6 kernel into the
   Markov-Reward quantities and write ``customer_trajectory``:
   ``(customer_id, score_date, horizon, expected_lifetime, churn_prob,
   expected_clv, prob_reach_high_power, predicted_destination,
   predicted_delta)``.

The Spark entry points are thin: they broadcast the kernel and a small reward
vector and apply the pure-pandas/numpy composition per customer partition via
``applyInPandas``. The pandas core functions are unit-testable without Spark.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.config import get_horizon, load_config
from ..common.state_contract import StateContract
from ..composition.markov_dynamics import absorption_analysis, clv_mrp, n_step
from ..kernel.markov_kernel import MarkovKernel


# --------------------------------------------------------------------------- #
# Pure core: per-customer trajectory from a single kernel
# --------------------------------------------------------------------------- #
def trajectory_from_matrix(
    P: np.ndarray,
    origin_rank: int,
    reward_vector: np.ndarray,
    gamma: float,
    contract: StateContract,
    horizon_steps: int,
) -> dict[str, float]:
    """Derive trajectory metrics for one customer from their 6x6 kernel.

    Args:
        P: The customer's row-stochastic 6x6 transition matrix.
        origin_rank: The customer's current (origin) transient rank.
        reward_vector: Per-transient-state reward vector (length n_transient).
        gamma: Discount factor.
        contract: The immutable :class:`StateContract`.
        horizon_steps: Number of steps for the horizon-specific destination/churn.

    Returns:
        Dict with expected_lifetime, churn_prob, expected_clv,
        prob_reach_high_power, predicted_destination (name), predicted_delta.
    """
    n_transient = len(contract.transient_ranks)
    absorbing = contract.absorbing_rank

    absorption = absorption_analysis(P, n_transient)
    expected_lifetime = float(absorption.expected_lifetime[origin_rank])
    expected_clv = float(clv_mrp(P, reward_vector, gamma, n_transient)[origin_rank])

    # Horizon-specific distribution from the current origin.
    p_h = n_step(P, horizon_steps)[origin_rank]
    churn_prob = float(p_h[absorbing])
    prob_reach_high_power = float(p_h[contract.growth_target_rank])

    predicted_dest_rank = int(np.argmax(p_h))
    predicted_destination = contract.rank_to_name[predicted_dest_rank]
    predicted_delta = (
        float("nan")
        if predicted_dest_rank == absorbing
        else float(predicted_dest_rank - origin_rank)
    )
    return {
        "expected_lifetime": expected_lifetime,
        "churn_prob": churn_prob,
        "expected_clv": expected_clv,
        "prob_reach_high_power": prob_reach_high_power,
        "predicted_destination": predicted_destination,
        "predicted_delta": predicted_delta,
    }


def score_partition(
    features: pd.DataFrame,
    kernel: MarkovKernel,
    config: Mapping[str, object],
    horizon_name: str,
    *,
    col_origin: str = "origin_cluster",
    col_id: str = "customer_id",
    col_date: str = "score_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a pandas partition: returns (long transitions, trajectory) frames.

    Args:
        features: Feature frame for a set of customers (one row each), including
            the origin rank, customer id and score date columns.
        kernel: The fitted :class:`MarkovKernel`.
        config: Loaded config.
        horizon_name: Horizon to compute (e.g. ``"6m"``).
        col_origin: Origin rank column.
        col_id: Customer id column.
        col_date: Score date column.

    Returns:
        Tuple of (long-format transition frame, trajectory frame).
    """
    contract = StateContract.from_config(config)
    horizon = get_horizon(config, horizon_name)
    gamma = float(config["mrp"]["gamma"])  # type: ignore[index]

    # Reward vector: per-period reward per transient state. Sourced from the mean
    # configured reward column by origin if available, else uniform from config.
    reward_col = str(config["mrp"]["reward_column"])  # type: ignore[index]
    reward_vector = _reward_vector(features, contract, reward_col, col_origin)

    stack = kernel.transition_matrices(features)
    names = contract.rank_to_name

    long_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    ids = features[col_id].to_numpy()
    dates = features[col_date].to_numpy()
    origins = features[col_origin].astype(int).to_numpy()

    for i in range(len(features)):
        P = stack[i]
        # Long-format transitions for this customer/horizon.
        for o in contract.all_ranks:
            for d in contract.all_ranks:
                long_rows.append(
                    {
                        col_id: ids[i],
                        col_date: dates[i],
                        "horizon": horizon_name,
                        "origin": names[o],
                        "dest": names[d],
                        "prob": float(P[o, d]),
                    }
                )
        metrics = trajectory_from_matrix(
            P, origins[i], reward_vector, gamma, contract, horizon.months_ahead
        )
        traj_rows.append(
            {col_id: ids[i], col_date: dates[i], "horizon": horizon_name, **metrics}
        )

    return pd.DataFrame(long_rows), pd.DataFrame(traj_rows)


def _reward_vector(
    features: pd.DataFrame,
    contract: StateContract,
    reward_col: str,
    col_origin: str,
) -> np.ndarray:
    """Build a per-transient-state reward vector from the snapshot.

    Uses the mean of the configured reward column within each origin state as the
    per-period reward. Falls back to zeros where an origin is absent in this
    partition (composition still well defined; CLV simply lower).
    """
    n_transient = len(contract.transient_ranks)
    reward = np.zeros(n_transient)
    if reward_col in features.columns:
        means = features.groupby(col_origin)[reward_col].mean()
        for rank in contract.transient_ranks:
            if rank in means.index:
                reward[rank] = float(means.loc[rank])
    return reward


# --------------------------------------------------------------------------- #
# Spark orchestration (thin)
# --------------------------------------------------------------------------- #
def run_batch_scoring(
    spark: Any,
    score_date: str,
    horizon_name: str,
    config_path: str | None = None,
    *,
    logger: ActionLogger | None = None,
) -> None:  # pragma: no cover - requires a Spark + MLflow + Delta runtime
    """Monthly Spark batch scoring entry point.

    Loads the registered pyfunc kernel, reads the point-in-time feature snapshot
    for ``score_date``, scores per partition and writes the long-format
    transition table and the trajectory table to Delta (Unity Catalog).

    Args:
        spark: Active ``SparkSession``.
        score_date: Month-end score date (``YYYY-MM-DD``).
        horizon_name: Horizon to score.
        config_path: Optional explicit config path.
        logger: Optional :class:`ActionLogger`.
    """
    import mlflow

    logger = logger or ActionLogger("scoring")
    config = load_config(config_path)
    scoring_cfg = config["scoring"]
    feats_cfg = config["features"]

    # Load the registered kernel pyfunc (Production stage by default).
    model_uri = f"models:/{scoring_cfg['registered_model_name']}/{scoring_cfg['model_stage']}"
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    kernel: MarkovKernel = pyfunc_model._model_impl._kernel  # underlying object

    fs = feats_cfg["feature_store"]
    feature_table = f"{fs['catalog']}.{fs['schema']}.{fs['table']}"
    sdf = (
        spark.table(feature_table)
        .where(f"{feats_cfg['snapshot_date_column']} = '{score_date}'")
        .withColumn("score_date", spark.sql(f"select to_date('{score_date}')").first()[0])
    )

    feature_cols = list(feats_cfg["base_features"]) + list(feats_cfg["path_features"])
    out_cols = scoring_cfg["long_format_columns"]
    traj_cols = scoring_cfg["trajectory_columns"]

    long_schema = ", ".join(
        [
            "customer_id string",
            "score_date date",
            "horizon string",
            "origin string",
            "dest string",
            "prob double",
        ]
    )
    traj_schema = (
        "customer_id string, score_date date, horizon string, "
        "expected_lifetime double, churn_prob double, expected_clv double, "
        "prob_reach_high_power double, predicted_destination string, predicted_delta double"
    )

    broadcast_kernel = spark.sparkContext.broadcast(kernel)
    broadcast_config = spark.sparkContext.broadcast(config)

    def _score(pdf: pd.DataFrame) -> pd.DataFrame:
        long_df, _ = score_partition(
            pdf, broadcast_kernel.value, broadcast_config.value, horizon_name
        )
        return long_df[out_cols]

    def _traj(pdf: pd.DataFrame) -> pd.DataFrame:
        _, traj_df = score_partition(
            pdf, broadcast_kernel.value, broadcast_config.value, horizon_name
        )
        return traj_df[traj_cols]

    long_out = sdf.groupBy(feats_cfg["customer_id_column"]).applyInPandas(
        _score, schema=long_schema
    )
    traj_out = sdf.groupBy(feats_cfg["customer_id_column"]).applyInPandas(
        _traj, schema=traj_schema
    )

    (
        long_out.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"score_date = '{score_date}' AND horizon = '{horizon_name}'")
        .saveAsTable(scoring_cfg["output"]["transitions_table"])
    )
    (
        traj_out.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"score_date = '{score_date}' AND horizon = '{horizon_name}'")
        .saveAsTable(scoring_cfg["output"]["trajectory_table"])
    )
    logger.info(
        "batch_scoring.complete",
        score_date=score_date,
        horizon=horizon_name,
        transitions_table=scoring_cfg["output"]["transitions_table"],
        trajectory_table=scoring_cfg["output"]["trajectory_table"],
    )
    logger.flush_to_mlflow()
