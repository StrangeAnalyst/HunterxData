"""Databricks job entry point: value layer (action queue + value expectation).

Runs immediately downstream of scoring in the same DAB job. Reads the scored
transition + trajectory tables, builds the action queue, the expected actioning
volume and the value-expectation projection, and (when prior-period actuals are
available) settles the realized-vs-expected closed loop.
"""

from __future__ import annotations

import argparse

from ..common.action_logger import ActionLogger
from ..common.config import load_config


def main() -> None:  # pragma: no cover - Databricks runtime entry point
    parser = argparse.ArgumentParser(description="Markov value layer")
    parser.add_argument("--score-date", required=True)
    parser.add_argument("--horizon", default="6m")
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    import numpy as np

    from ..business_value.actionability import (
        build_action_queue,
        compute_downgrade_prob,
        expected_actioning_volume,
        value_expectation,
    )
    from ..common.state_contract import StateContract

    spark = SparkSession.builder.getOrCreate()
    logger = ActionLogger("jobs.value")
    config = load_config(args.config_path)
    contract = StateContract.from_config(config)
    bv = config["business_value"]
    scoring = config["scoring"]

    where = f"score_date = '{args.score_date}' AND horizon = '{args.horizon}'"
    trajectory = (
        spark.table(scoring["output"]["trajectory_table"]).where(where).toPandas()
    )
    transitions = (
        spark.table(scoring["output"]["transitions_table"]).where(where).toPandas()
    )

    # Action queue + volume.
    downgrade = compute_downgrade_prob(transitions, contract)
    # Restore the customer's origin onto the trajectory frame for the queue.
    if "origin" not in trajectory.columns:
        origins = transitions[["customer_id", "origin"]].drop_duplicates("customer_id")
        trajectory = trajectory.merge(origins, on="customer_id", how="left")
    queue = build_action_queue(trajectory, downgrade, contract, config, logger=logger)
    volume = expected_actioning_volume(queue, config)

    spark.createDataFrame(queue).write.format("delta").mode("overwrite").option(
        "replaceWhere", where
    ).saveAsTable(bv["action_queue"]["table"])
    spark.createDataFrame(volume).write.format("delta").mode("overwrite").option(
        "replaceWhere", f"score_date = '{args.score_date}'"
    ).saveAsTable(bv["actioning_volume"]["table"])

    # Value expectation from the aggregate baseline matrix + current mix.
    baseline_P, initial_mix = _aggregate_matrix_and_mix(transitions, contract, np)
    ve = value_expectation(
        baseline_P, initial_mix, contract, config, score_date=args.score_date, logger=logger
    )
    spark.createDataFrame(ve).write.format("delta").mode("overwrite").option(
        "replaceWhere", f"score_date = '{args.score_date}'"
    ).saveAsTable(bv["value_expectation"]["table"])
    logger.info("value_job.complete", score_date=args.score_date)


def _aggregate_matrix_and_mix(transitions, contract, np):  # pragma: no cover
    """Aggregate per-customer transitions into a portfolio matrix + initial mix."""
    names_to_rank = {**contract.cluster_rank, contract.absorbing_name: contract.absorbing_rank}
    n = contract.n_states
    P = np.zeros((n, n))
    counts = transitions.groupby(["origin", "dest"])["prob"].mean()
    for (o, d), p in counts.items():
        P[names_to_rank[o], names_to_rank[d]] = p
    for i in contract.transient_ranks:
        s = P[i].sum()
        if s > 0:
            P[i] /= s
    P[contract.absorbing_rank] = 0.0
    P[contract.absorbing_rank, contract.absorbing_rank] = 1.0
    # Initial mix: distribution of current origins.
    origin_counts = (
        transitions[["customer_id", "origin"]]
        .drop_duplicates("customer_id")["origin"]
        .map(names_to_rank)
        .value_counts(normalize=True)
    )
    mix = np.zeros(n)
    for rank, share in origin_counts.items():
        mix[int(rank)] = share
    return P, mix


if __name__ == "__main__":
    main()
