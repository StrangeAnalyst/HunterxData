"""Databricks job entry point: weekly transition-drift + feature-drift monitoring.

Compares the current-month aggregate transition matrix with the prior month,
computes per-row PSI and the n-step backtest error, writes the drift table and
raises an alert row when a threshold is breached or a retrain trigger fires.
"""

from __future__ import annotations

import argparse

from ..common.action_logger import ActionLogger
from ..common.config import load_config


def main() -> None:  # pragma: no cover - Databricks runtime entry point
    parser = argparse.ArgumentParser(description="Markov drift monitoring")
    parser.add_argument("--score-date", required=True)
    parser.add_argument("--baseline-date", required=True, help="Prior month-end")
    parser.add_argument("--horizon", default="6m")
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    from ..common.state_contract import StateContract
    from ..monitoring.transition_drift import (
        drift_report_to_frame,
        transition_matrix_psi,
    )
    from .value_job import _aggregate_matrix_and_mix
    import numpy as np

    spark = SparkSession.builder.getOrCreate()
    logger = ActionLogger("jobs.monitoring")
    config = load_config(args.config_path)
    contract = StateContract.from_config(config)
    scoring = config["scoring"]
    mon = config["monitoring"]

    def _matrix(date: str):
        df = (
            spark.table(scoring["output"]["transitions_table"])
            .where(f"score_date = '{date}' AND horizon = '{args.horizon}'")
            .toPandas()
        )
        P, _ = _aggregate_matrix_and_mix(df, contract, np)
        return P

    baseline_P = _matrix(args.baseline_date)
    current_P = _matrix(args.score_date)

    report = transition_matrix_psi(
        baseline_P, current_P, contract, config, logger=logger
    )
    frame = drift_report_to_frame(report, args.score_date)
    spark.createDataFrame(frame).write.format("delta").mode("overwrite").option(
        "replaceWhere", f"score_date = '{args.score_date}'"
    ).saveAsTable(mon["output"]["drift_table"])

    if report.breached_rows or report.retrain_triggered:
        alert = spark.createDataFrame(
            [
                {
                    "score_date": args.score_date,
                    "max_psi": report.max_psi,
                    "breached_rows": ",".join(report.breached_rows),
                    "retrain_triggered": report.retrain_triggered,
                }
            ]
        )
        alert.write.format("delta").mode("append").saveAsTable(
            mon["output"]["alert_table"]
        )
    logger.info("monitoring_job.complete", retrain=report.retrain_triggered)


if __name__ == "__main__":
    main()
