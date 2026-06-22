"""Databricks job entry point: monthly batch scoring + composition.

Invoked by the DAB ``markov_scoring`` job. Resolves the score date (month-end of
the prior close), loads the registered kernel and writes the long-format
transition table and the customer_trajectory table. The trajectory drives the
downstream value-layer task in the same job.
"""

from __future__ import annotations

import argparse

from ..common.action_logger import ActionLogger
from ..scoring.batch_score import run_batch_scoring


def main() -> None:  # pragma: no cover - Databricks runtime entry point
    parser = argparse.ArgumentParser(description="Markov monthly batch scoring")
    parser.add_argument("--score-date", required=True, help="Month-end YYYY-MM-DD")
    parser.add_argument("--horizon", default="6m", help="Horizon name (e.g. 6m)")
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    logger = ActionLogger("jobs.score")
    run_batch_scoring(
        spark, args.score_date, args.horizon, args.config_path, logger=logger
    )


if __name__ == "__main__":
    main()
