"""Databricks job entry point: train + calibrate + register the kernel.

Used by the trigger-based retraining job (champion-challenger). Builds the
labelled training frame from the feature store over the configured rolling
window, fits the row models, calibrates, assembles the kernel and registers it
as a new model version in the requested stage (default ``Staging`` — promotion
to ``Production`` is deliberate per the runbook).
"""

from __future__ import annotations

import argparse

from ..common.action_logger import ActionLogger
from ..common.config import load_config


def main() -> None:  # pragma: no cover - Databricks runtime entry point
    parser = argparse.ArgumentParser(description="Markov kernel training")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--stage", default="Staging")
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    from ..training.pipeline import train_kernel

    spark = SparkSession.builder.getOrCreate()
    logger = ActionLogger("jobs.train")
    config = load_config(args.config_path)

    # The labelled training frame is materialised upstream by the feature/target
    # jobs into a training Delta table; read it here over the rolling window.
    feats = config["features"]
    train_table = f"{feats['feature_store']['catalog']}.{feats['feature_store']['schema']}.markov_training_labelled"
    labelled = spark.table(train_table).toPandas()

    train_kernel(labelled, config, register=args.register, logger=logger)
    logger.info("train_job.complete", stage=args.stage, registered=args.register)


if __name__ == "__main__":
    main()
