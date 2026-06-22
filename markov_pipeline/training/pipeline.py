"""pipeline — end-to-end training orchestration for the Markov kernel.

Ties the layers together under one MLflow parent run:

    feature snapshot -> row-stratified training -> per-row calibration
        -> kernel assembly -> log + register pyfunc

The temporal train/calibration split is taken from config (no shuffling — this
is time-series data). All artifacts are tracked; the resulting registered
pyfunc is what the monthly scoring job loads.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from ..calibration.calibrate import calibrate_rows
from ..common.action_logger import ActionLogger
from ..common.config import load_config
from ..common.state_contract import StateContract
from ..features.feature_builder import feature_matrix
from ..kernel.markov_kernel import MarkovKernel
from .train_row_models import train_row_models


def temporal_split(
    df: pd.DataFrame,
    config: Mapping[str, object],
    *,
    col_date: str = "snapshot_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labelled data temporally into train and calibration/holdout frames.

    The last ``training.test_split.holdout_snapshots`` distinct snapshot dates
    are reserved for calibration/validation; everything earlier trains the row
    models. No row is shuffled across the time boundary.

    Args:
        df: Labelled frame with a snapshot date column.
        config: Loaded config (reads ``training.test_split``).
        col_date: Snapshot date column.

    Returns:
        Tuple ``(train_df, calib_df)``.
    """
    holdout = int(config["training"]["test_split"]["holdout_snapshots"])  # type: ignore[index]
    snapshots = sorted(df[col_date].unique())
    if len(snapshots) <= holdout:
        # Not enough distinct snapshots — fall back to a tail fraction.
        cutoff_idx = max(1, int(len(df) * 0.8))
        return df.iloc[:cutoff_idx], df.iloc[cutoff_idx:]
    cutoff = snapshots[-holdout]
    train_df = df[df[col_date] < cutoff]
    calib_df = df[df[col_date] >= cutoff]
    return train_df, calib_df


def train_kernel(
    labelled: pd.DataFrame,
    config: Mapping[str, object] | None = None,
    *,
    register: bool = False,
    logger: ActionLogger | None = None,
) -> MarkovKernel:
    """Fit row models, calibrate, and assemble (optionally register) the kernel.

    Args:
        labelled: Labelled training frame (features + origin + destination +
            snapshot date).
        config: Loaded config; loaded from default path if omitted.
        register: When True, logs + registers the pyfunc kernel to MLflow.
        logger: Optional :class:`ActionLogger`.

    Returns:
        The assembled :class:`MarkovKernel`.
    """
    config = config or load_config()
    logger = logger or ActionLogger("training.pipeline")
    contract = StateContract.from_config(config)
    _, feature_cols = feature_matrix(labelled, config)

    with logger.timed("train_kernel"):
        train_df, calib_df = temporal_split(labelled, config)
        logger.info("split", n_train=len(train_df), n_calib=len(calib_df))

        models = train_row_models(
            train_df, feature_cols, contract, config, logger=logger
        )
        calibrators = calibrate_rows(
            models, calib_df, list(feature_cols), contract, config, logger=logger
        )
        kernel = MarkovKernel(models, calibrators, contract, list(feature_cols))

    if register:  # pragma: no cover - MLflow runtime
        import mlflow

        from ..kernel.register import log_and_register_kernel

        experiment = str(config["mlflow"]["experiment_path"])  # type: ignore[index]
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="train_markov_kernel"):
            mlflow.log_params(
                {
                    "estimator": config["row_models"]["estimator"],  # type: ignore[index]
                    "calibration_method": config["calibration"]["method"],  # type: ignore[index]
                    "gamma": config["mrp"]["gamma"],  # type: ignore[index]
                }
            )
            log_and_register_kernel(
                kernel, config, train_df[feature_cols].head(5), logger=logger
            )

    logger.flush_to_mlflow()
    return kernel
