"""Tests for row-model training, calibration improvement, and kernel assembly."""

from __future__ import annotations

import numpy as np

from markov_pipeline.calibration.calibrate import (
    RowCalibrator,
    expected_calibration_error,
)
from markov_pipeline.common.guardrails import l1_check_absorbing, l1_check_row_stochastic
from markov_pipeline.kernel.markov_kernel import MarkovKernel
from markov_pipeline.training.train_row_models import (
    empirical_transition_matrix,
    train_row_models,
)


def _feature_cols(config) -> list[str]:
    return list(config["features"]["base_features"]) + list(
        config["features"]["path_features"]
    )


def test_empirical_matrix_is_row_stochastic(labelled_frame, contract, config) -> None:
    P = empirical_transition_matrix(labelled_frame, contract, config)
    l1_check_row_stochastic(P, config)
    l1_check_absorbing(P, contract)


def test_kernel_emits_valid_matrices(labelled_frame, contract, config) -> None:
    fcols = _feature_cols(config)
    models = train_row_models(labelled_frame, fcols, contract, config, log_mlflow=False)
    kernel = MarkovKernel(models, None, contract, fcols)
    stack = kernel.transition_matrices(labelled_frame.head(20))
    assert stack.shape == (20, 6, 6)
    for m in stack:
        l1_check_row_stochastic(m, config)
        l1_check_absorbing(m, contract)


def test_kernel_long_output_shape(labelled_frame, contract, config) -> None:
    fcols = _feature_cols(config)
    models = train_row_models(labelled_frame, fcols, contract, config, log_mlflow=False)
    kernel = MarkovKernel(models, None, contract, fcols)
    sample = labelled_frame.head(3).assign(customer_id=[1, 2, 3])
    long = kernel.predict(sample)
    # 3 customers x 6 origins x 6 dests.
    assert len(long) == 3 * 36
    assert set(long.columns) >= {"customer_id", "origin", "dest", "prob"}


def test_calibration_improves_or_holds_ece(labelled_frame, contract, config) -> None:
    """Isotonic calibration should not worsen ECE on held-out data."""
    fcols = _feature_cols(config)
    train = labelled_frame.iloc[:2000]
    calib = labelled_frame.iloc[2000:]
    models = train_row_models(train, fcols, contract, config, log_mlflow=False)

    n_bins = int(config["calibration"]["reliability"]["n_bins"])
    improved_or_equal = 0
    evaluated = 0
    for origin_rank, model in models.items():
        odf = calib[calib["origin_cluster"] == origin_rank]
        if len(odf) < 50:
            continue
        x = odf[fcols]
        y = odf["destination_rank"].astype(int).to_numpy()
        raw = model.predict_proba(x)
        cal = RowCalibrator(origin_rank, "isotonic", contract.n_states).fit(raw, y)
        calibrated = cal.transform(raw)
        ece_before = np.mean(
            [expected_calibration_error(raw, y, c, n_bins) for c in contract.all_ranks]
        )
        ece_after = np.mean(
            [
                expected_calibration_error(calibrated, y, c, n_bins)
                for c in contract.all_ranks
            ]
        )
        evaluated += 1
        if ece_after <= ece_before + 1e-3:
            improved_or_equal += 1
    assert evaluated > 0
    # Calibration should hold/improve for the large majority of rows.
    assert improved_or_equal >= evaluated - 1
