"""markov_kernel — the registered MLflow pyfunc artifact.

This bundles, into ONE deployable object:

* the 5 ROW models (one conditional model per transient origin state),
* the per-row probability calibrators,
* the immutable ``CLUSTER_RANK`` state contract, and
* the matrix-reconstruction logic.

``MarkovKernel.predict`` returns, per customer, a full **6x6 row-stochastic
transition matrix P** (rows sum to 1; ``churned`` is absorbing). Because the
kernel is *non-homogeneous*, the customer's own features parameterise the entire
kernel: every transient origin row is produced by evaluating that origin's model
on the same feature row.

The estimator/calibrator implementations live in ``training`` / ``calibration``;
this module only depends on the small structural protocols defined here, so the
interface is stable regardless of which estimator backend is configured.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..common.state_contract import StateContract


# --------------------------------------------------------------------------- #
# Structural interfaces the training/calibration layers must satisfy
# --------------------------------------------------------------------------- #
@runtime_checkable
class RowModel(Protocol):
    """A conditional model for one origin state predicting the destination.

    Implementations predict a probability distribution over ALL states (the
    columns are ordered by rank, including the absorbing ``churned`` column).
    """

    #: Origin rank this model is conditioned on (0..4).
    origin_rank: int

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return an (n_rows x n_states) array of destination probabilities."""
        ...


@runtime_checkable
class RowCalibrator(Protocol):
    """Per-row probability calibrator (isotonic or Platt)."""

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Map raw (n x n_states) probabilities to calibrated probabilities.

        Implementations should return values that are renormalised to sum to 1
        per row by the kernel; calibrating then renormalising is the contract.
        """
        ...


class _IdentityCalibrator:
    """No-op calibrator used when a row has no fitted calibrator."""

    def transform(self, probabilities: np.ndarray) -> np.ndarray:  # noqa: D102
        return probabilities


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
class MarkovKernel:
    """Assembles per-customer 6x6 transition kernels from row models.

    Args:
        row_models: Mapping origin_rank -> :class:`RowModel`.
        calibrators: Mapping origin_rank -> :class:`RowCalibrator`.
        contract: The immutable :class:`StateContract`.
        feature_columns: Ordered feature columns the row models expect.
        row_sum_atol: Tolerance used when renormalising rows.
    """

    def __init__(
        self,
        row_models: Mapping[int, RowModel],
        calibrators: Mapping[int, RowCalibrator] | None,
        contract: StateContract,
        feature_columns: list[str],
        row_sum_atol: float = 1e-6,
    ) -> None:
        self.row_models = dict(row_models)
        self.calibrators = dict(calibrators or {})
        self.contract = contract
        self.feature_columns = list(feature_columns)
        self.row_sum_atol = float(row_sum_atol)

    # ------------------------------------------------------------------ #
    # Core: build dense per-customer kernels
    # ------------------------------------------------------------------ #
    def transition_matrices(self, features: pd.DataFrame) -> np.ndarray:
        """Build a dense (n_customers x n_states x n_states) kernel stack.

        For each transient origin rank, evaluate its row model on every customer
        row, calibrate, and place the resulting distribution into that origin's
        row. The absorbing row is set to the unit vector on ``churned``.

        Args:
            features: Feature frame containing ``self.feature_columns``.

        Returns:
            Array of shape ``(n, n_states, n_states)`` of row-stochastic kernels.
        """
        n = len(features)
        n_states = self.contract.n_states
        x = features[self.feature_columns]
        stack = np.zeros((n, n_states, n_states), dtype=float)

        for origin_rank in self.contract.transient_ranks:
            model = self.row_models[origin_rank]
            raw = np.asarray(model.predict_proba(x), dtype=float)
            calibrator = self.calibrators.get(origin_rank, _IdentityCalibrator())
            calibrated = np.asarray(calibrator.transform(raw), dtype=float)
            stack[:, origin_rank, :] = self._normalise_rows(calibrated)

        # Absorbing row: churned -> churned with probability 1.
        absorbing = self.contract.absorbing_rank
        stack[:, absorbing, :] = 0.0
        stack[:, absorbing, absorbing] = 1.0
        return stack

    def _normalise_rows(self, probs: np.ndarray) -> np.ndarray:
        """Clip negatives and renormalise each row to sum to 1."""
        clipped = np.clip(probs, 0.0, None)
        row_sums = clipped.sum(axis=1, keepdims=True)
        # Guard against all-zero rows (degenerate calibrator output).
        row_sums[row_sums <= self.row_sum_atol] = 1.0
        return clipped / row_sums

    # ------------------------------------------------------------------ #
    # Long / wide outputs
    # ------------------------------------------------------------------ #
    def predict(
        self, model_input: pd.DataFrame, params: Mapping[str, Any] | None = None
    ) -> pd.DataFrame:
        """MLflow pyfunc entry point.

        Args:
            model_input: Feature frame. If it carries the configured customer-id
                and score-date columns they are propagated to the output.
            params: Optional dict. ``{"output": "wide"}`` returns one row per
                customer with 36 ``p_{origin}_{dest}`` columns; the default
                ``"long"`` returns ``(customer_id, score_date, origin, dest,
                prob)`` rows.

        Returns:
            A long- or wide-format :class:`pandas.DataFrame` of transition
            probabilities.
        """
        params = dict(params or {})
        output = params.get("output", "long")
        stack = self.transition_matrices(model_input)
        names = self.contract.rank_to_name

        id_cols = self._passthrough_ids(model_input)
        if output == "wide":
            return self._to_wide(stack, names, id_cols, model_input.index)
        return self._to_long(stack, names, id_cols, model_input.index)

    def _passthrough_ids(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        cols: dict[str, pd.Series] = {}
        for candidate in ("customer_id", "score_date", "snapshot_date"):
            if candidate in df.columns:
                cols[candidate] = df[candidate].reset_index(drop=True)
        return cols

    def _to_long(
        self,
        stack: np.ndarray,
        names: Mapping[int, str],
        id_cols: dict[str, pd.Series],
        index: pd.Index,
    ) -> pd.DataFrame:
        n, n_states, _ = stack.shape
        rows: list[dict[str, Any]] = []
        for i in range(n):
            base = {k: v.iloc[i] for k, v in id_cols.items()}
            for o in range(n_states):
                for d in range(n_states):
                    rows.append(
                        {
                            **base,
                            "origin": names[o],
                            "dest": names[d],
                            "prob": float(stack[i, o, d]),
                        }
                    )
        return pd.DataFrame(rows)

    def _to_wide(
        self,
        stack: np.ndarray,
        names: Mapping[int, str],
        id_cols: dict[str, pd.Series],
        index: pd.Index,
    ) -> pd.DataFrame:
        n, n_states, _ = stack.shape
        data: dict[str, Any] = {k: v.to_numpy() for k, v in id_cols.items()}
        for o in range(n_states):
            for d in range(n_states):
                data[f"p_{names[o]}__{names[d]}"] = stack[:, o, d]
        return pd.DataFrame(data, index=index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# MLflow pyfunc wrapper
# --------------------------------------------------------------------------- #
def _import_pyfunc_base() -> type:
    """Import ``mlflow.pyfunc.PythonModel`` lazily.

    Keeps MLflow an optional dependency for unit tests of the pure kernel logic.
    """
    from mlflow.pyfunc import PythonModel  # local import

    return PythonModel


class MarkovKernelPyfunc:  # pragma: no cover - thin MLflow adapter
    """Adapter exposing :class:`MarkovKernel` as an ``mlflow.pyfunc.PythonModel``.

    The class is materialised against the MLflow base class at construction time
    so importing this module never hard-requires MLflow. The fitted
    :class:`MarkovKernel` is loaded from artifacts in ``load_context``.
    """

    @staticmethod
    def build() -> Any:
        base = _import_pyfunc_base()

        class _Impl(base):  # type: ignore[misc, valid-type]
            def load_context(self, context: Any) -> None:
                import cloudpickle

                with open(context.artifacts["kernel"], "rb") as handle:
                    self._kernel: MarkovKernel = cloudpickle.load(handle)

            def predict(
                self, context: Any, model_input: pd.DataFrame, params: dict | None = None
            ) -> pd.DataFrame:
                return self._kernel.predict(model_input, params)

        return _Impl()
