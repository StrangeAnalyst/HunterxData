"""train_row_models — one conditional destination model PER ORIGIN state.

The non-homogeneous kernel is fit as five **row-stratified** conditional models:
for each transient origin rank we train a multiclass classifier predicting the
destination state (over all 6 states, including the absorbing ``churned``).

Two interchangeable estimators sit behind a single interface, selected by config:

* ``multinomial_logistic`` — governable / regulator-facing (interpretable).
* ``gradient_boosting``    — LightGBM/XGBoost multinomial (performance).

**Partial pooling** shrinks each per-customer prediction toward the origin's
global empirical destination distribution, with weight ``w = n / (n + kappa)``.
This stabilises sparse origins — especially the ``high_value`` origin row — where
a flexible model would otherwise overfit a handful of transitions.

Every fitted model, its params and metrics are logged to MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.state_contract import StateContract


# --------------------------------------------------------------------------- #
# Global empirical matrix (the pooling prior)
# --------------------------------------------------------------------------- #
def empirical_transition_matrix(
    df: pd.DataFrame,
    contract: StateContract,
    config: Mapping[str, object],
    *,
    col_origin: str = "origin_cluster",
    col_dest: str = "destination_rank",
) -> np.ndarray:
    """Compute the smoothed global empirical transition matrix.

    Dirichlet/Laplace smoothing (``row_models.smoothing.alpha``) is applied so no
    cell is exactly zero, which keeps the pooling prior and downstream log-space
    composition well behaved.

    Args:
        df: Labelled frame with origin and destination rank columns.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads ``row_models.smoothing.alpha``).
        col_origin: Origin rank column.
        col_dest: Destination rank column.

    Returns:
        An (n_states x n_states) row-stochastic matrix; the absorbing row is the
        unit vector on ``churned``.
    """
    alpha = float(config["row_models"]["smoothing"]["alpha"])  # type: ignore[index]
    n = contract.n_states
    counts = np.zeros((n, n), dtype=float)
    grouped = df.groupby([col_origin, col_dest]).size()
    for (origin, dest), c in grouped.items():
        counts[int(origin), int(dest)] += float(c)

    matrix = np.zeros((n, n), dtype=float)
    for origin in contract.transient_ranks:
        row = counts[origin] + alpha
        matrix[origin] = row / row.sum()
    absorbing = contract.absorbing_rank
    matrix[absorbing] = 0.0
    matrix[absorbing, absorbing] = 1.0
    return matrix


# --------------------------------------------------------------------------- #
# Estimator interface + implementations
# --------------------------------------------------------------------------- #
class _BaseRowEstimator:
    """Common base aligning any classifier's classes_ to the full state space."""

    def __init__(self, origin_rank: int, n_states: int) -> None:
        self.origin_rank = origin_rank
        self.n_states = n_states
        self._clf: Any = None
        self._class_to_col: dict[int, int] = {}

    def _align(self, proba: np.ndarray) -> np.ndarray:
        """Expand a classifier's class-subset proba to full n_states columns."""
        full = np.zeros((proba.shape[0], self.n_states), dtype=float)
        for cls, col in self._class_to_col.items():
            full[:, cls] = proba[:, col]
        return full

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_BaseRowEstimator":
        self._clf.fit(x, y)
        self._class_to_col = {int(c): i for i, c in enumerate(self._clf.classes_)}
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self._align(self._clf.predict_proba(features))


class MultinomialLogisticRowModel(_BaseRowEstimator):
    """Multinomial logistic regression row model (governable / interpretable)."""

    def __init__(self, origin_rank: int, n_states: int, params: Mapping[str, Any]) -> None:
        super().__init__(origin_rank, n_states)
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 500)),
            class_weight=params.get("class_weight", "balanced"),
            solver=str(params.get("solver", "lbfgs")),
        )


class GradientBoostingRowModel(_BaseRowEstimator):
    """Gradient-boosted multinomial row model (LightGBM/XGBoost backend)."""

    def __init__(self, origin_rank: int, n_states: int, params: Mapping[str, Any]) -> None:
        super().__init__(origin_rank, n_states)
        backend = str(params.get("backend", "lightgbm")).lower()
        common = dict(
            n_estimators=int(params.get("n_estimators", 400)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=float(params.get("subsample", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
        )
        if backend == "xgboost":
            from xgboost import XGBClassifier

            self._clf = XGBClassifier(
                objective="multi:softprob",
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                eval_metric="mlogloss",
                **common,
            )
        else:
            from lightgbm import LGBMClassifier

            self._clf = LGBMClassifier(
                objective="multiclass",
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                verbose=-1,
                **common,
            )


_ESTIMATORS = {
    "multinomial_logistic": MultinomialLogisticRowModel,
    "gradient_boosting": GradientBoostingRowModel,
}


# --------------------------------------------------------------------------- #
# Partial-pooling wrapper (satisfies the kernel's RowModel protocol)
# --------------------------------------------------------------------------- #
@dataclass
class PooledRowModel:
    """Wraps a base estimator and shrinks predictions toward the empirical prior.

    Implements the :class:`~markov_pipeline.kernel.markov_kernel.RowModel`
    protocol: ``origin_rank`` attribute + ``predict_proba``.

    Attributes:
        origin_rank: Origin state this model is conditioned on.
        base: The fitted base estimator.
        empirical_row: The origin's global empirical destination distribution.
        shrink_weight: ``w`` in ``w * model + (1 - w) * empirical``.
        feature_columns: Ordered feature columns.
    """

    origin_rank: int
    base: _BaseRowEstimator
    empirical_row: np.ndarray
    shrink_weight: float
    feature_columns: Sequence[str]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return pooled (n x n_states) destination probabilities."""
        model_p = self.base.predict_proba(features[list(self.feature_columns)])
        pooled = self.shrink_weight * model_p + (1.0 - self.shrink_weight) * self.empirical_row
        row_sums = pooled.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return pooled / row_sums


def _shrink_weight(n_origin: int, kappa: float, enabled: bool) -> float:
    if not enabled:
        return 1.0
    return float(n_origin) / (float(n_origin) + float(kappa))


# --------------------------------------------------------------------------- #
# Fit orchestration
# --------------------------------------------------------------------------- #
def train_row_models(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    contract: StateContract,
    config: Mapping[str, object],
    *,
    col_origin: str = "origin_cluster",
    col_dest: str = "destination_rank",
    logger: ActionLogger | None = None,
    log_mlflow: bool = True,
) -> dict[int, PooledRowModel]:
    """Fit one pooled conditional model per transient origin state.

    Args:
        train_df: Labelled training frame (features + origin + destination ranks).
        feature_columns: Ordered model feature columns.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads the ``row_models`` block).
        col_origin: Origin rank column.
        col_dest: Destination rank column.
        logger: Optional :class:`ActionLogger`.
        log_mlflow: Whether to log models/params/metrics to MLflow.

    Returns:
        Mapping origin_rank -> :class:`PooledRowModel`.
    """
    logger = logger or ActionLogger("training")
    rm_cfg = config["row_models"]  # type: ignore[index]
    estimator_key = str(rm_cfg["estimator"])
    estimator_cls = _ESTIMATORS[estimator_key]
    estimator_params = rm_cfg["estimators"][estimator_key]
    pooling = rm_cfg["partial_pooling"]
    kappa = float(pooling["kappa"])
    pooling_enabled = bool(pooling["enabled"])
    sparse_warn = int(rm_cfg["sparse_origin_warn_count"])

    empirical = empirical_transition_matrix(
        train_df, contract, config, col_origin=col_origin, col_dest=col_dest
    )

    models: dict[int, PooledRowModel] = {}
    for origin_rank in contract.transient_ranks:
        origin_df = train_df[train_df[col_origin] == origin_rank]
        n_origin = len(origin_df)
        if n_origin == 0:
            logger.warn("train_row.empty_origin", origin_rank=origin_rank)
            continue
        if n_origin < sparse_warn:
            logger.warn(
                "train_row.sparse_origin",
                origin_rank=origin_rank,
                n=n_origin,
                threshold=sparse_warn,
            )

        x = origin_df[list(feature_columns)]
        y = origin_df[col_dest].astype(int).to_numpy()

        base = estimator_cls(origin_rank, contract.n_states, estimator_params)
        with logger.timed("train_row.fit", origin_rank=origin_rank, n=n_origin):
            base.fit(x, y)

        weight = _shrink_weight(n_origin, kappa, pooling_enabled)
        models[origin_rank] = PooledRowModel(
            origin_rank=origin_rank,
            base=base,
            empirical_row=empirical[origin_rank],
            shrink_weight=weight,
            feature_columns=list(feature_columns),
        )
        metrics = _row_metrics(models[origin_rank], x, y, contract)
        logger.info(
            "train_row.fit.metrics",
            origin_rank=origin_rank,
            shrink_weight=round(weight, 4),
            **{k: round(v, 5) for k, v in metrics.items()},
        )
        if log_mlflow:
            _log_row_to_mlflow(
                origin_rank, estimator_key, estimator_params, weight, metrics
            )

    logger.flush_to_mlflow()
    return models


def _row_metrics(
    model: PooledRowModel,
    x: pd.DataFrame,
    y: np.ndarray,
    contract: StateContract,
) -> dict[str, float]:
    """Compute log-loss and accuracy for a fitted pooled row model."""
    from sklearn.metrics import accuracy_score, log_loss

    proba = model.predict_proba(x)
    labels = contract.all_ranks
    preds = np.argmax(proba, axis=1)
    return {
        "log_loss": float(log_loss(y, proba, labels=labels)),
        "accuracy": float(accuracy_score(y, preds)),
    }


def _log_row_to_mlflow(
    origin_rank: int,
    estimator_key: str,
    estimator_params: Mapping[str, Any],
    shrink_weight: float,
    metrics: Mapping[str, float],
) -> None:  # pragma: no cover - MLflow side effect
    """Log one row model's params and metrics under a nested MLflow run."""
    try:
        import mlflow

        with mlflow.start_run(run_name=f"row_model_origin_{origin_rank}", nested=True):
            mlflow.log_params(
                {
                    "origin_rank": origin_rank,
                    "estimator": estimator_key,
                    "shrink_weight": shrink_weight,
                    **{f"hp_{k}": v for k, v in estimator_params.items()},
                }
            )
            mlflow.log_metrics(dict(metrics))
    except Exception:
        pass
