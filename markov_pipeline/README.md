# Non-Homogeneous Markov Cluster-Transition Pipeline

A production-grade, config-driven pipeline that models customer value-cluster
dynamics as a **non-homogeneous Markov chain**. Per customer we produce a
feature-dependent 6×6 transition kernel, then compose it as a **Markov Reward
Process** to derive expected lifetime, churn absorption and CLV — and a
closed-loop **value layer** that turns those into an actionable queue and an
auditable growth expectation.

Everything numeric (thresholds, business numbers, table names, switches) lives
in [`config/markov_config.yaml`](config/markov_config.yaml) — the single source
of truth. No business number is hardcoded in Python.

## State contract (ordinal — never reordered)

`low_value(0) < emerging_user(1) < growing_user(2) < high_power(3) <
high_value(4)` + absorbing `churned(5)`. Transient ranks `0..4`; absorbing `5`.

## Architecture (layered)

```
config/markov_config.yaml      single source of truth
common/                        config loader, StateContract, ActionLogger, L1/L2/L3 guardrails
targets/                       destination-state target (conditioned on origin), persistence rules
features/                      PIT-correct month-end snapshots + duration/path features
training/                      row-stratified models (logit | GBM), partial pooling, MLflow
calibration/                   per-row isotonic/Platt calibration + ECE/reliability
kernel/                        MarkovKernel pyfunc (6x6 per customer) + registration
composition/                   n-step, stationary, fundamental matrix, absorption, CLV-MRP
scoring/                       monthly Spark batch -> long Delta + customer_trajectory
monitoring/                    per-row PSI drift, n-step backtest, retrain trigger
business_value/                action queue, actioning volume, value expectation, closed loop
jobs/                          Databricks job entry points (score/value/monitoring/train)
```

## Run order

### One-time / per-retrain (training)
1. **Feature + target build** (`features/feature_builder.py`,
   `targets/build_transition_target.py`) → labelled training Delta table.
2. **Train + calibrate + register** (`training/pipeline.train_kernel`,
   `kernel/register.py`) → registered `markov_cluster_kernel` pyfunc in MLflow.
   - Estimator switch: `row_models.estimator` ∈ `{multinomial_logistic,
     gradient_boosting}`.
   - Partial pooling shrinks sparse origins toward the global empirical matrix.
   - Per-row calibration is mandatory; ECE gates BAU acceptance.

### Monthly (BAU)
3. **Scoring** (`jobs/score_job.py` → `scoring/batch_score.py`): loads the
   `Production` kernel, writes `transition_scores` (long) + `customer_trajectory`.
4. **Value layer** (`jobs/value_job.py` → `business_value/actionability.py`):
   `action_queue`, `actioning_volume`, `value_expectation` (+ `value_realized`
   once actuals land).

### Weekly (BAU)
5. **Monitoring** (`jobs/monitoring_job.py` → `monitoring/transition_drift.py`):
   per-row PSI vs prior month, n-step backtest error, retrain trigger.

## Local development

```bash
pip install -e ".[gbm,mlflow,dev]"
pytest                       # 28 invariant tests
```

Quick kernel demo:

```python
import pandas as pd
from markov_pipeline.common.config import load_config
from markov_pipeline.training.pipeline import train_kernel

cfg = load_config()
labelled = ...               # features + origin_cluster + destination_rank + snapshot_date
kernel = train_kernel(labelled, cfg, register=False)
P_stack = kernel.transition_matrices(labelled.head())   # (n, 6, 6) row-stochastic
```

## pyfunc interface

`MarkovKernel.predict(model_input, params)` returns, per customer, a full 6×6
row-stochastic transition matrix:
- `params={"output": "long"}` (default) → `(customer_id, score_date, origin,
  dest, prob)`
- `params={"output": "wide"}` → one row per customer with 36 `p_{origin}__{dest}`
  columns.

## Databricks Asset Bundle

See [`../databricks/`](../databricks). Deploy with
`databricks bundle deploy -t prod`; jobs: `markov_scoring` (monthly),
`markov_monitoring` (weekly), `markov_retraining` (trigger + quarterly floor).

## Operating model / BAU

See [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md): project→BAU acceptance gate,
cadence, RACI, SLA, observability, and the **causal `P_action` seam** (NBA
effects come from Double ML / uplift — never from observational counts).
