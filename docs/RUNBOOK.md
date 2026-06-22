# RUNBOOK — Non-Homogeneous Markov Cluster-Transition Pipeline

This runbook defines **when the project becomes BAU**, the **steady-state
cadence**, and **ownership / SLA / observability**. It is the operational
contract; everything numeric here is mirrored in `config/markov_config.yaml`
(`bau:` block) so the schedules and thresholds have a single source of truth.

---

## 0. What this system is

A non-homogeneous Markov chain over customer value clusters. Per customer we
produce a feature-dependent 6×6 transition kernel (5 transient value clusters +
the absorbing `churned` state). The kernel is composed into a Markov Reward
Process to derive **expected lifetime**, **churn absorption probability** and
**CLV**, and a value layer converts those into an **action queue** and an
**auditable growth expectation** (expected vs realized).

State contract (ordinal, never reordered):

| rank | state          | type      |
|------|----------------|-----------|
| 0    | low_value      | transient |
| 1    | emerging_user  | transient |
| 2    | growing_user   | transient |
| 3    | high_power     | transient |
| 4    | high_value     | transient |
| 5    | churned        | absorbing |

---

## A. PROJECT → BAU ACCEPTANCE CRITERIA

The model may **not** be handed to BAU until **all** of the following are
recorded (gate). Config keys live under `bau.acceptance_criteria`.

1. **Shadow run ≥ 3 monthly cycles** in parallel with stable output
   (`shadow_run_cycles: 3`).
2. **Per-row ECE ≤ threshold** on every origin row (`max_per_row_ece: 0.05`) —
   calibration holds (see `calibration/calibrate.py`).
3. **n-step backtest error within tolerance** (`n_step_backtest_tolerance:
   0.05`) — the Markov property is validated: `P^t` reproduces the empirical
   t-step transitions (see `composition/markov_dynamics.markov_property_error`).
4. **Transition-matrix PSI stable month-over-month** (`psi_stable_threshold:
   0.25`) across the shadow window.
5. **Sign-off recorded** by promoting the registered model to
   `Production` in the MLflow Model Registry
   (`require_registry_production_stage: true`).

Until the gate passes, scoring writes to a shadow schema and is **not** consumed
by Campaigns.

---

## B. BAU CADENCE

Encoded in the Databricks Asset Bundle (`databricks/`); config under
`bau.cadence`.

| Job | Trigger | Notes |
|-----|---------|-------|
| **Scoring** | Monthly, **T+N business days** after month-end close (`score_offset_business_days: 5`, cron `0 6 5 * *`) | Loads the `Production` kernel pyfunc; writes long-format transitions + `customer_trajectory`. |
| **Composition** | Immediately downstream of scoring (same DAB job, dependent task) | n-step, absorption, CLV-MRP, `action_queue`, `value_expectation`. |
| **Monitoring** | **Weekly** (cron `0 6 * * 1`) | Transition-matrix PSI (per row) + feature drift; alerts on breach. |
| **Calibration refresh** | **Monthly** (`calibration_refresh: monthly`) | Re-fit per-row calibrators on the latest labelled window. |
| **Retraining** | **Trigger-based** (PSI / backtest breach) with a **quarterly floor** (`retraining_quarterly_floor: true`) | Champion–challenger; **never** a silent overwrite. New kernel registered as a challenger, promoted only after passing the acceptance gate. |

### Champion–challenger
A retrain trigger fits a challenger kernel and registers a new model version in
`Staging`. It is shadow-scored and compared on ECE, backtest error and PSI
stability. Promotion to `Production` is a deliberate, recorded action — the
champion is never overwritten automatically.

---

## C. OWNERSHIP / SLA / OBSERVABILITY

### RACI

| Activity | Model Owner (Customer Analytics) | Data Owner (Data Platform) | Campaign Consumer (Marketing/Campaigns) |
|----------|:--------------------------------:|:--------------------------:|:---------------------------------------:|
| Feature store correctness / PIT | C | **A/R** | I |
| Model training / calibration | **A/R** | C | I |
| Drift monitoring & retrain decision | **A/R** | C | I |
| Action-queue consumption / campaigns | I | I | **A/R** |
| Lift / causal-effect calibration (P_action) | **A/R** (with causal team) | C | C |
| BAU acceptance sign-off | **A/R** | C | C |

(A = Accountable, R = Responsible, C = Consulted, I = Informed.)

### SLA
- Scored tables (`transition_scores`, `customer_trajectory`, `action_queue`,
  `value_expectation`) available by **T+N business days**
  (`sla.scored_tables_available_by_business_days: 5`).
- **Alert routing**: job failure → on-call model owner; drift breach → model
  owner + data owner (`sla.alert_on_job_failure`, `sla.alert_on_drift_breach`).

### Observability dashboards (`bau.observability.dashboards`)
1. **action_queue volume** — count entering the queue by segment/channel
   (capacity planning, Module 11.2).
2. **value_expectation vs value_realized** — modeled vs realized growth
   (Module 11.3 / 11.4); the closed-loop that keeps the system defensible to
   Finance.
3. **transition-matrix heatmap month-over-month** — visual drift.
4. **drift status** — per-row PSI, backtest error, retrain-trigger state.

---

## D. The causal seam (do not skip)

Action (NBA) effects **must** come from the causal pipeline (Double ML / uplift),
**never** from observational transition counts (confounded). `P_action` is built
in `business_value/actionability.apply_policy_to_matrix` from a `CausalLift`
resolved by `CausalLiftProvider`:

- `business_value.value_expectation.causal_lift.source: assumed` → lifts come
  from `assumed_lifts` in config, `assumption_flag = true` (surfaced in
  `value_expectation`). **Audit-only; not for committed growth targets.**
- `source: causal_pipeline` → lifts read from
  `ml_prod.causal.nba_uplift_effects`, `assumption_flag = false`.

Switching the source is a config change, not a code change. The realized-vs-
expected loop (Module 11.4) feeds back the ratio to recalibrate the assumed
lift until the causal layer is live.

---

## E. Operational playbook

| Symptom | First check | Action |
|---------|-------------|--------|
| Scoring job failed | Cluster/MLflow model load; feature snapshot freshness | Re-run; if model URI missing, confirm `Production` stage exists |
| PSI breach on a row | `transition_drift` table; which origin row moved | Open retrain decision (champion–challenger); inform Campaigns |
| ECE above threshold | Latest calibration metrics in MLflow | Trigger calibration refresh; if persistent, retrain |
| value_realized ≪ value_expectation | Lift assumption flag; take-rate | Recalibrate lift; escalate to causal team to wire `causal_pipeline` |
| Non-finite expected_lifetime | `(I-Q)` near-singular (a transient state became effectively absorbing) | Inspect the offending origin row; likely data/labelling issue |
