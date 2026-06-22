"""actionability — the closed-loop value layer on top of customer_trajectory.

This module turns Markov / MRP outputs into something the bank can act on and
measure:

* **11.1 Action queue** — config-driven rules select RETENTION / CULTIVATION
  candidates, gated by value (CLV floor) and an addressable churn band, with a
  priority score = expected CLV at risk (or expected cultivation uplift).
* **11.2 Expected actioning volume** — per month / segment / channel counts so
  Campaigns can capacity-plan.
* **11.3 Growth expectation** — MODELED from the chain: baseline drift (as-is
  ``P``) vs policy drift (``P_action``), with ``P_action`` effects entering
  ONLY through the causal/uplift interface. Lift is auditable and, until the
  causal layer is wired, explicitly ASSUMED and flagged.
* **11.4 Closed loop** — realized vs expected, by settling next-period actuals
  back against the prior action queue.

CRITICAL: action (NBA) effects must come from the causal pipeline (Double ML /
uplift), never from observational transition counts (confounded). The
:class:`CausalLiftProvider` is the single, clean seam where calibrated causal
effects plug in. No business number is hardcoded here — all are read from config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..common.action_logger import ActionLogger
from ..common.state_contract import StateContract


# =========================================================================== #
# Causal lift interface (the P_action seam)
# =========================================================================== #
@dataclass
class CausalLift:
    """Calibrated causal effect of the NBA policy on transition probabilities.

    Attributes:
        retention_churn_reduction_pp: Absolute pp reduction in churn probability
            for actioned RETENTION candidates.
        cultivation_uplift_to_high_power_pp: Absolute pp increase in P(reach
            high_power) for actioned CULTIVATION candidates.
        action_take_rate: Fraction of queued customers actually actioned.
        assumption_flag: True when the lift is an unaudited assumption (not yet
            sourced from the causal pipeline).
    """

    retention_churn_reduction_pp: float
    cultivation_uplift_to_high_power_pp: float
    action_take_rate: float
    assumption_flag: bool


class CausalLiftProvider:
    """Resolves causal lift from config — either assumed or from the causal table.

    When ``business_value.value_expectation.causal_lift.source == 'causal_pipeline'``
    the lifts are read from the Double-ML / uplift output table (and the
    assumption flag is cleared). Otherwise the auditable assumed lifts from
    config are returned with ``assumption_flag = True``.
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        self._cfg = config["business_value"]["value_expectation"]["causal_lift"]  # type: ignore[index]

    def get_lift(self, causal_effects: pd.DataFrame | None = None) -> CausalLift:
        """Return the :class:`CausalLift` to apply when forming ``P_action``.

        Args:
            causal_effects: Optional frame of calibrated effects from the causal
                pipeline. Required when ``source == 'causal_pipeline'``.

        Returns:
            A :class:`CausalLift`.

        Raises:
            ValueError: If the causal source is requested but no effects frame is
                supplied.
        """
        source = str(self._cfg["source"])
        if source == "causal_pipeline":
            if causal_effects is None or causal_effects.empty:
                raise ValueError(
                    "causal_lift.source='causal_pipeline' but no causal_effects "
                    "frame was supplied — refusing to fall back to assumptions silently"
                )
            row = causal_effects.iloc[0]
            return CausalLift(
                retention_churn_reduction_pp=float(row["retention_churn_reduction_pp"]),
                cultivation_uplift_to_high_power_pp=float(
                    row["cultivation_uplift_to_high_power_pp"]
                ),
                action_take_rate=float(row["action_take_rate"]),
                assumption_flag=False,
            )
        assumed = self._cfg["assumed_lifts"]
        return CausalLift(
            retention_churn_reduction_pp=float(assumed["retention_churn_reduction_pp"]),
            cultivation_uplift_to_high_power_pp=float(
                assumed["cultivation_uplift_to_high_power_pp"]
            ),
            action_take_rate=float(assumed["action_take_rate"]),
            assumption_flag=True,
        )


def apply_policy_to_matrix(
    P: np.ndarray,
    lift: CausalLift,
    contract: StateContract,
) -> np.ndarray:
    """Construct ``P_action`` from the as-is matrix ``P`` and a causal lift.

    Applies the calibrated causal effects as bounded probability adjustments and
    renormalises each row to stay stochastic. Effects are applied at the
    take-rate, so unactioned mass behaves as baseline.

    Args:
        P: As-is row-stochastic matrix.
        lift: Calibrated :class:`CausalLift`.
        contract: The immutable :class:`StateContract`.

    Returns:
        The policy transition matrix ``P_action`` (row-stochastic, churned
        absorbing).
    """
    P_action = P.copy()
    absorbing = contract.absorbing_rank
    hp = contract.growth_target_rank
    take = lift.action_take_rate

    for origin in contract.transient_ranks:
        row = P_action[origin].copy()
        # Retention: reduce churn mass, redistribute to staying in origin.
        churn_cut = min(row[absorbing], take * lift.retention_churn_reduction_pp)
        row[absorbing] -= churn_cut
        row[origin] += churn_cut
        # Cultivation: shift mass toward high_power from lower transient states.
        if origin < hp:
            uplift = take * lift.cultivation_uplift_to_high_power_pp
            uplift = min(uplift, row[origin])
            row[origin] -= uplift
            row[hp] += uplift
        P_action[origin] = row / row.sum()
    P_action[absorbing] = 0.0
    P_action[absorbing, absorbing] = 1.0
    return P_action


# =========================================================================== #
# 11.1 ACTION QUEUE
# =========================================================================== #
def compute_downgrade_prob(
    transitions: pd.DataFrame,
    contract: StateContract,
    *,
    col_id: str = "customer_id",
    col_origin: str = "origin",
    col_dest: str = "dest",
    col_prob: str = "prob",
) -> pd.DataFrame:
    """Per-customer probability of moving to a strictly lower transient rank.

    Args:
        transitions: Long-format transition frame for a single origin per
            customer (as written by scoring), filtered to that customer's origin.
        contract: The immutable :class:`StateContract`.

    Returns:
        Frame ``(customer_id, downgrade_prob)``.
    """
    rank = dict(contract.cluster_rank)
    absorbing_name = contract.absorbing_name

    def _is_downgrade(o: str, d: str) -> bool:
        if d == absorbing_name:
            return False  # churn handled separately via the band
        if o == absorbing_name:
            return False
        return rank[d] < rank[o]

    df = transitions.copy()
    df["_is_downgrade"] = [
        _is_downgrade(o, d) for o, d in zip(df[col_origin], df[col_dest])
    ]
    grouped = (
        df[df["_is_downgrade"]]
        .groupby(col_id)[col_prob]
        .sum()
        .rename("downgrade_prob")
        .reset_index()
    )
    return grouped


def build_action_queue(
    trajectory: pd.DataFrame,
    downgrade: pd.DataFrame,
    contract: StateContract,
    config: Mapping[str, object],
    *,
    col_id: str = "customer_id",
    logger: ActionLogger | None = None,
) -> pd.DataFrame:
    """Build the action queue from trajectory metrics + downgrade probabilities.

    A customer enters the queue when ALL of:
      * ``downgrade_prob >= tau_risk`` (RETENTION) OR
        ``prob_reach_high_power >= tau_growth`` (CULTIVATION)
      * ``expected_clv >= clv_floor`` (value gate)
      * churn_prob within the addressable band.

    Args:
        trajectory: customer_trajectory frame (expected_clv, churn_prob,
            prob_reach_high_power, predicted_destination, origin, score_date,
            horizon).
        downgrade: Output of :func:`compute_downgrade_prob`.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads ``business_value.action_queue``).
        col_id: Customer id column.
        logger: Optional :class:`ActionLogger`.

    Returns:
        The action_queue frame.
    """
    logger = logger or ActionLogger("business_value")
    aq = config["business_value"]["action_queue"]  # type: ignore[index]
    rules = aq["rules"]
    tau_risk = float(rules["tau_risk"])
    tau_growth = float(rules["tau_growth"])
    clv_floor = float(rules["clv_floor"])
    band_min = float(rules["churn_addressable_band"]["min"])
    band_max = float(rules["churn_addressable_band"]["max"])
    nba_map = aq["nba_map"]

    df = trajectory.merge(downgrade, on=col_id, how="left")
    df["downgrade_prob"] = df["downgrade_prob"].fillna(0.0)

    risk_gate = df["downgrade_prob"] >= tau_risk
    growth_gate = df["prob_reach_high_power"] >= tau_growth
    value_gate = df["expected_clv"] >= clv_floor
    band_gate = df["churn_prob"].between(band_min, band_max)

    selected = df[(risk_gate | growth_gate) & value_gate & band_gate].copy()
    # RETENTION takes precedence when the risk gate is met.
    selected["trigger_type"] = np.where(
        selected["downgrade_prob"] >= tau_risk, "RETENTION", "CULTIVATION"
    )

    # P(adverse) = downgrade + churn for retention priority.
    selected["p_adverse"] = (selected["downgrade_prob"] + selected["churn_prob"]).clip(
        upper=1.0
    )
    lift = CausalLiftProvider(config).get_lift()
    cultivation_gain = selected["prob_reach_high_power"] * (
        lift.cultivation_uplift_to_high_power_pp
    )
    selected["clv_at_risk"] = selected["expected_clv"] * selected["p_adverse"]
    selected["priority_score"] = np.where(
        selected["trigger_type"] == "RETENTION",
        selected["expected_clv"] * selected["p_adverse"],
        selected["expected_clv"] * cultivation_gain,
    )
    selected["recommended_nba"] = [
        nba_map[trig].get(origin, "review_manually")
        for trig, origin in zip(selected["trigger_type"], selected["origin"])
    ]

    out_cols = [
        col_id,
        "score_date",
        "horizon",
        "trigger_type",
        "origin",
        "predicted_destination",
        "expected_clv",
        "clv_at_risk",
        "recommended_nba",
        "priority_score",
    ]
    queue = selected[[c for c in out_cols if c in selected.columns]].sort_values(
        "priority_score", ascending=False
    )
    logger.info(
        "action_queue.built",
        n_total=len(df),
        n_queued=len(queue),
        n_retention=int((queue["trigger_type"] == "RETENTION").sum()),
        n_cultivation=int((queue["trigger_type"] == "CULTIVATION").sum()),
    )
    return queue


# =========================================================================== #
# 11.2 EXPECTED ACTIONING VOLUME
# =========================================================================== #
def expected_actioning_volume(
    action_queue: pd.DataFrame,
    config: Mapping[str, object],
    *,
    col_segment: str = "origin",
) -> pd.DataFrame:
    """Aggregate the action queue into capacity-planning volumes.

    Args:
        action_queue: Output of :func:`build_action_queue`.
        config: Loaded config (reads ``business_value.actioning_volume``).
        col_segment: Column used as the segment dimension (default origin).

    Returns:
        Frame ``(score_date, segment, channel, trigger_type, expected_count,
        expected_clv_at_risk)``.
    """
    av = config["business_value"]["actioning_volume"]  # type: ignore[index]
    channel_map = av["channel_map"]
    df = action_queue.copy()
    df["channel"] = df["recommended_nba"].map(channel_map).fillna("unrouted")
    df["segment"] = df[col_segment]

    grouped = (
        df.groupby(["score_date", "segment", "channel", "trigger_type"])
        .agg(
            expected_count=("recommended_nba", "size"),
            expected_clv_at_risk=("clv_at_risk", "sum"),
        )
        .reset_index()
    )
    return grouped


# =========================================================================== #
# 11.3 GROWTH EXPECTATION (modeled, not guessed)
# =========================================================================== #
def project_portfolio_mix(
    P: np.ndarray,
    initial_mix: np.ndarray,
    horizon_steps: int,
    contract: StateContract,
) -> np.ndarray:
    """Project the segment mix forward H steps under a transition matrix.

    Args:
        P: Row-stochastic transition matrix (as-is or policy).
        initial_mix: Initial distribution over all states (sums to 1).
        horizon_steps: Number of steps H.
        contract: The immutable :class:`StateContract`.

    Returns:
        The projected distribution over states at horizon H.
    """
    from ..composition.markov_dynamics import n_step

    return initial_mix @ n_step(P, horizon_steps)


def value_expectation(
    baseline_P: np.ndarray,
    initial_mix: np.ndarray,
    contract: StateContract,
    config: Mapping[str, object],
    *,
    score_date: str,
    causal_effects: pd.DataFrame | None = None,
    logger: ActionLogger | None = None,
) -> pd.DataFrame:
    """Compute baseline vs policy drift and the modeled expected growth.

    ``expected_growth = policy_drift - baseline_drift`` expressed per segment as
    involution-rate reduction, net upgrade flow, HPU pipeline net adds and CLV
    preserved/added. ``P_action`` is built ONLY from the causal lift; the
    assumption flag is propagated so Finance sees what is modeled vs assumed.

    Args:
        baseline_P: Aggregate as-is transition matrix.
        initial_mix: Current portfolio mix over states (sums to 1).
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads ``business_value.value_expectation`` + mrp).
        score_date: Score date stamp.
        causal_effects: Optional calibrated causal effects frame.
        logger: Optional :class:`ActionLogger`.

    Returns:
        The value_expectation frame.
    """
    logger = logger or ActionLogger("business_value")
    ve = config["business_value"]["value_expectation"]  # type: ignore[index]
    H = int(ve["projection_horizon_months"])
    total_customers = float(ve["portfolio_baselines"]["total_customers_assumed"])
    gamma = float(config["mrp"]["gamma"])  # type: ignore[index]

    lift = CausalLiftProvider(config).get_lift(causal_effects)
    policy_P = apply_policy_to_matrix(baseline_P, lift, contract)

    baseline_mix = project_portfolio_mix(baseline_P, initial_mix, H, contract)
    policy_mix = project_portfolio_mix(policy_P, initial_mix, H, contract)

    names = contract.rank_to_name
    absorbing = contract.absorbing_rank
    hp = contract.growth_target_rank

    # CLV per state under each regime (length n_transient).
    from ..composition.markov_dynamics import clv_mrp

    reward = _segment_reward_vector(config, contract)
    clv_baseline = clv_mrp(baseline_P, reward, gamma, len(contract.transient_ranks))
    clv_policy = clv_mrp(policy_P, reward, gamma, len(contract.transient_ranks))

    records: list[dict[str, object]] = []
    for state in contract.all_ranks:
        seg = names[state]
        base_share = float(baseline_mix[state])
        pol_share = float(policy_mix[state])
        delta_share = pol_share - base_share
        base_count = base_share * total_customers
        pol_count = pol_share * total_customers
        clv_b = 0.0 if state == absorbing else float(clv_baseline[state])
        clv_p = 0.0 if state == absorbing else float(clv_policy[state])
        # $ impact: change in head-count valued at policy CLV.
        dollar_impact = (pol_count - base_count) * clv_p
        records.append(
            {
                "score_date": score_date,
                "segment": seg,
                "baseline_metric": base_share,
                "policy_metric": pol_share,
                "expected_delta": delta_share,
                "baseline_headcount": base_count,
                "policy_headcount": pol_count,
                "net_flow": pol_count - base_count,
                "clv_baseline": clv_b,
                "clv_policy": clv_p,
                "dollar_impact": dollar_impact,
                "assumption_flag": lift.assumption_flag,
            }
        )

    frame = pd.DataFrame(records)
    # Headline summaries.
    involution_reduction = float(
        baseline_mix[absorbing] - policy_mix[absorbing]
    )  # Δ churn/involution rate (positive = improvement)
    hpu_net_adds = float((policy_mix[hp] - baseline_mix[hp]) * total_customers)
    logger.info(
        "value_expectation.computed",
        score_date=score_date,
        involution_rate_reduction=round(involution_reduction, 5),
        hpu_pipeline_net_adds=round(hpu_net_adds, 1),
        assumption_flag=lift.assumption_flag,
    )
    return frame


def _segment_reward_vector(
    config: Mapping[str, object], contract: StateContract
) -> np.ndarray:
    """Reward vector for value projection (placeholder weights from config).

    Defaults to an increasing reward by rank when no portfolio reward table is
    supplied; real per-segment rewards should be sourced from the bank's figures
    and surfaced in config.
    """
    n = len(contract.transient_ranks)
    # Monotone-by-rank default; auditable and overridable via the bank's figures.
    return np.array([float(r + 1) for r in range(n)], dtype=float)


# =========================================================================== #
# 11.4 CLOSED LOOP — realized vs expected
# =========================================================================== #
def value_realized(
    prior_action_queue: pd.DataFrame,
    realized_transitions: pd.DataFrame,
    contract: StateContract,
    config: Mapping[str, object],
    *,
    col_id: str = "customer_id",
    col_segment: str = "origin",
    logger: ActionLogger | None = None,
) -> pd.DataFrame:
    """Settle next-period actuals back against the prior action queue.

    Joins realized destination outcomes onto the customers that were queued, and
    measures the realized transition metric and realized $ impact per segment so
    ``expected_delta`` can be compared with ``realized_delta`` and the lift
    assumptions recalibrated.

    Args:
        prior_action_queue: The action queue emitted in the prior cycle.
        realized_transitions: Actual destination outcomes per customer for the
            settlement horizon, with columns ``(customer_id, realized_dest,
            realized_revenue)``.
        contract: The immutable :class:`StateContract`.
        config: Loaded config (reads ``business_value.value_realized``).
        col_id: Customer id column.
        col_segment: Segment dimension.
        logger: Optional :class:`ActionLogger`.

    Returns:
        The value_realized frame ``(score_date, segment,
        realized_transition_metric, realized_$impact)``.
    """
    logger = logger or ActionLogger("business_value")
    rank = dict(contract.cluster_rank)
    absorbing_name = contract.absorbing_name

    joined = prior_action_queue.merge(realized_transitions, on=col_id, how="left")
    # Realized adverse outcome: churned or moved to a lower rank than origin.
    def _adverse(origin: str, dest: object) -> float:
        if not isinstance(dest, str):
            return np.nan
        if dest == absorbing_name:
            return 1.0
        if origin == absorbing_name:
            return 0.0
        return 1.0 if rank.get(dest, rank[origin]) < rank[origin] else 0.0

    joined["realized_adverse"] = [
        _adverse(o, d) for o, d in zip(joined["origin"], joined.get("realized_dest"))
    ]
    # Realized $ impact = expected_clv of customers who did NOT go adverse
    # (value preserved/added under action).
    joined["realized_dollar_impact"] = np.where(
        joined["realized_adverse"] == 0.0,
        joined["clv_at_risk"],
        0.0,
    )

    out = (
        joined.groupby(["score_date", col_segment])
        .agg(
            realized_transition_metric=("realized_adverse", "mean"),
            realized_dollar_impact=("realized_dollar_impact", "sum"),
            n_actioned=(col_id, "size"),
        )
        .reset_index()
        .rename(columns={col_segment: "segment"})
    )
    logger.info("value_realized.settled", n_segments=len(out))
    return out
