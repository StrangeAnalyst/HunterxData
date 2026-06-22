"""Tests for the business-value layer: policy matrix, action queue, value expectation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markov_pipeline.business_value.actionability import (
    CausalLiftProvider,
    apply_policy_to_matrix,
    build_action_queue,
    compute_downgrade_prob,
    expected_actioning_volume,
    value_expectation,
)
from markov_pipeline.common.guardrails import l1_check_absorbing, l1_check_row_stochastic


def test_policy_matrix_stays_stochastic_and_absorbing(absorbing_chain, contract, config) -> None:
    lift = CausalLiftProvider(config).get_lift()
    P_action = apply_policy_to_matrix(absorbing_chain, lift, contract)
    l1_check_row_stochastic(P_action, config)
    l1_check_absorbing(P_action, contract)


def test_policy_reduces_churn_mass(absorbing_chain, contract, config) -> None:
    lift = CausalLiftProvider(config).get_lift()
    P_action = apply_policy_to_matrix(absorbing_chain, lift, contract)
    absorbing = contract.absorbing_rank
    for origin in contract.transient_ranks:
        assert P_action[origin, absorbing] <= absorbing_chain[origin, absorbing] + 1e-12


def test_assumed_lift_is_flagged(config) -> None:
    lift = CausalLiftProvider(config).get_lift()
    assert lift.assumption_flag is True


def test_action_queue_value_and_band_gates(contract, config) -> None:
    traj = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "score_date": "2026-06-30",
            "horizon": "6m",
            "origin": ["high_value", "low_value", "growing_user"],
            "predicted_destination": ["high_power", "low_value", "high_power"],
            "expected_clv": [5000.0, 100.0, 1500.0],  # cust 2 below clv_floor
            "churn_prob": [0.2, 0.9, 0.3],  # cust 2 outside addressable band
            "prob_reach_high_power": [0.05, 0.0, 0.4],
        }
    )
    rows = []
    for cid, o in [(1, "high_value"), (2, "low_value"), (3, "growing_user")]:
        for d in contract.ordered_names:
            p = 0.5 if (o == "high_value" and d == "low_value") else 0.1
            rows.append({"customer_id": cid, "origin": o, "dest": d, "prob": p})
    transitions = pd.DataFrame(rows)

    downgrade = compute_downgrade_prob(transitions, contract)
    queue = build_action_queue(traj, downgrade, contract, config)
    # Customer 2 is filtered (value + band). Customers 1 and 3 remain.
    assert set(queue["customer_id"]) == {1, 3}
    assert queue.iloc[0]["priority_score"] >= queue.iloc[-1]["priority_score"]


def test_value_expectation_carries_assumption_flag(absorbing_chain, contract, config) -> None:
    mix = np.array([0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    frame = value_expectation(
        absorbing_chain, mix, contract, config, score_date="2026-06-30"
    )
    assert frame["assumption_flag"].all()
    assert set(frame["segment"]) == set(contract.ordered_names)
    # Portfolio mix shares are valid probabilities.
    assert (frame["policy_metric"] >= -1e-9).all()
