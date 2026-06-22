"""Tests for destination-target construction and reconstruction equivalence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markov_pipeline.targets.build_transition_target import (
    build_destination_target,
    diagnose_transition_distribution,
    resolve_future_state_by_persistence,
)
from markov_pipeline.common.config import get_horizon


def test_build_destination_target_delta_and_reconstruction(contract, config) -> None:
    df = pd.DataFrame(
        {
            "origin_cluster": [0, 2, 4, 3],
            "future_state": [1, 2, 4, 5],  # last one churned
        }
    )
    out = build_destination_target(
        df,
        col_origin="origin_cluster",
        col_future="future_state",
        cluster_rank=contract.cluster_rank,
        contract=contract,
    )
    # Destination names map correctly.
    assert out["destination_state"].tolist() == [
        "emerging_user",
        "growing_user",
        "high_value",
        "churned",
    ]
    # delta = dest - origin for transient; NaN for churn.
    assert out["delta_rank"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(out["delta_rank"].iloc[3])

    # Reconstruction equivalence: clip(origin + delta, 0, 4) == destination rank
    # for all transient rows.
    transient = out[out["destination_rank"] != contract.absorbing_rank]
    for _, row in transient.iterrows():
        recon = contract.reconstruct_destination(
            int(row["origin_cluster"]), int(row["delta_rank"])
        )
        assert recon == int(row["destination_rank"])


def test_persistence_rule_6_of_9(contract, config) -> None:
    horizon = get_horizon(config, "9m")
    assert horizon.required_months == 6
    # Customer A: stays in rank 2 for 7 of 9 months -> resolves to 2.
    # Customer B: churns once -> resolves to absorbing.
    rows = []
    for m in range(1, 10):
        rows.append({"customer_id": "A", "month_offset": m, "state_rank": 2 if m <= 7 else 3, "is_churned": False})
    for m in range(1, 10):
        rows.append({"customer_id": "B", "month_offset": m, "state_rank": 1, "is_churned": m == 4})
    monthly = pd.DataFrame(rows)
    resolved = resolve_future_state_by_persistence(monthly, horizon, contract)
    resolved = resolved.set_index("customer_id")["future_state_rank"]
    assert resolved["A"] == 2
    assert resolved["B"] == contract.absorbing_rank


def test_diagnose_warns_on_rare_classes(contract, config) -> None:
    # Origin 2 sends almost everything to rank 2, with a tiny (<3%) leak to rank 4.
    n = 1000
    df = pd.DataFrame(
        {
            "origin_cluster": [2] * n,
            "destination_rank": [2] * 990 + [4] * 5 + [3] * 5,
        }
    )
    diag = diagnose_transition_distribution(df, contract, config)
    rare = diag[(diag["origin"] == "growing_user") & (diag["is_rare"])]
    assert not rare.empty
