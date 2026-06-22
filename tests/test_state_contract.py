"""Tests for the immutable ordinal state contract and reconstruction logic."""

from __future__ import annotations

import pytest

from markov_pipeline.common.state_contract import StateContract


def test_ordinal_hierarchy_is_fixed(contract: StateContract) -> None:
    assert contract.cluster_rank == {
        "low_value": 0,
        "emerging_user": 1,
        "growing_user": 2,
        "high_power": 3,
        "high_value": 4,
    }
    assert contract.absorbing_rank == 5
    assert contract.n_states == 6
    assert contract.ordered_names[-1] == "churned"


@pytest.mark.parametrize(
    "origin,delta,expected",
    [(0, -1, 0), (2, 1, 3), (4, 2, 4), (3, -3, 0), (1, 0, 1)],
)
def test_reconstruct_destination_clips(
    contract: StateContract, origin: int, delta: int, expected: int
) -> None:
    """destination = clip(origin_rank + delta, 0, 4)."""
    assert contract.reconstruct_destination(origin, delta) == expected


def test_is_downgrade_and_absorbing(contract: StateContract) -> None:
    assert contract.is_downgrade(3, 1) is True
    assert contract.is_downgrade(2, 4) is False
    assert contract.is_downgrade(2, contract.absorbing_rank) is True
    assert contract.is_absorbing(contract.absorbing_rank) is True
    assert contract.is_absorbing(0) is False
