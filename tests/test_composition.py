"""Tests for Markov composition: stochasticity, absorption, fundamental matrix,
CLV-MRP and the n-step Markov-property check."""

from __future__ import annotations

import numpy as np
import pytest

from markov_pipeline.common.guardrails import (
    GuardrailViolation,
    l1_check_absorbing,
    l1_check_row_stochastic,
)
from markov_pipeline.composition.markov_dynamics import (
    absorption_analysis,
    clv_mrp,
    fundamental_matrix,
    markov_property_error,
    n_step,
    partition,
    stationary_distribution,
)


def test_rows_sum_to_one(absorbing_chain, config) -> None:
    l1_check_row_stochastic(absorbing_chain, config)
    np.testing.assert_allclose(absorbing_chain.sum(axis=1), 1.0, atol=1e-9)


def test_churned_is_absorbing(absorbing_chain, contract) -> None:
    l1_check_absorbing(absorbing_chain, contract)
    absorbing = contract.absorbing_rank
    assert absorbing_chain[absorbing, absorbing] == 1.0
    assert absorbing_chain[absorbing].sum() == 1.0


def test_non_stochastic_matrix_rejected(config) -> None:
    bad = np.eye(6)
    bad[0, 0] = 0.5  # row no longer sums to 1
    with pytest.raises(GuardrailViolation):
        l1_check_row_stochastic(bad, config)


def test_fundamental_matrix_finite(absorbing_chain, contract) -> None:
    Q, _ = partition(absorbing_chain, len(contract.transient_ranks))
    N = fundamental_matrix(Q)
    assert np.all(np.isfinite(N))
    # Expected lifetime is positive and finite.
    res = absorption_analysis(absorbing_chain, len(contract.transient_ranks))
    assert np.all(res.expected_lifetime > 0)
    assert np.all(np.isfinite(res.expected_lifetime))


def test_absorption_probability_sums_to_one(absorbing_chain, contract) -> None:
    """With a single absorbing state, lifetime absorption prob is 1 everywhere."""
    res = absorption_analysis(absorbing_chain, len(contract.transient_ranks))
    np.testing.assert_allclose(res.absorption_prob.sum(axis=1), 1.0, atol=1e-8)


def test_clv_mrp_positive_and_monotone_in_gamma(absorbing_chain, contract) -> None:
    k = len(contract.transient_ranks)
    r = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    v_low = clv_mrp(absorbing_chain, r, 0.5, k)
    v_high = clv_mrp(absorbing_chain, r, 0.95, k)
    assert np.all(v_low > 0)
    # Higher discount factor -> higher discounted CLV.
    assert np.all(v_high >= v_low)


def test_clv_rejects_bad_gamma(absorbing_chain, contract) -> None:
    with pytest.raises(ValueError):
        clv_mrp(absorbing_chain, np.ones(5), 1.5, len(contract.transient_ranks))


def test_n_step_matches_empirical_within_tolerance(config) -> None:
    """Composing the one-step kernel reproduces the empirical t-step transitions."""
    rng = np.random.default_rng(5)
    # Construct a genuine first-order chain and simulate to get empirical t-step.
    n = 6
    P = np.zeros((n, n))
    for i in range(5):
        P[i] = rng.dirichlet(np.ones(n))
    P[5, 5] = 1.0

    t = 3
    n_paths = 200_000
    starts = rng.integers(0, 5, size=n_paths)
    states = starts.copy()
    for _ in range(t):
        current = states.copy()  # snapshot: avoid double-stepping within a step
        for s in range(n):
            mask = current == s
            if mask.any():
                states[mask] = rng.choice(n, size=int(mask.sum()), p=P[s])
    empirical = np.zeros((n, n))
    for s in range(n):
        rows = states[starts == s]
        if len(rows):
            empirical[s] = np.bincount(rows, minlength=n) / len(rows)
    empirical[5] = 0.0
    empirical[5, 5] = 1.0

    tol = float(config["composition"]["markov_property_test"]["tolerance"])
    err = markov_property_error(P, empirical, t)
    # Monte-Carlo noise: allow a small multiple of the configured tolerance.
    assert err < tol * 2.5


def test_stationary_distribution_is_a_distribution(absorbing_chain, contract) -> None:
    pi = stationary_distribution(absorbing_chain, len(contract.transient_ranks))
    assert np.all(pi >= 0)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-8)
