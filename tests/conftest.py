"""Shared pytest fixtures for the Markov pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_pipeline.common.config import load_config
from markov_pipeline.common.state_contract import StateContract


@pytest.fixture(scope="session")
def config() -> dict:
    """The loaded pipeline config."""
    return load_config()


@pytest.fixture(scope="session")
def contract(config) -> StateContract:
    """The immutable state contract."""
    return StateContract.from_config(config)


@pytest.fixture
def absorbing_chain(contract) -> np.ndarray:
    """A valid row-stochastic absorbing chain (5 transient + churned)."""
    rng = np.random.default_rng(7)
    n = contract.n_states
    P = np.zeros((n, n))
    for i in contract.transient_ranks:
        P[i] = rng.dirichlet(np.ones(n) * 2.0)
    absorbing = contract.absorbing_rank
    P[absorbing] = 0.0
    P[absorbing, absorbing] = 1.0
    return P


@pytest.fixture
def labelled_frame(config, contract) -> pd.DataFrame:
    """Synthetic labelled training frame with features, origin and destination."""
    rng = np.random.default_rng(11)
    feature_cols = list(config["features"]["base_features"]) + list(
        config["features"]["path_features"]
    )
    n = 3000
    df = pd.DataFrame({c: rng.normal(size=n) for c in feature_cols})
    origin = rng.integers(0, 5, size=n)
    df["origin_cluster"] = origin
    # Destination depends on origin + a feature signal; ~10% churn.
    signal = (df[feature_cols[0]] > 0).astype(int)
    dest = np.clip(origin + signal - rng.integers(0, 2, size=n), 0, 4)
    churn = rng.random(n) < 0.1
    df["destination_rank"] = np.where(churn, contract.absorbing_rank, dest)
    return df
