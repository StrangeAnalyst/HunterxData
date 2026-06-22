"""State contract for the Markov cluster-transition pipeline.

The ordinal cluster hierarchy is the contract and is **never** reordered:

    low_value(0) < emerging_user(1) < growing_user(2) < high_power(3) < high_value(4)

plus the absorbing state ``churned`` (rank 5). Transient ranks are 0..4 and the
absorbing rank is 5. All matrix construction, reconstruction and validation flow
through this module so the contract has exactly one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class StateContract:
    """Immutable view of the ordinal state space.

    Attributes:
        cluster_rank: Mapping of transient cluster name -> rank (0..4).
        absorbing_name: Name of the absorbing state (``churned``).
        absorbing_rank: Rank of the absorbing state (5).
        delta_clip_min: Lower bound for destination reconstruction.
        delta_clip_max: Upper bound for destination reconstruction.
        growth_target_state: Cultivation target cluster name.
    """

    cluster_rank: Mapping[str, int]
    absorbing_name: str
    absorbing_rank: int
    delta_clip_min: int
    delta_clip_max: int
    growth_target_state: str

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "StateContract":
        """Build the contract from the parsed ``state_contract`` config block."""
        sc = config["state_contract"]  # type: ignore[index]
        absorbing = sc["absorbing_state"]
        return cls(
            cluster_rank=dict(sc["cluster_rank"]),
            absorbing_name=str(absorbing["name"]),
            absorbing_rank=int(absorbing["rank"]),
            delta_clip_min=int(sc["delta_clip_min"]),
            delta_clip_max=int(sc["delta_clip_max"]),
            growth_target_state=str(sc["growth_target_state"]),
        )

    # ------------------------------------------------------------------ #
    # Derived views
    # ------------------------------------------------------------------ #
    @property
    def n_states(self) -> int:
        """Total number of states (transient + absorbing)."""
        return len(self.cluster_rank) + 1

    @property
    def transient_ranks(self) -> list[int]:
        """Sorted transient ranks (0..4)."""
        return sorted(self.cluster_rank.values())

    @property
    def all_ranks(self) -> list[int]:
        """All ranks including the absorbing state, ascending."""
        return self.transient_ranks + [self.absorbing_rank]

    @property
    def rank_to_name(self) -> dict[int, str]:
        """Inverse of ``cluster_rank`` extended with the absorbing state."""
        inv = {rank: name for name, rank in self.cluster_rank.items()}
        inv[self.absorbing_rank] = self.absorbing_name
        return inv

    @property
    def ordered_names(self) -> list[str]:
        """State names ordered by rank, including the absorbing state."""
        return [self.rank_to_name[r] for r in self.all_ranks]

    @property
    def growth_target_rank(self) -> int:
        """Rank of the cultivation target state."""
        return self.cluster_rank[self.growth_target_state]

    # ------------------------------------------------------------------ #
    # Reconstruction logic
    # ------------------------------------------------------------------ #
    def reconstruct_destination(self, origin_rank: int, delta: int) -> int:
        """Reconstruct a transient destination rank from origin + delta.

        ``destination = clip(origin_rank + delta, delta_clip_min, delta_clip_max)``.

        Note: this is only for *transient* moves. The absorbing ``churned`` state
        is never produced by delta arithmetic — it is labelled directly by the
        churn rule.

        Args:
            origin_rank: Origin transient rank (0..4).
            delta: Signed rank change.

        Returns:
            Clipped destination rank within the transient band.
        """
        return int(np.clip(origin_rank + delta, self.delta_clip_min, self.delta_clip_max))

    def is_absorbing(self, rank: int) -> bool:
        """Return True if ``rank`` is the absorbing state."""
        return rank == self.absorbing_rank

    def is_downgrade(self, origin_rank: int, dest_rank: int) -> bool:
        """Adverse transition: move to a strictly lower transient rank or churn."""
        if dest_rank == self.absorbing_rank:
            return True
        return dest_rank < origin_rank
