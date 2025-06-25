from __future__ import annotations

from typing import List


class GameHistory:
    """Container storing data from a single self-play game."""

    def __init__(self) -> None:
        """Initialize empty history buffers."""

        self.observations: List[object] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.root_values: List[float] = []
        self.policies: List[List[float]] = []

    def store_search_statistics(self, root: "TreeNode") -> None:
        """Record the policy and value estimates from the MCTS root."""

        visit_counts = [child.visit_count for child in root.children.values()]
        policy = [count / sum(visit_counts) for count in visit_counts]
        self.policies.append(policy)
        self.root_values.append(root.value())

    def append(self, observation: object, action: int, reward: float) -> None:
        """Add a single timestep to the history."""

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
