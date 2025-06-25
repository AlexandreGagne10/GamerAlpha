from __future__ import annotations

class GameHistory:
    """Container for a single self-play game."""

    def __init__(self) -> None:
        self.observations: list = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.root_values: list[float] = []
        self.policies: list[list[float]] = []

    def store_search_statistics(self, root: "TreeNode") -> None:
        """Record the search policy and value for the current position."""

        visit_counts = [child.visit_count for child in root.children.values()]
        policy = [count / sum(visit_counts) for count in visit_counts]
        self.policies.append(policy)
        self.root_values.append(root.value())

    def append(self, observation, action: int, reward: float) -> None:
        """Append a transition to the history."""

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
