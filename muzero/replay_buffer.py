"""Simple replay buffer used to store game histories for training."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List

class ReplayBuffer:
    """Fixed-size buffer that stores completed games."""

    def __init__(self, capacity: int = 10000) -> None:
        """Create a new buffer.

        Parameters
        ----------
        capacity : int, optional
            Maximum number of games to keep, by default ``10000``.
        """

        self.capacity = capacity
        self.buffer: Deque[object] = deque(maxlen=capacity)

    def add_game(self, game_history: object) -> None:
        """Add a game history to the buffer."""

        self.buffer.append(game_history)

    def sample(self, batch_size: int) -> List[object]:
        """Randomly sample game histories from the buffer."""

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return the number of stored games."""

        return len(self.buffer)
