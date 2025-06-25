"""Simple replay buffer storing complete game histories."""

from __future__ import annotations

import random
from collections import deque

from .game_history import GameHistory

class ReplayBuffer:
    """Fixed-size buffer for self-play games."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add_game(self, game_history: GameHistory) -> None:
        """Add a finished game to the buffer."""

        self.buffer.append(game_history)

    def sample(self, batch_size: int):
        """Return a random batch of game histories."""

        if len(self.buffer) < batch_size:
            if len(self.buffer) == 0:
                raise ValueError("Cannot sample from an empty buffer")
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
