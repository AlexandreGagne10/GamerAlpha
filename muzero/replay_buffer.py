import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add_game(self, game_history):
        self.buffer.append(game_history)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            if len(self.buffer) == 0:
                raise ValueError("Cannot sample from an empty buffer")
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
