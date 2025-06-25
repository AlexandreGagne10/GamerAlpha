"""Training loop for a minimal MuZero implementation."""

from __future__ import annotations

import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from muzero.model import MuZeroNetwork
from muzero.mcts import run_mcts
from muzero.replay_buffer import ReplayBuffer
from muzero.game_history import GameHistory


def select_action(root) -> Tuple[int, List[float]]:
    """Choose an action from the root node's visit counts."""

    visit_counts = [child.visit_count for child in root.children.values()]
    actions = list(root.children.keys())
    probs = torch.tensor(visit_counts, dtype=torch.float)
    probs = probs / probs.sum()
    action = int(torch.multinomial(probs, num_samples=1))
    return action, probs.tolist()


def update_weights(
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    batch: List[GameHistory],
    action_space: int,
    discount: float = 0.997,
) -> None:
    """Perform a single network update from a batch of games."""

    obs_batch = []
    actions_batch = []
    targets_value = []
    targets_policy = []
    targets_reward = []
    for game in batch:
        for i in range(len(game.actions)):
            obs_batch.append(game.observations[i])
            actions_batch.append(game.actions[i])
            targets_value.append(game.root_values[i])
            targets_policy.append(game.policies[i])
            targets_reward.append(game.rewards[i])

    obs_batch = torch.tensor(obs_batch, dtype=torch.float)
    actions_batch = torch.tensor(actions_batch, dtype=torch.long)
    targets_value = torch.tensor(targets_value, dtype=torch.float).unsqueeze(1)
    targets_reward = torch.tensor(targets_reward, dtype=torch.float).unsqueeze(1)
    targets_policy = torch.tensor(targets_policy, dtype=torch.float)

    latent, value, reward, policy_logits = network.initial_inference(obs_batch)
    value_loss = nn.functional.mse_loss(value, targets_value)
    reward_loss = nn.functional.mse_loss(reward, targets_reward)
    policy_loss = nn.functional.cross_entropy(policy_logits, actions_batch)

    loss = value_loss + reward_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def play_game(
    env: gym.Env,
    network: MuZeroNetwork,
    action_space: int,
    num_simulations: int,
) -> GameHistory:
    """Run one episode using MCTS to select actions."""

    observation, _ = env.reset()
    done = False
    history = GameHistory()
    while not done:
        obs_tensor = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        root = run_mcts(network, obs_tensor, action_space, num_simulations)
        action, policy = select_action(root)
        history.store_search_statistics(root)
        next_observation, reward, done, trunc, info = env.step(action)
        history.append(observation, action, reward)
        observation = next_observation
    return history


def main() -> None:
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--simulations', type=int, default=10)
    args = parser.parse_args()

    env = gym.make(args.env)
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n

    network = MuZeroNetwork(observation_shape, action_space)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(100)

    for episode in range(args.episodes):
        game = play_game(env, network, action_space, args.simulations)
        buffer.add_game(game)
        if len(buffer) >= 1:
            batch = buffer.sample(1)
            update_weights(network, optimizer, batch, action_space)
        print(f'Episode {episode+1} finished with {len(game.rewards)} steps')


if __name__ == '__main__':
    main()
