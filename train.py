"""Training script for the minimal MuZero implementation."""

from __future__ import annotations

import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim

from muzero.model import MuZeroNetwork
from muzero.mcts import run_mcts
from muzero.replay_buffer import ReplayBuffer
from muzero.game_history import GameHistory


def select_action(root: "TreeNode") -> tuple[int, list[float]]:
    """Sample an action from the visit count distribution.

    Parameters
    ----------
    root : TreeNode
        Root node after MCTS search.

    Returns
    -------
    tuple[int, list[float]]
        The selected action and the search policy as a list.
    """

    visit_counts = [child.visit_count for child in root.children.values()]
    probs = torch.tensor(visit_counts, dtype=torch.float)
    probs = probs / probs.sum()
    action = torch.multinomial(probs, num_samples=1).item()
    return action, probs.tolist()


def update_weights(
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    batch: list[GameHistory],
    action_space: int,
    device: torch.device,
    discount: float = 0.997,
) -> None:
    """Update network weights from a batch of game histories."""

    obs_batch = []
    targets_value = []
    targets_policy = []
    targets_reward = []
    for game in batch:
        for i in range(len(game.actions)):
            obs_batch.append(game.observations[i])
            targets_value.append(game.root_values[i])
            targets_policy.append(game.policies[i])
            targets_reward.append(game.rewards[i])

    obs_batch = torch.tensor(obs_batch, dtype=torch.float, device=device)
    targets_value = torch.tensor(targets_value, dtype=torch.float, device=device).unsqueeze(1)
    targets_reward = torch.tensor(targets_reward, dtype=torch.float, device=device).unsqueeze(1)
    targets_policy = torch.tensor(targets_policy, dtype=torch.float, device=device)

    latent, value, reward, policy_logits = network.initial_inference(obs_batch)
    value_loss = nn.functional.mse_loss(value, targets_value)
    reward_loss = nn.functional.mse_loss(reward, targets_reward)
    log_probs = nn.functional.log_softmax(policy_logits, dim=1)
    policy_loss = -(targets_policy * log_probs).sum(dim=1).mean()

    loss = value_loss + reward_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def play_game(
    env: gym.Env,
    network: MuZeroNetwork,
    action_space: int,
    num_simulations: int,
    device: torch.device,
) -> GameHistory:
    """Play one game in the environment using MCTS for action selection."""

    observation, _ = env.reset()
    done = False
    history = GameHistory()
    while not done:
        obs_tensor = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
        root = run_mcts(network, obs_tensor, action_space, num_simulations)
        action, policy = select_action(root)
        history.store_search_statistics(root)
        next_observation, reward, done, trunc, info = env.step(action)
        history.append(observation, action, reward)
        observation = next_observation
    return history


def main() -> None:
    """Entry point to run self-play training."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--simulations', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run the model on')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    env = gym.make(args.env)
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n

    network = MuZeroNetwork(observation_shape, action_space).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(100)

    for episode in range(args.episodes):
        game = play_game(env, network, action_space, args.simulations, device)
        buffer.add_game(game)
        if len(buffer) >= 1:
            batch = buffer.sample(1)
            update_weights(network, optimizer, batch, action_space, device)
        print(f'Episode {episode+1} finished with {len(game.rewards)} steps')


if __name__ == '__main__':
    main()
