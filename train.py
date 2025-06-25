"""Training script for the minimal MuZero implementation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from muzero.model import MuZeroNetwork
from muzero.mcts import run_mcts
from muzero.replay_buffer import ReplayBuffer
from muzero.game_history import GameHistory


def save_checkpoint(
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    episode: int,
    path: Path,
) -> None:
    """Save model and optimizer state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model_state": network.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    path: Path,
) -> int:
    """Load model and optimizer state. Returns the episode to resume from."""

    checkpoint = torch.load(path, map_location="cpu")
    network.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    return int(checkpoint.get("episode", 0))


def evaluate(
    env: gym.Env,
    network: MuZeroNetwork,
    episodes: int,
    action_space: int,
    num_simulations: int,
    device: torch.device,
    dirichlet_alpha: float | None = None,
    exploration_fraction: float = 0.25,
) -> float:
    """Run evaluation episodes and return the average reward."""

    total_reward = 0.0
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(
                observation, dtype=torch.float, device=device
            ).unsqueeze(0)
            root = run_mcts(
                network,
                obs_tensor,
                action_space,
                num_simulations,
                dirichlet_alpha=dirichlet_alpha,
                exploration_fraction=exploration_fraction,
            )
            action, _ = select_action(root)
            observation, reward, done, trunc, _ = env.step(action)
            done = done or trunc
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / episodes


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
) -> tuple[float, float, float, float]:
    """Update network weights from a batch of game histories.

    Returns
    -------
    tuple[float, float, float, float]
        Value loss, reward loss, policy loss and total loss.
    """

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

    return (
        value_loss.item(),
        reward_loss.item(),
        policy_loss.item(),
        loss.item(),
    )


def play_game(
    env: gym.Env,
    network: MuZeroNetwork,
    action_space: int,
    num_simulations: int,
    dirichlet_alpha: float | None,
    exploration_fraction: float,
    device: torch.device,
) -> GameHistory:
    """Play one game in the environment using MCTS for action selection."""

    observation, _ = env.reset()
    done = False
    history = GameHistory()
    while not done:
        obs_tensor = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
        root = run_mcts(
            network,
            obs_tensor,
            action_space,
            num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            exploration_fraction=exploration_fraction,
        )
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
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='Path to save/load training checkpoint')
    parser.add_argument('--eval-interval', type=int, default=0,
                        help='Run evaluation every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=1,
                        help='Number of evaluation episodes')
    parser.add_argument('--early-stop', type=int, default=0,
                        help='Stop if no improvement after this many evaluations')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run the model on')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3,
                        help='Dirichlet alpha for root exploration noise')
    parser.add_argument('--exploration-fraction', type=float, default=0.25,
                        help='Fraction of Dirichlet noise to add to root prior')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    )

    env = gym.make(args.env)
    try:
        observation_shape = env.observation_space.shape
        action_space = env.action_space.n

        network = MuZeroNetwork(observation_shape, action_space).to(device)
        optimizer = optim.Adam(network.parameters(), lr=1e-3)

        start_episode = 0
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            start_episode = load_checkpoint(network, optimizer, checkpoint_path)
            logging.info("Loaded checkpoint from %s (episode %d)", checkpoint_path, start_episode)

        buffer = ReplayBuffer(100)

        best_reward = float('-inf')
        no_improve = 0

        for episode in range(start_episode, args.episodes):
            game = play_game(
                env,
                network,
                action_space,
                args.simulations,
                args.dirichlet_alpha,
                args.exploration_fraction,
                device,
            )
            buffer.add_game(game)
            if len(buffer) >= 1:
                batch = buffer.sample(1)
                v_loss, r_loss, p_loss, loss = update_weights(
                    network, optimizer, batch, action_space, device
                )
                logging.info(
                    "Episode %d training loss v=%.4f r=%.4f p=%.4f total=%.4f",
                    episode + 1,
                    v_loss,
                    r_loss,
                    p_loss,
                    loss,
                )
            logging.info("Episode %d finished with %d steps", episode + 1, len(game.rewards))

            if args.eval_interval and (episode + 1) % args.eval_interval == 0:
                avg_reward = evaluate(
                    env,
                    network,
                    args.eval_episodes,
                    action_space,
                    args.simulations,
                    device,
                    dirichlet_alpha=None,
                    exploration_fraction=args.exploration_fraction,
                )
                logging.info("Evaluation reward after episode %d: %.2f", episode + 1, avg_reward)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    no_improve = 0
                else:
                    no_improve += 1
                if args.early_stop and no_improve >= args.early_stop:
                    logging.info("Early stopping at episode %d", episode + 1)
                    save_checkpoint(network, optimizer, episode + 1, checkpoint_path)
                    return

            save_checkpoint(network, optimizer, episode + 1, checkpoint_path)
    finally:
        env.close()


if __name__ == '__main__':
    main()
