import os, sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gym
import torch

from muzero.model import MuZeroNetwork
from muzero.mcts import run_mcts


def select_action(root):
    visit_counts = [child.visit_count for child in root.children.values()]
    actions = list(root.children.keys())
    probs = torch.tensor(visit_counts, dtype=torch.float)
    probs = probs / probs.sum()
    action = int(torch.multinomial(probs, num_samples=1))
    return action


def test_cartpole_episode():
    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=0)
    network = MuZeroNetwork(env.observation_space.shape, env.action_space.n)
    done = False
    steps = 0
    while not done and steps < 5:
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        root = run_mcts(network, obs_tensor, env.action_space.n, num_simulations=3)
        action = select_action(root)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    env.close()
    assert steps > 0
