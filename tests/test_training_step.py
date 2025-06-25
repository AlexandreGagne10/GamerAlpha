import gym
import torch

from muzero.model import MuZeroNetwork
from muzero.replay_buffer import ReplayBuffer
from train import play_game, update_weights


def test_training_step():
    env = gym.make("CartPole-v1")
    device = torch.device("cpu")
    network = MuZeroNetwork(env.observation_space.shape, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(10)
    game = play_game(
        env,
        network,
        env.action_space.n,
        num_simulations=1,
        dirichlet_alpha=None,
        exploration_fraction=0.0,
        device=device,
    )
    buffer.add_game(game)
    batch = buffer.sample(1)
    losses = update_weights(network, optimizer, batch, env.action_space.n, device)
    env.close()
    assert all(l >= 0 for l in losses)

