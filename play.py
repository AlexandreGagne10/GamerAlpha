import argparse
import gym
import torch

from muzero.model import MuZeroNetwork
from muzero.mcts import run_mcts
from train import select_action


def load_model(path: str, network: MuZeroNetwork, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    network.load_state_dict(state_dict)


def play(env_name: str, model_path: str, episodes: int, simulations: int, device: torch.device) -> None:
    env = gym.make(env_name, render_mode="human")
    observation_shape = env.observation_space.shape
    action_space = env.action_space.n

    network = MuZeroNetwork(observation_shape, action_space).to(device)
    load_model(model_path, network, device)

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            env.render()
            obs_tensor = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
            root = run_mcts(network, obs_tensor, action_space, simulations)
            action, _ = select_action(root)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        print(f"Episode {episode + 1} finished")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play episodes with a trained MuZero model")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    play(args.env, args.model, args.episodes, args.simulations, device)


if __name__ == "__main__":
    main()
