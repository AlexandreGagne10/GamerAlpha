# GamerAlpha

This repository contains a minimal implementation of the MuZero algorithm in PyTorch. The agent can be trained on standard Gym environments such as `CartPole-v1`, `LunarLander-v3` and `Acrobot-v1`.

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script with the desired environment name. The script supports
saving checkpoints and periodic evaluation:

```bash
python train.py --env CartPole-v1 --episodes 10 --simulations 50 \
    --checkpoint checkpoint.pth --eval-interval 5 --eval-episodes 2
```

Use `--checkpoint` to resume training from a previous run. Set `--eval-interval`
to periodically evaluate the agent during training.

The code is lightweight and meant for educational purposes. It does not implement all optimisations from the original MuZero paper but demonstrates the core ideas: learning a latent model of the environment, planning with Monte Carlo Tree Search and training from self-play games.

Run tests with:

```bash
pytest -q
```
