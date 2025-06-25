"""Minimal Monte Carlo Tree Search implementation."""

from __future__ import annotations

import math
import numpy as np
import torch

from .model import MuZeroNetwork

class TreeNode:
    """Node used by MCTS."""

    def __init__(self, prior: float) -> None:
        self.visit_count: int = 0
        self.to_play: int = 0
        self.prior: float = prior
        self.value_sum: float = 0.0
        self.children: dict[int, "TreeNode"] = {}
        self.reward: float = 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child(node: TreeNode, c_puct: float = 1.0) -> tuple[int, TreeNode]:
    """Select the child with maximum UCB score."""

    best_score = -float("inf")
    best_action = 0
    best_child = node
    for action, child in node.children.items():
        pb_c = c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = child.value() + pb_c
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def expand_node(node: TreeNode, latent, action_space: int) -> None:
    """Expand a leaf node using the network's policy."""

    policy_logits = latent[3]
    policy = torch.softmax(policy_logits, dim=1)[0].detach().cpu().numpy()
    for a in range(action_space):
        node.children[a] = TreeNode(policy[a])


def backpropagate(search_path: list[TreeNode], value: float, discount: float) -> None:
    """Update node statistics on the path back to the root."""

    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + discount * value


def run_mcts(network: MuZeroNetwork, root_observation: torch.Tensor, action_space: int, num_simulations: int = 50, discount: float = 0.997) -> TreeNode:
    """Run Monte Carlo Tree Search starting from the given observation."""

    device = root_observation.device
    root = TreeNode(1.0)
    latent = network.initial_inference(root_observation)
    expand_node(root, latent, action_space)
    root.reward = latent[2].item()

    for _ in range(num_simulations):
        node = root
        latent_state = latent[0]
        search_path = [node]
        while node.expanded():
            action, node = select_child(node)
            action_one_hot = torch.zeros(1, action_space, device=device)
            action_one_hot[0, action] = 1.0
            latent_state, value, reward, policy_logits = network.recurrent_inference(latent_state, action_one_hot)
            node.reward = reward.item()
            search_path.append(node)
        expand_node(node, (latent_state, value, reward, policy_logits), action_space)
        backpropagate(search_path, value.item(), discount)
    return root
