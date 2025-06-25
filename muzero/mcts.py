"""Monte Carlo tree search utilities for MuZero."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from muzero.model import MuZeroNetwork

class TreeNode:
    """Node in the search tree used by MCTS."""

    def __init__(self, prior: float) -> None:
        """Create a new tree node.

        Parameters
        ----------
        prior : float
            Prior probability of choosing this node.
        """

        self.visit_count: int = 0
        self.to_play: int = 0
        self.prior: float = prior
        self.value_sum: float = 0.0
        self.children: Dict[int, TreeNode] = {}
        self.reward: float = 0.0

    def expanded(self) -> bool:
        """Return True if the node has been expanded."""

        return len(self.children) > 0

    def value(self) -> float:
        """Average value of this node based on simulations."""

        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child(node: TreeNode, c_puct: float = 1.0) -> Tuple[int, TreeNode]:
    """Select a child according to the PUCT rule.

    Parameters
    ----------
    node : TreeNode
        Parent node from which to select.
    c_puct : float, optional
        Exploration constant, by default ``1.0``.

    Returns
    -------
    Tuple[int, TreeNode]
        The action index and the selected child node.
    """

    best_score = -float("inf")
    best_action: int | None = None
    best_child: TreeNode | None = None
    for action, child in node.children.items():
        pb_c = c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = child.value() + pb_c
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_action is not None and best_child is not None
    return best_action, best_child


def expand_node(
    node: TreeNode,
    latent: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    action_space: int,
) -> None:
    """Expand a leaf node using the policy from the network.

    Parameters
    ----------
    node : TreeNode
        Node to expand.
    latent : tuple[Tensor, Tensor, Tensor, Tensor]
        Network output containing latent state, value, reward and policy logits.
    action_space : int
        Number of possible actions.
    """

    policy_logits = latent[3]
    policy = torch.softmax(policy_logits, dim=1)[0].detach().cpu().numpy()
    for a in range(action_space):
        node.children[a] = TreeNode(float(policy[a]))


def backpropagate(search_path: Iterable[TreeNode], value: float, discount: float) -> None:
    """Update nodes along the search path with the simulation results."""

    for node in reversed(list(search_path)):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + discount * value


def run_mcts(
    network: MuZeroNetwork,
    root_observation: torch.Tensor,
    action_space: int,
    num_simulations: int = 50,
    discount: float = 0.997,
) -> TreeNode:
    """Execute MCTS starting from the given observation.

    Parameters
    ----------
    network : MuZeroNetwork
        Network used to evaluate states and actions.
    root_observation : Tensor
        Initial environment observation.
    action_space : int
        Number of discrete actions.
    num_simulations : int, optional
        How many simulations to perform, by default ``50``.
    discount : float, optional
        Discount factor used during backpropagation, by default ``0.997``.

    Returns
    -------
    TreeNode
        The root node after simulations.
    """

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
            action_one_hot = torch.zeros(1, action_space)
            action_one_hot[0, action] = 1.0
            latent_state, value, reward, policy_logits = network.recurrent_inference(latent_state, action_one_hot)
            node.reward = reward.item()
            search_path.append(node)
        expand_node(node, (latent_state, value, reward, policy_logits), action_space)
        backpropagate(search_path, value.item(), discount)
    return root
