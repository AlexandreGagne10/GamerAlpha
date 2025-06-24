import math
import numpy as np
import torch

class TreeNode:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.reward = 0.0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def select_child(node, c_puct=1.0):
    best_score = -float('inf')
    best_action = None
    best_child = None
    for action, child in node.children.items():
        pb_c = c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = child.value() + pb_c
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def expand_node(node, latent, action_space):
    policy_logits = latent[3]
    policy = torch.softmax(policy_logits, dim=1)[0].detach().cpu().numpy()
    for a in range(action_space):
        node.children[a] = TreeNode(policy[a])


def backpropagate(search_path, value, discount):
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + discount * value


def run_mcts(network, root_observation, action_space, num_simulations=50, discount=0.997):
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
