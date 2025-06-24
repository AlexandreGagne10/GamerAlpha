class GameHistory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.root_values = []
        self.policies = []

    def store_search_statistics(self, root):
        visit_counts = [child.visit_count for child in root.children.values()]
        policy = [count / sum(visit_counts) for count in visit_counts]
        self.policies.append(policy)
        self.root_values.append(root.value())

    def append(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
