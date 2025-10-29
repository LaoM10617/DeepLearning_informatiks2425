import numpy as np


class ReinforceAgent:
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.actions = range(env.action_space.n)  # actions
        self.theta = np.random.random(4+len(self.actions)) # must match dimension of feature_vector
        self.probs = np.random.uniform(0, 1, len(self.actions))

    def _feature_vector(self, state, action):
        # feature vector consists of state and one-hot-encoded action
        x, y = state 
        action_vec = np.zeros(len(self.actions))
        action_vec[action] = 1
        feature = (1, x, y, x*y)
        feature_vec = np.append(feature, action_vec)
        return feature_vec

    def softmax_policy(self, state, action):
        # probability of choosing action a in state s  = exp(x(s, a)^T  theta) / sum_b  exp(x(s, b)^T theta)
        x = self._feature_vector(state, action)
        numerator = np.e ** (np.dot(np.transpose(x), self.theta))
        normalizer = np.sum([np.e ** (np.dot(np.transpose(self._feature_vector(state, a)), self.theta)) for a in self.actions])
        return numerator / normalizer

    def score_function(self, state, action):
        # gradient(log (pi (a | s))) =  x(s, a) - \sum_b pi(b | s) * x(s, b)
        x = self._feature_vector(state, action)
        expected_value = np.sum([self.softmax_policy(state, a) * self._feature_vector(state, a) for a in self.actions], axis = 0)  # = sum_b pi(b | s) * x(s, b)
        gradient_log = x - expected_value
        return gradient_log

    def choose_action(self, state):
        probs = [self.softmax_policy(state, a) for a in self.actions]
        action = np.random.choice(range(len(self.actions)), p=probs)
        return action

    def update_montecarlo(self, states, actions, rewards):
        nr_steps = len(states)
        discounted_return = 0
        for i in range(nr_steps, 0, -1):  # iterate over time steps from last to first
            s = states[i-1]
            a = actions[i-1]
            r = rewards[i-1]
            discounted_return = self.gamma * discounted_return + r
            self.theta += self.alpha * self.gamma**i * discounted_return * self.score_function(s, a)
