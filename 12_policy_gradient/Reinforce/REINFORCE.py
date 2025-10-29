import numpy as np


class ReinforceAgent:
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.actions = range(env.action_space.n)  # actions
        self.theta = np.random.random(4+len(self.actions))  # must match dimension of feature_vector
        self.probs = np.random.uniform(0, 1, len(self.actions))

    def _feature_vector(self, state, action):
        # feature vector consists of state and one-hot-encoded action
        x, y = state 
        action_vec = np.zeros(len(self.actions))
        action_vec[action] = 1
        feature = (1, x, y, x*y)  #TODO: try different feature vectors
        feature_vec = np.append(feature, action_vec)
        return feature_vec

    def softmax_policy(self, state, action):
        '''TODO 
        return the probability of choosing action a in state s  = exp(x(s, a)^T  theta) / sum_b  exp(x(s, b)^T theta) 
        where x(s,a) is the corresponding feature vector for (s, a)
        '''
        pass

    def score_function(self, state, action):
        '''TODO
        returns the score function: gradient(log (pi (a | s))) =  x(s, a) - \sum_b p i(b | s) * x(s, b)
        where x(s,a) is the feature vector for (s, a)
        '''
        pass

    def choose_action(self, state):
        '''TODO
        samples an action with probability self.softmax_policy(state, a) for each action a
        '''
        pass

    def update_montecarlo(self, states, actions, rewards):
        '''TODO
        iterate over the time steps from the last to the first
        update self.theta at each step i by using the discounted return of that time step, i.e. self.theta += alpha * gamma**i * discounted_return * score_function
        '''
        pass