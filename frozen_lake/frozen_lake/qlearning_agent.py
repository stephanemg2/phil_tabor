import numpy as np


class Agent:
    def __init__(self, num_states, num_actions,
                 epsilon_i=1.0,
                 epsilon_f=0.0,
                 n_epsilon=0.1,
                 lr=0.01,
                 gamma=0.95):
        """
        num_states: the number of states in the discrete state space
        num_actions: the number of actions in the discrete action space
        epsilon_i: the initial value for epsilon
        epsilon_f: the final value for epsilon
        n_epsilon: a float between 0 and 1 determining at which point
        during training epsilon should have decayed from epsilon_i to
        epsilon_f
        lr: the learning rate
        gamma: the decay rate
        """

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon = epsilon_i
        self.n_epsilon = n_epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))

    def decay_epsilon(self, n):
        """
        Decays the agent's exploration rate according to n, which is a
        float between 0 and 1 describing how far along training is,
        with 0 meaning 'just started' and 1 meaning 'done'.
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))

    def act(self, s_t):
        """
        Epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[s_t])

    def update(self, s_t, a_t, r_t, s_t_next):
        """
        Uses the q-learning update rule to update the agent's predictions
        for Q(s_t, a_t).
        """
        q_next = np.max(self.Q[s_t_next])
        self.Q[s_t, a_t] = self.Q[s_t, a_t] + self.lr * (r_t + self.gamma * q_next - self.Q[s_t, a_t])
