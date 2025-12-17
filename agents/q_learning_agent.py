import numpy as np
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, action_space_n, **kwargs):
        super().__init__(action_space_n, **kwargs)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_n)
        
        q_values = [self.get_q(state, a) for a in range(self.action_space_n)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q(state, action)
        
        # Max Q for next state (Off-Policy)
        next_max_q = max([self.get_q(next_state, a) for a in range(self.action_space_n)])
        
        # Q-Learning Update Rule
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        
        self.q_table[(state, action)] = new_q
