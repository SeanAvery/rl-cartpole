import gym
import random
from time import sleep
import numpy as np

class Simulation():
    def __init__(self, num_episodes=2):
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes
        self.buckets = (0, 0, 5, 11)
        self.upper_bounds = [self.env.observation_space.high[0], 3.4, self.env.observation_space.high[2], 3.4]
        self.lower_bounds = [self.env.observation_space.low[0], -3.4, self.env.observation_space.low[2], -3.4]
        self.state_length = len(self.lower_bounds)

        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        print('empty q_table', self.q_table, self.q_table.shape)
        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.97
        self.alpha = 1
        self.min_alpha = 0.05
        self.alpha_decay = 0.97
        self.gamma = 0.995

        print('lower bounds', self.lower_bounds)
        print('upper bounds', self.upper_bounds)

    def choose_action(self, state):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            # choose from q-table

    def calc_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        if epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = epsilon

    def calc_alpha(self):
        alpha = self.alpha * self.alpha_decay
        if alpha < self.min_alpha:
            self.alpha = self.min_alpha
        else:
            self.epsilon = epsilon

    def uppdate_q_table(self, old_state, action, reward, new_state):
        self.q_table[old_state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[old_state][action])


    def reduce(self, state):
        ratios = [(state[i] + abs(self.lower_bounds[i])) / (abs(self.upper_bounds[i]) - self.lower_bounds[i]) for i in range(self.state_length)]
        reduced_state = [int((self.buckets[i] - 1) * ratios[i]) for i in range(self.state_length)]
        return (reduced_state)

    def run(self):
        for i in range(self.num_episodes):
            old_state = self.reduce(self.env.reset())

            while True:
                self.env.render()
                action = self.choose_action(state)
                state, reward, done, info = self.env.step(action)
                new_state = self.reduce(state)
                self.update_q_table(self, old_state, action, reward, new_state)
                old_state = new_state
                sleep(0.01)
                if done:
                    self.calc_epsilon()
                    self.calc_alpha()
                    break

        self.env.close()

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
