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
        self.min_epsilon = 0.05
        self.min_alpha = 0.05
        self.gamma = 0.995

        print('lower bounds', self.lower_bounds)
        print('upper bounds', self.upper_bounds)

    def choose_action(self, state):
        return 0 if state[3] < 5 else 1

    def reduce(self, state):
        ratios = [(state[i] + abs(self.lower_bounds[i])) / (abs(self.upper_bounds[i]) - self.lower_bounds[i]) for i in range(self.state_length)]
        reduced_state = [int((self.buckets[i] - 1) * ratios[i]) for i in range(self.state_length)]
        return (reduced_state)

    def run(self):
        for i in range(self.num_episodes):
            state = self.reduce(self.env.reset())

            while True:
                self.env.render()
                action = self.choose_action(state)
                state, reward, done, info = self.env.step(action)
                state = self.reduce(state)
                sleep(0.1)
                if done:
                    break

        self.env.close()

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
