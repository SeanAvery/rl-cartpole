import gym
import random
from time import sleep
import numpy as np
from Graph import Graph

class Simulation():
    def __init__(self, graph):
        # graph utility for results
        self.Graph = Graph

        # environment
        self.env = gym.make('CartPole-v0')

        # state
        self.buckets = (3, 3, 5, 11)
        self.upper_bounds = [self.env.observation_space.high[0], 3.4, self.env.observation_space.high[2], 3.4]
        self.lower_bounds = [self.env.observation_space.low[0], -3.4, self.env.observation_space.low[2], -3.4]
        self.state_length = len(self.lower_bounds)

        # model
        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.97
        self.alpha = 1
        self.min_alpha = 0.05
        self.alpha_decay = 0.97
        self.gamma = 0.995
        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            # choose from q-table
            return np.argmax(self.q_table[state])

    def calc_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.min_epsilon, epsilon)

    def calc_alpha(self):
        alpha = self.alpha * self.alpha_decay
        self.alpha = max(self.min_alpha, alpha)

    def update_q_table(self, old_state, action, reward, new_state):
        self.q_table[old_state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[old_state][action])


    def reduce(self, state):
        ratios = [(state[i] + abs(self.lower_bounds[i])) / (abs(self.upper_bounds[i]) - self.lower_bounds[i]) for i in range(self.state_length)]
        reduced_state = [int((self.buckets[i] - 1) * ratios[i]) for i in range(self.state_length)]
        return tuple(reduced_state)

    def run(self, num_episodes, isTraining):
        if not isTraining:
            self.Graph.init_results_graph(self)
        sleep(0.1)
        for i in range(num_episodes):
            old_state = self.reduce(self.env.reset())
            ticks = 0
            while True:
                ticks += 1
                if not isTraining:
                    self.env.render()
                action = self.choose_action(old_state)
                state, reward, done, info = self.env.step(action)
                new_state = self.reduce(state)
                self.update_q_table(old_state, action, reward, new_state)
                old_state = new_state
                if done:
                    self.calc_epsilon()
                    self.calc_alpha()
                    if not isTraining:
                        self.Graph.append_results_graph(self, i, ticks)
                    break

        self.env.close()

if __name__ == '__main__':
    # init graph
    graph = Graph()
    # init simulation
    simulation = Simulation(graph)
    # run training
    simulation.run(10000 , True)
    # run tests
    print('made it here!')
    simulation.run(100, False)
