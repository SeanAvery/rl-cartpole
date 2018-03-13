import gym
import numpy as np
import random

class Simulation():
    def __init__(self, model):
        self.model = model
        self.build_gym()
        self.memory = []

        # hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        self.batch_size = 2

    def build_gym(self):
        self.env = gym.make('CartPole-v0')
        self.set_action_size()
        self.set_state_size()

    def set_action_size(self):
        self.action_size = self.env.action_space.n

    def set_state_size(self):
        self.state_size = self.env.observation_space.shape[0]

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            random_choice = random.randrange(self.action_size)
            print('random choice', random_choice)
            return random_choice
        else:
            choice = np.argmax(self.model.predict(state)[0])
            print('choice', choice)
            return choice

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            for old_state, action, reward, new_state, done in mini_batch:
                if not done:
                    target = (reward + self.gamma * np.amax(self.model.predict(new_state)))
                    print('calculated target', target)
                else:
                    target = reward
                    print('target', target)

                target_f = self.model.predict(old_state)
                print('target_f', target_f)
                target_f[0][action] = target

                self.model.fit(old_state, target_f, epochs=1, verbose=0)
                print('self.model', self.model)
                
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

''' TESTING '''
if __name__ == '__main__':
    simulation = Simulation()
    print('action_size', simulation.action_size, type(simulation.action_size))
    print('state_size', simulation.state_size)
