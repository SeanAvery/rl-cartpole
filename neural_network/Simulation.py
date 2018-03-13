import gym
import numpy as np
import random

class Simulation():
    def __init__(self, model):
        self.Model = model
        self.build_gym()
        self.memory = []

    def build_gym(self):
        self.env = gym.make('CartPole-v0')
        self.set_action_size()
        self.set_state_size()

    def set_action_size(self):
        self.action_size = self.env.action_space.n

    def set_state_size(self):
        self.state_size = self.env.observation_space.shape[0]

    def choose_action(self, state):
        if np.random.rand() <= self.Model.epsilon:
            random_choice = random.randrange(self.action_size)
            print('random choice', random_choice)
            return random_choice
        else:
            choice = np.argmax(self.Model.predict(state)[0])
            print('choice', choice)
            return choice

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        else:
            mini_batch = random.sample(self.memory, self.Model.batch_size)
            for old_state, action, reward, new_state, done in mini_batch:
                if not done:
                    target = (reward + self.Model.gamma * np.amax(self.Model.predict(new_state)))
                    print('calculated target', target)
                else:
                    target = reward
                    print('target', target)

                target_f = self.Model.predict(old_state)
                print('target_f', target_f)
                target_f[0][action] = target

                self.Model.fit(old_state, target_f, epochs=1, verbose=0)
                print('self.Model', self.Model)

            if self.Model.epsilon > self.Model.epsilon_min:
                self.Model.epsilon *= self.Model.epsilon_decay

''' TESTING '''
if __name__ == '__main__':
    simulation = Simulation()
    print('action_size', simulation.action_size, type(simulation.action_size))
    print('state_size', simulation.state_size)
