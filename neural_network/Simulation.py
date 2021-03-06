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
        self.env.max_episode_steps = 500
        self.set_action_size()
        self.set_state_size()

    def set_action_size(self):
        self.action_size = self.env.action_space.n

    def set_state_size(self):
        self.state_size = self.env.observation_space.shape[0]

    def choose_action(self, state):
        if np.random.rand() <= self.Model.epsilon:
            random_choice = random.randrange(self.action_size)
            return random_choice
        else:
            choice = np.argmax(self.Model.model.predict(state))
            return choice

    def replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(self.memory, min(len(self.memory), self.Model.batch_size))

        for old_state, action, reward, new_state, done in mini_batch:
            y_target = self.Model.model.predict(old_state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.Model.gamma * np.max(self.Model.model.predict(new_state)[0])
            x_batch.append(old_state[0])
            y_batch.append(y_target[0])

        self.Model.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        if self.Model.epsilon > self.Model.epsilon_min:
            self.Model.epsilon *= self.Model.epsilon_decay

''' TESTING '''
if __name__ == '__main__':
    simulation = Simulation()
    print('action_size', simulation.action_size, type(simulation.action_size))
    print('state_size', simulation.state_size)
