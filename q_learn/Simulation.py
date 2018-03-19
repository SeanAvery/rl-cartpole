import gym
import random

class Simulation():
    def __init__(self, num_episodes=10):
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes

    def choose_action(self):
        return random.randint(0, 1)

    def run(self):
        for _ in range(self.num_episodes):
            self.env.reset()

            while True:
                self.env.render()
                action = self.choose_action()
                new_state, reward, done, info = self.env.step(action)

                if done:
                    break

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
