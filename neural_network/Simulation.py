import gym

class Simulation():
    def __init__(self):
        self.build_gym()
        self.memory = []

        # hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 32

    def build_gym(self):
        self.env = gym.make('CartPole-v0')
        selft.set_action_size()
        self.set_state_size()

    def set_action_size(self):
        self.action_size = self.env.action_space.n

    def set_state_size(self):
        self.state_size = self.env.observation_space.shape[0]

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])



''' TESTING '''
if __name__ == '__main__':
    simulation = Simulation()
    print('action_size', simulation.action_size, type(simulation.action_size))
    print('state_size', simulation.state_size)
