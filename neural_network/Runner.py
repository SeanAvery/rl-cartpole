from Model import *
from Simulation import Simulation
from Graph import Graph

class Runner():
    def __init__(self, model, Simulation, Graph):
        self.model = model
        self.Simulation = Simulation
        self.Graph = Graph

    def run(self, is_training, num_episodes):
        if not is_training:
            self.Graph.init_results_graph()

        for episode in range(num_episodes):
            print('episode', episode)
            old_state = self.Simulation.env.reset().reshape(1, self.Simulation.state_size)
            done = False
            total_reward = 0
            ticks = 0

            while not done:
                if not is_training:
                    self.Simulation.env.render()
                ticks += 1
                action = self.Simulation.choose_action(old_state)
                new_state, reward, done, info = self.Simulation.env.step(action)
                new_state = new_state.reshape(1, self.Simulation.env.observation_space.shape[0])
                total_reward += reward

                if is_training:
                    self.Simulation.memory.append((old_state, action, reward, new_state, done))

                new_state = old_state

                if done:
                    if not is_training:
                        self.Graph.append_results_graph(episode, ticks)
                    break

            if is_training:
                self.Simulation.replay()
                self.model.save_weights()

if __name__ == '__main__':
    model = build_model()
    save_model(model.to_json(), 'dense_model_1')

    simulation = Simulation(model)
    graph = Graph()
    runner = Runner(model, simulation, graph)

    # training
    runner.run(True, 10)

    # testing
    # runner.run(False, 20)
