from Model import * as model_utils
from Simulation import Simulation
from Graph import Graph

class Runner():
    def __init__(self, model, Simulation, Graph):
        self.model = model
        self.Simulation = Simulation
        self.Graph =

    def run(self, is_training, num_episodes):
        for episode in range(num_episodes):
            old_state = self.Simulation.env.reset(1, self.Simulation.state_space)
            done = False
            total_reward = 0
            ticks = 0
            if not is_training:
                this.Graph.init_results_graph()

            while not done:
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
                        this.Graph.append_results_graph(episode, ticks)
                    break

            if do_train:
                self.Simulation.replay()

if __name__ == '__main__':
    model = model_utils.build_model()
    model_utils.save_model(model)

    simulation = Simulation()
    graph = Graph()
    runner = Runner(model, simulation, graph)

    # training
    runner.run(True, 1000)

    # testing
    runner.run(False, 200)
