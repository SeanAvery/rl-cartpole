import matplotlib.pyplot as plt
from time import sleep

class Graph():
    def __init__(self):
        plt.ion()

    def init_results_graph(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 300)
        plt.draw()
        self.x, self.y = [], []

    def append_results_graph(self, episode, ticks):
        self.x.append(episode)
        self.y.append(ticks)
        self.ax.plot(self.x, self.y)

    def init_epsilon_graph(self):
        self.fig2, self.ax2 = plt.subplots()
        self.ax.set_ylim(0, 1)
        plt.draw()
        self.epsilon_x = []
        self.epsilon_y = []

    def append_epsilon_graph(self, episode, epsilon):
        self.x_epsilon.append(episode)
        self.y_epsilon.append(epsilon)
        self.ax2.plot(self.x_epsilon, self.y_epsilon)

''' TESTING '''

if __name__ == '__main__':
    graph = Graph()
    graph.init_results_graph()

    for i in range(100):
        sleep(0.5)
        graph.append_results_graph(i, i * 2)
