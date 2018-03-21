import matplotlib.pyplot as plt

class Graph():
    def __init__(self):
        plt.ion()

    def init_results_graph(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 210)
        plt.draw()
        self.x, self.y = [], []

    def append_results_graph(self, episode, ticks):
        self.x.append(episode)
        self.y.append(ticks)
        self.ax.plot(self.x, self.y)
