import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Graph():
    def __init__(self):
        plt.ion()

    def init_results_graph(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 210)
        # plt.draw()
        self.x, self.y = [], []

    def append_results_graph(self, episode, ticks):
        self.x.append(episode)
        self.y.append(ticks)
        self.ax.plot(self.x, self.y)

    def save_to_gif(self):
        self.init_gif_graph()
        anim = FuncAnimation(self.fig_gif, self.update_results_gif, frames=np.arange(0, 50), interval=100)
        anim.save('results.gif', dpi=80, writer='imagemagick')

    def init_gif_graph(self):
        self.fig_gif, self.ax_gif = plt.subplots()
        self.ax_gif.set_ylim(0, 210)

    def update_results_gif(self, i):
        self.x_gif = self.x[0:i]
        self.y_gif = self.y[0:i]
        plot, = self.ax_gif.plot(self.x_gif, self.y_gif)
        return plot, self.ax_gif
