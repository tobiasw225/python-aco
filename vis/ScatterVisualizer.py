import numpy as np

import matplotlib.pyplot as plt
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# __filename__: my_visualilation.py
#
# __description__: contains Scatter_Visualizer, a simple scatter plot class

#
# also: variables for 'print'
# __remark__:
#
# __todos__: stop vis when stopping!
#
# Created by Tobias Wenzel in September 2017
# Copyright (c) 2017 Tobias Wenzel

class ScatterVisualizer:

    def __init__(self, interactive=False, xlim=0, ylim=0,
                 offset=50, log_scale=False, sexify=False):
        self.interactive= interactive
        self.my_plot = self.fig = self.ax = None
        if interactive:
            plt.ion()
            # self.fig, self.ax = plt.subplots()
            # self.my_plot = self.ax.scatter([], [])
            # self.ax.set_xlim(0, xlim)
            # self.my_plot.set_edgecolor('white')
            self.init_me()
            self.ax.set_xlim(0, xlim)

        self.log_scale = log_scale

        if self.log_scale and interactive:
            self.ax.set_yscale('log')
        # if sexify:
        #     sexiplot.latexify()

        self.offset = offset
        self.xlim = xlim
        self.ylim = ylim

    def init_me(self):
        self.fig, self.ax = plt.subplots()
        self.my_plot = self.ax.scatter([], [])
        self.my_plot.set_edgecolor('white')

    def set_vis_title(self,title=""):
        self.ax.set_title(title)

    def set_labels(self, xlabel="", ylabel=""):
        self.ax.set_xlabel(xlabel=xlabel)
        self.ax.set_ylabel(ylabel=ylabel)

    def set_yx_lim(self, x=[],y=[]):
            """

            :param x:
            :param y:
            :return:
            """
            self.ax.set_xlim(x[0], x[1])
            self.ax.set_ylim(y[0], y[1])

    def plot_my_data(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """

        self.init_me()
        self.ax.set_xlim(0, self.xlim)
        #print(np.min(y, axis=0))
        try:
            self.ax.set_ylim([np.min(y, axis=0)[0]-self.offset,
                                 np.max(y, axis=0)[0]+self.offset])
        except IndexError:
            self.ax.set_ylim([np.min(y, axis=0)-self.offset,
                                 np.max(y, axis=0)+self.offset])
        self.set_labels(xlabel="(time)", ylabel="(value)")
        if self.log_scale:
            self.ax.set_yscale('log')
        self.ax.plot(x, y, 'o-')
        #plt.show()
        #time.sleep(3)

    def save_me(self, filename=""):
        plt.savefig(filename)