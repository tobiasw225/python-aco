# __filename__: main.py
#
# __description__: implementation of ant colony optimizer + Population Based ACO
# PACO is derived from ACO
#
# __remark__:
#
# __todos__:
"""
@todo visualisation not working
"""
# Created by Tobias Wenzel in September 2017
# Copyright (c) 2017 Tobias Wenzel

from __future__ import division

from operator import itemgetter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from vis.Path2dVis import Path2DVis


def collect(l, index):
    return map(itemgetter(index), l)

from aco.common_aco import Aco
from aco.paco import Paco


if __name__ == '__main__':

    num_runs = 50
    tau_zero = 1
    num_ants = 20
    num_cities = 17
    tau_delta = 1  # 5
    gamma = .1  # (1-gamma) .1
    alpha = 1
    beta = 5

    # aco = Aco(tsp_file="../res/ulysses16.tsp",
    #           tau_zero=tau_zero, num_ants=num_ants,
    #           num_cities=num_cities, tau=tau_delta, gamma=gamma,
    #           alpha=alpha, beta=beta)
    aco = Paco(tsp_file="../res/rd100.tsp",
               tau_zero=tau_zero,
               num_ants=num_ants,
               num_cities=num_cities,
               tau=tau_delta,
               gamma=gamma,
               alpha=alpha,
               beta=beta,
               population_size=5)

    plot_pathlengths = True
    path_vis = True
    #plot_ph_matrix = True
    path_visualiser = ph_mtrx_visualiser = error_visualiser = None


    if path_vis:
        path_visualiser = Path2DVis(xymin=-50, xymax=1100,
                                    num_runs=num_runs, offset=50,
                                    interactive=True, sleep_interval=0)
    # @todo
    # if plot_ph_matrix:
    #     ph_mtrx_visualiser = Pheromone_Ways_2DVis(xymin=-50, xymax=1100,
    #                                               num_runs=1, offset=50,
    #                                               interactive=True,
    #                                               sleep_interval=0)

    print(aco)
    aco.run_aco(num_runs=num_runs, path_visualiser=path_visualiser)

    if plot_pathlengths:

        values = aco.error_rates
        df = pd.DataFrame(list(zip(np.arange(len(values)), values)),
                          columns=['path-length', 'iteration'])
        sns.lineplot(x="iteration", y="path-length", data=df)
        plt.show()
