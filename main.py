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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aco.common_aco import Aco
from aco.tsp_parser import load_tsp_problem

# from vis.Path2dVis import Path2DVis

if __name__ == "__main__":
    num_runs = 50
    tau_zero = 1
    num_ants = 20
    num_cities = 5
    tau_delta = 1  # 5
    gamma = 0.1  # (1-gamma) .1
    alpha = 1
    beta = 5

    aco = Aco(
        tau_zero=tau_zero,
        num_ants=num_ants,
        tau=tau_delta,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
    )
    # aco = Paco(
    #     tau_zero=tau_zero,
    #     num_ants=num_ants,
    #     tau=tau_delta,
    #     gamma=gamma,
    #     alpha=alpha,
    #     beta=beta,
    #     population_size=5,
    # )

    plot_pathlengths = True
    path_vis = True
    plot_ph_matrix = True
    path_visualiser = ph_mtrx_visualiser = error_visualiser = None

    # if path_vis:
    #     path_visualiser = Path2DVis(
    #         xymin=-50,
    #         xymax=1100,
    #         num_runs=num_runs,
    #         offset=50,
    #         interactive=True,
    #         sleep_interval=0,
    #     )
    # @todo
    # if plot_ph_matrix:
    #     ph_mtrx_visualiser = Pheromone_Ways_2DVis(xymin=-50, xymax=1100,
    #                                               num_runs=1, offset=50,
    #                                               interactive=True,
    #                                               sleep_interval=0)

    print(aco)
    points = load_tsp_problem("tsp_problems/rd100.tsp", num_cities)
    aco.run_aco(num_runs=num_runs, points=points)

    if plot_pathlengths:
        values = aco.path_lengths
        df = pd.DataFrame(
            list(zip(np.arange(len(values)), values)),
            columns=["path-length", "iteration"],
        )
        sns.lineplot(x="iteration", y="path-length", data=df)
        plt.show()
