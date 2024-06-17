# __filename__: main.py
#
# __description__: implementation of ant colony optimizer + Population Based ACO
# PACO is derived from ACO
#
# Created by Tobias Wenzel in September 2017
# Copyright (c) 2017 Tobias Wenzel
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from aco.common_aco import Aco
from aco.paco import Paco
from aco.tsp_parser import load_tsp_problem
from vis.visualization import animate

if __name__ == "__main__":
    num_runs = 50
    tau_zero = 1
    num_ants = 50
    num_cities = 15
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
        keep_paths=True,
    )
    aco = Paco(
        tau_zero=tau_zero,
        num_ants=num_ants,
        tau=tau_delta,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        population_size=5,
        keep_paths=True,
    )

    plot_path_lengths = True
    path_vis = False
    print(aco)
    points = load_tsp_problem("tsp_problems/rd100.tsp", num_cities)
    aco.run_paco(num_runs=num_runs, points=points)

    if plot_path_lengths:
        values = aco.path_lengths
        df = pd.DataFrame(
            list(zip(np.arange(len(values)), values)),
            columns=["path-length", "iteration"],
        )
        sns.lineplot(x="iteration", y="path-length", data=df)
        plt.show()
    if path_vis:
        x_values = [value[0] for value in aco.best_paths]
        y_values = [value[1] for value in aco.best_paths]
        animate(x_values=x_values, y_values=y_values)
