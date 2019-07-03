# __filename__: main.py
#
# __description__: implementation of ant colony optimizer + Population Based ACO
# PACO is derived from ACO
#
# __remark__:
#
# __todos__:
"""
@done vis-object dem optimierer übergeben.
@done: welche parameter?
-> war ein bug in update ph.
-> trotzdem noch nicht perfekt.
@done singleton!
@done pheromone-matrix plot- muss ich wohl in der gleichen klasse machen
@done-> ne, ich übergebe immer instanz
"""
# Created by Tobias Wenzel in September 2017
# Copyright (c) 2017 Tobias Wenzel

from __future__ import division

from operator import itemgetter

# error+2d vis
from vis.Path2dVis import Path2DVis
from vis.ErrorVis import ErrorVis


def collect(l, index):
    return map(itemgetter(index), l)

from aco.common_aco import Aco
from aco.paco import Paco


if __name__ == '__main__':

    num_runs = 100
    tau_zero = 1
    num_ants = 20
    num_cities = 10
    tau_delta = 1  # 5
    gamma = .1  # (1-gamma) .1
    alpha = 1
    beta = 5
    aco = Paco(tsp_file="/home/tobias/mygits/python-aco/tsp_problems/rd100.tsp",
              tau_zero=tau_zero, num_ants=num_ants,
              num_cities=num_cities, tau=tau_delta, gamma=gamma,
              alpha=alpha, beta=beta, population_size=5)
    error_plot = True
    live_error = False
    path_vis = False
    plot_ph_matrix = True
    path_visualiser = ph_mtrx_visualiser = error_visualiser = None
    if error_plot:

        error_visualiser = ErrorVis(interactive=live_error, offset=20,
                                    xlim=num_runs, log_scale=False)
        if live_error:
            error_visualiser.set_vis_title(aco.tsp_name + ": " + str(aco.path_length) + " cities\n"
                                           + str(aco.num_ants) + " ants, "
                                           + " tau delta: " + str(aco.tau_delta) + " pop-size: " + str(
                aco.population_size))
    if path_vis:
        path_visualiser = Path2DVis(xymin=-50, xymax=1100,
                                    num_runs=num_runs, offset=50,
                                    interactive=True, sleep_interval=0)
    # if plot_ph_matrix:
    #     ph_mtrx_visualiser = Pheromone_Ways_2DVis(xymin=-50, xymax=1100,
    #                                               num_runs=1, offset=50,
    #                                               interactive=True,
    #                                               sleep_interval=0)

    print(aco)
    output_file = "/home/tobias/Bilder/"
    output_file += "_".join(['paco','ants'+str(num_ants),'cities'+str(num_cities), str(num_runs)+'runs'])+".pdf"
    aco.run_paco(num_runs=num_runs, live_error=live_error, output_file=output_file,
                 path_visualiser= path_visualiser, ph_mtrx_visualiser = ph_mtrx_visualiser,
                 error_visualiser = error_visualiser)
