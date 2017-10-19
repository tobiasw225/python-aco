# __filename__: my_ant.py
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

import sys
import itertools
import heapq
import numpy as np
import time

from scipy.spatial import distance
from operator import itemgetter
#from matplotlib.lines import Line2D
from multiprocessing import Queue

from src.tsp_parser import *
sys.path.append('/home/tobias/Dropbox/Programme/python_vis/src')
# error+2d vis
from my_visualisation import *

def collect(l, index):
    return map(itemgetter(index), l)


class Ant:
    def __init__(self, path_length):
        """

        :param path_length:
        """
        self.path_length = path_length
        self.current_solution = []#np.zeros((self.path_length,1))
        self.has_seen_cities = []

    def refresh_cities(self):
        self.has_seen_cities = []

    def start_city(self):
        """

        :return:
        """
        return np.random.randint(low=0,high=self.path_length)


class Aco:
    class __Aco:
        def __init__(self, num_ants=10,
                     tau_zero=0.4, tsp_file="",
                     num_cities=10, tau=0.5, gamma=0.8,
                     alpha=1, beta=5):
            """

            :param num_ants:
            :param tau_zero:
            :param tsp_file:
            :param num_cities:
            :param tau:
            :param gamma:
            :param alpha:
            :param beta:
            """
            data = read_tsp_data(tsp_file)
            dimension = detect_dimension(data)
            cities_set = get_cities(data, dimension, num_cities=num_cities)

            self.cities_tups = city_tup(cities_set)
            self.cities_dict = create_cities_dict(cities_tups)
            self.path_length = len(self.cities_dict)
            self.tsp_name = tsp_file.split("/")[-1].split(".")[0]

            self.dist_matrix = []
            self.tau_matrix = []
            self.tau_zero = tau_zero
            #self.make_dist_matrix()
            self.make_tau_matrix()
            self.num_ants = num_ants
            self.ants = [Ant(self.path_length) for i in range(num_ants)]
            self.beta = beta
            self.alpha = alpha
            self.length_of_path = 0.0
            self.tau_delta = tau  # pheromone-value added if win
            self.gamma = gamma  # verdunstung

            self.plot_ph_matrix = self.plot_build_ants_path = False
            self.error_plot = self.path_vis = False

        """
            Plot Functions
        """
        def plot_options(self, plot_ph_matrix, error_plot,
                         plot_build_ants_path, path_vis):

            self.plot_ph_matrix = plot_ph_matrix
            self.plot_build_ants_path = plot_build_ants_path
            self.error_plot = error_plot
            self.path_vis = path_vis


        def get_best_ant_path(self, best_ant_index=0):
            """
            gets coordinates of shortest paths and forwards it to the vis
            :param best_ant_index:
            :return:
            """
            x, y = [], []
            winner_ant = self.ants[best_ant_index]
            for i in range(0, self.path_length):
                if i != self.path_length - 1:
                    x.append(self.cities_dict[winner_ant.current_solution[i] + 1][0])
                    x.append(self.cities_dict[winner_ant.current_solution[i + 1] + 1][0])
                    y.append(self.cities_dict[winner_ant.current_solution[i] + 1][1])
                    y.append(self.cities_dict[winner_ant.current_solution[i + 1] + 1][1])
                else:
                    x.append(self.cities_dict[winner_ant.current_solution[-1] + 1][0])
                    x.append(self.cities_dict[winner_ant.current_solution[0] + 1][0])
                    y.append(self.cities_dict[winner_ant.current_solution[-1] + 1][1])
                    y.append(self.cities_dict[winner_ant.current_solution[0] + 1][1])
            return x, y

        """
            Heuristic Functions + Pheromone-Evalution
        """
        def get_heuristic(self, i, j):
            """

            :param i:
            :param j:
            :return:
            """
            res = 0.0
            try:
                res = (1/distance.euclidean(self.cities_dict[i+1],
                                            self.cities_dict[j+1]))**self.beta
            except ZeroDivisionError:
                res= 1
            return res

        def get_tau(self, i, j):
            return (self.tau_matrix[i][j])**self.alpha

        def get_tau_times_heuristic(self, i, j):
            return self.get_heuristic(i,j)*self.get_tau(i,j)

        """
            Functions to get new paths + evaluation of length
        """
        def find_new_ways(self, ant_solutions):
            """
            find new paths for each ant in colony
            :param ant_solutions:
            :return:
            """
            for a, ant in zip(range(self.num_ants), self.ants):
                ant.current_solution = []#np.zeros((self.path_length,1))
                # starte bei zufälliger Stadt
                start_index = ant.start_city()
                ant.current_solution.append(start_index)
                ant.has_seen_cities.append(start_index)
                if self.plot_build_ants_path:
                    path_array = list(self.cities_dict[start_index + 1])
                    self.path_plot.set_offsets(path_array)
                    time.sleep(self.pause_interval)
                # jetzt suche stadt mit höchster W'keit
                # dafür summiere ich zunächst alle tau werte, die noch nicht im set sind.

                j = 1
                while j < self.path_length:
                    # theoretisch muss ich die summe wirklich nur 1x bilden!
                    sum_of_not_visited = 0.0
                    for city in self.cities_dict:
                        # cities starten mit 1
                        if city - 1 not in ant.has_seen_cities:
                            #print(city-1)
                            sum_of_not_visited += \
                                self.get_tau_times_heuristic(start_index, city - 1)

                    pij = np.zeros((len(self.cities_dict),1))
                    i = 0
                    for city in self.cities_dict:
                        # cities starten mit 1
                        if city-1 not in ant.has_seen_cities:
                            pij[i] = self.get_tau_times_heuristic(
                                start_index, city-1)/sum_of_not_visited
                        i += 1
                    best_city_index = np.argmax(pij)  # greatest prob

                    ant.current_solution.append(best_city_index)
                    ant.has_seen_cities.append(best_city_index)
                    # # this way i don't have to compute the rest every time
                    ## läuft leider gar nicht
                    # sum_of_not_visited -= self.get_tau_times_heuristic(
                    # #      start_index, best_city_index)
                    # print(sum_of_not_visited, self.get_tau_times_heuristic(
                    #       start_index, best_city_index))
                    start_index = best_city_index  # continue with this city
                    if self.plot_build_ants_path:
                        path_array = self.path_plot.get_offsets()
                        path_array = np.append(path_array,
                                               list(self.cities_dict[best_city_index+1]))
                        self.path_plot.set_offsets(path_array)
                        self.path_fig.canvas.draw()
                        time.sleep(self.pause_interval)

                    j += 1
                    # end for path_length
                if self.plot_build_ants_path:
                    self.path_plot.set_offsets([])
                res = self.get_length_of_path(ant)

                ant_solutions[a] = res
                if len(set(ant.current_solution)) != self.path_length:
                    print(len(ant.current_solution)," should be ",len(set(ant.current_solution)))
                    break
                ant.refresh_cities()

        def get_length_of_path(self, ant):
            """
            not to confuse with path_length!
            :return:
            """
            length_of_path = 0.0
            for i in range(-1, len(ant.current_solution)-1):
                length_of_path += distance.euclidean(
                    self.cities_dict[ant.current_solution[i]+1],
                    self.cities_dict[ant.current_solution[i+1]+1])
            ant.length_of_path = length_of_path
            return length_of_path

        def update_tau_matrix(self, ant):
            """
            best ant is allowed to update
            :param ant:
            :return:
            """
            self.tau_matrix *= (1-self.gamma)
            for i in range(0, len(ant.current_solution)):
                try:
                    self.tau_matrix\
                        [ant.current_solution[i]][ant.current_solution[i+1]]\
                        += self.tau_delta
                except IndexError:
                    self.tau_matrix\
                        [ant.current_solution[i]][ant.current_solution[0]] \
                        += self.tau_delta

        def run_aco(self, num_runs=10, output_file="", live_error=True,
                    path_visualiser=None, ph_mtrx_visualiser=None,
                    error_visualiser=None):
            """
            Implementation of the ACO with the possibility to visualise error,
            shortest path and the pheromone-matrix. This is done in the following way:
            each ant guesses a paths through all points/cities according to a heuristic
            and the current pheromone value on the connection between the corresponding
            points. Meaning: The more pheromone and the closer, the higher the probability
            of choosing the city as next position in the path. After an iteration, the lenghts
            of the paths are compared and the connections between the cities of the winning ant
            get all of the pheromone (could be done otherwise). Also, each connection is decreased
            by (1-gamma). This is done until i==num_runs.

            :param num_runs:
            :param output_file:
            :param live_error:
            :param path_visualiser:
            :param ph_mtrx_visualiser:
            :param error_visualiser:
            :return:
            """

            best_solutions = np.zeros((num_runs,1))
            for iteration in range(num_runs):
                ant_solutions = np.zeros((self.num_ants, 1))
                self.find_new_ways(ant_solutions=ant_solutions)
                best_ant_index = np.argmin(ant_solutions)  # greatest prob
                self.update_tau_matrix(self.ants[best_ant_index])

                if ph_mtrx_visualiser:
                    ph_mtrx_visualiser.plot_ph_matrix_fn(self)

                if path_visualiser:
                    x, y = self.get_best_ant_path(best_ant_index)
                    path_visualiser.plot_path(x, y, iteration)

                best_solutions[iteration] = self.ants[best_ant_index].length_of_path
                if error_visualiser and live_error:
                    error_visualiser.update_with_point(x=iteration,
                                                       y=self.ants[best_ant_index].length_of_path)
                print(self.ants[best_ant_index].length_of_path)

            if not live_error and error_visualiser:
                error_visualiser.plot_my_data(range(0, num_runs), best_solutions)
                time.sleep(5)
            elif error_visualiser:
                plt.savefig(output_file)


        """
            Idea: don't calculate path every time but create matrix in
            the beginning.
        """
        def make_dist_matrix(self):
            """
            wäre cool, wenn ich das zu beginn so aufbauen könnte, muss aber nicht unbedingt.
            -> wird im moment nicht benutzt
            :return:
            """
            city_coords = self.cities_dict.values()
            city_coords_tuples = list(itertools.combinations(city_coords, 2))
            city_keys = self.cities_dict.keys()
            city_keys_tuples = list(itertools.combinations(city_keys, 2))

            # warum nicht einfach mit ner city,city matrix?
            for key_pair, coord_pair in zip(city_keys_tuples, city_coords_tuples):
                # (Stadt-Indices (A,B), und ihre Distance)
                heapq.heappush(
                    self.dist_matrix, (key_pair,
                                       distance.euclidean(coord_pair[0], coord_pair[1])))

        def make_tau_matrix(self):
            """
            @todo: diagonale =0
            :return:
            """
            self.tau_matrix = np.zeros((self.path_length+1,self.path_length+1))
            self.tau_matrix.fill(self.tau_zero)

            np.fill_diagonal(self.tau_matrix, 0)

    instance = None

    def __init__(self, num_ants=10,
                 tau_zero=0.4, tsp_file="",
                 num_cities=10, tau=0.5, gamma=0.8,
                 alpha=1, beta=5):
        if not Aco.instance:
            Aco.instance = Aco.__Aco(num_ants,
                 tau_zero, tsp_file,
                 num_cities, tau, gamma,
                 alpha, beta)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __str__(self):
        res = ""
        res += colon_line+"\n"+ half_space_line+"ACO\n"+colon_line+"\n"
        res += "ant number "+str(self.num_ants)+"\n"
        res += "tau "+ str(self.tau_delta) +"\n"
        res += "gamma " + str(self.gamma) +"\n"
        res += "num cities "+str(self.path_length)
        return res


class Paco(Aco):
    class __Paco(Aco):
        def __init__(self, num_ants=10,
                     tau_zero=0.4, tsp_file="",
                     num_cities=10, tau=0.5, gamma=0.0,
                     alpha=1, beta=5, population_size=5):
            super().__init__(num_ants,
                     tau_zero, tsp_file,
                     num_cities, tau, gamma,
                     alpha, beta)

            self.fifo_solution_q = Queue(maxsize=population_size)
            self.population_size=population_size

        def add_solution(self, ant_index):
            """

            :param solution:
            :return:
            """
            solution = self.ants[ant_index].current_solution
            if self.fifo_solution_q.full():
                self.remove_pheromone_of_solution(self.fifo_solution_q.get())
            self.add_pheromone_of_solution(solution)
            self.fifo_solution_q.put(solution)

        def add_pheromone_of_solution(self, solution):
            """
            :param solution:
            :return:
            """

            for i in range(-1, len(solution)-1):
                self.tau_matrix \
                    [solution[i]][solution[i + 1]] \
                    += self.tau_delta

        def remove_pheromone_of_solution(self, solution):
            """

            :param solution:
            :return:
            """
            for i in range(-1, len(solution)-1):
                self.tau_matrix \
                    [solution[i]][solution[i + 1]] \
                    -= self.tau_delta

        def run_paco(self, num_runs=50, live_error=True, output_file="",
                     path_visualiser=None, ph_mtrx_visualiser=None,
                     error_visualiser=None):
            """
            Simple implementation of the P-ACO algorithm. This is similar to ACO, but there
            is no evaporation step. In this case a population of solution influences the choice
            of the ants. After 'population_size' steps, the solution looses it's impact and the
            corresponding pheromone value is removed from the pheromone matrix.
            :param num_runs:
            :param live_error:
            :param output_file:
            :param path_visualiser:
            :param ph_mtrx_visualiser:
            :param error_visualiser:
            :return:
            """

            best_solutions = np.zeros((num_runs, 1))
            for iteration in range(num_runs):
                ant_solutions = np.zeros((self.num_ants, 1))
                self.find_new_ways(ant_solutions=ant_solutions)
                best_ant_index = np.argmin(ant_solutions)  # greatest prob
                self.add_solution(best_ant_index)
                best_solutions[iteration] = self.ants[best_ant_index].length_of_path
                print(iteration, self.ants[best_ant_index].length_of_path)
                if ph_mtrx_visualiser:
                    ph_mtrx_visualiser.plot_ph_matrix_fn(self, iteration)  # incl. save_fig
                if path_visualiser:
                    x, y = self.get_best_ant_path(best_ant_index)
                    path_visualiser.plot_path(x, y, iteration)  # incl. save_fig
                if error_visualiser and live_error:
                    error_visualiser.update_with_point(x=iteration,
                                                       y=self.ants[best_ant_index].length_of_path)

            if not live_error and error_visualiser:
                # error_visualiser.set_vis_title(aco.tsp_name + ": " + str(aco.path_length) + " cities\n"
                #                                + str(aco.num_ants) + " ants, "
                #                                + " tau delta: " + str(aco.tau_delta) + " pop-size: " + str(
                #     aco.population_size))
                error_visualiser.plot_my_data(range(0, num_runs), best_solutions)
                plt.tight_layout()  # needed to have the labels in the figure
                plt.savefig(output_file)
                #plt.show()
            elif error_visualiser:
                # @todo bug
                plt.savefig(output_file)
            # if ph_mtrx_visualiser:
            #     create_video(source_path="'/home/tobias/Bilder/tsp/ph_vis/*.png'",
            #                  frame_rate=5,
            #                  output_file="/home/tobias/Bilder/tsp/ph_vis/my_video.mp4")
            # if path_visualiser:
            #     create_video(source_path="'/home/tobias/Bilder/tsp/path_vis/*.png'",
            #                  frame_rate=5,
            #                  output_file="/home/tobias/Bilder/tsp/path_vis/my_video.mp4")

    instance = None

    def __init__(self, num_ants=10,
                 tau_zero=0.4, tsp_file="",
                 num_cities=10, tau=0.5, gamma=0.8,
                 alpha=1, beta=5, population_size=5):
        if not Paco.instance:
            Paco.instance = Paco.__Paco(num_ants,
                 tau_zero, tsp_file,
                 num_cities, tau, gamma,
                 alpha, beta, population_size)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __str__(self):
        res = ""
        res += colon_line+"\n"+ half_space_line+"P-ACO\n"+colon_line+"\n"
        res += "ant number "+str(self.num_ants)+"\n"
        res += "tau "+ str(self.tau_delta) +"\n"
        res += "pop-size "+str(self.population_size)+"\n"
        res += "num cities "+str(self.path_length)
        return res

if __name__ == '__main__':


    num_runs = 100
    tau_zero = 1
    num_ants = 20
    num_cities = 50
    tau_delta = 1  # 5
    gamma = .1  # (1-gamma) .1
    alpha = 1
    beta = 5
    aco = Paco(tsp_file="/home/tobias/Dropbox/Programme/tsp/tsp_problems/rd100.tsp",
              tau_zero=tau_zero, num_ants=num_ants,
              num_cities=num_cities, tau=tau_delta, gamma=gamma,
              alpha=alpha, beta=beta, population_size=5)
    error_plot = False
    live_error = False
    path_vis = False
    plot_ph_matrix = True
    path_visualiser = ph_mtrx_visualiser = error_visualiser = None
    if error_plot:
        sexify = False
        if not live_error:
            sexify = True
        error_visualiser = ErrorVis(interactive=live_error, offset=20,
                                    xlim=num_runs, log_scale=False,
                                    sexify=sexify)
        if live_error:
            error_visualiser.set_vis_title(aco.tsp_name + ": " + str(aco.path_length) + " cities\n"
                                           + str(aco.num_ants) + " ants, "
                                           + " tau delta: " + str(aco.tau_delta) + " pop-size: " + str(
                aco.population_size))
    if path_vis:
        path_visualiser = Path2DVis(xymin=-50, xymax=1100,
                                    num_runs=num_runs, offset=50, interactive=True, sleep_interval=0)
    if plot_ph_matrix:
        ph_mtrx_visualiser = Pheromone_Ways_2DVis(xymin=-50, xymax=1100,
                                                  num_runs=1, offset=50,
                                                  interactive=True, sexify=False,
                                                  sleep_interval=0)

    print(aco)
    output_file = "/home/tobias/Bilder/tsp/"
    output_file += "_".join(['paco','ants'+str(num_ants),'cities'+str(num_cities), str(num_runs)+'runs'])+".pdf"
    aco.run_paco(num_runs=num_runs, live_error=live_error, output_file=output_file,
                 path_visualiser= path_visualiser, ph_mtrx_visualiser = ph_mtrx_visualiser,
                 error_visualiser = error_visualiser)
