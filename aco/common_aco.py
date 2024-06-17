from __future__ import division

import numpy as np
from scipy.spatial import distance

from aco.ant import Ant
from aco.constants import colon_line, half_space_line


class Aco:
    def __init__(
        self,
        num_ants=10,
        tau_zero=0.4,
        tau=0.5,
        gamma=0.8,
        alpha=1,
        beta=5,
        keep_paths=False,
    ):
        self.points = None
        self.tau_zero = tau_zero
        self.beta = beta
        self.alpha = alpha
        self.tau_delta = tau  # pheromone-value added if win
        self.gamma = gamma  # evaporation rate
        self.tau_matrix = None
        self.num_ants = num_ants
        self.colony = [Ant() for _ in range(num_ants)]

        self.plot_ph_matrix = False
        self.path_vis = False
        # todo rename
        self.path_length = None
        self.path_lengths = []
        self.keep_paths = keep_paths
        self.best_paths = []

    def make_tau_matrix(self):
        self.tau_matrix = np.zeros((self.path_length + 1, self.path_length + 1))
        self.tau_matrix.fill(self.tau_zero)
        np.fill_diagonal(self.tau_matrix, 0)

    def ant_path(self, ant: Ant):
        """
        gets coordinates of shortest paths and forwards it to the vis
        """
        x, y = [], []
        x1 = [sol + 1 for sol in ant.current_solution]
        x2 = np.roll(x1, 1)
        for i, j in zip(x1, x2):
            c1, c2 = self.points[i], self.points[j]
            x.extend([c1[0], c2[0]])
            y.extend([c1[1], c2[1]])

        return x, y

    def heuristic(self, i, j) -> float:
        h = distance.euclidean(self.points[i + 1], self.points[j + 1]) or 1
        return (1 / h) ** self.beta

    def tau(self, i, j):
        return self.tau_matrix[i, j] ** self.alpha

    def tau_times_heuristic(self, i, j):
        return self.heuristic(i, j) * self.tau(i, j)

    def shortest_path(self):
        """
        find new paths for each ant in colony
        """
        ant_solutions = np.zeros(self.num_ants)

        for a, ant in enumerate(self.colony):
            # start at random city
            start_i = np.random.randint(self.path_length)
            ant.current_solution = [start_i]
            ant.has_seen_cities.append(start_i)

            for step in range(1, self.path_length):
                # sum up all tau
                pij = np.zeros(len(self.points))

                for i, city in enumerate(self.points):
                    # cities start with 1
                    if city - 1 not in ant.has_seen_cities:
                        pij[i] = self.tau_times_heuristic(start_i, city - 1)

                # highest probability wins.
                next_city = np.argmax(pij / np.sum(pij))

                ant.current_solution.append(next_city)
                ant.has_seen_cities.append(next_city)
                start_i = next_city  # continue with this city

            ant_solutions[a] = self.calc_length_of_path(ant)
            ant.has_seen_cities = []
        return np.argmin(ant_solutions)

    def calc_length_of_path(self, ant):
        ant.length_of_path = 0.0
        # shift array and compute the distances.
        for i0, i1 in zip(ant.current_solution, np.roll(ant.current_solution, 1)):
            ant.length_of_path += distance.euclidean(
                self.points[i0 + 1], self.points[i1 + 1]
            )
        return ant.length_of_path

    def update_tau_matrix(self, ant):
        """
        best ant is allowed to update
        """
        for cs, ns in zip(ant.current_solution, np.roll(ant.current_solution, 1)):
            self.tau_matrix[cs, ns] += self.tau_delta

    def run_aco(self, num_runs=10, points=None):
        """
        Implementation of the ACO with the possibility to visualise the shortest path
        and the pheromone-matrix. This is done in the following way:
        each ant guesses a paths through all points/cities according to a heuristic
        and the current pheromone value on the connection between the corresponding
        points. Meaning: The more pheromone and the closer, the higher the probability
        of choosing the city as next position in the path. After an iteration, the lengths
        of the paths are compared and the connections between the cities of the winning ant
        get all of the pheromone (could be done otherwise). Also, each connection is decreased
        by (1-gamma).
        """
        self.points = points
        self.path_length = len(self.points)
        self.make_tau_matrix()
        for i in range(num_runs):
            self.tau_matrix *= 1 - self.gamma  # evaporation step before updating.
            best_ant_index = self.shortest_path()
            best_ant = self.colony[best_ant_index]
            self.update_tau_matrix(best_ant)
            if self.keep_paths:
                x, y = self.ant_path(best_ant)
                # breakpoint()

                self.best_paths.append((x, y))
            self.path_lengths.append(best_ant.length_of_path)
            print(self.path_lengths[-1])

    def __str__(self):
        res = ""
        res += colon_line + "\n" + half_space_line + "ACO\n" + colon_line + "\n"
        res += f"ant number {self.num_ants} \n"
        res += f"tau {self.tau_delta} \n"
        res += f"gamma {self.gamma} \n"
        return res
