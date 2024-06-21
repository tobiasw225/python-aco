from __future__ import division

import numpy as np
from scipy.spatial import distance

from aco.ant import Ant
from aco.constants import colon_line, half_space_line
from aco.tau_matrix import TauMatrix


class Aco:
    def __init__(
        self,
        tau_matrix: TauMatrix,
        points: dict,
        num_ants=10,
        tau=0.5,
        gamma=0.8,
        alpha=1,
        beta=5,
    ):
        self.tau_matrix = tau_matrix
        self.points = points
        self.beta = beta
        self.alpha = alpha
        self.tau_delta = tau  # pheromone-value added if win
        self.gamma = gamma  # evaporation rate
        self.num_ants = num_ants
        self.colony = [Ant() for _ in range(num_ants)]
        # todo rename
        self.path_length = len(self.points)
        self.path_lengths = []
        self.best_paths = []

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
        return self.tau_matrix.get(i, j) ** self.alpha

    def tau_times_heuristic(self, i, j):
        return self.heuristic(i, j) * self.tau(i, j)

    def _sum_up_all_tau(self, ant: Ant, start_index: int):
        # calculate the tau value for all points
        # which the ant has not seen yet.
        pij = np.zeros(len(self.points))
        for i, city in enumerate(self.points):
            # cities start with 1
            current_city = city - 1
            if current_city not in ant.has_seen_cities_in_iteration:
                pij[i] = self.tau_times_heuristic(i=start_index, j=current_city)
        return pij

    def _step(self, ant: Ant, start_index: int):
        # the ant walks one step further
        # depending on the highest probability.
        pij = self._sum_up_all_tau(ant=ant, start_index=start_index)
        next_city = np.argmax(pij / np.sum(pij))
        ant.current_solution.append(next_city)
        ant.has_seen_cities_in_iteration.append(next_city)
        return next_city  # continue with this city

    def fastest_ant(self):
        """
        find new paths for each ant in colony

        :return: index of fastest ant
        """
        ant_solutions = []

        for ant in self.colony:
            # start at random city
            start_index = np.random.randint(self.path_length)
            ant.current_solution = [start_index]
            ant.has_seen_cities_in_iteration.append(start_index)

            for step in range(1, self.path_length):
                start_index = self._step(ant=ant, start_index=start_index)

            ant_solutions.append(self.calc_length_of_path(ant))
            ant.has_seen_cities_in_iteration = []
        return self.colony[np.argmin(ant_solutions)]

    def calc_length_of_path(self, ant: Ant) -> float:
        ant.length_of_path = 0.0
        # shift array and compute the distances.
        for i0, i1 in zip(ant.current_solution, np.roll(ant.current_solution, 1)):
            ant.length_of_path += distance.euclidean(
                self.points[i0 + 1], self.points[i1 + 1]
            )
        return ant.length_of_path

    def update_tau_matrix(self, ant: Ant) -> None:
        """
        best ant is allowed to update
        """
        for cs, ns in zip(ant.current_solution, np.roll(ant.current_solution, 1)):
            self.tau_matrix.update(i=cs, j=ns, delta=self.tau_delta)

    def evaporate(self):
        self.tau_matrix.tau_matrix *= 1 - self.gamma

    def run(self, num_runs: int = 0) -> None:
        """
        Implementation of the ACO with the possibility to visualise the shortest path
        and the pheromone-matrix. This is done in the following way:
        each ant guesses a paths through all points/cities according to a heuristic
        and the current pheromone value on the connection between the corresponding
        points. Meaning: The more pheromone and the closer, the higher the probability
        of choosing the city as next position in the path. After an iteration, the lengths
        of the paths are compared and the connections between the cities of the winning ant
        get all the pheromone (could be done otherwise). Also, each connection is decreased
        by (1-gamma).
        """
        for i in range(num_runs):
            self.evaporate()
            best_ant = self.fastest_ant()
            self.update_tau_matrix(ant=best_ant)
            self.save_solutions(ant=best_ant)

    def save_solutions(self, ant: Ant):
        self.best_paths.append(self.ant_path(ant=ant))
        self.path_lengths.append(ant.length_of_path)
        print(self.path_lengths[-1])

    def __str__(self) -> str:
        res = ""
        res += colon_line + "\n" + half_space_line + "ACO\n" + colon_line + "\n"
        res += f"ant number {self.num_ants} \n"
        res += f"tau {self.tau_delta} \n"
        res += f"gamma {self.gamma} \n"
        return res
