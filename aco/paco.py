from multiprocessing import Queue

import numpy as np

from aco.common_aco import Aco
from aco.constants import colon_line, half_space_line


class Paco(Aco):
    def __init__(
        self,
        num_ants=10,
        tau_zero=0.4,
        tau=0.5,
        gamma=0.0,
        alpha=1,
        beta=5,
        population_size=5,
    ):
        super().__init__(
            num_ants=num_ants,
            tau_zero=tau_zero,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
        )
        self.fifo_solution_q = Queue(maxsize=population_size)
        self.population_size = population_size

    def add_solution(self, solution):
        if self.fifo_solution_q.full():
            self.remove_pheromone(self.fifo_solution_q.get())
        self.add_pheromone(solution)
        self.fifo_solution_q.put(solution)

    def add_pheromone(self, solution):
        for i0, i1 in zip(solution, np.roll(solution, 1)):
            self.tau_matrix[i0, i1] += self.tau_delta

    def remove_pheromone(self, solution):
        for i0, i1 in zip(solution, np.roll(solution, 1)):
            self.tau_matrix[i0, i1] -= self.tau_delta

    def run_paco(self, num_runs=50, points=None):
        """
        Simple implementation of the P-ACO algorithm. This is similar to ACO, but there
        is no evaporation step (for all ants).
         In this case a population of solution influences the choice
        of the ants. After 'population_size' steps, the solution looses it's impact and the
        corresponding pheromone value is removed from the pheromone matrix.
        """
        self.points = points
        self.path_length = len(self.points)
        self.make_tau_matrix()
        for i in range(num_runs):
            best_ant_index = self.shortest_path()
            best_ant = self.colony[best_ant_index]
            self.add_solution(best_ant.current_solution)
            # todo extract me
            # """
            #     Visualisation & current status of algorithm
            # """
            #
            # if ph_mtrx_visualiser:
            #     ph_mtrx_visualiser.plot_ph_matrix_fn(i)
            # if path_visualiser:
            #     x, y = self.get_best_ant_path(best_ant)
            #     path_visualiser.plot_path(x, y)

            self.path_lengths.append(best_ant.length_of_path)
            print(self.path_lengths[-1])

    def __str__(self):
        res = ""
        res += colon_line + "\n" + half_space_line + "P-ACO\n" + colon_line + "\n"
        res += f"ant number {self.num_ants} \n"
        res += f"tau {self.tau_delta} \n"
        res += f"pop-size {self.population_size}\n"
        return res
