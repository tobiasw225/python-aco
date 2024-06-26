from multiprocessing import Queue

import numpy as np

from aco.common_aco import Aco
from aco.constants import colon_line, half_space_line
from aco.tau_matrix import TauMatrix


class Paco(Aco):
    def __init__(
        self,
        tau_matrix: TauMatrix,
        points: dict,
        num_ants: int,
        tau: float,
        alpha: int,
        beta: int,
        population_size: int,
    ):
        super().__init__(
            tau_matrix=tau_matrix,
            points=points,
            num_ants=num_ants,
            tau=tau,
            alpha=alpha,
            beta=beta,
        )
        self.fifo_solution_q = Queue(maxsize=population_size)
        self.population_size = population_size

    def add_solution(self, solution: list):
        if self.fifo_solution_q.full():
            self.remove_pheromone(self.fifo_solution_q.get())
        self.add_pheromone(solution)
        self.fifo_solution_q.put(solution)

    def add_pheromone(self, solution: list):
        for i0, i1 in zip(solution, np.roll(solution, 1)):
            self.tau_matrix.update(i=i0, j=i1, delta=self.tau_delta)

    def remove_pheromone(self, solution: list):
        for i0, i1 in zip(solution, np.roll(solution, 1)):
            self.tau_matrix.update(i=i0, j=i1, delta=-self.tau_delta)

    def run(self, num_runs: int = 0) -> None:
        """
        Simple implementation of the P-ACO algorithm. This is similar to ACO, but there
        is no evaporation step (for all ants).
        In this case a population of solution influences the choice
        of the ants. After 'population_size' steps, the solution looses its impact and the
        corresponding pheromone value is removed from the pheromone matrix.
        """
        for i in range(num_runs):
            best_ant = self.fastest_ant()
            self.add_solution(best_ant.current_solution)
            self.save_solutions(ant=best_ant)

    def __str__(self):
        res = ""
        res += colon_line + "\n" + half_space_line + "P-ACO\n" + colon_line + "\n"
        res += f"ant number {self.num_ants} \n"
        res += f"tau {self.tau_delta} \n"
        res += f"pop-size {self.population_size}\n"
        return res
