
import numpy as np
from multiprocessing import Queue

from aco.common_aco import Aco
from aco.constants import *
from aco.tsp_parser import *


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
                    ph_mtrx_visualiser.plot_ph_matrix_fn( iteration)  # incl. save_fig
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