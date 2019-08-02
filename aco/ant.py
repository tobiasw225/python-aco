import numpy as np

class Ant:
    def __init__(self, path_length):
        """

        :param path_length:
        """
        self.path_length = path_length
        self.current_solution = []
        self.has_seen_cities = []

    def refresh_cities(self):
        self.has_seen_cities = []

    def start_city(self):
        """

        :return:
        """
        return np.random.randint(low=0, high=self.path_length)
