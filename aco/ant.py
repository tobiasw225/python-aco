import sys


class Ant:
    def __init__(self):
        self.current_solution = []
        self.has_seen_cities = []
        self.length_of_path = sys.maxsize

    def refresh_cities(self):
        self.has_seen_cities = []

