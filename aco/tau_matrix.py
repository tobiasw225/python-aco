from __future__ import division

import numpy as np


class TauMatrix:
    def __init__(self, path_length, tau_zero):
        self.tau_matrix = np.zeros((path_length + 1, path_length + 1))
        self.tau_matrix.fill(tau_zero)
        np.fill_diagonal(self.tau_matrix, 0)

    def get(self, i, j):
        return self.tau_matrix[i, j]

    def update(self, i, j, delta):
        self.tau_matrix[i, j] += delta
