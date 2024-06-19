import pytest

from aco.tau_matrix import TauMatrix
from aco.tsp_parser import load_tsp_problem


@pytest.fixture(scope="session")
def points():
    return load_tsp_problem("tsp_problems/rd100.tsp", 5)


@pytest.fixture(scope="session")
def tau_matrix(points):
    return TauMatrix(path_length=len(points), tau_zero=1)
