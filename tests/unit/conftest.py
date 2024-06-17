import pytest

from aco.tsp_parser import load_tsp_problem


@pytest.fixture(scope="session")
def points():
    return load_tsp_problem("tsp_problems/rd100.tsp", 5)
