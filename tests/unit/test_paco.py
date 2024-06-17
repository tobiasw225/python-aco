from unittest.mock import MagicMock

import pytest

from aco.paco import Paco


@pytest.fixture
def paco():
    return Paco(
        tau_zero=1,
        num_ants=20,
        tau=0.1,
        gamma=0.1,
        alpha=1,
        beta=5,
        population_size=5,
    )


def test_add_solution(paco):
    paco.add_pheromone = MagicMock()
    paco.remove_pheromone = MagicMock()
    for _ in range(paco.population_size + 2):
        paco.add_solution(
            solution=[
                12,
                14,
                0,
                7,
                17,
                13,
                11,
                3,
                2,
                9,
                5,
                15,
                10,
                4,
                6,
                1,
                18,
                16,
                19,
                8,
            ]
        )
    assert paco.add_pheromone.call_count == paco.population_size + 2
    assert paco.remove_pheromone.call_count == 2
    assert paco.fifo_solution_q.qsize() == paco.population_size


def test_run_paco(paco, points):
    paco.make_tau_matrix = MagicMock()
    paco.add_solution = MagicMock()
    paco.shortest_path = MagicMock()
    num_runs = 5
    paco.run_paco(num_runs=num_runs, points=points)
    paco.make_tau_matrix.assert_called_once()
    assert paco.add_solution.call_count == num_runs
    assert paco.shortest_path.call_count == num_runs
