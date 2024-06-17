from unittest.mock import MagicMock

import numpy as np
import pytest

from aco.common_aco import Aco


@pytest.fixture
def aco():
    return Aco(
        tau_zero=1,
        num_ants=20,
        tau=0.1,
        gamma=0.1,
        alpha=1,
        beta=5,
    )


def test_heuristic(aco, points):
    aco.points = points
    assert aco.heuristic(0, 1) == 5.324407902849726e-16


def test_tau(aco, points):
    aco.points = points
    aco.path_length = len(points)
    aco.make_tau_matrix()
    assert aco.tau(0, 4) == 1.0


def test_tau_times_heuristic(aco):
    aco.heuristic = MagicMock()
    aco.tau = MagicMock()
    aco.tau_times_heuristic(0, 9)
    aco.heuristic.assert_called_with(0, 9)
    aco.tau.assert_called_with(0, 9)


def test_shortest_path(aco, points):
    aco.points = points
    aco.path_length = len(points)
    aco.make_tau_matrix()
    np.random.seed(1)
    assert aco.shortest_path() == 11


def test_calc_length_of_path(aco, points):
    aco.points = points
    aco.path_length = len(points)
    aco.make_tau_matrix()
    ant = aco.colony[0]
    assert aco.calc_length_of_path(ant=ant) == 0.0


def test_update_tau_matrix(aco, points):
    aco.points = points
    aco.path_length = len(points)
    aco.make_tau_matrix()
    ant = aco.colony[0]
    aco.update_tau_matrix(ant=ant)
    assert (
        aco.tau_matrix
        == np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
    ).all()


def test_run_aco(aco, points):
    aco.shortest_path = MagicMock()
    aco.update_tau_matrix = MagicMock()
    num_runs = 5
    aco.run_aco(num_runs=num_runs, points=points)
    assert aco.shortest_path.call_count == num_runs
    assert aco.update_tau_matrix.call_count == num_runs
