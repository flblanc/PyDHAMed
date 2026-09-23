import numpy as np
from numpy.testing import assert_array_equal

from pydhamed.determine_transition_counts import count_matrix, loop_traj_count_matrix


def test_count_matrix_default_n_states():
    # states 0..3 present -> matrix must be 4x4, not 3x3
    traj = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    b = count_matrix(traj, lag=1)
    assert b.shape == (4, 4)


def test_loop_traj_count_matrix_no_double_counting():
    traj_dict = {
        "0": np.array([0, 1, 0, 1, 0, 1]),
        "1": np.array([0, 1, 0, 1, 0, 1]),
    }
    n_states = 2

    individual = {k: count_matrix(v, n_states=n_states) for k, v in traj_dict.items()}
    expected = individual["0"] + individual["1"]

    comb = loop_traj_count_matrix(traj_dict, n_states=n_states, trj1_index="0")

    assert_array_equal(comb, expected)
    # original per-trajectory matrix for trj1_index must be untouched
    assert_array_equal(count_matrix(traj_dict["0"], n_states=n_states), individual["0"])
