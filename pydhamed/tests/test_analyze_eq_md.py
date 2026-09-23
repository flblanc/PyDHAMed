import numpy as np

from pydhamed.analyze_eq_md import pop_from_tba_eq_traj, block_average_pop_eq_tba


def test_pop_from_tba_eq_traj_runs_on_python3():
    # regression test for Counter(...).iteritems(), removed in Python 3
    tba = np.array([1, 1, 2, 2, 2, 3])
    p_ar = pop_from_tba_eq_traj(tba, n_states=3)
    assert np.isclose(p_ar.sum(), 1.0)


def test_block_average_pop_eq_tba_runs_on_python3():
    tba = np.array([1, 1, 2, 2, 2, 3, 1, 2, 3, 3])
    pop_bl_ar = block_average_pop_eq_tba(tba, n_blocks=2, n_states=3)
    assert pop_bl_ar.shape == (3, 2)
