from collections import Counter
import numpy as np


def pop_from_tba_eq_traj(tba, verbose=False, n_states=32):
    # NOTE: tba is indexed 1-based here (p_ar[s-1]), unlike the 0-based
    # state convention used by determine_transition_counts.count_matrix.
    # Confirm this is intentional for the input format before relying on it.
    p_ar = np.zeros(n_states)
    traj_time = len(tba) * 1.0
    for s,c in Counter(tba).items():
        if verbose:
            print(s, c)
        p_i = c/ float(traj_time)
        p_ar[s-1] = p_i
    return p_ar


def block_average_pop_eq_tba(tba, n_blocks, n_states=32):
    tba_bl = np.split(tba, n_blocks)
    pop_bl_ar = np.zeros((n_states, n_blocks))

    for bi, b in enumerate(tba_bl):
        pop_bl_ar[:,bi] = pop_from_tba_eq_traj(b, n_states=n_states)
    return pop_bl_ar
