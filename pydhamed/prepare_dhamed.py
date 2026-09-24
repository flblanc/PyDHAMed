import numpy as np
from collections import defaultdict
from dataclasses import dataclass


def state_lifetimes_counts(transition_count_matrix_l):
    """

    Calculate lifetimes in each of the states (for each run/window)

    Parameters:
    -----------
    transition_count_matrix_l: list of arrays
        List of arrays with transition count matrices. One array for
        each run/windows.

    Returns:
    --------
    t_ar: array_like
        Array, n x nwin, where n is number of states,
        nwin is number of windows with aggregate lifetimes in the states.

    """
    # sum over the column gives all counts in a state
    return np.stack(transition_count_matrix_l, axis=-1).sum(axis=0)


def counts_in_out(transition_count_matrix_l, n):
    """
    Parameters:
    -----------
    transition_count_matrix_l: list of arrays
        List of arrays with transition count matrices. One array for
        each run/windows.
    n: int
        Number of (structural) states

    Returns:
    --------
    n_in: array_like
        Array length n. Total number of transitions into state i.
    n_out: array_like
        Array length n. Total number of transitions out of state j.

    """
    n_in = np.zeros(n)
    n_out = np.zeros(n)

    for count_matrix in transition_count_matrix_l:
        diag = np.diag(count_matrix)
        # row/column sums minus the diagonal exclude the i == j terms
        n_in += count_matrix.sum(axis=1) - diag
        n_out += count_matrix.sum(axis=0) - diag
    return n_in, n_out


def eligible_states(n_in, n_out):
    """
    States with at least one observed transition into and out of them,
    the minimum requirement for an equilibrium to be established for
    that state.

    Parameters:
    -----------
    n_in: array
        Number of transitions into given states.
    n_out: array
        Number of transitions from given states.

    Returns:
    --------
    eligible: array of bool
        True for states with both n_in > 0 and n_out > 0.

    """
    return (n_in > 0.0) & (n_out > 0.0)


def find_transition_pairs(transition_count_matrix_l, eligible, t_ar):
    """
    Enumerate the (window, i, j) triples of distinct eligible states that
    are actually connected: at least one transition between them and a
    nonzero combined residence time, in a given run/window.

    This is the single enumeration shared by the state-pairing bookkeeping
    (check_transition_pairs) and by the DHAMed input assembly
    (prepare_dhamed_input_pairs), so the two agree by construction on
    which pairs are used instead of each re-deriving the same criteria.

    Parameters:
    -----------
    transition_count_matrix_l: list of arrays
        List of arrays with transition count matrices. One array for
        each run/window.
    eligible: array of bool
        Per-state mask of states allowed to take part in a pair (see
        eligible_states).
    t_ar: array_like
        n x nwin array of aggregate lifetimes in the states.

    Returns:
    --------
    pairs: list of (iwin, i, j) tuples
        Indices (into transition_count_matrix_l/t_ar) of each connected
        pair of states, with i < j.

    """
    n_states = len(eligible)
    pairs = []
    for iwin, count_matrix in enumerate(transition_count_matrix_l):
        for i in range(n_states - 1):
            if not eligible[i]:
                continue
            for j in range(i + 1, n_states):
                if not eligible[j]:
                    continue
                if count_matrix[i, j] + count_matrix[j, i] <= 0.0:
                    continue
                if t_ar[i, iwin] + t_ar[j, iwin] <= 0.0:
                    continue
                pairs.append((iwin, i, j))
    return pairs


def check_transition_pairs(pairs, n_states):
    """
    Count, for each state, in how many (window, partner) pairs it appears.
    Unpaired states are subsequently excluded from the analysis since no
    proper equilibrium can be established for them.

    Parameters:
    -----------
    pairs: list of (iwin, i, j) tuples
        Connected pairs, as returned by find_transition_pairs.
    n_states: integer

    Returns:
    --------
    paired_ar: array
        Number of transition pairs for each state.

    """
    paired_ar = np.zeros(n_states)
    for _, i, j in pairs:
        paired_ar[i] += 1
        paired_ar[j] += 1
    return paired_ar


def actual_transition_pairs(eligible, paired_ar, n_states, verbose=False):
    """
    Generate indeces of transition pairs. The indices of transition pairs are
    required to setup the input for the actual optimization of the DHAMed
    effective likelihood.

    Parameters:
    -----------
    eligible: array of bool
        Per-state mask of states with at least one transition in and out
        (see eligible_states).
    paired_ar: array
        Number of transition pairs for each state.
    n_states: int
    verbose: Boolean

    Returns:
    --------
    pair_idx_d: defaultdict
        Indeces of the paired states. Shifts the indices to account for
        excluded, unpaired states.
    n_actual: int
        Actual number of states included in the DHAMed calculation.

    """
    n_actual = 0
    pair_idx_d = defaultdict(list)
    # index of included states/bins
    for i in range(n_states):
        if eligible[i] and paired_ar[i] > 0:
            pair_idx_d[i].append(n_actual)
            n_actual += 1
        else:
            print("bin {} excluded".format(i))
    if verbose:
        print(n_actual)
    return pair_idx_d, n_actual


def prepare_dhamed_input_pairs(transition_count_matrix_l, pairs, t_ar, pair_idx_d, v_ar):
    """
    Prepare formatted inputs for DHAMed optimization.

    Parameters:
    -----------
    transition_count_matrix_l: list of arrays
        List of transition count matrices
    pairs: list of (iwin, i, j) tuples
        Connected pairs, as returned by find_transition_pairs. Every state
        appearing here is guaranteed to already have an entry in
        pair_idx_d, since find_transition_pairs was built from the same
        eligible-states mask used to construct pair_idx_d.
    t_ar: array_like
        n x nwin, where n is number of states, nwin is number of windows.
    pair_idx_d: dictionary
        Indeces of the paired states. Shifts the indices to account for
        excluded, unpaired states.
    v_ar: array
        Bias potentials in units of kT.

    Returns:
    --------
    ip: array_like
        npair entries, list of indices of bin i in transition pair.
    jp: array_like
        npair entries, list of indices of bin j in transition pair.
    vi: array_like
        npair entries, list of potentials in kT units at bin i of a pair.
    vj: array_like
        npair entries, list of potentials in kT units at bin j of a pair.
    ti: array_like
        npair entries, list of residence times in bin i of a pair.
    tj: array like
        npair entries, list of residence times in bin j of a pair.
    nijp: array_like
        npair entries, number of j->i and i->j transitions combined for a pair.

    """
    ip_l = []
    jp_l = []
    vi = []
    vj = []
    ti = []
    tj = []
    nijp = []

    for iwin, i, j in pairs:
        count_matrix = transition_count_matrix_l[iwin]
        ip_l.append(pair_idx_d[i][0])
        jp_l.append(pair_idx_d[j][0])
        vi.append(v_ar[i, iwin])
        vj.append(v_ar[j, iwin])
        ti.append(t_ar[i, iwin])
        tj.append(t_ar[j, iwin])
        nijp.append(count_matrix[i, j] + count_matrix[j, i])

    print("Number of transition pairs {}".format(len(pairs)))
    return (np.array(ip_l, dtype=int), np.array(jp_l, dtype=int),
            np.array(vi), np.array(vj), np.array(ti), np.array(tj), np.array(nijp))


def check_total_transition_counts(n_out, eligible, paired_ar, n_actual):
    """
    Remove excluded states/bin from the array with the total number of transitions
    out of state/bin i.

    Parameters:
    -----------
    n_out: array_like
        Total number of transitions out of state/bin i, with length N, the numer of states.
    eligible: array of bool
        Per-state mask of states with at least one transition in and out
        (see eligible_states).
    paired_ar: array_like
        Number of transition pairs for each state, with length N, the number of states.
    n_actual: int
        Actual number of connected states which can be analyzed.

    Returns:
    --------
    n_k: array_like
        Total number of transitions out of state/bin i. Excluding states for which
        no proper equilibrium can be established. The array has the length N_actual.
    """
    n_k = np.zeros(n_actual)
    c = 0
    for i in range(len(n_out)):
        if eligible[i] and paired_ar[i] > 0:
            n_k[c] = n_out[i]
            c += 1
    return n_k


@dataclass
class DhamedPairData:
    """
    Per-transition-pair inputs to the DHAMed effective likelihood, plus the
    total per-state transition counts. ip/jp/ti/tj/vi/vj/nijp each have one
    entry per transition pair; nk has one entry per included state.
    """
    ip: np.ndarray
    jp: np.ndarray
    ti: np.ndarray
    tj: np.ndarray
    vi: np.ndarray
    vj: np.ndarray
    nk: np.ndarray
    nijp: np.ndarray


def generate_dhamed_input(c_l, v_ar, n_states, return_included_state_indices=False):
    """
    Converts a list of count matrices and an array of bias potentials
    to the input for DHAMed. For efficient calculation DHAMed input data
    is organized into transition pairs.

    Parameters:
    -----------
    c_l: list,
        List of arrays. Each array contains a transition count matrix.
    v_ar: array
        Array of bias potentials
    n_states: int
        Number of states/bins.
    return_included_state_indices: boolean, optional
        Also return the mapping from original state index to the index
        used in the DHAMed calculation.

    Returns:
    --------
    data: DhamedPairData
        Per-pair inputs and per-state transition counts for the DHAMed
        effective likelihood.
    pair_idx_d: dictionary, optional
        Indices of the states included in the DHAMed calculation, keyed
        by their original state index (only returned if
        return_included_state_indices is True).

    """
    t = state_lifetimes_counts(c_l)
    n_in, n_out = counts_in_out(c_l, n_states)
    eligible = eligible_states(n_in, n_out)

    pairs = find_transition_pairs(c_l, eligible, t)
    paired_ar = check_transition_pairs(pairs, n_states)
    pair_idx_d, n_actual = actual_transition_pairs(eligible, paired_ar, n_states)
    ip, jp, vi, vj, ti, tj, nijp = prepare_dhamed_input_pairs(c_l, pairs, t, pair_idx_d, v_ar)
    # Remove excluded counts from the total number of transitions out of state.
    nk = check_total_transition_counts(n_out, eligible, paired_ar, n_actual)

    data = DhamedPairData(ip=ip, jp=jp, ti=ti, tj=tj, vi=vi, vj=vj, nk=nk, nijp=nijp)

    if return_included_state_indices:
        return data, pair_idx_d
    return data
