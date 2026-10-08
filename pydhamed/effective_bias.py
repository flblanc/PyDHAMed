"""Effective bias factors of states when the bias is not constant within them (DHAMed paper, eqs 36-39).

e^{-u_i^a} = < e^{-u^a(x)} >_i, the average over the unbiased distribution within state i of the bias of run a
(in kT), estimated by binless WHAM restricted to the configurations of state i (eqs 37-39). Binless WHAM is MBAR,
so each state is solved with pymbar: the runs are the sampled thermodynamic states, restricted to state i, and an
unsampled zero-bias state gives the unbiased reference. This covers e.g. macrostates in umbrella sampling and
temperature replica exchange, where u^a(x) = (beta_a - beta) U(x) is not constant within a conformational state.

L. S. Stelzl, A. Kells, E. Rosta, G. Hummer, J. Chem. Theory Comput. 13, 6328 (2017);
E. Rosta, M. Nowotny, W. Yang, G. Hummer, J. Am. Chem. Soc. 133, 8934 (2011) (binless WHAM);
M. R. Shirts, J. D. Chodera, J. Chem. Phys. 129, 124105 (2008) (MBAR).
"""

import numpy as np


def effective_bias(u_kn, state_n, run_n, n_states=None, return_frame_weights=False, initial_f_k=None,
                   max_iterations=1000, **mbar_kwargs):
    """
    Effective bias factors u_i^a (kT) of every state i in every run a.

    Parameters:
    -----------
    u_kn: array (n_runs, n_frames), reduced bias energy u^a(x_n) of every frame n (from any run) in every run a,
          in kT: the run's reduced potential minus that of the unbiased reference ensemble.
    state_n: array (n_frames,), state index of every frame (negative: frame not assigned, e.g. between core sets).
    run_n: array (n_frames,), index of the run that sampled every frame.
    n_states: number of states (default: max(state_n) + 1).
    return_frame_weights: also return, for every frame, its log-weight in the unbiased distribution within its
        own state (normalised to sum to one within each state; -inf for unassigned frames). Multiplied by the
        DHAMed populations p_i, these give unbiased frame weights for observables other than the states.
    initial_f_k: optional free energies (kT) of the runs from a global MBAR, used as starting point for every
        state. Otherwise each state starts from the solution of the previous one (the per-state free energies of
        the runs differ little between states), which makes large problems (e.g. 64 temperatures) much faster.
    max_iterations: iteration cap of pymbar's adaptive solver for each state.
    mbar_kwargs: passed to pymbar.MBAR (e.g. a different solver_protocol).

    Returns:
    --------
    bias_ar: array (n_states, n_runs), the DHAMed bias array (NaN for states without frames);
    (log_w_n: array (n_frames,), if return_frame_weights).
    """
    try:
        from pymbar import MBAR
    except ImportError as error:
        raise ImportError("effective_bias needs pymbar >= 4: pip install PyDHAMed[mbar]") from error
    from scipy.special import logsumexp

    u_kn = np.asarray(u_kn, dtype=float)
    state_n = np.asarray(state_n, dtype=int)
    run_n = np.asarray(run_n, dtype=int)
    n_runs = u_kn.shape[0]
    n_states = int(state_n.max()) + 1 if n_states is None else n_states
    bias_ar = np.full((n_states, n_runs), np.nan)
    log_w_n = np.full(len(state_n), -np.inf)
    mbar_kwargs.setdefault("verbose", False)
    mbar_kwargs.setdefault("solver_protocol", [{"method": "adaptive",
                                                "options": {"maximum_iterations": max_iterations, "min_sc_iter": 0}}])
    previous = None if initial_f_k is None else np.append(np.asarray(initial_f_k, dtype=float), 0.0)
    for i in range(n_states):
        sel = np.flatnonzero(state_n == i)
        if sel.size == 0:
            continue
        N_k = np.append(np.bincount(run_n[sel], minlength=n_runs), 0)       # + unsampled zero-bias state
        u = np.vstack([u_kn[:, sel], np.zeros(sel.size)])
        sampled = N_k > 0
        if sampled.sum() == 1:
            # One run only: f_a - f_0 = -ln < e^{-(u_a - u_s)} > in the sampled run s, relative to the zero bias.
            s = np.flatnonzero(sampled)[0]
            f = -logsumexp(-(u - u[s]), axis=1) + np.log(sel.size)
            log_w = -u[-1] + u[s]
        else:
            # Shift every state's energies by their minimum (a constant per state, absorbed in f) so that large
            # reduced energies (temperatures) do not stall pymbar's solvers.
            shifts = u.min(axis=1)
            start = None
            if previous is not None:
                start = previous - shifts
                start = start - start[0]
            mbar = MBAR(u - shifts[:, None], N_k, initial_f_k=start, **mbar_kwargs)
            f = np.asarray(mbar.f_k) + shifts
            if initial_f_k is None:
                previous = f - f[0]
            log_w = -u[-1] - logsumexp(np.log(N_k[sampled])[:, None] + f[sampled, None] - u[sampled], axis=0)
        bias_ar[i] = f[:-1] - f[-1]
        log_w_n[sel] = log_w - logsumexp(log_w)
    return (bias_ar, log_w_n) if return_frame_weights else bias_ar
