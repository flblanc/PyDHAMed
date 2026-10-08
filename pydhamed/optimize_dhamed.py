import dataclasses
import logging
import time

import numpy as np
import numba
from scipy.optimize import minimize

from .prepare_dhamed import generate_dhamed_input
from .result import DhamedResult

logger = logging.getLogger("pydhamed")


@numba.jit(nopython=True)
def effective_log_likelihood_count_list(g,  ip, jp, ti, tj, vi, vj, nk, nijp,
                                        jit_gradient=False):
    """
    Effective negative log-likelihood for sovling the DHAMed equations by numerical 
    optimization. 
    
    Parameters:
    -----------
    g: array_like
        Dimensionless free energy of the states, with gi = ln pi = -beta Gi
    ip: array_like
        npair entries, list of indices of bin i in transition pair.
    jp: array_like
        npair entries, list of indices of bin j in transition pair.
    ti: array_like
        npair entries, list of residence times in bin i of a pair.
    tj: array_like
        npair entries, list of residence times in bin j of a pair.
    vi: array_like
        npair entries, list of potentials in kT units at bin i of a pair.  
    vj: array_like
        npair entries, list of potentials in kT units at bin j of a pair.
    nk: array_like
        Total number of transitions out bin i.
    nijp: array_like
        npair entries, number of j->i and i->j transitions combined for a pair.        
    jit_gradient: Boolean, optional
        Use Numba to speed up the calculation.
        
    Returns:
    --------
    F: float
        Effective negative log-likelihood for DHAMed.
    """
    xlogp = 0
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        _vi = vi[ipair]
        _vj = vj[ipair]
        w = 0.5 * (_vi - g[i] + _vj - g[j])
        taui = ti[ipair]*np.exp(_vi-g[i] -w)
        tauj = tj[ipair]*np.exp(_vj-g[j] -w)
        xlogp += nijp[ipair]* (np.log(taui+tauj)+w)

    return xlogp + np.sum(nk*g)


# Pure-Python fallback (no numba dispatch overhead / no numba dependency at
# call time): the undecorated function underlying the jitted version above,
# exposed by numba rather than kept as a hand-maintained duplicate.
effective_log_likelihood_count_ref = effective_log_likelihood_count_list.py_func


@numba.jit(nopython=True)
def _loop_grad_dhamed_likelihood_0(grad, g,  ip, jp, ti, tj, vi, vj, nijp):
    """
    Shared inner loop accumulating the pairwise contributions to the gradient
    of the effective log-likelihood into `grad`:
    -nijp * a_i / (a_i + a_j) for state i (and symmetrically for j), with a_i = ti exp(vi - g_i),
    evaluated as a logistic function of ln a_j - ln a_i so that large biases do not overflow.
    """
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        n = nijp[ipair]
        if ti[ipair] > 0 and tj[ipair] > 0:
            d = np.log(tj[ipair]) + vj[ipair] - g[j] - np.log(ti[ipair]) - vi[ipair] + g[i]
            if d > 0:
                e = np.exp(-d)
                grad[i] += -n * e / (1.0 + e)
                grad[j] += -n / (1.0 + e)
            else:
                e = np.exp(d)
                grad[i] += -n / (1.0 + e)
                grad[j] += -n * e / (1.0 + e)
        elif ti[ipair] > 0:
            grad[i] += -n
        elif tj[ipair] > 0:
            grad[j] += -n
    return grad


# Pure-Python fallback: the undecorated function underlying the jitted version
# above, exposed by numba rather than kept as a hand-maintained duplicate.
_loop_grad_dhamed_likelihood_0_ref = _loop_grad_dhamed_likelihood_0.py_func


@numba.jit(nopython=True)
def grad_dhamed_likelihood(g,  ip, jp, ti, tj, vi, vj, nk, nijp):
    grad = np.zeros(g.shape)
    grad += nk
    return _loop_grad_dhamed_likelihood_0(grad, g, ip, jp, ti, tj, vi, vj, nijp)


grad_dhamed_likelihood_ref = grad_dhamed_likelihood.py_func


def wrapper_ll(g_prime, data, jit_gradient=False):
    """
    Adding the extra zero when minimizing N-1 relative weights.

    Parameters:
    -----------
    data: DhamedPairData
        Per-pair inputs and per-state transition counts (see
        generate_dhamed_input).
    """
    g_i = np.append(g_prime, [0], axis=0)
    return effective_log_likelihood_count_list(g_i, data.ip, data.jp, data.ti, data.tj,
                                               data.vi, data.vj, data.nk, data.nijp)


def grad_dhamed_likelihood_ref_0(g_prime, data, jit_gradient=False):
    g = np.append(g_prime, [0], axis=0)
    grad = np.zeros(g.shape[0] )
    grad[:-1]  += data.nk[:-1]
    loop = _loop_grad_dhamed_likelihood_0 if jit_gradient else _loop_grad_dhamed_likelihood_0_ref
    grad = loop(grad, g, data.ip, data.jp, data.ti, data.tj, data.vi, data.vj, data.nijp)
    return grad[:-1]


def solve_dhamed(count_list, bias_ar, g_init=None, numerical_gradients=False, jit_gradient=False, **kwargs):
    """
    Solve the DHAMed equations (Stelzl, Kells, Rosta, Hummer, JCTC 13, 6328 (2017), eq 12) and return a
    DhamedResult with the populations, free energies, their statistical errors (eqs 14-18) and rates (eq 26).

    Parameters:
    -----------
    count_list: list of arrays, NxN transition counts for each simulation (window), C[i, j] = number of j -> i
                transitions within the lag time; the diagonal holds the counts of staying in a state.
    bias_ar: array, (N x nwin) bias acting on each state in each simulation, in units of kT
             (see effective_bias for biases that are not constant within states).
    g_init: initial log-weights of all N states (excluded states are dropped).
    Other keyword arguments are passed as options to scipy.optimize.minimize(method="BFGS").
    """
    n_states = count_list[0].shape[0]
    bias_ar = np.asarray(bias_ar, dtype=float)
    data, state_index = generate_dhamed_input(count_list, bias_ar, n_states, return_included_state_indices=True)
    included = np.zeros(n_states, dtype=bool)
    included[list(state_index)] = True
    g0 = np.zeros(len(data.nk)) if g_init is None else np.asarray(g_init, dtype=float)[included]

    # A constant added to all biases of a run shifts F by a constant: subtract each run's smallest bias among the
    # states it visited, which keeps F well scaled for large biases (e.g. temperatures). The result keeps the
    # unshifted pair data, which the rates (eq 26) need.
    visited = np.stack(count_list, axis=-1).sum(axis=0) > 0
    shift = np.where(visited.any(axis=0), np.where(visited, bias_ar, np.inf).min(axis=0), 0.0)
    opt_data = generate_dhamed_input(count_list, bias_ar - shift[None, :], n_states)
    # F is linear in the counts: dividing them by the total leaves the optimum unchanged and makes BFGS's absolute
    # gradient tolerance meaningful whatever the amount of data.
    scale = 1.0 / max(opt_data.nk.sum(), 1.0)
    opt_data = dataclasses.replace(opt_data, nk=opt_data.nk * scale, nijp=opt_data.nijp * scale)

    start = time.time()
    fprime = None if numerical_gradients else grad_dhamed_likelihood_ref_0
    kwargs.setdefault("gtol", 1e-9)        # per transition count, after the scaling above
    result = minimize(wrapper_ll, g0[:-1], args=(opt_data, jit_gradient), jac=fprime, method="BFGS", options=kwargs)
    logger.info("DHAMed: %d states, %d transition pairs, %d iterations, %.2f s",
                len(data.nk), len(data.ip), result.nit, time.time() - start)
    # BFGS often stops on "precision loss" once the gradient is at machine precision: judge convergence by the
    # gradient per transition count instead.
    converged = bool(result.success or np.abs(result.jac).max() < 1e-6)
    if not converged:
        logger.warning("DHAMed optimisation did not converge: %s (max |gradient| %.3g, per transition count). Check "
                       "the state definitions and connectivity, or pass g_init / maxiter.", result.message,
                       np.abs(result.jac).max())
    g = np.append(result.x, 0.0)
    return DhamedResult(g=g - np.log(np.sum(np.exp(g))), included=included, data=data, optimizer=result,
                        converged=converged)


def run_dhamed(count_list, bias_ar, numerical_gradients=False, g_init=None,
               jit_gradient=False, last_g_zero=True, **kwargs):
    """
    Run DHAMed from a list of count matrices and an array specfying the
    biases in each simulation (window).

    The list of the individual count matrices C contain the transition counts
    between the different states (or bins in umbrella sampling). C[i,j] where
    i is the product state and j the reactent state. The first row contains
    thus all the transitions into state 0.The first column C[:,0] all
    transition out of state 0.

    The bias array contains a bias value for each state and for each simulation
    (or window in umbrella sampling. The bias NEEDS to be given in units to kBT.

    Most parameters besides count_list and bias_ar are only relevant for testing
    and further code developement. solve_dhamed returns a DhamedResult with errors and rates.

    The function takes keyword arguments passed on as `options` to
    scipy.optimize.minimize(method="BFGS"), such as gtol and maxiter.

    Parameters:
    -----------
    count_list: list of arrays, NxN the transition counts for
                each simulation (window)
    bias_ar: array, (Nxnwin) the bias acting on each state in each
             simulation (window)
    numerical_gradients: Boolean. default False, use analytical gradients.
    g_init: initial log-weights,

    Returns:
    --------
    og: array-like, optimized log-weights of the included states (last one set to 0)
    """

    n_states = count_list[0].shape[0]

    data = generate_dhamed_input(count_list, bias_ar, n_states)
    if g_init is None:
       g_init = np.zeros(len(data.nk))

    start = time.time()

    if numerical_gradients:
       fprime = None
    else:
         if jit_gradient:
            fprime = grad_dhamed_likelihood
         else:
              fprime = grad_dhamed_likelihood_ref

    if last_g_zero:
       og = min_dhamed_bfgs(g_init, data, jit_gradient=jit_gradient,
                            numerical_gradients=numerical_gradients, **kwargs)

    else:
         result = minimize(effective_log_likelihood_count_list, g_init*1.0,
                           args=(data.ip, data.jp, data.ti, data.tj, data.vi, data.vj,
                                 data.nk, data.nijp),
                           jac=fprime, method="BFGS", options=kwargs)
         og = result.x
    logger.info("time elapsed %.2f s", time.time() - start)

    return og


def min_dhamed_bfgs(g_init, data, jit_gradient=False,
                    numerical_gradients=False, **kwargs):
    """
    Find the optimal weights to solve the DHAMed equations by
    determining the N-1 optimal relative weights of the states.

    Parameters:
    -----------
    g_init: array, N entries, initial log weights
    data: DhamedPairData
        Per-pair inputs and per-state transition counts (see
        generate_dhamed_input).

    """
    g_prime = g_init[:-1].T

    if numerical_gradients:
        fprime=None
    else:
        fprime=grad_dhamed_likelihood_ref_0

    result = minimize(wrapper_ll, g_prime,
                      args=(data, jit_gradient),
                      jac=fprime, method="BFGS", options=kwargs)
    return np.append(result.x, 0)
