import numpy as np
import numba
import time
from scipy.optimize import minimize

from .prepare_dhamed import generate_dhamed_input


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
    of the effective log-likelihood into `grad`.
    """
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        vij = np.exp(vj[ipair]-g[j]-vi[ipair]+g[i])
        # don't think I need to test if ti exists
        if ti[ipair] > 0:
            grad[i] += -nijp[ipair] / (1.0 + tj[ipair]*vij/ti[ipair])
        if tj[ipair] >0 :
            grad[j] += -nijp[ipair] / (1.0 + ti[ipair]/(vij*tj[ipair]))
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


def wrapper_ll(g_prime, ip, jp, ti, tj, vi, vj, nk, nijp,
               jit_gradient=False):
    """
    Adding the extra zero when minimizing N-1 relative weights.
    """
    g_i = np.append(g_prime, [0], axis=0)
    l = effective_log_likelihood_count_list(g_i,  ip, jp, ti, tj, vi, vj, nk, nijp)
    return l


def grad_dhamed_likelihood_ref_0(g_prime, ip, jp, ti, tj, vi, vj, nk, nijp,
                                jit_gradient=False):
    g = np.append(g_prime, [0], axis=0)
    grad = np.zeros(g.shape[0] )
    grad[:-1]  += nk[:-1]
    loop = _loop_grad_dhamed_likelihood_0 if jit_gradient else _loop_grad_dhamed_likelihood_0_ref
    grad = loop(grad, g, ip, jp, ti, tj, vi, vj, nijp)
    return grad[:-1]


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
    and further code developement.

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
    og: array-like, optimized log-weights
    """

    n_states = count_list[0].shape[0]

    n_out, ip, jp, vi, vj, ti, tj, nijp, n_actual = generate_dhamed_input(count_list,
                                                                          bias_ar,
                                                                          n_states,
                                                                          return_included_state_indices=False)
    if g_init is None:
       g_init = np.zeros(n_actual)

    start = time.time()

    if numerical_gradients:
       fprime = None
    else:
         if jit_gradient:
            fprime = grad_dhamed_likelihood
         else:
              fprime = grad_dhamed_likelihood_ref

    if last_g_zero:
       og = min_dhamed_bfgs(g_init, ip, jp, ti, tj, vi, vj, n_out, nijp, jit_gradient=jit_gradient,
                            numerical_gradients=numerical_gradients, **kwargs)

    else:
         result = minimize(effective_log_likelihood_count_list, g_init*1.0,
                           args=(ip, jp, ti, tj, vi, vj, n_out, nijp),
                           jac=fprime, method="BFGS", options=kwargs)
         og = result.x
    end = time.time()
    print("time elapsed {} s".format(end-start))

    return og


def min_dhamed_bfgs(g_init, ip, jp, ti, tj, vi, vj, n_out, nijp, jit_gradient=False,
                    numerical_gradients=False, **kwargs):
    """
    Find the optimal weights to solve the DHAMed equations by
    determining the N-1 optimal relative weights of the states.

    Parameters:
    -----------
    g_init: array, N entries, initial log weights
    ip: array of integers,

    """
    g_prime = g_init[:-1].T

    if numerical_gradients:
        fprime=None
    else:
        fprime=grad_dhamed_likelihood_ref_0

    result = minimize(wrapper_ll, g_prime,
                      args=(ip, jp, ti, tj, vi, vj, n_out, nijp, jit_gradient),
                      jac=fprime, method="BFGS", options=kwargs)
    return np.append(result.x, 0)
