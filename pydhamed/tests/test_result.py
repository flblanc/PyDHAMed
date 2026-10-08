"""solve_dhamed: populations, Hessian errors and rates; effective_bias with pymbar."""

import numpy as np
import pytest
from scipy.linalg import expm

from pydhamed import solve_dhamed, run_dhamed, count_matrix, effective_bias
from pydhamed.optimize_dhamed import effective_log_likelihood_count_ref

# Unbiased 4-state chain with detailed balance.
P_EXACT = np.array([0.1, 0.4, 0.3, 0.2])
K_EXACT = np.zeros((4, 4))
for i, j, k in ((1, 0, 0.05), (2, 1, 0.02), (3, 2, 0.04)):        # k for j -> i, reverse by detailed balance
    K_EXACT[i, j] = k
    K_EXACT[j, i] = k * P_EXACT[j] / P_EXACT[i]
K_EXACT -= np.diag(K_EXACT.sum(axis=0))


def biased_trajectory(u, n, rng):
    """ Discrete-time trajectory (dt = 1) of the master equation biased by u (kT) with the TST rate model
    (eq 27): rates out of j are multiplied by e^{u_j}. """
    K = K_EXACT * np.exp(u)[None, :]
    K -= np.diag(np.diag(K))
    K -= np.diag(K.sum(axis=0))
    T = expm(K)
    traj = np.zeros(n, dtype=int)
    traj[0] = 1
    for t in range(1, n):
        traj[t] = rng.choice(4, p=T[:, traj[t - 1]])
    return traj


@pytest.fixture(scope="module")
def biased_runs():
    rng = np.random.default_rng(0)
    biases = [np.zeros(4), np.array([1.5, 0.0, -0.5, 0.5]), np.array([0.0, 1.0, 1.0, 0.0])]
    counts = [count_matrix(biased_trajectory(u, 150_000, rng), n_states=4) for u in biases]
    return counts, np.column_stack(biases)


def test_populations_and_compatibility(biased_runs):
    counts, bias = biased_runs
    res = solve_dhamed(counts, bias)
    assert np.allclose(res.populations.sum(), 1.0)
    assert np.allclose(res.populations, P_EXACT, rtol=0.05)
    og = run_dhamed(counts, bias)
    assert np.allclose(og - og[-1], res.g - res.g[-1], atol=1e-4)


def test_hessian_matches_finite_differences(biased_runs):
    counts, bias = biased_runs
    res = solve_dhamed(counts, bias)
    d = res.data
    F = lambda g: effective_log_likelihood_count_ref(g, d.ip, d.jp, d.ti, d.tj, d.vi, d.vj, d.nk, d.nijp)
    h, n = 1e-4, len(res.g)
    H_fd = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            ea, eb = np.eye(n)[a] * h, np.eye(n)[b] * h
            H_fd[a, b] = (F(res.g + ea + eb) - F(res.g + ea - eb) - F(res.g - ea + eb) + F(res.g - ea - eb)) / (4 * h * h)
    assert np.allclose(res.hessian(), H_fd, rtol=1e-3, atol=1e-2)
    sigma = res.free_energy_errors()
    assert np.all(sigma > 0) and np.all(np.abs(-np.log(res.populations) + np.log(P_EXACT)
                                               - np.mean(-np.log(res.populations) + np.log(P_EXACT))) < 4 * sigma + 1e-3)


def test_rates_tst_model(biased_runs):
    counts, bias = biased_runs
    K = solve_dhamed(counts, bias).rate_matrix("tst", dt=1.0)
    for i, j in ((1, 0), (2, 1), (3, 2), (0, 1), (1, 2), (2, 3)):
        assert abs(K[i, j] / K_EXACT[i, j] - 1) < 0.1
    assert np.allclose(K.sum(axis=0), 0.0)


def test_effective_bias_constant_within_states():
    rng = np.random.default_rng(1)
    n_runs, n_states = 3, 4
    true = rng.normal(0, 1, (n_states, n_runs))
    state_n = rng.integers(0, n_states, 2000)
    run_n = rng.integers(0, n_runs, 2000)
    u_kn = true[state_n].T                                            # bias depends on the state only
    assert np.allclose(effective_bias(u_kn, state_n, run_n), true)


def test_effective_bias_variable_within_state():
    # One state, x uniform on [0, 1) in the unbiased ensemble, runs at bias u_a(x) = c_a x; sample each run exactly.
    rng = np.random.default_rng(2)
    c = np.array([0.0, 2.0, 5.0])
    xs, runs = [], []
    for a, ca in enumerate(c):
        x = rng.random(400_000)
        if ca:
            x = -np.log(1 - x * (1 - np.exp(-ca))) / ca                 # density ~ exp(-c x) on [0, 1)
        xs.append(x)
        runs.append(np.full(x.size, a))
    x, run_n = np.concatenate(xs), np.concatenate(runs)
    u_kn = c[:, None] * x[None, :]
    bias, log_w = effective_bias(u_kn, np.zeros(x.size, dtype=int), run_n, return_frame_weights=True)
    safe_c = np.where(c > 0, c, 1.0)
    exact = np.where(c > 0, -np.log((1 - np.exp(-safe_c)) / safe_c), 0.0)   # -ln <e^{-c x}> over uniform x
    assert np.allclose(bias[0], exact, atol=5e-3)
    w = np.exp(log_w)
    assert np.isclose(w.sum(), 1.0) and abs(np.sum(w * x) - 0.5) < 5e-3    # unbiased mean of x
