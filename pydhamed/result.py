"""Result of a DHAMed calculation: populations, free energies, statistical errors and rate coefficients.

Equation numbers refer to L. S. Stelzl, A. Kells, E. Rosta, G. Hummer, "Dynamic Histogram Analysis To Determine
Free Energies and Rates from Biased Simulations", J. Chem. Theory Comput. 13, 6328-6342 (2017).
"""

from dataclasses import dataclass

import numpy as np

RATE_MODELS = ("tst", "linear", "metropolis", "anti_metropolis")


@dataclass
class DhamedResult:
    """
    g: log-populations ln p_i of the included states, normalised so that sum_i p_i = 1.
    included: boolean mask over all states (states without a transition in and out, or without a transition
        partner, are excluded, see eq 10).
    data: the DhamedPairData the likelihood was built from (pair indices refer to included states).
    optimizer: the scipy OptimizeResult of the minimisation of F (eq 12).
    """
    g: np.ndarray
    included: np.ndarray
    data: object
    optimizer: object

    @property
    def n_states(self):
        return len(self.included)

    @property
    def state_indices(self):
        """ Original indices of the included states. """
        return np.flatnonzero(self.included)

    def _expand(self, values, fill=np.nan):
        out = np.full(self.n_states, fill, dtype=float)
        out[self.included] = values
        return out

    @property
    def populations(self):
        """ p_i over all states (0 for excluded states). """
        return self._expand(np.exp(self.g), fill=0.0)

    @property
    def free_energies(self):
        """ G_i = -ln p_i in kT over all states (NaN for excluded states), minimum at 0. """
        G = -self.g
        return self._expand(G - G.min())

    def hessian(self):
        """ Hessian of F (eqs 14, 15) at the optimum, over the included states. """
        d, g = self.data, self.g
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a = d.ti * np.exp(d.vi - g[d.ip])
            b = d.tj * np.exp(d.vj - g[d.jp])
            h = np.nan_to_num(d.nijp / (2.0 + a / b + b / a))
        n = len(g)
        H = np.zeros((n, n))
        np.add.at(H, (d.ip, d.jp), -h)
        np.add.at(H, (d.jp, d.ip), -h)
        np.add.at(H, (d.ip, d.ip), h)
        np.add.at(H, (d.jp, d.jp), h)
        return H

    def covariance(self):
        """ Covariance of the g_i under the normalisation constraint, C = (H + O)^-1 - O (eq 17), included states. """
        n = len(self.g)
        O = np.full((n, n), 1.0 / n)
        return np.linalg.inv(self.hessian() + O) - O

    def free_energy_errors(self):
        """ Standard errors sigma_i of G_i (in kT) over all states (NaN for excluded states) (eq 18). """
        return self._expand(np.sqrt(np.clip(np.diag(self.covariance()), 0.0, None)))

    def free_energy_difference_errors(self):
        """ Matrix of standard errors of G_j - G_i, sqrt(C_ii - 2 C_ij + C_jj) (in kT), over all states. """
        C = self.covariance()
        d = np.diag(C)
        sub = np.sqrt(np.clip(d[:, None] - 2 * C + d[None, :], 0.0, None))
        out = np.full((self.n_states, self.n_states), np.nan)
        out[np.ix_(self.included, self.included)] = sub
        return out

    def rate_matrix(self, rate_model="tst", b=0.5, dt=1.0):
        """ Unbiased rate matrix K (eq 26), over all states: K[i, j] is the rate of j -> i for i != j, and
        K[j, j] = -sum_i K[i, j], so that dP/dt = K P (eq 3). Rates in units of 1 / dt, where dt is the time
        between the frames the counts were taken from (the lag time).

        rate_model: how the bias changes the rates (eqs 27-30), through v_ij:
            "tst"             v = 1                          (bias lifts the initial well; eq 27)
            "linear"          v = exp(-b (u_i + u_j))        (linear free-energy relation, b = 1/2 halfway TS; eq 28)
            "metropolis"      v = exp(-max(u_i, u_j))        (eq 29)
            "anti_metropolis" v = exp(-min(u_i, u_j))        (eq 30)
        For unbiased runs (u = 0) all models coincide.
        """
        if rate_model not in RATE_MODELS:
            raise ValueError(f"rate_model must be one of {RATE_MODELS}.")
        d = self.data
        p = np.exp(self.g)
        ui, uj = d.vi, d.vj
        v = {"tst": np.ones_like(ui), "linear": np.exp(-b * (ui + uj)),
             "metropolis": np.exp(-np.maximum(ui, uj)), "anti_metropolis": np.exp(-np.minimum(ui, uj))}[rate_model]
        n = len(p)
        numerator = np.zeros((n, n))
        denominator = np.zeros((n, n))
        np.add.at(numerator, (d.ip, d.jp), d.nijp)
        np.add.at(denominator, (d.ip, d.jp),
                  v * dt * (d.tj * np.exp(uj) + d.ti * np.exp(ui) * p[d.jp] / p[d.ip]))
        with np.errstate(divide="ignore", invalid="ignore"):
            k_ij = np.where(denominator > 0, numerator / denominator, 0.0)   # j -> i, for pairs stored with i < j
        K = k_ij + (k_ij * (p[None, :] / p[:, None])).T                       # i -> j by detailed balance
        K -= np.diag(K.sum(axis=0))
        out = np.zeros((self.n_states, self.n_states))
        out[np.ix_(self.included, self.included)] = K
        return out
