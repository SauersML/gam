"""Independent Poisson LAML and smoothing-mixture mean audit for #1561.

Usage: python poisson_posterior_audit.py poisson-tensor.json [MAX_ORDER]
This diagnostic asks whether averaging smoothing uncertainty improves the
original tensor fixture. It does not tune priors or model choices to its truth.
It integrates the standard LAML expression reconstructed below. Agreement of
the production criterion's rho-dependent corrections needs a separate audit;
the reported scalar criterion discrepancy is retained, not normalized away.
"""

import itertools
import json
import math
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits


def run(path, max_order=25):
    with open(path) as handle:
        data = json.load(handle)
    assert data["family"] == "poisson"
    x, y, truth, beta_mode, rho_mode = (np.asarray(data[key], dtype=float)
                                      for key in ("x", "y", "truth", "beta", "rho"))
    n, p = x.shape
    penalties = []
    for block in data["penalties"]:
        matrix = np.zeros((p, p))
        indices = slice(block["start"], block["end"])
        matrix[indices, indices] = block["matrix"]
        penalties.append(matrix)
    penalties = np.asarray(penalties)
    eigenvalues, vectors = np.linalg.eigh(penalties.sum(axis=0))
    penalty_range = vectors[:, eigenvalues > p * np.finfo(float).eps * eigenvalues.max()]
    theta = -math.log(0.01) / 10.0
    epsilon = np.finfo(float).eps
    gamma_n = n * epsilon / (1 - n * epsilon)
    gamma_p = p * epsilon / (1 - p * epsilon)
    abs_x = np.abs(x)

    def evaluate(rho):
        penalty = np.einsum("i,ijk->jk", np.exp(rho), penalties)
        beta = beta_mode.copy()
        for iteration in range(80):
            eta = x @ beta
            mean = np.exp(eta)
            objective = np.sum(mean - y * eta) + 0.5 * beta @ penalty @ beta
            gradient = x.T @ (mean - y) + penalty @ beta
            factor = cho_factor(x.T @ (mean[:, None] * x) + penalty, lower=True)
            step = cho_solve(factor, gradient)
            decrement = gradient @ step
            # Componentwise forward-error budget for eta, exp(eta), the two
            # dot products, and their subtraction/addition. A zero gradient
            # inside this budget cannot be distinguished from cancellation.
            mean_error = mean * (gamma_p * (abs_x @ np.abs(beta)) + epsilon)
            gradient_resolution = ((gamma_n + epsilon) * (abs_x.T @ (mean + y))
                                   + abs_x.T @ mean_error
                                   + (gamma_p + epsilon) * (np.abs(penalty) @ np.abs(beta)))
            if np.all(np.abs(gradient) <= gradient_resolution):
                break
            if decrement <= n * np.finfo(float).eps * (1 + abs(objective)):
                # The objective decrease is below summation resolution. Newton's
                # analytic coefficient correction remains resolvable.
                beta -= step
                continue
            scale = 1.0
            while scale > np.finfo(float).eps:
                proposal = beta - scale * step
                candidate_eta = x @ proposal
                candidate = np.exp(candidate_eta).sum() - y @ candidate_eta + 0.5 * proposal @ penalty @ proposal
                if candidate <= objective - 1e-4 * scale * decrement:
                    beta = proposal
                    break
                scale *= 0.5
            else:
                raise AssertionError("independent Poisson Newton line search did not converge")
        else:
            raise AssertionError(f"independent Poisson Newton solve did not converge at rho={rho}: residual/resolution={np.max(np.abs(gradient) / gradient_resolution)}")
        sign, logdet_penalty = np.linalg.slogdet(penalty_range.T @ penalty @ penalty_range)
        assert sign > 0
        criterion = objective + np.log(np.diag(factor[0])).sum() - 0.5 * logdet_penalty
        covariance = cho_solve(factor, np.eye(p))
        variance = np.einsum("ij,jk,ik->i", x, covariance, x)
        return criterion, beta, mean, np.exp(eta + 0.5 * variance)

    started = time.monotonic()
    mode_cost, beta, mode_mean, conditional_mean = evaluate(rho_mode)
    mode_error = float(np.max(np.abs(beta - beta_mode)))
    criterion_error = float(abs(mode_cost - data["criterion"]))
    mode_penalty = np.einsum("i,ijk->jk", np.exp(rho_mode), penalties)
    mode_hessian = x.T @ (mode_mean[:, None] * x) + mode_penalty
    # Dense Gram assembly and log determinants lose absolute accuracy in
    # proportion to their condition numbers. Use their arithmetic resolution
    # to check the independently assembled criterion, and publish the bound.
    operation_count = max(n, p)
    gamma = operation_count * np.finfo(float).eps / (1 - operation_count * np.finfo(float).eps)
    criterion_resolution = 0.5 * p * gamma * (
        np.linalg.cond(mode_hessian) + np.linalg.cond(penalty_range.T @ mode_penalty @ penalty_range))
    stored = data["conditional_covariance"]
    stored_covariance = np.array(stored["data"]).reshape(stored["dim"])
    covariance = np.linalg.inv(mode_hessian)
    covariance_error = np.linalg.norm(covariance - stored_covariance) / np.linalg.norm(covariance)
    rmse = lambda mean: float(np.sqrt(np.mean((mean - truth) ** 2)))
    print(json.dumps({"rows": n, "coefficients": p, "rho_dim": len(rho_mode),
                      "beta_error": mode_error, "criterion_error": criterion_error,
                      "criterion_arithmetic_resolution": criterion_resolution,
                      "conditional_covariance_relative_error": covariance_error,
                      "mode_rmse": rmse(mode_mean), "conditional_posterior_mean_rmse": rmse(conditional_mean)}), flush=True)
    assert mode_error < 1e-7 and covariance_error < 1e-7, "independent Poisson conditional problem disagrees"
    if criterion_error > criterion_resolution:
        print(json.dumps({"production_density_equivalence": "not certified",
                          "reason": "reported criterion differs beyond the dense reconstruction's roundoff estimate; results below target independently reconstructed standard LAML"}), flush=True)
    previous = None
    for order in (7, 13, 25, 49):
        if order > max_order:
            break
        nodes, weights = roots_legendre(order)
        uniform = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        rho_nodes = -2 * np.log(-np.log1p(uniform - 1) / theta)
        mass = 0.0
        mixture_mean = np.zeros(n)
        conditional_mode_mixture = np.zeros(n)
        for indices in itertools.product(range(order), repeat=len(rho_mode)):
            cost, _, mean, posterior_mean = evaluate(rho_nodes[list(indices)])
            weight = math.prod(weights[i] for i in indices) * math.exp(mode_cost - cost)
            mass += weight
            mixture_mean += weight * posterior_mean
            conditional_mode_mixture += weight * mean
        mixture_mean /= mass
        conditional_mode_mixture /= mass
        result = {"order": order, "evaluations": order ** len(rho_mode), "mass": mass,
                  "posterior_mean_rmse": rmse(mixture_mean),
                  "conditional_mode_mixture_rmse": rmse(conditional_mode_mixture),
                  "elapsed_seconds": time.monotonic() - started}
        if previous is not None:
            result["successive_prediction_change_max"] = float(np.max(np.abs(mixture_mean - previous)))
        print(json.dumps(result), flush=True)
        previous = mixture_mean


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 25)
