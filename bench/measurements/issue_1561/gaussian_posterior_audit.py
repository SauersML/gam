"""Independent Gaussian REML / smoothing-posterior quadrature for #1561.

Reads the model-service export, certifies the reconstructed coefficient mode,
conditional covariance and REML criterion, then integrates under the declared
PC prior. Gauss-Legendre nodes act on independent prior CDF coordinates, so the
rule has positive weights, unbounded rho support and no Gaussian tail proposal.
Dispersion is fixed at the exported estimate, as in the production cubature.

Usage: python gaussian_posterior_audit.py gaussian-integration-problem.json
"""

import argparse
import json
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits


def run(path, undo_bias_transform):
    with open(path) as handle:
        data = json.load(handle)
    x, y, rho_mode = (np.asarray(data[k], dtype=float) for k in ("x", "y", "rho"))
    n, p = x.shape
    assert len(rho_mode) == 2, "this independent audit integrates the exported two-penalty fixture"
    penalties = []
    penalized_columns = np.zeros(p, dtype=bool)
    for block in data["penalties"]:
        matrix = np.zeros((p, p))
        indices = slice(block["start"], block["end"])
        penalized_columns[indices] = True
        matrix[indices, indices] = block["matrix"]
        penalties.append(matrix)
    penalties = np.asarray(penalties)
    eigenvalues, vectors = np.linalg.eigh(penalties.sum(axis=0))
    positive = eigenvalues > p * np.finfo(float).eps * np.max(eigenvalues)
    penalty_range = vectors[:, positive]
    nu = n - (p - np.sum(positive))
    gram, rhs = x.T @ x, x.T @ y
    phi = data["phi"]
    theta = -math.log(0.01) / 10.0  # existing declared distribution policy
    # The solver standardizes unpenalized nonconstant columns. Scaling such a
    # column by 1/s changes 1/2 log|H| by -log(s), with no penalty determinant
    # contribution. Centering against the intercept has determinant one.
    unpenalized_scales = np.std(x[:, ~penalized_columns], axis=0)
    criterion_offset = -np.log(unpenalized_scales[unpenalized_scales > 1e-12]).sum()

    def evaluate(rho):
        penalty = np.einsum("i,ijk->jk", np.exp(rho), penalties)
        factor = cho_factor(gram + penalty, lower=True)
        beta = cho_solve(factor, rhs)
        covariance = phi * cho_solve(factor, np.eye(p))
        residual = y - x @ beta
        penalized_deviance = residual @ residual + beta @ penalty @ beta
        logdet_h = 2 * np.log(np.diag(factor[0])).sum()
        sign, logdet_s = np.linalg.slogdet(penalty_range.T @ penalty @ penalty_range)
        assert sign > 0
        cost = criterion_offset + 0.5 * (nu * (1 + math.log(2 * math.pi * penalized_deviance / nu)) + logdet_h - logdet_s)
        return cost, beta, covariance

    cost_mode, beta_mode, conditional = evaluate(rho_mode)
    beta_error = np.max(np.abs(beta_mode - data["beta"]))
    covariance_error = np.linalg.norm(conditional - data["conditional_covariance"]) / np.linalg.norm(conditional)
    criterion_error = abs(cost_mode - data["criterion"])
    print(json.dumps(dict(n=n, p=p, rank=int(positive.sum()), beta_error=beta_error,
                          conditional_relative_error=covariance_error, criterion_error=criterion_error)))
    assert beta_error < 1e-7 and covariance_error < 1e-7 and criterion_error < 1e-7, "independent reconstruction disagrees with fitted problem"

    baseline = np.einsum("ij,jk,ik->i", x, conditional, x)
    reported = np.einsum("ij,jk,ik->i", x, data["marginal_covariance"], x)
    if undo_bias_transform:
        penalty_mode = np.einsum("i,ijk->jk", np.exp(rho_mode), penalties)
        bias_jacobian = np.eye(p) + conditional @ penalty_mode / phi
        inverse_bias = np.linalg.inv(bias_jacobian)
        raw_cubature = inverse_bias @ data["marginal_covariance"] @ inverse_bias.T
        raw_cubature_variance = np.einsum("ij,jk,ik->i", x, raw_cubature, x)
    previous = None
    for order in (17, 33, 65, 129):
        nodes, weights = roots_legendre(order)
        uniform = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        # F_rho(r) = exp(-theta exp(-r/2)); inverse CDF uses log1p near one.
        rho_nodes = -2 * np.log(-np.log1p(uniform - 1) / theta)
        mass = 0.0
        mean_shift = np.zeros(p)
        second_moment = np.zeros((p, p))
        for i in range(order):
            for j in range(order):
                cost, beta, covariance = evaluate(np.array([rho_nodes[i], rho_nodes[j]]))
                weight = weights[i] * weights[j] * math.exp(cost_mode - cost)
                shift = beta - beta_mode
                mass += weight
                mean_shift += weight * shift
                second_moment += weight * (covariance + np.outer(shift, shift))
        mean_shift /= mass
        marginal = second_moment / mass - np.outer(mean_shift, mean_shift)
        prediction_variance = np.einsum("ij,jk,ik->i", x, marginal, x)
        width_ratio = np.sqrt(prediction_variance / baseline)
        result = dict(order=order, evaluations=order**2, mass=mass,
                      width_ratio_min=float(width_ratio.min()), width_ratio_max=float(width_ratio.max()),
                      reported_width_relative_error_max=float(np.max(np.abs(np.sqrt(reported / prediction_variance) - 1))),
                      posterior_mean_shift_max=float(np.max(np.abs(x @ mean_shift))))
        if undo_bias_transform:
            result["pre_bias_transform_width_relative_error_max"] = float(
                np.max(np.abs(np.sqrt(raw_cubature_variance / prediction_variance) - 1)))
        if previous is not None:
            result["successive_width_relative_change_max"] = float(np.max(np.abs(np.sqrt(prediction_variance / previous) - 1)))
        print(json.dumps(result), flush=True)
        previous = prediction_variance
    assert result["successive_width_relative_change_max"] < 1e-5, "independent quadrature has not converged"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem")
    parser.add_argument("--undo-bias-transform", action="store_true",
                        help="attribute the old solver's incorrectly transformed covariance")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.problem, args.undo_bias_transform)
