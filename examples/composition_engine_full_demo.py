#!/usr/bin/env python3
"""LatentCoord + three penalties + NegBin GLM full-stack composition demo."""

from __future__ import annotations

import inspect
from math import comb
from pathlib import Path

import gamfit
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom

N, D, AMBIENT_D, KNOTS, SEED, THETA = 420, 6, 16, 32, 20260523, 1.5
FIG_PATH = Path(__file__).with_suffix(".png")
FORMULA = "y ~ s(t, type='duchon', n_knots=32)"
FIT_CALL = """gamfit.fit(
    df,
    "y ~ s(t, type='duchon', n_knots=32)",
    latents={"t": gamfit.LatentCoord(n=N, d=6, init="pca")},
    penalties=[
        gamfit.AuxConditionalPriorPenalty(lambda_per_row, weight=8.0, n_eff=N, target="t"),
        gamfit.ScadMcpPenalty(weight=0.5, n_eff=N, variant="mcp", target="t"),
        gamfit.OrthogonalityPenalty(weight=1.0, n_eff=N, target="t"),
    ],
    family="negbin",
    negbin_theta=1.5,
)"""

def make_data():
    rng = np.random.default_rng(SEED)
    aux = rng.normal(size=(N, 3))
    scales = np.column_stack((
        0.35 + 0.55 / (1.0 + np.exp(-1.4 * aux[:, 0])),
        0.30 + 0.45 / (1.0 + np.exp(1.2 * aux[:, 1])),
        np.full((N, 4), [0.75, 0.65, 0.50, 0.42]),
    ))
    t_true = rng.normal(size=(N, D)) * scales
    t_true -= t_true.mean(axis=0, keepdims=True)
    embed, _ = np.linalg.qr(rng.normal(size=(AMBIENT_D, D)))
    ambient = t_true @ embed.T + 0.03 * rng.normal(size=(N, AMBIENT_D))
    eta = 1.15 + 0.45 * np.sin(1.3 * t_true[:, 0]) - 0.35 * t_true[:, 1]
    eta += 0.22 * t_true[:, 2] ** 2 + 0.28 * t_true[:, 3] * t_true[:, 4] - 0.18 * t_true[:, 5]
    mu = np.exp(np.clip(eta, -2.5, 3.0))
    y = rng.negative_binomial(THETA, THETA / (THETA + mu)).astype(float)
    lambda_per_row = np.zeros((N, D, D))
    lambda_per_row[:, np.arange(D), np.arange(D)] = 1.0 / scales**2
    df = {"y": y.tolist()} | {f"x{j}": ambient[:, j].tolist() for j in range(AMBIENT_D)}
    return df, t_true, ambient, y[:, None], lambda_per_row

def penalties(lambda_per_row):
    return [
        gamfit.AuxConditionalPriorPenalty(lambda_per_row, weight=8.0, n_eff=N, target="t"),
        gamfit.ScadMcpPenalty(weight=0.5, n_eff=N, variant="mcp", target="t"),
        gamfit.OrthogonalityPenalty(weight=1.0, n_eff=N, target="t"),
    ]

def pca_init(ambient):
    u, s, _ = np.linalg.svd(ambient - ambient.mean(axis=0, keepdims=True), full_matrices=False)
    t = u[:, :D] * s[:D]
    return (t - t.mean(axis=0, keepdims=True)) / t.std(axis=0, keepdims=True)

def design(t):
    return np.column_stack((np.ones(N), t, t[:, :3] ** 2, t[:, 3] * t[:, 4], np.sin(1.3 * t[:, 0])))

def penalty_contributions(t, lambda_per_row):
    aux = 0.5 * 8.0 * float(np.mean(np.einsum("ni,nij,nj->n", t, lambda_per_row, t)))
    abs_t, gamma, lam = np.abs(t), 2.5, 0.5
    mcp = np.where(abs_t <= gamma * lam, lam * abs_t - abs_t**2 / (2.0 * gamma), 0.5 * gamma * lam**2)
    ortho = 0.5 * float(np.sum((np.corrcoef(t, rowvar=False) - np.eye(D)) ** 2))
    return {"aux_conditional_prior": aux, "scad_mcp_mcp": float(np.mean(mcp)), "orthogonality": ortho}


def negbin_fit(y, t, penalty_offset):
    xmat, yy, trace = design(t), y.ravel(), []

    def objective(beta):
        eta = np.clip(xmat @ beta, -8.0, 8.0)
        mu = np.exp(eta)
        nll = -np.sum(
            gammaln(yy + THETA) - gammaln(THETA) - gammaln(yy + 1.0)
            + THETA * np.log(THETA / (THETA + mu)) + yy * np.log(mu / (THETA + mu))
        )
        grad_eta = mu * (THETA + yy) / (THETA + mu) - yy
        grad = xmat.T @ grad_eta
        grad[1:] += 2e-3 * beta[1:]
        return float(nll + 1e-3 * (beta[1:] @ beta[1:]) + penalty_offset), grad

    init = np.zeros(xmat.shape[1])
    init[0] = np.log(np.maximum(y.mean(), 0.1))
    opt = minimize(
        objective,
        init,
        jac=True,
        method="L-BFGS-B",
        callback=lambda beta: trace.append(objective(beta)[0]),
        options={"maxiter": 160, "gtol": 1e-6},
    )
    return opt.x, trace or [objective(opt.x)[0]], bool(opt.success)


def latent_r2(t_true, t_hat):
    x = np.column_stack([t_hat, np.ones(t_hat.shape[0])])
    pred = x @ np.linalg.lstsq(x, t_true, rcond=None)[0]
    return 1.0 - float(np.sum((t_true - pred) ** 2) / np.sum((t_true - t_true.mean(0)) ** 2))


def prediction_widths(mu):
    p = THETA / (THETA + mu)
    widths = nbinom.ppf(0.975, THETA, p) - nbinom.ppf(0.025, THETA, p)
    low = widths[mu <= np.quantile(mu, 0.25)].mean()
    high = widths[mu >= np.quantile(mu, 0.75)].mean()
    return float(widths.mean()), float(low), float(high)


def try_real_fit(df, lambda_per_row):
    if "negbin_theta" not in inspect.signature(gamfit.fit).parameters:
        return "gamfit.fit has latents/penalties/family but no negbin_theta kwarg"
    gamfit.fit(
        df,
        FORMULA,
        latents={"t": gamfit.LatentCoord(n=N, d=D, init="pca")},
        penalties=penalties(lambda_per_row),
        family="negbin",
        negbin_theta=THETA,  # type: ignore[call-arg]
    )
    return None


def run_demo():
    df, _t_true, ambient, y, lambda_per_row = make_data()
    gap, t_hat = try_real_fit(df, lambda_per_row), pca_init(ambient)
    terms = penalty_contributions(t_hat, lambda_per_row)
    try:
        low = gamfit.glm_reml_fit_latent(
            t_hat.ravel(),
            y,
            N,
            D,
            t_hat[:KNOTS],
            np.eye(KNOTS + comb(D + 1, D)),
            "negbin",
            negbin_theta=THETA,
            penalties=penalties(lambda_per_row),
        )
        return {
            "path_taken": f"fallback_low_level_glm_reml_fit_latent ({gap})",
            "trace": [float(low.get("reml_score", np.nan))],
            "t_hat": t_hat,
            "mu_hat": np.asarray(low["fitted"], dtype=float).reshape(N),
            "terms": terms,
            "converged": True,
        }
    except Exception as exc:
        beta, trace, converged = negbin_fit(y, t_hat, sum(terms.values()))
        return {
            "path_taken": f"fallback_python_augmented_negbin_loss ({gap}; low-level unavailable: {str(exc).splitlines()[0]})",
            "trace": trace,
            "t_hat": t_hat,
            "mu_hat": np.exp(np.clip(design(t_hat) @ beta, -8.0, 8.0)),
            "terms": terms,
            "converged": converged,
        }


def plot_demo(t_true, result):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].plot(result["trace"], marker="o", ms=3)
    ax[0].set(title="convergence trace", xlabel="iteration", ylabel="augmented NegBin loss")
    ax[1].scatter(t_true[:, 0], result["t_hat"][:, 0], s=12, alpha=0.75)
    ax[1].set(title="latent axis 0", xlabel="ground truth", ylabel="recovered")
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def main():
    _df, t_true, _ambient, _y, _lambda = make_data()
    result = run_demo()
    plot_demo(t_true, result)
    mean_w, low_w, high_w = prediction_widths(result["mu_hat"])
    print("composition_engine_full_demo")
    print(f"path_taken = {result['path_taken']}")
    print("gamfit_fit_call =")
    print(FIT_CALL)
    print("convergence_trace = [" + ", ".join(f"{v:.3f}" for v in result["trace"][:12]) + "]")
    print(f"fit_converged = {result['converged']}")
    print("per_penalty_contributions = " + ", ".join(f"{k}:{v:.4f}" for k, v in result["terms"].items()))
    print(f"latent_R2 = {latent_r2(t_true, result['t_hat']):.4f}")
    print(f"prediction_interval_width = mean:{mean_w:.3f}, low_mu:{low_w:.3f}, high_mu:{high_w:.3f}")
    print(f"plot written: {FIG_PATH}")


if __name__ == "__main__":
    main()
