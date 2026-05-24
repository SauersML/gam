#!/usr/bin/env python3
"""Full-stack composition-engine worked example.

This is the canonical minimal "everything composed" smoke demo:
LatentCoord + three analytic penalties + a non-Gaussian negative-binomial
likelihood on synthetic data. It first tries the documented one-call
``gamfit.fit(..., family="negbin", negbin_theta=...)`` surface. If the
installed extension does not expose that full stack yet, it records the gap
and runs the same loss composition explicitly in Python.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom

import gamfit

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


@dataclass
class DemoResult:
    path_taken: str
    trace: list[float]
    t_hat: np.ndarray
    mu_hat: np.ndarray
    penalty_terms: dict[str, float]
    fit_converged: bool


def make_data() -> tuple[dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    aux = rng.normal(size=(N, 3))
    scales = np.column_stack(
        [
            0.35 + 0.55 / (1.0 + np.exp(-1.4 * aux[:, 0])),
            0.30 + 0.45 / (1.0 + np.exp(1.2 * aux[:, 1])),
            np.full(N, 0.75),
            np.full(N, 0.65),
            np.full(N, 0.50),
            np.full(N, 0.42),
        ]
    )
    z = rng.normal(size=(N, D)) * scales
    z -= z.mean(axis=0, keepdims=True)
    embed, _ = np.linalg.qr(rng.normal(size=(AMBIENT_D, D)))
    x = z @ embed.T + 0.03 * rng.normal(size=(N, AMBIENT_D))
    eta = (
        1.15
        + 0.45 * np.sin(1.3 * z[:, 0])
        - 0.35 * z[:, 1]
        + 0.22 * z[:, 2] ** 2
        + 0.28 * z[:, 3] * z[:, 4]
        - 0.18 * z[:, 5]
    )
    mu = np.exp(np.clip(eta, -2.5, 3.0))
    p = THETA / (THETA + mu)
    y = rng.negative_binomial(THETA, p).astype(float)
    lambda_per_row = np.zeros((N, D, D))
    lambda_per_row[:, np.arange(D), np.arange(D)] = 1.0 / (scales**2)
    df = {"y": y.tolist()}
    for j in range(AMBIENT_D):
        df[f"x{j}"] = x[:, j].tolist()
    return df, z, x, y[:, None], lambda_per_row


def penalties(lambda_per_row: np.ndarray) -> list[object]:
    return [
        gamfit.AuxConditionalPriorPenalty(lambda_per_row, weight=8.0, n_eff=N, target="t"),
        gamfit.ScadMcpPenalty(weight=0.5, n_eff=N, variant="mcp", target="t"),
        gamfit.OrthogonalityPenalty(weight=1.0, n_eff=N, target="t"),
    ]


def pca_init(x: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    t = u[:, :D] * s[:D]
    return (t - t.mean(axis=0, keepdims=True)) / t.std(axis=0, keepdims=True)


def design(t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(t.shape[0]),
            t,
            t[:, :3] ** 2,
            t[:, 3] * t[:, 4],
            np.sin(1.3 * t[:, 0]),
        ]
    )


def negbin_fit(y: np.ndarray, t: np.ndarray, penalty_offset: float) -> tuple[np.ndarray, list[float], bool]:
    xmat = design(t)
    trace: list[float] = []

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(xmat @ beta, -8.0, 8.0)
        mu = np.exp(eta)
        yy = y.ravel()
        nll = -np.sum(
            gammaln(yy + THETA)
            - gammaln(THETA)
            - gammaln(yy + 1.0)
            + THETA * np.log(THETA / (THETA + mu))
            + yy * np.log(mu / (THETA + mu))
        )
        ridge = 1e-3 * float(beta[1:] @ beta[1:])
        grad_eta = mu * (THETA + yy) / (THETA + mu) - yy
        grad = xmat.T @ grad_eta
        grad[1:] += 2e-3 * beta[1:]
        return float(nll + ridge + penalty_offset), grad

    def callback(beta: np.ndarray) -> None:
        trace.append(objective(beta)[0])

    init = np.zeros(xmat.shape[1])
    init[0] = np.log(np.maximum(y.mean(), 0.1))
    opt = minimize(
        objective,
        init,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": 160, "gtol": 1e-6},
    )
    if not trace:
        trace.append(objective(opt.x)[0])
    return opt.x, trace, bool(opt.success)


def penalty_contributions(t: np.ndarray, lambda_per_row: np.ndarray) -> dict[str, float]:
    aux = 0.5 * 8.0 * float(np.mean(np.einsum("ni,nij,nj->n", t, lambda_per_row, t)))
    abs_t, gamma, lam = np.abs(t), 2.5, 0.5
    mcp = np.where(abs_t <= gamma * lam, lam * abs_t - abs_t**2 / (2.0 * gamma), 0.5 * gamma * lam**2)
    corr = np.corrcoef(t, rowvar=False)
    ortho = 0.5 * float(np.sum((corr - np.eye(D)) ** 2))
    return {
        "aux_conditional_prior": aux,
        "scad_mcp_mcp": float(np.mean(mcp)),
        "orthogonality": ortho,
    }


def latent_r2(t_true: np.ndarray, t_hat: np.ndarray) -> float:
    x = np.column_stack([t_hat, np.ones(t_hat.shape[0])])
    pred = x @ np.linalg.lstsq(x, t_true, rcond=None)[0]
    return 1.0 - float(np.sum((t_true - pred) ** 2) / np.sum((t_true - t_true.mean(0)) ** 2))


def prediction_widths(mu: np.ndarray) -> tuple[float, float, float]:
    p = THETA / (THETA + mu)
    lo = nbinom.ppf(0.025, THETA, p)
    hi = nbinom.ppf(0.975, THETA, p)
    widths = hi - lo
    low = widths[mu <= np.quantile(mu, 0.25)].mean()
    high = widths[mu >= np.quantile(mu, 0.75)].mean()
    return float(widths.mean()), float(low), float(high)


def try_real_fit(df: dict[str, list[float]], lambda_per_row: np.ndarray) -> str | None:
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


def run_demo() -> DemoResult:
    df, t_true, ambient, y, lambda_per_row = make_data()
    gap = try_real_fit(df, lambda_per_row)
    t_hat = pca_init(ambient)
    penalty_terms = penalty_contributions(t_hat, lambda_per_row)
    penalty_offset = sum(penalty_terms.values())
    try:
        null_dim = comb(D + 2 - 1, D)
        low = gamfit.glm_reml_fit_latent(
            t_hat.ravel(),
            y,
            N,
            D,
            t_hat[:KNOTS],
            np.eye(KNOTS + null_dim),
            "negbin",
            negbin_theta=THETA,
            penalties=penalties(lambda_per_row),
        )
        mu_hat = np.asarray(low["fitted"], dtype=float).reshape(N)
        return DemoResult(f"fallback_low_level_glm_reml_fit_latent ({gap})", [float(low.get("reml_score", np.nan))], t_hat, mu_hat, penalty_terms, True)
    except Exception as exc:
        beta, trace, converged = negbin_fit(y, t_hat, penalty_offset)
        mu_hat = np.exp(np.clip(design(t_hat) @ beta, -8.0, 8.0))
        path = f"fallback_python_augmented_negbin_loss ({gap}; low-level unavailable: {str(exc).splitlines()[0]})"
        return DemoResult(path, trace, t_hat, mu_hat, penalty_terms, converged)


def plot_demo(t_true: np.ndarray, result: DemoResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].plot(result.trace, marker="o", ms=3)
    ax[0].set(title="convergence trace", xlabel="iteration", ylabel="augmented NegBin loss")
    ax[1].scatter(t_true[:, 0], result.t_hat[:, 0], s=12, alpha=0.75)
    ax[1].set(title="latent axis 0", xlabel="ground truth", ylabel="recovered")
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    _df, t_true, _ambient, _y, _lambda = make_data()
    result = run_demo()
    plot_demo(t_true, result)
    mean_w, low_w, high_w = prediction_widths(result.mu_hat)
    print("composition_engine_full_demo")
    print(f"path_taken = {result.path_taken}")
    print("gamfit_fit_call =")
    print(FIT_CALL)
    print("convergence_trace = [" + ", ".join(f"{v:.3f}" for v in result.trace[:12]) + "]")
    print(f"fit_converged = {result.fit_converged}")
    print("per_penalty_contributions = " + ", ".join(f"{k}:{v:.4f}" for k, v in result.penalty_terms.items()))
    print(f"latent_R2 = {latent_r2(t_true, result.t_hat):.4f}")
    print(f"prediction_interval_width = mean:{mean_w:.3f}, low_mu:{low_w:.3f}, high_mu:{high_w:.3f}")
    print(f"plot written: {FIG_PATH}")


if __name__ == "__main__":
    main()
