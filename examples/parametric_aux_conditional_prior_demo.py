#!/usr/bin/env python3
"""ParametricAuxConditionalPriorPenalty worked example.

Commit ``buhl4ip5t`` adds the learnable sibling of the fixed
``AuxConditionalPriorPenalty`` from ``b31w36m23``. FIXED requires the user to
pre-compute one ``Lambda_n`` per row. PARAMETRIC gives REML the aux rows and
learns ``Lambda_k(u_n) = alpha_k + beta_k * ||u_n - mu_k||^2`` directly.

The setup mirrors ``aux_conditional_prior_demo.py``: four latent axes in an
8D ambient space, with a 3D aux variable correlated with the two true axes
and independent from two nuisance axes. ARD-alone remains gauge-invariant;
the aux prior anchors the basis; pairing parametric aux with ARD prunes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import warnings

import numpy as np


N, D, AMBIENT_D, TRUE_D, SEED = 720, 4, 8, 2, 902
AUX_WEIGHT, AXIS_CUTOFF, AUX_CUTOFF = 8.0, 0.04, 0.35
FIG_PATH = Path(__file__).with_suffix(".png")
CONFIGS = (
    ("ARD only", "gamfit.fit(..., penalties=[ARDPenalty('t')])"),
    ("FIXED AuxConditional only", "gamfit.fit(..., penalties=[AuxConditionalPriorPenalty(lambda_per_row, weight=8.0, n_eff=N)])"),
    ("PARAMETRIC AuxConditional only", "gamfit.fit(..., penalties=[ParametricAuxConditionalPriorPenalty(aux, alpha_init, beta_init, mu_init, weight=8.0, n_eff=N)])"),
    ("PARAMETRIC AuxConditional + ARD paired", "gamfit.fit(..., penalties=[ParametricAuxConditionalPriorPenalty(...), ARDPenalty('t')])"),
)


@dataclass
class FitReport:
    name: str; call: str; coords: np.ndarray


@dataclass
class ParamPath:
    alpha: np.ndarray; beta: np.ndarray; mu: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    aux = rng.normal(size=(N, 3))
    s0 = 0.16 + 1.05 * sigmoid(1.4 * aux[:, 0] - 0.6 * aux[:, 1])
    s1 = 0.14 + 0.95 * sigmoid(-1.1 * aux[:, 0] + 0.9 * aux[:, 2])
    signs = rng.choice([-1.0, 1.0], size=(N, TRUE_D))
    latent = np.column_stack([
        s0 * signs[:, 0] + 0.04 * rng.standard_normal(N),
        s1 * signs[:, 1] + 0.04 * rng.standard_normal(N),
        1.15 * rng.standard_normal(N),
        0.95 * rng.standard_normal(N),
    ])
    latent -= latent.mean(axis=0, keepdims=True)
    rotate, _ = np.linalg.qr(rng.normal(size=(D, D)))
    embed, _ = np.linalg.qr(rng.normal(size=(AMBIENT_D, D)))
    x = (latent @ rotate.T) @ embed.T + 0.025 * rng.standard_normal((N, AMBIENT_D))
    diag = np.column_stack([1 / s0**2, 1 / s1**2, np.full(N, 1 / 25), np.full(N, 1 / 25)])
    lambdas = np.zeros((N, D, D))
    lambdas[:, np.arange(D), np.arange(D)] = diag
    return x - x.mean(axis=0, keepdims=True), aux, lambdas


def initial_latent(x: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    t = u[:, :D] * s[:D]
    gauge, _ = np.linalg.qr(np.random.default_rng(SEED + 5).normal(size=(D, D)))
    t = t @ gauge.T
    return t - t.mean(axis=0, keepdims=True)


def aux_design(aux: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(N), aux, aux * aux,
                            aux[:, 0] * aux[:, 1], aux[:, 0] * aux[:, 2], aux[:, 1] * aux[:, 2]])


def axis_aux_r2(coords: np.ndarray, aux: np.ndarray) -> np.ndarray:
    design, vals = aux_design(aux), []
    for j in range(coords.shape[1]):
        y = coords[:, j] ** 2
        y -= y.mean()
        denom = float(np.sum(y * y))
        if denom < 1e-12:
            vals.append(0.0); continue
        pred = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        vals.append(max(0.0, 1.0 - float(np.sum((y - pred) ** 2)) / denom))
    return np.asarray(vals)


def axis_variance(coords: np.ndarray) -> np.ndarray:
    return np.var(coords, axis=0, ddof=1)


def axes_kept(report: FitReport) -> int:
    var = axis_variance(report.coords)
    return int(np.count_nonzero(var > AXIS_CUTOFF * max(float(var.max()), 1e-12)))


def aux_score(report: FitReport, aux: np.ndarray) -> float:
    return float(np.mean(np.sort(axis_aux_r2(report.coords, aux))[-TRUE_D:]))


def aux_align(t0: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    diag, rng = np.diagonal(lambdas, axis1=1, axis2=2), np.random.default_rng(SEED + 11)
    starts = [np.eye(D)] + [np.linalg.qr(rng.normal(size=(D, D)))[0] for _ in range(31)]
    best: tuple[float, np.ndarray] | None = None
    for q in starts:
        val, step = aux_objective(t0, q, diag), 0.08
        for _ in range(260):
            y = t0 @ q
            cand, _ = np.linalg.qr(q - step * AUX_WEIGHT * (t0.T @ (diag * y) / N))
            if np.linalg.det(cand) < 0.0:
                cand[:, 0] *= -1.0
            cand_val = aux_objective(t0, cand, diag)
            if cand_val < val:
                q, val, step = cand, cand_val, min(step * 1.03, 0.2)
            else:
                step *= 0.5
        if best is None or val < best[0]:
            best = (val, q)
    return t0 @ best[1]


def aux_objective(t0: np.ndarray, q: np.ndarray, diag: np.ndarray) -> float:
    y = t0 @ q
    return float(0.5 * np.mean(np.sum(diag * y * y, axis=1)))


def parametric_lambda(aux: np.ndarray, path: ParamPath) -> np.ndarray:
    diff = aux[:, None, :] - path.mu[None, :, :]
    return path.alpha[None, :] + path.beta[None, :] * np.sum(diff * diff, axis=2)


def learn_parametric(coords: np.ndarray, aux: np.ndarray) -> tuple[ParamPath, list[ParamPath]]:
    log_alpha = np.log(np.array([0.80, 0.70, 0.09, 0.09]))
    raw_beta = inverse_softplus(np.array([0.25, 0.22, 0.01, 0.01]))
    mu = np.array([[0.40, -0.20, 0.00], [-0.30, 0.10, 0.20], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ma = np.zeros_like(log_alpha); va = np.zeros_like(log_alpha)
    mb = np.zeros_like(raw_beta); vb = np.zeros_like(raw_beta)
    mm = np.zeros_like(mu); vm = np.zeros_like(mu)
    history: list[ParamPath] = []
    for it in range(2000):
        alpha, beta = np.exp(log_alpha), softplus(raw_beta)
        diff = aux[:, None, :] - mu[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        lam = np.clip(alpha[None, :] + beta[None, :] * dist2, 1e-8, None)
        score = coords * coords - 1.0 / lam
        ga = 0.5 * np.mean(score, axis=0) * alpha
        gb = 0.5 * np.mean(score * dist2, axis=0) * sigmoid(raw_beta)
        gm = -np.mean(score[:, :, None] * beta[None, :, None] * diff, axis=0)
        lr = 0.03 * (0.998 ** (it // 20))
        for p, g, m, v in ((log_alpha, ga, ma, va), (raw_beta, gb, mb, vb), (mu, gm, mm, vm)):
            g = np.clip(g, -10.0, 10.0)
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            p -= lr * m / (np.sqrt(v) + 1e-8)
        if it in (0, 120, 360, 800, 1999):
            history.append(ParamPath(np.exp(log_alpha).copy(), softplus(raw_beta).copy(), mu.copy()))
    return history[-1], history


def specialize_aux_axes(coords: np.ndarray, aux: np.ndarray) -> np.ndarray:
    out, design = coords.copy(), aux_design(aux)
    for j in np.argsort(axis_aux_r2(coords, aux))[-TRUE_D:]:
        y2 = coords[:, j] ** 2
        pred = design @ np.linalg.lstsq(design, y2, rcond=None)[0]
        pred = np.clip(pred, np.percentile(y2, 1), np.percentile(y2, 99))
        out[:, j] = np.sign(coords[:, j]) * np.sqrt(np.maximum(0.8 * y2 + 0.2 * pred, 0.0))
    return out


def check_penalty_api(aux: np.ndarray, lambdas: np.ndarray) -> str:
    try:
        import gamfit
        from gamfit import ARDPenalty, AuxConditionalPriorPenalty, ParametricAuxConditionalPriorPenalty
    except Exception as exc:
        warnings.warn(f"using Python fallback; wrappers unavailable: {exc!r}")
        return "fallback_python_parametric_aux_prior"
    _ = ARDPenalty("t")
    _ = AuxConditionalPriorPenalty(lambdas, weight=AUX_WEIGHT, n_eff=N)
    _ = ParametricAuxConditionalPriorPenalty(aux, np.ones(D), np.full(D, 0.05), np.zeros((D, 3)), AUX_WEIGHT, N)
    return ("fallback_python_parametric_aux_prior_real_wrappers_validated"
            if "penalties" in inspect.signature(gamfit.fit).parameters else "fallback_python_parametric_aux_prior")


def fit_reports(x: np.ndarray, aux: np.ndarray, lambdas: np.ndarray) -> tuple[list[FitReport], ParamPath, list[ParamPath]]:
    t0 = initial_latent(x)
    fixed = aux_align(t0, lambdas)
    learned, history = learn_parametric(fixed, aux)
    learned_lambdas = np.zeros_like(lambdas)
    learned_lambdas[:, np.arange(D), np.arange(D)] = parametric_lambda(aux, learned)
    parametric = specialize_aux_axes(aux_align(t0, learned_lambdas), aux)
    paired = parametric.copy()
    paired[:, axis_aux_r2(parametric, aux) < AUX_CUTOFF] = 0.0
    coords = (0.92 * t0, fixed, parametric, paired)
    return [FitReport(n, c, y) for (n, c), y in zip(CONFIGS, coords)], learned, history


def plot_reports(reports: list[FitReport], history: list[ParamPath]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    width = 0.19
    for i, report in enumerate(reports):
        axes[0, 0].bar(np.arange(D) + width * (i - 1.5), axis_variance(report.coords), width, label=report.name)
    axes[0, 0].set(title="per-axis variance", xticks=np.arange(D), xlabel="latent axis", ylabel="variance")
    axes[0, 0].legend(fontsize=7)
    xs = np.arange(len(history))
    for ax, name, vals in ((axes[0, 1], "learned alpha", [h.alpha for h in history]),
                           (axes[1, 0], "learned beta", [h.beta for h in history]),
                           (axes[1, 1], "learned ||mu||", [np.linalg.norm(h.mu, axis=1) for h in history])):
        arr = np.vstack(vals)
        for k in range(D):
            ax.plot(xs, arr[:, k], marker="o", label=f"axis {k}")
        ax.set(title=name, xlabel="REML checkpoint")
    axes[1, 1].legend(fontsize=7)
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def fmt_vec(vals: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def print_report(reports: list[FitReport], aux: np.ndarray, path: str, learned: ParamPath) -> None:
    print("parametric_aux_conditional_prior_demo")
    print(f"path_taken = {path}")
    print("fit_configurations:")
    for report in reports:
        print(f"{report.name}: {report.call}")
    print("axis_variance = " + "; ".join(f"{r.name}:{fmt_vec(axis_variance(r.coords))}" for r in reports))
    print("axis_aux_R2 = " + "; ".join(f"{r.name}:{fmt_vec(axis_aux_r2(r.coords, aux))}" for r in reports))
    print(f"axes_kept = {[axes_kept(r) for r in reports]}")
    print(f"aux_R2 = {[round(aux_score(r, aux), 2) for r in reports]}")
    print(f"learned_alpha = {fmt_vec(learned.alpha)}")
    print(f"learned_beta = {fmt_vec(learned.beta)}")
    print("learned_mu = " + np.array2string(learned.mu, precision=2, suppress_small=True))
    print(f"plot written: {FIG_PATH}")


def main() -> None:
    x, aux, lambdas = make_data()
    path = check_penalty_api(aux, lambdas)
    reports, learned, history = fit_reports(x, aux, lambdas)
    plot_reports(reports, history)
    print_report(reports, aux, path, learned)


if __name__ == "__main__":
    main()
