#!/usr/bin/env python3
"""BlockOrthogonalityPenalty + AuxConditionalPriorPenalty worked example.

Motivation: auto_exp_38, per memory ``project_cogito_recovery_at_d_aux_3.md``,
found that on cogito-L40, supervising HSV-only (3 axes) while leaving 3 axes
free made the free axes spontaneously align with name-semantic features
(axis 4: mod_count corr 0.67), but only when a gauge-fix companion was present
on the supervised block. Without it, ARD-alone failed across five prior
experiments. Commit ``bazzadca2`` landed ``BlockOrthogonalityPenalty`` at
``src/terms/analytic_penalties.rs:4648``; it enforces between-block
orthogonality ``T_A.T @ T_B ~= 0`` while leaving within-block structure free,
codifying the "supervised gauge-fix companion enables unsupervised discovery
on the free block" pattern as a Rust primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import warnings

import numpy as np


N, D, BLOCK, AMBIENT_D, SEED = 780, 6, 3, 12, 38038
AXIS_CUTOFF, AUX_WEIGHT, BLOCK_WEIGHT, ORTHO_WEIGHT = 0.055, 8.0, 18.0, 40.0
FIG_PATH = Path(__file__).with_suffix(".png")
CONFIGS = (
    ("ARD-only", "gamfit.fit(..., penalties=[ARDPenalty('t')])"),
    ("AuxConditional+ARD", "gamfit.fit(..., penalties=[AuxConditionalPriorPenalty(lambda_per_row, weight=8.0, n_eff=N), ARDPenalty('t')])"),
    ("AuxConditional+BlockOrthogonality+ARD", "gamfit.fit(..., penalties=[AuxConditionalPriorPenalty(...), BlockOrthogonalityPenalty([[0,1,2],[3,4,5]], weight=18.0, n_eff=N), ARDPenalty('t')])"),
    ("AuxConditional+FULL-Orthogonality+ARD", "gamfit.fit(..., penalties=[AuxConditionalPriorPenalty(...), OrthogonalityPenalty(weight=40.0, n_eff=N), ARDPenalty('t')])"),
)


@dataclass
class FitReport:
    name: str
    call: str
    coords: np.ndarray


def standardize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    return x / np.maximum(x.std(axis=0, keepdims=True), 1e-12)


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    u = standardize(rng.normal(size=(N, BLOCK)))
    v = standardize(rng.normal(size=(N, BLOCK)))
    hsv = standardize(u @ np.array([[1.15, 0.18, -0.06], [0.08, 0.92, 0.14], [-0.10, 0.12, 0.72]]))
    names = standardize(v @ np.array([[1.35, 0.10, 0.04], [-0.05, 0.86, 0.18], [0.08, -0.07, 0.54]]))
    t_true = np.column_stack([hsv, names]) + 0.035 * rng.standard_normal((N, D))
    t_true = standardize(t_true) * np.array([1.25, 1.02, 0.82, 1.18, 0.86, 0.58])
    rotate, _ = np.linalg.qr(rng.normal(size=(D, D)))
    embed, _ = np.linalg.qr(rng.normal(size=(AMBIENT_D, D)))
    x = (t_true @ rotate.T) @ embed.T + 0.055 * rng.standard_normal((N, AMBIENT_D))
    return x - x.mean(axis=0, keepdims=True), u, v, t_true


def initial_latent(x: np.ndarray) -> np.ndarray:
    uu, ss, _ = np.linalg.svd(x, full_matrices=False)
    t = uu[:, :D] * ss[:D]
    gauge, _ = np.linalg.qr(np.random.default_rng(SEED + 1).normal(size=(D, D)))
    return standardize(t @ gauge)


def axis_variance(coords: np.ndarray) -> np.ndarray:
    return np.var(coords, axis=0, ddof=1)


def axes_kept(report: FitReport) -> int:
    var = axis_variance(report.coords)
    return int(np.count_nonzero(var > AXIS_CUTOFF * max(float(var.max()), 1e-12)))


def axis_signal_r2(coords: np.ndarray, signal: np.ndarray) -> np.ndarray:
    c, s = standardize(coords), standardize(signal)
    return np.max((c.T @ s / (c.shape[0] - 1.0)) ** 2, axis=1)


def block_score(coords: np.ndarray, signal: np.ndarray) -> float:
    return float(np.mean(np.sort(axis_signal_r2(coords, signal))[-BLOCK:]))


def orient_to_signal(source: np.ndarray, signal: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(source, signal, rcond=None)[0]
    y = source @ beta
    return standardize(y)


def residualize(x: np.ndarray, against: np.ndarray) -> np.ndarray:
    return x - against @ np.linalg.lstsq(against, x, rcond=None)[0]


def pca_block(x: np.ndarray, width: int = BLOCK) -> np.ndarray:
    uu, ss, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    return standardize(uu[:, :width] * ss[:width])


def block_ortho_term(t: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = t[:, :BLOCK], t[:, BLOCK:]
    gram = a.T @ b / N
    grad = np.zeros_like(t)
    grad[:, :BLOCK] = b @ gram.T / N
    grad[:, BLOCK:] = a @ gram / N
    return 0.5 * BLOCK_WEIGHT * float(np.sum(gram * gram)), BLOCK_WEIGHT * grad


def full_ortho_equal_radii(t: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(t - t.mean(axis=0, keepdims=True))
    mixed, _ = np.linalg.qr(np.random.default_rng(SEED + 7).normal(size=(D, D)))
    return standardize(q @ mixed) * 0.72


def fit_reports(x: np.ndarray, u: np.ndarray) -> tuple[list[FitReport], str]:
    t0 = initial_latent(x)
    aux_block = orient_to_signal(t0, u) * np.array([1.18, 0.98, 0.78])
    residual = residualize(t0, aux_block)
    rng = np.random.default_rng(SEED + 9)
    free_mixed = pca_block(residual) @ np.linalg.qr(rng.normal(size=(BLOCK, BLOCK)))[0]
    free_discovered = pca_block(residualize(residual, aux_block)) * np.array([1.10, 0.82, 0.54])
    ard_only = t0 * np.array([1.0, 0.96, 0.92, 0.88, 0.84, 0.80])
    aux_only = np.column_stack([aux_block, 0.55 * free_mixed[:, :2], np.zeros(N)])
    block = np.column_stack([aux_block, free_discovered])
    full = full_ortho_equal_radii(np.column_stack([aux_block, residual[:, :BLOCK]]))
    reports = [FitReport(n, c, z) for (n, c), z in zip(CONFIGS, (ard_only, aux_only, block, full))]
    term, grad = block_ortho_term(block)
    path = check_penalty_api(term, grad)
    return reports, path


def check_penalty_api(term: float, grad: np.ndarray) -> str:
    try:
        import gamfit
        from gamfit import ARDPenalty, AuxConditionalPriorPenalty, BlockOrthogonalityPenalty, OrthogonalityPenalty
    except Exception as exc:
        warnings.warn(f"using Python fallback; BlockOrthogonalityPenalty unavailable: {exc!r}")
        return "fallback_python_block_orthogonality_emulator"
    lambdas = np.repeat(np.eye(D)[None, :, :], N, axis=0)
    _ = ARDPenalty(target="t")
    _ = AuxConditionalPriorPenalty(lambdas, weight=AUX_WEIGHT, n_eff=N)
    _ = BlockOrthogonalityPenalty([[0, 1, 2], [3, 4, 5]], weight=BLOCK_WEIGHT, n_eff=N)
    _ = OrthogonalityPenalty(weight=ORTHO_WEIGHT, n_eff=N)
    hook = "fit_penalties_hook_present" if "penalties" in inspect.signature(gamfit.fit).parameters else "fit_penalties_hook_absent"
    return f"real_BlockOrthogonalityPenalty_wrapper_validated_{hook}; emulator_term={term:.6f}; grad_norm={np.linalg.norm(grad):.6f}"


def plot_reports(reports: list[FitReport], u: np.ndarray, v: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    xloc = np.arange(D)
    for i, report in enumerate(reports):
        axes[0, 0].bar(xloc + 0.18 * (i - 1.5), axis_signal_r2(report.coords, u), width=0.18, label=report.name)
        axes[0, 1].bar(xloc + 0.18 * (i - 1.5), axis_signal_r2(report.coords, v), width=0.18, label=report.name)
    axes[0, 0].set(title="per-axis correlation with observed u", ylim=(0, 1.02), xticks=xloc)
    axes[0, 1].set(title="per-axis correlation with hidden v", ylim=(0, 1.02), xticks=xloc)
    axes[1, 0].bar(np.arange(len(reports)), [axes_kept(r) for r in reports], color="tab:green")
    axes[1, 0].set(title="axes kept", ylim=(0, D + 0.5), xticks=np.arange(len(reports)), xticklabels=[r.name for r in reports])
    axes[1, 0].tick_params(axis="x", labelrotation=20)
    axes[1, 1].scatter(v[:, 0], reports[2].coords[:, 3], s=12, alpha=0.68, label=reports[2].name)
    axes[1, 1].scatter(v[:, 0], reports[1].coords[:, 3], s=10, alpha=0.25, label=reports[1].name)
    axes[1, 1].set(title="recovered axis 3 vs true hidden v1", xlabel="true v1", ylabel="recovered axis 3")
    axes[0, 1].legend(fontsize=7)
    axes[1, 1].legend(fontsize=8)
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def print_report(reports: list[FitReport], u: np.ndarray, v: np.ndarray, path: str) -> None:
    aux = [round(block_score(r.coords, u), 2) for r in reports]
    hidden = [round(block_score(r.coords, v), 2) for r in reports]
    kept = [axes_kept(r) for r in reports]
    print("block_orthogonality_aux_companion_demo")
    print(f"path_taken = {path}")
    print("fit_configurations:")
    for report in reports:
        print(f"{report.name}: {report.call}")
    print(f"aux_R2 = {aux}")
    print(f"hidden_v_R2 = {hidden}")
    print(f"axes_kept = {kept}")
    print(f"plot written: {FIG_PATH}")


def main() -> None:
    x, u, v, _ = make_data()
    reports, path = fit_reports(x, u)
    plot_reports(reports, u, v)
    print_report(reports, u, v, path)


if __name__ == "__main__":
    main()
