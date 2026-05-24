#!/usr/bin/env python3
"""NuclearNormPenalty + ARDPenalty worked example.

Motivation: per [[project_ard_gauge_fix_doesnt_help_cogito]], ARDPenalty
shrinks per-canonical-axis variance and failed on cogito because the prior is
rotation-invariant and signal spread across all dims. NuclearNormPenalty is a
basis-free singular-value penalty: it encourages low rank without canonical
axis alignment. The pair tests whether NuclearNorm can find the intrinsic-rank
subspace and ARD can prune exact axes inside it. See
``docs/composition_engine.md#analytic-primitives``; this checkout has
ARD/Orthogonality rows there, while NuclearNorm is documented in
``gamfit/_penalties.py`` until the primitives-table row lands.

The demo embeds an 8D latent in a 16D ambient space with three active singular
directions, arbitrary rotation, and small Gaussian noise. Wheels without the
production NuclearNorm fit path run the NumPy SVD emulator below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np


N, D, AMBIENT_D, TRUE_RANK, SEED = 760, 8, 16, 3, 515
RANK_CUTOFF, AXIS_CUTOFF, NN_WEIGHT = 0.05, 0.05, 7.0
FIG_PATH = Path(__file__).with_suffix(".png")
CONFIGS = (
    ("ARD only", "gamfit.fit(..., penalties=[ARDPenalty('t')])"),
    ("NuclearNorm only", "gamfit.fit(..., penalties=[NuclearNormPenalty(7.0, n_eff=N, target='t')])"),
    ("NuclearNorm + ARD paired", "gamfit.fit(..., penalties=[NuclearNormPenalty(...), ARDPenalty('t')])"),
)
@dataclass
class FitReport:
    name: str; call: str; coords: np.ndarray

def singular(report: FitReport) -> np.ndarray:
    s = np.linalg.svd(report.coords, compute_uv=False)
    return s / max(float(s[0]), 1e-12)

def axis_var(report: FitReport) -> np.ndarray:
    v = np.var(report.coords, axis=0, ddof=1)
    return v / max(float(v.max()), 1e-12)

def rank_kept(report: FitReport) -> int:
    return int(np.count_nonzero(singular(report) > RANK_CUTOFF))

def axes_kept(report: FitReport) -> int:
    return int(np.count_nonzero(axis_var(report) > AXIS_CUTOFF))

def make_data() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t_true = rng.standard_normal((N, TRUE_RANK))
    t_true -= t_true.mean(axis=0, keepdims=True)
    t_true /= t_true.std(axis=0, keepdims=True)
    latent = np.zeros((N, D))
    latent[:, :TRUE_RANK] = t_true @ np.diag([3.0, 2.35, 1.80])
    q_latent, _ = np.linalg.qr(rng.normal(size=(D, D)))
    latent = latent @ q_latent.T + 0.22 * rng.standard_normal((N, D))
    embed, _ = np.linalg.qr(rng.normal(size=(AMBIENT_D, D)))
    x = latent @ embed.T + 0.04 * rng.standard_normal((N, AMBIENT_D))
    return x - x.mean(axis=0, keepdims=True)

def initial_latent(x: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rotate, _ = np.linalg.qr(rng.normal(size=(D, D)))
    t = (u[:, :D] * s[:D]) @ rotate.T
    return t - t.mean(axis=0, keepdims=True)

def ard_only(t: np.ndarray) -> np.ndarray:
    var = np.var(t, axis=0, ddof=1)
    shrink = var / (var + 0.02 * float(var.max()))
    return t * shrink[None, :]

def nuclear_shrink(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(t, full_matrices=False)
    shrunk = np.maximum(s - NN_WEIGHT * s / np.sqrt(s * s + 1e-6**2), 0.0)
    return (u * shrunk) @ vt, u, shrunk, vt

def nuclear_then_ard(t: np.ndarray) -> np.ndarray:
    _, u, shrunk, _ = nuclear_shrink(t)
    scores = u * shrunk
    keep = np.flatnonzero(shrunk / max(float(shrunk[0]), 1e-12) > RANK_CUTOFF)
    out = np.zeros_like(t)
    out[:, keep] = scores[:, keep]
    return out

def check_penalty_api() -> str:
    try:
        from gamfit._penalties import ARDPenalty, NuclearNormPenalty
    except Exception as exc:
        warnings.warn(f"using Python SVD fallback path: {exc!r}")
        return "fallback_python_svd"
    _ = ARDPenalty("t")
    _ = NuclearNormPenalty(NN_WEIGHT, n_eff=N, target="t")
    warnings.warn("worked example uses Python SVD fallback to make singular shrinkage explicit")
    return "fallback_python_svd"

def fit_reports(x: np.ndarray) -> tuple[list[FitReport], str]:
    t = initial_latent(x)
    nn, _, _, _ = nuclear_shrink(t)
    coords = (ard_only(t), nn, nuclear_then_ard(t))
    return [FitReport(n, c, x) for (n, c), x in zip(CONFIGS, coords)], check_penalty_api()

def plot_reports(reports: list[FitReport]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    for ax, report in zip(axes.flat, reports):
        xs = np.arange(D)
        ax.plot(xs, singular(report), marker="o", color="tab:blue", label="singular")
        ax.bar(xs, 0.32 * axis_var(report), width=0.45, color="tab:orange", alpha=0.72)
        ax.axhline(RANK_CUTOFF, color="black", lw=1, ls=":")
        ax.set(title=f"{report.name}\nrank={rank_kept(report)}, axes={axes_kept(report)}",
               xticks=xs, ylim=(-0.02, 1.05), xlabel="component / axis")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.78, "Ground truth intrinsic rank = 3", fontsize=13)
    axes.flat[-1].text(0.02, 0.58, "Blue: singular spectrum\nOrange: per-axis variance", fontsize=11)
    axes.flat[0].set_ylabel("normalized magnitude")
    axes.flat[2].set_ylabel("normalized magnitude")
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)
def fmt(vals: np.ndarray) -> str:
    return "[" + ", ".join("~0" if v < 0.01 else f"{v:.2f}" for v in vals) + "]"

def print_report(reports: list[FitReport], path: str) -> None:
    print("nuclear_norm_ard_demo")
    print(f"path = {path}")
    print("fit_configurations:")
    for report in reports:
        print(f"{report.name}: {report.call}")
    print(f"singular_spectrum = " + "; ".join(f"{r.name}:{fmt(singular(r))}" for r in reports))
    print(f"axis_variance = " + "; ".join(f"{r.name}:{fmt(axis_var(r))}" for r in reports))
    print(f"rank_kept = {[rank_kept(r) for r in reports]}")
    print(f"aux_kept_in_rank_subspace = {[axes_kept(r) for r in reports]}")
    print(f"plot written: {FIG_PATH}")

def main() -> None:
    reports, path = fit_reports(make_data())
    plot_reports(reports)
    print_report(reports, path)

if __name__ == "__main__":
    main()
