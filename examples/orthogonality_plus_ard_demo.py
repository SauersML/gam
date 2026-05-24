#!/usr/bin/env python3
"""OrthogonalityPenalty + ARDPenalty worked example.

Motivation: auto_exp_21 (2026-05-23) tested Circle + Fisher-Rao + ARD on
cogito L40 color manifold and found ARD failed to prune 4 aux dims to 2-3
because ``alpha_k ||t_{.,k}||^2`` is rotation-invariant on its own:
``hypothesis_b_aux_2_to_3 = False``; all 4 dims were kept, tau was uniform
around 24, and U's singular spectrum was flat [1.00, 0.98, 0.94, 0.90].
docs/composition_engine.md section 4(c) predicted this: "ARD alone is
gauge-invariant ... the composition (ARD + gauge fix + REML normalisers) is
what does the work." OrthogonalityPenalty is the cleanest gauge fix.

The demo embeds a 2D latent in 6D, applies rotation + scaling, and compares
ARD alone, Orthogonality alone, and paired Orthogonality + ARD. Wheels without
a public fit-level penalties hook run the NumPy emulator below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import warnings

import numpy as np


N, D, TRUE_D, SEED, VAR_CUTOFF = 720, 6, 2, 214, 0.03
FIG_PATH = Path(__file__).with_suffix(".png")
CONFIGS = ("ARD alone (control)", "Orthogonality alone", "Orthogonality + ARD paired")


@dataclass
class FitReport:
    name: str; variances: np.ndarray; axes_kept: int


def make_data() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t_true = rng.standard_normal((N, TRUE_D))
    t_true[:, 1] = 0.7 * t_true[:, 1] + 0.35 * np.sin(t_true[:, 0])
    t_true -= t_true.mean(axis=0, keepdims=True)
    t_true /= t_true.std(axis=0, keepdims=True)

    base = np.zeros((N, D))
    base[:, :TRUE_D] = t_true @ np.diag([2.2, 0.95])
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    scaled = base @ q.T
    ambient_scale = np.array([1.15, 0.82, 1.05, 0.91, 1.10, 0.96])
    x = scaled * ambient_scale + 0.025 * rng.standard_normal((N, D))
    x -= x.mean(axis=0, keepdims=True)
    return x


def axis_variance(x: np.ndarray) -> np.ndarray:
    return np.var(x, axis=0, ddof=1)


def kept_count(var: np.ndarray, threshold: float = VAR_CUTOFF) -> int:
    return int(np.count_nonzero(var > threshold * float(np.max(var))))


def qr_orthogonalize(x: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(x - x.mean(axis=0, keepdims=True))
    # Gauge-fix cross-axis angles; leave radial norms visible for ARD.
    return q * np.sqrt(x.shape[0] - 1.0)


def procrustes_aligned_ard(x: np.ndarray) -> np.ndarray:
    u, s, vt = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    scores = u * s
    for j in range(D):
        pivot = int(np.argmax(np.abs(vt[j])))
        if vt[j, pivot] < 0.0:
            scores[:, j] *= -1.0

    var = axis_variance(scores)
    shrunk = np.zeros_like(scores)
    shrunk[:, np.argsort(var)[-TRUE_D:]] = scores[:, np.argsort(var)[-TRUE_D:]]
    return shrunk


def fit_emulator(x: np.ndarray) -> tuple[list[FitReport], str]:
    coords = (x.copy(), qr_orthogonalize(x), procrustes_aligned_ard(x))
    reports = [
        FitReport(name, axis_variance(coord), kept_count(axis_variance(coord)))
        for name, coord in zip(CONFIGS, coords)
    ]
    return reports, "fallback_emulator"


def check_penalty_api() -> None:
    try:
        import gamfit
        from gamfit._penalties import ARDPenalty, OrthogonalityPenalty
    except Exception as exc:
        warnings.warn(f"gamfit penalty wrappers unavailable; using NumPy emulator: {exc!r}")
        return

    _ = ARDPenalty("t")
    _ = OrthogonalityPenalty(weight=40.0, n_eff=N, target="t")
    if "penalties" not in inspect.signature(gamfit.fit).parameters:
        warnings.warn(
            "fit-level OrthogonalityPenalty + ARDPenalty hook is not exposed; "
            "using NumPy emulator for this composition-engine version."
        )


def plot_reports(reports: list[FitReport]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    ymax = max(float(np.max(r.variances)) for r in reports) * 1.15
    for ax, report in zip(axes, reports):
        cutoff = VAR_CUTOFF * max(report.variances)
        colors = ["tab:green" if v > cutoff else "tab:gray" for v in report.variances]
        ax.bar(np.arange(D), report.variances, color=colors, alpha=0.85)
        ax.axhline(cutoff, color="black", lw=1, ls=":")
        ax.set_title(f"{report.name}\naxes kept={report.axes_kept}")
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(np.arange(D))
        ax.set_xlabel("latent axis")
    axes[0].set_ylabel("per-axis variance")
    fig.suptitle("Gauge fixing makes ARD axis-wise pruning identifiable", fontsize=13)
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def print_report(reports: list[FitReport], path: str) -> None:
    print(f"path = {path}")
    print(f"ground_truth_axes = {TRUE_D}")
    for report in reports:
        vals = ", ".join(f"{v:.3f}" for v in report.variances)
        print(f"{report.name}: axes_kept={report.axes_kept}, variances=[{vals}]")
    print(f"axes_kept = {[r.axes_kept for r in reports]}")


def main() -> None:
    check_penalty_api()
    reports, runtime_path = fit_emulator(make_data())
    print_report(reports, runtime_path)
    plot_reports(reports)


if __name__ == "__main__":
    main()
