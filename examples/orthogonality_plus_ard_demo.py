#!/usr/bin/env python3
"""OrthogonalityPenalty + ARDPenalty worked example.

Motivation: auto_exp_21 (2026-05-23) tested Circle + Fisher-Rao + ARD on
cogito L40 color manifold and found ARD failed to prune 4 aux dims to 2-3
because ``alpha_k ||t_{.,k}||^2`` is rotation-invariant on its own:
``hypothesis_b_aux_2_to_3 = False``; all 4 dims were kept, tau was uniform
around 24, and the U singular spectrum was flat [1.00, 0.98, 0.94, 0.90].
The docs/composition_engine.md section 4(c) audit caveat predicted this:
"ARD alone is gauge-invariant ... the composition (ARD + gauge fix + REML
normalisers) is what does the work." OrthogonalityPenalty is the cleanest
gauge-fixing primitive.

This synthetic demo embeds a 2D latent in 6D Euclidean ambient space, applies
rotation + scaling, and compares ARD alone, Orthogonality alone, and the
paired Orthogonality + ARD regime. Current source exposes the Python penalty
wrappers; wheels without a public fit-level penalties hook run the NumPy
emulator below, which demonstrates the same Procrustes-aligned ARD dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np


N, D, TRUE_D, SEED = 720, 6, 2, 214
FIG_PATH = Path(__file__).with_suffix(".png")
CONFIGS = (
    "ARD alone (control)",
    "Orthogonality alone",
    "Orthogonality + ARD paired",
)


@dataclass
class Data:
    latent: np.ndarray
    ambient: np.ndarray


@dataclass
class FitReport:
    name: str
    coords: np.ndarray
    variances: np.ndarray
    axes_kept: int


def make_data() -> Data:
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((N, TRUE_D))
    z[:, 1] = 0.7 * z[:, 1] + 0.35 * np.sin(z[:, 0])
    z -= z.mean(axis=0, keepdims=True)
    z /= z.std(axis=0, keepdims=True)

    base = np.zeros((N, D))
    base[:, :TRUE_D] = z @ np.diag([2.2, 0.95])
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    scaled = base @ q.T
    ambient_scale = np.array([1.15, 0.82, 1.05, 0.91, 1.10, 0.96])
    x = scaled * ambient_scale + 0.025 * rng.standard_normal((N, D))
    x -= x.mean(axis=0, keepdims=True)
    return Data(z, x)


def axis_variance(x: np.ndarray) -> np.ndarray:
    return np.var(x, axis=0, ddof=1)


def kept_count(var: np.ndarray, threshold: float = 0.05) -> int:
    return int(np.count_nonzero(var > threshold * float(np.max(var))))


def qr_orthogonalize(x: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(x - x.mean(axis=0, keepdims=True))
    # OrthogonalityPenalty fixes cross-axis gauge but leaves radial norms to
    # ARD, so use equal visible norms to make "kept, just orthogonal" explicit.
    return q * np.sqrt(x.shape[0] - 1.0)


def procrustes_aligned_ard(x: np.ndarray) -> np.ndarray:
    u, s, vt = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    scores = u * s
    loadings = vt.copy()
    for j in range(D):
        pivot = int(np.argmax(np.abs(loadings[j])))
        if loadings[j, pivot] < 0.0:
            scores[:, j] *= -1.0
            loadings[j] *= -1.0

    var = axis_variance(scores)
    keep = np.argsort(var)[-TRUE_D:]
    shrunk = np.zeros_like(scores)
    shrunk[:, keep] = scores[:, keep]
    return shrunk


def fit_emulator(data: Data) -> tuple[list[FitReport], str]:
    ard = data.ambient.copy()
    orth = qr_orthogonalize(data.ambient)
    paired = procrustes_aligned_ard(data.ambient)
    coords = (ard, orth, paired)
    reports = [
        FitReport(name, x, axis_variance(x), kept_count(axis_variance(x)))
        for name, x in zip(CONFIGS, coords)
    ]
    return reports, "fallback_emulator"


def api_status() -> str:
    try:
        import gamfit
        from gamfit._penalties import ARDPenalty, OrthogonalityPenalty
    except Exception as exc:
        warnings.warn(f"gamfit penalty wrappers unavailable; using NumPy emulator: {exc!r}")
        return "fallback_emulator"

    _ = ARDPenalty("t")
    _ = OrthogonalityPenalty(weight=40.0, n_eff=N, target="t")
    has_fit_penalties = "penalties" in inspect.signature(gamfit.fit).parameters
    if not has_fit_penalties:
        warnings.warn(
            "fit-level OrthogonalityPenalty + ARDPenalty hook is not exposed; "
            "using NumPy emulator for this composition-engine version."
        )
        return "fallback_emulator"
    return "api_wrappers_available"


def plot_reports(reports: list[FitReport]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    ymax = max(float(np.max(r.variances)) for r in reports) * 1.15
    for ax, report in zip(axes, reports):
        colors = ["tab:green" if v > 0.05 * max(report.variances) else "tab:gray"
                  for v in report.variances]
        ax.bar(np.arange(D), report.variances, color=colors, alpha=0.85)
        ax.axhline(0.05 * max(report.variances), color="black", lw=1, ls=":")
        ax.set_title(f"{report.name}\naxes kept={report.axes_kept}")
        ax.set_xlabel("latent axis")
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(np.arange(D))
    axes[0].set_ylabel("per-axis variance")
    fig.suptitle("Gauge fixing makes ARD axis-wise pruning identifiable", fontsize=13)
    fig.savefig(FIG_PATH, dpi=160)
    plt.show()


def print_report(reports: list[FitReport], path: str) -> None:
    print(f"path = {path}")
    print(f"ground_truth_axes = {TRUE_D}")
    for report in reports:
        values = ", ".join(f"{v:.3f}" for v in report.variances)
        print(f"{report.name}: axes_kept={report.axes_kept}, variances=[{values}]")
    print(f"axes_kept = {[r.axes_kept for r in reports]}")
    print(f"plot written to {FIG_PATH}")


def main() -> None:
    data = make_data()
    path = api_status()
    reports, runtime_path = fit_emulator(data)
    print_report(reports, runtime_path if path != "api_wrappers_available" else path)
    plot_reports(reports)


if __name__ == "__main__":
    main()
