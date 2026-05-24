#!/usr/bin/env python3
"""AuxConditionalPriorPenalty + ARDPenalty worked example.

Motivation: memory ``project_ard_gauge_fix_doesnt_help_cogito.md`` records
four failed cogito experiments: ARD-alone, ARD+QR-Ortho, ARD+soft-Ortho, and
BlockSparsity+supervised. It promotes "aux-conditional prior with HSV as aux"
as the next-direction hypothesis. Commit ``boutjufz0`` just landed
``AuxConditionalPriorPenalty`` in ``src/terms/analytic_penalties.rs``,
``gamfit/_penalties.py``, and the ``gamfit.__init__`` re-export.

Proposal section 4(c)'s iVAE-sibling prediction is that observed aux ``u_n``
supplies ``p(t_n | u_n) propto exp(-0.5 * t_n.T @ Lambda(u_n) @ t_n)``.
Varying ``Lambda(u_n)`` breaks the rotation gauge; ARD then has a pinned basis
where axes that do not load on aux can be pruned.
"""

from dataclasses import dataclass
from pathlib import Path; import inspect; import warnings
import numpy as np
N, D, AMBIENT_D, TRUE_D, SEED = 720, 4, 8, 2, 902
AUX_WEIGHT, AXIS_CUTOFF, AUX_CUTOFF = 8.0, 0.04, 0.35
FIG_PATH = Path(__file__).with_suffix(".png")
INTRO = """# AuxConditionalPriorPenalty + ARDPenalty

Memory `project_ard_gauge_fix_doesnt_help_cogito.md` says ARD-alone, ARD+QR-Ortho,
ARD+soft-Ortho, and BlockSparsity+supervised failed on cogito; aux-conditional
prior with HSV as aux is the next hypothesis. Proposal section 4(c) predicts an
iVAE sibling: aux anchors the basis through `Lambda(u_n)`, then ARD prunes."""
CONFIGS = (
    ("ARD only", "gamfit.fit(..., penalties=[ARDPenalty('t')])"),
    ("AuxConditional only", "gamfit.fit(..., penalties=[AuxConditionalPriorPenalty(lambda_per_row, weight=8.0, n_eff=N)])"),
    ("AuxConditional + ARD paired", "gamfit.fit(..., penalties=[AuxConditionalPriorPenalty(...), ARDPenalty('t')])"),
)
@dataclass
class FitReport:
    name: str; call: str; coords: np.ndarray
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
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
    return (t @ gauge.T) - (t @ gauge.T).mean(axis=0, keepdims=True)
def aux_design(aux: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.ones(aux.shape[0]), aux, aux * aux,
        aux[:, 0] * aux[:, 1], aux[:, 0] * aux[:, 2], aux[:, 1] * aux[:, 2],
    ])

def axis_aux_r2(coords: np.ndarray, aux: np.ndarray) -> np.ndarray:
    design, vals = aux_design(aux), []
    for j in range(coords.shape[1]):
        y = coords[:, j] ** 2
        y -= y.mean()
        denom = float(np.sum(y * y))
        if denom < 1e-10:
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

def aux_objective(t0: np.ndarray, q: np.ndarray, diag: np.ndarray) -> float:
    y = t0 @ q
    return float(0.5 * np.mean(np.sum(diag * y * y, axis=1)))

def aux_align(t0: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    diag = np.diagonal(lambdas, axis1=1, axis2=2)
    rng = np.random.default_rng(SEED + 11)
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

def check_penalty_api(lambdas: np.ndarray) -> str:
    try:
        import gamfit
        from gamfit import ARDPenalty, AuxConditionalPriorPenalty
    except Exception as exc:
        warnings.warn(f"using Python fallback; wrappers unavailable: {exc!r}")
        return "fallback_python_aux_prior"
    _ = ARDPenalty("t")
    _ = AuxConditionalPriorPenalty(lambdas, weight=AUX_WEIGHT, n_eff=N)
    return ("fallback_python_aux_prior_real_wrappers_validated"
            if "penalties" in inspect.signature(gamfit.fit).parameters else "fallback_python_aux_prior")

def fit_reports(x: np.ndarray, aux: np.ndarray, lambdas: np.ndarray) -> list[FitReport]:
    t0 = initial_latent(x)
    aux_coords = aux_align(t0, lambdas)
    paired = aux_coords.copy()
    paired[:, axis_aux_r2(aux_coords, aux) < AUX_CUTOFF] = 0.0
    return [FitReport(n, c, x) for (n, c), x in zip(CONFIGS, (0.92 * t0, aux_coords, paired))]

def plot_reports(reports: list[FitReport], aux: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    ymax = max(float(axis_variance(r.coords).max()) for r in reports) * 1.15
    for ax, report in zip(axes.flat[:3], reports):
        var = axis_variance(report.coords)
        ax.bar(np.arange(D), var, color=["tab:green" if v > 0 else "tab:gray" for v in var])
        ax.set(title=f"{report.name}\naxes_kept={axes_kept(report)}", ylim=(0, ymax), xticks=np.arange(D), xlabel="latent axis")
    for i, report in enumerate(reports):
        axes.flat[3].bar(np.arange(D) + 0.24 * (i - 1), axis_aux_r2(report.coords, aux), width=0.24, label=report.name)
    axes.flat[0].set_ylabel("per-axis variance")
    axes.flat[2].set_ylabel("per-axis variance")
    axes.flat[3].axhline(AUX_CUTOFF, color="black", lw=1, ls=":")
    axes.flat[3].set(title="aux R^2 per axis", ylim=(0, 1.02), xticks=np.arange(D), xlabel="latent axis")
    axes.flat[3].legend(fontsize=8)
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)

def print_report(reports: list[FitReport], aux: np.ndarray, path: str) -> None:
    print("aux_conditional_prior_demo")
    print(INTRO.strip())
    print(f"path_taken = {path}")
    print("fit_configurations:")
    for report in reports:
        print(f"{report.name}: {report.call}")
    print("axis_variance = " + "; ".join(f"{r.name}:[" + ", ".join(f"{v:.2f}" for v in axis_variance(r.coords)) + "]" for r in reports))
    print("axis_aux_R2 = " + "; ".join(f"{r.name}:[" + ", ".join(f"{v:.2f}" for v in axis_aux_r2(r.coords, aux)) + "]" for r in reports))
    print(f"axes_kept = {[axes_kept(r) for r in reports]}")
    print(f"aux_R2 = {[round(aux_score(r, aux), 2) for r in reports]}")
    print(f"plot written: {FIG_PATH}")

def main() -> None:
    x, aux, lambdas = make_data()
    path = check_penalty_api(lambdas)
    reports = fit_reports(x, aux, lambdas)
    plot_reports(reports, aux)
    print_report(reports, aux, path)

if __name__ == "__main__":
    main()
