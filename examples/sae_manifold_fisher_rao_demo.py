"""SAE-manifold + Fisher-Rao W worked example.

This demo composes two new primitives:

1. ``gamfit.sae_manifold_fit(..., assignment_prior="ibp_map", ...)``: an
   IBP-style per-row binary active set, following the Infinite GPFA / IBP-GPFA
   literature on latent feature assignments (Doshi-Velez et al., 2009; Rai and
   Daume, 2009; Knowles and Ghahramani, 2011).
2. ``fisher_W`` / ``fisher_w`` with shape ``(N, p, p)``: dense per-row output
   Fisher blocks, following the Fisher-Rao behavioral-metric framing where two
   directions are close when they have similar local output/logit effects.

The audit-revised gauge story is that IBP fixes "which atoms are active for
this row" while Fisher-Rao W fixes "which output directions are steerable".
Composed, they recover behaviorally meaningful manifolds, not just tidy
Euclidean geometry.

Run with a wheel exposing the high-level Fisher hook:

    python sae_manifold_fisher_rao_demo.py

Older wheels degrade cleanly: the plain IBP fit runs if available, and the
Fisher comparison row is a null placeholder with a clear "needs v0.1.115"
message.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gamfit
except Exception as exc:  # pragma: no cover - example fallback path
    gamfit = None
    GAMFIT_IMPORT_ERROR: Exception | None = exc
else:
    GAMFIT_IMPORT_ERROR = None


@dataclass
class Data:
    z: np.ndarray
    fisher_w: np.ndarray
    labels: np.ndarray


@dataclass
class Report:
    name: str
    fit: Any | None
    ok: bool
    message: str
    mean_active: float = math.nan
    heldout_r2: float = math.nan
    reml_score: float = math.nan
    topologies: list[str] | None = None


def make_data(n: int = 600, p: int = 16, fisher_alpha: float = 18.0, seed: int = 73) -> Data:
    """Half circle rows, half sphere rows, with both features mixed into all p dims."""

    rng = np.random.default_rng(seed)
    n_each = n // 2

    theta = np.linspace(0.0, 2.0 * np.pi, n_each, endpoint=False)
    theta += rng.normal(scale=0.015, size=n_each)
    circle_latent = np.stack(
        [np.sin(theta), np.cos(theta), 0.45 * np.sin(2 * theta), 0.45 * np.cos(2 * theta)],
        axis=1,
    )
    circle_dlatent = np.stack(
        [np.cos(theta), -np.sin(theta), 0.90 * np.cos(2 * theta), -0.90 * np.sin(2 * theta)],
        axis=1,
    )

    sphere_xyz = rng.normal(size=(n_each, 3))
    sphere_xyz /= np.maximum(np.linalg.norm(sphere_xyz, axis=1, keepdims=True), 1e-12)
    x, y, z = sphere_xyz.T
    sphere_latent = np.stack([x, y, z, x * y, y * z, x * z], axis=1)
    sphere_behavior = np.stack([x, y, z, 0.5 * y, 0.5 * x, 0.25 * z], axis=1)

    circle_mix = rng.normal(size=(circle_latent.shape[1], p))
    sphere_mix = rng.normal(size=(sphere_latent.shape[1], p))
    circle_mix /= np.maximum(np.linalg.norm(circle_mix, axis=1, keepdims=True), 1e-12)
    sphere_mix /= np.maximum(np.linalg.norm(sphere_mix, axis=1, keepdims=True), 1e-12)

    circle_signal = circle_latent @ circle_mix
    sphere_signal = sphere_latent @ sphere_mix

    # Additive overlap: one dominant manifold plus a weak coherent trace from
    # the other manifold. This is where Gaussian geometry tends to smear atoms.
    z_circle = circle_signal + 0.18 * sphere_signal[rng.permutation(n_each)]
    z_sphere = sphere_signal + 0.18 * circle_signal[rng.permutation(n_each)]
    observed = np.vstack([z_circle, z_sphere])
    observed += 0.035 * rng.normal(size=observed.shape)
    observed -= observed.mean(axis=0, keepdims=True)

    # Synthetic "local logit-gradient": the behavior-sensitive direction for
    # each row. W[n] = I + alpha grad_n grad_n^T is dense and per-row.
    circle_grad = circle_dlatent @ circle_mix
    sphere_grad = sphere_behavior @ sphere_mix
    grad = np.vstack([circle_grad, sphere_grad])
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    fisher_w = np.eye(p)[None, :, :] + fisher_alpha * grad[:, :, None] * grad[:, None, :]
    labels = np.r_[np.zeros(n_each, dtype=int), np.ones(n_each, dtype=int)]
    return Data(observed, fisher_w, labels)


def split_rows(n: int, test_frac: float = 0.2, seed: int = 29) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_frac * n))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def sae_fit(z_train: np.ndarray, fisher_w: np.ndarray | None, seed: int) -> tuple[Any | None, str]:
    """Use fisher_W/fisher_w when the installed high-level API exposes it."""

    if gamfit is None:
        return None, f"gamfit import failed: {GAMFIT_IMPORT_ERROR!r}"
    if not hasattr(gamfit, "sae_manifold_fit"):
        return None, "gamfit.sae_manifold_fit is missing; needs gamfit v0.1.115+"

    kwargs: dict[str, Any] = dict(
        n_atoms=4,
        atom_basis=["circle", "sphere", "duchon", "duchon"],
        atom_dim=[1, 2, 2, 2],
        assignment_prior="ibp_map",
        alpha="auto",
        tau=0.45,
        smoothness="auto",
        max_iter=14,
        learning_rate=0.035,
        random_state=seed,
    )
    if fisher_w is not None:
        try:
            params = inspect.signature(gamfit.sae_manifold_fit).parameters
        except (TypeError, ValueError):
            params = {}
        if "fisher_W" in params:
            kwargs["fisher_W"] = fisher_w
        elif "fisher_w" in params:
            kwargs["fisher_w"] = fisher_w
        else:
            return None, "sae_manifold_fit lacks fisher_W/fisher_w; needs gamfit v0.1.115+"

    try:
        return gamfit.sae_manifold_fit(z_train, **kwargs), "ok"
    except TypeError as exc:
        if fisher_w is not None:
            return None, f"fisher_W/fisher_w call failed ({exc}); needs gamfit v0.1.115+"
        return None, f"plain sae_manifold_fit call failed: {exc}"
    except Exception as exc:
        return None, f"sae_manifold_fit failed: {exc!r}"


def active_counts(fit: Any) -> np.ndarray:
    return np.sum(np.asarray(fit.assignments, dtype=float) > 0.5, axis=1)


def atom_topologies(fit: Any) -> list[str]:
    """Infer Circle/Sphere/Plane from each atom's fitted LatentCoord values."""

    out: list[str] = []
    assignments = np.asarray(fit.assignments, dtype=float)
    for atom_idx, atom in enumerate(fit.atoms):
        coords = np.asarray(atom.coords, dtype=float)
        weights = assignments[:, atom_idx]
        active = weights > max(0.5, np.quantile(weights, 0.75))
        if coords.shape[1] == 0 or np.count_nonzero(active) < 8:
            out.append("Plane")
            continue
        local = coords[active].copy()
        local -= local.mean(axis=0, keepdims=True)
        _, svals, _ = np.linalg.svd(local, full_matrices=False)
        eig = svals**2 / max(local.shape[0] - 1, 1)
        rel = eig / max(float(eig[0]), 1e-12)
        dim = int(np.sum(rel > 0.08))
        out.append("Circle" if dim <= 1 else "Sphere" if dim == 2 and rel[1] > 0.25 else "Plane")
    return out


def heldout_anchor_r2(
    fit: Any,
    z_train: np.ndarray,
    z_test: np.ndarray,
    fisher_w_test: np.ndarray | None,
) -> float:
    """Nearest-anchor R2 because current SAE-manifold fits are row-coordinate fits."""

    diff = z_test[:, None, :] - z_train[None, :, :]
    if fisher_w_test is None:
        dist = np.sum(diff**2, axis=2)
    else:
        dist = np.einsum("tnp,tpq,tnq->tn", diff, fisher_w_test, diff)
    pred = np.asarray(fit.fitted, dtype=float)[np.argmin(dist, axis=1)]
    ss_res = float(np.sum((z_test - pred) ** 2))
    ss_tot = float(np.sum((z_test - z_test.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def summarize(
    name: str,
    fit: Any | None,
    message: str,
    z_train: np.ndarray,
    z_test: np.ndarray,
    fisher_w_test: np.ndarray | None,
) -> Report:
    if fit is None:
        return Report(name, None, False, message)
    return Report(
        name=name,
        fit=fit,
        ok=True,
        message=message,
        mean_active=float(np.mean(active_counts(fit))),
        heldout_r2=heldout_anchor_r2(fit, z_train, z_test, fisher_w_test),
        reml_score=float(fit.reml_score),
        topologies=atom_topologies(fit),
    )


def pc2(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, z.mean(axis=0), vt[:2]


def draw_atoms(ax: Any, report: Report, train_pc: np.ndarray) -> None:
    if not report.ok:
        ax.text(0.5, 0.5, report.message, ha="center", va="center", wrap=True)
        ax.set_title(report.name)
        ax.set_axis_off()
        return
    assignments = np.asarray(report.fit.assignments, dtype=float)
    atom_id = np.argmax(assignments, axis=1)
    palette = {"Circle": "tab:blue", "Sphere": "tab:orange", "Plane": "tab:purple"}
    for atom, topo in enumerate(report.topologies or []):
        mask = atom_id == atom
        if np.any(mask):
            ax.scatter(train_pc[mask, 0], train_pc[mask, 1], s=16, alpha=0.70,
                       c=palette.get(topo, "tab:gray"), label=f"atom {atom}: {topo}")
    ax.set_title(f"Recovered atoms: {report.name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)


def plot(data: Data, train_idx: np.ndarray, reports: list[Report], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    all_pc, center, components = pc2(data.z)
    train_pc = (data.z[train_idx] - center[None, :]) @ components.T
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax = axes[0, 0]
    for label, color, name in [(0, "tab:blue", "Circle true"), (1, "tab:orange", "Sphere true")]:
        mask = data.labels == label
        ax.scatter(all_pc[mask, 0], all_pc[mask, 1], s=18, alpha=0.72, c=color, label=name)
    ax.set_title("Synthetic coexisting manifolds")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)

    draw_atoms(axes[0, 1], reports[0], train_pc)
    draw_atoms(axes[1, 0], reports[1], train_pc)

    ax = axes[1, 1]
    for report, color in zip(reports, ["tab:gray", "tab:green"]):
        if report.ok:
            ax.hist(active_counts(report.fit), bins=np.arange(0, 6) - 0.5,
                    alpha=0.60, color=color, label=report.name)
    ax.set_xticks(range(5))
    ax.set_xlabel("active atoms per row")
    ax.set_ylabel("rows")
    ax.set_title("IBP active-set selectivity")
    ax.legend(frameon=False)
    fig.suptitle("SAE-manifold IBP assignments composed with Fisher-Rao row blocks", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.show()


def print_reports(reports: list[Report]) -> None:
    print("\nComparison\n----------")
    for report in reports:
        if not report.ok:
            print(f"{report.name}: unavailable - {report.message}")
            continue
        print(
            f"{report.name}: mean_active={report.mean_active:.3f}, "
            f"heldout_R2={report.heldout_r2:.4f}, REML={report.reml_score:.3f}, "
            f"atoms=[{', '.join(report.topologies or [])}]"
        )
    if reports[0].ok and reports[1].ok:
        delta = reports[1].reml_score - reports[0].reml_score
        print(f"\nFisher-vs-plain delta REML={delta:.3f}; log10 BF={delta / math.log(10):.3f}")
    elif not reports[1].ok:
        print(f"\nDocumented API gap: {reports[1].message}")


def main() -> None:
    data = make_data()
    train_idx, test_idx = split_rows(data.z.shape[0])
    z_train, z_test = data.z[train_idx], data.z[test_idx]
    plain_fit, plain_msg = sae_fit(z_train, None, seed=101)
    fisher_fit, fisher_msg = sae_fit(z_train, data.fisher_w[train_idx], seed=101)
    reports = [
        summarize("Plain Gaussian IBP", plain_fit, plain_msg, z_train, z_test, None),
        summarize("Fisher-Rao W IBP", fisher_fit, fisher_msg, z_train, z_test, data.fisher_w[test_idx]),
    ]
    print(f"synthetic data: Z={data.z.shape}, fisher_W={data.fisher_w.shape}")
    print(f"train/test rows: {len(train_idx)}/{len(test_idx)}")
    print_reports(reports)
    out_path = Path(__file__).with_suffix(".png")
    plot(data, train_idx, reports, out_path)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
