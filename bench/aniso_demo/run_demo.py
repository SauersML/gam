"""
Anisotropy demo: induce PGS shifts along PC1 and PC3 only, cluster into
populations along the (PC1, PC3) continuum, then run CTN with and without
--scale-dimensions and compare normalized z-scores per population.

Goal: with anisotropy ON, REML should learn small kappa for PC1/PC3 (axes that
deform the conditional CDF) and large kappa for PC2/PC4/PC5 (irrelevant). With
anisotropy OFF, a single kappa has to compromise across all axes. The
normalized z by population should look identical across pops if normalization
is working.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "bench" / "aniso_demo"
GAM = ROOT / "target" / "release" / "gam"

N = 300
SEED = 7
N_POPS = 5
PC_DIM = 5
CENTERS = 12
DUCHON_ORDER = 0
DUCHON_POWER = 8
DUCHON_LENGTH = 1.0


def simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    # PCs uncorrelated, unit variance (already "standardized").
    pcs = rng.standard_normal((N, PC_DIM))
    # Shift PGS along PC1 and PC3 only. PC2/PC4/PC5 are pure noise dimensions.
    # Use a mildly nonlinear shape so CTN has something to bend.
    shift = 1.4 * pcs[:, 0] + 0.6 * (pcs[:, 0] ** 2 - 1.0) \
        + 1.0 * pcs[:, 2] + 0.4 * np.tanh(pcs[:, 2])
    noise = 0.6 * rng.standard_normal(N)
    pgs_raw = shift + noise
    # Cluster on (PC1, PC3) into a continuum of populations.
    coords = pcs[:, [0, 2]]
    km = KMeans(n_clusters=N_POPS, n_init=10, random_state=SEED).fit(coords)
    # Order populations along the PC1+PC3 axis so colors map to the gradient.
    order = np.argsort(km.cluster_centers_[:, 0] + km.cluster_centers_[:, 1])
    relabel = np.empty(N_POPS, dtype=int)
    relabel[order] = np.arange(N_POPS)
    pop = relabel[km.labels_]
    return pcs, pgs_raw, pop


def write_csv(path: Path, pcs: np.ndarray, pgs_raw: np.ndarray) -> None:
    cols = ["pgs_raw"] + [f"pc{i+1}_std" for i in range(PC_DIM)]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(pgs_raw)):
            w.writerow([f"{pgs_raw[i]:.6f}"] + [f"{pcs[i, j]:.6f}" for j in range(PC_DIM)])


def formula() -> str:
    pc_cols = ", ".join(f"pc{i+1}_std" for i in range(PC_DIM))
    return (
        f"pgs_raw ~ duchon({pc_cols}, centers={CENTERS}, "
        f"order={DUCHON_ORDER}, power={DUCHON_POWER}, length_scale={DUCHON_LENGTH:g})"
    )


def run_fit_predict(tag: str, scale_dims: bool, csv_in: Path) -> np.ndarray:
    model = DEMO / f"model_{tag}.json"
    pred = DEMO / f"pred_{tag}.csv"
    log = DEMO / f"fit_{tag}.log"
    cmd = [str(GAM), "fit", "--transformation-normal"]
    if scale_dims:
        cmd.append("--scale-dimensions")
    cmd += ["--out", str(model), str(csv_in), formula()]
    print(f"[fit:{tag}] {' '.join(cmd)}")
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"fit {tag} failed; see {log}")
    cmd_p = [str(GAM), "predict", str(model), str(csv_in), "--out", str(pred)]
    print(f"[predict:{tag}] {' '.join(cmd_p)}")
    rc = subprocess.run(cmd_p, cwd=ROOT).returncode
    if rc != 0:
        raise RuntimeError(f"predict {tag} failed")
    rows = list(csv.DictReader(pred.open()))
    for key in ("z", "z_score", "transformed", "eta", "mean"):
        if rows and key in rows[0]:
            return np.array([float(r[key]) for r in rows])
    raise RuntimeError(f"no z column in {pred}; keys = {list(rows[0].keys()) if rows else []}")


def raincloud(
    ax: Axes,
    values_by_pop: list[np.ndarray],
    colors: Sequence[str | tuple[float, float, float, float]],
    labels: list[str],
    title: str,
) -> None:
    """Half-violin (KDE) above, jittered strip + box below, per population."""
    rng = np.random.default_rng(0)
    n = len(values_by_pop)
    xs_grid = np.linspace(
        min(v.min() for v in values_by_pop) - 0.3,
        max(v.max() for v in values_by_pop) + 0.3,
        400,
    )
    width = 0.45
    for k in range(n):
        v = values_by_pop[k]
        y0 = -k  # rows go downward so pop 0 sits at top
        # KDE half-violin (upward).
        kde = gaussian_kde(v)
        density = kde(xs_grid)
        density = density / density.max() * width * 0.95
        ax.fill_between(xs_grid, y0, y0 + density, color=colors[k], alpha=0.55, linewidth=0)
        ax.plot(xs_grid, y0 + density, color=colors[k], lw=1.0)
        # Jittered scatter strip below the row baseline.
        jitter = rng.uniform(-0.10, -0.02, size=v.size)
        ax.scatter(v, np.full_like(v, y0) + jitter, s=2, color=colors[k], alpha=0.18, linewidths=0)
        # Box: median + IQR.
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [y0 - 0.18, y0 - 0.18], color="black", lw=2.5, solid_capstyle="butt")
        ax.scatter([med], [y0 - 0.18], color="white", edgecolors="black", s=22, zorder=5)
    # Reference standard normal curve at top.
    sn = np.exp(-0.5 * xs_grid ** 2) / np.sqrt(2 * np.pi)
    sn = sn / sn.max() * width * 0.95
    ax.plot(xs_grid, 1 + sn, color="black", lw=1.2, ls="--", label="N(0,1) target")
    ax.set_yticks([1] + [-k for k in range(n)])
    ax.set_yticklabels(["N(0,1)"] + labels)
    ax.set_xlim(xs_grid[0], xs_grid[-1])
    ax.set_ylim(-n, 1.6)
    ax.axvline(0, color="grey", lw=0.6, alpha=0.5)
    ax.set_title(title)


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    print(f"simulating n={N}, pc_dim={PC_DIM}, n_pops={N_POPS}")
    pcs, pgs_raw, pop = simulate()
    print(f"PGS raw: mean={pgs_raw.mean():.3f}, sd={pgs_raw.std():.3f}")
    for k in range(N_POPS):
        m = pop == k
        print(f"  pop{k}: n={m.sum()}, PC1_mean={pcs[m,0].mean():+.2f}, "
              f"PC3_mean={pcs[m,2].mean():+.2f}, PGS_mean={pgs_raw[m].mean():+.2f}, "
              f"PGS_sd={pgs_raw[m].std():.2f}")

    csv_path = DEMO / "data.csv"
    write_csv(csv_path, pcs, pgs_raw)

    # Plot 1: PC space colored by pop + raw PGS raincloud by pop.
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(N_POPS - 1, 1)) for i in range(N_POPS)]
    labels = [f"pop{k}" for k in range(N_POPS)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for k in range(N_POPS):
        m = pop == k
        axes[0].scatter(pcs[m, 0], pcs[m, 2], s=4, color=colors[k], alpha=0.5, label=labels[k])
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC3")
    axes[0].set_title("Population clusters in (PC1, PC3) space")
    axes[0].legend(markerscale=3, loc="best")
    raincloud(axes[1], [pgs_raw[pop == k] for k in range(N_POPS)], colors, labels,
              "Raw PGS by population (uncalibrated)")
    axes[1].set_xlabel("pgs_raw")
    fig.tight_layout()
    fig.savefig(DEMO / "01_raw.png", dpi=140)
    plt.close(fig)
    print(f"saved {DEMO/'01_raw.png'}")

    # Run CTN twice.
    z_iso = run_fit_predict("iso", scale_dims=False, csv_in=csv_path)
    z_aniso = run_fit_predict("aniso", scale_dims=True, csv_in=csv_path)

    def report(tag: str, z: np.ndarray) -> None:
        print(f"[{tag}] overall: mean={z.mean():+.3f}, sd={z.std():.3f}")
        for k in range(N_POPS):
            zz = z[pop == k]
            print(f"  pop{k}: mean={zz.mean():+.3f}, sd={zz.std():.3f}")

    report("iso", z_iso)
    report("aniso", z_aniso)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), sharex=True)
    raincloud(axes[0], [z_iso[pop == k] for k in range(N_POPS)], colors, labels,
              "CTN normalized z — isotropic kappa  (--scale-dimensions OFF)")
    raincloud(axes[1], [z_aniso[pop == k] for k in range(N_POPS)], colors, labels,
              "CTN normalized z — anisotropic per-axis kappa  (--scale-dimensions ON)")
    for ax in axes:
        ax.set_xlabel("z = h(pgs_raw; PC)")
    fig.suptitle("If normalization is working, all populations should overlap N(0,1).", y=1.02)
    fig.tight_layout()
    fig.savefig(DEMO / "02_normalized.png", dpi=140)
    plt.close(fig)
    print(f"saved {DEMO/'02_normalized.png'}")


if __name__ == "__main__":
    main()
