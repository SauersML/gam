"""D2 chart transport: the weekday circle's equation of motion across layers.

Fits the cyclic (circle) atom independently at three residual-stream layers
(L11 / L17 / L23 of Qwen3.6-35B-A3B, from the MSI activation cache), then asks,
for each layer hop, whether the hop *carries* the circle (near-isometric
transport) or *computes* on it (rotation / rescale / re-embedding) — with
certificates, not vibes. All estimation math lives in the Rust core
(``sae_manifold_fit``, ``layer_transport_fit``, ``chart_transfer_operator``,
``certify_chart_transfer``); this script is the HF/figure boundary only.

Three stacked evidence levels per hop j -> k:

1. AMBIENT (gauge-invariant): principal angles between the two fitted circle
   planes ``P_j, P_k`` (2048x2 orthonormal frames). Cosines near 1 mean the
   circle lives in the SAME ambient plane at both layers; away from 1 the
   representation is re-embedded.
2. DATA-level chart transport (Rust ``layer_transport_fit``): the fitted map
   ``theta_k = h(theta_j)`` with winding degree, isometry defect (+SE), and
   rotation offset. Defects and degree are invariant to per-layer chart phase.
   The rotation offset is reported in the AMBIENT-ANCHORED gauge: every
   layer's frame is aligned to the first layer's frame by the polar factor of
   ``P_l^T P_ref`` (ambient parallel transport), so a nonzero offset is a real
   phase advance of the representation, not a chart artifact. (Pinning each
   layer's phase by labels would absorb any true rotation into the gauges and
   read zero by construction — deliberately NOT done.)
3. MODEL-level (the mechanistic prize; Rust ``chart_transfer_operator``): the
   pulled-back 2x2 operator ``A_kj = (P_k^T P_k)^-1 P_k^T J_F(x) P_j`` from
   frozen-model JVPs (supplied by the model lane as an .npz), with the
   density-weighted mean, per-token band, SO(2) polar rotation angle (+SE),
   transport defect, and Lie-equivariance defect against the circle generator.

Run modes
---------
--synthetic          end-to-end validation on planted 3-layer data (a known
                     in-plane rotation at hop 2, identity at hop 1) including a
                     synthetic linear "model" JVP; asserts recovery.
--shards DIR         real cache: safetensors shards holding acts_L11/17/23.

Every fit cell is guarded by SIGALRM (--fit-timeout); wrap the whole run in
``timeout`` in the sbatch as well.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TWO_PI = 2.0 * math.pi
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# The circle's infinitesimal rotation generator: the Lie-equivariance reference.
CIRCLE_GENERATOR = np.array([[0.0, -1.0], [1.0, 0.0]])


class FitTimeout(RuntimeError):
    pass


@contextmanager
def fit_alarm(seconds: int, label: str):
    """SIGALRM guard so a hanging fit cell is recorded, not a stall."""

    def _raise(signum, frame):  # noqa: ARG001
        raise FitTimeout(f"fit cell '{label}' exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


def load_layers(
    shards_dir: Path, layer_keys: list[str], n_max: int, token_indices: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Row-aligned activations per layer from the safetensors cache.

    Shards are concatenated in sorted filename order so every layer sees the
    same token rows; the optional token subset (weekday probe tokens) indexes
    into that shared row order.
    """
    from safetensors import safe_open

    files = sorted(shards_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors shards under {shards_dir}")
    if token_indices is not None:
        token_indices = np.sort(np.asarray(token_indices, dtype=np.int64))[:n_max]
    per_layer: dict[str, list[np.ndarray]] = {k: [] for k in layer_keys}
    offset = 0  # global row offset of the current shard
    taken = 0
    for f in files:
        with safe_open(str(f), framework="np") as sf:
            names = set(sf.keys())
            missing = [k for k in layer_keys if k not in names]
            if missing:
                raise SystemExit(f"{f} lacks tensors {missing}; has {sorted(names)[:8]}")
            # Row count from the first layer key; every key must agree.
            shard_rows = sf.get_slice(layer_keys[0]).get_shape()[0]
            if token_indices is not None:
                local = token_indices[
                    (token_indices >= offset) & (token_indices < offset + shard_rows)
                ] - offset
                if local.size:
                    for k in layer_keys:
                        arr = sf.get_tensor(k)  # one shard resident at a time
                        per_layer[k].append(np.asarray(arr[local], dtype=np.float64))
                    taken += local.size
            else:
                take = min(shard_rows, max(0, n_max - taken))
                if take:
                    for k in layer_keys:
                        arr = sf.get_tensor(k)
                        per_layer[k].append(np.asarray(arr[:take], dtype=np.float64))
                    taken += take
        offset += shard_rows
        if token_indices is None and taken >= n_max:
            break
    if token_indices is not None and taken < token_indices.size:
        raise SystemExit(
            f"token subset asks for {token_indices.size} rows but the cache holds "
            f"only {offset}; {token_indices.size - taken} indices out of range"
        )
    if not any(per_layer[layer_keys[0]]):
        raise SystemExit("no rows selected from the cache")
    return {k: np.concatenate(v, axis=0) for k, v in per_layer.items()}


# --------------------------------------------------------------------------
# per-layer circle fit -> chart (angles + ambient plane frame)
# --------------------------------------------------------------------------


@dataclass
class LayerChart:
    layer: str
    theta: np.ndarray  # (N,) angles in [0, 2pi)
    plane: np.ndarray  # (ambient, 2) orthonormal circle-plane frame
    center: np.ndarray  # (ambient,) plane center
    centered: np.ndarray  # (N, ambient) centered fitted curve rows
    reconstruction_r2: float
    gauge: str = "arc"  # "arc" (per-layer arbitrary phase) | "ambient-anchored"
    notes: list[str] = field(default_factory=list)


def fit_layer_circle(
    x: np.ndarray, layer: str, seed: int, timeout_s: int, pca_rank: int | None = None
) -> LayerChart:
    """K=1 curved circle fit (never-fails path); chart = (angles, plane frame).

    With ``pca_rank`` the fit runs in the top-``r`` PCA coordinates (DOSE's
    recipe — rank ~48 keeps the fit fast and conditions the smoother) and the
    fitted curve is lifted back to ambient before the plane frame is taken, so
    every cross-layer quantity stays in ambient coordinates.
    """
    import gamfit

    lift = None
    x_fit = x
    x_mean = x.mean(axis=0, keepdims=True)
    if pca_rank is not None and 2 <= pca_rank < min(x.shape):
        xc = x - x_mean
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        lift = vt[:pca_rank]  # (r, ambient)
        x_fit = xc @ lift.T
    with fit_alarm(timeout_s, f"circle@{layer}"):
        res = gamfit.sae_manifold_fit(
            x_fit,
            K=1,
            d_atom=1,
            atom_topology="circle",
            assignment="ibp_map",
            random_state=seed,
            # DOSE's proven-good K=1 kwargs on MSI (r2=0.997): the default
            # isometry_weight=1.0 / n_iter=50 path can live-lock on K=1 (STATE
            # #9); isometry_weight=0.0 with a bounded iteration count fits
            # cleanly. The circle's shape is carried by the topology prior, not
            # the isometry penalty, so zeroing it does not distort the chart.
            isometry_weight=0.0,
            n_iter=30,
        )
    # Certificate-gated honest angle (#2081): raises on a degenerate chart
    # rather than returning an arbitrary coordinate.
    u_arc = res.atom_angle_coordinate(0)
    notes: list[str] = []
    fitted = np.asarray(res.fitted, dtype=np.float64)
    if lift is not None:
        fitted = fitted @ lift + x_mean  # back to ambient
        notes.append(f"fit in top-{lift.shape[0]} PCA coords, lifted to ambient")
    center = fitted.mean(axis=0)
    centered = fitted - center
    # The fitted decoder samples live (numerically) on the circle's 2-plane:
    # its top-2 right singular vectors are the plane frame.
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    plane = vt[:2].T  # (ambient, 2), orthonormal
    planarity = float((svals[:2] ** 2).sum() / max((svals**2).sum(), 1e-300))
    notes.append(f"planarity(top2 EV share of fitted curve)={planarity:.4f}")

    proj = centered @ plane  # (N, 2)
    phi = np.mod(np.arctan2(proj[:, 1], proj[:, 0]), TWO_PI)
    if u_arc is None:
        raise SystemExit(
            f"{layer}: coordinate-fidelity chart is degenerate (no u_arc); "
            "refusing to substitute raw t. Inspect the fit."
        )
    theta = np.mod(TWO_PI * np.asarray(u_arc, dtype=np.float64).ravel(), TWO_PI)
    # Orient the plane frame so the projected angle runs WITH the honest
    # arc-length coordinate (flip the second axis if reversed) — a pure gauge
    # choice, fixed identically at every layer so hops are comparable. The
    # comparison is between RESULTANT MAGNITUDES of phi -/+ theta (invariant
    # to the arbitrary phase offset between the two coordinates); the mean
    # cosine alone would confound a phase offset with an orientation flip.
    arc_angle = TWO_PI * np.asarray(u_arc, dtype=np.float64).ravel()
    corr = abs(np.exp(1j * (phi - arc_angle)).mean())
    anti = abs(np.exp(1j * (-phi - arc_angle)).mean())
    if anti > corr:
        plane = plane[:, [0, 1]] * np.array([1.0, -1.0])
        notes.append("plane orientation flipped to match u_arc direction")
    notes.append(f"reconstruction_r2={float(res.reconstruction_r2):.4f}")
    return LayerChart(
        layer=layer,
        theta=theta,
        plane=plane,
        center=center,
        centered=centered,
        reconstruction_r2=float(res.reconstruction_r2),
        notes=notes,
    )


def anchor_gauges_to_first_layer(charts: list[LayerChart]) -> None:
    """Pin every chart's phase/orientation by AMBIENT parallel transport.

    Per-layer charts carry an arbitrary phase, so a naive per-layer pinning
    (e.g. by labels at every layer) would absorb any real inter-layer rotation
    into the gauges and read zero by construction. The meaningful "the hop
    rotates the circle by Z radians" needs a gauge that is chosen ONCE and
    carried across layers by the ambient geometry: align each layer's plane
    frame to the FIRST layer's frame via the polar (Procrustes) factor of
    ``P_l^T P_ref`` — the closest in-plane rotation to ambient-parallel
    transport of the reference frame. The phase advance of the tokens'
    arc-length coordinate relative to that transported frame is then a real,
    layer-to-layer rotation, not a gauge artifact. Requires the planes to be
    non-orthogonal (principal angles < 90 deg); a hop whose planes are near
    orthogonal is a re-embedding and gets no anchored angle.
    """
    ref = charts[0].plane
    for chart in charts:
        m = chart.plane.T @ ref  # (2, 2)
        det = float(np.linalg.det(m))
        if abs(det) < 1e-12:
            chart.notes.append(
                "plane near-orthogonal to reference; anchored gauge undefined"
            )
            continue
        if det < 0.0:
            # Orientation flip relative to the reference: mirror the chart so
            # the anchored frames share orientation, and reverse the angle.
            chart.plane = chart.plane * np.array([1.0, -1.0])
            chart.theta = np.mod(-chart.theta, TWO_PI)
            m = chart.plane.T @ ref
            chart.notes.append("orientation mirrored to match reference frame")
        # Polar factor U of m: the in-plane rotation aligning P_l to ref.
        u_svd, _, vt_svd = np.linalg.svd(m)
        u_rot = u_svd @ vt_svd  # det +1 by construction now
        anchored = chart.plane @ u_rot
        # Phase offset between the arc-length gauge and the anchored frame:
        # delta = circular mean of (projected-angle - arc-angle); the anchored
        # coordinate keeps arc-length honesty, only its zero moves.
        proj = chart.centered @ anchored
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        delta = math.atan2(
            float(np.sin(phi - chart.theta).mean()),
            float(np.cos(phi - chart.theta).mean()),
        )
        chart.theta = np.mod(chart.theta + delta, TWO_PI)
        chart.plane = anchored
        chart.gauge = "ambient-anchored"


# --------------------------------------------------------------------------
# per-hop evidence
# --------------------------------------------------------------------------


def principal_plane_angles(p_from: np.ndarray, p_to: np.ndarray) -> list[float]:
    """Principal angles (radians) between the two circle planes in ambient."""
    svals = np.linalg.svd(p_to.T @ p_from, compute_uv=False)
    return [float(math.acos(min(1.0, max(-1.0, s)))) for s in svals]


def hop_evidence(
    chart_from: LayerChart,
    chart_to: LayerChart,
    jvp_npz: Path | None,
    hop_index: int,
) -> dict[str, Any]:
    import gamfit

    out: dict[str, Any] = {
        "from": chart_from.layer,
        "to": chart_to.layer,
        "gauge": chart_from.gauge,
    }
    out["plane_principal_angles_rad"] = principal_plane_angles(
        chart_from.plane, chart_to.plane
    )
    # DATA level: REML transport map on the shared tokens' angles.
    out["data_transport"] = gamfit.layer_transport_fit(
        chart_from.theta,
        chart_to.theta,
        "circle",
        "circle",
        layer_from=hop_index,
        layer_to=hop_index + 1,
    )
    # Frame-level alignment: P_to^T P_from as a 2x2 operator (how the plane
    # maps under the identity ambient pushforward — 'is it the same plane,
    # same orientation, same phase').
    has_transfer_ffi = hasattr(gamfit, "certify_chart_transfer") and hasattr(
        gamfit, "chart_transfer_operator"
    )
    m = chart_to.plane.T @ chart_from.plane
    if has_transfer_ffi:
        out["frame_alignment"] = gamfit.certify_chart_transfer(
            m, CIRCLE_GENERATOR, CIRCLE_GENERATOR
        )
        out["frame_alignment"]["operator"] = m.tolist()
    else:
        out["frame_alignment"] = {"operator": m.tolist(), "skipped": "gamfit lacks chart-transfer FFI"}

    # MODEL level: pulled-back transfer operator from frozen-model JVPs.
    if jvp_npz is not None and not has_transfer_ffi:
        print(
            "WARNING: JVPs supplied but this gamfit lacks chart_transfer_operator; "
            "model-level evidence skipped (upgrade the wheel)",
            file=sys.stderr,
        )
    elif jvp_npz is not None:
        with np.load(jvp_npz) as z:
            jvp = np.asarray(z["jvp"], dtype=np.float64)  # (N, ambient, 2)
            weights = (
                np.asarray(z["weights"], dtype=np.float64) if "weights" in z else None
            )
        n = jvp.shape[0]
        jets = np.broadcast_to(
            chart_to.plane[None, :, :], (n, *chart_to.plane.shape)
        ).copy()
        rep = gamfit.chart_transfer_operator(jets, jvp, weights)
        mean_op = np.asarray(rep["mean"])
        cert = gamfit.certify_chart_transfer(
            mean_op, CIRCLE_GENERATOR, CIRCLE_GENERATOR
        )
        out["model_transfer"] = {
            "mean": mean_op.tolist(),
            "variance": np.asarray(rep["variance"]).tolist(),
            "effective_n": rep["effective_n"],
            "rotation_angle_mean": rep["rotation_angle_mean"],
            "rotation_angle_se": rep["rotation_angle_se"],
            "rotation_angle_of_mean": rep["rotation_angle_of_mean"],
            "transport_defect": cert["transport_defect"],
            "equivariance_defect": cert["equivariance_defect"],
        }
    return out


def classify_hop(hop: dict[str, Any]) -> str:
    """Transport vs compute, from the certificates.

    'transport' when the data-level isometry defect is within 2 SE of zero AND
    the rotation angle (model-level when present, else data rotation offset)
    is within 2 SE of zero; 'rotation' when the angle is significant but the
    isometry defect is not; 'compute' otherwise. The 2-SE studentization is a
    conventional two-sided criterion, reported alongside the raw numbers so a
    reader can re-threshold.
    """
    data = hop["data_transport"]
    iso = data["isometry_defect"]
    iso_se = max(data["isometry_defect_se"], 1e-300)
    iso_sig = iso / iso_se > 2.0
    if "model_transfer" in hop and hop["model_transfer"]["rotation_angle_mean"] is not None:
        ang = hop["model_transfer"]["rotation_angle_mean"]
        ang_se = max(hop["model_transfer"]["rotation_angle_se"] or 0.0, 1e-300)
    else:
        ang = data["rotation_offset"]
        # The data-level offset carries no SE; reuse the isometry SE scale as a
        # conservative studentizer and flag it.
        ang_se = max(iso_se, 1e-300)
    # Wrap to (-pi, pi] for the significance read.
    ang_wrapped = math.atan2(math.sin(ang), math.cos(ang))
    ang_sig = abs(ang_wrapped) / ang_se > 2.0
    if not iso_sig and not ang_sig:
        return "transport"
    if ang_sig and not iso_sig:
        return "rotation"
    return "compute"


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def make_figure(
    charts: list[LayerChart],
    hops: list[dict[str, Any]],
    labels: np.ndarray | None,
    label_names: list[str] | None,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_l = len(charts)
    fig, axes = plt.subplots(
        2, max(n_l, n_l - 1), figsize=(4.2 * n_l, 8.0), facecolor=SURFACE
    )
    fig.suptitle(
        "Chart transport of the cyclic atom across layers", color=INK, fontsize=14
    )

    for ax_row in axes:
        for ax in np.atleast_1d(ax_row):
            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, color=GRID, linewidth=0.6)

    # Row 1: each layer's circle in its own fitted plane.
    for i, chart in enumerate(charts):
        ax = axes[0][i]
        th = chart.theta
        if labels is not None:
            for u in np.unique(labels):
                mask = labels == u
                name = label_names[int(u)] if label_names else str(u)
                ax.scatter(
                    np.cos(th[mask]),
                    np.sin(th[mask]),
                    s=9,
                    color=SERIES[int(u) % len(SERIES)],
                    label=name if i == 0 else None,
                    linewidths=0,
                )
        else:
            ax.scatter(np.cos(th), np.sin(th), s=9, color=SERIES[0], linewidths=0)
        ax.set_title(f"{chart.layer} ({chart.gauge} gauge)", color=INK2, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
    if labels is not None:
        axes[0][0].legend(
            loc="center", fontsize=7, frameon=False, labelcolor=INK2, handletextpad=0.2
        )

    # Row 2: angle-transport scatter per hop with the certificate annotation.
    for h, hop in enumerate(hops):
        ax = axes[1][h]
        cf, ct = charts[h], charts[h + 1]
        ax.scatter(cf.theta, ct.theta, s=6, color=SERIES[0], alpha=0.5, linewidths=0)
        data = hop["data_transport"]
        verdict = hop["classification"]
        ang = None
        band = ""
        if "model_transfer" in hop and hop["model_transfer"]["rotation_angle_mean"] is not None:
            ang = hop["model_transfer"]["rotation_angle_mean"]
            se = hop["model_transfer"]["rotation_angle_se"]
            band = f" ± {2 * se:.3f} (2SE, model)"
        else:
            ang = data["rotation_offset"]
            band = " (data gauge offset)"
        ax.set_title(
            f"{cf.layer} → {ct.layer}: {verdict}\n"
            f"rot {ang:+.3f} rad{band}; iso defect "
            f"{data['isometry_defect']:.3g}±{data['isometry_defect_se']:.2g}, "
            f"deg {data['degree']}",
            color=INK2,
            fontsize=9,
        )
        ax.set_xlabel(f"θ {cf.layer} (rad)", color=MUTED, fontsize=8)
        ax.set_ylabel(f"θ {ct.layer} (rad)", color=MUTED, fontsize=8)
        ax.set_xlim(0, TWO_PI)
        ax.set_ylim(0, TWO_PI)
    for h in range(len(hops), axes.shape[1]):
        axes[1][h].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=160)
    print(f"figure -> {out_png}")


# --------------------------------------------------------------------------
# synthetic end-to-end validation
# --------------------------------------------------------------------------


def synthetic_layers(
    n: int, ambient: int, planted_rot: float, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Three planted layers + a linear 'model' whose JVPs we know exactly.

    Layer A: circle in the (e1, e2) plane. Hop A->B: identity (pure
    transport). Hop B->C: in-plane rotation by ``planted_rot`` plus a mild
    isotropic rescale (a 'compute' hop with a known angle).
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 7, size=n)
    theta = labels * (TWO_PI / 7.0) + rng.normal(0.0, 0.05, size=n)
    basis = np.linalg.qr(rng.normal(size=(ambient, ambient)))[0]
    p = basis[:, :2]
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n, 2)
    noise = 0.02
    x_a = circle @ p.T + rng.normal(0.0, noise, size=(n, ambient))
    # F1 = identity; F2 = rotation by planted_rot inside the plane, 1.1x scale.
    c, s = math.cos(planted_rot), math.sin(planted_rot)
    rot2 = np.array([[c, -s], [s, c]])
    f2 = np.eye(ambient) + p @ (1.1 * rot2 - np.eye(2)) @ p.T
    x_b = x_a  # identity hop
    x_c = x_b @ f2.T + rng.normal(0.0, noise, size=(n, ambient))
    layers = {"A": x_a, "B": x_b + rng.normal(0.0, noise, size=(n, ambient)), "C": x_c}
    jvps = {"A->B": np.eye(ambient), "B->C": f2}
    return layers, labels, jvps


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, help="safetensors cache directory")
    ap.add_argument("--probe-npz", type=Path,
                    help="weekday probe harvest .npz (acts_L*, labels, template_ids) "
                         "from weekday_probe_harvest.py; overrides --shards")
    ap.add_argument("--demean-by-template", action="store_true",
                    help="per-template demeaning of probe activations (isolates the "
                         "weekday signal from template context, per DOSE's recipe)")
    ap.add_argument("--dump-planes", type=Path,
                    help="write fitted plane frames (plane_L{l}) to this .npz for the "
                         "JVP stage")
    ap.add_argument(
        "--layer-keys",
        nargs="+",
        default=["acts_L11", "acts_L17", "acts_L23"],
        help="tensor names, in depth order",
    )
    ap.add_argument("--tokens-json", type=Path, help="weekday probe subset: "
                    '{"indices": [...], "labels": [...], "label_names": [...]}')
    ap.add_argument("--jvp", action="append", type=Path, default=None,
                    help=".npz with 'jvp' (N, ambient, 2) per hop, in hop order")
    ap.add_argument("--n-max", type=int, default=30000)
    ap.add_argument("--pca-rank", type=int, default=None,
                    help="fit each layer's circle in its top-r PCA coordinates "
                         "(lifted back to ambient afterwards); e.g. 48 for the "
                         "70-prompt weekday battery")
    ap.add_argument("--fit-timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("chart_transport_out"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-rot", type=float, default=0.9)
    ap.add_argument("--synthetic-n", type=int, default=800)
    ap.add_argument("--synthetic-ambient", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = label_names = None
    jvp_paths: list[Path | None]

    if args.synthetic:
        layers, labels, jvp_mats = synthetic_layers(
            args.synthetic_n, args.synthetic_ambient, args.synthetic_rot, args.seed
        )
        label_names = [f"day{i}" for i in range(7)]
        layer_keys = list(layers)
    elif args.probe_npz is not None:
        with np.load(args.probe_npz) as z:
            layer_keys = sorted(
                (k for k in z.files if k.startswith("acts_L")),
                key=lambda k: int(k.split("acts_L")[1]),
            )
            layers = {k: np.asarray(z[k], dtype=np.float64) for k in layer_keys}
            labels = np.asarray(z["labels"], dtype=np.int64) if "labels" in z else None
            template_ids = (
                np.asarray(z["template_ids"], dtype=np.int64)
                if "template_ids" in z
                else None
            )
        label_names = WEEKDAY_NAMES if labels is not None and labels.max() < 7 else None
        if args.demean_by_template:
            if template_ids is None:
                raise SystemExit("--demean-by-template needs template_ids in the npz")
            for k in layer_keys:
                x = layers[k]
                for t in np.unique(template_ids):
                    m = template_ids == t
                    x[m] -= x[m].mean(axis=0, keepdims=True)
    else:
        if args.shards is None:
            ap.error("--shards, --probe-npz, or --synthetic required")
        token_indices = None
        if args.tokens_json is not None:
            meta = json.loads(args.tokens_json.read_text())
            token_indices = np.asarray(meta["indices"], dtype=np.int64)
            labels = np.asarray(meta["labels"], dtype=np.int64)
            if labels.shape != token_indices.shape:
                raise SystemExit("tokens-json labels/indices length mismatch")
            # The loader returns rows in ascending index order; align labels.
            order = np.argsort(token_indices)
            labels = labels[order][: args.n_max]
            label_names = meta.get("label_names")
        layers = load_layers(args.shards, args.layer_keys, args.n_max, token_indices)
        layer_keys = args.layer_keys

    charts: list[LayerChart] = []
    for key in layer_keys:
        chart = fit_layer_circle(
            layers[key], key, args.seed, args.fit_timeout, pca_rank=args.pca_rank
        )
        charts.append(chart)
    # ONE gauge for the whole ladder: ambient parallel transport of the first
    # layer's frame. Per-layer pinning (e.g. by labels at every layer) would
    # absorb any real rotation into the charts and read zero by construction.
    anchor_gauges_to_first_layer(charts)
    for chart in charts:
        print(f"{chart.layer}: gauge={chart.gauge}; {'; '.join(chart.notes)}")
    if args.dump_planes is not None:
        np.savez(
            args.dump_planes,
            **{
                f"plane_L{c.layer.split('acts_L')[-1]}": c.plane for c in charts
            },
        )
        print(f"planes -> {args.dump_planes}")

    # Model JVPs: synthetic mode manufactures them from the known linear maps.
    hop_names = [f"{layer_keys[i]}->{layer_keys[i + 1]}" for i in range(len(charts) - 1)]
    jvp_paths = [None] * len(hop_names)
    if args.synthetic:
        for i, name in enumerate(hop_names):
            f_mat = jvp_mats[["A->B", "B->C"][i]]
            jv = np.einsum("ab,bc->ac", f_mat, charts[i].plane)  # (ambient, 2)
            n = charts[i].theta.shape[0]
            arr = np.broadcast_to(jv[None], (n, *jv.shape)).copy()
            path = args.out / f"jvp_{i}.npz"
            np.savez(path, jvp=arr)
            jvp_paths[i] = path
    elif args.jvp:
        for i, p in enumerate(args.jvp[: len(hop_names)]):
            jvp_paths[i] = p

    hops = []
    for i in range(len(charts) - 1):
        hop = hop_evidence(charts[i], charts[i + 1], jvp_paths[i], i)
        hop["classification"] = classify_hop(hop)
        hops.append(hop)
        mt = hop.get("model_transfer")
        print(
            f"{hop['from']} -> {hop['to']}: {hop['classification']}; "
            f"data iso {hop['data_transport']['isometry_defect']:.4g}"
            f"±{hop['data_transport']['isometry_defect_se']:.2g}, "
            f"deg {hop['data_transport']['degree']}, "
            f"plane angles {['%.3f' % a for a in hop['plane_principal_angles_rad']]}"
            + (
                f"; MODEL rot {mt['rotation_angle_mean']:+.4f}"
                f"±{mt['rotation_angle_se']:.4f} rad, "
                f"transport defect {mt['transport_defect']:.4g}, "
                f"equivariance defect {mt['equivariance_defect']:.4g}"
                if mt
                else ""
            )
        )

    summary = {
        "layers": [
            {
                "layer": c.layer,
                "gauge": c.gauge,
                "notes": c.notes,
            }
            for c in charts
        ],
        "hops": [
            {k: v for k, v in h.items() if k != "frame_alignment"}
            | {"frame_alignment": h["frame_alignment"]}
            for h in hops
        ],
    }
    (args.out / "chart_transport_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    make_figure(charts, hops, labels, label_names, args.out / "chart_transport.png")

    if args.synthetic:
        if "model_transfer" not in hops[0]:
            sys.exit("synthetic validation needs the chart-transfer FFI (new wheel)")
        # Falsification gate: hop 1 must classify transport with |angle| small;
        # hop 2 must recover the planted rotation within its 2SE band (plus the
        # fit's own angle-estimation error, bounded here by 0.05 rad).
        mt1, mt2 = hops[0]["model_transfer"], hops[1]["model_transfer"]
        a1, a2 = mt1["rotation_angle_mean"], mt2["rotation_angle_mean"]
        tol2 = 2 * mt2["rotation_angle_se"] + 0.05
        ok1 = abs(a1) < 2 * mt1["rotation_angle_se"] + 0.05
        ok2 = abs(a2 - args.synthetic_rot) < tol2
        print(
            f"SYNTHETIC CHECK: hop1 angle {a1:+.4f} (want ~0) -> {'PASS' if ok1 else 'FAIL'}; "
            f"hop2 angle {a2:+.4f} (want {args.synthetic_rot:+.4f} ± {tol2:.4f}) -> "
            f"{'PASS' if ok2 else 'FAIL'}"
        )
        if not (ok1 and ok2):
            sys.exit(1)


if __name__ == "__main__":
    main()
