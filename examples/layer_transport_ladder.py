#!/usr/bin/env python3
"""Isometry-defect-by-depth ladder via the Rust layer-transport engine (#1013).

Loads ``runs/OLMO3_32B_TRAJ_SFT/<step>/extra/activations.npy`` (shape
``(n_prompts, n_layers, d_model)``), fits the hue-circle chart per layer as
the leading PCA phase (the same per-layer chart pattern as
``examples/layer_transport_maps.py``), then hands the whole coordinate ladder
to ``gamfit.layer_transport.layer_transport_ladder``. The Rust core fits every
adjacent and two-hop transport map with REML, reads the winding degree,
computes the data-density-weighted isometry defect with a delta-method SE, and
tests the composition law ``h_{l->l+2} =? h_{l+1->l+2} o h_{l->l+1}`` per
triple — TRANSPORT layers show near-zero defect, COMPUTE layers reshape the
chart metric.

Writes an isometry-defect-by-depth CSV table and, when matplotlib is
available, a defect-by-depth plot. All math is in Rust; this driver is glue.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi
DEFAULT_BANK_GLOBS = (
    "runs/OLMO3_32B_TRAJ_SFT/*/extra/activations.npy",
    "/home/azuser/Manifold-SAE/runs/OLMO3_32B_TRAJ_SFT/*/extra/activations.npy",
)
DEFAULT_LAYERS = list(range(20, 31))
DEFAULT_OUT_PREFIX = "layer_transport_ladder"


def find_activations(bank: str | None) -> Path | None:
    if bank is not None:
        candidate = Path(bank)
        if candidate.is_file():
            return candidate
        nested = candidate / "activations.npy"
        return nested if nested.is_file() else None
    for pattern in DEFAULT_BANK_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    return None


def standardize_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    x = np.asarray(acts[:, layer, :], dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    return np.ascontiguousarray((x - mu) / sd)


def layer_circle_chart(acts: np.ndarray, layer: int) -> np.ndarray:
    """Leading PCA phase of the layer's standardized activations.

    The chart's own rotation/reflection gauge is irrelevant downstream: the
    transport estimand is the double coset, and the Rust composition test
    quotients the certified circle isometries before testing.
    """
    x = standardize_layer(acts, layer)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    if u.shape[1] < 2:
        raise ValueError(f"layer {layer} PCA chart needs at least two components")
    return np.mod(np.arctan2(u[:, 1] * s[1], u[:, 0] * s[0]), TAU)


def write_defect_table(path: Path, ladder: dict) -> None:
    fields = [
        "layer_from",
        "layer_to",
        "kind",
        "degree",
        "topology_preserved",
        "isometry_defect",
        "isometry_defect_se",
        "transport_edf",
        "composition_defect",
        "composition_p_value",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for kind in ("adjacent", "two_hop"):
            for report in ladder[kind]:
                row = {field: report.get(field) for field in fields}
                row["kind"] = kind
                writer.writerow(row)


def maybe_plot(path: Path, ladder: dict) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    adjacent = ladder["adjacent"]
    depth = [r["layer_from"] for r in adjacent]
    defect = [r["isometry_defect"] for r in adjacent]
    se = [r["isometry_defect_se"] for r in adjacent]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(depth, defect, yerr=se, marker="o", capsize=3, label="adjacent h_{l->l+1}")
    two_hop = ladder["two_hop"]
    if two_hop:
        ax.plot(
            [r["layer_from"] for r in two_hop],
            [r["composition_defect"] for r in two_hop],
            marker="s",
            linestyle="--",
            label="composition RMS defect",
        )
    ax.set_xlabel("layer l")
    ax.set_ylabel("isometry defect  E[(|h'| - 1)^2]")
    ax.set_title("Inter-layer concept transport: isometry defect by depth (#1013)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=None, help="activations.npy file or its parent dir")
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    activations_path = find_activations(args.bank)
    if activations_path is None:
        print(
            "No activation bank found (looked for "
            + ", ".join(DEFAULT_BANK_GLOBS)
            + "). Point --bank at runs/OLMO3_32B_TRAJ_SFT/<step>/extra/activations.npy "
            "to run the ladder; nothing to do.",
        )
        return 0

    acts = np.load(activations_path, mmap_mode="r")
    n_layers = acts.shape[1]
    layers = [layer for layer in args.layers if 0 <= layer < n_layers]
    if len(layers) < 2:
        print(f"need at least two valid layers below {n_layers}, got {layers}")
        return 1

    print(f"bank: {activations_path}  shape: {tuple(acts.shape)}  layers: {layers}")
    coords = [layer_circle_chart(acts, layer) for layer in layers]

    from gamfit.layer_transport import layer_transport_ladder

    ladder = layer_transport_ladder(coords, topology="circle", layers=layers)

    for report in ladder["adjacent"]:
        print(
            f"h_{{{report['layer_from']}->{report['layer_to']}}}: "
            f"degree={report['degree']} preserved={report['topology_preserved']} "
            f"isometry_defect={report['isometry_defect']:.4f}"
            f"±{report['isometry_defect_se']:.4f} edf={report['transport_edf']:.2f}"
        )
    for report in ladder["two_hop"]:
        print(
            f"composition {report['layer_from']}->{report['layer_to']}: "
            f"rms_defect={report['composition_defect']:.4f} "
            f"p={report['composition_p_value']:.4g} "
            f"reflected={report['composition_gauge_reflected']}"
        )

    table_path = Path(f"{args.out_prefix}_defect_by_depth.csv")
    write_defect_table(table_path, ladder)
    print(f"wrote {table_path}")
    json_path = Path(f"{args.out_prefix}_report.json")
    json_path.write_text(json.dumps(ladder, indent=2, sort_keys=True, default=str))
    print(f"wrote {json_path}")
    plot_path = Path(f"{args.out_prefix}_defect_by_depth.png")
    if maybe_plot(plot_path, ladder):
        print(f"wrote {plot_path}")
    else:
        print("matplotlib unavailable; skipped the plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
