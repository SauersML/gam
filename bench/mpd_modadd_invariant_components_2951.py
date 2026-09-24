"""#2951: the grokked modular-addition embedding's components from its own isometry-invariant operators.

Thin driver of the Rust surface operation ``invariant_components`` (``parameter_decomposition::invariant_spectrum``):
the centred embedding rows go in, the joint eigenspaces of the Hadamard powers of their Gram and the derived components
come out. No group, rank, scale or threshold is given. The known Fourier planes are used only to score the result:
each component's row coordinates are compared with every frequency's cos/sin pair by principal angles
(``cos theta_1 * cos theta_2 = 1`` means the component IS that plane).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from mpd_modadd_2951 import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import run_parameter_decomposition
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    model = build_model(run["config"])
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    p = run["config"]["p"]
    rows = model.W_E.detach().double()[:p].numpy()
    request = {"schema": "gam.mpd-request", "schema_version": 1,
               "operation": {"kind": "invariant_components", "rows": "rows"}}
    out = run_parameter_decomposition(request, {"rows": rows})
    report, arrays = out.report["result"], out.arrays
    coords = arrays["coordinates"]
    a = np.arange(p)
    D = rows - rows.mean(0)
    planes = {}
    for k in range(1, (p - 1) // 2 + 1):
        m = np.stack([np.cos(2 * math.pi * k * a / p), np.sin(2 * math.pi * k * a / p)], 1)
        planes[k] = (np.linalg.qr(m)[0], float(((m.T @ D) ** 2).sum()))
    total = sum(v for _, v in planes.values())
    key = sorted(planes, key=lambda k: -planes[k][1])[:6]
    print(f"[inv] key frequencies (power share): {[(k, round(planes[k][1] / total, 3)) for k in key]}", flush=True)
    rows_out = []
    for comp in report["components"][:12]:
        q = np.linalg.qr(coords[:, comp])[0]
        scores = {k: float(np.prod(np.linalg.svd(q.T @ planes[k][0], compute_uv=False)[:min(2, len(comp))]))
                  for k in planes}
        best = max(scores, key=scores.get)
        rows_out.append({"directions": comp, "dim": len(comp), "best_frequency": best, "overlap": scores[best]})
        print(f"[inv] component dim {len(comp)} {comp}: best plane k={best} overlap {scores[best]:.4f}", flush=True)
    sizes = [len(c) for c in report["components"]]
    print(f"[inv] {len(sizes)} components, sizes {sorted(sizes, reverse=True)[:12]}; band {report['band']}", flush=True)
    with open(args.out, "w") as handle:
        json.dump({"key": key, "components": rows_out, "sizes": sizes, "band": report["band"]}, handle, indent=1)


if __name__ == "__main__":
    main()
