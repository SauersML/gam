#!/usr/bin/env python3
"""#2234 — the crosscoder transport test, run fit-free on real activations.

`certify_chart_transfer(operator, input_generator, output_generator)` asks two
questions about a square map A between two charts:

  transport_defect    = ||A^T A - I||_F      -- does the layer CARRY the atom
                                                (near-isometry) or COMPUTE with it?
  equivariance_defect = ||A G_in - G_out A||_F -- does A commute with the charts'
                                                infinitesimal symmetry generator?

For a circle chart the generator is the rotation `G = [[0,-1],[1,0]]`, so
equivariance is exactly the phase-transport law this issue's acceptance test
measures: A is equivariant iff it acts on the chart as a rotation-and-scale,
i.e. iff transporting a token from layer L to layer L' is a pure phase shift.

Nothing here needs a fitted SAE. The charts come from the top-2 plane of the
SAME real token cloud at two layers, so A is the honest least-squares transfer
between two real charts of the same tokens, and the certificate is the shipped
Rust one.

Controls, because a defect is only informative against a scale:
  identity     A = I                     (perfect transport, perfect equivariance)
  rotation     A = R(phi)                (a pure phase shift: the law HOLDING)
  shear        A = [[1, s], [0, 1]]      (the law failing in a known way)
  random       A ~ N(0, 1)               (no structure)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


MONTHS = [" January", " February", " March", " April", " May", " June",
          " July", " August", " September", " October", " November",
          " December"]

# infinitesimal generator of rotation on a circle chart
G_CIRCLE = np.array([[0.0, -1.0], [1.0, 0.0]])


def chart_plane(cloud):
    """Top-2 plane of a cloud, orthonormal rows, plus the coords in it."""
    mu = cloud.mean(0)
    xc = cloud - mu
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(w)[::-1][:2]
    B = v[:, order].T                       # (2, p), orthonormal
    return mu, B, xc @ B.T, w[order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.expanduser("~/f2_site"))
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--split-module",
                    default=os.path.expanduser("~/i2502-baselines"))
    ap.add_argument("--layer-in", type=int, default=16)
    ap.add_argument("--layer-out", type=int, default=22)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.site)
    sys.path.insert(0, args.split_module)
    import gamfit
    from issue_2502_doc_split import row_side, split_manifest
    from transformers import AutoTokenizer

    H = args.harvest
    doc_ids = np.load(os.path.join(H, "doc_ids.npy"))
    man = split_manifest(doc_ids)
    side = row_side(doc_ids)
    token_ids = np.load(os.path.join(H, "token_ids.npy"))

    prov = {"node": platform.node(), "gamfit": gamfit.__version__,
            "code_pin": open(os.path.expanduser("~/f2_wheel/PIN.txt")).read().strip(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",
                                                   "<unset>"),
            "split_hash": man["split_hash"],
            "harvest_doc_digest": man["harvest_doc_digest"],
            "model": "Qwen/Qwen3.5-4B-Base",
            "layer_in": args.layer_in, "layer_out": args.layer_out}
    log(f"provenance {json.dumps(prov)}")

    def emit(rec):
        rec["provenance"] = prov
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base",
                                        trust_remote_code=True)
    cls = [tok.encode(n, add_special_tokens=False)[0] for n in MONTHS
           if len(tok.encode(n, add_special_tokens=False)) == 1]
    rows = np.flatnonzero(
        np.isin(token_ids, np.asarray(cls, dtype=token_ids.dtype)) & ~side)
    log(f"month cloud: {rows.size} train-side rows, shared across both layers")

    charts = {}
    for L in (args.layer_in, args.layer_out):
        acts = np.load(os.path.join(H, f"resid_L{L}.npy"), mmap_mode="r")
        cloud = np.asarray(acts[rows], dtype=np.float64)
        mu, B, coords, eig = chart_plane(cloud)
        charts[L] = coords
        log(f"L{L} chart plane eigenvalues {eig.tolist()} "
            f"(ratio {eig[0]/eig[1]:.3f}), median radius "
            f"{float(np.median(np.linalg.norm(coords, axis=1))):.3f}")
        emit({"record": "chart", "layer": L,
              "plane_eigenvalues": [float(e) for e in eig],
              "n_rows": int(rows.size),
              "median_radius": float(np.median(np.linalg.norm(coords, axis=1)))})

    Cin, Cout = charts[args.layer_in], charts[args.layer_out]
    # unit-scale both charts so the transfer is about SHAPE, not layer norm growth
    sin = float(np.sqrt((Cin ** 2).sum(1).mean()))
    sout = float(np.sqrt((Cout ** 2).sum(1).mean()))
    Cin_n, Cout_n = Cin / sin, Cout / sout
    A, *_ = np.linalg.lstsq(Cin_n, Cout_n, rcond=None)      # (2, 2)
    A = np.ascontiguousarray(A.T)   # act on column coords: c_out = A c_in
    pred = Cin_n @ A.T
    r2 = 1.0 - float(((Cout_n - pred) ** 2).sum()) / float(
        ((Cout_n - Cout_n.mean(0)) ** 2).sum())
    log(f"measured transfer A = {A.tolist()}, chart-to-chart R2 = {r2:.4f}, "
        f"radius scale {sout/sin:.4f}")

    def run(name, Aop, extra=None):
        t0 = time.perf_counter()
        try:
            rep = gamfit.sae.certify_chart_transfer(
                np.ascontiguousarray(Aop),
                np.ascontiguousarray(G_CIRCLE),
                np.ascontiguousarray(G_CIRCLE))
            rec = {"record": "transfer", "arm": name,
                   "operator": [[float(x) for x in r] for r in Aop],
                   "report": json.loads(json.dumps(rep, default=str)),
                   "wall_s": time.perf_counter() - t0}
        except Exception as exc:  # noqa: BLE001
            rec = {"record": "transfer", "arm": name,
                   "operator": [[float(x) for x in r] for r in Aop],
                   "error": f"{type(exc).__name__}: {exc}"[:400],
                   "wall_s": time.perf_counter() - t0}
        if extra:
            rec.update(extra)
        emit(rec)
        rep = rec.get("report") or {}
        log(f"{name:22s} transport={rep.get('transport_defect')} "
            f"equivariance={rep.get('equivariance_defect')} "
            f"{rec.get('error','')}")
        return rec

    run("measured_L%d_to_L%d" % (args.layer_in, args.layer_out), A,
        {"chart_to_chart_r2": r2, "radius_scale": sout / sin})

    # GAUGE. Each chart's basis is an `eigh` output and eigenvector SIGNS are
    # arbitrary, so flipping one output basis vector flips the chart's
    # orientation and turns a rotation into a reflection — which ANTI-commutes
    # with G. The equivariance defect is therefore gauge-dependent and means
    # nothing until it is minimised over the chart's sign group; taken raw it
    # can report numpy's eigenvector convention as a broken law. (The transport
    # defect is gauge-invariant: A^T A is unchanged by an orthogonal left
    # factor, which is a useful self-check that the sweep is doing what it says.)
    best = None
    for s0 in (1, -1):
        for s1 in (1, -1):
            Ag = np.diag([s0, s1]) @ A
            rec = run(f"gauge_diag({s0:+d},{s1:+d})", Ag,
                      {"gauge": [s0, s1], "det": float(np.linalg.det(Ag))})
            rep = rec.get("report") or {}
            ed = rep.get("equivariance_defect")
            if ed is not None and (best is None or ed < best[0]):
                best = (ed, (s0, s1), Ag, rep.get("transport_defect"))
    if best is not None:
        ed, gauge, Ag, td = best
        # closest rotation-and-scale to the gauge-fixed transfer
        phi = float(np.arctan2(Ag[1, 0] - Ag[0, 1], Ag[0, 0] + Ag[1, 1]))
        scale = float(np.linalg.norm(Ag) / np.sqrt(2.0))
        R = scale * np.array([[math.cos(phi), -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        resid = float(np.linalg.norm(Ag - R))
        emit({"record": "gauge_minimised",
              "gauge": list(gauge),
              "equivariance_defect": ed, "transport_defect": td,
              "phase_radians": phi, "phase_turns": phi / (2 * math.pi),
              "scale": scale,
              "residual_frobenius": resid,
              "operator_frobenius": float(np.linalg.norm(Ag)),
              "phase_shift_fraction": 1.0 - resid / float(np.linalg.norm(Ag))})
        log(f"GAUGE-MINIMISED equivariance={ed:.6f} transport={td:.6f} "
            f"phase={phi:.4f} rad ({phi/(2*math.pi):+.4f} turns) "
            f"scale={scale:.4f} residual={resid:.6f}")

    # controls
    run("control_identity", np.eye(2))
    for phi in (0.1, 0.5, math.pi / 2):
        c, s = math.cos(phi), math.sin(phi)
        run(f"control_rotation_{phi:.3f}", np.array([[c, -s], [s, c]]),
            {"phi": phi})
    for sh in (0.05, 0.2, 1.0):
        run(f"control_shear_{sh:.2f}", np.array([[1.0, sh], [0.0, 1.0]]),
            {"shear": sh})
    rng = np.random.default_rng(0)
    run("control_random", rng.standard_normal((2, 2)))
    # a pure scale: isometric up to a factor, but still equivariant
    run("control_scale_2x", 2.0 * np.eye(2))

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
