#!/usr/bin/env python3
"""Controlled demonstration of the gam#2233 MDL crossover theorem + a validation
of the #1026 Eq-4 bits pipeline (bits_eq4 / arm_featurizers) on planted data.

Why this exists
---------------
The #1026 close reports bits-at-R^2 (their Eq. 4) alongside EV for hybrid-vs-flat
at a 32K overcomplete dictionary (gam#2233 task 3). On the real creditscope cloud
that is an MSI fit; here we plant circles we KNOW the geometry of and score the
two encodings with the EXACT zoo scorer (``bits_eq4.description_length``), so the
theorem's per-term decomposition is checkable against ground truth and the
scoring surface is exercised end-to-end before any expensive run.

Faithful setup (matches the real 32K/L0=32 regime)
--------------------------------------------------
The zoo's ``support_bits`` is a GLOBAL ``C(G, round(mean L0))`` -- it is blind to
sub-integer L0 differences, so a RARE planted feature shows zero support win (the
per-firing-token saving washes out in the mean). The real #1026 code is DENSE
(L0=32 active/token). We reproduce that: every token fires ``bg_l0`` background
linear atoms PLUS ``k_circ`` circle features. Then

    flat   L0 = bg_l0 + 2*k_circ   (each circle = 2 ambient dirs, code_dims 1)
    hybrid L0 = bg_l0 + 1*k_circ   (each circle = 1 curved atom, code_dims d+1=2)

so the mean L0 differs by an integer ``k_circ`` and the support term is live.

The theorem (gam#2233), specialized to a circle (d=1, ambient span s=2,
m=2H+1 basis atoms):

    * code bits   : flat sends s=2 scalars/circle, curved sends d+1=2 => EQUAL
                    (the surprise: circles win ZERO code bits).
    * support bits: flat lights 2 atoms/circle, curved lights 1 => saves k_circ
                    slots; each slot costs ~log2(G/L0) -- the term that SCALES
                    with overcompleteness (the 32K regime).
    * residual    : equal at matched recon.
    * dict bits   : curved SURCHARGE k_circ*(m-2)*P*0.5*log2(N)/N -- tiny.

=> EV-at-matched-actives (prices code) is THIN; Eq-4 bits (prices support+code+
resid) is WIDE. Reproduced here on data where the answer is known.

Pure numpy + the zoo scorer; no wheel/torch. Lean enough for the local box.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from bits_eq4 import description_length, make_fitted_featurizer, scorer_source


def build_cloud(n, p, *, k_circ, bg_l0, g_dict, radius, noise, seed):
    """Dense planted cloud: bg_l0 background linear atoms + k_circ circles, all
    active on every token. Returns (X, dirs, circ, bg_idx, recon_signal).

    ``dirs`` = orthonormal ambient frame (p, 2*k_circ+bg_l0); the first
    ``2*k_circ`` columns are the circle planes (u_j, v_j), the rest are the
    background linear directions. Atom slots in the shared G-dict:
      circle j flat atoms -> columns (2j, 2j+1); background atom b -> 2*k_circ+b.
    """
    rng = np.random.default_rng(seed)
    n_dir = 2 * k_circ + bg_l0
    assert n_dir <= p, "need p >= 2*k_circ + bg_l0"
    q, _ = np.linalg.qr(rng.standard_normal((p, n_dir)))
    dirs = q[:, :n_dir]  # (p, n_dir) orthonormal

    signal = np.zeros((n, p))
    # circles
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(n, k_circ))
    for j in range(k_circ):
        u, v = dirs[:, 2 * j], dirs[:, 2 * j + 1]
        signal += radius * (np.cos(thetas[:, j])[:, None] * u[None, :]
                            + np.sin(thetas[:, j])[:, None] * v[None, :])
    # background linear atoms (unit-ish Gaussian codes)
    bg_codes = rng.standard_normal((n, bg_l0)) * radius
    signal += bg_codes @ dirs[:, 2 * k_circ:].T
    X = signal + noise * rng.standard_normal((n, p))
    return X, dirs, thetas, bg_codes


def _lazy(fn):
    class _L:
        __slots__ = ()
        def __getitem__(self, idx):
            return fn(np.asarray(idx))
    return _L()


def flat_featurizer(X, dirs, *, k_circ, bg_l0, g_dict):
    """FLAT: every circle spends 2 flat atoms (its ambient dirs); background
    spends bg_l0 flat atoms. code_dims=1; dictionary = full G-atom flat dict.
    """
    n, p = X.shape
    n_used = 2 * k_circ + bg_l0
    codes = X @ dirs  # (N, n_used) projections onto the used frame
    gate = np.zeros((n, g_dict), dtype=np.float32)
    gate[:, :n_used] = np.abs(codes).astype(np.float32)
    recon = codes @ dirs.T

    def contribution(g):
        if g < n_used:
            d = dirs[:, g]
            return _lazy(lambda take, d=d: (X[take] @ d)[:, None] * d[None, :])
        return _lazy(lambda take: np.zeros((np.asarray(take).size, p)))

    return make_fitted_featurizer(
        name="flat", gate=gate, atom_contribution=contribution,
        code_dims=np.ones(g_dict, dtype=int), dictionary_params=g_dict * p,
        recon=recon, fit_seconds=0.0)


def hybrid_featurizer(X, dirs, *, k_circ, bg_l0, g_dict, n_harm):
    """HYBRID: each circle = ONE curved atom (code_dims=2: angle+amplitude);
    background still bg_l0 flat atoms. Same recon as flat. Dictionary = shared
    G-atom flat dict + k_circ curved atoms' basis surcharge (m=2H+1 per atom).
    """
    n, p = X.shape
    m = 2 * n_harm + 1
    # background flat atoms occupy the first bg_l0 flat slots; curved atoms are
    # appended after the G-dict.
    bg_dirs = dirs[:, 2 * k_circ:]
    bg_codes = X @ bg_dirs  # (N, bg_l0)
    n_atoms = g_dict + k_circ
    gate = np.zeros((n, n_atoms), dtype=np.float32)
    gate[:, :bg_l0] = np.abs(bg_codes).astype(np.float32)
    # curved atom amplitudes
    recon = bg_codes @ bg_dirs.T
    for j in range(k_circ):
        u, v = dirs[:, 2 * j], dirs[:, 2 * j + 1]
        cu, cv = X @ u, X @ v
        amp = np.sqrt(cu ** 2 + cv ** 2)
        gate[:, g_dict + j] = amp.astype(np.float32)
        recon = recon + cu[:, None] * u[None, :] + cv[:, None] * v[None, :]

    code_dims = np.ones(n_atoms, dtype=int)
    code_dims[g_dict:] = 2  # each curved atom: angle coord + amplitude

    def contribution(g):
        if g < bg_l0:
            d = bg_dirs[:, g]
            return _lazy(lambda take, d=d: (X[take] @ d)[:, None] * d[None, :])
        if g >= g_dict:
            j = g - g_dict
            u, v = dirs[:, 2 * j], dirs[:, 2 * j + 1]
            return _lazy(lambda take, u=u, v=v: (X[take] @ u)[:, None] * u[None, :]
                         + (X[take] @ v)[:, None] * v[None, :])
        return _lazy(lambda take: np.zeros((np.asarray(take).size, p)))

    return make_fitted_featurizer(
        name="hybrid", gate=gate, atom_contribution=contribution,
        code_dims=code_dims, dictionary_params=g_dict * p + k_circ * m * p,
        recon=recon, fit_seconds=0.0)


def _ev(X, recon):
    ss_res = float(np.sum((X - recon) ** 2))
    ss_tot = float(np.sum((X - X.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--p", type=int, default=96)
    ap.add_argument("--g-dict", type=int, default=8192,
                    help="overcomplete flat dict both arms carry")
    ap.add_argument("--k-circ", type=int, default=4, help="planted circle features")
    ap.add_argument("--bg-l0", type=int, default=28, help="dense background linear atoms/token")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--n-harm", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-g", type=int, default=32768,
                    help="report the analytic per-slot support cost log2(G/L0) at this G")
    ap.add_argument("--out", default="results/crossover_synthetic.jsonl")
    args = ap.parse_args()

    X, dirs, thetas, bg_codes = build_cloud(
        args.n, args.p, k_circ=args.k_circ, bg_l0=args.bg_l0,
        g_dict=args.g_dict, radius=args.radius, noise=args.noise, seed=args.seed)

    targets = (0.99, 0.95, 0.90, 0.80)
    # Score sequentially so the two (N, G) gates are never both resident.
    flat = flat_featurizer(X, dirs, k_circ=args.k_circ, bg_l0=args.bg_l0, g_dict=args.g_dict)
    ev_flat = _ev(X, flat.recon)
    # The synthetic corpus is scored on all N rows, so the estimation subsample IS
    # N here; the declared dictionary amortisation horizon is that same planted
    # token count. Passing it explicitly (never implicitly from the row count)
    # keeps the score well-defined under #2283.
    dl_flat = description_length(
        flat, X, amortization_horizon=args.n, r2_targets=targets
    )
    del flat

    hyb = hybrid_featurizer(X, dirs, k_circ=args.k_circ, bg_l0=args.bg_l0,
                            g_dict=args.g_dict, n_harm=args.n_harm)
    ev_hyb = _ev(X, hyb.recon)
    dl_hyb = description_length(
        hyb, X, amortization_horizon=args.n, r2_targets=targets
    )
    del hyb

    l0_flat = dl_flat["achieved_block_l0"]
    l0_hyb = dl_hyb["achieved_block_l0"]
    # Analytic per-slot support cost the theorem scales by (reported at report-g).
    slot_bits = math.log2(max(args.report_g, 2) / max(l0_hyb, 1.0))

    print(f"[crossover] N={args.n} p={args.p} G={args.g_dict} k_circ={args.k_circ} "
          f"bg_l0={args.bg_l0} radius={args.radius} noise={args.noise} H={args.n_harm}")
    print(f"[crossover] scorer = {scorer_source()}")
    print(f"[crossover] EV: flat={ev_flat:.6f}  hybrid={ev_hyb:.6f}  "
          f"margin={ev_hyb - ev_flat:+.2e}  <- the THIN scoreboard (matched recon)")
    print(f"[crossover] mean L0: flat={l0_flat:.2f} (~{args.bg_l0}+2*{args.k_circ})  "
          f"hybrid={l0_hyb:.2f} (~{args.bg_l0}+{args.k_circ})  "
          f"=> curved saves {l0_flat - l0_hyb:.1f} active slots/token")
    print(f"[crossover] support_bits: flat={dl_flat['support_bits']:.1f}  "
          f"hybrid={dl_hyb['support_bits']:.1f}  "
          f"(analytic slot cost log2(G/L0) at G={args.report_g}: {slot_bits:.1f} bits/slot)")
    print(f"\n  {'R2':>5} {'bits(flat)':>12} {'bits(hyb)':>12} {'d=flat-hyb':>12}"
          f"   (dsupport / dcode / dresid)  bits/token")
    rows = []
    for t in targets:
        key = f"bits_at_r2_{t:g}"
        bf, bh = dl_flat[key], dl_hyb[key]
        dsup = dl_flat["support_bits"] - dl_hyb["support_bits"]
        dcode = dl_flat[f"code_bits_at_r2_{t:g}"] - dl_hyb[f"code_bits_at_r2_{t:g}"]
        dres = dl_flat[f"resid_bits_at_r2_{t:g}"] - dl_hyb[f"resid_bits_at_r2_{t:g}"]
        print(f"  {t:>5.2f} {bf:>12.1f} {bh:>12.1f} {bf - bh:>12.1f}"
              f"   ({dsup:+.1f} / {dcode:+.1f} / {dres:+.1f})   "
              f"flat={bf / args.n:.3f} hyb={bh / args.n:.3f}")
        rows.append({"r2": t, "bits_flat": bf, "bits_hyb": bh, "delta_total": bf - bh,
                     "delta_support": dsup, "delta_code": dcode, "delta_resid": dres})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rec = {"kind": "crossover_synthetic", "n": args.n, "p": args.p,
           "g_dict": args.g_dict, "k_circ": args.k_circ, "bg_l0": args.bg_l0,
           "radius": args.radius, "noise": args.noise, "n_harm": args.n_harm,
           "seed": args.seed, "ev_flat": ev_flat, "ev_hyb": ev_hyb,
           "l0_flat": l0_flat, "l0_hyb": l0_hyb, "report_g": args.report_g,
           "slot_bits_at_report_g": slot_bits, "scorer": scorer_source(),
           "per_target": rows}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n[crossover] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
