#!/usr/bin/env python3
"""#2263 item 4 — the L20 weekday/month structure_certificate panel + #2266 external arm.

The #2263 closure gate for `structure_certificate` on real activations:

  * NATIVE panel (the historical 0/39 evidence set): every attempted native fit on
    a real weekday/month feature at an intervention layer must EITHER mint a
    genuinely converged fit and a parseable, NON-EMPTY structure certificate, OR
    return its TYPED non-convergence error. Closure requires the intended fits to
    MINT; a certificate fabricated from a non-converged fit, or an untyped crash,
    is a hard failure.

  * EXTERNAL arm (#2266): feed externally-trained (torch-lane) arrays into
    `gamfit.sae_manifold_certify_external(..., run_structure_search=True)` and
    require a parseable NON-EMPTY structure certificate. A parseable-but-empty
    certificate is not acceptance.

Data:
  --acts-dir DIR  a directory of real (N, p) mean-centered .npy activation slices
                  named "<feature>_<layer>.npy" (e.g. weekday_L20.npy). Missing
                  (feature, layer) slices are skipped and reported, not failed.
  Absent --acts-dir, a synthetic ring fallback stands in: weekday = 7 phases,
  month = 12 phases, evenly spaced on a unit circle in a random 2-plane of R^p
  plus noise. Clean ring structure SHOULD mint, so the fallback self-tests the
  harness and the certificate machinery end to end.

  --torch-state PATH.npz  real torch-trained arrays for the external arm
                  (keys: decoder_blocks_*, coords_*, logits, geometry_plans meta,
                  reg state). Absent, the external arm uses the #2266 honest
                  replay: fit NATIVELY, pull the genuinely-converged arrays, and
                  feed them back through certify_external (labelled external_replay,
                  distinct from external_torch).

Prints one machine-parseable RESULT line and a PASS / PARTIAL / FAIL VERDICT.
PASS = all intended fits mint. PARTIAL = some honest typed refusals, none
fabricated. FAIL = any fabricated certificate or untyped crash.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


def ring_feature(n_phases, n, p, radius, noise, seed):
    """N tokens on `n_phases` evenly-spaced points of a unit circle in a random
    2-plane of R^p, mean-centered. Clean, low-rank, genuinely circular — the
    structure a periodic atom should certify."""
    rng = np.random.default_rng(seed)
    frame, _ = np.linalg.qr(rng.standard_normal((p, 2)))
    phase_idx = rng.integers(0, n_phases, size=n)
    angles = 2.0 * math.pi * phase_idx / n_phases
    coords = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    x = coords @ frame.T + noise * rng.standard_normal((n, p))
    x -= x.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_slice(acts_dir, feature, layer):
    path = os.path.join(acts_dir, f"{feature}_{layer}.npy")
    if not os.path.exists(path):
        return None
    x = np.load(path).astype(np.float64)
    if x.ndim != 2:
        raise SystemExit(f"[#2263] {path} must be 2-D (N, p); got {x.shape}")
    x -= x.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(x, dtype=np.float32)


def parse_certificate(raw):
    """Return (n_entries, n_confirmed) for a parseable non-empty cert, else None."""
    if raw is None:
        return None
    try:
        cert = json.loads(raw)
    except (TypeError, ValueError):
        return None
    entries = cert.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        return None
    if not all(math.isfinite(float(e.get("log_e", "nan"))) for e in entries):
        return None
    n_conf = sum(1 for e in entries if e.get("confirmed"))
    return len(entries), n_conf


def is_typed_nonconvergence(exc):
    msg = str(exc).lower()
    typed = ("did not converge" in msg or "off-optimum" in msg
             or "inner solve" in msg or "objective stalled" in msg
             or "refusing to rank" in msg)
    return typed


def preflight():
    import gamfit

    ver = getattr(gamfit, "__version__", "?")
    where = os.path.dirname(getattr(gamfit, "__file__", "?"))
    for name in ("sae_manifold_fit", "sae_manifold_certify_external"):
        if not hasattr(gamfit, name):
            raise SystemExit(f"[preflight] gamfit {ver} at {where} has no {name}")
    # Prove the fit accepts run_structure_search and exposes the certificate field.
    rng = np.random.default_rng(0)
    x = ring_feature(7, 210, 16, 1.0, 0.05, seed=0)
    try:
        m = gamfit.sae_manifold_fit(
            X=x, K=1, d_atom=2, atom_topology="circle", assignment="softmax",
            n_iter=3, random_state=0, run_structure_search=True)
    except TypeError as exc:
        raise SystemExit(
            f"[preflight] gamfit {ver} at {where} rejects run_structure_search: {exc}; "
            f"upgrade the wheel.")
    if not hasattr(m, "structure_certificate_json"):
        raise SystemExit(
            f"[preflight] fitted model has no structure_certificate_json; upgrade the wheel.")
    print(f"[#2263] preflight OK: gamfit {ver} at {where}", flush=True)


def fit_native(x, feature, seed, n_iter):
    """One native panel fit. Returns a classified row."""
    import gamfit

    # 7 weekday phases → K up to a few atoms; a single periodic atom already
    # certifies a clean ring, and structure search may confirm/contest more.
    k = 2
    t0 = time.time()
    outcome, detail, cert = "minted", "", None
    try:
        m = gamfit.sae_manifold_fit(
            X=np.ascontiguousarray(x), K=k, d_atom=2, atom_topology="circle",
            assignment="ordered_beta_bernoulli", n_iter=n_iter, learning_rate=0.04,
            random_state=seed, run_structure_search=True)
    except Exception as exc:  # noqa: BLE001 — classify, do not swallow
        detail = str(exc).splitlines()[0][:200]
        outcome = "typed_refusal" if is_typed_nonconvergence(exc) else "untyped_crash"
        dt = time.time() - t0
        return {"feature": feature, "outcome": outcome, "detail": detail,
                "cert_entries": None, "cert_confirmed": None, "seconds": dt}
    dt = time.time() - t0
    cert = parse_certificate(getattr(m, "structure_certificate_json", None))
    if cert is None:
        # Fit returned but no parseable non-empty certificate despite requesting
        # structure search: a fabricated / empty certificate is the hard failure.
        outcome, detail = "fabricated_or_empty", "structure search requested but no parseable non-empty certificate"
        return {"feature": feature, "outcome": outcome, "detail": detail,
                "cert_entries": None, "cert_confirmed": None, "seconds": dt}
    entries, confirmed = cert
    return {"feature": feature, "outcome": "minted", "detail": "",
            "cert_entries": entries, "cert_confirmed": confirmed, "seconds": dt}


def external_arm(x, seed, n_iter):
    """#2266 external-state cert with structure search. Honest replay: fit natively,
    pull the genuinely-converged arrays, feed them back through certify_external."""
    import gamfit

    row = {"arm": "external_replay"}
    t0 = time.time()
    try:
        fit = gamfit.sae_manifold_fit(
            X=np.ascontiguousarray(x), K=2, d_atom=2, atom_topology="circle",
            assignment="ordered_beta_bernoulli", n_iter=n_iter, learning_rate=0.04,
            random_state=seed, run_structure_search=False)
    except Exception as exc:  # noqa: BLE001
        row.update(outcome=("typed_refusal" if is_typed_nonconvergence(exc)
                            else "untyped_crash"),
                   detail=str(exc).splitlines()[0][:200], seconds=time.time() - t0)
        return row
    try:
        report = gamfit.sae_manifold_certify_external(
            X=x,
            geometry_plans=list(fit.geometry_plans),
            decoder_blocks=[np.asarray(b, dtype=float) for b in fit.decoder_blocks],
            t_init=[np.asarray(b, dtype=float) for b in fit.coords],
            a_init=np.asarray(fit.low_level_logits, dtype=float),
            log_lambda_smooth=[float(v) for v in fit.selected_log_lambda_smooth],
            log_ard=[[float(v) for v in a] for a in fit.selected_log_ard],
            assignment=fit.assignment, alpha=float(fit.alpha), tau=float(fit.tau),
            log_lambda_sparse=float(fit.selected_log_lambda_sparse),
            tier0_mean=np.asarray(fit.training_mean, dtype=float),
            tier0_scale=np.asarray(fit.tier0_scale, dtype=float),
            learnable_alpha=bool(fit.learnable_alpha),
            top_k=None if fit.top_k is None else int(fit.top_k),
            threshold_gate_threshold=float(fit.threshold_gate_threshold),
            run_structure_search=True)
    except Exception as exc:  # noqa: BLE001
        row.update(outcome=("typed_refusal" if is_typed_nonconvergence(exc)
                            else "untyped_crash"),
                   detail=str(exc).splitlines()[0][:200], seconds=time.time() - t0)
        return row
    dt = time.time() - t0
    status = report.get("status")
    raw_cert = report.get("structure_certificate")
    cert = parse_certificate(raw_cert if isinstance(raw_cert, str)
                             else json.dumps(raw_cert) if raw_cert else None)
    if status == "certified" and cert is not None:
        row.update(outcome="minted", detail="", status=status,
                   cert_entries=cert[0], cert_confirmed=cert[1], seconds=dt)
    else:
        row.update(outcome="fabricated_or_empty", status=status,
                   detail="external cert with run_structure_search=True was empty/absent",
                   cert_entries=None, seconds=dt)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default=None,
                    help="dir of <feature>_<layer>.npy real activation slices")
    ap.add_argument("--features", nargs="+", default=["weekday", "month"])
    ap.add_argument("--layers", nargs="+", default=["L20"])
    ap.add_argument("--n", type=int, default=1400, help="synthetic tokens per feature")
    ap.add_argument("--p", type=int, default=64, help="synthetic ambient dim")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_2263_panel.jsonl")
    args = ap.parse_args()

    preflight()

    phases = {"weekday": 7, "month": 12}
    native_rows, skipped = [], []
    for f_idx, feature in enumerate(args.features):
        for layer in args.layers:
            if args.acts_dir is not None:
                x = load_slice(args.acts_dir, feature, layer)
                if x is None:
                    skipped.append(f"{feature}_{layer}")
                    continue
                src = "real"
            else:
                # Deterministic per-feature seed offset (PYTHONHASHSEED-independent).
                x = ring_feature(phases.get(feature, 7), args.n, args.p, 1.0,
                                 args.noise, seed=args.seed + 101 * (f_idx + 1))
                src = "synthetic_ring"
            print(f"[#2263] NATIVE fit {feature}/{layer} src={src} X{tuple(x.shape)}",
                  flush=True)
            row = fit_native(x, f"{feature}/{layer}", args.seed, args.n_iter)
            row["src"] = src
            print(f"[#2263]   {feature}/{layer}: outcome={row['outcome']} "
                  f"cert_entries={row['cert_entries']} ({row['seconds']:.0f}s) "
                  f"{row['detail']!r}", flush=True)
            native_rows.append(row)

    # External arm on a clean synthetic ring (or the first available real slice).
    ext_x = None
    if args.acts_dir is not None:
        for feature in args.features:
            for layer in args.layers:
                ext_x = load_slice(args.acts_dir, feature, layer)
                if ext_x is not None:
                    break
            if ext_x is not None:
                break
    if ext_x is None:
        ext_x = ring_feature(7, args.n, args.p, 1.0, args.noise, seed=args.seed)
    print("[#2263] EXTERNAL arm (certify_external, run_structure_search=True)", flush=True)
    ext_row = external_arm(ext_x, args.seed, args.n_iter)
    print(f"[#2263]   external: {ext_row}", flush=True)

    all_rows = native_rows + [ext_row]
    fabricated = [r for r in all_rows
                  if r["outcome"] in ("fabricated_or_empty", "untyped_crash")]
    refused = [r for r in all_rows if r["outcome"] == "typed_refusal"]
    minted = [r for r in all_rows if r["outcome"] == "minted"]
    if fabricated:
        verdict = "FAIL"
    elif refused:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"

    result = {
        "issue": 2263, "panel": "structure_certificate",
        "native": native_rows, "external": ext_row, "skipped": skipped,
        "n_minted": len(minted), "n_typed_refusal": len(refused),
        "n_fabricated_or_untyped": len(fabricated), "verdict": verdict,
    }
    print("[#2263] RESULT " + json.dumps(result), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[#2263] VERDICT={verdict} minted={len(minted)} "
          f"typed_refusal={len(refused)} fabricated/untyped={len(fabricated)} "
          f"skipped={skipped}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
