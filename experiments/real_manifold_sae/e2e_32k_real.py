"""E2E massively-overcomplete TRUE-MANIFOLD SAE on real frontier activations.

The goal contract: a K=32,000 curved-atom `sae_manifold_fit` must RUN end to end
on real activations, must not lose to a matched 32k LINEAR SAE (losing = bug),
must not be slow (slow = bug), and every user-facing feature must work on the
resulting fit (each feature failure is recorded as a named bug).

Arms (identical PCA-reduced data):
  linear32k   — block_sparse_dictionary_fit, 32,000 blocks of size 1 (a plain
                TopK linear SAE: pure directions, the baseline to beat).
  manifold32k — sae_manifold_fit, K=32,000 circle atoms (d_atom=1, jumprelu
                assignment: per-row independent, streams at massive K), the
                known-good convergence regime from bench/massive_k_manifold_validate.

Feature battery on the manifold fit (each recorded ok/error):
  summary / repr / description_length (bits per token), save->load roundtrip
  (shape-band survival), out-of-sample encode, curvature / incoherence /
  diagnostics reports, steer smoke test.

Output JSONL is append-per-record so a crash loses nothing. Run through
heimdall (mandatory on datasci nodes).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1024**3 if sys.platform == "darwin" else 1024**2
    return r / div


def load_slice(root: Path, category: str, n_rows: int, seed: int) -> np.ndarray:
    acts = np.load(root / category / "activations.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(acts.shape[0], size=min(n_rows, acts.shape[0]), replace=False))
    return np.asarray(acts[idx], dtype=np.float32)


def gpu_pca(z: np.ndarray, d: int, device: str):
    import torch

    with torch.no_grad():
        t = torch.from_numpy(z).to(device=device, dtype=torch.float32)
        t -= t.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(t, full_matrices=False)
        proj = (t @ vh[:d].T).cpu().numpy().astype(np.float64)
        ev_frac = float((s[:d] ** 2).sum() / (s**2).sum())
    return proj, ev_frac


def emit(out: Path, rec: dict) -> None:
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[e2e32k] {rec.get('record')}: {json.dumps(rec)[:240]}", flush=True)


class _LinearReused(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/models/k25_tokens"))
    ap.add_argument("--category", default="emotions")
    ap.add_argument("--n-rows", type=int, default=60000)
    ap.add_argument("--n-oos", type=int, default=2000)
    ap.add_argument("--p", type=int, default=512)
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--assignment", default="jumprelu",
                    help="manifold gate: jumprelu | topk (sparsity by construction)")
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("e2e_32k_real.jsonl"))
    args = ap.parse_args()

    import gamfit

    z_raw = load_slice(args.root, args.category, args.n_rows + args.n_oos, args.seed)
    proj, ev_frac = gpu_pca(z_raw, args.p, args.device)
    train, oos = proj[: args.n_rows], proj[args.n_rows :]
    print(f"[e2e32k] train {train.shape} oos {oos.shape} pca_ev={ev_frac:.3f}", flush=True)
    base = dict(category=args.category, n_rows=args.n_rows, p=args.p, k=args.k,
                top_k=args.top_k, pca_ev=round(ev_frac, 4))

    # ── ARM 1: matched 32k LINEAR SAE (the baseline to beat) ────────────────
    # Resumable: a completed linear arm in the output JSONL is reused, not
    # refit — the baseline costs ~1 GPU-hour and its number is deterministic
    # at fixed (category, n_rows, p, k, top_k, seed).
    lin_ev = None
    lin_wall = None
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("record") == "linear32k" and r.get("status") == "ok"
                    and r.get("k") == args.k and r.get("p") == args.p
                    and r.get("n_rows") == args.n_rows):
                lin_ev = float(r["ev"])
                lin_wall = float(r["wall_s"])
                print(f"[e2e32k] linear32k reused from {args.out}: ev={lin_ev}", flush=True)
                break
    t0 = time.perf_counter()
    try:
        if lin_ev is not None:
            raise _LinearReused
        lin = gamfit.block_sparse_dictionary_fit(
            train, args.k, block_size=1, block_topk=args.top_k, max_epochs=30,
        )
        lin_wall = time.perf_counter() - t0
        lin_ev = float(lin.explained_variance)
        emit(args.out, {**base, "record": "linear32k", "status": "ok",
                        "ev": lin_ev, "wall_s": round(lin_wall, 1),
                        "peak_rss_gb": round(peak_rss_gb(), 2)})
    except _LinearReused:
        pass
    except Exception as exc:  # noqa: BLE001 - the class IS the datum
        emit(args.out, {**base, "record": "linear32k", "status": type(exc).__name__,
                        "error": str(exc)[:800], "wall_s": round(time.perf_counter() - t0, 1)})

    # ── ARM 2: the TRUE-MANIFOLD 32k fit ─────────────────────────────────────
    model = None
    t0 = time.perf_counter()
    try:
        model = gamfit.sae_manifold_fit(
            train,
            K=args.k,
            d_atom=1,
            atom_topology="circle",
            assignment=args.assignment,
            n_iter=args.n_iter,
            random_state=args.seed,
            top_k=args.top_k,
            sparsity_weight=0.01,
            smoothness_weight=0.01,
            isometry_weight=0.0,
            learning_rate=1.0,
            ard_per_atom=False,
        )
        man_wall = time.perf_counter() - t0
        man_r2 = float(model.reconstruction_r2)
        emit(args.out, {**base, "record": "manifold32k:" + args.assignment, "status": "ok",
                        "reconstruction_r2": man_r2, "wall_s": round(man_wall, 1),
                        "peak_rss_gb": round(peak_rss_gb(), 2),
                        "beats_linear": (lin_ev is not None and man_r2 >= lin_ev),
                        "wall_ratio_vs_linear": (round(man_wall / lin_wall, 2)
                                                  if lin_wall else None)})
    except Exception as exc:  # noqa: BLE001
        emit(args.out, {**base, "record": "manifold32k:" + args.assignment, "status": type(exc).__name__,
                        "error": str(exc)[:800],
                        "traceback_tail": traceback.format_exc()[-1200:],
                        "wall_s": round(time.perf_counter() - t0, 1)})

    if model is None:
        print("[e2e32k] manifold fit failed — feature battery skipped", flush=True)
        return

    # ── FEATURE BATTERY: every failure is a named bug ────────────────────────
    def feature(name: str, fn) -> None:
        t = time.perf_counter()
        try:
            payload = fn()
            emit(args.out, {"record": f"feature:{name}", "status": "ok",
                            "wall_s": round(time.perf_counter() - t, 2),
                            "payload": payload})
        except Exception as exc:  # noqa: BLE001
            emit(args.out, {"record": f"feature:{name}", "status": type(exc).__name__,
                            "error": str(exc)[:500],
                            "wall_s": round(time.perf_counter() - t, 2)})

    feature("summary", lambda: {k: v for k, v in list(model.summary().items())[:8]
                                 if isinstance(v, (int, float, str, type(None)))})
    feature("repr", lambda: repr(model)[:200])
    feature("description_length", lambda: model.description_length())
    feature("diagnostics", lambda: sorted((model.diagnostics or {}).keys())[:12])
    feature("curvature_report", lambda: None if model.curvature_report is None
            else {"n_atoms": len(model.curvature_report.get("atoms", []))})
    feature("incoherence_report", lambda: None if model.incoherence_report is None
            else model.incoherence_report.get("mu_hat"))

    def roundtrip():
        d = model.to_dict()
        m2 = type(model).from_dict(d)
        return {"r2_matches": abs(float(m2.reconstruction_r2)
                                   - float(model.reconstruction_r2)) < 1e-9}
    feature("save_load_roundtrip", roundtrip)

    def oos_encode():
        codes = model.encode(oos)
        arr = np.asarray(codes if not isinstance(codes, tuple) else codes[0])
        return {"oos_rows": int(arr.shape[0])}
    feature("oos_encode", oos_encode)

    def steer_smoke():
        s = model.steer(atom=0, delta=0.1)
        keys = s.keys() if hasattr(s, "keys") else dir(s)
        return {"has_predicted_nats": "predicted_nats" in list(keys)}
    feature("steer_smoke", steer_smoke)

    print("[e2e32k] DONE", flush=True)


if __name__ == "__main__":
    main()
