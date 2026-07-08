"""Phase 2: curved-vs-flat T2 on frontier activations, warm-started from the
datasci team's own TopK SAE as the frozen T1 tier.

Protocol (mirrors run_curved_vs_flat_32b.py, self-contained for datasci):
  T1  : the team's trained TopK SAE checkpoint (W_enc/b_enc/W_dec/b_dec,
        16x overcomplete, k=64) run as a FROZEN forward pass on GPU. This is
        the strongest possible flat warm start: a converged production flat
        dictionary, not one we trained.
  T2  : gamfit.sae_manifold_fit_stagewise on the T1 RESIDUAL (PCA-reduced on
        GPU to --resid-dim for tractability; capture fraction reported), run
        per atom_topology in {circle, linear} on the IDENTICAL residual.
        Same budget, only the atom manifold type differs.

HEADLINE: dEV = EV_residual_tier(circle) - EV_residual_tier(linear), plus
composed EV over T1 and the certificate payloads (theta, basis kinds,
births, collapse events).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


def load_slice(root: Path, category: str, n_rows: int, seed: int) -> np.ndarray:
    acts = np.load(root / category / "activations.npy", mmap_mode="r")
    n_total = acts.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=min(n_rows, n_total), replace=False))
    return np.asarray(acts[idx], dtype=np.float32)


def t1_topk_forward(x: np.ndarray, ckpt_path: Path, k: int, device: str, chunk: int = 8192):
    """Frozen TopK-SAE forward exactly as trained: pre-acts on centered input,
    top-k by value on ReLU pre-activations, decode + b_dec."""
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    w_enc = state["W_enc"].to(device=device, dtype=torch.float32)   # (F, D)
    b_enc = state["b_enc"].to(device=device, dtype=torch.float32)   # (F,)
    w_dec = state["W_dec"].to(device=device, dtype=torch.float32)   # (D, F)
    b_dec = state["b_dec"].to(device=device, dtype=torch.float32)   # (D,)

    recon = np.empty_like(x)
    n_alive_sum = 0
    with torch.no_grad():
        for a in range(0, x.shape[0], chunk):
            xb = torch.from_numpy(x[a : a + chunk]).to(device=device, dtype=torch.float32)
            pre = (xb - b_dec) @ w_enc.T + b_enc                    # (b, F)
            pre = torch.relu(pre)
            vals, idx = torch.topk(pre, k, dim=1)
            lat = torch.zeros_like(pre)
            lat.scatter_(1, idx, vals)
            n_alive_sum += int((vals > 0).sum())
            xh = lat @ w_dec.T + b_dec
            recon[a : a + chunk] = xh.cpu().numpy()
    meta = {
        "ckpt": str(ckpt_path),
        "F": int(w_enc.shape[0]),
        "k": int(k),
        "mean_active": n_alive_sum / x.shape[0],
    }
    return recon, meta


def explained_variance(x: np.ndarray, xh: np.ndarray) -> float:
    """Standard EV against the mean baseline, float64 accumulation."""
    x64 = x.astype(np.float64)
    resid = x64 - xh.astype(np.float64)
    centered = x64 - x64.mean(axis=0, keepdims=True)
    return float(1.0 - (resid**2).sum() / (centered**2).sum())


def gpu_pca_reduce(r: np.ndarray, d: int, device: str):
    import torch

    with torch.no_grad():
        t = torch.from_numpy(r).to(device=device, dtype=torch.float32)
        mu = t.mean(dim=0, keepdim=True)
        t -= mu
        _, s, vh = torch.linalg.svd(t, full_matrices=False)
        reduced = (t @ vh[:d].T).cpu().numpy().astype(np.float64)
        capture = float((s[:d] ** 2).sum() / (s**2).sum())
    return reduced, capture


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def run_t2_arm(residual_reduced: np.ndarray, topo: str, args) -> dict:
    import gamfit

    t0 = time.time()
    rec: dict = {"atom_topology": topo}
    try:
        sw = gamfit.sae_manifold_fit_stagewise(
            residual_reduced,
            d_atom=args.d_atom,
            atom_topology=topo,
            assignment="ibp_map",
            max_births=args.max_births,
            min_effect_ev=0.0,
            structured_whitening=True,
            n_iter=args.n_iter,
            random_state=0,
        )
        recon = np.asarray(sw.fitted, dtype=np.float64)  # in-sample reconstruction
        rec["ev_residual_tier"] = explained_variance(residual_reduced, recon)
        for attr in (
            "k",
            "births_accepted",
            "births_rejected",
            "stopped_reason",
            "ev_trace",
            "collapse_events",
            "terminal_joint_reml",
            "terminal_data_fit",
        ):
            try:
                rec[attr] = _jsonable(getattr(sw, attr))
            except Exception as exc:  # noqa: BLE001
                rec[attr] = f"<err {exc}>"
        try:
            kinds = [str(getattr(a, "basis_kind", getattr(a, "basis", "?"))) for a in sw.atoms]
            rec["basis_kind_counts"] = dict(Counter(kinds))
            rec["n_atoms"] = len(kinds)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        rec["status"] = type(exc).__name__
        rec["error"] = str(exc)[:2000]
        rec["traceback_tail"] = traceback.format_exc()[-1200:]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-root", type=Path, default=Path("/models/k25_tokens"))
    ap.add_argument("--category", default="emotions")
    ap.add_argument("--ckpt", type=Path, default=Path("/models/sae/k25-145M-16x-k64_latest.pt"))
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--t1-k", type=int, default=64)
    ap.add_argument("--resid-dim", type=int, default=256)
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--max-births", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("phase2_results.json"))
    args = ap.parse_args()

    import gamfit

    print(f"[phase2] gamfit {gamfit.__version__}; slice {args.rows} rows of "
          f"{args.acts_root}/{args.category}", flush=True)
    x = load_slice(args.acts_root, args.category, args.rows, args.seed)
    print(f"[phase2] activations {x.shape}", flush=True)

    t1_recon, t1_meta = t1_topk_forward(x, args.ckpt, args.t1_k, args.device)
    ev_t1 = explained_variance(x, t1_recon)
    print(f"[phase2] T1 frozen TopK: EV={ev_t1:.4f} meta={t1_meta}", flush=True)

    residual = x - t1_recon
    residual_reduced, capture = gpu_pca_reduce(residual, args.resid_dim, args.device)
    print(f"[phase2] residual PCA {residual.shape[1]} -> {args.resid_dim} "
          f"(capture {capture:.3f})", flush=True)

    results = {
        "acts": f"{args.acts_root}/{args.category}",
        "rows": int(x.shape[0]),
        "D": int(x.shape[1]),
        "T1": {**t1_meta, "ev": ev_t1},
        "residual_pca": {"dim": args.resid_dim, "capture": capture},
        "T2_arms": {},
        "headline": {},
    }
    for topo in ("circle", "linear"):
        print(f"[phase2] === T2 {topo} (max_births={args.max_births}) ===", flush=True)
        rec = run_t2_arm(residual_reduced, topo, args)
        results["T2_arms"][topo] = rec
        print(f"[phase2] {topo}: ev_residual_tier={rec.get('ev_residual_tier')} "
              f"births={rec.get('births_accepted')} stop={rec.get('stopped_reason')} "
              f"wall={rec['wall_s']}s err={rec.get('error', '')[:200]}", flush=True)

    circle = results["T2_arms"]["circle"].get("ev_residual_tier")
    linear = results["T2_arms"]["linear"].get("ev_residual_tier")
    if circle is not None and linear is not None:
        results["headline"] = {
            "delta_ev_curved_minus_flat_residual_tier": circle - linear,
        }
        print(f"[phase2] HEADLINE dEV(curved-flat) = {circle - linear:+.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(results), indent=1))
    print(f"[phase2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
