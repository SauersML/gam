"""Phase 1: does the REML manifold fit now SURVIVE real frontier activations?

Context: the manifold SAE's REML fit historically ABORTED on real LM
activations for every circle config (dictionary co-collapse -> indefinite
joint Hessian -> arrow-Schur Schur Cholesky non-PD pivot -> every seed
rejected -> RemlConvergenceError). The self-concordant barrier fix
(4b198bc72 dense rank-1 + 58df43cfc framed SparseRankOnePenaltyOp) landed
2026-07-07 and killed that genus on the synthetic gates; this script is the
first REAL-DATA re-validation.

Data: layer-40 residual-stream activations from the datasci corpora
(default /models/k25_tokens/<category>, Kimi-K2.5, hidden 7168), row-sampled
and PCA-reduced on GPU. Configs sweep the exact historical failure family
(circle atoms, D in {2,4,8}, K in {1,2}) plus mid-size research shapes.

Each config records: survived / RemlConvergenceError / other exception,
wall time, reconstruction EV, and the certificate diagnostics when the fit
completes. Output JSON is append-per-config so a crash loses nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


def load_slice(root: Path, category: str, n_rows: int, seed: int, cache_dir: Path | None) -> np.ndarray:
    """Row-sample the activation memmap, with an atomic shared cache so parallel
    jobs and restarts pay the 700GB-memmap gather exactly once per slice key."""
    cache = None
    if cache_dir is not None:
        cache = cache_dir / f"slice_{category}_n{n_rows}_s{seed}.npy"
        if cache.exists():
            print(f"[phase1] slice cache hit: {cache}", flush=True)
            return np.load(cache)
    acts = np.load(root / category / "activations.npy", mmap_mode="r")
    n_total = acts.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=min(n_rows, n_total), replace=False))
    z = np.asarray(acts[idx], dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # np.save appends ".npy" unless the name already ends with it — the tmp
        # name must carry the suffix or the atomic rename source never exists.
        tmp = cache.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(tmp, z)
        os.replace(tmp, cache)  # atomic: concurrent writers race harmlessly
    return z


def already_done(out: Path) -> set[str]:
    """Config names already recorded in the output JSONL — the resume set."""
    done: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("name"):
                done.add(rec["name"])
    return done


class PcaCache:
    """One SVD for the whole sweep; every config slices its top-d projection.

    The economy SVD of the (n, 7168) slice is the dominant preprocessing cost —
    recomputing it per config multiplied it by the config count for no reason.
    """

    def __init__(self, z: np.ndarray, device: str) -> None:
        import torch

        with torch.no_grad():
            t = torch.from_numpy(z).to(device=device, dtype=torch.float32)
            t -= t.mean(dim=0, keepdim=True)
            _, s, vh = torch.linalg.svd(t, full_matrices=False)
            self._t = t
            self._s = s
            self._vh = vh

    def project(self, d: int):
        import torch

        with torch.no_grad():
            proj = (self._t @ self._vh[:d].T).cpu().numpy().astype(np.float64)
            ev_frac = float((self._s[:d] ** 2).sum() / (self._s**2).sum())
        return proj, ev_frac


def run_config(z: np.ndarray, cfg: dict) -> dict:
    import gamfit

    out = dict(cfg)
    t0 = time.time()
    try:
        model = gamfit.sae_manifold_fit(
            z,
            K=cfg["K"],
            d_atom=cfg.get("d_atom", 1),
            atom_topology=cfg.get("atom_topology", "circle"),
            top_k=cfg.get("top_k"),
            n_iter=cfg.get("n_iter", 50),
            random_state=cfg.get("random_state", 0),
        )
        out["status"] = "survived"
        out["reconstruction_r2"] = float(model.reconstruction_r2)
        out["penalized_loss_score"] = (
            None if model.penalized_loss_score is None else float(model.penalized_loss_score)
        )
        diag = model.diagnostics or {}
        out["diagnostics_keys"] = sorted(diag.keys())
        for key in ("cocollapse_reseeds", "collapse_events", "stopped_reason"):
            if key in diag:
                out[key] = diag[key]
        if model.curvature_report is not None:
            out["kappa_hat"] = [a.get("kappa_hat") for a in model.curvature_report.get("atoms", [])]
        if model.incoherence_report is not None:
            out["mu_hat"] = model.incoherence_report.get("mu_hat")
    except Exception as exc:  # noqa: BLE001 - the exception CLASS is the result here
        out["status"] = type(exc).__name__
        out["error"] = str(exc)[:2000]
        out["traceback_tail"] = traceback.format_exc()[-1500:]
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/models/k25_tokens"))
    ap.add_argument("--category", default="emotions")
    ap.add_argument("--n-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("phase1_results.jsonl"))
    ap.add_argument("--only", default=None, help="comma-separated config names to run")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="shared dir for the sampled-slice cache (resume + cross-job reuse)")
    args = ap.parse_args()

    # Three families:
    #  repro_* — the historical co-collapse family (all aborted pre-fix); kept
    #            as the regression proof for the barrier fix.
    #  mid_*   — modest undercomplete warm-ups.
    #  oc_*    — the REAL manifold-SAE regime: overcomplete dictionaries
    #            (atoms >> dims), per-row identifiability carried by top_k
    #            sparsity + curvature. This is genus-2 frontier (pairwise
    #            barrier known-inadequate pre-Jeffreys), so the K/D survival
    #            threshold itself is the measurement.
    configs = [
        dict(name="repro_D2_K1", d_pca=2, K=1),
        dict(name="repro_D4_K1", d_pca=4, K=1),
        dict(name="repro_D4_K2", d_pca=4, K=2),
        dict(name="repro_D8_K2", d_pca=8, K=2),
        dict(name="mid_D32_K4", d_pca=32, K=4, top_k=2),
        dict(name="mid_D64_K8", d_pca=64, K=8, top_k=3),
        dict(name="wide_D128_K16", d_pca=128, K=16, top_k=4, n_iter=40),
        dict(name="oc_D16_K64", d_pca=16, K=64, top_k=4, n_iter=40),
        dict(name="oc_D32_K128", d_pca=32, K=128, top_k=4, n_iter=40),
        dict(name="oc_D32_K256", d_pca=32, K=256, top_k=4, n_iter=30),
        dict(name="oc_D64_K512", d_pca=64, K=512, top_k=8, n_iter=30),
    ]
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in wanted]

    done = already_done(args.out)
    configs = [c for c in configs if c["name"] not in done]
    if done:
        print(f"[phase1] resume: skipping {sorted(done)}", flush=True)
    if not configs:
        print("[phase1] ALL CONFIGS DONE (resume: nothing left)", flush=True)
        return

    print(f"[phase1] loading {args.n_rows} rows from {args.root}/{args.category}", flush=True)
    z_raw = load_slice(args.root, args.category, args.n_rows, args.seed, args.cache_dir)
    print(f"[phase1] slice {z_raw.shape} dtype={z_raw.dtype}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pca = PcaCache(z_raw, args.device)
    for cfg in configs:
        proj, ev_frac = pca.project(cfg["d_pca"])
        cfg["pca_ev_fraction"] = round(ev_frac, 4)
        cfg["category"] = args.category
        cfg["n_rows"] = int(proj.shape[0])
        print(f"[phase1] {cfg['name']}: fit begins (pca_ev={ev_frac:.3f})", flush=True)
        result = run_config(proj, cfg)
        with args.out.open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        print(
            f"[phase1] {cfg['name']}: {result['status']} "
            f"r2={result.get('reconstruction_r2')} t={result['wall_seconds']}s",
            flush=True,
        )
    print("[phase1] ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
