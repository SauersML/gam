#!/usr/bin/env python
"""#1026 — autonomous HILLCLIMB that MAXIMIZES held-out reconstruction EV.

This is the optimization counterpart of ``sae_ev_vs_k_olmo.py``. Instead of
sweeping a fixed (K, topology) grid, it runs an iterative optimizer over the full
production ``sae_manifold_fit`` configuration space and CLIMBS held-out
reconstruction explained-variance (EV) on real OLMo-3-32B activations.

Objective (honest, no leakage):
  * PCA fit on TRAIN ONLY, project both splits, unit-RMS scale from TRAIN only
    (identical protocol to sae_ev_vs_k_olmo.py).
  * Each candidate = one production SAE fit on z_train; EV measured on z_test via
    ``m.reconstruct(z_test)`` (frozen decoder re-seated on held-out rows).

Search space (per-knob, see ``mutate``):
  * K                     dictionary size
  * topology mix          per-atom basis fractions over
                          {euclidean, periodic(circle), sphere, torus}
  * d_atom                intrinsic dim (basis-compatible: sphere/torus need >=2)
  * assignment            ibp_map / softmax / jumprelu
  * jumprelu_threshold    hard-gate sparsity (jumprelu only)
  * sparsity_weight       gate sparsity strength
  * smoothness_weight     decoder roughness
  * nuclear_norm_weight   decoder embedding-rank shrink
  * decoder_incoherence_weight  cross-atom incoherence
  * top_k                 final assignment support projection
  * n_iter                solver iterations

Optimizer: greedy coordinate-ascent + evolutionary mutation of the best-so-far
config. Each iteration proposes a batch of mutated candidates, evaluates them in
parallel across the GPU pool (one worker process per visible device), and keeps
any candidate that strictly improves held-out EV. Trajectory (iteration ->
best EV) is logged. Plateau detection stops after a configurable number of
non-improving iterations.

NO fitting math lives here; it only drives the production engine.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# data + metric (identical protocol to sae_ev_vs_k_olmo.py)
# ----------------------------------------------------------------------------
BASES = ["euclidean", "periodic", "sphere", "torus"]
ASSIGNMENTS = ["ibp_map", "softmax", "jumprelu"]
# minimum intrinsic dim each basis needs to be well posed
MIN_D = {"euclidean": 1, "periodic": 1, "sphere": 2, "torus": 2}


def _load_activations(npy, olmo_layer):
    arr = np.load(npy)
    if arr.ndim == 3:
        if olmo_layer is None:
            raise SystemExit("3D activations.npy needs --olmo-layer")
        arr = arr[:, olmo_layer, :]
    return np.asarray(arr, dtype=np.float64)


def _pca_project(train, test, pcs):
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    comp = vt[:pcs].T
    z_tr = tc @ comp
    z_te = (test - mean) @ comp
    scale = np.sqrt(np.mean(z_tr**2)) or 1.0
    return z_tr / scale, z_te / scale


def _ev(target, fitted):
    resid = target - fitted
    denom = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid**2)) / denom


# ----------------------------------------------------------------------------
# config representation
# ----------------------------------------------------------------------------
@dataclass
class Config:
    K: int = 2
    bases: tuple = ("periodic", "periodic")  # per-atom basis (len K)
    d_atom: int = 1
    assignment: str = "ibp_map"
    jumprelu_threshold: float = 0.0
    sparsity_weight: float = 1.0
    smoothness_weight: float = 1.0
    nuclear_norm_weight: float = 1.0
    decoder_incoherence_weight: float = 1.0
    top_k: int = 0  # 0 == disabled
    n_iter: int = 40

    def normalized(self):
        """Repair invariants: len(bases)==K, d_atom >= max basis requirement."""
        c = copy.deepcopy(self)
        b = list(c.bases)
        if len(b) < c.K:
            b = b + [b[-1] if b else "periodic"] * (c.K - len(b))
        b = b[: c.K]
        c.bases = tuple(b)
        need = max(MIN_D[x] for x in c.bases)
        if c.d_atom < need:
            c.d_atom = need
        if c.top_k is not None and c.top_k > c.K:
            c.top_k = c.K
        return c

    def key(self):
        c = self.normalized()
        return json.dumps(
            {
                "K": c.K, "bases": list(c.bases), "d_atom": c.d_atom,
                "assignment": c.assignment, "jt": round(c.jumprelu_threshold, 4),
                "sw": round(c.sparsity_weight, 4), "smw": round(c.smoothness_weight, 4),
                "nnw": round(c.nuclear_norm_weight, 4),
                "diw": round(c.decoder_incoherence_weight, 4),
                "top_k": c.top_k, "n_iter": c.n_iter,
            },
            sort_keys=True,
        )

    def to_kwargs(self, seed):
        c = self.normalized()
        kw = dict(
            K=c.K,
            d_atom=c.d_atom,
            atom_basis=list(c.bases),
            assignment=c.assignment,
            n_iter=c.n_iter,
            sparsity_weight=c.sparsity_weight,
            smoothness_weight=c.smoothness_weight,
            nuclear_norm_weight=c.nuclear_norm_weight,
            decoder_incoherence_weight=c.decoder_incoherence_weight,
            random_state=seed,
        )
        if c.assignment == "jumprelu":
            kw["jumprelu_threshold"] = c.jumprelu_threshold
        if c.top_k and c.top_k > 0:
            kw["top_k"] = c.top_k
        return kw


def mutate(cfg, rng):
    """Return one mutated copy of cfg (single-knob coordinate move, mostly)."""
    c = copy.deepcopy(cfg)
    knob = rng.choice([
        "K", "basis_one", "basis_all", "d_atom", "assignment", "jt",
        "sw", "smw", "nnw", "diw", "top_k", "n_iter",
    ])
    if knob == "K":
        step = rng.choice([-2, -1, 1, 2, 3])
        c.K = max(1, c.K + step)
    elif knob == "basis_one":
        i = rng.randrange(len(c.bases))
        b = list(c.bases)
        b[i] = rng.choice(BASES)
        c.bases = tuple(b)
    elif knob == "basis_all":
        b = rng.choice(BASES)
        c.bases = tuple([b] * c.K)
    elif knob == "d_atom":
        c.d_atom = max(1, c.d_atom + rng.choice([-1, 1, 1]))
    elif knob == "assignment":
        c.assignment = rng.choice(ASSIGNMENTS)
    elif knob == "jt":
        c.jumprelu_threshold = max(0.0, c.jumprelu_threshold + rng.choice([-0.2, -0.1, 0.1, 0.2, 0.4]))
    elif knob == "sw":
        c.sparsity_weight = max(0.0, c.sparsity_weight * rng.choice([0.3, 0.5, 2.0, 3.0]))
    elif knob == "smw":
        c.smoothness_weight = max(0.0, c.smoothness_weight * rng.choice([0.3, 0.5, 2.0, 3.0]))
    elif knob == "nnw":
        c.nuclear_norm_weight = max(0.0, c.nuclear_norm_weight * rng.choice([0.0, 0.3, 0.5, 2.0]))
    elif knob == "diw":
        c.decoder_incoherence_weight = max(0.0, c.decoder_incoherence_weight * rng.choice([0.0, 0.3, 0.5, 2.0]))
    elif knob == "top_k":
        c.top_k = max(0, min(c.K, c.top_k + rng.choice([-1, 0, 1, 2])))
    elif knob == "n_iter":
        c.n_iter = int(min(120, max(15, c.n_iter + rng.choice([-15, 15, 30]))))
    return c.normalized()


# ----------------------------------------------------------------------------
# worker: evaluate one candidate (runs in its own process, own GPU)
# ----------------------------------------------------------------------------
_W = {}  # per-process cached split


def _worker_init(npy, olmo_layer, pcs, test_frac, seed, gpu_ids, counter, lock):
    # pin this worker to a DISTINCT GPU via a shared atomic counter so the
    # N persistent workers map one-to-one onto the N visible devices.
    with lock:
        wid = counter.value % max(1, len(gpu_ids))
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[wid])
    x = _load_activations(npy, olmo_layer)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    z_tr, z_te = _pca_project(x[train_idx], x[test_idx], pcs)
    _W["z_tr"], _W["z_te"], _W["seed"] = z_tr, z_te, seed


def _eval_candidate(args):
    cfg, fit_seed = args
    z_tr, z_te = _W["z_tr"], _W["z_te"]
    if cfg.normalized().K >= z_tr.shape[0]:
        return cfg.key(), float("-inf"), "K>=N_train"
    from gamfit import sae_manifold_fit

    t0 = time.time()
    try:
        m = sae_manifold_fit(z_tr, **cfg.to_kwargs(fit_seed))
        fitted = m.reconstruct(z_te)
        ev = _ev(z_te, fitted)
        if not np.isfinite(ev):
            return cfg.key(), float("-inf"), "nonfinite_EV"
        return cfg.key(), float(ev), f"{time.time()-t0:.1f}s"
    except Exception as e:  # noqa: BLE001
        return cfg.key(), float("-inf"), f"ERR:{type(e).__name__}:{str(e)[:120]}"


# ----------------------------------------------------------------------------
# hillclimb driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--olmo-layer", type=int, default=25)
    ap.add_argument("--pcs", type=int, default=32)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-seed", type=int, default=42)
    ap.add_argument("--gpus", default="0,1,2,3", help="visible GPU ids for workers")
    ap.add_argument("--batch", type=int, default=4, help="candidates per iteration (== #GPUs ideal)")
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--plateau", type=int, default=12, help="stop after this many non-improving iters")
    ap.add_argument("--time-budget", type=float, default=25000.0, help="wall-clock seconds")
    ap.add_argument("--out", default="hillclimb_result.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gpu_ids = [int(s) for s in args.gpus.split(",") if s.strip() != ""]
    n_workers = max(1, len(gpu_ids))

    # baseline = the sae_ev_vs_k_olmo.py reference config (curved, K=2, ibp_map)
    baseline = Config(K=2, bases=("periodic", "periodic"), d_atom=1,
                      assignment="ibp_map", n_iter=40).normalized()

    evaluated = {}  # key -> (ev, note, cfg)
    trajectory = []  # list of (iter, best_ev)
    t_start = time.time()

    import multiprocessing as _mp
    _ctx = _mp.get_context("spawn")
    _counter = _ctx.Value("i", 0)
    _lock = _ctx.Lock()
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=_ctx,
        initializer=_worker_init,
        initargs=(args.npy, args.olmo_layer, args.pcs, args.test_frac, args.seed,
                  gpu_ids, _counter, _lock),
    )

    def evaluate(cfgs):
        out = []
        todo = []
        for c in cfgs:
            k = c.key()
            if k in evaluated:
                out.append((k, evaluated[k][0], c))
            else:
                todo.append(c)
        if todo:
            results = list(pool.map(_eval_candidate, [(c, args.fit_seed) for c in todo]))
            for c, (k, ev, note) in zip(todo, results):
                evaluated[k] = (ev, note, c)
                out.append((k, ev, c))
                print(f"    cand EV={ev: .6f}  [{note}]  {k}", flush=True)
        return out

    print("=== #1026 SAE hillclimb on OLMo activations ===", flush=True)
    print(f"npy={args.npy} layer={args.olmo_layer} pcs={args.pcs} "
          f"gpus={gpu_ids} batch={args.batch} max_iters={args.max_iters}", flush=True)

    # seed the climb with the baseline
    print("[iter 0] evaluating baseline", flush=True)
    evaluate([baseline])
    best_key = baseline.key()
    best_ev = evaluated[best_key][0]
    best_cfg = baseline
    baseline_ev = best_ev
    trajectory.append((0, best_ev))
    print(f"[iter 0] baseline EV = {best_ev:.6f}", flush=True)

    no_improve = 0
    it = 0
    while it < args.max_iters:
        it += 1
        if time.time() - t_start > args.time_budget:
            print(f"[stop] time budget {args.time_budget}s reached", flush=True)
            break
        # propose: mutate best-so-far into a batch
        cands = []
        seen_local = {best_cfg.key()}
        tries = 0
        while len(cands) < args.batch and tries < args.batch * 8:
            tries += 1
            c = mutate(best_cfg, rng)
            if c.key() not in seen_local and c.key() not in evaluated:
                cands.append(c)
                seen_local.add(c.key())
        if not cands:  # neighborhood exhausted near best -> random restart kick
            cands = [mutate(mutate(best_cfg, rng), rng) for _ in range(args.batch)]
        print(f"[iter {it}] proposing {len(cands)} candidates (best EV so far {best_ev:.6f})", flush=True)
        results = evaluate(cands)
        improved = False
        for k, ev, c in results:
            if ev > best_ev + 1e-9:
                best_ev, best_key, best_cfg = ev, k, c
                improved = True
        trajectory.append((it, best_ev))
        if improved:
            no_improve = 0
            print(f"[iter {it}] *** NEW BEST EV = {best_ev:.6f}  {best_key}", flush=True)
        else:
            no_improve += 1
            print(f"[iter {it}] no improvement ({no_improve}/{args.plateau})", flush=True)
        # checkpoint each iter
        _dump(args.out, baseline, baseline_ev, best_cfg, best_ev, trajectory, evaluated, it)
        if no_improve >= args.plateau:
            print(f"[stop] plateau: {no_improve} non-improving iters", flush=True)
            break

    pool.shutdown(wait=True)
    print("\n=== HILLCLIMB DONE ===", flush=True)
    print(f"baseline EV = {baseline_ev:.6f}", flush=True)
    print(f"best     EV = {best_ev:.6f}   (delta = {best_ev - baseline_ev:+.6f})", flush=True)
    print(f"best config = {best_key}", flush=True)
    print("climb curve (iter -> best EV):", flush=True)
    for i, e in trajectory:
        print(f"  {i:>3}  {e:.6f}", flush=True)
    _dump(args.out, baseline, baseline_ev, best_cfg, best_ev, trajectory, evaluated, it)
    print(f"result written to {args.out}", flush=True)


def _dump(path, baseline, baseline_ev, best_cfg, best_ev, trajectory, evaluated, it):
    payload = {
        "iter": it,
        "baseline_ev": baseline_ev,
        "baseline_config": json.loads(baseline.key()),
        "best_ev": best_ev,
        "best_config": json.loads(best_cfg.key()),
        "delta_ev": best_ev - baseline_ev,
        "trajectory": trajectory,
        "n_evaluated": len(evaluated),
        "leaderboard": sorted(
            [{"ev": v[0], "note": v[1], "config": json.loads(k)}
             for k, v in evaluated.items() if np.isfinite(v[0])],
            key=lambda r: -r["ev"],
        )[:15],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
