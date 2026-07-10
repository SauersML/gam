"""Real gamfit ManifoldSAE on the REAL 32B MoE (Qwen3-30B-A3B) L17 residual harvest.

Mirrors the frozen msae_l17 two-tier product (CONFIG_FROZEN.md):
  Tier-0 space : x' = (x - per_dim_mean)/global_rms_scale  (tier0_recentered.json;
                 the recentered mean fixes the L2~1.07 offset -- see repo memory).
  T1 (linear)  : gamfit.sparse_dictionary_fit -- a GENUINELY OVERCOMPLETE dictionary,
                 K >> p=2048 (8192 / 16384), active=32. This is the overcomplete tier.
  T2 (curved)  : gamfit.sae_manifold_fit_stagewise on the T1 residual -- the real
                 typed-manifold engine (d_atom=1 circles, evidence-gated births).
  typed-select : a small-K gamfit.sae_manifold_fit with structure search on to show
                 WHICH typed manifolds (Circle/Torus/Sphere/Euclidean) get selected.

All EV uses the frozen metric EV = 1 - ||X-recon||^2/||X||^2 on the tier-0 space
with ev_baseline="zero" (origin == train mean on tier-0 rows).
"""
import argparse, json, os, time, numpy as np
from collections import Counter


def load_tier0(path):
    d = json.load(open(path))
    mean = np.asarray(d["per_dim_mean"], dtype=np.float64)
    scale = float(d["global_rms_scale"])
    return mean, scale, d


def ev_of(X, recon, tot):
    return 1.0 - float(((X - recon) ** 2).sum()) / tot


def linear_pca_ev(X, Ms, tot):
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    out = {}
    for M in Ms:
        recon = (X @ Vt[:M].T) @ Vt[:M]
        out[f"linear_pca_M{M}"] = ev_of(X, recon, tot)
    return out


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tier0", required=True)
    ap.add_argument("--rows", type=int, default=150000)
    ap.add_argument("--sae-K", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--sparse-epochs", type=int, default=30)
    ap.add_argument("--max-births", type=int, default=24)
    ap.add_argument("--stagewise-rows", type=int, default=100000)
    ap.add_argument("--stagewise-niter", type=int, default=64)
    ap.add_argument("--typed-K", type=int, default=16)
    ap.add_argument("--typed-rows", type=int, default=20000)
    ap.add_argument("--typed-niter", type=int, default=12)
    ap.add_argument("--skip-typed", action="store_true")
    ap.add_argument("--skip-stagewise", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gamfit
    ver = getattr(gamfit, "__version__", "?")
    os.makedirs(args.out, exist_ok=True)

    mean, scale, tier0meta = load_tier0(args.tier0)
    Xmm = np.load(args.train, mmap_mode="r")
    n_rows, D = Xmm.shape
    N = min(args.rows, n_rows)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n_rows, N, replace=False))
    X = np.asarray(Xmm[idx], dtype=np.float64)
    X = (X - mean) / scale  # tier-0 recentered space (ev_baseline zero == origin)
    tot = float((X ** 2).sum())
    print(f"loaded L17 {Xmm.shape} -> subsample {X.shape} gamfit={ver} tier0-recentered", flush=True)

    results = {
        "model": "Qwen3-30B-A3B (q36b, MoE) L17 residual",
        "gamfit_version": ver,
        "train_path": args.train,
        "tier0": os.path.basename(args.tier0),
        "tier0_transform": tier0meta.get("transform"),
        "N_subsample": int(N),
        "D": int(D),
        "ev_baseline": "zero (tier-0 origin == train mean)",
        "references": {},
        "overcomplete_sparse_dict": [],
        "stagewise_manifold": None,
        "typed_manifold_fit": None,
    }

    # ---- linear PCA references (matched capacity) ----
    refs = linear_pca_ev(X, [1, 2, 4, 8, 16, 32, 64], tot)
    results["references"] = refs
    print("LINEAR PCA:", {k: round(v, 4) for k, v in refs.items()}, flush=True)

    # ---- OVERCOMPLETE linear sparse dictionary (K >> p) ----
    for K in args.sae_K:
        print(f"=== sparse_dictionary_fit K={K} (overcomplete, p={D}) active={args.active} ===", flush=True)
        t0 = time.time()
        try:
            sd = gamfit.sparse_dictionary_fit(X, K, active=args.active,
                                              max_epochs=args.sparse_epochs)
            wall = time.time() - t0
            rec = {
                "K": K, "active": args.active, "overcomplete_ratio": K / D,
                "explained_variance": float(sd.explained_variance),
                "epochs": int(sd.epochs), "converged": bool(sd.converged),
                "wall_s": wall,
            }
            print(f"  K={K} EV={rec['explained_variance']:.4f} epochs={rec['epochs']} "
                  f"conv={rec['converged']} wall={wall:.1f}s", flush=True)
            results["overcomplete_sparse_dict"].append(rec)
            # keep the K=16384 fitted for the T2 residual (largest K)
            if K == max(args.sae_K):
                t1_recon = np.asarray(sd.fitted, dtype=np.float64)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["overcomplete_sparse_dict"].append({"K": K, "error": str(e),
                                                        "wall_s": time.time() - t0})

    # ---- T2 curved: stagewise manifold on the T1 residual ----
    if not args.skip_stagewise and results["overcomplete_sparse_dict"] and "t1_recon" in dir():
        resid = X - t1_recon
        srng = np.random.default_rng(0)
        M = min(args.stagewise_rows, resid.shape[0])
        sidx = np.sort(srng.choice(resid.shape[0], M, replace=False))
        Xs = resid[sidx]
        tots = float((Xs ** 2).sum())
        print(f"=== sae_manifold_fit_stagewise on T1 residual rows={M} max_births={args.max_births} ===",
              flush=True)
        t0 = time.time()
        try:
            sw = gamfit.sae_manifold_fit_stagewise(
                Xs, d_atom=1, atom_topology="circle", assignment="ibp_map",
                structured_whitening=True, max_births=args.max_births,
                max_backfit_sweeps=4, n_iter=args.stagewise_niter, random_state=0)
            wall = time.time() - t0
            recon = np.asarray(sw.reconstruct(Xs), dtype=np.float64)
            srec = {
                "engine": "sae_manifold_fit_stagewise",
                "on": "T1 residual (K=%d overcomplete dict)" % max(args.sae_K),
                "rows": M,
                "resid_ev_of_curved_tier": ev_of(Xs, recon, tots),
                "wall_s": wall,
            }
            for attr in ["k", "reconstruction_ev", "births_accepted", "births_rejected",
                         "stopped_reason", "ev_trace",
                         "rank_charge_evidence_used"]:
                try: srec[attr] = _jsonable(getattr(sw, attr))
                except Exception as e: srec[attr] = f"<err {e}>"
            # per-atom curvature Theta (which atoms are genuinely curved)
            try:
                atoms = list(sw.atoms)
                srec["n_atoms"] = len(atoms)
                thetas = []
                for a in atoms:
                    th = getattr(a, "theta", getattr(a, "curvature", getattr(a, "angle_span", None)))
                    thetas.append(_jsonable(th))
                srec["atom_theta"] = thetas
                srec["n_curved_theta_gt1"] = sum(1 for t in thetas if isinstance(t, (int, float)) and t > 1.0)
            except Exception as e:
                srec["atoms_err"] = str(e)
            print(f"  stagewise: k={srec.get('k')} births_acc={srec.get('births_accepted')} "
                  f"resid_tier_EV={srec['resid_ev_of_curved_tier']:.4f} wall={wall:.1f}s "
                  f"stop={srec.get('stopped_reason')}", flush=True)
            results["stagewise_manifold"] = srec
        except Exception as e:
            import traceback; traceback.print_exc()
            results["stagewise_manifold"] = {"error": str(e), "wall_s": time.time() - t0}

    # ---- typed manifold selection: small-K sae_manifold_fit, structure search ON ----
    if not args.skip_typed:
        trng = np.random.default_rng(1)
        M = min(args.typed_rows, X.shape[0])
        tidx = np.sort(trng.choice(X.shape[0], M, replace=False))
        Xt = X[tidx]
        tott = float((Xt ** 2).sum())
        print(f"=== sae_manifold_fit K={args.typed_K} (typed-select, struct search) rows={M} ===",
              flush=True)
        t0 = time.time()
        try:
            model = gamfit.sae_manifold_fit(Xt, K=args.typed_K, d_atom=1,
                                            n_iter=args.typed_niter, random_state=0)
            wall = time.time() - t0
            recon = np.asarray(model.reconstruct(Xt), dtype=np.float64)
            trec = {
                "engine": "sae_manifold_fit (structure search)",
                "K_requested": args.typed_K, "rows": M,
                "ev_reconstruct": ev_of(Xt, recon, tott),
                "wall_s": wall,
            }
            try: trec["chosen_k"] = _jsonable(model.chosen_k)
            except Exception as e: trec["chosen_k"] = f"<err {e}>"
            try:
                tops = [str(t) for t in list(model.atom_topologies)]
                trec["atom_topologies"] = tops
                trec["atom_topology_counts"] = dict(Counter(tops))
                trec["n_atoms"] = len(tops)
            except Exception as e:
                trec["atom_topologies_err"] = str(e)
            try: trec["reml_score"] = _jsonable(model.reml_score)
            except Exception as e: trec["reml_score"] = f"<err {e}>"
            try:
                td = gamfit.sae_trust_diagnostics(model.to_dict())
                trec["trust_diagnostics"] = _jsonable(td)
            except Exception as e:
                trec["trust_diagnostics"] = f"<err {e}>"
            print(f"  typed: chosen_k={trec.get('chosen_k')} EV={trec['ev_reconstruct']:.4f} "
                  f"topo={trec.get('atom_topology_counts')} wall={wall:.1f}s", flush=True)
            results["typed_manifold_fit"] = trec
        except Exception as e:
            import traceback; traceback.print_exc()
            results["typed_manifold_fit"] = {"error": str(e), "wall_s": time.time() - t0}

    with open(os.path.join(args.out, "numbers.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    print("WROTE", os.path.join(args.out, "numbers.json"), flush=True)


if __name__ == "__main__":
    main()
