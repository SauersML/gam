"""Real ManifoldSAE curved-vs-flat headline on Qwen3-30B-A3B (32B MoE) L17.

Faithful to the frozen msae_l17 product (CONFIG_FROZEN.md): reuses the SHIPPING
driver helpers from driver/compose_l17_stagewise.py (imported, not reimplemented):
  T1  : frozen OVERCOMPLETE linear sparse dictionary decoder (decoder_K32000.npy,
        K=32000 >> p=2048, active=32) -> CPU top-active transform (_BareDecoderT1 /
        _t1_recon). This is the overcomplete tier.
  T2  : gamfit.sae_manifold_fit_stagewise on the T1 RESIDUAL, run TWICE on the
        IDENTICAL residual, matched atom budget + sparsity, differing ONLY in
        atom_topology -> "circle" (CURVED) vs "linear" (FLAT). This isolates
        curvature: same capacity, only the atom manifold type changes.

HEADLINE: dEV = EV(T1 + T2-curved) - EV(T1 + T2-flat), composed on the tier-0
recentered space with ev_baseline="zero" (the frozen scoring convention).

Also reports standalone tiers (EV T1 alone / +curved / +flat), atom counts, per-atom
turning Theta, stopped_reason, collapse events, and a small-K structure-search
sae_manifold_fit chart-type selection breakdown (which typed manifolds get chosen).
"""
import argparse, json, os, sys, time, numpy as np
from pathlib import Path
from collections import Counter

import gamfit

C = None


def load_product_driver(root):
    """Load the frozen product driver from the caller-supplied data root."""
    global C
    driver_dir = root / "msae_l17" / "driver"
    sys.path.insert(0, str(driver_dir))
    import compose_l17_stagewise as product_driver
    C = product_driver


def t1_ridge_recon(X, dec, active, ridge=1e-6, chunk=4096):
    """Frozen T1 recipe: per row pick top-|score| active atoms (score = X @ decoderᵀ),
    then ridge least-squares codes over that active support -> dense reconstruction.
    Chunked to bound memory (gathering (chunk, active, p) blocks). CPU only."""
    N, p = X.shape
    K = dec.shape[0]
    s = min(active, K)
    recon = np.empty_like(X)
    eye = ridge * np.eye(s)
    for a in range(0, N, chunk):
        Xc = X[a:a + chunk]                      # (b, p)
        scores = Xc @ dec.T                      # (b, K)
        top = np.argpartition(-np.abs(scores), s - 1, axis=1)[:, :s]  # (b, s)
        Da = dec[top]                            # (b, s, p)
        # G = Da Daᵀ (b,s,s); rhs = Da·x (b,s); codes = G⁻¹ rhs
        G = np.einsum("bsp,btp->bst", Da, Da) + eye
        rhs = np.einsum("bsp,bp->bs", Da, Xc)
        codes = np.linalg.solve(G, rhs[:, :, None])[:, :, 0]  # (b, s); explicit colvec
        recon[a:a + chunk] = np.einsum("bs,bsp->bp", codes, Da)
    return recon


def _jsonable(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    return o


def _atom_thetas_and_bases(sw):
    """Per-atom turning Theta (via the driver's own _turning_from_curve) + selected basis."""
    thetas, bases = [], []
    try:
        for a in list(sw.atoms):
            coords = C._get(a, "on_atom_coords_t")
            dec = C._get(a, "decoder_B")
            basis = C._get(a, "basis_kind") or C._get(a, "basis")
            bases.append(str(basis))
            try:
                thetas.append(_jsonable(C._turning_from_curve(coords, dec, basis)))
            except Exception:
                thetas.append(None)
    except Exception as e:
        return f"<err {e}>", f"<err {e}>"
    return thetas, bases


def run_t2(residual, topo, args):
    t0 = time.time()
    def _cb(bi, partial):
        ev = partial.get("ev_trace")
        print(f"  [{topo}] birth {bi} ev_trace_tail={ev[-1] if isinstance(ev, list) and ev else '?'}",
              flush=True)
    sw = C._fit_stagewise_t2(
        residual, d_atom=args.d_atom, atom_topology=topo, assignment="ordered_beta_bernoulli",
        max_births=args.max_births, min_effect_ev=0.0, structured_whitening=True,
        sample_weights=None, max_iter=args.n_iter, random_state=0, birth_callback=_cb)
    wall = time.time() - t0
    recon = np.asarray(sw.reconstruct(np.ascontiguousarray(residual, dtype=np.float32)),
                       dtype=np.float64)
    rec = {"atom_topology": topo, "wall_s": wall}
    for attr in ["k", "reconstruction_ev", "births_accepted", "births_rejected",
                 "stopped_reason", "ev_trace", "collapse_events"]:
        try: rec[attr] = _jsonable(getattr(sw, attr))
        except Exception as e: rec[attr] = f"<err {e}>"
    thetas, bases = _atom_thetas_and_bases(sw)
    rec["atom_theta"] = thetas
    rec["atom_basis_kinds"] = bases
    if isinstance(bases, list):
        rec["basis_kind_counts"] = dict(Counter(bases))
    try: rec["n_atoms"] = len(list(sw.atoms))
    except Exception: pass
    if isinstance(thetas, list):
        rec["n_curved_theta_gt1"] = sum(1 for t in thetas
                                        if isinstance(t, (int, float)) and abs(t) > 1.0)
    return sw, recon, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root directory containing the frozen msae_l17 product files")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--t1-decoder")
    ap.add_argument("--t1-active", type=int, default=32)
    ap.add_argument("--tier0")
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--max-births", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=64)
    ap.add_argument("--typed-K", type=int, default=12)
    ap.add_argument("--typed-rows", type=int, default=15000)
    ap.add_argument("--typed-niter", type=int, default=10)
    ap.add_argument("--skip-typed", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    load_product_driver(root)
    if args.t1_decoder is None:
        args.t1_decoder = str(root / "msae_l17" / "t1_out" / "decoder_K32000.npy")
    if args.tier0 is None:
        args.tier0 = str(root / "msae_l17" / "tier0_recentered.json")
    os.makedirs(args.out, exist_ok=True)
    ver = getattr(gamfit, "__version__", "?")

    # ---- tier-0 recentered space ----
    t0meta = json.load(open(args.tier0))
    mean = np.asarray(t0meta["per_dim_mean"], dtype=np.float64)
    scale = float(t0meta["global_rms_scale"])
    train = str(root / "msae_l17" / "L17_train.f32.npy")
    Xmm = np.load(train, mmap_mode="r")
    n_rows, D = Xmm.shape
    N = min(args.rows, n_rows)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n_rows, N, replace=False))
    X = ((np.asarray(Xmm[idx], dtype=np.float64) - mean) / scale)
    print(f"loaded L17 {Xmm.shape} -> {X.shape} gamfit={ver} tier0=recentered", flush=True)

    # ---- T1: frozen overcomplete decoder, CPU top-active + RIDGE transform ----
    # CONFIG_FROZEN recipe: T1 decoder rows carry scale and reconstruction is
    # top-active support + ridge least-squares codes (NOT unit-norm dot-product
    # projection). decoder_K32000.npy is a unit-norm overcomplete dictionary, so
    # dot-product codes overshoot catastrophically on correlated atoms (EV ~ -64);
    # ridge-LS over the active support is the correct, product-faithful recipe.
    dec = np.load(args.t1_decoder).astype(np.float64)
    K1 = dec.shape[0]
    t1_recon = t1_ridge_recon(X, dec, args.t1_active, ridge=1e-6, chunk=4096)
    residual = np.ascontiguousarray(X - t1_recon, dtype=np.float32)
    ev_t1 = C.explained_variance(X, t1_recon, baseline="zero")
    print(f"T1 K={K1} (overcomplete {K1/D:.1f}x) active={args.t1_active}: "
          f"EV(T1 alone)={ev_t1:.4f}", flush=True)

    results = {
        "model": "Qwen3-30B-A3B (q36b, MoE) L17 residual",
        "gamfit_version": ver, "N_subsample": int(N), "D": int(D),
        "tier0": os.path.basename(args.tier0),
        "tier0_transform": t0meta.get("transform"),
        "ev_baseline": "zero (tier-0 origin == train mean)",
        "T1": {"decoder": os.path.basename(args.t1_decoder), "K": int(K1),
               "overcomplete_ratio": K1 / D, "active": args.t1_active,
               "ev_t1_alone": ev_t1},
        "T2_arms": {}, "headline": {}, "typed_manifold_fit": None,
    }

    # ---- T2 curved vs flat on the IDENTICAL residual ----
    arms = {}
    for topo in ["circle", "linear"]:
        print(f"=== T2 stagewise atom_topology={topo} rows={N} max_births={args.max_births} ===",
              flush=True)
        try:
            sw, t2recon, rec = run_t2(residual, topo, args)
            composed = t1_recon + t2recon
            rec["ev_composed"] = C.explained_variance(X, composed, baseline="zero")
            rec["ev_residual_tier"] = C.explained_variance(
                residual.astype(np.float64), t2recon, baseline="zero")
            rec["delta_ev_over_t1"] = rec["ev_composed"] - ev_t1
            print(f"  [{topo}] k={rec.get('k')} births_acc={rec.get('births_accepted')} "
                  f"EV_composed={rec['ev_composed']:.4f} (+{rec['delta_ev_over_t1']:.4f} over T1) "
                  f"stop={rec.get('stopped_reason')} wall={rec['wall_s']:.1f}s", flush=True)
            arms[topo] = rec
        except Exception as e:
            import traceback; traceback.print_exc()
            arms[topo] = {"atom_topology": topo, "error": str(e)}
    results["T2_arms"] = arms

    if "ev_composed" in arms.get("circle", {}) and "ev_composed" in arms.get("linear", {}):
        dev = arms["circle"]["ev_composed"] - arms["linear"]["ev_composed"]
        dev_res = arms["circle"]["ev_residual_tier"] - arms["linear"]["ev_residual_tier"]
        results["headline"] = {
            "delta_ev_curved_minus_flat_composed": dev,
            "delta_ev_curved_minus_flat_residual_tier": dev_res,
            "ev_composed_curved": arms["circle"]["ev_composed"],
            "ev_composed_flat": arms["linear"]["ev_composed"],
            "atoms_curved": arms["circle"].get("n_atoms"),
            "atoms_flat": arms["linear"].get("n_atoms"),
        }
        print(f"HEADLINE dEV(curved-flat) composed={dev:+.5f} residual_tier={dev_res:+.5f}", flush=True)

    # ---- typed manifold chart selection (small-K structure search) ----
    if not args.skip_typed:
        trng = np.random.default_rng(1)
        M = min(args.typed_rows, X.shape[0])
        tidx = np.sort(trng.choice(X.shape[0], M, replace=False))
        Xt = np.ascontiguousarray(X[tidx], dtype=np.float64)
        tott = float((Xt ** 2).sum())
        print(f"=== typed sae_manifold_fit K={args.typed_K} (structure search) rows={M} ===", flush=True)
        t0 = time.time()
        try:
            model = gamfit.sae_manifold_fit(Xt, K=args.typed_K, d_atom=1,
                                            n_iter=args.typed_niter, random_state=0)
            wall = time.time() - t0
            recon = np.asarray(model.reconstruct(Xt), dtype=np.float64)
            trec = {"K_requested": args.typed_K, "rows": M, "wall_s": wall,
                    "ev_reconstruct": 1.0 - float(((Xt - recon) ** 2).sum()) / tott}
            try: trec["chosen_k"] = _jsonable(model.chosen_k)
            except Exception as e: trec["chosen_k"] = f"<err {e}>"
            try:
                tops = [str(t) for t in list(model.atom_topologies)]
                trec["atom_topologies"] = tops
                trec["atom_topology_counts"] = dict(Counter(tops))
            except Exception as e: trec["atom_topologies_err"] = str(e)
            try: trec["trust_diagnostics"] = _jsonable(gamfit.sae_trust_diagnostics(model.to_dict()))
            except Exception as e: trec["trust_diagnostics"] = f"<err {e}>"
            print(f"  typed chosen_k={trec.get('chosen_k')} EV={trec['ev_reconstruct']:.4f} "
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
