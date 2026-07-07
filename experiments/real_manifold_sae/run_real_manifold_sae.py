"""Real gamfit ManifoldSAE fit on REAL Qwen3-8B L18 activations (pos0 sink-peeled).

This is the actual product path -- gamfit.sae_manifold_fit / sae_manifold_fit_stagewise
on the SAME pos0-null-gated peeled input as curved_vs_linear.py Run-2 -- NOT the
numpy kmeans+PCA proxy atlas. We compare the real fitter's reconstruction EV to the
Run-2 references at matched capacity (linear PCA d=1, proxy atlas K32 d=1).

Peel functions are replicated verbatim from
experiments/curved_vs_linear/curved_vs_linear.py so the input is byte-identical.

Usage (on MSI compute node):
  GAM_MSI_DATA=$R saevenv/bin/python run_real_manifold_sae.py \
      --positions $R/qwen_positions.npy --rows 30000 --K 32 16 --out <dir>
"""
import argparse, json, os, time, numpy as np

PMAX = 512
NULL_PERMUTATION_REPLICATES = 200


# --- peel (verbatim from curved_vs_linear.py) ------------------------------
def early_indicators(positions, k):
    return np.stack([(positions == q).astype(np.float64) for q in range(k)], axis=1)


def peel_design_pos0(positions):
    n = positions.shape[0]
    intercept = np.ones((n, 1))
    z = np.concatenate([intercept, early_indicators(positions, 1)], axis=1)
    return z, "pos0 (first-token indicator)"


def regress_out(Xc, Z):
    B = np.linalg.pinv(Z.T @ Z) @ (Z.T @ Xc)
    return Xc - Z @ B, B


def absorbed_r2(Xc, Z):
    resid, _ = regress_out(Xc, Z)
    ss_res = float((resid ** 2).sum())
    ss_tot = float((Xc ** 2).sum())
    return 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot


def null_permutation_absorbed(Xc, Z, replicates, seed):
    rng = np.random.default_rng(seed)
    n = Xc.shape[0]
    return np.array([absorbed_r2(Xc, Z[rng.permutation(n)]) for _ in range(replicates)])


def sink_peel_pos0(Xc, positions, null_replicates=NULL_PERMUTATION_REPLICATES, seed=0):
    Z, label = peel_design_pos0(positions)
    observed = absorbed_r2(Xc, Z)
    null_dist = null_permutation_absorbed(Xc, Z, null_replicates, seed)
    null_max = float(null_dist.max())
    peel_is_causal = observed > null_max
    if peel_is_causal:
        Xp, _ = regress_out(Xc, Z)
        n_peeled = int(Z.shape[1] - 1)
    else:
        Xp = Xc.copy()
        n_peeled = 0
    audit = dict(
        peel_design=label,
        n_directions_peeled=n_peeled,
        peel_is_causal=bool(peel_is_causal),
        absorbed_observed=float(observed),
        absorbed_permuted_null_max=null_max,
        absorbed_permuted_null_mean=float(null_dist.mean()),
        null_replicates=int(null_replicates),
        frac_rows_pos0=float((positions == 0).mean()),
    )
    return Xp, audit


# --- reference recomputation (same ev metric as curved_vs_linear.py) --------
def ev_of(Xp, recon, tot):
    return 1.0 - float(((Xp - recon) ** 2).sum()) / tot


def kmeans_np(Xin, K, iters=25, seed=0):
    rg = np.random.default_rng(seed)
    C = Xin[rg.choice(len(Xin), K, replace=False)].copy()
    lab = np.zeros(len(Xin), dtype=int)
    xn = (Xin ** 2).sum(1)
    for _ in range(iters):
        d2 = xn[:, None] - 2 * Xin @ C.T + (C ** 2).sum(1)[None, :]
        newlab = d2.argmin(1)
        if (newlab == lab).all() and _ > 0:
            break
        lab = newlab
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = Xin[m].mean(0)
    return lab


def linear_pca_ev(Xp, M, tot):
    _, _, Vtp = np.linalg.svd(Xp, full_matrices=False)
    recon = (Xp @ Vtp[:M].T) @ Vtp[:M]
    return ev_of(Xp, recon, tot)


def proxy_atlas_ev(Xp, K, d, tot, seed=0):
    lab = kmeans_np(Xp, K, seed=seed)
    recon = np.zeros_like(Xp)
    for k in range(K):
        m = lab == k
        if m.sum() < d + 1:
            recon[m] = Xp[m].mean(0)
            continue
        Ck = Xp[m]
        ck = Ck.mean(0)
        Cc = Ck - ck
        _, _, vv = np.linalg.svd(Cc, full_matrices=False)
        recon[m] = ck + (Cc @ vv[:d].T) @ vv[:d]
    return ev_of(Xp, recon, tot)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def extract(model, Xp, tot, label):
    """Pull EV + atom/chart-type breakdown + diagnostics from a fitted ManifoldSAE."""
    out = {"fit": label}
    recon = np.asarray(model.reconstruct(Xp), dtype=np.float64)
    out["ev_reconstruct"] = ev_of(Xp, recon, tot)
    for attr in ["chosen_k", "reml_score", "reconstruction_r2"]:
        try:
            out[attr] = _jsonable(getattr(model, attr))
        except Exception as e:
            out[attr] = f"<err {e}>"
    try:
        tops = list(getattr(model, "atom_topologies"))
        out["atom_topologies"] = [str(t) for t in tops]
        # count by type
        from collections import Counter
        out["atom_topology_counts"] = dict(Counter(str(t) for t in tops))
        out["n_atoms"] = len(tops)
    except Exception as e:
        out["atom_topologies"] = f"<err {e}>"
    try:
        out["summary"] = _jsonable(model.summary())
    except Exception as e:
        out["summary"] = f"<err {e}>"
    try:
        coords = np.asarray(model.coords)
        out["coords_shape"] = list(coords.shape)
    except Exception as e:
        out["coords_shape"] = f"<err {e}>"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--K", type=int, nargs="+", default=[32])
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--stagewise", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate API")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gamfit
    ver = getattr(gamfit, "__version__", "?")
    R = os.environ["GAM_MSI_DATA"]
    os.makedirs(args.out, exist_ok=True)

    path = f"{R}/harvest_out/qwen3_8b_wikitext/resid_L{args.layer}.npy"
    Xmm = np.load(path, mmap_mode="r")
    n_rows = Xmm.shape[0]
    N = min(args.rows, n_rows)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n_rows, N, replace=False))
    X = np.asarray(Xmm[idx], dtype=np.float64)
    print(f"loaded L{args.layer} {X.shape} gamfit={ver}", flush=True)
    Xc = X - X.mean(0)

    positions = np.load(args.positions).astype(np.int32)
    if positions.shape[0] < n_rows:
        raise SystemExit(f"positions ({positions.shape[0]}) shorter than harvest rows ({n_rows})")
    positions = positions[:n_rows][idx]
    Xp, audit = sink_peel_pos0(Xc, positions)
    print(f"peel: peeled {audit['n_directions_peeled']} dir(s) causal={audit['peel_is_causal']} "
          f"observed={audit['absorbed_observed']:.4f} null_max={audit['absorbed_permuted_null_max']:.5f}",
          flush=True)

    tot = float((Xp ** 2).sum())

    # references on the identical peeled input
    refs = {}
    for M in [1, 2, 4]:
        refs[f"linear_pca_M{M}"] = linear_pca_ev(Xp, M, tot)
    for (K, d) in [(16, 1), (32, 1)]:
        refs[f"proxy_atlas_K{K}d{d}"] = proxy_atlas_ev(Xp, K, d, tot)
    print("REFS:", {k: round(v, 4) for k, v in refs.items()}, flush=True)

    results = {
        "gamfit_version": ver,
        "layer": args.layer,
        "N": int(N),
        "D": int(Xp.shape[1]),
        "peel": "pos0",
        "peel_audit": audit,
        "references": refs,
        "real_fits": [],
    }

    if args.smoke:
        Ks = args.K[:1]
        n_iter = min(args.n_iter, 8)
    else:
        Ks = args.K
        n_iter = args.n_iter

    for K in Ks:
        print(f"=== REAL sae_manifold_fit K={K} d_atom={args.d_atom} n_iter={n_iter} ===", flush=True)
        t0 = time.time()
        try:
            model = gamfit.sae_manifold_fit(Xp, K=K, d_atom=args.d_atom, n_iter=n_iter, random_state=0)
            wall = time.time() - t0
            rec = extract(model, Xp, tot, f"sae_manifold_fit K={K} d_atom={args.d_atom}")
            rec["K_requested"] = K
            rec["d_atom"] = args.d_atom
            rec["n_iter"] = n_iter
            rec["wall_s"] = wall
            print(f"  EV={rec['ev_reconstruct']:.4f} chosen_k={rec.get('chosen_k')} "
                  f"topo={rec.get('atom_topology_counts')} wall={wall:.1f}s", flush=True)
            # trust diagnostics
            try:
                td = gamfit.sae_trust_diagnostics(model.to_dict())
                rec["trust_diagnostics"] = _jsonable(td)
            except Exception as e:
                rec["trust_diagnostics"] = f"<err {e}>"
            results["real_fits"].append(rec)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["real_fits"].append({"fit": f"sae_manifold_fit K={K}", "error": str(e),
                                         "wall_s": time.time() - t0})

    if args.stagewise:
        print("=== REAL sae_manifold_fit_stagewise ===", flush=True)
        t0 = time.time()
        try:
            sw = gamfit.sae_manifold_fit_stagewise(Xp, d_atom=args.d_atom, n_iter=n_iter, random_state=0)
            wall = time.time() - t0
            recon = np.asarray(sw.reconstruct(Xp), dtype=np.float64)
            srec = {
                "fit": "sae_manifold_fit_stagewise",
                "ev_reconstruct": ev_of(Xp, recon, tot),
                "wall_s": wall,
                "d_atom": args.d_atom,
            }
            for attr in ["k", "reconstruction_ev", "births_accepted", "births_rejected",
                         "stopped_reason", "ev_trace", "backfit_ev_trace",
                         "cone_atom_recovery_used", "rank_charge_evidence_used"]:
                try:
                    srec[attr] = _jsonable(getattr(sw, attr))
                except Exception as e:
                    srec[attr] = f"<err {e}>"
            try:
                atoms = list(sw.atoms)
                from collections import Counter
                tops = [str(getattr(a, "topology", getattr(a, "atom_topology", type(a).__name__))) for a in atoms]
                srec["atom_topology_counts"] = dict(Counter(tops))
                srec["n_atoms"] = len(atoms)
            except Exception as e:
                srec["atom_topologies"] = f"<err {e}>"
            print(f"  stagewise EV={srec['ev_reconstruct']:.4f} k={srec.get('k')} "
                  f"topo={srec.get('atom_topology_counts')} wall={wall:.1f}s", flush=True)
            results["real_fits"].append(srec)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["real_fits"].append({"fit": "sae_manifold_fit_stagewise", "error": str(e),
                                         "wall_s": time.time() - t0})

    with open(os.path.join(args.out, "numbers.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    print("WROTE", os.path.join(args.out, "numbers.json"), flush=True)


if __name__ == "__main__":
    main()
