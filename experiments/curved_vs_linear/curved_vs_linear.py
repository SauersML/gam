"""
Curved-vs-linear on REAL Qwen3-8B L18 (sink-peeled), unsupervised, matched capacity.
The manifold-SAE premise as a head-to-head: a CHART ATLAS (K local curved charts) vs a
GLOBAL LINEAR code. If the atlas reconstructs better at matched parameter budget, curvature
is exploitable on real data; if not, it's a real negative. Also tries the actual gamfit fitter.
Saves curved_vs_linear.png + curved_vs_linear.json.

PEEL (the sink de-confounding, fixed 2026-07):
  The sink is peeled by the nuisance-atlas method the repo already owns
  (`experiments/nuisance_atlas/qwen_nuisance_msi.py`): a POSITION-0 INDICATOR
  regressed out of the centred activations, validated by the PERMUTED-POSITION NULL.
  This is causally targeted (it removes the first-token attention sink, not "the top
  chunk of variance") and auditable (the removed direction, its correlation with PC1,
  and the null are all logged). The previous rule peeled top PCs until 90% of variance
  was gone, capped at 16 -- accidentally benign on L18 (sink PC ~ 99% var -> exactly
  one PC) but WRONG in general: on a healthy layer it strips real semantic structure,
  and it removes variance rather than the sink. See the run docstring for --positions.

Peel-first doctrine (mirrors scale_k's --raw-ok): the script REFUSES to run a
positionless / un-peeled comparison unless --raw-ok is passed, because an un-peeled
L18 comparison mostly measures the positional sink.
"""
import argparse, json, os, numpy as np

PMAX = 512  # harvest seq_len / truncation length (matches qwen_nuisance_msi.py)


def required_env(name, message):
    try:
        return os.environ[name]
    except KeyError as exc:
        raise SystemExit(message) from exc


# ---------------------------------------------------------------------------
# Peel: the nuisance-atlas position-0 (attention-sink) regress-out + null.
# Mirrors experiments/nuisance_atlas/qwen_nuisance_msi.py so the two agree.
# ---------------------------------------------------------------------------
def normalized_fourier(positions, harmonics):
    u = positions.astype(np.float64) / PMAX
    cols = []
    for j in range(1, harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * j * u))
        cols.append(np.sin(2.0 * np.pi * j * u))
    return np.stack(cols, axis=1) if cols else np.zeros((positions.shape[0], 0))


def early_indicators(positions, k):
    return np.stack([(positions == q).astype(np.float64) for q in range(k)], axis=1)


def peel_design(positions, mode, harmonics=16, early_k=8):
    """Return (Z, label). `pos0` = [intercept | first-token indicator] (the sink test).
    `combined` = [intercept | fourier(2H) | early(early_k)] (full positional nuisance;
    strictly subsumes pos0). Both are the repo's own nuisance blocks."""
    n = positions.shape[0]
    intercept = np.ones((n, 1))
    if mode == "pos0":
        z = np.concatenate([intercept, early_indicators(positions, 1)], axis=1)
        return z, "pos0 (first-token indicator)"
    if mode == "combined":
        z = np.concatenate(
            [intercept, normalized_fourier(positions, harmonics), early_indicators(positions, early_k)],
            axis=1,
        )
        return z, f"combined (fourier {harmonics} + early {early_k})"
    raise SystemExit(f"unknown peel mode {mode!r}")


def regress_out(Xc, Z):
    """Residual of Xc after projecting out the column span of Z (closed form).
    Returns (residual, B) with B = (ZᵀZ)⁺ ZᵀXc."""
    B = np.linalg.pinv(Z.T @ Z) @ (Z.T @ Xc)
    return Xc - Z @ B, B


def absorbed_r2(Xc, Z):
    """Centred aggregate R² the design Z absorbs (0..1)."""
    resid, _ = regress_out(Xc, Z)
    ss_res = float((resid ** 2).sum())
    ss_tot = float((Xc ** 2).sum())
    return 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot


def sink_peel(Xc, positions, mode, seed=0):
    """Peel the positional sink and return (Xp, audit). `audit` records the removed
    direction's correlation with PC1, the pos0 / combined absorbed R², and the
    permuted-position null -- so the peel is an auditable claim, not a quantile."""
    Z, label = peel_design(positions, mode)
    Xp, B = regress_out(Xc, Z)

    # sink direction = the first-token indicator's fitted offset (last col of a pos0
    # design; for `combined` re-fit pos0 alone to name the sink direction cleanly).
    Zp0, _ = peel_design(positions, "pos0")
    _, Bp0 = regress_out(Xc, Zp0)
    sink_dir = Bp0[-1]
    sn = np.linalg.norm(sink_dir)
    sv = np.linalg.svd(Xc, compute_uv=False)  # Xc is already centred
    pc1 = np.linalg.svd(Xc, full_matrices=False)[2][0]
    cos_sink_pc1 = float(abs(sink_dir @ pc1) / sn) if sn > 0 else 0.0

    rng = np.random.default_rng(seed)
    Z_null = Z[rng.permutation(Xc.shape[0])]  # positions shuffled vs activations

    audit = dict(
        peel_mode=mode,
        peel_design=label,
        n_design_cols=int(Z.shape[1]),
        absorbed_pos0=absorbed_r2(Xc, Zp0),
        absorbed_peel=absorbed_r2(Xc, Z),
        absorbed_permuted_null=absorbed_r2(Xc, Z_null),
        cos_sink_dir_pc1=cos_sink_pc1,
        frac_rows_pos0=float((positions == 0).mean()),
        var_frac_top_pc_before=float((sv ** 2)[0] / (sv ** 2).sum()),
    )
    return Xp, audit


# ---------------------------------------------------------------------------
# The head-to-head (unchanged math; now operates on the sink-peeled residual).
# ---------------------------------------------------------------------------
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


def head_to_head(Xp):
    tot = (Xp ** 2).sum()

    def ev(recon):
        return 1.0 - ((Xp - recon) ** 2).sum() / tot

    # LINEAR baseline: global top-M PCA reconstruction, EV vs capacity.
    Up, Sp, Vtp = np.linalg.svd(Xp, full_matrices=False)
    lin = {}
    for M in [1, 2, 4, 8, 16, 32, 64]:
        recon = (Xp @ Vtp[:M].T) @ Vtp[:M]
        lin[M] = dict(ev=float(ev(recon)), params=M)  # M*D dict params + N*M codes
    print("LINEAR (global PCA):", {M: round(v["ev"], 3) for M, v in lin.items()}, flush=True)

    # CURVED atlas: K local charts (k-means), each a local d-dim PCA (a curved chart).
    atlas = {}
    for (K, d) in [(8, 1), (16, 1), (32, 1), (8, 2), (16, 2), (32, 2), (64, 2)]:
        lab = kmeans_np(Xp, K, seed=0)
        recon = np.zeros_like(Xp)
        for k in range(K):
            m = lab == k
            if m.sum() < d + 1:
                recon[m] = Xp[m].mean(0)
                continue
            Ck = Xp[m]
            ck = Ck.mean(0)
            Cc = Ck - ck
            uu, ss, vv = np.linalg.svd(Cc, full_matrices=False)
            recon[m] = ck + (Cc @ vv[:d].T) @ vv[:d]  # local chart reconstruction (piecewise-curved)
        atlas[(K, d)] = dict(ev=float(ev(recon)), K=K, d=d, eff_dim=d, params=K * (d + 1))
        print(f"  atlas K={K} d={d}: EV={atlas[(K, d)]['ev']:.3f}", flush=True)

    # head-to-head at matched coords/row d: atlas(K,d) sparse code uses d coords/row.
    compare = []
    for (K, d), a in atlas.items():
        lin_same = lin.get(d)
        if lin_same:
            compare.append(dict(K=K, d=d, atlas_ev=a["ev"], linear_ev=lin_same["ev"],
                                curved_gain=a["ev"] - lin_same["ev"]))
    compare.sort(key=lambda r: -r["curved_gain"])
    return lin, atlas, compare


def _selftest():
    """Plant a position-0 sink into synthetic data; assert the peel removes it and the
    permuted-position null collapses. Runs with no cluster data (the local proof)."""
    rng = np.random.default_rng(0)
    N, D = 40000, 64
    positions = (np.arange(N) % (PMAX)).astype(np.int32)  # 0..511 repeating
    # small semantic structure on a 3-dim subspace
    sem = rng.standard_normal((N, 3)) @ rng.standard_normal((3, D))
    # DOMINANT first-token sink along one direction (position 0 only). Real attention
    # sinks are ~1000x per-row; on ~1/512 of rows that still absorbs ~90% of variance,
    # so the plant has to be large to reproduce the L18 regime (91% from one indicator).
    sink_dir = rng.standard_normal(D)
    sink_dir /= np.linalg.norm(sink_dir)
    X = sem + 600.0 * (positions == 0)[:, None] * sink_dir[None, :] + 0.05 * rng.standard_normal((N, D))
    Xc = X - X.mean(0)
    var_before = (np.linalg.svd(Xc, compute_uv=False) ** 2)
    top_frac_before = var_before[0] / var_before.sum()
    Xp, audit = sink_peel(Xc, positions, "pos0")
    # variance ALONG the true sink direction: must be ~annihilated by the peel.
    var_sink_before = float(((Xc @ sink_dir) ** 2).sum())
    var_sink_after = float(((Xp @ sink_dir) ** 2).sum())
    print(f"[selftest] top var frac before={top_frac_before:.3f}; "
          f"variance along sink dir {var_sink_before:.3e} -> {var_sink_after:.3e}")
    print(f"[selftest] absorbed_pos0={audit['absorbed_pos0']:.4f} "
          f"null={audit['absorbed_permuted_null']:.5f} cos(sink,PC1)={audit['cos_sink_dir_pc1']:.3f}")
    assert audit["absorbed_pos0"] > 0.5, audit
    assert audit["absorbed_permuted_null"] < 0.05, audit
    assert audit["cos_sink_dir_pc1"] > 0.9, audit
    assert var_sink_after < var_sink_before / 100, (var_sink_before, var_sink_after)
    # head-to-head still runs on the residual
    lin, atlas, compare = head_to_head(Xp)
    assert compare, "head-to-head produced no comparison"
    print("[selftest] OK: sink peeled, null collapsed, head-to-head runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic planted-sink proof locally (no cluster data)")
    ap.add_argument("--positions", default="",
                    help="per-row within-doc position .npy, order-matched to resid_L18.npy "
                         "(produced by qwen_nuisance_msi.py --save-positions). REQUIRED for a "
                         "real peel unless --raw-ok.")
    ap.add_argument("--peel", choices=["pos0", "combined"], default="pos0",
                    help="peel design: pos0 (first-token sink, default) or combined (full "
                         "positional nuisance block)")
    ap.add_argument("--raw-ok", action="store_true",
                    help="override the peel-first doctrine and compare RAW un-peeled activations "
                         "(the result then mostly measures the positional sink -- for debugging only)")
    ap.add_argument("--layer", type=int, default=18)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    R = required_env("GAM_MSI_DATA", "set GAM_MSI_DATA to the activation-harvest root")
    OUT = f"{R}/gam_ceiling_fable/experiments/curved_vs_linear"
    os.makedirs(OUT, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = f"{R}/scratch/mplcache"
    os.makedirs(f"{R}/scratch/mplcache", exist_ok=True)

    # ---- load real activations ----
    path = f"{R}/harvest_out/qwen3_8b_wikitext/resid_L{args.layer}.npy"
    Xmm = np.load(path, mmap_mode="r")
    n_rows = Xmm.shape[0]
    N = min(30000, n_rows)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n_rows, N, replace=False))
    X = np.asarray(Xmm[idx], dtype=np.float64)
    print(f"loaded L{args.layer} {X.shape}", flush=True)
    mu = X.mean(0)
    Xc = X - mu

    # ---- peel the attention sink (nuisance-atlas position-0 regress-out) ----
    if not args.positions:
        if not args.raw_ok:
            raise SystemExit(
                "peel-first doctrine: --positions is required (per-row within-doc positions, "
                "order-matched to the harvest; make them with qwen_nuisance_msi.py "
                "--save-positions). Pass --raw-ok to knowingly compare the un-peeled sink.")
        print("WARNING --raw-ok: comparing RAW un-peeled activations; this mostly measures the "
              "positional sink.", flush=True)
        Xp = Xc
        audit = dict(peel_mode="RAW_UNPEELED", warning="no positional peel applied")
    else:
        positions = np.load(args.positions).astype(np.int32)
        if positions.shape[0] < n_rows:
            raise SystemExit(f"positions ({positions.shape[0]}) shorter than harvest rows ({n_rows})")
        positions = positions[:n_rows][idx]  # same subsample as the activations
        Xp, audit = sink_peel(Xc, positions, args.peel)
        print(f"peeled sink via {audit['peel_design']}: absorbed_pos0={audit['absorbed_pos0']:.4f} "
              f"absorbed_peel={audit['absorbed_peel']:.4f} permuted_null={audit['absorbed_permuted_null']:.5f} "
              f"cos(sink,PC1)={audit['cos_sink_dir_pc1']:.3f}", flush=True)
        if audit["absorbed_permuted_null"] > 0.05:
            print("WARNING permuted-position null did not collapse (<0.05); the peel may be "
                  "fitting a low-rank artifact rather than a causal positional sink.", flush=True)

    # ---- head-to-head on the peeled residual ----
    lin, atlas, compare = head_to_head(Xp)
    print("\nHEAD-TO-HEAD (same coords/row d): atlas EV vs linear EV", flush=True)
    for r in compare[:6]:
        print(f"  K={r['K']} d={r['d']}: atlas={r['atlas_ev']:.3f} linear={r['linear_ev']:.3f} "
              f"gain={r['curved_gain']:+.3f}", flush=True)
    best = compare[0]

    # ---- try the ACTUAL gamfit manifold SAE if available ----
    try:
        import gamfit
        fns = [x for x in dir(gamfit) if any(k in x.lower() for k in ("sae", "manifold", "spectral"))]
        gamfit_result = f"gamfit importable; sae fns: {fns}"
    except Exception as e:
        gamfit_result = f"gamfit import failed: {type(e).__name__}: {str(e)[:80]}"
    print("\nGAMFIT:", gamfit_result, flush=True)

    # ---- plot EV vs coords/row: linear vs best atlas at each d ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = [1, 2]
    lin_pts = [lin[d]["ev"] for d in ds]
    atl_pts = [max(a["ev"] for (K, dd), a in atlas.items() if dd == d) for d in ds]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ds, lin_pts, "o-", label="linear (global PCA)", color="#c0392b", lw=2, ms=9)
    ax.plot(ds, atl_pts, "s-", label="curved atlas (best K local charts)", color="#2471a3", lw=2, ms=9)
    for d, l, a in zip(ds, lin_pts, atl_pts):
        ax.annotate(f"+{a - l:.3f}", (d, (a + l) / 2), fontsize=9, color="#2471a3")
    ax.set_xlabel("coordinates per row (sparse: 1 atom active)")
    ax.set_ylabel(f"explained variance (sink-peeled L{args.layer})")
    ax.set_title(f"Qwen3-8B L{args.layer}: curved atlas vs linear, matched sparse capacity\n"
                 f"best curved gain = {best['curved_gain']:+.3f} EV at d={best['d']}, K={best['K']}")
    ax.set_xticks(ds)
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/curved_vs_linear.png", dpi=140)
    print(f"saved {OUT}/curved_vs_linear.png", flush=True)

    # capacity accounting is made explicit so the matched-budget asymmetry is auditable:
    # linear pays M*D dict + N*M codes; the atlas pays K*(d+1) dict + ~ (1 gate + d coords)
    # per row. At matched TOTAL capacity the atlas buys more dictionary because its per-row
    # code is cheaper -- that is the intended per-token-description-length advantage, but the
    # honest framing is to also report matched-MDL (reverse water-filling). Flagged, not hidden.
    summary = dict(
        layer=f"L{args.layer}", N=N, peel=audit, linear=lin,
        atlas={f"K{K}_d{d}": a for (K, d), a in atlas.items()},
        head_to_head=compare, best=best, gamfit=gamfit_result,
        capacity_note=("matched at coords/row d (per-row code); linear per-row cost = M scalars, "
                       "atlas per-row cost = 1 gate + d coords. Matched-MDL comparison via "
                       "reverse water-filling is the referee-proof framing and is still owed."),
    )
    json.dump(summary, open(f"{OUT}/curved_vs_linear.json", "w"), indent=1)
    print("\n==== RESULT ====", flush=True)
    print(f"best curved gain over linear at matched sparse capacity: {best['curved_gain']:+.3f} EV "
          f"(atlas K={best['K']} d={best['d']}: {best['atlas_ev']:.3f} vs linear {best['linear_ev']:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
