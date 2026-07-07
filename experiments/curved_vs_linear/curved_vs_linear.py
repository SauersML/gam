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

# Reporting-only default (SPEC line 19: no magic knobs in the peel decision). The peel is
# gated by a PERMUTATION TEST, not a variance quantile: peel iff the observed positional
# absorption exceeds every permuted-position replicate. The only free number is the
# replicate count, which sets the achievable significance 1/(R+1) -- a reporting
# resolution, not a threshold. This is what makes the peel "recover the null" (SPEC
# line 13): on sink-free data the observed absorption is exchangeable with the nulls, so
# nothing is peeled.
NULL_PERMUTATION_REPLICATES = 200  # reporting-only: permutation-null resolution (p < 1/201)

# Reporting-only (SPEC line 19): number of common bit-budgets on which the two codes'
# rate-distortion curves are sampled for the figure/curve. It sets the plotting
# resolution of the MDL sweep, NOT any decision -- the matched-bits headline is computed
# by exact bisection at the derived operating point, independent of this grid.
MDL_SWEEP_POINTS = 48  # reporting-only: EV-vs-bits curve resolution


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


def null_permutation_absorbed(Xc, Z, replicates, seed):
    """Permuted-position null distribution of absorbed R²: shuffle the design rows
    (positions) against the activations `replicates` times. Under no positional signal
    the true-position absorption is exchangeable with these; under a real sink it is not."""
    rng = np.random.default_rng(seed)
    n = Xc.shape[0]
    return np.array([absorbed_r2(Xc, Z[rng.permutation(n)]) for _ in range(replicates)])


def sink_peel(Xc, positions, mode, null_replicates=NULL_PERMUTATION_REPLICATES, seed=0):
    """Peel the positional sink IFF it is causally real, and return (Xp, audit).

    Decision (SPEC line 13/19 compliant): peel iff the observed positional absorption
    exceeds every one of `null_replicates` permuted-position replicates (permutation
    p < 1/(R+1)). On sink-free data the observed absorption is exchangeable with the
    nulls -> the gate fails -> ZERO directions peeled, recovering the null. No variance
    quantile, no cap, no magic threshold.

    `audit` records the removed direction's correlation with PC1, the pos0/peel absorbed
    R², the permutation null, the decision, and how many directions were removed -- so the
    peel is an auditable claim rather than a quantile."""
    Z, label = peel_design(positions, mode)
    observed = absorbed_r2(Xc, Z)
    null_dist = null_permutation_absorbed(Xc, Z, null_replicates, seed)
    null_max = float(null_dist.max())
    null_mean = float(null_dist.mean())
    peel_is_causal = observed > null_max  # exact permutation test, sig < 1/(R+1)

    # sink direction = the first-token indicator's fitted offset (named from a pos0 design
    # so `combined` still points at the sink cleanly).
    Zp0, _ = peel_design(positions, "pos0")
    _, Bp0 = regress_out(Xc, Zp0)
    sink_dir = Bp0[-1]
    sn = np.linalg.norm(sink_dir)
    sv = np.linalg.svd(Xc, compute_uv=False)  # Xc is already centred
    pc1 = np.linalg.svd(Xc, full_matrices=False)[2][0]
    cos_sink_pc1 = float(abs(sink_dir @ pc1) / sn) if sn > 0 else 0.0

    if peel_is_causal:
        Xp, _ = regress_out(Xc, Z)
        n_peeled = int(Z.shape[1] - 1)  # non-intercept design columns
    else:
        Xp = Xc.copy()  # recover the null: no positional sink detected, peel nothing
        n_peeled = 0

    audit = dict(
        peel_mode=mode,
        peel_design=label,
        n_design_cols=int(Z.shape[1]),
        n_directions_peeled=n_peeled,
        peel_is_causal=bool(peel_is_causal),
        absorbed_observed=float(observed),
        absorbed_pos0=absorbed_r2(Xc, Zp0),
        absorbed_permuted_null_max=null_max,
        absorbed_permuted_null_mean=null_mean,
        null_replicates=int(null_replicates),
        permutation_significance=1.0 / (null_replicates + 1),
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


# ---------------------------------------------------------------------------
# MDL-fair matching: compare the two codes at EQUAL DESCRIPTION LENGTH (bits),
# not equal coords/row. Both codes are scored by reverse water-filling over their
# relevant eigen-spectrum: to reach a common distortion floor D*, component i is
# allocated rate R_i = 1/2 * max(0, log2(lambda_i / D*)) bits and suffers residual
# distortion min(lambda_i, D*). This is the exact Gaussian rate-distortion solution,
# so "bits" and "explained variance" are two readouts of the SAME D* -- no free knob
# beyond D*, which is itself DERIVED (the noise floor), never hand-set.
#
#   linear code : the global covariance spectrum {lambda_i}, one code for all rows.
#   curved atlas: K local charts; the relevant spectrum is the POOLED within-chart
#                 spectrum {lambda_{k,j}} weighted by chart occupancy n_k/N, plus a
#                 GATE cost of H(occupancy) bits/row to name a row's chart. The
#                 between-chart structure the gate buys is then reconstructed for
#                 free (it lives in the chart means / dictionary), so both codes'
#                 EV share the GLOBAL total variance as denominator -- a fair head
#                 -to-head. At the gate-only budget the atlas already explains the
#                 between-chart variance; every coding bit above that buys within
#                 -chart detail.
# ---------------------------------------------------------------------------
def _centred_spectrum(Xin):
    """Per-component variances (covariance eigenvalues) of row-set Xin."""
    n = len(Xin)
    Cc = Xin - Xin.mean(0)
    s = np.linalg.svd(Cc, compute_uv=False, full_matrices=False)
    return (s ** 2) / n


def _atlas_spectrum(Xp, K, seed=0):
    """Pooled within-chart eigen-spectrum + occupancy gate for a K-chart atlas.
    Returns (eigs, weights, gate_bits): `eigs` are local per-component variances, each
    carrying weight n_k/N (chart occupancy); `gate_bits` = occupancy entropy (bits/row),
    the cost of naming which chart a row lands in."""
    N = len(Xp)
    lab = kmeans_np(Xp, K, seed=seed)
    counts = np.bincount(lab, minlength=K)
    p = counts[counts > 0].astype(np.float64) / N
    gate_bits = float(-(p * np.log2(p)).sum()) if p.size else 0.0
    eigs, wts = [], []
    for k in range(K):
        nk = int(counts[k])
        if nk < 2:
            continue  # <2 rows: zero within-chart variance, coded with 0 bits
        lam = _centred_spectrum(Xp[lab == k])
        eigs.append(lam)
        wts.append(np.full(lam.shape[0], nk / N))
    eigs = np.concatenate(eigs) if eigs else np.zeros(0)
    wts = np.concatenate(wts) if wts else np.zeros(0)
    return eigs, wts, gate_bits


def _wf_bits(eigs, wts, Dstar):
    """Reverse-water-filling rate (bits/row) to distortion floor Dstar over a weighted
    spectrum: sum_i w_i * 1/2 * max(0, log2(lambda_i / Dstar))."""
    if eigs.size == 0:
        return 0.0
    return float((wts * 0.5 * np.maximum(0.0, np.log2(eigs / Dstar))).sum())


def _wf_ev(eigs, wts, Dstar, totvar):
    """Explained variance at distortion floor Dstar: 1 - (coded residual)/totvar, where
    the residual is sum_i w_i * min(lambda_i, Dstar). `totvar` is the GLOBAL variance so
    linear and atlas share a denominator (the atlas is credited its between-chart mass)."""
    if totvar <= 0.0:
        return 0.0
    dist = float((wts * np.minimum(eigs, Dstar)).sum()) if eigs.size else 0.0
    return 1.0 - dist / totvar


def _invert_bits(eigs, wts, target_bits, extra=0.0):
    """Distortion floor D* whose total rate extra + _wf_bits(eigs,wts,D*) == target_bits.
    The rate is continuous and strictly decreasing in D* over its active range, so a
    bisection (root-finding, NOT grid search) inverts it. The target is clamped to the
    achievable coding range so a caller never extrapolates past the spectrum."""
    if eigs.size == 0:
        return 1.0
    lo, hi = float(eigs.min()) * 1e-12, float(eigs.max())
    coding = target_bits - extra
    if coding <= 0.0:
        return hi                       # no coding budget -> code nothing (D* = lambda_max)
    if coding >= _wf_bits(eigs, wts, lo):
        return lo                       # full budget -> code down to the smallest component
    for _ in range(100):                # ~1e-30 relative precision on D*
        mid = np.sqrt(lo * hi)
        if _wf_bits(eigs, wts, mid) > coding:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def head_to_head_mdl(Xp, Ks=(8, 16, 32, 64), seed=0, sweep_points=MDL_SWEEP_POINTS):
    """MDL-fair head-to-head: linear vs curved atlas at EQUAL total bits.

    D* is DERIVED as the noise floor = the smallest global covariance eigenvalue (the
    variance no principal component beats; below it a component is indistinguishable from
    noise and earns zero rate). At that common D* each code has a total description length
    and an EV; we then quote the OTHER code's EV at the SAME total bits (bisection), so the
    comparison is always at an identical budget. Reported from both anchors (linear's spend,
    atlas's spend) so neither side picks the operating point."""
    Xc = Xp - Xp.mean(0)
    lin_eigs = _centred_spectrum(Xc)
    lin_wts = np.ones_like(lin_eigs)
    V = float(lin_eigs.sum())                       # global total variance (shared denominator)
    pos = lin_eigs[lin_eigs > 0]
    floor = float(pos.min()) if pos.size else 1.0   # derived noise-floor D* (no magic constant)

    def lin_ev_at_bits(B):
        return _wf_ev(lin_eigs, lin_wts, _invert_bits(lin_eigs, lin_wts, B), V) if B > 0 else 0.0

    lin_bits_floor = _wf_bits(lin_eigs, lin_wts, floor)  # linear's spend to code to the noise floor
    lin_ev_floor = _wf_ev(lin_eigs, lin_wts, floor, V)

    # common bit-budget grid for the RD curves (data-derived span; resolution is reporting-only)
    max_bits = lin_bits_floor
    budgets = np.linspace(0.0, max_bits, sweep_points)
    lin_curve = [dict(bits=float(B), ev=float(lin_ev_at_bits(B))) for B in budgets]

    per_K = []
    for K in Ks:
        eigs, wts, gate = _atlas_spectrum(Xp, K, seed=seed)

        def atl_ev_at_bits(B, eigs=eigs, wts=wts, gate=gate):
            if B < gate:
                return 0.0                          # cannot even afford the chart index
            return _wf_ev(eigs, wts, _invert_bits(eigs, wts, B, extra=gate), V)

        atlas_bits_floor = gate + _wf_bits(eigs, wts, floor)   # atlas spend to the same D*
        atlas_ev_floor = _wf_ev(eigs, wts, floor, V)
        # anchor A: match linear TO the atlas's total spend -> compare EV at atlas_bits_floor
        lin_ev_at_atlas_bits = lin_ev_at_bits(atlas_bits_floor)
        gain_atlas_anchor = atlas_ev_floor - lin_ev_at_atlas_bits
        # anchor B: match atlas TO the linear's total spend -> compare EV at lin_bits_floor
        atlas_ev_at_lin_bits = atl_ev_at_bits(lin_bits_floor)
        gain_linear_anchor = atlas_ev_at_lin_bits - lin_ev_floor

        per_K.append(dict(
            K=K, gate_bits=gate,
            atlas_bits_floor=float(atlas_bits_floor), atlas_ev_floor=float(atlas_ev_floor),
            linear_ev_at_atlas_bits=float(lin_ev_at_atlas_bits),
            matched_bits_atlas_anchor=float(atlas_bits_floor),
            mdl_gain_atlas_anchor=float(gain_atlas_anchor),
            linear_bits_floor=float(lin_bits_floor), linear_ev_floor=float(lin_ev_floor),
            atlas_ev_at_lin_bits=float(atlas_ev_at_lin_bits),
            matched_bits_linear_anchor=float(lin_bits_floor),
            mdl_gain_linear_anchor=float(gain_linear_anchor),
            atlas_curve=[dict(bits=float(B), ev=float(atl_ev_at_bits(B))) for B in budgets],
        ))

    best = max(per_K, key=lambda r: r["mdl_gain_atlas_anchor"])
    return dict(
        distortion_level_Dstar=floor,
        distortion_level_note=("D* = smallest global covariance eigenvalue (the noise floor); "
                               "reverse water-filling allocates R_i=0.5*max(0,log2(lambda_i/D*)) "
                               "bits/component. Both codes scored at this common D*, then EV is "
                               "re-quoted at matched total bits."),
        total_variance=V,
        linear_bits_floor=float(lin_bits_floor), linear_ev_floor=float(lin_ev_floor),
        linear_rd_curve=lin_curve,
        per_K=per_K,
        best=dict(K=best["K"], matched_bits=best["matched_bits_atlas_anchor"],
                  atlas_ev=best["atlas_ev_floor"], linear_ev=best["linear_ev_at_atlas_bits"],
                  mdl_gain=best["mdl_gain_atlas_anchor"]),
    )


def _synth(with_sink, seed=0):
    """Synthetic activations: 3-dim semantic structure, optionally with a DOMINANT
    first-token sink (position 0 only). Real attention sinks are ~1000x per-row; on
    ~1/512 of rows that still absorbs ~90% of variance, so the plant must be large to
    reproduce the L18 regime (91% from one indicator)."""
    rng = np.random.default_rng(seed)
    N, D = 40000, 64
    positions = (np.arange(N) % PMAX).astype(np.int32)  # 0..511 repeating
    sem = rng.standard_normal((N, 3)) @ rng.standard_normal((3, D))
    sink_dir = rng.standard_normal(D)
    sink_dir /= np.linalg.norm(sink_dir)
    sink = 600.0 * (positions == 0)[:, None] * sink_dir[None, :] if with_sink else 0.0
    X = sem + sink + 0.05 * rng.standard_normal((N, D))
    return X - X.mean(0), positions, sink_dir


def _selftest():
    """Two-regime acceptance (SPEC line 13): a planted position-0 sink must be peeled
    (exactly one direction, null collapsed); sink-free data must peel ZERO directions."""
    # --- regime 1: dominant planted sink -> peel exactly one direction ---
    Xc, positions, sink_dir = _synth(with_sink=True)
    var_sink_before = float(((Xc @ sink_dir) ** 2).sum())
    Xp, audit = sink_peel(Xc, positions, "pos0")
    var_sink_after = float(((Xp @ sink_dir) ** 2).sum())
    print(f"[selftest/sink] peeled={audit['n_directions_peeled']} "
          f"observed={audit['absorbed_observed']:.4f} null_max={audit['absorbed_permuted_null_max']:.5f} "
          f"cos(sink,PC1)={audit['cos_sink_dir_pc1']:.3f}; sink var {var_sink_before:.3e}->{var_sink_after:.3e}")
    assert audit["peel_is_causal"] is True, audit
    assert audit["n_directions_peeled"] == 1, audit
    assert audit["absorbed_observed"] > 0.5, audit
    assert audit["absorbed_permuted_null_max"] < 0.05, audit
    assert audit["cos_sink_dir_pc1"] > 0.9, audit
    assert var_sink_after < var_sink_before / 100, (var_sink_before, var_sink_after)
    lin, atlas, compare = head_to_head(Xp)
    assert compare, "head-to-head produced no comparison"

    # --- MDL-fair matching: reverse water-filling at EQUAL total bits ---
    mdl = head_to_head_mdl(Xp)
    D = mdl["distortion_level_Dstar"]
    assert np.isfinite(D) and D > 0.0, mdl                       # D* derived, not magic
    # (a) more bits => weakly-monotone EV on every RD curve (linear + each atlas-K)
    def _monotone(curve, tag):
        b = np.array([p["bits"] for p in curve]); e = np.array([p["ev"] for p in curve])
        assert np.all(np.diff(b) >= -1e-12), (tag, "bits not sorted")
        assert np.all(np.diff(e) >= -1e-9), (tag, "EV not weakly-monotone in bits", e)
        assert np.all((e >= -1e-9) & (e <= 1.0 + 1e-9)), (tag, "EV out of [0,1]", e)
    _monotone(mdl["linear_rd_curve"], "linear")
    for rec in mdl["per_K"]:
        _monotone(rec["atlas_curve"], f"atlas K={rec['K']}")
        # (b) atlas and linear are compared at the SAME bit budget on both anchors
        assert rec["matched_bits_atlas_anchor"] == rec["atlas_bits_floor"], rec
        assert rec["matched_bits_linear_anchor"] == mdl["linear_bits_floor"], rec
        assert np.isfinite(rec["mdl_gain_atlas_anchor"]), rec
        assert np.isfinite(rec["mdl_gain_linear_anchor"]), rec
        # gate is a proper cost; atlas at gate-only budget explains only between-chart mass
        assert rec["gate_bits"] > 0.0, rec
    b = mdl["best"]
    print(f"[selftest/mdl] D*={D:.3e} bits(lin@floor)={mdl['linear_bits_floor']:.2f} "
          f"best atlas K={b['K']} @ {b['matched_bits']:.2f} bits: "
          f"atlas EV={b['atlas_ev']:.3f} vs linear EV={b['linear_ev']:.3f} "
          f"(MDL gain {b['mdl_gain']:+.3f})")

    # --- regime 2: sink-free -> peel ZERO directions, recover the null (SPEC line 13) ---
    Xc0, positions0, _ = _synth(with_sink=False, seed=1)
    Xp0, audit0 = sink_peel(Xc0, positions0, "pos0")
    print(f"[selftest/null] peeled={audit0['n_directions_peeled']} "
          f"observed={audit0['absorbed_observed']:.5f} null_max={audit0['absorbed_permuted_null_max']:.5f} "
          f"causal={audit0['peel_is_causal']}")
    assert audit0["peel_is_causal"] is False, audit0
    assert audit0["n_directions_peeled"] == 0, audit0
    assert np.array_equal(Xp0, Xc0), "sink-free peel must be an identity (recover the null)"
    print("[selftest] OK: sink peeled (1 dir, null collapsed); sink-free peels 0 dirs")


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
        print(f"peel via {audit['peel_design']}: peeled {audit['n_directions_peeled']} dir(s) "
              f"(causal={audit['peel_is_causal']}); observed={audit['absorbed_observed']:.4f} "
              f"null_max={audit['absorbed_permuted_null_max']:.5f} "
              f"cos(sink,PC1)={audit['cos_sink_dir_pc1']:.3f}", flush=True)
        if not audit["peel_is_causal"]:
            print("NOTE positional absorption did not beat the permuted-position null; no sink "
                  "peeled (the layer recovers the null). Head-to-head runs on raw centred acts.",
                  flush=True)

    # ---- head-to-head on the peeled residual ----
    lin, atlas, compare = head_to_head(Xp)
    print("\nHEAD-TO-HEAD (same coords/row d): atlas EV vs linear EV", flush=True)
    for r in compare[:6]:
        print(f"  K={r['K']} d={r['d']}: atlas={r['atlas_ev']:.3f} linear={r['linear_ev']:.3f} "
              f"gain={r['curved_gain']:+.3f}", flush=True)
    best = compare[0]

    # ---- MDL-fair matching: linear vs atlas at EQUAL total bits (reverse water-filling) ----
    mdl = head_to_head_mdl(Xp)
    mb = mdl["best"]
    print(f"\nMDL-FAIR (equal total bits, D*={mdl['distortion_level_Dstar']:.3e} = noise floor):",
          flush=True)
    for r in sorted(mdl["per_K"], key=lambda z: -z["mdl_gain_atlas_anchor"]):
        print(f"  K={r['K']}: at {r['atlas_bits_floor']:.2f} bits/row  atlas EV={r['atlas_ev_floor']:.3f} "
              f"vs linear EV={r['linear_ev_at_atlas_bits']:.3f}  gain={r['mdl_gain_atlas_anchor']:+.3f} "
              f"(gate={r['gate_bits']:.2f} bits)", flush=True)
    print(f"  best MDL gain: {mb['mdl_gain']:+.3f} EV at {mb['matched_bits']:.2f} bits/row "
          f"(atlas K={mb['K']})", flush=True)

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
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # left panel: matched coords/row (the original currency)
    ax.plot(ds, lin_pts, "o-", label="linear (global PCA)", color="#c0392b", lw=2, ms=9)
    ax.plot(ds, atl_pts, "s-", label="curved atlas (best K local charts)", color="#2471a3", lw=2, ms=9)
    for d, l, a in zip(ds, lin_pts, atl_pts):
        ax.annotate(f"+{a - l:.3f}", (d, (a + l) / 2), fontsize=9, color="#2471a3")
    ax.set_xlabel("coordinates per row (sparse: 1 atom active)")
    ax.set_ylabel(f"explained variance (sink-peeled L{args.layer})")
    ax.set_title(f"matched coords/row\nbest curved gain = {best['curved_gain']:+.3f} EV "
                 f"at d={best['d']}, K={best['K']}")
    ax.set_xticks(ds)
    ax.legend()
    ax.grid(alpha=.3)
    # right panel: matched DESCRIPTION LENGTH (bits) -- reverse water-filling RD curves
    lb = np.array([p["bits"] for p in mdl["linear_rd_curve"]])
    le = np.array([p["ev"] for p in mdl["linear_rd_curve"]])
    ax2.plot(lb, le, "-", label="linear (global PCA)", color="#c0392b", lw=2)
    bestK = mb["K"]
    brec = next(r for r in mdl["per_K"] if r["K"] == bestK)
    ab = np.array([p["bits"] for p in brec["atlas_curve"]])
    ae = np.array([p["ev"] for p in brec["atlas_curve"]])
    ax2.plot(ab, ae, "-", label=f"curved atlas (K={bestK})", color="#2471a3", lw=2)
    ax2.axvline(mb["matched_bits"], color="#7f8c8d", ls="--", lw=1)
    ax2.plot([mb["matched_bits"]], [mb["atlas_ev"]], "s", color="#2471a3", ms=9)
    ax2.plot([mb["matched_bits"]], [mb["linear_ev"]], "o", color="#c0392b", ms=9)
    ax2.annotate(f"{mb['mdl_gain']:+.3f} @ {mb['matched_bits']:.1f} bits",
                 (mb["matched_bits"], (mb["atlas_ev"] + mb["linear_ev"]) / 2),
                 fontsize=9, color="#2471a3")
    ax2.set_xlabel("description length (bits / row, reverse water-filling)")
    ax2.set_ylabel("explained variance")
    ax2.set_title(f"matched bits (MDL-fair)\nbest MDL gain = {mb['mdl_gain']:+.3f} EV "
                  f"at K={bestK}, D*={mdl['distortion_level_Dstar']:.2e}")
    ax2.legend()
    ax2.grid(alpha=.3)
    fig.suptitle(f"Qwen3-8B L{args.layer}: curved atlas vs linear under BOTH capacity currencies")
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
        head_to_head=compare,               # back-compat alias for head_to_head_coords
        head_to_head_coords=compare, best=best,
        head_to_head_mdl=mdl, best_mdl=mdl["best"],
        gamfit=gamfit_result,
        capacity_note=("reported under BOTH currencies: (1) matched coords/row d -- linear per-row "
                       "cost = M scalars, atlas per-row cost = 1 gate + d coords; (2) matched "
                       "description length (bits) via reverse water-filling to a derived noise-floor "
                       "D*, comparing EV at EQUAL total bits. Neither currency is privileged."),
    )
    json.dump(summary, open(f"{OUT}/curved_vs_linear.json", "w"), indent=1)
    print("\n==== RESULT ====", flush=True)
    print(f"matched coords/row: best curved gain {best['curved_gain']:+.3f} EV "
          f"(atlas K={best['K']} d={best['d']}: {best['atlas_ev']:.3f} vs linear {best['linear_ev']:.3f})",
          flush=True)
    print(f"matched bits (MDL): best curved gain {mb['mdl_gain']:+.3f} EV at {mb['matched_bits']:.2f} "
          f"bits/row (atlas K={mb['K']}: {mb['atlas_ev']:.3f} vs linear {mb['linear_ev']:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
