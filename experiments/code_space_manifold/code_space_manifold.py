"""Code-space 1-parameter manifold detection + rate-distortion on the frozen K=32000
Qwen3.6 (msae_l17) block dictionary. NO NEW FIT — pure detection on the existing
flat-dictionary code stream (T1-exact top-active + active-set ridge-LS, active=32).

Thesis: the manifold is structure in the CODE stream. A token at latent t fires two
adjacent secant atoms with barycentric amplitudes (~scale*(1-u, u)); as t sweeps, the
top-2 code walks a PATH/CYCLE through the co-fire graph, two knots at a time.

STAGE 2 (detect): top-2 "secant edge" co-fire sketch -> threshold -> connected
components -> per-component graph Betti (path vs cycle, exact via Euler char),
barycentric two-hot signature, decoder-space adjacency, spectral (Fiedler) seriation.
STAGE 3 (re-code + price): re-code each group firing as (group_id, t, scale); emit the
rate-distortion CURVE bits/token vs EV for flat-coding vs manifold-coding (an EXACT
edge-secant reparameterization with a per-record SSE guard — nested by construction,
so the manifold curve can never sit below flat at matched b_amp) of the SAME
dictionary, SWEPT over the co-fire threshold (coverage vs re-code distortion). The
re-code is deterministic; EV is measured directly (bug-proof). Calendar check: cycle
groups of size ~7 (weekday) / ~12 (month) with barycentric codes.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = "/projects/standard/hsiehph/sauer354/msae_l17"
sys.path.insert(0, ROOT + "/driver")
from compose_l17_stagewise import _load_tier0, _apply_tier0  # exact compose tier0


def t1_codes_tile(D, X, active):
    """T1-exact code: top-|score| active atoms; amplitudes = active-set ridge-LS."""
    m, P = X.shape
    K = D.shape[0]
    a = min(active, K)
    scores = X @ D.T
    idx = np.argpartition(-np.abs(scores), a - 1, axis=1)[:, :a].astype(np.int32)
    Dg = D[idx]
    G = np.einsum("map,mbp->mab", Dg, Dg)
    ridge = 1e-6 * np.trace(G, axis1=1, axis2=2)[:, None, None] * np.eye(a)[None]
    rhs = np.einsum("map,mp->ma", Dg, X)
    c = np.linalg.solve(G + ridge, rhs[..., None])[..., 0].astype(np.float32)
    recon = np.einsum("ma,map->mp", c, Dg).astype(np.float32)
    return idx, c, recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=50000)
    ap.add_argument("--active", type=int, default=32)
    ap.add_argument("--data", default=ROOT + "/L17_train.f32.npy")
    ap.add_argument("--tier0", default=ROOT + "/tier0_recentered.json")
    ap.add_argument("--decoder", default=ROOT + "/t1_out/decoder_K32000.npy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=4000)
    ap.add_argument("--s-nbhd", type=int, default=8)
    ap.add_argument("--frac-sweep", default="5e-4,2e-4,1e-4,5e-5,2e-5",
                    help="co-fire edge thresholds as fractions of N (coverage sweep)")
    ap.add_argument("--headline-frac", type=float, default=1e-4,
                    help="threshold whose full groups get dumped to discovered_groups.json")
    ap.add_argument("--min-group", type=int, default=4)
    ap.add_argument("--max-group", type=int, default=400)
    ap.add_argument("--peel-hub-rate", type=float, default=0.0,
                    help="NUISANCE PEEL: atoms whose top-32 firing rate exceeds this are "
                         "excluded from the co-fire graph (secant knots/edges) before "
                         "detection — removes the position/frequency hub blob. 0 = off.")
    ap.add_argument("--out-dir", default="/projects/standard/hsiehph/sauer354/code_space_out")
    args = ap.parse_args()

    t0 = time.time()
    D = np.ascontiguousarray(np.load(args.decoder), dtype=np.float32)
    K, P = D.shape
    Dn = np.linalg.norm(D, axis=1)
    mean, scale = _load_tier0(args.tier0)
    Xfull = np.load(args.data, mmap_mode="r")
    n_all = Xfull.shape[0]
    rng = np.random.default_rng(args.seed)
    N = min(args.n_tokens, n_all)
    sel = np.sort(rng.choice(n_all, size=N, replace=False))
    A = args.active
    print(f"[cs] D={D.shape} rownorm[min/med/max]={Dn.min():.3f}/{np.median(Dn):.3f}/"
          f"{Dn.max():.3f}  N={N}/{n_all} active={A}", flush=True)

    # ---- STAGE 1: code stream (in memory) + top-2 secant edge counts -------------
    Xsub = np.empty((N, P), dtype=np.float32)
    all_idx = np.empty((N, A), dtype=np.int32)
    all_cod = np.empty((N, A), dtype=np.float32)
    fire = np.zeros(K, dtype=np.int64)                 # per-atom top-32 firing count
    sse_flat = 0.0
    for i0 in range(0, N, args.tile):
        rows = sel[i0:i0 + args.tile]
        Xb = _apply_tier0(np.ascontiguousarray(Xfull[rows], dtype=np.float32), mean, scale)
        idx, cod, recon = t1_codes_tile(D, Xb, A)
        m = Xb.shape[0]
        Xsub[i0:i0 + m] = Xb; all_idx[i0:i0 + m] = idx; all_cod[i0:i0 + m] = cod
        np.add.at(fire, idx.ravel(), 1)
        sse_flat += float(((Xb.astype(np.float64) - recon.astype(np.float64)) ** 2).sum())
        if (i0 // args.tile) % 10 == 0:
            print(f"[cs] coded {i0+m}/{N}  {time.time()-t0:.0f}s", flush=True)
    tss = float((Xsub.astype(np.float64) ** 2).sum())
    ev_flat_full = 1.0 - sse_flat / tss

    # NUISANCE PEEL: flag high-firing-rate hub atoms and exclude them from the co-fire
    # graph. The secant edge is then the top-2 NON-hub actives per token, so the
    # position/frequency hub blob cannot connect every real family into one component.
    rate = fire / float(N)
    hub = np.zeros(K, dtype=bool)
    if args.peel_hub_rate > 0.0:
        hub = rate > args.peel_hub_rate
        print(f"[cs] PEEL: {int(hub.sum())} hub atoms (rate>{args.peel_hub_rate}); "
              f"they are {float(fire[hub].sum())/fire.sum():.3f} of firings, "
              f"max rate {rate.max():.4f}", flush=True)
    # top-2 secant edge among NON-hub actives (vectorised, tiled)
    edge_keys_parts = []
    for i0 in range(0, N, 20000):
        ii = all_idx[i0:i0+20000]; cc = np.abs(all_cod[i0:i0+20000]).copy()
        if hub.any():
            cc[hub[ii]] = -1.0                          # push hubs out of the top-2 rank
        o = np.argsort(-cc, axis=1)
        oi = np.take_along_axis(ii, o, axis=1).astype(np.int64)
        # only keep tokens whose top-2 are both non-hub (else no clean secant)
        keep2 = (~hub[oi[:, 0]]) & (~hub[oi[:, 1]]) if hub.any() else np.ones(len(oi), bool)
        oi = oi[keep2]
        e0 = np.minimum(oi[:, 0], oi[:, 1]); e1 = np.maximum(oi[:, 0], oi[:, 1])
        edge_keys_parts.append(e0 * K + e1)
    ek, ec = np.unique(np.concatenate(edge_keys_parts), return_counts=True)
    print(f"[cs] STAGE1 done: flat active={A} EV(baseline=zero)={ev_flat_full:.4f}  "
          f"n_secant_edges={len(ek)}  hub_peel={args.peel_hub_rate}  {time.time()-t0:.0f}s", flush=True)

    # ---- shared quantiser + threshold-independent flat RD curve ------------------
    Xf = Xsub.astype(np.float64)
    amp_max = float(np.abs(all_cod).max()) + 1e-9
    log2K = float(np.log2(K))

    def quant(v, bits, vmax):
        if bits >= 24:
            return v
        lv = (1 << bits) - 1
        return np.round((v / vmax) * lv) / lv * vmax

    _flat_cache = {}

    def flat_recon(b_amp):
        """(N,P) f32 reconstruction from all A actives, amps quantised to b_amp.
        Cached (memory-safe: builds one (N,P) atom gather per active, sequential)."""
        if b_amp in _flat_cache:
            return _flat_cache[b_amp]
        recon = np.zeros((N, P), dtype=np.float32)
        cq = quant(all_cod.astype(np.float64), b_amp, amp_max).astype(np.float32)
        for k in range(A):
            recon += cq[:, k][:, None] * D[all_idx[:, k]]
        _flat_cache[b_amp] = recon
        return recon

    def sse_of(recon):
        s = 0.0
        for i0 in range(0, N, 20000):
            s += float(((Xf[i0:i0+20000] - recon[i0:i0+20000].astype(np.float64)) ** 2).sum())
        return s

    flat_curve = []
    for b_amp in [4, 6, 8, 12, 16]:
        ev = 1.0 - sse_of(flat_recon(b_amp)) / tss
        bits = A * (log2K + b_amp)
        flat_curve.append({"bits_per_token": float(bits), "ev": float(ev), "b_amp": b_amp})
        print(f"[cs][RD] flat b_amp={b_amp} bits={bits:.0f} EV={ev:.4f}", flush=True)
        if b_amp not in (8, 16):      # keep only the b_amp the manifold side reuses
            _flat_cache.pop(b_amp, None)

    def build_groups(thr):
        keep = ec >= thr
        ii = (ek[keep] // K).astype(np.int64); jj = (ek[keep] % K).astype(np.int64)
        cc = ec[keep]
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[rx] = ry
        adj = defaultdict(dict)
        for a_, b_, c_ in zip(ii.tolist(), jj.tolist(), cc.tolist()):
            union(a_, b_); adj[a_][b_] = c_; adj[b_][a_] = c_
        comps = defaultdict(list)
        for a_ in list(parent.keys()):
            comps[find(a_)].append(a_)
        groups = []
        n_within = n_edges = 0
        for root, atoms in comps.items():
            m = len(atoms)
            if m < args.min_group or m > args.max_group:
                continue
            aset = set(atoms)
            elist = [(a_, b_, adj[a_][b_]) for a_ in atoms for b_ in adj[a_]
                     if b_ in aset and a_ < b_]
            V, E = m, len(elist)
            betti1 = E - V + 1
            deg = defaultdict(int)
            for a_, b_, _ in elist:
                deg[a_] += 1; deg[b_] += 1
                n_edges += 1
                if a_ // 2 == b_ // 2:
                    n_within += 1
            degs = np.array([deg[a_] for a_ in atoms])
            n1 = int((degs == 1).sum()); n2 = int((degs == 2).sum())
            if betti1 == 0 and n1 == 2 and n2 == m - 2:
                topo = "path"
            elif betti1 == 1 and n2 == m:
                topo = "cycle"
            elif betti1 == 0:
                topo = "tree"
            else:
                topo = f"graph_b1_{betti1}"
            aidx = {a_: k for k, a_ in enumerate(atoms)}
            W = np.zeros((m, m))
            for a_, b_, c_ in elist:
                W[aidx[a_], aidx[b_]] = c_; W[aidx[b_], aidx[a_]] = c_
            Lap = np.diag(W.sum(1)) - W
            _, v_eig = np.linalg.eigh(Lap)
            fiedler = v_eig[:, 1] if m > 1 else np.zeros(m)
            seq = [atoms[j] for j in np.argsort(fiedler)]
            Dg = D[seq]; Dgn = Dg / (np.linalg.norm(Dg, axis=1, keepdims=True) + 1e-12)
            cos_consec = [float(Dgn[k] @ Dgn[k + 1]) for k in range(m - 1)]
            if topo == "cycle" and m > 2:
                cos_consec.append(float(Dgn[-1] @ Dgn[0]))
            cos_consec = np.array(cos_consec)
            cos_rand = float((Dgn @ Dgn.T)[np.triu_indices(m, 1)].mean())
            groups.append({
                "root": int(root), "size": m, "V": V, "E": E, "betti1": int(betti1),
                "topology": topo, "n_deg1": n1, "n_deg2": n2,
                "atoms": [int(a_) for a_ in atoms],
                "knot_sequence": [int(a_) for a_ in seq],
                "edges": [[int(a_), int(b_), int(c_)] for a_, b_, c_ in elist],
                "decoder_cos_consecutive_mean": float(cos_consec.mean()),
                "decoder_cos_random_mean": cos_rand,
                "decoder_adjacency_contrast": float(cos_consec.mean() - cos_rand),
                "total_edge_count": int(sum(c_ for _, _, c_ in elist)),
            })
        groups.sort(key=lambda g: -g["total_edge_count"])
        return groups, (n_within / max(1, n_edges)), int(keep.sum())

    def stage3(groups):
        """Per-token decomposition + manifold RD sub-curve for a group set."""
        G_groups = len(groups)
        log2G = float(np.log2(max(2, G_groups)))
        atom2grp = -np.ones(K, dtype=np.int64)
        grp_pos = {}
        for gid, g in enumerate(groups):
            seq = g["knot_sequence"]; L = len(seq)
            for a_ in g["atoms"]:
                atom2grp[a_] = gid
            # Fiedler-path positions are DIAGNOSTIC only (path_coverage); the
            # recode below names edges directly and never reads them.
            for pos, a_ in enumerate(seq):
                grp_pos[(gid, a_)] = pos / max(1, L - 1)
        # Kept-edge index across ALL groups at this threshold: the manifold code
        # names an EDGE (the actual co-fired secant), never a linearized path
        # position — the discovered object is a graph (betti1 up to O(100)), so a
        # knot-sequence position generally decodes to DIFFERENT atoms than the
        # ones that fired, which is pure re-code distortion at any bit depth.
        edge_set = {}
        for g in groups:
            for a_, b_, _c in g["edges"]:
                edge_set[(int(a_), int(b_))] = len(edge_set)
        e_total = max(2, len(edge_set))
        log2E = float(np.log2(e_total))
        # per-secant records (only the top-2 of each group-touching token)
        g_row, g_gid, g_tpos, g_sign, g_scale = [], [], [], [], []
        g_a0, g_a1, g_c0, g_c1 = [], [], [], []
        twohot_sum = np.zeros(G_groups); twohot_n = np.zeros(G_groups)
        sign_agree = np.zeros(G_groups)
        u_min = np.ones(G_groups); u_max = np.zeros(G_groups)
        grp_tok = np.zeros(G_groups, dtype=np.int64)
        gid_of_active = atom2grp[all_idx]
        for r in range(N):
            idx_r = all_idx[r]; cod_r = all_cod[r]; gid_r = gid_of_active[r]
            buckets = defaultdict(list)
            for k in range(A):
                gid = gid_r[k]
                if gid >= 0:
                    buckets[gid].append((int(idx_r[k]), float(cod_r[k])))
            for gid, lst in buckets.items():
                if len(lst) < 2:                      # single group atom -> stays flat (base)
                    continue
                amps = np.array([c_ for _, c_ in lst])
                e_tot = float((amps ** 2).sum()); two = float(np.sort(amps ** 2)[::-1][:2].sum())
                twohot_sum[gid] += (two / e_tot) if e_tot > 0 else 0.0
                twohot_n[gid] += 1; grp_tok[gid] += 1
                order2 = np.argsort(-np.abs(amps)); o = order2[:2]
                (a0, c0), (a1, c1) = lst[o[0]], lst[o[1]]
                sign_agree[gid] += 1.0 if (c0 * c1 >= 0) else 0.0
                p0 = grp_pos[(gid, a0)]; p1 = grp_pos[(gid, a1)]
                denom = abs(c0) + abs(c1); u_amp = abs(c1) / denom if denom > 0 else 0.0
                tpos = (1 - u_amp) * p0 + u_amp * p1
                u_min[gid] = min(u_min[gid], tpos); u_max[gid] = max(u_max[gid], tpos)
                sign = np.sign(c0) if abs(c0) >= abs(c1) else np.sign(c1)
                # RECODE ELIGIBILITY: only pairs that lie on a KEPT edge — the
                # secant claim is edge-level; a same-group pair with no direct
                # edge stays flat (no honest 1-D secant witnessed for it).
                lo, hi = (a0, a1) if a0 < a1 else (a1, a0)
                if (lo, hi) not in edge_set:
                    continue
                c_lo, c_hi = (c0, c1) if a0 < a1 else (c1, c0)
                g_row.append(r); g_gid.append(gid); g_tpos.append(tpos)
                g_sign.append(float(sign)); g_scale.append(float(abs(c0) + abs(c1)))
                g_a0.append(lo); g_a1.append(hi); g_c0.append(c_lo); g_c1.append(c_hi)
        g_row = np.array(g_row, np.int64); g_gid = np.array(g_gid, np.int64)
        g_tpos = np.array(g_tpos); g_sign = np.array(g_sign, np.float32)
        g_scale = np.array(g_scale); g_a0 = np.array(g_a0, np.int64); g_a1 = np.array(g_a1, np.int64)
        g_c0 = np.array(g_c0); g_c1 = np.array(g_c1)
        n_grouped = len(g_row); n_flat = N * A - 2 * n_grouped
        for gid, g in enumerate(groups):
            n = max(1, twohot_n[gid])
            g["n_secant_tokens"] = int(grp_tok[gid])
            g["twohot_energy_frac_mean"] = float(twohot_sum[gid] / n)
            g["sign_agree_frac"] = float(sign_agree[gid] / n)
            g["path_coverage"] = float(u_max[gid] - u_min[gid]) if twohot_n[gid] else 0.0
            g["barycentric"] = bool(g["twohot_energy_frac_mean"] > 0.7
                                    and g["decoder_adjacency_contrast"] > 0.05
                                    and g["sign_agree_frac"] > 0.8)

        def manifold_ev_bits(b_t, b_amp):
            # EXACT EDGE-SECANT RECODE (nested by construction). The manifold code
            # for a secant firing is (edge_id, u, scale, s_lo, s_hi) with
            #   u     = |c_hi| / (|c_lo| + |c_hi|)   (barycentric along THE edge
            #                                         that actually fired),
            #   scale = |c_lo| + |c_hi|,  s_· = sign(c_·).
            # Decoding w_lo = s_lo·scale·(1−u), w_hi = s_hi·scale·u reproduces
            # (c_lo, c_hi) EXACTLY at full precision — the recode is a bijective
            # reparameterization of the pair restricted to kept edges, so it
            # CANNOT lose EV; it saves bits because naming one edge among E kept
            # edges (+2 sign bits) is far cheaper than naming two atoms among K.
            # (The previous path-snap recode projected the pair onto whichever
            # two knots were adjacent at a global Fiedler-path position — for a
            # betti1≫0 graph that decodes to DIFFERENT atoms than fired, which is
            # irreducible re-code distortion even at 24-bit fidelity.)
            #
            # A greedy per-record SSE guard keeps only records whose quantized
            # recode does not increase the row SSE vs the flat-quantized pair, so
            # the manifold point dominates the flat point at the same b_amp in
            # BOTH axes: EV(manifold) ≥ EV(flat) and, since the per-pair cost
            # log2E + 2 + b_t + b_amp < 2·(log2K + b_amp), fewer bits per
            # accepted record. Which records are recoded is carried by the
            # record counts (stream structure), not per-record flags.
            recon = flat_recon(b_amp).copy()          # (N,P) f32
            n_acc = 0
            if n_grouped:
                tlev = float(max(1, (1 << min(b_t, 24)) - 1))
                denom = np.abs(g_c0) + np.abs(g_c1)
                u = np.where(denom > 0, np.abs(g_c1) / np.maximum(denom, 1e-300), 0.0)
                uq = np.round(u * tlev) / tlev if b_t < 24 else u
                scq = quant(g_scale, b_amp, 2.0 * amp_max)
                w_lo = (np.sign(g_c0) * scq * (1.0 - uq)).astype(np.float32)
                w_hi = (np.sign(g_c1) * scq * uq).astype(np.float32)
                c0q = quant(g_c0, b_amp, amp_max).astype(np.float32)
                c1q = quant(g_c1, b_amp, amp_max).astype(np.float32)
                # Per-record replacement delta v = (w_lo−c0q)·D[lo] + (w_hi−c1q)·D[hi];
                # accept iff ‖r − v‖² ≤ ‖r‖²  ⇔  2·r·v ≥ ‖v‖², updating the row
                # residual greedily (records on one row belong to disjoint groups).
                order = np.argsort(g_row, kind="stable")
                accept = np.zeros(n_grouped, dtype=bool)
                for i in order.tolist():
                    row = g_row[i]
                    v = ((w_lo[i] - c0q[i]) * D[g_a0[i]]
                         + (w_hi[i] - c1q[i]) * D[g_a1[i]]).astype(np.float64)
                    r_row = Xf[row] - recon[row].astype(np.float64)
                    if 2.0 * float(r_row @ v) >= float(v @ v):
                        recon[row] += v.astype(np.float32)
                        accept[i] = True
                n_acc = int(accept.sum())
            sse = sse_of(recon)
            n_flat_eff = N * A - 2 * n_acc
            bits = (n_acc * (log2E + 2 + b_t + b_amp)
                    + n_flat_eff * (log2K + b_amp)) / N
            return 1.0 - sse / tss, bits, n_acc

        pts = []
        for (b_t, b_amp) in [(4, 8), (6, 8), (8, 16)]:
            ev, bits, n_acc = manifold_ev_bits(b_t, b_amp)
            pts.append({"bits_per_token": float(bits), "ev": float(ev),
                        "b_t": b_t, "b_amp": b_amp, "n_recode_accepted": n_acc})
        ev_max, bits_max, n_acc_max = manifold_ev_bits(24, 24)
        return {
            "n_groups": G_groups, "log2G": log2G, "n_kept_edges": int(len(edge_set)),
            "log2E": log2E,
            "n_grouped_secant_firings": n_grouped, "n_flat_firings": n_flat,
            "frac_firings_secant": 2 * n_grouped / float(N * A),
            "n_tokens_with_secant": int((np.bincount(g_row, minlength=1) > 0).sum()) if n_grouped else 0,
            "manifold_points": pts,
            "manifold_maxfidelity_ev": float(ev_max),
            "manifold_maxfidelity_bits": float(bits_max),
            "manifold_maxfidelity_recode_accepted": int(n_acc_max),
            "recode_distortion_ev_drop": float(ev_flat_full - ev_max),
        }

    # ---- sweep thresholds --------------------------------------------------------
    fracs = [float(x) for x in args.frac_sweep.split(",")]
    sweep = []
    headline_groups = None
    for frac in fracs:
        thr = max(2, int(frac * N))
        groups, within_frac, n_kept = build_groups(thr)
        s3 = stage3(groups)
        topo_hist = dict(Counter(g["topology"] for g in groups))
        cyc = [{"gid": gid, "size": g["size"], "topology": g["topology"],
                "barycentric": g["barycentric"], "sign_agree_frac": g["sign_agree_frac"],
                "twohot_energy_frac_mean": g["twohot_energy_frac_mean"],
                "decoder_adjacency_contrast": g["decoder_adjacency_contrast"],
                "atoms": g["atoms"]}
               for gid, g in enumerate(groups)
               if g["topology"] == "cycle" and (6 <= g["size"] <= 8 or 11 <= g["size"] <= 13)]
        rec = {"edge_min_frac": frac, "threshold": thr, "n_edges_kept": n_kept,
               "within_block_edge_frac": within_frac,
               "topology_histogram": topo_hist,
               "size_histogram": dict(sorted(Counter(g["size"] for g in groups).items())),
               "n_barycentric_groups": int(sum(g["barycentric"] for g in groups)),
               "n_cycle_groups": int(sum(1 for g in groups if g["topology"] == "cycle")),
               "calendar_cycle_candidates": cyc, **s3}
        sweep.append(rec)
        print(f"[cs][SWEEP] frac={frac:g} thr={thr} groups={s3['n_groups']} "
              f"cov(firings)={s3['frac_firings_secant']:.3f} "
              f"maxfid_EV={s3['manifold_maxfidelity_ev']:.4f} (flat {ev_flat_full:.4f}) "
              f"within_block={within_frac:.2f} cycles={rec['n_cycle_groups']} "
              f"calendar={len(cyc)}  {time.time()-t0:.0f}s", flush=True)
        if abs(frac - args.headline_frac) < 1e-12:
            headline_groups = groups
    if headline_groups is None:
        headline_groups, _, _ = build_groups(max(2, int(args.headline_frac * N)))

    # ---- outputs -----------------------------------------------------------------
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    rd = {"flat": flat_curve, "sweep": sweep, "flat_active_recon_ev": ev_flat_full,
          "N_tokens": N, "K": K, "active": A, "log2K": log2K}
    # overall calendar hits across the sweep
    calendar = []
    seen = set()
    for rec in sweep:
        for c in rec["calendar_cycle_candidates"]:
            key = tuple(sorted(c["atoms"]))
            if key not in seen:
                seen.add(key); calendar.append({**{k: v for k, v in c.items() if k != "atoms"},
                                                "atoms": c["atoms"], "at_frac": rec["edge_min_frac"]})
    summary = {
        "K": K, "P": P, "active": A, "N_tokens": N, "seed": args.seed,
        "flat_active_recon_ev": ev_flat_full,
        "frac_sweep": fracs, "headline_frac": args.headline_frac,
        "n_calendar_cycle_groups": len(calendar), "calendar": calendar,
        "sweep_brief": [{"frac": r["edge_min_frac"], "n_groups": r["n_groups"],
                         "frac_firings_secant": r["frac_firings_secant"],
                         "manifold_maxfidelity_ev": r["manifold_maxfidelity_ev"],
                         "recode_distortion_ev_drop": r["recode_distortion_ev_drop"],
                         "n_cycle_groups": r["n_cycle_groups"],
                         "within_block_edge_frac": r["within_block_edge_frac"]} for r in sweep],
        "wall_s": time.time() - t0,
        "decoder": args.decoder, "tier0": args.tier0, "data": args.data,
    }
    (outd / "discovered_groups.json").write_text(json.dumps(
        {"groups": headline_groups, "headline_frac": args.headline_frac, "summary": summary}, indent=2))
    (outd / "rate_distortion.json").write_text(json.dumps(rd, indent=2))
    (outd / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[cs] SUMMARY " + json.dumps(summary), flush=True)
    print(f"[cs] wrote {outd}/  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
