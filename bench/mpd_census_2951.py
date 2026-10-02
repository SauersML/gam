"""Shared-structure census (#2951): how many bits do the target's site matrices save, beyond coding
each matrix on its own, when structure shared across neurons, heads, matrices and layers is used?

Every code below is a two-part code with the SAME guarantee as dense rounding at b = 4 (the S1 row
'dense target b=4', 2.29e8 bits on VPD 4L): each matrix W_i is reconstructed to within its own lattice
step delta_i = 2^-p_i, p_i = 4 - round(log2 rms W_i), because the decoder adds the residual
Q_delta_i(W_i - prediction) to a prediction built from already-decoded reals. So
    bits = L(structure reals, own lattice) + L(indices) + sum_i L(residual_i on delta_i)
and 'saved' is the drop against the best code that treats each matrix alone (per-matrix SVD + residual,
rank and factor precision chosen by total code length; rank 0 = dense rounding). The coder is gam's
LatticeCode (vpd_bits.lattice_bits: Elias-omega header, signed Elias-delta indices), table-driven here.

Families:
  gauge     exact symmetries of the forward: neuron and head permutations (log2 n! by canonical order),
            the continuous orbit (OV: GL(hd) per head; QK: one complex scalar per RoPE plane, GL of the
            non-rotary dims), which is an upper bound (orbit reals x their bits/real); and heads equal up to
            that gauge (gauge-invariant distances, then the realized code of predicting each head from its
            nearest head through a fitted gauge)
  subspace  one basis shared by a group of matrices (joint SVD of the tolerance-scaled reads stacked
            vertically, or writes side by side), per-matrix coefficients, residual
  clusters  MLP neurons (c_fc row + down column) as scaled copies of K centroids (k-means on directions)

usage: mpd_census_2951.py {vpd4l|pythia70m} OUT.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from vpd_bits import delta_len, lattice_bits, omega_len, signed_len  # noqa: E402

MODEL, OUT = sys.argv[1:3]
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
B_REF = 4
BGRID = [3, 4, 5, 6, 7, 8, 9]

# ------------------------------------------------------------------ weights

if MODEL == "vpd4l":
    from safetensors.torch import load_file
    from vpd_model import TARGET_DIR, site_names
    sd = load_file(str(TARGET_DIR / "model_step_99999.safetensors"))
    names = site_names()
    W = {n: sd[f"{n}.weight"].double().numpy() for n in names}
    L, H, HD, RD = 4, 6, 128, 128
else:
    from mpd_pythia_model_2951 import KINDS, load_pythia, site_name
    m = load_pythia("cpu")
    names = [site_name(i, k) for i in range(m.n_layer) for k in KINDS]
    W = {n: m.site(n).W.double().numpy() for n in names}
    L, H, HD, RD = m.n_layer, m.n_head, m.hd, m.rd
    del m
layer = lambda n: int(n.split(".")[1])
kind = lambda n: n.split(".")[-1]
READS = [n for n in names if kind(n) in ("q_proj", "k_proj", "v_proj", "c_fc")]
WRITES = [n for n in names if kind(n) in ("o_proj", "down_proj")]

# ------------------------------------------------------------------ the coder, table-driven


TAB_N = 1 << 20
DELTA_TAB = delta_len(np.arange(1, TAB_N + 1)).astype(np.int64)  # DELTA_TAB[z] = delta(z + 1)


def prec(x, b):
    r = math.sqrt(float(np.mean(np.square(x))))
    return b - round(math.log2(r)) if r > 0 else b


def q(x, p):
    return np.round(x * 2.0**p) * 2.0**-p


def lb(x, p):
    """Exact LatticeCode length of x at precision p (vpd_bits.lattice_bits, by table)."""
    k = np.round(np.asarray(x) * 2.0**p).astype(np.int64).ravel()
    z = np.where(k >= 0, 2 * k, -2 * k - 1)
    small = z < TAB_N
    bits = int(DELTA_TAB[z[small]].sum())
    if not small.all():
        bits += int(delta_len(z[~small].astype(np.uint64) + np.uint64(1)).sum())
    return bits + int(omega_len(np.array([k.size + 1]))[0]) + int(signed_len(np.array([p]))[0])


def lbq(x, b):
    """Code x on its own b-bit lattice: (bits, decoded x)."""
    p = prec(x, b)
    return lb(x, p), q(x, p)


omega = lambda n: int(omega_len(np.array([n]))[0])
P = {n: prec(W[n], B_REF) for n in names}
DENSE = {n: lb(W[n], P[n]) for n in names}
_chk = names[0]
assert DENSE[_chk] == lattice_bits(torch.from_numpy(W[_chk]), P[_chk]), "table coder disagrees with vpd_bits"
TOTAL = sum(DENSE.values())
log(f"dense b={B_REF}: {TOTAL} bits over {sum(w.size for w in W.values())} reals")
res = {"model": MODEL, "b_ref": B_REF, "dense_bits": TOTAL, "dense_per_site": DENSE}

# ------------------------------------------------------------------ per-matrix baseline: SVD + residual

SVD = {n: sl.svd(W[n], full_matrices=False, lapack_driver="gesvd") for n in names}
RANKS = [0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024]


def lowrank_code(n):
    Pm, s, Qt = SVD[n]
    best = (DENSE[n], 0, None)
    for k in RANKS:
        if k == 0 or k > len(s):
            continue
        Lf, Rf = Pm[:, :k] * np.sqrt(s[:k]), np.sqrt(s[:k])[:, None] * Qt[:k]
        for b in BGRID:
            bl, Lq = lbq(Lf, b)
            br, Rq = lbq(Rf, b)
            tot = omega(k + 1) + bl + br + lb(W[n] - Lq @ Rq, P[n])
            if tot < best[0]:
                best = (tot, k, b)
    return best


prev = json.load(open(OUT)) if Path(OUT).exists() else {}
if "per_matrix" in prev:  # resume: the per-matrix codes are deterministic
    PER = {n: (v["bits"], v["rank"], v["b"]) for n, v in prev["per_matrix"]["sites"].items()}
else:
    PER = {}
    for n in names:
        PER[n] = lowrank_code(n)
        log(f"per-matrix {n}: dense {DENSE[n]} -> {PER[n][0]} (rank {PER[n][1]}, b {PER[n][2]})")
PER_TOTAL = sum(v[0] + 1 for v in PER.values())  # +1: a flag per matrix, dense or factored
res["per_matrix"] = {"bits": PER_TOTAL, "saved_vs_dense": TOTAL - PER_TOTAL,
                     "sites": {n: {"bits": v[0], "rank": v[1], "b": v[2]} for n, v in PER.items()}}

# ------------------------------------------------------------------ shared subspaces


def shared_code(group, side):
    """One basis shared by the group: reads share the input side (rows stacked), writes the output side.
    Each matrix is scaled by 1/delta_i in the joint SVD (error in units of its own tolerance)."""
    sc = {n: 2.0 ** P[n] for n in group}
    if side == "in":
        M = np.vstack([W[n] * sc[n] for n in group])
        _, s, Bt = sl.svd(M, full_matrices=False, lapack_driver="gesvd")
        basis = Bt.T  # [d, r]
    else:
        M = np.hstack([W[n] * sc[n] for n in group])
        basis, s, _ = sl.svd(M, full_matrices=False, lapack_driver="gesvd")
    base = sum(PER[n][0] + 1 for n in group)
    best = (base, 0, None)
    for k in RANKS:
        if k == 0 or k > len(s):
            continue
        for bq in BGRID:
            bQ, Qq = lbq(basis[:, :k], bq)
            G = np.linalg.pinv(Qq) if side == "in" else np.linalg.pinv(Qq)
            for bc in (bq - 1, bq, bq + 1):
                tot = omega(k + 1) + bQ
                for n in group:
                    C = W[n] @ G.T if side == "in" else G @ W[n]
                    bC, Cq = lbq(C, bc)
                    pred = Cq @ Qq.T if side == "in" else Qq @ Cq
                    tot += bC + lb(W[n] - pred, P[n])
                    if tot >= best[0]:
                        break
                if tot < best[0]:
                    best = (tot, k, (bq, bc))
    return {"sites": group, "side": side, "per_matrix_bits": base, "shared_bits": best[0], "rank": best[1],
            "b": best[2], "saved": base - best[0], "sv_top": [float(x) for x in s[:8]]}


groups = [("reads, all layers", READS, "in"), ("writes, all layers", WRITES, "out")]
for li in range(L):
    groups.append((f"reads, layer {li}", [n for n in READS if layer(n) == li], "in"))
    groups.append((f"writes, layer {li}", [n for n in WRITES if layer(n) == li], "out"))
for kd in ("q_proj", "k_proj", "v_proj", "c_fc"):
    groups.append((f"{kd} across layers", [n for n in READS if kind(n) == kd], "in"))
for kd in ("o_proj", "down_proj"):
    groups.append((f"{kd} across layers", [n for n in WRITES if kind(n) == kd], "out"))
res["subspace"] = prev.get("subspace", {})
for name, g, side in groups:
    if name in res["subspace"]:
        continue
    r = shared_code(g, side)
    res["subspace"][name] = r
    log(f"shared {side} basis, {name}: per-matrix {r['per_matrix_bits']} -> {r['shared_bits']} (rank {r['rank']}) saved {r['saved']}")
    json.dump(res, open(OUT, "w"), indent=1)

# ------------------------------------------------------------------ neuron clusters


def cluster_code(mlps, K, seed=0):
    """Neurons j of the MLPs as (a_j mu_in[c_j], b_j mu_out[c_j]); centroids, fixed-length indices,
    two scales per neuron, residuals."""
    fin = np.vstack([W[f] for f, _ in mlps])  # [N, d] c_fc rows
    fout = np.vstack([W[d].T for _, d in mlps])  # [N, d] down columns
    din = np.repeat([2.0 ** P[f] for f, _ in mlps], [W[f].shape[0] for f, _ in mlps])[:, None]
    dout = np.repeat([2.0 ** P[d] for _, d in mlps], [W[d].shape[1] for _, d in mlps])[:, None]
    z = np.hstack([fin * din, fout * dout])
    u = z / np.linalg.norm(z, axis=1, keepdims=True)
    cent, lab = kmeans2(u, K, iter=25, minit="++", seed=seed)
    base = sum(PER[f][0] + PER[d][0] + 2 for f, d in mlps)
    best = (base, None)
    d0 = fin.shape[1]
    for b in BGRID:
        mu_in, mu_out = cent[:, :d0] / din.mean(), cent[:, d0:] / dout.mean()
        bmi, mi = lbq(mu_in, b)
        bmo, mo = lbq(mu_out, b)
        A, Bs = mi[lab], mo[lab]
        a = (fin * A).sum(1) / np.maximum((A * A).sum(1), 1e-30)
        bb = (fout * Bs).sum(1) / np.maximum((Bs * Bs).sum(1), 1e-30)
        for bs in BGRID:
            ba, aq = lbq(a, bs)
            bbb, bq_ = lbq(bb, bs)
            tot = omega(K + 1) + bmi + bmo + ba + bbb + len(lab) * math.ceil(math.log2(K))
            r_in, r_out = fin - aq[:, None] * A, fout - bq_[:, None] * Bs
            o = 0
            for f, d in mlps:
                nf = W[f].shape[0]
                tot += lb(r_in[o:o + nf], P[f]) + lb(r_out[o:o + nf].T, P[d])
                o += nf
            if tot < best[0]:
                best = (tot, (b, bs))
    return {"K": K, "per_matrix_bits": base, "cluster_bits": best[0], "b": best[1], "saved": base - best[0]}


mlps = [(f"h.{i}.mlp.c_fc", f"h.{i}.mlp.down_proj") for i in range(L)]
res["clusters"] = {}
for scope, ms in [(f"layer {i}", [mlps[i]]) for i in range(L)] + [("all layers", mlps)]:
    rows = []
    for K in (4, 16, 64, 256):  # K = 1024 on VPD 4L layer 0 also saved 0 bits (39 min of k-means)
        r = cluster_code(ms, K)
        rows.append(r)
        log(f"clusters {scope} K={K}: per-matrix {r['per_matrix_bits']} -> {r['cluster_bits']} saved {r['saved']}")
    res["clusters"][scope] = rows
    json.dump(res, open(OUT, "w"), indent=1)

# ------------------------------------------------------------------ gauge


def lgfact2(n):
    return math.lgamma(n + 1) / math.log(2)


n_ff = W[mlps[0][0]].shape[0]
perm_bits = L * lgfact2(n_ff) + L * lgfact2(H)
bits_per_real = {n: DENSE[n] / W[n].size for n in names}
orbit_ov = L * H * HD * HD
nr = HD - RD
orbit_qk = L * H * ((RD // 2) * 2 + nr * nr)
bpr = lambda kinds: np.mean([bits_per_real[n] for n in names if kind(n) in kinds])
gauge = {"permutation_bits": perm_bits,
         "orbit_reals_ov": orbit_ov, "orbit_reals_qk": orbit_qk,
         "orbit_bound_bits": orbit_ov * bpr(("v_proj", "o_proj")) + orbit_qk * bpr(("q_proj", "k_proj"))}


def heads(n_l, kd):
    Wm = W[f"h.{n_l}.attn.{kd}"]
    return [Wm[h * HD:(h + 1) * HD] if kd != "o_proj" else Wm[:, h * HD:(h + 1) * HD] for h in range(H)]


allh = [(li, h) for li in range(L) for h in range(H)]
Q = {(li, h): heads(li, "q_proj")[h] for li, h in allh}
K_ = {(li, h): heads(li, "k_proj")[h] for li, h in allh}
V = {(li, h): heads(li, "v_proj")[h] for li, h in allh}
O = {(li, h): heads(li, "o_proj")[h] for li, h in allh}
half = RD // 2


def cplx(M):  # rotate-half planes (j, j + RD/2) as complex rows
    return M[:half] + 1j * M[half:RD]


def qk_inner(a, b):
    """<QK_a, QK_b> over the gauge-invariant forms: per-plane complex q k^H, plus the non-rotary q^T k."""
    qa, ka, qb, kb = cplx(Q[a]), cplx(K_[a]), cplx(Q[b]), cplx(K_[b])
    rot = np.real(np.sum((qa.conj() * qb).sum(1) * (kb.conj() * ka).sum(1)))
    nrt = np.trace((Q[a][RD:] @ Q[b][RD:].T) @ (K_[b][RD:] @ K_[a][RD:].T)) if nr else 0.0
    return rot + nrt


def ov_inner(a, b):
    return np.trace((O[a].T @ O[b]) @ (V[b] @ V[a].T))


dist = {"qk": {}, "ov": {}}
for nm, f in (("qk", qk_inner), ("ov", ov_inner)):
    nn = {a: f(a, a) for a in allh}
    for a in allh:
        dist[nm][a] = min(((nn[a] + nn[b] - 2 * f(a, b)) / nn[a], b) for b in allh if b != a)
gauge["head_nearest_rel_dist_qk_min"] = float(min(v[0] for v in dist["qk"].values()))
gauge["head_nearest_rel_dist_ov_min"] = float(min(v[0] for v in dist["ov"].values()))


def head_predict_bits(a, src):
    """Realized two-part code of head a's (q, k, v, o) predicted from head src through a fitted gauge:
    OV: G = V_a V_src^+ (coded), V ~ G V_src, O ~ O_src G^-1; QK: per plane z = <q_src, q_a>/|q_src|^2
    (coded, 2 reals), q ~ z q_src, k ~ k_src / conj(z), non-rotary block by least squares likewise."""
    la, ls = a[0], src[0]
    pq, pk, pv, po = (P[f"h.{la}.attn.{k}"] for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
    dense = sum(lb(M, p) for M, p in ((Q[a], pq), (K_[a], pk), (V[a], pv), (O[a], po)))
    best = dense
    for b in BGRID:
        G = V[a] @ np.linalg.pinv(V[src])
        bG, Gq = lbq(G, b)
        try:
            Gi = np.linalg.inv(Gq)
        except np.linalg.LinAlgError:
            continue
        ov = bG + lb(V[a] - Gq @ V[src], pv) + lb(O[a] - O[src] @ Gi, po)
        qs, qa_, ks = cplx(Q[src]), cplx(Q[a]), cplx(K_[src])
        zc = (qs.conj() * qa_).sum(1) / np.maximum((np.abs(qs) ** 2).sum(1), 1e-30)
        bz, zr = lbq(np.concatenate([zc.real, zc.imag]), b)
        zq = zr[:half] + 1j * zr[half:]
        qp, kp = zq[:, None] * qs, ks / np.conj(np.where(np.abs(zq) > 0, zq, 1))[:, None]
        Qp, Kp = np.vstack([qp.real, qp.imag]), np.vstack([kp.real, kp.imag])
        qk = bz
        if nr:
            Gn = Q[a][RD:] @ np.linalg.pinv(Q[src][RD:])
            bn, Gnq = lbq(Gn, b)
            Gni = np.linalg.inv(Gnq)
            Qp, Kp = np.vstack([Qp, Gnq @ Q[src][RD:]]), np.vstack([Kp, Gni.T @ K_[src][RD:]])
            qk += bn
        qk += lb(Q[a] - Qp, pq) + lb(K_[a] - Kp, pk)
        best = min(best, ov + qk + math.ceil(math.log2(len(allh))))
    return dense, best


saved_heads = 0
for ia, a in enumerate(allh[1:], 1):  # each head predicted from its nearest EARLIER head (a DAG)
    nn_ov = {b: ov_inner(b, b) for b in allh[:ia]}
    na = ov_inner(a, a)
    src = min(allh[:ia], key=lambda b: (na + nn_ov[b] - 2 * ov_inner(a, b)) / na)
    d_, b_ = head_predict_bits(a, src)
    saved_heads += max(0, d_ - b_)
gauge["heads_up_to_gauge_saved_bits"] = saved_heads
# neuron near-duplicates (permutation is the only exact MLP gauge under GELU)
dup = {}
for f, d in mlps:
    z = np.hstack([W[f], W[d].T])
    u = z / np.linalg.norm(z, axis=1, keepdims=True)
    c = u @ u.T
    np.fill_diagonal(c, 0)
    dup[f] = {"max_cos": float(np.abs(c).max()), "pairs_cos_gt_0.9": int((np.abs(np.triu(c)) > 0.9).sum())}
gauge["neuron_duplicates"] = dup
res["gauge"] = gauge
log(f"gauge: {json.dumps(gauge)}")
json.dump(res, open(OUT, "w"), indent=1)
log("done")
