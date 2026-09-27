"""#2951 operator-first: how many independent QK operators does a RoPE attention layer hold?

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). numpy + safetensors only,
CPU float64 on the exact bf16 / f32 weights (-> f64 is exact), one layer at a time.

For query head h reading key/value head g(h) = h // (H / KV), with rotary planes j = coords (j, j + hd/2)
(HF rotate_half pairing) and omega_j = theta^(-2j/hd), the pre-softmax score is

    s = sigma * q_h(x)^T R(Delta) k_g(y),   R(Delta) = sum_j cos(omega_j Delta) P_j + sin(omega_j Delta) J_j,

Delta = key position - query position, J_j the quarter turn (a, b) -> (-b, a) on plane j. The architecture is
read from config (bench/mpd_opfirst_decoder_2951.py). The q/k-norm gains, and in a pre-norm model the
input RMSNorm gain, are folded in as diagonal factors: q_h = Q_h x / r_q[h], k_g = K_g y / r_k[g], with x, y
the attention-block inputs (the RMS-normalised residual rows in a pre-norm model such as Qwen3; the raw
residual in OLMo 2, which has no pre-attention norm). The normalisers r are EXCLUDED: per head in Qwen3
(q_norm over each head's hd coordinates), and a single scalar per token shared by every head in OLMo 2
(q_norm / k_norm over the full H * hd projection), so for OLMo 2 the ratio of two heads' scores for one token
pair is exact. Everything below is the bilinear operator M_h(Delta) = sum_j cos(omega_j Delta) A_hj +
sin(omega_j Delta) B_hj with

    A_hj = Q_hj^T K_gj,   B_hj = Q_hj^T J K_gj     (Q_hj, K_gj the 2 x d rows of plane j; rank <= 2),

kept factored as U V^T with two columns. Frobenius inner products in factored form,
<U_a V_a^T, U_b V_b^T> = tr[(U_a^T U_b)(V_b^T V_a)] = sum_{pq} (U^T U)_{ab,pq} (V^T V)_{ab,pq}.
The self-check scores random rows through the literal HF path (raw projections, q/k RMSNorm at its declared
scope with eps, gains, rotate_half) and compares with the expansion divided by r_q r_k.

Reports, per layer: the span dimension of {A_hj, B_hj} (numerical rank of the Gram at a stated threshold,
plus a Weyl certificate when the smallest eigenvalue clears it), the eigenvalue counts reaching 90 / 99 /
99.9 % of the Gram trace against the block-diagonal null "every head (or plane) in its own orthogonal
subspace", per-plane sharing across heads, and cross-head overlap split by shared vs distinct K head.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import Decoder, compact_json  # noqa: E402

EPS = np.finfo(np.float64).eps
LEVELS = (0.9, 0.99, 0.999)


def spectrum_counts(eigs):
    """Numbers of leading eigenvalues reaching each trace fraction, plus participation ratio."""
    e = np.sort(np.clip(eigs, 0.0, None))[::-1]
    c = np.cumsum(e) / e.sum()
    out = {f"k{int(round(l * 1000)) / 10:g}": int(np.searchsorted(c, l) + 1) for l in LEVELS}
    out["participation_ratio"] = float(e.sum() ** 2 / (e**2).sum())
    return out


def rank_report(gram):
    """Numerical rank of a Gram matrix at tau = n * eps * lambda_max (n = order).

    Gram entries are sums of ~4d products of f64 numbers; the computed Gram differs from the exact one by
    O(n eps ||G||) in operator norm, so eigenvalues above tau are resolved (Weyl). When lambda_min > tau
    the span is certified full-dimensional (up to that rounding model); below it, "numerical" only.
    Equivalently the atom stack has singular values resolved down to sqrt(tau)."""
    n = gram.shape[0]
    eigs = np.linalg.eigvalsh(gram)
    tau = n * EPS * eigs[-1]
    return eigs, {
        "max_dim": n,
        "tau": float(tau),
        "numerical_rank": int((eigs > tau).sum()),
        "lambda_max": float(eigs[-1]),
        "lambda_min": float(eigs[0]),
        "cond_lambda": float(eigs[-1] / max(eigs[0], 1e-300)),
        "full_rank_certified": bool(eigs[0] > tau),
    }


def block_null(gram, blocks):
    """Eigenvalues of the block-diagonal part of the Gram (the 'no sharing between blocks' null)."""
    return np.concatenate([np.linalg.eigvalsh(gram[np.ix_(b, b)]) for b in blocks])


def captured(gram, a, b):
    """Fraction of the Frobenius energy of atom set b lying in span(atom set a)."""
    gaa = gram[np.ix_(a, a)]
    w, v = np.linalg.eigh(gaa)
    keep = w > len(a) * EPS * w[-1]
    pinv = (v[:, keep] / w[keep]) @ v[:, keep].T
    gab = gram[np.ix_(a, b)]
    return float(np.trace(gab.T @ pinv @ gab) / np.trace(gram[np.ix_(b, b)]))


def top_cosine(gram, a, b):
    """Largest principal cosine between span(a) and span(b)."""

    def isqrt(g):
        w, v = np.linalg.eigh(g)
        keep = w > len(g) * EPS * w[-1]
        return v[:, keep] / np.sqrt(w[keep])

    ia, ib = isqrt(gram[np.ix_(a, a)]), isqrt(gram[np.ix_(b, b)])
    return float(np.linalg.svd(ia.T @ gram[np.ix_(a, b)] @ ib, compute_uv=False)[0])


def atoms(Q, K):
    """U, V stacks (d x 2N): atom i = (h, j, t) with t = 0 -> A_hj, t = 1 -> B_hj; columns 2i, 2i+1."""
    H, hd, d = Q.shape
    KV, P = K.shape[0], hd // 2
    grp = H // KV
    U = np.empty((H, P, 2, d, 2))
    V = np.empty((H, P, 2, d, 2))
    for h in range(H):
        g = h // grp
        for j in range(P):
            q = Q[h, [j, j + P]]  # 2 x d
            k = K[g, [j, j + P]]
            jk = np.stack([-k[1], k[0]])  # J k: (a, b) -> (-b, a)
            U[h, j, 0] = U[h, j, 1] = q.T
            V[h, j, 0] = k.T
            V[h, j, 1] = jk.T
    N = H * P * 2
    U = U.transpose(3, 0, 1, 2, 4).reshape(d, 2 * N)
    V = V.transpose(3, 0, 1, 2, 4).reshape(d, 2 * N)
    return U, V


def gram_from_factors(U, V):
    N = U.shape[1] // 2
    return ((U.T @ U) * (V.T @ V)).reshape(N, 2, N, 2).sum(axis=(1, 3))


def selfcheck(D, layer, Q, K, fold, rng):
    """Literal HF scoring (raw q/k projections, q/k RMSNorm at the declared scope with eps, gains,
    rotate_half rope) vs (sum_j cos A_hj + sin B_hj) / (r_q[h] r_k[g]) from the factored atoms, on random
    attention-block inputs at query position m, key position n."""
    H, hd, d = Q.shape
    P, grp = hd // 2, H // K.shape[0]
    omega = D.theta ** (-np.arange(P) * 2.0 / hd)
    eps = D.config["rms_norm_eps"]
    pre = f"model.layers.{layer}.self_attn."
    gamma = D.attn_input_gain(layer) if fold else None
    U, V = atoms(Q, K)
    x, y = rng.standard_normal(d), rng.standard_normal(d)
    xin, yin = (x, y) if gamma is None else (gamma * x, gamma * y)
    qr = (D(pre + "q_proj.weight") @ xin).reshape(H, hd)
    kr = (D(pre + "k_proj.weight") @ yin).reshape(-1, hd)
    rq, rk = D.qk_norm_rms(qr, H, eps), D.qk_norm_rms(kr, kr.shape[0], eps)
    if D.qk_norm is not None:
        qr = qr * D(pre + "q_norm.weight").reshape(-1, hd)
        kr = kr * D(pre + "k_norm.weight").reshape(-1, hd)
    q, k = qr / rq[:, None], kr / rk[:, None]
    xu = (x @ U).reshape(H, P, 2, 2)
    yv = (y @ V).reshape(H, P, 2, 2)
    bil = (xu * yv).sum(-1)  # x^T A_hj y, x^T B_hj y

    def rope(v, pos):
        c, s = np.cos(omega * pos), np.sin(omega * pos)
        return np.concatenate([v[:P] * c - v[P:] * s, v[P:] * c + v[:P] * s])

    m, n = 17, 1234
    err = 0.0
    for h in range(H):
        g = h // grp
        direct = rope(q[h], m) @ rope(k[g], n)
        expand = (np.cos(omega * (n - m)) * bil[h, :, 0] + np.sin(omega * (n - m)) * bil[h, :, 1]).sum()
        err = max(err, abs(direct - expand / (rq[h] * rk[g])) / np.abs(q[h]).sum() / np.abs(k[g]).max())
    return err


def analyse_layer(Q, K, layer, with_pairs, theta):
    H, hd, d = Q.shape
    KV, P = K.shape[0], hd // 2
    grp = H // KV
    U, V = atoms(Q, K)
    G = gram_from_factors(U, V)
    N = G.shape[0]
    idx = np.arange(N).reshape(H, P, 2)

    eigs, rk = rank_report(G)
    rec = {"layer": layer, "rank": rk, "actual": spectrum_counts(eigs)}
    rec["null_heads_independent"] = spectrum_counts(block_null(G, [idx[h].ravel() for h in range(H)]))
    rec["null_planes_independent"] = spectrum_counts(block_null(G, [idx[:, j].ravel() for j in range(P)]))
    rec["null_groups_independent"] = spectrum_counts(
        block_null(G, [idx[g * grp:(g + 1) * grp].ravel() for g in range(KV)]))
    rec["null_all_atoms_orthogonal"] = spectrum_counts(np.diag(G).copy())
    rec["per_head_k90"] = [spectrum_counts(np.linalg.eigvalsh(G[np.ix_(idx[h].ravel(), idx[h].ravel())]))["k90"]
                           for h in range(H)]

    # per-plane sharing across heads: span of {A_hj, B_hj : h}, 2H atoms
    diag = np.sqrt(np.diag(G))
    C = G / np.outer(diag, diag)
    planes = []
    for j in range(P):
        b = idx[:, j].ravel()
        e, r = rank_report(G[np.ix_(b, b)])
        ec = np.linalg.eigvalsh(C[np.ix_(b, b)])
        same, diff = [], []
        for h in range(H):
            for h2 in range(h + 1, H):
                blk = np.abs(C[np.ix_(idx[h, j], idx[h2, j])]).max()
                (same if h // grp == h2 // grp else diff).append(blk)
        planes.append({
            "j": j,
            "wavelength": float(2 * np.pi / (theta ** (-2.0 * j / hd))),
            "energy": float(np.trace(G[np.ix_(b, b)])),
            "numerical_rank": r["numerical_rank"],
            "counts": spectrum_counts(e),
            "cos_pr": spectrum_counts(ec)["participation_ratio"],  # of the cosine Gram: max 2H
            "max_abs_cos_same_k": float(np.mean(same)) if same else None,
            "max_abs_cos_diff_k": float(np.mean(diff)),
        })
    rec["planes"] = planes

    if with_pairs:
        same, diff = {"captured": [], "top_cos": []}, {"captured": [], "top_cos": []}
        for h in range(H):
            for h2 in range(h + 1, H):
                a, b = idx[h].ravel(), idx[h2].ravel()
                cap = 0.5 * (captured(G, a, b) + captured(G, b, a))
                tc = top_cosine(G, a, b)
                tgt = same if h // grp == h2 // grp else diff
                tgt["captured"].append(cap)
                tgt["top_cos"].append(tc)
        rec["head_pairs"] = {  # "same_k" (GQA siblings) is absent when every head has its own K head
            key: {k: float(np.mean(v)) for k, v in grp_.items()} | {"n": len(grp_["captured"]),
                                                                     "captured_max": float(np.max(grp_["captured"]))}
            for key, grp_ in (("same_k", same), ("diff_k", diff)) if grp_["captured"]}
    return rec



def random_null(H, KV, hd, d, rng):
    """Same shapes, iid Gaussian weights: what 'no structure' looks like for the head-pair metrics."""
    Q = rng.standard_normal((H, hd, d))
    K = rng.standard_normal((KV, hd, d))
    U, V = atoms(Q, K)
    G = gram_from_factors(U, V)
    idx = np.arange(G.shape[0]).reshape(H, hd // 2, 2)
    grp = H // KV
    out = {"actual": spectrum_counts(np.linalg.eigvalsh(G)),
           "null_heads_independent": spectrum_counts(block_null(G, [idx[h].ravel() for h in range(H)])),
           "captured_diff_k": captured(G, idx[0].ravel(), idx[grp].ravel()),
           "top_cos_diff_k": top_cosine(G, idx[0].ravel(), idx[grp].ravel())}
    if grp > 1:
        out["captured_same_k"] = captured(G, idx[0].ravel(), idx[1].ravel())
        out["top_cos_same_k"] = top_cosine(G, idx[0].ravel(), idx[1].ravel())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fold-input-norm", action="store_true")
    ap.add_argument("--pairs-every", type=int, default=1, help="head-pair analysis every k-th layer")
    args = ap.parse_args()
    t0 = time.time()
    D = Decoder(args.model, args.revision)
    rng = np.random.default_rng(2951)
    fold = not args.no_fold_input_norm

    layers, check = [], 0.0
    for layer in range(D.L):
        tl = time.time()
        Q, K = D.qk(layer, fold)
        check = max(check, selfcheck(D, layer, Q, K, fold, rng))
        rec = analyse_layer(Q, K, layer, layer % args.pairs_every == 0, D.theta)
        rec["seconds"] = time.time() - tl
        layers.append(rec)
        a, nh = rec["actual"], rec["null_heads_independent"]
        hp = rec.get("head_pairs", {})
        print(f"L{layer:2d} rank {rec['rank']['numerical_rank']}/{rec['rank']['max_dim']} "
              f"cert={rec['rank']['full_rank_certified']} k90/99/99.9 {a['k90']}/{a['k99']}/{a['k99.9']} "
              f"(heads-indep null {nh['k90']}/{nh['k99']}/{nh['k99.9']}) PR {a['participation_ratio']:.1f}/"
              f"{nh['participation_ratio']:.1f} pairs same/diff="
              f"{hp.get('same_k', {}).get('captured', float('nan')):.3f}/"
              f"{hp.get('diff_k', {}).get('captured', float('nan')):.3f} {rec['seconds']:.1f}s", flush=True)

    excluded = {"head": "per-head RMS normalisers of q_norm/k_norm",
                "full": "ONE full-projection RMS normaliser per token for q and for k, shared by all heads",
                None: "none (no q/k norm)"}[D.qk_norm]
    if D.where == "pre":
        excluded += "; the residual RMS of the input RMSNorm"
    out = {
        "model": args.model,
        "architecture": D.describe(),
        "config": {"H": D.H, "KV": D.KV, "head_dim": D.hd, "d_model": D.d, "layers": D.L, "rope_theta": D.theta},
        "fold": {"q_norm_k_norm_gain": D.qk_norm, "input_layernorm_gain": fold and D.where == "pre",
                 "excluded": excluded},
        "rank_threshold": "tau = n * eps_f64 * lambda_max(Gram); full rank certified iff lambda_min > tau",
        "operator_expansion_selfcheck_max_rel_err": check,
        "random_gaussian_null": random_null(D.H, D.KV, D.hd, D.d, rng),
        "layers": layers,
        "seconds_total": time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(compact_json(out))
    print(f"total {out['seconds_total']:.1f}s  selfcheck {check:.2e}")


if __name__ == "__main__":
    main()
