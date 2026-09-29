"""#2951 operator-first: how many independent QK operators does a RoPE attention layer hold?

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). numpy + safetensors only,
CPU float64 on the exact bf16 / f32 weights (-> f64 is exact), one layer at a time. Every eigen/singular
decomposition goes through bench/mpd_opfirst_linalg_2951.py (faer).

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

kept factored as U V^T with two columns. The factors and their Frobenius Gram
<U_a V_a^T, U_b V_b^T> = tr[(U_a^T U_b)(V_b^T V_a)] come from parameter_decomposition::joint_operators
(query_key_operators, family_gram) through the MPD surface (op ``joint_operators``), on the folded Q/K rows.
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
import mpd_opfirst_linalg_2951 as la  # noqa: E402
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
    eigs = la.eigvalsh(gram)
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
    return np.concatenate([la.eigvalsh(gram[np.ix_(b, b)]) for b in blocks])


def captured(gram, a, b):
    """Fraction of the Frobenius energy of atom set b lying in span(atom set a)."""
    gaa = gram[np.ix_(a, a)]
    w, v = la.eigh(gaa)
    keep = w > len(a) * EPS * w[-1]
    pinv = (v[:, keep] / w[keep]) @ v[:, keep].T
    gab = gram[np.ix_(a, b)]
    return float(np.trace(gab.T @ pinv @ gab) / np.trace(gram[np.ix_(b, b)]))


def top_cosine(gram, a, b):
    """Largest principal cosine between span(a) and span(b)."""

    def isqrt(g):
        w, v = la.eigh(g)
        keep = w > len(g) * EPS * w[-1]
        return v[:, keep] / np.sqrt(w[keep])

    ia, ib = isqrt(gram[np.ix_(a, a)]), isqrt(gram[np.ix_(b, b)])
    return float(la.svdvals(ia.T @ gram[np.ix_(a, b)] @ ib)[0])


def qk_gram(Q, K, inverse_frequencies):
    """joint_operators::query_key_operators of the folded rows Q (H, hd, d) and K (KV, hd, d) under half-split
    rotary planes, and family_gram over every atom (h, j, t), t = 0 -> A_hj, t = 1 -> B_hj, through the MPD
    surface. Returns the factor stacks U, V (d x 2N; atom i in columns 2i, 2i+1), the Gram (N x N) and its
    entrywise rounding band."""
    from gamfit.sae import run_parameter_decomposition

    H, hd, d = Q.shape
    KV, P = K.shape[0], len(inverse_frequencies)
    atoms_list = [{"kind": kind, "head": h, "plane": j} for h in range(H) for j in range(P) for kind in ("cosine", "sine")]
    attention = {"geometry": {"model_dim": d, "n_heads": H, "n_kv_heads": KV, "head_dim": hd},
                 "rotary": {"pairing": "half_split", "inverse_frequencies": [float(w) for w in inverse_frequencies],
                            "attention_scaling": 1.0},
                 "score_scale": 1.0, "query": {"weight": "q", "bias": None}, "key": {"weight": "k", "bias": None},
                 "value": {"weight": "v", "bias": None}, "output": {"weight": "o", "bias": None},
                 "query_key_norm": None}
    # The gains are already folded into Q and K; the value/output rows are not read by the QK operators.
    out = run_parameter_decomposition(
        {"schema": "gam.mpd-request", "schema_version": 1,
         "operation": {"kind": "joint_operators", "attention": attention, "input_gain": None, "context_length": 1.0,
                       "grams": [atoms_list], "comparisons": []}},
        {"q": Q.reshape(H * hd, d), "k": K.reshape(KV * hd, d), "v": np.zeros((KV * hd, d)),
         "o": np.zeros((d, H * hd))})
    report = out.report["result"]
    U = np.empty((d, 2 * len(atoms_list)))
    V = np.empty_like(U)
    for i, atom in enumerate(atoms_list):
        factors = report["query_key"][atom["head"]][atom["kind"]][atom["plane"]]
        U[:, 2 * i:2 * i + 2] = out.arrays[factors["left"]]
        V[:, 2 * i:2 * i + 2] = out.arrays[factors["right"]]
    gram = report["grams"][0]
    return U, V, out.arrays[gram["gram"]], out.arrays[gram["band"]]


def inverse_frequencies(theta, hd):
    return theta ** (-np.arange(hd // 2) * 2.0 / hd)


def selfcheck(D, layer, Q, K, U, V, fold, rng):
    """Literal HF scoring (raw q/k projections, q/k RMSNorm at the declared scope with eps, gains,
    rotate_half rope) vs (sum_j cos A_hj + sin B_hj) / (r_q[h] r_k[g]) from the factored atoms, on random
    attention-block inputs at query position m, key position n."""
    H, hd, d = Q.shape
    P, grp = hd // 2, H // K.shape[0]
    omega = D.theta ** (-np.arange(P) * 2.0 / hd)
    eps = D.config["rms_norm_eps"]
    pre = f"model.layers.{layer}.self_attn."
    gamma = D.attn_input_gain(layer) if fold else None
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


def analyse_layer(Q, K, G, band, layer, with_pairs, theta):
    H, hd, d = Q.shape
    KV, P = K.shape[0], hd // 2
    grp = H // KV
    N = G.shape[0]
    idx = np.arange(N).reshape(H, P, 2)

    eigs, rk = rank_report(G)
    rec = {"layer": layer, "rank": rk, "actual": spectrum_counts(eigs), "gram_band_max": float(band.max())}
    rec["null_heads_independent"] = spectrum_counts(block_null(G, [idx[h].ravel() for h in range(H)]))
    rec["null_planes_independent"] = spectrum_counts(block_null(G, [idx[:, j].ravel() for j in range(P)]))
    rec["null_groups_independent"] = spectrum_counts(
        block_null(G, [idx[g * grp:(g + 1) * grp].ravel() for g in range(KV)]))
    rec["null_all_atoms_orthogonal"] = spectrum_counts(np.diag(G).copy())
    rec["per_head_k90"] = [spectrum_counts(la.eigvalsh(G[np.ix_(idx[h].ravel(), idx[h].ravel())]))["k90"]
                           for h in range(H)]

    # per-plane sharing across heads: span of {A_hj, B_hj : h}, 2H atoms
    diag = np.sqrt(np.diag(G))
    C = G / np.outer(diag, diag)
    planes = []
    for j in range(P):
        b = idx[:, j].ravel()
        e, r = rank_report(G[np.ix_(b, b)])
        ec = la.eigvalsh(C[np.ix_(b, b)])
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
    G = qk_gram(Q, K, inverse_frequencies(10000.0, hd))[2]
    idx = np.arange(G.shape[0]).reshape(H, hd // 2, 2)
    grp = H // KV
    out = {"actual": spectrum_counts(la.eigvalsh(G)),
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
        U, V, G, band = qk_gram(Q, K, inverse_frequencies(D.theta, D.hd))
        check = max(check, selfcheck(D, layer, Q, K, U, V, fold, rng))
        rec = analyse_layer(Q, K, G, band, layer, layer % args.pairs_every == 0, D.theta)
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
        "env": la.env_record(),
        "seconds_total": time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(compact_json(out))
    print(f"total {out['seconds_total']:.1f}s  selfcheck {check:.2e}")


if __name__ == "__main__":
    main()
