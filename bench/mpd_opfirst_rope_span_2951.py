"""#2951 operator-first: how many independent QK operators does a RoPE attention layer hold?

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). numpy + safetensors only,
CPU float64 on the exact bf16 weights (bf16 -> f64 is exact).

For query head h reading key/value head g(h) = h // (H / KV), with rotary planes j = coords (j, j + hd/2)
(Qwen3's half-split pairing) and omega_j = theta^(-2j/hd), the pre-softmax score is

    s = sigma * q_h(x)^T R(Delta) k_g(y),   R(Delta) = sum_j cos(omega_j Delta) P_j + sin(omega_j Delta) J_j,

Delta = key position - query position, J_j the quarter turn (a, b) -> (-b, a) on plane j. With the per-head
q_norm / k_norm gains and the input_layernorm gain folded in as diagonal factors, q_h = Q_h x_hat / r_q and
k_g = K_g y_hat / r_k, where x_hat, y_hat are the RMS-normalised residual rows and r_q, r_k the per-head RMS
normaliser scalars. Those two scalars are EXCLUDED: everything below is the bilinear operator on the
normalised vectors, M_h(Delta) = sum_j cos(omega_j Delta) A_hj + sin(omega_j Delta) B_hj with

    A_hj = Q_hj^T K_gj,   B_hj = Q_hj^T J K_gj     (Q_hj, K_gj the 2 x d rows of plane j; rank <= 2),

kept factored as U V^T with two columns. Frobenius inner products in factored form,
<U_a V_a^T, U_b V_b^T> = tr[(U_a^T U_b)(V_b^T V_a)] = sum_{pq} (U^T U)_{ab,pq} (V^T V)_{ab,pq}.

Reports, per layer: the span dimension of {A_hj, B_hj} (numerical rank of the Gram at a stated threshold,
plus a Weyl certificate when the smallest eigenvalue clears it), the eigenvalue counts reaching 90 / 99 /
99.9 % of the Gram trace against the block-diagonal null "every head (or plane) in its own orthogonal
subspace", per-plane sharing across heads, and cross-head overlap split by shared vs distinct K head.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import time

import numpy as np

EPS = np.finfo(np.float64).eps
LEVELS = (0.9, 0.99, 0.999)


class Weights:
    """Minimal safetensors reader (bf16 / f32) returning float64 arrays, all shards of a snapshot."""

    def __init__(self, snapshot):
        self.index = {}
        for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            for name, meta in header.items():
                if name != "__metadata__":
                    self.index[name] = (path, 8 + n, meta)
        self.config = json.load(open(os.path.join(snapshot, "config.json")))

    def __call__(self, name):
        path, base, meta = self.index[name]
        lo, hi = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + lo)
            raw = f.read(hi - lo)
        if meta["dtype"] == "BF16":
            a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] == "F32":
            a = np.frombuffer(raw, dtype=np.float32)
        else:
            raise SystemExit(f"{name}: dtype {meta['dtype']} not handled")
        return a.astype(np.float64).reshape(meta["shape"])


def snapshot_dir(model):
    root = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots")
    for snap in sorted(glob.glob(os.path.join(root, "*"))):
        if glob.glob(os.path.join(snap, "*.safetensors")):
            return snap
    raise SystemExit(f"no safetensors snapshot for {model} under {root}")


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


def layer_factors(W, cfg, layer, fold_input_norm):
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", cfg["hidden_size"] // H)
    p = f"model.layers.{layer}."
    Q = W(p + "self_attn.q_proj.weight").reshape(H, hd, -1) * W(p + "self_attn.q_norm.weight")[None, :, None]
    K = W(p + "self_attn.k_proj.weight").reshape(KV, hd, -1) * W(p + "self_attn.k_norm.weight")[None, :, None]
    if fold_input_norm:
        gamma = W(p + "input_layernorm.weight")
        Q, K = Q * gamma, K * gamma
    return Q, K


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


def selfcheck(Q, K, cfg, rng):
    """Direct HF-style rope score (rotate_half convention) vs sum_j cos A_hj + sin B_hj built from the
    factored atoms, on random normalised rows at query position m, key position n."""
    H, hd, d = Q.shape
    P, grp = hd // 2, H // K.shape[0]
    omega = cfg["rope_theta"] ** (-np.arange(P) * 2.0 / hd)
    U, V = atoms(Q, K)
    x, y = rng.standard_normal(d), rng.standard_normal(d)
    xu = (x @ U).reshape(H, P, 2, 2)
    yv = (y @ V).reshape(H, P, 2, 2)
    bil = (xu * yv).sum(-1)  # x^T A_hj y, x^T B_hj y

    def rope(v, pos):
        c, s = np.cos(omega * pos), np.sin(omega * pos)
        return np.concatenate([v[:P] * c - v[P:] * s, v[P:] * c + v[:P] * s])

    m, n = 17, 1234
    err = 0.0
    for h in range(H):
        direct = rope(Q[h] @ x, m) @ rope(K[h // grp] @ y, n)
        expand = (np.cos(omega * (n - m)) * bil[h, :, 0] + np.sin(omega * (n - m)) * bil[h, :, 1]).sum()
        err = max(err, abs(direct - expand) / np.abs(Q[h] @ x).sum() / np.abs(K[h // grp] @ y).max())
    return err


def analyse_layer(Q, K, layer, with_pairs):
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
            "wavelength": float(2 * np.pi / (CFG_THETA ** (-2.0 * j / hd))),
            "energy": float(np.trace(G[np.ix_(b, b)])),
            "numerical_rank": r["numerical_rank"],
            "counts": spectrum_counts(e),
            "cos_pr": spectrum_counts(ec)["participation_ratio"],  # of the cosine Gram: max 2H
            "max_abs_cos_same_k": float(np.mean(same)),
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
        rec["head_pairs"] = {
            "same_k": {k: float(np.mean(v)) for k, v in same.items()} | {"n": len(same["captured"])},
            "diff_k": {k: float(np.mean(v)) for k, v in diff.items()} | {"n": len(diff["captured"])},
            "same_k_captured_max": float(np.max(same["captured"])),
            "diff_k_captured_max": float(np.max(diff["captured"])),
        }
    return rec


CFG_THETA = 1e6


def random_null(H, KV, hd, d, rng):
    """Same shapes, iid Gaussian weights: what 'no structure' looks like for the head-pair metrics."""
    Q = rng.standard_normal((H, hd, d))
    K = rng.standard_normal((KV, hd, d))
    U, V = atoms(Q, K)
    G = gram_from_factors(U, V)
    idx = np.arange(G.shape[0]).reshape(H, hd // 2, 2)
    grp = H // KV
    return {
        "actual": spectrum_counts(np.linalg.eigvalsh(G)),
        "captured_same_k": captured(G, idx[0].ravel(), idx[1].ravel()),
        "captured_diff_k": captured(G, idx[0].ravel(), idx[grp].ravel()),
        "top_cos_same_k": top_cosine(G, idx[0].ravel(), idx[1].ravel()),
        "top_cos_diff_k": top_cosine(G, idx[0].ravel(), idx[grp].ravel()),
    }


def main():
    global CFG_THETA
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fold-input-norm", action="store_true")
    ap.add_argument("--pairs-every", type=int, default=1, help="head-pair analysis every k-th layer")
    args = ap.parse_args()
    t0 = time.time()
    snap = snapshot_dir(args.model)
    W = Weights(snap)
    cfg = W.config
    CFG_THETA = float(cfg["rope_theta"])
    H, KV, L, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"], cfg["hidden_size"]
    hd = cfg.get("head_dim", d // H)
    rng = np.random.default_rng(2951)
    fold = not args.no_fold_input_norm

    layers, check = [], 0.0
    for layer in range(L):
        tl = time.time()
        Q, K = layer_factors(W, cfg, layer, fold)
        check = max(check, selfcheck(Q, K, cfg, rng))
        rec = analyse_layer(Q, K, layer, layer % args.pairs_every == 0)
        rec["seconds"] = time.time() - tl
        layers.append(rec)
        a, nh = rec["actual"], rec["null_heads_independent"]
        print(f"L{layer:2d} rank {rec['rank']['numerical_rank']}/{rec['rank']['max_dim']} "
              f"cert={rec['rank']['full_rank_certified']} k90/99/99.9 {a['k90']}/{a['k99']}/{a['k99.9']} "
              f"(heads-indep null {nh['k90']}/{nh['k99']}/{nh['k99.9']}) "
              f"pairs={rec.get('head_pairs', {}).get('same_k', {}).get('captured', float('nan')):.3f}/"
              f"{rec.get('head_pairs', {}).get('diff_k', {}).get('captured', float('nan')):.3f} "
              f"{rec['seconds']:.1f}s", flush=True)

    out = {
        "model": args.model,
        "snapshot": snap,
        "config": {"H": H, "KV": KV, "head_dim": hd, "d_model": d, "layers": L, "rope_theta": CFG_THETA},
        "fold": {"q_norm_k_norm_gain": True, "input_layernorm_gain": fold,
                 "excluded": "per-head RMS normaliser scalars of q_norm/k_norm (and the residual RMS)"},
        "rank_threshold": "tau = n * eps_f64 * lambda_max(Gram); full rank certified iff lambda_min > tau",
        "operator_expansion_selfcheck_max_rel_err": check,
        "random_gaussian_null": random_null(H, KV, hd, d, rng),
        "layers": layers,
        "seconds_total": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"total {out['seconds_total']:.1f}s  selfcheck {check:.2e}")


if __name__ == "__main__":
    main()
