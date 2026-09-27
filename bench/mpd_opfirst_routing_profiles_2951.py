"""#2951 operator-first: do RoPE attention heads share routing profiles, or is each head x frequency its own block?

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). Weight analysis is numpy float64 on the
exact bf16 weights, one layer at a time; the executed check (SmolLM2 only) runs one real HF attention module in float64.

Complex plane factors. With the half-split rotary pairing (coords a = j, b = j + hd/2, omega_j = theta^(-2j/hd)) and
the input-norm gain (and, for Qwen3, the q_norm / k_norm gains) folded into the rows,

    X_hj = q_{h,a} + i q_{h,b},   Y_gj = k_{g,a} + i k_{g,b}   (vectors in C^d; g = g(h) the GQA key head),
    Z_hj = X_hj Y_gj^H           (d x d, complex rank 1),
    score(m -> n) = sum_j Re(e^{-i omega_j t} x^T Z_hj y) / sqrt(hd),   t = n - m (key minus query position),

exact for bias-free attention without q/k norm (Llama / SmolLM2). With q/k norm (Qwen3) it is the bilinear operator
on the normalised vectors; the per-head RMS normaliser scalars are excluded, as in mpd_opfirst_rope_span_2951.py.

Shared-profile class of a family {Z_j}: joint column space U, row space V (orthonormal, complex), cores
C_j = U^H Z_j V, reference C_* = sum b_j C_j (generic b), ratios T_j = C_j C_*^{-1}. The class holds iff the T_j
commute and are diagonalisable; then T_j = S diag(lambda_j) S^{-1}, P_a = S E_a S^{-1} and
Z_j = sum_a lambda_ja B_a with B_a = U P_a C_* V^H. The native finite edit X' = A X, A = I + sum_a (eta_a - 1) U P_a U^H,
gives Z'_j = sum_a eta_a lambda_ja B_a exactly; one plane with eta = g e^{-i omega delta} is a gain g and a lag delta.
Square cores need r_U = r_V. Where the ranks differ the class cannot be posed; the script reports the ranks and then
(i) a square-truncated test (U cut to its leading r_V directions, truncation residual reported separately) and
(ii) the natural restrictions: one head over frequencies (r_U = r_V = hd/2) and one GQA group at one frequency (shared
key factor, so the only possible sharing is collinear query factors).

Softmax / payload bounds on an edited layer: if the logits of one query row move by d with oscillation
w = max d - min d, then TV(p, p') <= tanh(w/4) (sharp), and the head's output moves by at most
TV * diam_k(W_O,h v_k) <= tanh(w/4) * diam_k(W_O,h v_k).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import Decoder, compact_json  # noqa: E402
from mpd_opfirst_rope_span_2951 import EPS, spectrum_counts  # noqa: E402

N_PAIRS = 4000


def plane_factors(D, layer):
    """X (H, P, d), Y (KV, P, d) complex, all multiplicative gains folded (Decoder.qk: q/k-norm gains at their
    declared scope, input RMSNorm gain); plus the raw q_norm gains per head, (H or 1, hd), or None."""
    P = D.hd // 2
    Q, K = D.qk(layer)
    qn = None
    if D.qk_norm is not None:
        qn = D(f"model.layers.{layer}.self_attn.q_norm.weight").reshape(-1, D.hd)
    return Q[:, :P] + 1j * Q[:, P:], K[:, :P] + 1j * K[:, P:], qn


def omegas(cfg):
    H = cfg["num_attention_heads"]
    hd = cfg.get("head_dim", cfg["hidden_size"] // H)
    return float(cfg["rope_theta"]) ** (-np.arange(hd // 2) * 2.0 / hd)


def selfcheck(X, Y, cfg, rng):
    """HF rotate_half rope score on random rows vs sum_j Re(e^{-i w t} x^T Z_hj y) from the complex factors."""
    H, P, d = X.shape
    grp = H // Y.shape[0]
    om = omegas(cfg)
    Q = np.concatenate([X.real, X.imag], axis=1)
    K = np.concatenate([Y.real, Y.imag], axis=1)
    x, y = rng.standard_normal(d), rng.standard_normal(d)

    def rope(v, pos):
        c, s = np.cos(om * pos), np.sin(om * pos)
        return np.concatenate([v[:P] * c - v[P:] * s, v[P:] * c + v[:P] * s])

    m, n = 17, 1234
    err = 0.0
    for h in range(H):
        q, k = Q[h] @ x, K[h // grp] @ y
        direct = rope(q, m) @ rope(k, n)
        z = (X[h] @ x) * np.conj(Y[h // grp] @ y)
        expand = np.sum((np.exp(-1j * om * (n - m)) * z).real)
        err = max(err, abs(direct - expand) / (np.abs(q).sum() * np.abs(k).max()))
    return err


def crank(M):
    """Numerical rank of a complex d x N stack at tau = max(d, N) eps sigma_1, with the singular values."""
    s = np.linalg.svd(M, compute_uv=False)
    tau = max(M.shape) * EPS * s[0]
    return int((s > tau).sum()), s


def joint_ranks(Xs, Ys):
    rU, sU = crank(Xs)
    rV, sV = crank(Ys)
    return {"r_U": rU, "r_V": rV, "sum_ranks": int(Xs.shape[1]),
            "sU_min_over_max": float(sU[-1] / sU[0]), "sV_min_over_max": float(sV[-1] / sV[0])}


def span_report(Xs, Ydup):
    """Dimension of span_C{Z_j} in matrix space (Gram <Z_a, Z_b> = (X_a^H X_b)(Y_b^H Y_a)) and its energy counts
    against the 'every Z_j orthogonal to every other' null. Any shared-profile class with K blocks has span <= K."""
    G = (Xs.conj().T @ Xs) * (Ydup.conj().T @ Ydup).T
    e = np.linalg.eigvalsh(G)
    tau = G.shape[0] * EPS * e[-1]
    return {"n": int(G.shape[0]), "span_dim": int((e > tau).sum()), "lambda_min_over_max": float(e[0] / e[-1]),
            "actual": spectrum_counts(e), "null_orthogonal": spectrum_counts(np.real(np.diag(G)).copy())}


def class_test(Xs, Ydist, Ydup, rng, truncate):
    """Shared-profile class test on the family Z_j = Xs[:, j] Ydup[:, j]^H.

    truncate=False requires r_U = r_V (exact test). truncate=True cuts U to the leading r = r_V left singular
    directions of the X stack (the square-truncated test) and reports the truncation residual separately."""
    d, N = Xs.shape
    Uf, sU, _ = np.linalg.svd(Xs, full_matrices=False)
    Vf, sV, _ = np.linalg.svd(Ydist, full_matrices=False)
    rU = int((sU > max(Xs.shape) * EPS * sU[0]).sum())
    rV = int((sV > max(Ydist.shape) * EPS * sV[0]).sum())
    rec = {"r_U": rU, "r_V": rV, "n": N}
    if rU != rV and not truncate:
        rec["posable"] = False
        return rec, None
    r = min(rU, rV)
    U, V = Uf[:, :r], Vf[:, :r]
    u = U.conj().T @ Xs  # r x N,  C_j = u_j v_j^H
    v = V.conj().T @ Ydup
    znorm2 = np.sum(np.abs(Xs) ** 2, 0) * np.sum(np.abs(Ydup) ** 2, 0)
    # ||Z_j - U C_j V^H||^2 = |x_perp|^2 |Y_j|^2 + |u_j|^2 |y_perp|^2 (no cancellation)
    xp2 = np.sum(np.abs(Xs - U @ u) ** 2, 0)
    yp2 = np.sum(np.abs(Ydup - V @ v) ** 2, 0)
    trunc = np.sqrt((xp2 * np.sum(np.abs(Ydup) ** 2, 0) + np.sum(np.abs(u) ** 2, 0) * yp2) / znorm2)
    b = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    Cs = (u * b) @ v.conj().T
    Bm = np.linalg.solve(Cs.conj().T, v)  # T_j = u_j Bm_j^H
    # commutators of rank-one ratios, relative to ||T_j|| ||T_k||
    na = np.linalg.norm(u, axis=0) * np.linalg.norm(Bm, axis=0)
    pairs = [(j, k) for j in range(N) for k in range(j + 1, N)]
    if len(pairs) > N_PAIRS:
        pairs = [pairs[i] for i in rng.choice(len(pairs), N_PAIRS, replace=False)]
    comm = np.empty(len(pairs))
    for i, (j, k) in enumerate(pairs):
        al, be = Bm[:, j].conj() @ u[:, k], Bm[:, k].conj() @ u[:, j]
        c = al * np.outer(u[:, j], Bm[:, k].conj()) - be * np.outer(u[:, k], Bm[:, j].conj())
        comm[i] = np.linalg.norm(c) / (na[j] * na[k])
    # simultaneous diagonalisation through a generic combination, then reconstruction
    c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    ev, S = np.linalg.eig((u * c) @ Bm.conj().T)
    Sinv = np.linalg.inv(S)
    lam = (Sinv @ u).T * (Bm.conj().T @ S)  # lam[j, a] = (S^-1 u_j)_a (Bm_j^H S)_a
    Wm = Sinv @ Cs
    core = np.empty(N)
    for j in range(N):
        core[j] = np.linalg.norm(np.outer(u[:, j], v[:, j].conj()) - (S * lam[j]) @ Wm)
    core /= np.sqrt(znorm2)
    gaps = np.abs(ev[:, None] - ev[None, :]) + np.diag(np.full(r, np.inf))
    rec.update({
        "posable": rU == rV, "r": r,
        "cond_C_star": float(np.linalg.cond(Cs)), "cond_S": float(np.linalg.cond(S)),
        "min_eig_gap_rel": float(gaps.min() / np.abs(ev).max()),
        "commutator_rel": {"median": float(np.median(comm)), "max": float(comm.max()), "pairs": len(pairs)},
        "truncation_rel": {"median": float(np.median(trunc)), "max": float(trunc.max())},
        "core_recon_rel": {"median": float(np.median(core)), "max": float(core.max())},
        "recon_rel_total_max": float(np.sqrt(trunc ** 2 + core ** 2).max()),
    })
    return rec, (U, V, S, Sinv, lam, Cs, u, v, b)


def head_exact(X, Y, grp, rng):
    """Per-head family over frequencies (equal ranks): exact class test, recovered blocks, and the native edit."""
    H, P, d = X.shape
    out = []
    for h in range(H):
        Xs, Ys = X[h].T, Y[h // grp].T
        rec, parts = class_test(Xs, Ys, Ys, rng, truncate=False)
        if parts is None:
            out.append(rec)
            continue
        U, V, S, Sinv, lam, Cs, u, v, b = parts
        # which frequency does each recovered block belong to: lam[j, a] b_j is the block weight
        wgt = np.abs(lam * b[:, None])
        purity = wgt.max(0) / wgt.sum(0)
        owner = wgt.argmax(0)
        # native finite edit on the recovered blocks: A = I + sum_a (eta_a - 1) U P_a U^H, check A X_j = eta_{a(j)} X_j
        eta = rng.uniform(0.5, 2.0, P) * np.exp(1j * rng.uniform(-np.pi, np.pi, P))
        Mid = (S * (eta - 1)) @ Sinv
        AX = Xs + U @ (Mid @ (U.conj().T @ Xs))
        want = Xs * eta[np.argsort(owner)][None, :] if len(set(owner)) == P else None
        edit_err = float(np.abs(AX - want).max() / np.abs(Xs).max()) if want is not None else float("nan")
        rec.update({"blocks": int(len(set(owner))), "min_purity": float(purity.min()), "edit_rel_err": edit_err})
        out.append(rec)
    return out


def group_frequency(X, Y, grp, rng):
    """Per (GQA group, frequency) the heads share one key factor, so they share a routing profile iff their query
    factors are complex-collinear. Scale-free: factors are normalised to unit length first. Reports the largest
    pairwise |cos| and sigma_min/sigma_max of the normalised group stack, and per frequency the participation ratio of
    all H normalised query factors, each against iid random directions."""
    H, P, d = X.shape
    unit = X / np.linalg.norm(X, axis=2, keepdims=True)
    R = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
    null = R / np.linalg.norm(R, axis=2, keepdims=True)
    stats = {}
    for name, F in (("actual", unit), ("null", null)):
        cos, ratio, pr = [], [], []
        for j in range(P):
            for g in range(H // grp):
                A = F[g * grp:(g + 1) * grp, j]
                c = np.abs(A.conj() @ A.T)
                cos.append((c - np.eye(grp)).max())
                s = np.linalg.svd(A, compute_uv=False)
                ratio.append(s[-1] / s[0])
            e = np.linalg.svd(F[:, j], compute_uv=False) ** 2
            pr.append(e.sum() ** 2 / (e ** 2).sum() / H)
        stats[name] = {"max_abs_cos": {"median": float(np.median(cos)), "max": float(np.max(cos))},
                       "sigma_min_over_max": {"median": float(np.median(ratio)), "min": float(np.min(ratio))},
                       "cross_head_PR_over_H": {"median": float(np.median(pr)), "min": float(np.min(pr)),
                                                "per_plane": [round(float(x), 4) for x in pr]}}
    # distance to the class, energy weighted: the best shared profile for a (group, frequency) cell replaces the
    # query factors by their projections on one direction; relative Frobenius residual of the cell's Z's is
    # sqrt(1 - sigma_1^2 / sum sigma^2) of the (unnormalised) group stack, since the key factor is common
    res, en, resn = np.empty((H // grp, P)), np.empty((H // grp, P)), np.empty((H // grp, P))
    for j in range(P):
        for g in range(H // grp):
            A = X[g * grp:(g + 1) * grp, j]
            s = np.linalg.svd(A, compute_uv=False) ** 2
            res[g, j] = np.sqrt(max(0.0, 1 - s[0] / s.sum()))
            en[g, j] = s.sum() * np.linalg.norm(Y[g, j]) ** 2
            sn = np.linalg.svd(null[g * grp:(g + 1) * grp, j] * np.linalg.norm(A, axis=1, keepdims=True),
                               compute_uv=False) ** 2
            resn[g, j] = np.sqrt(max(0.0, 1 - sn[0] / sn.sum()))
    w = en / en.sum()
    stats["shared_profile_residual"] = {
        "median": float(np.median(res)), "min": float(res.min()), "null_median": float(np.median(resn)),
        "energy_weighted_mean": float((w * res).sum()), "null_energy_weighted_mean": float((w * resn).sum()),
        "cells_below_0.1": int((res < 0.1).sum()), "cells_below_0.3": int((res < 0.3).sum()), "cells": int(res.size),
        "energy_share_below_0.1": float(w[res < 0.1].sum()), "energy_share_below_0.3": float(w[res < 0.3].sum()),
        "per_plane_median": [round(float(x), 4) for x in np.median(res, 0)],
        "per_plane_energy_share": [round(float(x), 5) for x in w.sum(0)]}
    return stats


def analyse_layer(X, Y, rng, full_class):
    H, P, d = X.shape
    KV = Y.shape[0]
    grp = H // KV
    Xs = X.reshape(H * P, d).T
    Ydup = np.stack([Y[h // grp] for h in range(H)]).reshape(H * P, d).T
    Ydist = Y.reshape(KV * P, d).T
    rec = {"joint_ranks": joint_ranks(Xs, Ydist), "span": span_report(Xs, Ydup)}
    if full_class:
        rec["class_layer_truncated"] = class_test(Xs, Ydist, Ydup, rng, truncate=True)[0]
    groups = []
    for g in range(KV):
        Xg = X[g * grp:(g + 1) * grp].reshape(grp * P, d).T
        Yg = np.tile(Y[g], (grp, 1)).T
        gr = {"g": g, "joint_ranks": joint_ranks(Xg, Y[g].T), "span": span_report(Xg, Yg)}
        gr["class_truncated"] = class_test(Xg, Y[g].T, Yg, rng, truncate=True)[0]
        groups.append(gr)
    rec["gqa_groups"] = groups
    heads = head_exact(X, Y, grp, rng)
    rec["per_head_exact"] = {
        "posable_heads": sum(1 for r in heads if r.get("posable")),
        "blocks": [r.get("blocks") for r in heads],
        "commutator_rel_max": max(r["commutator_rel"]["max"] for r in heads if "r" in r),
        "core_recon_rel_max": max(r["core_recon_rel"]["max"] for r in heads if "r" in r),
        "truncation_rel_max": max(r["truncation_rel"]["max"] for r in heads if "r" in r),
        "min_purity": min(r["min_purity"] for r in heads if "r" in r),
        "edit_rel_err_max": max(r["edit_rel_err"] for r in heads if "r" in r),
        "cond_C_star_max": max(r["cond_C_star"] for r in heads if "r" in r),
        "cond_S_max": max(r["cond_S"] for r in heads if "r" in r),
    }
    rec["group_frequency"] = group_frequency(X, Y, grp, rng)
    return rec


# ---------------------------------------------------------------- executed edit on a real layer (Llama / SmolLM2)

def edit_checks(model_id, D, layers, edits, rng):
    import copy

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import AttentionInterface

    from mpd_opfirst_mlp_oddeven_2951 import FALLBACK_TEXT

    torch.set_num_threads(6)
    captured = {}

    def capture_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        # eager_attention_forward with the softmax kept in float64 (eager casts it to float32) and the logits recorded
        rep = module.num_key_value_groups
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
        logits = torch.matmul(query, key.transpose(2, 3)) * scaling
        captured["logits"] = logits.detach().clone()
        w = torch.softmax(logits + attention_mask, dim=-1)
        captured["probs"] = w.detach().clone()
        captured["values"] = value.detach().clone()
        return torch.matmul(w, value).transpose(1, 2).contiguous(), w

    AttentionInterface.register("capture_f64", capture_attention)
    snap = D.snap
    tok = AutoTokenizer.from_pretrained(snap)
    ids = torch.tensor(tok(FALLBACK_TEXT).input_ids[:48])[None]
    T = ids.shape[1]
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    H, KV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    P, grp = hd // 2, H // KV
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    inv_freq = model.model.rotary_emb.inv_freq.detach().double()  # the model's own (float32) frequencies
    pos = torch.arange(T, dtype=torch.float64)
    fr = torch.outer(pos, inv_freq)
    emb = torch.cat([fr, fr], -1)
    pe = (emb.cos()[None], emb.sin()[None])
    mask = torch.triu(torch.full((T, T), -torch.inf, dtype=torch.float64), 1)[None, None]
    om = inv_freq.numpy()
    tmat = (np.arange(T)[None, :] - np.arange(T)[:, None]).astype(np.float64)  # t = n - m
    causal = np.tril(np.ones((T, T), bool))
    Wf = D
    results = []
    for L in layers:
        layer = model.model.layers[L]
        attn = layer.self_attn
        X, Y, _ = plane_factors(Wf, L)
        xh = hs[L].double()[0].numpy()
        xh = xh / np.sqrt(np.mean(xh ** 2, -1, keepdims=True) + cfg.rms_norm_eps)  # gain is folded in X, Y
        # input RMSNorm in float64 here: HF's LlamaRMSNorm upcasts to float32, a downcast for a float64 input
        xin = torch.from_numpy(xh * layer.input_layernorm.weight.detach().double().numpy())[None]
        qt = np.einsum("hjd,td->htj", X, xh)
        kt = np.einsum("gjd,td->gtj", Y, xh)
        zz = qt[:, :, None, :] * np.conj(kt[np.arange(H) // grp][:, None, :, :])  # h, m, n, j
        phase = np.exp(-1j * om[None, None, :] * tmat[:, :, None])
        pred = np.sum((phase[None] * zz).real, -1) / np.sqrt(hd)

        def run(attn64):
            with torch.no_grad():
                out, _ = attn64(xin, position_embeddings=pe, attention_mask=mask)
            return (out[0].numpy(), captured["logits"][0].numpy(), captured["probs"][0].numpy(),
                    captured["values"][0].numpy())

        def f64(module):
            m = copy.deepcopy(module).double()
            m.config._attn_implementation = "capture_f64"
            return m

        base = f64(attn)
        out0, lg0, p0, v0 = run(base)
        scale = np.abs(lg0[:, causal]).max()
        base_err = float(np.abs(lg0 - pred)[:, causal].max() / scale)
        # choose the plane: largest mean |q k| among planes whose wavelength fits the window
        wl = 2 * np.pi / om
        fit = np.where((wl >= 8) & (wl <= 4 * T))[0]
        mag = np.abs(zz).mean(axis=(1, 2))[:, fit]
        h, jj = np.unravel_index(np.argmax(mag), mag.shape)
        j = int(fit[jj])
        Wo = attn.o_proj.weight.detach().double().numpy()[:, h * hd:(h + 1) * hd]
        payload = v0[h] @ Wo.T  # T x d, the head's output per key
        diam = np.sqrt(((payload[:, None] - payload[None]) ** 2).sum(-1))
        for g, delta in edits:
            if delta is None:
                delta = float(np.pi / om[j])  # half a wavelength: a sign flip of the plane
            eta = g * np.exp(-1j * om[j] * delta)
            ed = f64(attn)
            with torch.no_grad():
                Wq = ed.q_proj.weight
                ra, rb = Wq[h * hd + j].clone(), Wq[h * hd + j + P].clone()
                Wq[h * hd + j] = eta.real * ra - eta.imag * rb
                Wq[h * hd + j + P] = eta.imag * ra + eta.real * rb
            out1, lg1, p1, _ = run(ed)
            # prediction: plane (h, j) term becomes g Re(e^{-i w (t + delta)} x^T Z y)
            zj = zz[h, :, :, j]
            dpred = ((g * np.exp(-1j * om[j] * (tmat + delta)) - np.exp(-1j * om[j] * tmat)) * zj).real / np.sqrt(hd)
            pred1 = pred.copy()
            pred1[h] += dpred
            edit_err = float(np.abs(lg1 - pred1)[:, causal].max() / scale)
            other = float(np.abs(np.delete(lg1 - lg0, h, 0))[:, causal].max())
            rows = []
            for m in range(1, T):
                dm = dpred[m, :m + 1]
                w = dm.max() - dm.min()
                tv = 0.5 * np.abs(p1[h, m, :m + 1] - p0[h, m, :m + 1]).sum()
                dmx = diam[:m + 1, :m + 1].max()
                dout = np.linalg.norm(out1[m] - out0[m])
                rows.append((w, tv, np.tanh(w / 4), dout, tv * dmx, np.tanh(w / 4) * dmx))
            R = np.array(rows)
            ok = R[:, 2] > 0
            results.append({
                "layer": L, "head": int(h), "plane": j, "wavelength": float(wl[j]), "gain": g, "lag": delta,
                "tokens": T, "unedited_score_identity_rel_err": base_err, "edited_score_rel_err": edit_err,
                "other_heads_max_abs_logit_change": other,
                "w": {"median": float(np.median(R[:, 0])), "max": float(R[:, 0].max())},
                "tv_over_tanh_w4": {"median": float(np.median(R[ok, 1] / R[ok, 2])), "max": float((R[ok, 1] / R[ok, 2]).max())},
                "tv": {"median": float(np.median(R[:, 1])), "max": float(R[:, 1].max())},
                "payload_measured_over_tv_diam": {"median": float(np.median(R[ok, 3] / R[ok, 4])),
                                                  "max": float((R[ok, 3] / R[ok, 4]).max())},
                "payload_measured_over_tanh_bound": {"median": float(np.median(R[ok, 3] / R[ok, 5])),
                                                     "max": float((R[ok, 3] / R[ok, 5]).max())},
                "payload_measured_max": float(R[:, 3].max()),
            })
            print(f"  edit L{L} h{h} j{j} (wl {wl[j]:.1f}) g={g} lag={delta:.2f}: identity {base_err:.1e} "
                  f"edited {edit_err:.1e} others {other:.1e} TV/tanh max {results[-1]['tv_over_tanh_w4']['max']:.3f} "
                  f"payload/bound max {results[-1]['payload_measured_over_tanh_bound']['max']:.3f}", flush=True)
    del model
    return {"text_source": "fixed passage (bench/mpd_opfirst_mlp_oddeven_2951.py FALLBACK_TEXT), first 48 tokens",
            "rotary": "cos/sin in float64 from the model's own float32 inv_freq buffer",
            "softmax": "eager attention with the softmax in float64 (eager casts it to float32); logits captured",
            "input_norm": "RMSNorm computed in float64 by the script (HF's upcasts to float32); attention module is HF's",
            "edits": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--full-class-every", type=int, default=1)
    ap.add_argument("--edit-layers", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    D = Decoder(args.model)
    snap, cfg = D.snap, D.config
    H, KV, L, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"], cfg["hidden_size"]
    hd = cfg.get("head_dim", d // H)
    layers = list(range(L)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(2951)
    qk_norm = D.qk_norm is not None
    out = {
        "model": args.model, "snapshot": snap,
        "config": {"H": H, "KV": KV, "head_dim": hd, "d_model": d, "layers": L, "rope_theta": float(cfg["rope_theta"]),
                   "qk_norm": qk_norm, "attention_bias": bool(cfg.get("attention_bias", False))},
        "convention": "X = q_a + i q_b, Y = k_a + i k_b (half-split planes, gains folded), Z = X Y^H, "
                      "score = sum_j Re(e^{-i w_j t} x^T Z_j y)/sqrt(hd), t = key pos - query pos",
        "rank_threshold": "tau = max(rows, cols) eps_f64 sigma_1 (stacks); n eps lambda_max (Grams)",
        "exactness": ("exact: bias-free, no q/k norm, so scores are exactly the complex bilinear sum and the native "
                      "edit is exact" if not qk_norm else
                      "NOT exact for the finite edit: q_norm divides each head's query by its RMS over all planes, "
                      "which the edit changes (exact only for |eta| = 1 with equal q_norm gains on the pair); the "
                      "weight analysis is the bilinear operator on normalised vectors, normaliser scalars excluded"),
    }
    recs, check = [], 0.0
    for layer in layers:
        tl = time.time()
        X, Y, qn = plane_factors(D, layer)
        check = max(check, selfcheck(X, Y, cfg, rng))
        rec = {"layer": layer}
        if qn is not None:
            P = hd // 2
            rec["q_norm_pair_gain_equal_frac"] = float(np.mean(qn[:, :P] == qn[:, P:]))
        rec |= analyse_layer(X, Y, rng, full_class=layer % args.full_class_every == 0)
        rec["seconds"] = time.time() - tl
        recs.append(rec)
        jr, sp, ph = rec["joint_ranks"], rec["span"], rec["per_head_exact"]
        ct = rec.get("class_layer_truncated", {})
        g0 = rec["gqa_groups"][0]
        print(f"L{layer:2d} rU/rV/sum {jr['r_U']}/{jr['r_V']}/{jr['sum_ranks']} span {sp['span_dim']}/{sp['n']} "
              f"k99 {sp['actual']['k99']} (null {sp['null_orthogonal']['k99']}) | layer-trunc comm "
              f"{ct.get('commutator_rel', {}).get('median', float('nan')):.2e} recon "
              f"{ct.get('recon_rel_total_max', float('nan')):.2f} | grp0 {g0['joint_ranks']['r_U']}/"
              f"{g0['joint_ranks']['r_V']} comm {g0['class_truncated']['commutator_rel']['median']:.2e} | head-exact "
              f"blocks {min(ph['blocks'])}-{max(ph['blocks'])} comm {ph['commutator_rel_max']:.1e} recon "
              f"{ph['core_recon_rel_max']:.1e} edit {ph['edit_rel_err_max']:.1e} | gf "
              f"{rec['group_frequency']['actual']['max_abs_cos']['max']:.3f} {rec['seconds']:.1f}s", flush=True)
    out["score_identity_selfcheck_max_rel_err"] = check
    # calibration: iid complex Gaussian factors of the same shapes and GQA structure
    Xn = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
    Yn = rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)
    nl = analyse_layer(Xn, Yn, rng, full_class=True)
    out["gaussian_null"] = {"class_layer_truncated": nl["class_layer_truncated"],
                            "class_group0_truncated": nl["gqa_groups"][0]["class_truncated"],
                            "span": nl["span"], "per_head_exact": nl["per_head_exact"],
                            "group_frequency": nl["group_frequency"]}
    out["layers"] = recs
    if args.edit_layers:
        out["executed_edit"] = edit_checks(args.model, D, [int(x) for x in args.edit_layers.split(",")],
                                           [(2.0, 3.0), (1.0, None)], rng)
    out["seconds_total"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(compact_json(out))
    print(f"total {out['seconds_total']:.1f}s selfcheck {check:.2e}")


if __name__ == "__main__":
    main()
