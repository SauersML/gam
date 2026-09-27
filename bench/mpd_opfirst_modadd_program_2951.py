"""An executable, weight-grounded program for the one-layer modular-addition transformer (#2951).

The model is bench/mpd_modadd_2951.py's ``ModAddTransformer`` (p = 113, no LayerNorm, 4 heads, ReLU MLP),
read from a ``train`` run file. Everything is float64 numpy on CPU. The receipts were made from
``mpd_modadd_2951.py train --labels addition --seed 0 --steps 20000 --eval-every 1000 --checkpoint-steps 20000``
(final test accuracy 1.0), with ``--read frac`` (opfirst_modadd_program.json), ``own`` and ``S_E``.

Exact rewrite R (checked against the direct forward on all p^2 inputs). Only the ``=`` position is read, so
only its residual matters. With characters D(w x) = (cos w x, sin w x), w_k = 2 pi k / p, k = 1..(p-1)/2:

* embedding  e(x) = c0 + sum_k U_k D(w_k x)               (DFT of W_E's rows over the token cycle, exact for odd p;
                                                           c0, U_k from cyclic_action::cyclic_planes and the pos0 edit
                                                           from cyclic_action::frequency_edit, both through the MPD
                                                           surface; ``chars`` evaluates D at the program's inputs)
* scores     s_hj = sig_hj + sum_k g_hk . D(w_k x_j)      (the query at ``=`` is input-independent; j = 0, 1)
* routing    alpha_h = softmax(s_h0, s_h1, sig_h2)
* moved      zeta_hk = alpha_h0 D(w_k a) + alpha_h1 D(w_k b)   (what head h carries in plane k: its OV image O_h U_k)
* MLP        pre_n = beta_n + sum_hj alpha_hj kap_nhj + sum_hk w_nhk . zeta_hk,   act_n = relu(pre_n)
* unembed    W_U[c] = u0 + sum_k V_k D(w_k c), so up to a c-constant, logit(c) = sum_k D(w_k c) . y_k,
             y_k = y0_k + sum_n rho_nk act_n + sum_hj dk_khj alpha_hj + sum_hk' dw_khk' . zeta_hk'   (direct path).

P is R restricted by weight-only choices, none fitted to model outputs: S_E = W_E frequencies with power
share > 1% (scores, values); S_U = W_U frequencies with share > 1% (readout planes); each neuron reads only its
own frequency k(n) = argmax over S_E of its read energy sum_h |w_nhk|^2 (``--read own``) or all of S_E
(``--read S_E``); every kap is folded at alpha = (1/2, 1/2, 0) into beta. The closed form T is
y_k = A_k D(w_k (a + b)) on S_U, A_k = mean over Z_p^2 of y_k^P . D(w_k (a + b)), the exact Fourier coefficient of
P's own output (a function of the weights through P).

Certificates are exhaustive over all p^2 inputs: every input is evaluated (Exact{Exhaustive} for the finite
family). Held-out counterfactuals, run on the model and predicted by P and T, none used to build them:
 (a) frequency_edit of the embedding at the pos0 use site, planes in a subset turned by w_k s, all s = 1..p-1;
 (b) at the attention output, the plane-k content every head writes, sum_h O_h U_k zeta_hk, interchanged from a
     donor input (a seeded permutation of all inputs) or replaced by its mean over inputs;
 (c) W_U's plane V_k scaled by lambda (a realizable parameter edit).

``--select greedy --greedy-epsilon eps`` replaces the 1% cutoffs by MDL structure selection over R (see the greedy
section): one-edit proposals (drop a plane, a head's score plane, a direct path or a neuron's read of a plane; route a
head by its law; fold a head's kap; coarsen a real group's declared precision), each accepted iff the decoded program
is strictly shorter and meets eps, with code lengths from the Rust codec (``code_lengths``) and the decision from
fit.rs ``decide_proposal``, both through the MPD surface. eps is the only declared input.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch


def chars(k, x, p):
    """D(w_k x) for frequencies k (K,) and integers x (N,): shape (N, K, 2)."""
    w = 2 * np.pi * np.outer(np.asarray(x), np.asarray(k)) / p
    return np.stack([np.cos(w), np.sin(w)], -1)


def mpd(operation, tensors):
    """One operation of the Rust MPD surface (gam_sae::parameter_decomposition::surface)."""
    from gamfit.sae import run_parameter_decomposition

    return run_parameter_decomposition({"schema": "gam.mpd-request", "schema_version": 1, "operation": operation},
                                       tensors)


def token_cycle(p, rows):
    """The declared successor x -> x + 1 (mod p) over a table's first p rows; any further row is fixed."""
    return [(x + 1) % p if x < p else x for x in range(rows)]


def cyclic_planes(table, p):
    """cyclic_action::cyclic_planes of ``table`` under the token cycle: c0 (d,), the planes U (K, 2, d) with
    U[k-1] = [u_kc, u_ks], and each plane's power (K,)."""
    out = mpd({"kind": "cyclic_planes", "table": "table", "successor": token_cycle(p, table.shape[0])},
              {"table": table})
    planes = out.arrays["planes"]
    return out.arrays["mean"], planes.T.reshape(-1, 2, planes.shape[0]), np.array(out.report["result"]["power"])


def frequency_edited(table, p, frequencies, shift):
    """cyclic_action::frequency_edit of the closed-form planes ``frequencies`` of ``table`` at ``shift``:
    the edited table E + left right^T (rows outside the cycle stay fixed)."""
    out = mpd({"kind": "frequency_edit", "successor": token_cycle(p, table.shape[0]),
               "basis": {"kind": "closed_form", "table": "table"},
               "frequencies": sorted(int(k) for k in frequencies), "shift": int(shift)}, {"table": table})
    return table + out.arrays["left"] @ out.arrays["right"].T


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def load(path):
    run = torch.load(path, map_location="cpu", weights_only=True)
    step = max(run["checkpoints"])
    return run["config"], step, {k: v.double().numpy() for k, v in run["checkpoints"][step].items()}


# ------------------------------------------------------------------------------------------------ the model

def model_forward(W, cfg, a, b, table0=None, attn_patch=None, W_U=None):
    """The model at ``=``. ``table0`` replaces W_E at the pos0 use site; ``attn_patch(alpha)`` is added to the
    attention output; ``W_U`` replaces the unembedding."""
    p, dh = cfg["p"], cfg["d_head"]
    E = W["W_E"]
    x0 = (E if table0 is None else table0)[a] + W["W_pos"][0]
    x2 = E[p] + W["W_pos"][2]
    xs = np.stack([x0, E[b] + W["W_pos"][1], np.broadcast_to(x2, x0.shape)], 1)
    q = np.einsum("hkd,d->hk", W["W_Q"], x2)
    alpha = softmax(np.einsum("hk,hkd,njd->nhj", q, W["W_K"], xs, optimize=True) / math.sqrt(dh))
    z = np.einsum("nhj,hkd,njd->nhk", alpha, W["W_V"], xs, optimize=True).reshape(len(a), -1)
    attn = z @ W["W_O"].T
    if attn_patch is not None:
        attn = attn + attn_patch(alpha)
    mid = x2 + attn
    pre = mid @ W["W_in"].T + W["b_in"]
    fin = mid + np.maximum(pre, 0) @ W["W_out"].T + W["b_out"]
    return {"alpha": alpha, "pre": pre, "logits": fin @ (W["W_U"] if W_U is None else W_U).T}


# ------------------------------------------------------------------------------------- the exact rewrite R

def exact_coordinates(W, cfg):
    """Every coefficient of R, all frequencies, from the weights alone."""
    p, H, dh = cfg["p"], cfg["n_heads"], cfg["d_head"]
    K = np.arange(1, (p - 1) // 2 + 1)
    c0, U, _ = cyclic_planes(W["W_E"][:p], p)                                       # d / K,2,d
    V = cyclic_planes(W["W_U"], p)[1]                                               # K,2,d
    x2 = W["W_E"][p] + W["W_pos"][2]
    base = np.stack([c0 + W["W_pos"][0], c0 + W["W_pos"][1], x2])                   # 3,d
    qk = np.einsum("hk,hkd->hd", np.einsum("hkd,d->hk", W["W_Q"], x2), W["W_K"]) / math.sqrt(dh)
    O = np.stack([W["W_O"][:, h * dh:(h + 1) * dh] @ W["W_V"][h] for h in range(H)])   # H,d,d
    OU = np.einsum("hde,kte->hktd", O, U)                                           # H,K,2,d
    Ob = np.einsum("hde,je->hjd", O, base)                                          # H,3,d
    return {
        "p": p, "K": K, "U": U, "V": V, "O": O,
        "g": np.einsum("hd,ktd->hkt", qk, U), "sig": qk @ base.T,                  # H,K,2 / H,3
        "beta": W["b_in"] + W["W_in"] @ x2,
        "kap": np.einsum("nd,hjd->nhj", W["W_in"], Ob),                             # n,H,3
        "w": np.einsum("nd,hktd->nhkt", W["W_in"], OU),                             # n,H,K,2
        "rho": np.einsum("ktd,dn->nkt", V, W["W_out"]),                             # n,K,2
        "y0": np.einsum("ktd,d->kt", V, x2 + W["b_out"]),                           # K,2
        "dk": np.einsum("ktd,hjd->kthj", V, Ob),                                    # K,2,H,3
        "dw": np.einsum("ktd,hjsd->kthjs", V, OU),                                  # K,2,H,K,2
    }


def rewrite_logits(X, a, b, att=None, read=None, write=None, direct=True, bias=True, kappa=True,
                   shift=None, zeta_patch=None, uscale=None):
    """R with optional restrictions: masks ``att`` (K,) on the scores, ``read``/``write`` (n,K) on each neuron;
    ``direct``/``bias`` keep the attention-to-W_U path and y0; ``kappa`` False folds kap at alpha = (1/2, 1/2, 0).
    Unrestricted, it equals the model's c-centered logits. ``shift``, ``zeta_patch`` and ``uscale`` are the
    counterfactual hooks of run_program, over all frequencies."""
    p, K = X["p"], X["K"]
    Da, Db = chars(K, a, p), chars(K, b, p)
    if shift is not None:
        sel = np.isin(K, shift[0])
        Da[:, sel] = chars(K[sel], a + shift[1], p)
    g = X["g"] if att is None else X["g"] * att[None, :, None]
    alpha = softmax(np.stack([X["sig"][:, 0] + np.einsum("hkt,nkt->nh", g, Da),
                              X["sig"][:, 1] + np.einsum("hkt,nkt->nh", g, Db),
                              np.broadcast_to(X["sig"][:, 2], (len(a), len(g)))], -1))
    zeta = alpha[:, :, 0, None, None] * Da[:, None] + alpha[:, :, 1, None, None] * Db[:, None]
    if zeta_patch is not None:
        zeta = zeta_patch(zeta)
    w = X["w"] if read is None else X["w"] * read[:, None, :, None]
    const = np.einsum("nhj,mhj->mn", X["kap"], alpha) if kappa else 0.5 * (X["kap"][:, :, 0] + X["kap"][:, :, 1]).sum(1)
    pre = X["beta"] + const + np.einsum("nhkt,mhkt->mn", w, zeta, optimize=True)
    rho = X["rho"] if write is None else X["rho"] * write[:, :, None]
    y = np.einsum("nkt,mn->mkt", rho, np.maximum(pre, 0), optimize=True)
    if bias:
        y = y + X["y0"]
    if direct:
        y = y + np.einsum("kthj,mhj->mkt", X["dk"], alpha) + np.einsum("kthjs,mhjs->mkt", X["dw"], zeta, optimize=True)
    if uscale:
        y = y * np.array([uscale.get(int(k), 1.0) for k in K])[None, :, None]
    return np.einsum("mkt,ckt->mc", y, chars(K, np.arange(p), p)), alpha, pre


# ---------------------------------------------------------------------------------------------- program P

def extract_program(X, s_embed, s_read, read="frac", read_frac=0.01):
    """The parameters of P, sliced from R. No model output is consulted. ``read``: ``own`` keeps each neuron's
    argmax frequency only, ``frac`` every S_E frequency holding >= read_frac of its S_E read energy, ``S_E`` all."""
    K = X["K"]
    ke, ku = np.searchsorted(K, s_embed), np.searchsorted(K, s_read)
    w = X["w"][:, :, ke]                                                            # n,H,Ke,2
    energy = np.einsum("nhkt->nk", w ** 2)
    own = energy.argmax(1)                                                          # index into s_embed
    mask = np.ones_like(energy)
    if read == "own":
        mask = np.zeros_like(energy)
        mask[np.arange(len(own)), own] = 1.0
    elif read == "frac":
        mask = (energy >= read_frac * energy.sum(1, keepdims=True)).astype(float)
    w = w * mask[:, None, :, None]
    return {
        "p": X["p"], "s_embed": np.asarray(s_embed), "s_read": np.asarray(s_read), "read": read, "own": own,
        "read_mask": mask,
        "g": X["g"][:, ke], "sig": X["sig"],
        "beta": X["beta"] + 0.5 * (X["kap"][:, :, 0] + X["kap"][:, :, 1]).sum(1),
        "w": w, "rho": X["rho"][:, ku], "y0": X["y0"][ku],
        "dk": X["dk"][ku], "dw": X["dw"][ku][:, :, :, ke],
    }


def program_size(prog):
    reals = {name: int(np.count_nonzero(prog[name])) for name in ("g", "sig", "beta", "w", "rho", "y0", "dk", "dw")}
    ints = {"p": 1, "S_E": len(prog["s_embed"]), "S_U": len(prog["s_read"]),
            "neuron_read_freqs": int(prog["read_mask"].sum()) if prog["read"] != "S_E" else 0}
    return {"reals": reals, "reals_total": sum(reals.values()), "ints": ints, "ints_total": sum(ints.values())}


def embed_stage(prog, a, b, shift=None):
    """Stage (i): token -> plane coordinates D(w_k x), k in S_E. ``shift=(T, s)`` is frequency_edit at pos0."""
    ks, p = prog["s_embed"], prog["p"]
    Da, Db = chars(ks, a, p), chars(ks, b, p)
    if shift is not None:
        sel = np.isin(ks, shift[0])
        Da[:, sel] = chars(ks[sel], a + shift[1], p)
    return Da, Db


def attention_stage(prog, Da, Db):
    """Stage (ii): routing alpha from the plane coordinates, and the moved content zeta_hk per head and plane."""
    g, sig = prog["g"], prog["sig"]
    alpha = softmax(np.stack([sig[:, 0] + np.einsum("hkt,nkt->nh", g, Da), sig[:, 1] + np.einsum("hkt,nkt->nh", g, Db),
                              np.broadcast_to(sig[:, 2], (len(Da), len(g)))], -1))
    return alpha, alpha[:, :, 0, None, None] * Da[:, None] + alpha[:, :, 1, None, None] * Db[:, None]


def mlp_stage(prog, zeta):
    """Stage (iii): pre_n = beta_n + sum_h w_nh . zeta_h,k(n); the ReLU is executed, not approximated."""
    pre = prog["beta"] + np.einsum("nhkt,mhkt->mn", prog["w"], zeta, optimize=True)
    return pre, np.maximum(pre, 0)


def unembed_stage(prog, act, alpha, zeta, uscale=None):
    """Stage (iv): readout coordinates y_k (k in S_U) and logit(c) = sum_k D(w_k c) . y_k."""
    y = prog["y0"] + np.einsum("nkt,mn->mkt", prog["rho"], act, optimize=True) \
        + np.einsum("kthj,mhj->mkt", prog["dk"], alpha) + np.einsum("kthjs,mhjs->mkt", prog["dw"], zeta, optimize=True)
    if uscale:
        y = y * np.array([uscale.get(int(k), 1.0) for k in prog["s_read"]])[None, :, None]
    return y


def readout(y, s_read, p):
    return np.einsum("mkt,ckt->mc", y, chars(s_read, np.arange(p), p))


def run_program(prog, a, b, shift=None, zeta_patch=None, uscale=None):
    Da, Db = embed_stage(prog, a, b, shift)
    alpha, zeta = attention_stage(prog, Da, Db)
    if zeta_patch is not None:
        zeta = zeta_patch(zeta)
    pre, act = mlp_stage(prog, zeta)
    y = unembed_stage(prog, act, alpha, zeta, uscale)
    return {"alpha": alpha, "zeta": zeta, "pre": pre, "y": y, "logits": readout(y, prog["s_read"], prog["p"])}


def closed_form(A, s_read, p, arg, scale=None):
    """T: y_k = A_k D(w_k arg_k); ``arg`` maps k to the integer the plane encodes (a + b unless edited)."""
    y = np.stack([A[i] * (scale or {}).get(int(k), 1.0) * chars([k], arg[int(k)] % p, p)[:, 0]
                  for i, k in enumerate(s_read)], 1)
    return readout(y, s_read, p)


# ------------------------------------------------------------------------------------------------ metrics

class Stats:
    """Streaming KL(ref || pred) and argmax agreement over rows."""

    def __init__(self):
        self.kl, self.agree = [], []

    def add(self, ref, pred):
        lp, lq = log_softmax(ref), log_softmax(pred)
        self.kl.append((np.exp(lp) * (lp - lq)).sum(-1))
        self.agree.append(ref.argmax(-1) == pred.argmax(-1))
        return self

    def summary(self, label=None):
        kl, ag = np.concatenate(self.kl), np.concatenate(self.agree)
        out = {"kl_mean": float(kl.mean()), "kl_q99": float(np.quantile(kl, 0.99)), "kl_max": float(kl.max()),
               "argmax_agree": float(ag.mean()), "rows": int(len(kl))}
        if label:
            print(label, json.dumps(out), flush=True)
        return out


def compare(ref, pred, label=None):
    return Stats().add(ref, pred).summary(label)


def y_of_logits(logits, ks, p):
    """The model's actual readout coordinates: the c-Fourier coefficients of its logits, (N, K, 2)."""
    planes = cyclic_planes(np.ascontiguousarray(logits.T), p)[1]                    # K,2,N
    return np.moveaxis(planes[np.asarray(ks) - 1], -1, 0)


def rel(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def ladder(X, a, b, M, prog):
    """Each restriction of P alone on top of R, then cumulatively."""
    K = X["K"]
    inS = np.isin(K, prog["s_embed"]).astype(float)
    inU = np.isin(K, prog["s_read"]).astype(float)
    own = np.zeros((len(prog["own"]), len(K)))
    own[np.arange(len(own)), np.searchsorted(K, prog["s_embed"])[prog["own"]]] = 1.0
    mine = np.zeros_like(own)
    mine[:, np.searchsorted(K, prog["s_embed"])] = prog["read_mask"]
    steps = {"scores_S_E": dict(att=inS), "read_S_E": dict(read=np.broadcast_to(inS, own.shape)),
             "read_own": dict(read=own), "read_P": dict(read=mine), "write_S_U": dict(write=np.broadcast_to(inU, own.shape)),
             "write_own_only": dict(write=own), "no_direct_path": dict(direct=False), "no_bias": dict(bias=False),
             "kappa_folded": dict(kappa=False)}
    for i, k in enumerate(prog["s_embed"]):
        drop = mine.copy()
        drop[prog["own"] != i, np.searchsorted(K, [k])[0]] = 0.0
        steps[f"read_P_without_cross_reads_of_{k}"] = dict(read=drop)
    out = {}
    for name, kw in steps.items():
        out["only_" + name] = compare(M["logits"], rewrite_logits(X, a, b, **kw)[0], f"LADDER only {name}")
    cum = {}
    for name in ("scores_S_E", "read_P", "write_S_U", "kappa_folded", "write_own_only"):
        cum.update(steps[name])
        out["cumulative_" + name] = compare(M["logits"], rewrite_logits(X, a, b, **cum)[0], f"LADDER cum {name}")
    return out


def dump(res, path):
    with open(path + ".partial", "w") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(str(k))}:{json.dumps(v, separators=(',', ':'))}"
                                    for k, v in res.items()) + "\n}\n")
    os.replace(path + ".partial", path)


# ------------------------------------------------------------------------ greedy structure selection (--select greedy)
#
# Replaces the 1% cutoffs of extract_program by structure selection over R. A program is R under a structure (which
# embedding planes S_E, readout planes S_U, per-head score and direct-path planes, per-neuron read planes, which heads
# route by their law, which heads' kap are folded) with every real on a declared dyadic lattice per real group. Each
# proposal is one structural edit of the current program; it is accepted iff the DECODED program is strictly shorter
# in bits AND meets the declared fidelity (decide_proposal). The only declared input is the tolerance eps of the
# contract: exhaustive max KL(model || P) <= eps over
#   clean     all p^2 inputs;
#   rotate_k  frequency_edit of plane k at pos0 by every shift s = 1..p-1, on every input, for every k = 1..(p-1)/2;
#   swap_k    plane k's attention-output content interchanged from the seeded donor permutation, every input, every k.
# A route-law head's routing is its mean pattern abar_h over the p^2 inputs under R (weight-derived); kap of a folded
# head enters beta at abar_h.

GROUPS = ("sig", "g", "abar", "beta", "kap", "w", "rho", "y0", "dk", "dw")
KINDS = ("coarsen", "drop_embed_plane", "drop_readout_plane", "route_law", "fold_kap", "drop_score_plane",
         "drop_direct_plane", "drop_read")


class Codec:
    """Code lengths and lattice codes from the Rust owners (codec.rs, precision.rs) through the surface's
    ``code_lengths``. A program's message is a sequence of items, so its length is the sum of Rust's item lengths;
    ``verify`` prices the whole message in one call. Rust's lengths are memoized by value: the signed-prefix length of
    each lattice index (a lattice message is count + 1 in the prefix code, its fraction bits and then each index in
    the signed prefix code: precision.rs LatticeCode::write), and each prefix-integer and subset length."""

    def __init__(self):
        self.items, self.groups = {}, {}

    def lengths(self, items):
        key = [json.dumps(it, sort_keys=True) for it in items]
        todo = [it for it, k in zip(items, key) if k not in self.items]
        if todo:
            got = mpd({"kind": "code_lengths", "items": todo}, {}).report["result"]["items"]
            self.items.update({json.dumps(it, sort_keys=True): g["bits"] for it, g in zip(todo, got)})
        return np.array([self.items[k] for k in key], dtype=np.int64)

    def prefix(self, v):
        return int(self.lengths([{"kind": "prefix_integer", "value": int(v)}])[0])

    def subset(self, n, k):
        return int(self.lengths([{"kind": "subset", "universe": int(n), "cardinality": int(k)}])[0])

    def signed(self, v):
        return int(self.lengths([{"kind": "signed_prefix_integer", "value": int(v)}])[0])

    def lattice(self, name, values, fraction_bits):
        """(decoded reals, each real's index length) of ``values`` on the lattice 2^-fraction_bits: Rust decodes;
        each index is decoded * 2^fraction_bits, priced by Rust's signed prefix code."""
        key = (name, int(fraction_bits), hash(values.tobytes()))
        if key not in self.groups:
            out = mpd({"kind": "code_lengths", "items": [{"kind": "lattice", "tensor": "x", "fraction_bits": int(fraction_bits)}]},
                      {"x": np.ascontiguousarray(values, dtype=np.float64)})
            dec = out.arrays[out.report["result"]["items"][0]["lattice"]["decoded"]]
            idx = np.rint(dec * 2.0 ** fraction_bits).astype(np.int64)
            u, inv = np.unique(idx, return_inverse=True)
            ub = self.lengths([{"kind": "signed_prefix_integer", "value": int(v)} for v in u])
            self.groups[key] = (dec, ub[inv].reshape(idx.shape))
        return self.groups[key]

    def verify(self, st, R, p, bits):
        """The whole message priced by Rust in one code_lengths call; it must equal the memoized sum ``bits``."""
        items, tensors = structure_items(st, p), {}
        for name in GROUPS:
            tensors[name] = np.ascontiguousarray(R[name][3][R[name][1]], dtype=np.float64)
            items.append({"kind": "lattice", "tensor": name, "fraction_bits": int(st["prec"][name])})
        total = mpd({"kind": "code_lengths", "items": items}, tensors).report["result"]["total_bits"]
        assert total == bits, (total, bits)
        return total


def start_structure(nK, H, n):
    """R itself: every plane, read, score plane and direct path; no head folded or routed by its law."""
    return {"SE": np.ones(nK, bool), "SU": np.ones(nK, bool), "G": np.ones((H, nK), bool), "DW": np.ones((H, nK), bool),
            "Rd": np.ones((n, nK), bool), "const": np.zeros(H, bool), "fold": np.zeros(H, bool), "prec": None}


def realize(X, st, fold_at, codec=None):
    """The reals P sends under ``st``: {group: (decoded full-shape array, presence mask, index lengths of the present
    reals, the reals before coding)}, each group on the lattice 2^-prec[group] through ``codec`` (exact reals when
    ``st["prec"]`` is None). Neurons with no read and every kap folded are constant and fold into y0; route-law heads
    fold kap into beta and dk into y0 at abar."""
    SE, SU, const = st["SE"], st["SU"], st["const"]
    fold, live = st["fold"] | const, ~const
    Rd = st["Rd"] & SE[None]
    vary = Rd.any(1) | (~fold).any()
    beta = X["beta"] + np.einsum("nhj,hj->n", X["kap"][:, fold], fold_at[fold])
    y0 = X["y0"] + np.einsum("n,nkt->kt", np.maximum(beta[~vary], 0), X["rho"][~vary]) \
        + np.einsum("kthj,hj->kt", X["dk"][:, :, const], fold_at[const])
    vals = {"sig": X["sig"], "g": X["g"], "abar": fold_at, "beta": beta, "kap": X["kap"], "w": X["w"], "rho": X["rho"],
            "y0": y0, "dk": X["dk"], "dw": X["dw"]}
    masks = {"sig": live[:, None], "g": (live[:, None] & st["G"] & SE[None])[:, :, None], "abar": const[:, None],
             "beta": vary, "kap": vary[:, None, None] & ~fold[None, :, None],
             "w": Rd[:, None, :, None], "rho": vary[:, None, None] & SU[None, :, None], "y0": SU[:, None],
             "dk": SU[:, None, None, None] & live[None, None, :, None],
             "dw": SU[:, None, None, None, None] & (st["DW"] & SE[None])[None, None, :, :, None]}
    out = {}
    for name in GROUPS:
        v = vals[name]
        m = np.broadcast_to(masks[name], v.shape)
        if st["prec"] is None:
            out[name] = (np.where(m, v, 0.0), m, None, v)
            continue
        dec, bits = codec.lattice(name, v, st["prec"][name])
        out[name] = (np.where(m, dec, 0.0), m, bits[m], v)
    return out


def structure_items(st, p):
    """The structure part of P's message as code_lengths items: the architecture integers (p, planes, heads, neurons)
    in the prefix code; S_E and S_U as subsets of the planes; per head a route-law flag and, if it routes, a kap-fold
    flag (fixed indices into 2) and its score planes as a subset of S_E; per head its direct-path planes and per
    neuron its read planes as subsets of S_E."""
    SE = st["SE"]
    nK, m, H = len(SE), int(SE.sum()), len(st["const"])
    items = [{"kind": "prefix_integer", "value": int(v)} for v in (p, nK, H, st["Rd"].shape[0])]
    items += [{"kind": "subset", "universe": nK, "cardinality": int(st[f].sum())} for f in ("SE", "SU")]
    for h in range(H):
        items.append({"kind": "fixed_index", "alphabet_size": 2})
        if not st["const"][h]:
            items += [{"kind": "fixed_index", "alphabet_size": 2},
                      {"kind": "subset", "universe": m, "cardinality": int((st["G"][h] & SE).sum())}]
        items.append({"kind": "subset", "universe": m, "cardinality": int((st["DW"][h] & SE).sum())})
    items += [{"kind": "subset", "universe": m, "cardinality": int(r)} for r in (st["Rd"] & SE[None]).sum(1)]
    return items


def code_length_bits(st, R, p, codec):
    """P's message length: Rust's length of every structure item plus, per real group, Rust's lattice message length
    (count + 1 prefix, fraction bits signed, each index signed)."""
    SE = st["SE"]
    m, H = int(SE.sum()), len(st["const"])
    sub = np.array([codec.subset(m, r) for r in range(m + 1)])
    bits = sum(codec.prefix(v) for v in (p, len(SE), H, st["Rd"].shape[0]))
    bits += codec.subset(len(SE), m) + codec.subset(len(SE), int(st["SU"].sum()))
    flag = int(codec.lengths([{"kind": "fixed_index", "alphabet_size": 2}])[0])
    bits += flag * (H + int((~st["const"]).sum())) + int(sub[(st["G"] & SE[None]).sum(1)][~st["const"]].sum())
    bits += int(sub[(st["DW"] & SE[None]).sum(1)].sum()) + int(sub[(st["Rd"] & SE[None]).sum(1)].sum())
    for name in GROUPS:
        bits += codec.prefix(len(R[name][2]) + 1) + codec.signed(st["prec"][name]) + int(R[name][2].sum())
    return bits


def evidence(fid, eps):
    """The contract evaluation as an EvidenceStatus on the wire: a counterexample row above eps, an exact exhaustive
    maximum, or (when any rotate plane was certified by the P-side bound) a uniform bound. numerical_error is the
    roundoff of the final KL sum given the computed log-probs (unit roundoff x (classes + 2) x max |log-prob|)."""
    err = fid["numerical_error"]
    if not fid["complete"]:
        return {"kind": "counterexample", "value": fid["max_kl"], "numerical_error": err, "threshold": eps,
                "witness": "/".join(fid["witness"])}
    if fid["bounded"]:
        return {"kind": "uniform_bound", "upper": fid["max_kl"], "numerical_error": err,
                "region": "clean + swap_k + rotate_k, every plane (rotate_k bounded via Kb, Mb where certified)"}
    return {"kind": "exact", "value": fid["max_kl"], "numerical_error": err, "basis": {"kind": "exhaustive",
            "cardinality": fid["rows"]}, "witness": None, "domain": "clean + swap_k + rotate_k, every plane"}


def decide_proposal(reference, candidate, eps):
    """fit.rs decide_proposal through the surface: ``reference`` and ``candidate`` are (bits, contract evaluation);
    every structural edit here is a Reduce. Returns (accepted, Rust's decision kind)."""
    (rb, rf), (cb, cf) = reference, candidate
    out = mpd({"kind": "decide_proposal", "proposal": "reduce", "tolerance": eps,
               "reference": {"bits": int(rb), "distortion": evidence(rf, eps)},
               "candidate": {"bits": int(cb), "distortion": evidence(cf, eps)}, "fidelity": evidence(cf, eps)}, {})
    kind = out.report["result"]["decision"]["kind"]
    return kind == "accepted", kind


def proposals(st, R):
    """Every single structural edit of ``st``, in KINDS order."""
    SE = st["SE"]
    out = [("coarsen", g) for g in GROUPS if R[g][1].any()]
    out += [("drop_embed_plane", int(k)) for k in np.flatnonzero(SE)]
    out += [("drop_readout_plane", int(k)) for k in np.flatnonzero(st["SU"])]
    for h in np.flatnonzero(~st["const"]):
        out.append(("route_law", int(h)))
        if not st["fold"][h]:
            out.append(("fold_kap", int(h)))
        out += [("drop_score_plane", (int(h), int(k))) for k in np.flatnonzero(st["G"][h] & SE)]
    out += [("drop_direct_plane", (int(h), int(k))) for h, k in zip(*np.nonzero(st["DW"] & SE[None]))]
    out += [("drop_read", (int(n), int(k))) for n, k in zip(*np.nonzero(st["Rd"] & SE[None]))]
    return out


def apply_proposal(st, prop):
    """The edited structure, or None when ``prop`` no longer applies to ``st``."""
    kind, key = prop
    new = {k: (v.copy() if isinstance(v, (np.ndarray, dict)) else v) for k, v in st.items()}
    if kind == "coarsen":
        new["prec"][key] -= 1
        return new
    field = {"drop_embed_plane": "SE", "drop_readout_plane": "SU", "drop_score_plane": "G", "drop_direct_plane": "DW",
             "drop_read": "Rd", "route_law": "const", "fold_kap": "fold"}[kind]
    if kind in ("route_law", "fold_kap"):
        if st["const"][key] or st[field][key]:
            return None
        new[field][key] = True
        return new
    if not st[field][key] or (isinstance(key, tuple) and not st["SE"][key[1]]):
        return None
    new[field][key] = False
    return new


# ---- torch executors (the contract runs on a device; the greedy screens in float32 on mps, the certificate is float64)

def torch_model(W, cfg, dev, dt):
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev, dtype=dt)                  # noqa: E731
    p, dh = cfg["p"], cfg["d_head"]
    x2 = W["W_E"][p] + W["W_pos"][2]
    qk = np.einsum("hk,hkd->hd", np.einsum("hkd,d->hk", W["W_Q"], x2), W["W_K"]) / math.sqrt(dh)
    return {"E": t(W["W_E"][:p]), "pos0": t(W["W_pos"][0]), "pos1": t(W["W_pos"][1]), "x2": t(x2), "qk": t(qk),
            "W_V": t(W["W_V"]), "W_O": t(W["W_O"]), "W_in": t(W["W_in"]), "b_in": t(W["b_in"]),
            "W_out": t(W["W_out"]), "b_out": t(W["b_out"]), "W_U": t(W["W_U"])}


def torch_model_logits(Mt, a, b, d0=None):
    """The model at ``=``; ``d0`` is added to the pos0 embedding (a frequency_edit)."""
    x0 = Mt["E"][a] + Mt["pos0"] if d0 is None else Mt["E"][a] + Mt["pos0"] + d0
    xs = torch.stack([x0, Mt["E"][b] + Mt["pos1"], Mt["x2"].expand_as(x0)], 1)
    alpha = torch.softmax(xs @ Mt["qk"].T, 1)                                                    # N,3,H
    z = torch.einsum("njh,njhe->nhe", alpha, torch.einsum("njd,hed->njhe", xs, Mt["W_V"])).reshape(len(a), -1)
    mid = Mt["x2"] + z @ Mt["W_O"].T
    return (mid + torch.relu(mid @ Mt["W_in"].T + Mt["b_in"]) @ Mt["W_out"].T + Mt["b_out"]) @ Mt["W_U"].T


def torch_program(R, st, T, dev, dt):
    """P's decoded reals on the device, restricted to S_E, S_U and the varying neurons. ``T`` = D(w_k x), (p, K, 2)."""
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev, dtype=dt)                  # noqa: E731
    se, su, nv = np.flatnonzero(st["SE"]), np.flatnonzero(st["SU"]), np.flatnonzero(R["beta"][1])
    d = {k: v[0] for k, v in R.items()}
    u, p = len(su), len(T)
    return {"se": se, "T": t(T[:, se]), "sig": t(d["sig"]), "g": t(d["g"][:, se]), "abar": t(d["abar"]),
            "const": torch.as_tensor(st["const"], device=dev), "beta": t(d["beta"][nv]),
            "kap": t(d["kap"][nv].reshape(len(nv), -1).T), "w": t(d["w"][nv][:, :, se].reshape(len(nv), -1).T),
            "rho": t(d["rho"][nv][:, su].reshape(len(nv), -1)), "y0": t(d["y0"][su].reshape(-1)),
            "dk": t(d["dk"][su].reshape(2 * u, -1).T), "dw": t(d["dw"][su][:, :, :, se].reshape(2 * u, -1).T),
            "Dc": t(T[:, su].reshape(p, -1).T)}


def torch_program_state(tp, a, b, rot=None, swap=None):
    """P's routing and moved content on inputs (a, b), flattened: (alpha (N, 3H), zeta (N, 2Hm)). ``rot = (j, a2)``:
    plane se[j] of operand a reads a2; ``swap = (j, donor)``: plane se[j]'s moved content zeta is the donor row's."""
    Da, Db = tp["T"][a], tp["T"][b]
    if rot is not None:
        Da = Da.clone()
        Da[:, rot[0]] = tp["T"][rot[1], rot[0]]
    sc = torch.stack([tp["sig"][:, 0] + torch.einsum("hkt,nkt->nh", tp["g"], Da),
                      tp["sig"][:, 1] + torch.einsum("hkt,nkt->nh", tp["g"], Db),
                      tp["sig"][:, 2].expand(len(a), -1)], -1)
    alpha = torch.where(tp["const"][None, :, None], tp["abar"][None], torch.softmax(sc, -1))
    zeta = alpha[..., 0, None, None] * Da[:, None] + alpha[..., 1, None, None] * Db[:, None]
    if swap is not None:
        zeta = zeta.clone()
        zeta[:, :, swap[0]] = zeta[swap[1]][:, :, swap[0]]
    return alpha.reshape(len(a), -1), zeta.reshape(len(a), -1)


def torch_program_logits(tp, a, b, rot=None, swap=None):
    """P on inputs (a, b) (see torch_program_state)."""
    af, zf = torch_program_state(tp, a, b, rot, swap)
    act = torch.relu(tp["beta"] + af @ tp["kap"] + zf @ tp["w"])
    return (tp["y0"] + act @ tp["rho"] + af @ tp["dk"] + zf @ tp["dw"]) @ tp["Dc"]


def torch_program_tables(tp):
    """P's sums regrouped by operand token (exact): per plane j and token x, the score g_h . D_j(x), the neuron read
    w_nh . D_j(x) and the direct-path logit contribution dw_h . D_j(x) through D(w_k c); their sums over S_E; and the
    readouts of act, alpha and y0 through D(w_k c). Cached on ``tp``."""
    if "tab" not in tp:
        H, (p, m, _), nv = len(tp["sig"]), tp["T"].shape, tp["beta"].shape[0]
        Wp = torch.einsum("nhkt,xkt->xkhn", tp["w"].T.reshape(nv, H, m, 2), tp["T"])            # p,m,H,nv
        Ep = torch.einsum("hktc,xkt->xkhc", (tp["dw"] @ tp["Dc"]).reshape(H, m, 2, p), tp["T"])  # p,m,H,p
        Gp = torch.einsum("hkt,xkt->xkh", tp["g"], tp["T"])                                     # p,m,H
        tp["tab"] = {"A": Wp.sum(1), "Wp": Wp, "E": Ep.sum(1), "Ep": Ep, "G": Gp.sum(1), "Gp": Gp,
                     "R": tp["rho"] @ tp["Dc"], "DK": tp["dk"] @ tp["Dc"], "Y0": tp["y0"] @ tp["Dc"]}
    return tp["tab"]


def torch_program_rot_logits(tp, j, s):
    """torch_program_logits with plane se[j] of operand a turned by each shift in ``s``, on every input (a, b): rows
    ordered (s, a, b). Regrouped by token (torch_program_tables) so a row's neuron reads cost O(H n), not O(H m n)."""
    tab = torch_program_tables(tp)
    p, S, H = tp["T"].shape[0], len(s), len(tp["sig"])
    x = torch.arange(p, device=s.device)
    a2 = (x[None] + s[:, None]) % p                                                             # S,p

    def turned(full, part):
        return full[None] - part[:, j][None] + part[a2, j]

    sc = torch.stack(torch.broadcast_tensors((tp["sig"][:, 0] + turned(tab["G"], tab["Gp"]))[:, :, None],
                                             (tp["sig"][:, 1] + tab["G"])[None, None],
                                             tp["sig"][:, 2].expand(1, 1, 1, H)), -1)            # S,a,b,H,3
    alpha = torch.where(tp["const"][:, None], tp["abar"], torch.softmax(sc, -1))
    af = alpha.reshape(S, p, p, 3 * H)

    def by_b(al, table):
        return (al.permute(2, 0, 1, 3).reshape(p, S * p, H) @ table).reshape(p, S, p, -1).permute(1, 2, 0, 3)

    act = torch.relu(tp["beta"] + af @ tp["kap"] + alpha[..., 0] @ turned(tab["A"], tab["Wp"]) + by_b(alpha[..., 1], tab["A"]))
    z = act @ tab["R"] + af @ tab["DK"] + tab["Y0"] + alpha[..., 0] @ turned(tab["E"], tab["Ep"]) + by_b(alpha[..., 1], tab["E"])
    return z.reshape(S * p * p, p)


def read_drop_delta(tp, tc, st, cand, prop, R, Rc):
    """When ``cand`` differs from ``st`` only in neuron n's read weights (a drop_read that leaves n varying and every
    other decoded real unchanged), the one changed column of P: {v, w_ref, dw, beta, kap, range}, where
    range = max_c r_n(c) - min_c r_n(c) for n's readout r_n(c) = sum_k rho_nk . D(w_k c). Otherwise None."""
    if prop[0] != "drop_read":
        return None
    n = prop[1][0]
    if not Rc["beta"][1][n] or any(not np.array_equal(R[g][0], Rc[g][0]) for g in GROUPS if g != "w"):
        return None
    diff = np.flatnonzero((R["w"][0] != Rc["w"][0]).reshape(len(R["w"][0]), -1).any(1))
    if len(diff) > 1 or (len(diff) == 1 and diff[0] != n):
        return None
    v = int(np.searchsorted(np.flatnonzero(R["beta"][1]), n))
    r = tp["rho"][v] @ tp["Dc"]
    return {"v": v, "w_ref": tp["w"][:, v], "dw": tc["w"][:, v] - tp["w"][:, v], "beta": tp["beta"][v],
            "kap": tp["kap"][:, v], "range": float(r.max() - r.min())}


def kl_rows(lm, logits_q):
    """KL(m || q) per row and the largest |log-prob| entering the sums (a 0-d tensor)."""
    lq = torch.log_softmax(logits_q, -1)
    return (lm.exp() * (lm - lq)).sum(-1), torch.maximum(lm.abs().amax(), lq.abs().amax())


class Contract:
    """The declared contract on one device. Model-side references: clean and swap_k log-probs are precomputed; rotate_k
    log-probs are recomputed per shift chunk (cached by the float32 screen for up to six planes). For every rotate_k row,
    KL(m_s || q_s) = KL(m_s || m_0) + sum_c m_s,c (log m_0,c - log q_s,c) <= Kb_k + sum_c Mb_k,c (log m_0,c - log q_s,c)^+
    with Kb_k = max_s KL(m_s || m_0) and Mb_k = max_s m_s (exact maxima over all shifts, precomputed): a certified upper
    bound that needs only P. A plane whose bound exceeds eps is evaluated exactly against the model's rotated outputs.

    A drop_read candidate differs from the reference only in one neuron's pre-activation, so its logits move by
    Delta_c = (act' - act) r_n(c), and KL(m || q') - KL(m || q) = sum_c m_c (-Delta_c) + LSE(z + Delta) - LSE(z)
    <= max Delta - min Delta = |act' - act| range(r_n): each rotate_k row's certified upper bound U (exact KL or the bound
    above, kept from the reference's evaluation) moves by at most that. Every shift chunk where the moved bound exceeds
    eps is evaluated exactly."""

    def __init__(self, W, cfg, X, donor, swap_lp, bound, deltas, dev, dt, shifts):
        self.p, self.dev, self.dt = cfg["p"], dev, dt
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev, dtype=dt)             # noqa: E731
        grid = np.arange(self.p ** 2)
        self.a, self.b = torch.as_tensor(grid // self.p, device=dev), torch.as_tensor(grid % self.p, device=dev)
        self.donor = torch.as_tensor(donor, device=dev) if donor is not None else None
        self.Mt = torch_model(W, cfg, dev, dt)
        self.T, self.D = t(X["T"]), t(deltas)
        self.lm0 = torch.log_softmax(torch_model_logits(self.Mt, self.a, self.b), -1)
        self.swap_lp = t(swap_lp) if swap_lp is not None else None
        self.Kb, self.Mb = (t(bound[0]), t(bound[1])) if bound is not None else (None, None)
        self.nK = X["U"].shape[0]
        self.shifts = np.asarray(shifts)
        self.chunk = max(1, (1 << 17) // self.p ** 2)
        self.front, self.hot, self.cache = [], set(), {}                  # early-exit order, bound-failed planes, lm

    def rot_chunks(self, k):
        """(chunk index, a, b, a2, shift index) over every input and shift of rotate_k, in shift chunks."""
        N = self.p ** 2
        for c, i in enumerate(range(0, len(self.shifts), self.chunk)):
            s = torch.as_tensor(self.shifts[i:i + self.chunk], device=self.dev)
            a, b = self.a.repeat(len(s)), self.b.repeat(len(s))
            yield c, a, b, (a + s.repeat_interleave(N)) % self.p, torch.arange(i, i + len(s), device=self.dev).repeat_interleave(N)

    def rot_model(self, k, c, a, b, si):
        """The model's log-probs on shift chunk c of rotate_k."""
        got = self.cache.get(k, {}).get(c)
        if got is not None:
            return got
        lm = torch.log_softmax(torch_model_logits(self.Mt, a, b, self.D[k][si, a]), -1)
        if self.dt == torch.float32 and (k in self.cache or len(self.cache) < 6):
            self.cache.setdefault(k, {})[c] = lm
        return lm

    def evaluate(self, tp, eps, stats=False, exact_planes=(), inc=None, U=None):
        """max KL over the contract with early exit above eps. ``stats``: every family in full, with mean KL and
        argmax agreement per family; planes in ``exact_planes`` are evaluated exactly, never by a bound. ``inc`` (a
        read_drop_delta) with ``U`` (the reference's per-row rotate bounds) moves the reference's bounds instead of
        re-evaluating P. A complete evaluation returns its per-row rotate bounds for the planes in S_E as "U"."""
        se = list(tp["se"])
        q0 = torch_program_logits(tp, self.a, self.b)
        lq0 = torch.log_softmax(q0, -1)
        units = [("clean",), ("swap_out",)] + [("swap", k) for k in se] + [("rot_out",)] \
            + [("rot", k) for k in range(self.nK) if k in se or k in self.hot or k in exact_planes]
        units = [u for u in self.front if u in units] + [u for u in units if u not in self.front]
        worst, fam, tally, newU = 0.0, {}, {"rows": 0, "lmax": torch.zeros((), device=self.dev), "bounded": False}, {}

        def note(name, ref, pred):
            nonlocal worst
            kl, lmax = kl_rows(ref, pred)
            v = float(kl.max())
            worst = max(worst, v)
            tally["rows"] += len(kl)
            tally["lmax"] = torch.maximum(tally["lmax"], lmax)
            if stats:
                f = fam.setdefault(name, {"kl_max": 0.0, "kl_sum": 0.0, "agree": 0, "rows": 0})
                f["kl_max"] = max(f["kl_max"], v)
                f["kl_sum"] += float(kl.sum())
                f["rows"] += len(kl)
                f["agree"] += int((ref.argmax(-1) == pred.argmax(-1)).sum())
            return kl

        def bounded(v):
            nonlocal worst
            worst = max(worst, v)
            tally["bounded"] = True

        def rotated(k, a, b, a2):
            if k not in se:
                return q0.repeat(len(a) // len(q0), 1)
            return torch_program_rot_logits(tp, se.index(k), ((a2 - a) % self.p)[::len(q0)])

        def exact_plane(k):
            rows = []
            for c, a, b, a2, si in self.rot_chunks(k):
                kl = note(f"rotate_k{k + 1}", self.rot_model(k, c, a, b, si), rotated(k, a, b, a2))
                rows.append(kl)
                if float(kl.max()) > eps and not stats:
                    return None
            return torch.cat(rows)

        for u in units:
            hit = False
            if u == ("clean",):
                hit = float(note("clean", self.lm0, q0).max()) > eps
            elif u == ("swap_out",):
                out_k = [k for k in range(self.nK) if k not in se]
                if not stats and out_k:
                    hit = float(note("swap_other", self.swap_lp[out_k].reshape(-1, self.p), q0.repeat(len(out_k), 1)).max()) > eps
                for k in out_k if stats else ():
                    name = f"swap_k{k + 1}" if k in exact_planes else "swap_other"
                    hit = float(note(name, self.swap_lp[k], q0).max()) > eps or hit
            elif u == ("rot_out",):
                out_k = [k for k in range(self.nK) if k not in se and k not in self.hot and k not in exact_planes]
                f = (self.lm0 - lq0).clamp(min=0)
                bnd = (self.Kb[out_k] + (self.Mb[out_k] * f).sum(-1)).amax(1).tolist() if out_k else []
                for k, v in zip(out_k, bnd):
                    if v <= eps:
                        bounded(v)
                        if stats:
                            g = fam.setdefault("rotate_bounded", {"bound_max": 0.0, "planes": []})
                            g["bound_max"] = max(g["bound_max"], v)
                            g["planes"].append(int(k + 1))
                        continue
                    self.hot.add(k)
                    hit = exact_plane(k) is None or hit
                    if hit and not stats:
                        break
            elif u[0] == "swap":
                pred = torch_program_logits(tp, self.a, self.b, swap=(se.index(u[1]), self.donor))
                hit = float(note(f"swap_k{u[1] + 1}", self.swap_lp[u[1]], pred).max()) > eps
            elif inc is not None and u[1] in se and u[1] not in exact_planes:
                k, rows, chunks = u[1], [], []
                j, Uk, off = se.index(u[1]), U[u[1]], 0
                for c, a, b, a2, si in self.rot_chunks(k):
                    af, zf = torch_program_state(tp, a, b, rot=(j, a2))
                    pre = inc["beta"] + af @ inc["kap"] + zf @ inc["w_ref"]
                    rows.append(Uk[off:off + len(a)] + (torch.relu(pre + zf @ inc["dw"]) - torch.relu(pre)).abs() * inc["range"])
                    chunks.append((c, a, b, a2, si))
                    off += len(a)
                for i, v in enumerate(torch.stack([r.max() for r in rows]).tolist()):
                    if v <= eps:
                        bounded(v)
                        continue
                    c, a, b, a2, si = chunks[i]
                    rows[i] = note(f"rotate_k{k + 1}", self.rot_model(k, c, a, b, si), rotated(k, a, b, a2))
                    if float(rows[i].max()) > eps:
                        hit = True
                        break
                if not hit:
                    newU[k] = torch.cat(rows)
            else:
                k = u[1]
                if k in se and k not in self.hot and k not in exact_planes:
                    bnd, rows = 0.0, []
                    for c, a, b, a2, si in self.rot_chunks(k):
                        S = len(a) // len(q0)
                        f = (self.lm0.repeat(S, 1) - torch.log_softmax(rotated(k, a, b, a2), -1)).clamp(min=0)
                        rows.append(self.Kb[k].repeat(S) + (self.Mb[k].repeat(S, 1) * f).sum(-1))
                        bnd = max(bnd, float(rows[-1].max()))
                        if bnd > eps:
                            break
                    if bnd <= eps:
                        bounded(bnd)
                        newU[k] = torch.cat(rows)
                        if stats:
                            f = fam.setdefault("rotate_bounded", {"bound_max": 0.0, "planes": []})
                            f["bound_max"] = max(f["bound_max"], bnd)
                            f["planes"].append(int(k + 1))
                        continue
                    self.hot.add(k)
                rows = exact_plane(k)
                hit = rows is None
                if not hit and k in se:
                    newU[k] = rows
            if hit and not stats:
                err = torch.finfo(self.dt).eps / 2 * (self.p + 2) * float(tally["lmax"])
                self.front = [u] + [v for v in self.front if v != u]
                return {"max_kl": worst, "complete": False, "witness": list(map(str, u)), "numerical_error": err,
                        "bounded": tally["bounded"], "rows": tally["rows"]}
        out = {"max_kl": worst, "complete": True,
               "numerical_error": torch.finfo(self.dt).eps / 2 * (self.p + 2) * float(tally["lmax"]),
               "bounded": tally["bounded"], "rows": tally["rows"]}
        if stats:
            out["families"] = {k: ({"kl_max": v["kl_max"], "kl_mean": v["kl_sum"] / v["rows"],
                                    "argmax_agree": v["agree"] / v["rows"], "rows": v["rows"]} if "rows" in v else v)
                               for k, v in fam.items()}
        else:
            out["U"] = newU
        return out


def edit_deltas(W, p, nK, shifts, path):
    """cyclic_action::frequency_edit (through the surface) of each single plane k at every shift: the pos0 table's row
    changes, (K, shifts, p, d); cached at ``path``."""
    if os.path.exists(path):
        return np.load(path)
    E = W["W_E"]
    out = np.stack([np.stack([frequency_edited(E, p, [k], s)[:p] - E[:p] for s in shifts]) for k in range(1, nK + 1)])
    np.save(path, out)
    return out


def rotation_bound(W, cfg, X, deltas, shifts, path):
    """Kb_k(a, b) = max_s KL(m_s || m_0) and Mb_k(a, b, c) = max_s m_s(c) over rotate_k, every plane, float64 CPU."""
    if os.path.exists(path):
        z = np.load(path)
        return z["Kb"], z["Mb"]
    c = Contract(W, cfg, X, None, None, None, deltas, "cpu", torch.float64, shifts)
    lm0, nK, N = c.lm0, X["U"].shape[0], c.p ** 2
    Kb, Mb = np.zeros((nK, N)), np.zeros((nK, N, c.p))
    with torch.no_grad():
        for k in range(nK):
            for ci, a, b, _, si in c.rot_chunks(k):
                lm = c.rot_model(k, ci, a, b, si).reshape(-1, N, c.p)
                Kb[k] = np.maximum(Kb[k], (lm.exp() * (lm - lm0)).sum(-1).amax(0).numpy())
                Mb[k] = np.maximum(Mb[k], lm.exp().amax(0).numpy())
            print(f"BOUND plane {k + 1} Kb_max={Kb[k].max():.3g}", flush=True)
    np.savez(path, Kb=Kb, Mb=Mb)
    return Kb, Mb


def swap_logprobs(W, cfg, X, a, b, donor):
    """Model log-probs under swap_k for every plane k (float64, the landed attn_out_interchange edit)."""
    M = model_forward(W, cfg, a, b)
    out = []
    for k in range(X["U"].shape[0]):
        Dk = np.stack([chars([k + 1], a, cfg["p"])[:, 0], chars([k + 1], b, cfg["p"])[:, 0]], 1)     # N,2,2
        zM = np.einsum("nhj,njt->nht", M["alpha"][:, :, :2], Dk)
        OU = np.einsum("hde,te->htd", X["O"], X["U"][k])
        patch = np.einsum("nht,htd->nd", zM[donor] - zM, OU)
        out.append(log_softmax(model_forward(W, cfg, a, b, attn_patch=lambda _al: patch)["logits"]))
    return np.stack(out)


def derive_precision(X, fold_at, eps, a, b):
    """Fraction bits per real group of R, derived from eps. A real on the lattice 2^-p moves by at most 2^-(p+1), so
    group G moves logit c by at most 2^-(p_G+1) S_G(x, c), S_G = sum over G's reals of |d logit_c / d real| (first
    order, hand-derived through R's stages, on the clean inputs). The logit change then has range over c at most
    2 sum_G 2^-(p_G+1) max S_G, and Hoeffding's lemma bounds KL(softmax(z) || softmax(z + d)) by range(d)^2 / 8, so
    sum_G 2^-(p_G+1) max S_G <= sqrt(2 eps) meets eps to first order; each of the len(GROUPS) groups takes an equal
    share. abar is scored as if every head routed by its law."""
    p, H = X["p"], X["sig"].shape[0]
    T = X["T"]
    C1 = np.abs(T).sum((1, 2))                                                                  # p
    r = np.einsum("nkt,ckt->nc", X["rho"], T)                                                    # n,p
    S = {g: 0.0 for g in GROUPS}
    for i in range(0, len(a), 64):
        Da, Db = T[a[i:i + 64]], T[b[i:i + 64]]
        B = len(Da)
        alpha = softmax(np.stack([X["sig"][:, 0] + np.einsum("hkt,nkt->nh", X["g"], Da),
                                  X["sig"][:, 1] + np.einsum("hkt,nkt->nh", X["g"], Db),
                                  np.broadcast_to(X["sig"][:, 2], (B, H))], -1))
        Dx = np.stack([Da, Db, np.zeros_like(Da)], 1)                                            # B,3,K,2
        zeta = np.einsum("bhj,bjkt->bhkt", alpha[:, :, :2], Dx[:, :2])
        pre = X["beta"] + np.einsum("nhj,bhj->bn", X["kap"], alpha) + np.einsum("nhkt,bhkt->bn", X["w"], zeta, optimize=True)
        act = np.maximum(pre, 0)
        gpre = (pre > 0)[:, :, None] * r[None]                                                   # B,n,p
        sb = np.abs(gpre).sum(1)
        za = np.abs(zeta).sum((1, 2, 3))[:, None]
        Q = X["kap"][None] + np.einsum("nhkt,bjkt->bnhj", X["w"], Dx, optimize=True)
        Rj = X["dk"][None] + np.einsum("kthls,bjls->bkthj", X["dw"], Dx, optimize=True)
        galpha = np.einsum("bnc,bnhj->bhjc", gpre, Q, optimize=True) + np.einsum("ckt,bkthj->bhjc", T, Rj, optimize=True)
        gs = alpha[..., None] * (galpha - (alpha[..., None] * galpha).sum(2, keepdims=True))
        cand = {"sig": np.abs(gs).sum((1, 2)),
                "g": np.abs(np.einsum("bhjc,bjkt->bhktc", gs[:, :, :2], Dx[:, :2])).sum((1, 2, 3)),
                "abar": np.abs(galpha).sum((1, 2)), "beta": sb, "kap": sb * H, "w": sb * za,
                "rho": act.sum(1)[:, None] * C1, "y0": C1[None], "dk": H * C1[None], "dw": za * C1}
        for g in GROUPS:
            S[g] = max(S[g], float(cand[g].max()))
    prec = {g: int(math.ceil(math.log2(len(GROUPS) * S[g] / math.sqrt(2 * eps)))) - 1 for g in GROUPS}
    return prec, S


def structure_summary(st, R, X, own_k):
    """What P keeps, in plane numbers k = 1..(p-1)/2."""
    K = X["K"]
    SE = st["SE"]
    Rd = st["Rd"] & SE[None]
    vary = R["beta"][1]
    reads = {}
    for n in np.flatnonzero(vary):
        for k in K[Rd[n]]:
            key = f"{own_k[n]}->{k}"
            reads[key] = reads.get(key, 0) + 1
    return {"S_E": K[SE].tolist(), "S_U": K[st["SU"]].tolist(),
            "score_planes": {h: K[st["G"][h] & SE].tolist() for h in range(len(st["const"])) if not st["const"][h]},
            "route_law_heads": np.flatnonzero(st["const"]).tolist(),
            "kap_folded_heads": np.flatnonzero(st["fold"] | st["const"]).tolist(),
            "direct_path_planes": {h: K[st["DW"][h] & SE].tolist() for h in range(len(st["const"]))},
            "varying_neurons": int(vary.sum()), "read_pairs": int(Rd[vary].sum()),
            "read_pairs_by_own_plane": dict(sorted(reads.items(), key=lambda kv: -kv[1])),
            "reals": {g: int(R[g][1].sum()) for g in GROUPS}, "reals_total": int(sum(R[g][1].sum() for g in GROUPS)),
            "nonzero_reals_total": int(sum(np.count_nonzero(R[g][0]) for g in GROUPS)), "precision": st["prec"]}


def greedy_select(X, st, fold_at, contract, eps, prog_args, codec):
    """Passes over the proposals, each pass ordered by bits saved against the program at its start (then KINDS order
    and key, deterministic); every proposal is re-scored against the current program and decided by decide_proposal.
    Ends at the first pass with no acceptance: no single proposal is then both shorter and within eps. A candidate that
    is not shorter is refused without evaluating it (decide_proposal's NoShorterCode needs no fidelity to fail)."""
    p, T, dev, dt = prog_args
    R = realize(X, st, fold_at, codec)
    bits = codec.verify(st, R, p, code_length_bits(st, R, p, codec))
    tp = torch_program(R, st, T, dev, dt)
    fid = contract.evaluate(tp, eps)
    trail, passes = [], []
    t0 = time.time()
    while True:
        scored = []
        for prop in proposals(st, R):
            cand = apply_proposal(st, prop)
            scored.append((bits - code_length_bits(cand, realize(X, cand, fold_at, codec), p, codec), KINDS.index(prop[0]),
                           str(prop[1]), prop))
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))
        why = {}
        n_acc = 0
        for _, _, _, prop in scored:
            cand = apply_proposal(st, prop)
            if cand is None:
                why["NoLongerApplies"] = why.get("NoLongerApplies", 0) + 1
                continue
            Rc = realize(X, cand, fold_at, codec)
            cb = code_length_bits(cand, Rc, p, codec)
            if cb >= bits:
                verdict = "no_shorter_code_not_evaluated"
            else:
                tc = torch_program(Rc, cand, T, dev, dt)
                inc = read_drop_delta(tp, tc, st, cand, prop, R, Rc)
                cf = contract.evaluate(tc, eps, inc=inc, U=fid["U"] if inc is not None else None)
                ok, verdict = decide_proposal((bits, fid), (cb, cf), eps)
                if ok:
                    tp = tc
                    codec.verify(cand, Rc, p, cb)
                    trail.append([prop[0], prop[1] if isinstance(prop[1], str) else np.asarray(prop[1]).tolist(),
                                  bits - cb, cf["max_kl"]])
                    st, R, bits, fid = cand, Rc, cb, cf
                    n_acc += 1
                    if prop[0] != "drop_read":
                        print(f"  ACCEPT {trail[-1]} bits={bits} {time.time() - t0:.0f}s", flush=True)
            why[verdict] = why.get(verdict, 0) + 1
            if sum(why.values()) % 500 == 0:
                print(f"  {sum(why.values())}/{len(scored)} accepted={n_acc} bits={bits} m={int(st['SE'].sum())} "
                      f"u={int(st['SU'].sum())} {time.time() - t0:.0f}s", flush=True)
        passes.append({"proposals": len(scored), "accepted": n_acc, "decisions": why, "bits": bits,
                       "seconds": time.time() - t0})
        print(f"PASS {len(passes)} eps={eps:g} {json.dumps(passes[-1])} m={int(st['SE'].sum())} u={int(st['SU'].sum())}",
              flush=True)
        if n_acc == 0:
            return st, R, bits, fid, trail, passes


def share_frequencies(W, X, p, embed_share, unembed_share):
    """The share path's weight-spectrum frequency sets: W_E and W_U frequencies above their power-share cutoffs."""
    def shares(T):
        pw = cyclic_planes(T, p)[2]
        return pw / pw.sum()
    se, su = shares(W["W_E"][:p]), shares(W["W_U"])
    return ([int(k) for k in X["K"][np.argsort(-se)] if se[k - 1] > embed_share],
            [int(k) for k in X["K"][np.argsort(-su)] if su[k - 1] > unembed_share], se, su)


def greedy_main(args, cfg, step, W, res):
    p = cfg["p"]
    grid = np.arange(p * p)
    a, b = grid // p, grid % p
    shifts = args.shifts or list(range(1, p))
    X = exact_coordinates(W, cfg)
    X["T"] = chars(X["K"], np.arange(p), p)
    M = model_forward(W, cfg, a, b)
    _, R_alpha, _ = rewrite_logits(X, a, b)
    fold_at = R_alpha.mean(0)                                                                    # abar_h, H,3
    donor = np.random.default_rng(args.seed).permutation(p * p)
    eps = args.greedy_epsilon
    res.update({"select": "greedy", "declared": {"epsilon": eps, "contract": "exhaustive max KL(model || P) over clean, "
                "rotate_k (every plane k, every shift) and swap_k (every plane k, seeded donor permutation)",
                "code": "Rust code_lengths (codec.rs, precision.rs) and decide_proposal (fit.rs) via the MPD surface",
                "screen": f"{args.device} float32; certificate float64 cpu"},
                "abar": fold_at.tolist(), "shifts": len(shifts)})
    t0 = time.time()
    swap_lp = swap_logprobs(W, cfg, X, a, b, donor)
    cache = os.path.join(os.path.dirname(os.path.abspath(args.run)), os.path.basename(args.run))
    deltas = edit_deltas(W, p, len(X["K"]), shifts, cache + f".frequency_edits_{len(shifts)}.npy")
    bound = rotation_bound(W, cfg, X, deltas, shifts, cache + f".rotation_bound_{len(shifts)}.npz")
    print(f"CACHE swap + rotation bound {time.time() - t0:.0f}s", flush=True)
    dev, dt = args.device, torch.float32
    cert = Contract(W, cfg, X, donor, swap_lp, bound, deltas, "cpu", torch.float64, shifts)
    screen = Contract(W, cfg, X, donor, swap_lp, bound, deltas, dev, dt, shifts)
    codec = Codec()
    H, n = X["sig"].shape[0], X["beta"].shape[0]
    own_k = X["K"][np.einsum("nhkt->nk", X["w"] ** 2).argmax(1)]

    # ---- the landed 1% program, on the same contract (exact reals, kap folded at (1/2, 1/2, 0))
    se_share, su_share = share_frequencies(W, X, p, args.embed_share, args.unembed_share)[:2]
    with torch.no_grad():
        if se_share:
            prog = extract_program(X, se_share, su_share, "frac", args.read_frac)
            st1 = start_structure(len(X["K"]), H, n)
            st1["SE"] = np.isin(X["K"], se_share)
            st1["SU"] = np.isin(X["K"], su_share)
            st1["G"][:] = st1["SE"]
            st1["DW"][:] = st1["SE"]
            st1["Rd"][:] = False
            st1["Rd"][:, np.searchsorted(X["K"], se_share)] = prog["read_mask"] > 0
            st1["fold"][:] = True
            half = np.tile([0.5, 0.5, 0.0], (H, 1))
            R1 = realize(X, st1, half)
            tp1 = torch_program(R1, st1, X["T"], "cpu", torch.float64)
            same = float(np.abs(torch_program_logits(tp1, cert.a, cert.b).numpy() - run_program(prog, a, b)["logits"]).max())
            c1 = cert.evaluate(tp1, math.inf, stats=True, exact_planes=np.searchsorted(X["K"], se_share).tolist())
            res["share_program_on_contract"] = {"emulation_vs_run_program_max_abs_logit": same,
                                                **structure_summary(st1, R1, X, own_k), "certificate_float64": c1}
            print("SHARE", json.dumps(c1), flush=True)

        # ---- the greedy program
        st = start_structure(len(X["K"]), H, n)
        st["prec"], S = derive_precision(X, fold_at, eps, a, b)
        res["precision_derivation"] = {"S_G_max": S, "start_fraction_bits": dict(st["prec"])}
        while True:
            R0 = realize(X, st, fold_at, codec)
            f0 = screen.evaluate(torch_program(R0, st, X["T"], dev, dt), eps)
            print("START", st["prec"], f0["max_kl"], f0["complete"], flush=True)
            if f0["complete"] and f0["max_kl"] <= eps:
                break
            st["prec"] = {g: v + 1 for g, v in st["prec"].items()}                                  # refine until R meets eps
        res["start"] = {"fraction_bits": dict(st["prec"]), "bits": codec.verify(st, R0, p, code_length_bits(st, R0, p, codec)),
                        "fidelity": {k: v for k, v in f0.items() if k != "U"},
                        "reals_total": int(sum(R0[g][1].sum() for g in GROUPS))}
        st, R, bits, fid, trail, passes = greedy_select(X, st, fold_at, screen, eps, (p, X["T"], dev, dt), codec)
        cert_exact = np.searchsorted(X["K"], se_share).tolist() if se_share else []
        c = cert.evaluate(torch_program(R, st, X["T"], "cpu", torch.float64), eps, stats=True, exact_planes=cert_exact)
    res["greedy"] = {"bits": bits, **structure_summary(st, R, X, own_k), "screen_fidelity": {k: v for k, v in fid.items() if k != "U"},
                     "certificate_float64": c, "meets_eps_float64": bool(c["max_kl"] <= eps),
                     "passes": passes, "accepted": trail}
    if se_share:
        s1 = res["share_program_on_contract"]
        g_reads = st["Rd"] & st["SE"][None]
        kept = {}
        for nn, k in zip(*np.nonzero(g_reads & ~st1["Rd"])):
            key = f"{own_k[nn]}->{X['K'][k]}"
            kept[key] = kept.get(key, 0) + 1
        res["greedy_kept_that_share_dropped"] = {
            "S_E": sorted(set(res["greedy"]["S_E"]) - set(s1["S_E"])), "S_U": sorted(set(res["greedy"]["S_U"]) - set(s1["S_U"])),
            "read_pairs": dict(sorted(kept.items(), key=lambda kv: -kv[1])),
            "kap_unfolded_heads": sorted(set(range(H)) - set(res["greedy"]["kap_folded_heads"]))}
        res["share_kept_that_greedy_dropped"] = {
            "S_E": sorted(set(s1["S_E"]) - set(res["greedy"]["S_E"])), "S_U": sorted(set(s1["S_U"]) - set(res["greedy"]["S_U"])),
            "read_pairs": int((st1["Rd"] & ~g_reads).sum())}
    res["seconds"] = time.time() - t0
    dump(res, args.out)
    print("RECEIPT", args.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="bench/mpd_modadd_2951.py train output (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--read", choices=("own", "frac", "S_E"), default="frac")
    ap.add_argument("--read-frac", type=float, default=0.01)
    ap.add_argument("--embed-share", type=float, default=0.01)
    ap.add_argument("--unembed-share", type=float, default=0.01)
    ap.add_argument("--shifts", type=int, nargs="*", default=None, help="frequency_edit shifts; default all 1..p-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=("share", "greedy"), default="share",
                    help="share: the weight-power cutoffs above; greedy: MDL structure selection at --greedy-epsilon")
    ap.add_argument("--greedy-epsilon", type=float, help="declared tolerance of the greedy contract (max KL)")
    ap.add_argument("--device", default="cpu", help="greedy screen device (float32); the certificate is float64 cpu")
    args = ap.parse_args()
    t0 = time.time()
    cfg, step, W = load(args.run)
    p = cfg["p"]
    grid = np.arange(p * p)
    a, b = grid // p, grid % p
    shifts = args.shifts or list(range(1, p))
    res = {"run": os.path.basename(args.run), "step": step, "config": cfg, "read": args.read, "read_frac": args.read_frac,
           "model_params": int(sum(v.size for k, v in W.items() if k != "causal"))}
    if args.select == "greedy":
        return greedy_main(args, cfg, step, W, res)

    # ---- R is exact
    M = model_forward(W, cfg, a, b)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpd_modadd_2951 import all_pairs, build_model
    tm = build_model(cfg).double()
    tm.load_state_dict({k: torch.from_numpy(v) for k, v in W.items() if k != "causal"}, strict=False)
    with torch.no_grad():
        torch_diff = float(np.abs(tm(all_pairs(p)).numpy() - M["logits"]).max())
    X = exact_coordinates(W, cfg)
    R_logits, R_alpha, R_pre = rewrite_logits(X, a, b)
    Mc = M["logits"] - M["logits"].mean(1, keepdims=True)
    res["exact_rewrite"] = {"numpy_forward_vs_torch_max_abs_logit": torch_diff,
                            "R_vs_forward_max_abs_centered_logit": float(np.abs(R_logits - Mc).max()),
                            "R_alpha_max_abs": float(np.abs(R_alpha - M["alpha"]).max()),
                            "R_pre_max_abs": float(np.abs(R_pre - M["pre"]).max())}
    print("EXACT", res["exact_rewrite"], flush=True)

    # ---- frequencies from weight spectra only
    s_embed, s_read, se, su = share_frequencies(W, X, p, args.embed_share, args.unembed_share)
    res["frequencies"] = {"S_E": s_embed, "S_U": s_read, "W_E_share": {k: float(se[k - 1]) for k in s_embed},
                          "W_U_share": {k: float(su[k - 1]) for k in s_read},
                          "W_E_nonconstant_energy_in_S_E": float(se[np.array(s_embed) - 1].sum()),
                          "W_U_nonconstant_energy_in_S_U": float(su[np.array(s_read) - 1].sum())}
    print("FREQ", res["frequencies"], flush=True)

    # ---- P
    prog = extract_program(X, s_embed, s_read, args.read, args.read_frac)
    size = program_size(prog)
    P = run_program(prog, a, b)
    ks_read = np.array(s_read)
    yM, yP = y_of_logits(M["logits"], ks_read, p), P["y"]
    Dab = chars(ks_read, (a + b) % p, p)                                            # N,Ku,2
    A = (yP * Dab).sum(-1).mean(0)
    AM = (yM * Dab).sum(-1).mean(0)                                                 # for the report only, never used by T
    arg0 = {k: a + b for k in s_read}
    T_logits = closed_form(A, ks_read, p, arg0)
    ks_e = np.array(s_embed)
    zM = M["alpha"][:, :, 0, None, None] * chars(ks_e, a, p)[:, None] + M["alpha"][:, :, 1, None, None] * chars(ks_e, b, p)[:, None]
    gp = (X["g"] ** 2).sum(-1)
    own_freq = ks_e[prog["own"]]
    read_e = np.einsum("nhkt->nk", X["w"][:, :, np.searchsorted(X["K"], s_embed)] ** 2)
    res["stage_laws"] = {
        "embedding": {"exact": "e(x) = c0 + sum_k U_k D(w_k x) over all k (DFT, odd p)",
                      "energy_fraction_kept_in_S_E": res["frequencies"]["W_E_nonconstant_energy_in_S_E"]},
        "attention": {
            "pattern_mean": M["alpha"].mean(0).round(5).tolist(), "pattern_std": M["alpha"].std(0).round(5).tolist(),
            "pattern_pos0_min_max": [[float(M["alpha"][:, h, 0].min()), float(M["alpha"][:, h, 0].max())]
                                     for h in range(cfg["n_heads"])],
            "max_weight_on_eq": float(M["alpha"][:, :, 2].max()),
            "uniform_half_half_max_abs_dev": float(np.abs(M["alpha"][:, :, :2] - 0.5).max()),
            "score_frequency_share_top3": [[[int(X["K"][i]), float(gp[h, i] / gp[h].sum())]
                                            for i in np.argsort(-gp[h])[:3]] for h in range(len(gp))]},
        "mlp": {"neurons_by_own_freq": {int(k): int((own_freq == k).sum()) for k in s_embed},
                "P_read_pairs_own_to_k": {int(k): {int(kk): int(prog["read_mask"][own_freq == k, j].sum())
                                                   for j, kk in enumerate(s_embed)} for k in s_embed},
                "own_read_energy_fraction_quantiles_0_5_25_50": np.quantile(
                    read_e.max(1) / read_e.sum(1), [0, 0.05, 0.25, 0.5]).round(4).tolist(),
                "never_active_neurons_model": int((~(M["pre"] > 0).any(0)).sum()),
                "always_active_neurons_model": int((M["pre"] > 0).all(0).sum())},
        "unembed": {"A_k_from_P": dict(zip(s_read, A.tolist())),
                    "A_k_of_model_reported_not_used": dict(zip(s_read, AM.tolist())),
                    "model_y_off_law_rel": {k: rel(yM[:, i], AM[i] * Dab[:, i]) for i, k in enumerate(s_read)},
                    "P_y_off_law_rel": {k: rel(yP[:, i], A[i] * Dab[:, i]) for i, k in enumerate(s_read)}}}
    res["program_P"] = {**size, "compression_vs_model_params": res["model_params"] / (size["reals_total"] + size["ints_total"]),
                        "vs_model": compare(M["logits"], P["logits"], "P vs model"),
                        "interfaces": {
                            "alpha_max_abs_err": float(np.abs(P["alpha"] - M["alpha"]).max()),
                            "zeta_S_E_max_abs_err": float(np.abs(P["zeta"] - zM).max()),
                            "pre_rel_err": rel(P["pre"], M["pre"]),
                            "pre_max_abs_err": float(np.abs(P["pre"] - M["pre"]).max()),
                            "pre_rms": float(np.sqrt((M["pre"] ** 2).mean())),
                            "relu_sign_agree": float(((P["pre"] > 0) == (M["pre"] > 0)).mean()),
                            "y_rel_err": {k: rel(yP[:, i], yM[:, i]) for i, k in enumerate(s_read)},
                            "y_max_abs_err": {k: float(np.abs(yP[:, i] - yM[:, i]).max()) for i, k in enumerate(s_read)},
                            "centered_logit_max_abs_err": float(np.abs(P["logits"] - P["logits"].mean(1, keepdims=True) - Mc).max())}}
    res["closed_form_T"] = {"reals": len(A), "ints": len(A) + 1, "A": dict(zip(s_read, A.tolist())),
                            "vs_model": compare(M["logits"], T_logits, "T vs model"), "vs_P": compare(P["logits"], T_logits),
                            "centered_logit_max_abs_err": float(np.abs(T_logits - Mc).max())}
    res["P_with_residual"] = {"definition": "P plus every dropped term is R",
                              "R_vs_model": compare(M["logits"], R_logits, "R vs model")}
    res["residual_ladder"] = ladder(X, a, b, M, prog)
    print("SIZE", size["reals_total"], size["ints_total"], res["model_params"], flush=True)

    # ---- held-out counterfactuals
    cf = {"frequency_edit_pos0": {}, "attn_out_interchange": {}, "attn_out_mean_ablate": {}, "unembed_plane_scale": {}}
    E = W["W_E"]
    for T in [[k] for k in s_embed] + [s_read, s_embed]:
        st = {"model_vs_P": Stats(), "model_vs_T": Stats(), "model_vs_unedited": Stats(), "model_vs_R": Stats()}
        hit_shift, hit_orig = 0, 0
        sel = np.isin(X["K"], T)
        for s in shifts:
            table0 = frequency_edited(E, p, X["K"][sel], s)
            Me = model_forward(W, cfg, a, b, table0=table0)["logits"]
            st["model_vs_P"].add(Me, run_program(prog, a, b, shift=(T, s))["logits"])
            st["model_vs_T"].add(Me, closed_form(A, ks_read, p, {k: a + b + (s if k in T else 0) for k in s_read}))
            st["model_vs_unedited"].add(Me, M["logits"])
            st["model_vs_R"].add(Me, rewrite_logits(X, a, b, shift=(T, s))[0])
            hit_shift += int((Me.argmax(-1) == (a + b + s) % p).sum())
            hit_orig += int((Me.argmax(-1) == (a + b) % p).sum())
        name = "+".join(map(str, T))
        cf["frequency_edit_pos0"][name] = {key: v.summary(f"CF-a T={name} {key}") for key, v in st.items()}
        cf["frequency_edit_pos0"][name].update({"model_argmax_a+b+s": hit_shift / (len(shifts) * p * p),
                                                "model_argmax_a+b": hit_orig / (len(shifts) * p * p)})

    donor = np.random.default_rng(args.seed).permutation(p * p)
    for i, k in enumerate(s_embed):
        OU = np.einsum("hde,te->htd", X["O"], X["U"][k - 1])                        # H,2,d
        for kind in ("interchange", "mean"):
            zrep = zM[donor, :, i] if kind == "interchange" else np.broadcast_to(zM[:, :, i].mean(0), zM[:, :, i].shape)
            Me = model_forward(W, cfg, a, b, attn_patch=lambda _al: np.einsum("nht,htd->nd", zrep - zM[:, :, i], OU))["logits"]

            def patch(z, col):
                z = z.copy()
                z[:, :, col] = z[donor][:, :, col] if kind == "interchange" else z[:, :, col].mean(0)
                return z
            Pe = run_program(prog, a, b, zeta_patch=lambda z: patch(z, i))["logits"]
            Re = rewrite_logits(X, a, b, zeta_patch=lambda z: patch(z, k - 1))[0]
            if kind == "interchange":
                Te = closed_form(A, ks_read, p, {kk: (a + b)[donor] if kk == k else a + b for kk in s_read})
            else:
                Te = closed_form(A, ks_read, p, arg0, scale={k: 0.0})
            out = {"model_vs_P": compare(Me, Pe, f"CF-b {kind} k={k} P"), "model_vs_T": compare(Me, Te, f"CF-b {kind} k={k} T"),
                   "model_vs_unedited": compare(Me, M["logits"]), "model_vs_R": compare(Me, Re),
                   "model_argmax_a+b": float((Me.argmax(-1) == (a + b) % p).mean())}
            if kind == "interchange":
                out["model_argmax_donor_a+b"] = float((Me.argmax(-1) == (a + b)[donor] % p).mean())
            cf["attn_out_interchange" if kind == "interchange" else "attn_out_mean_ablate"][str(k)] = out

    for k in s_read:
        Dc = chars([k], np.arange(p), p)[:, 0]
        for lam in (0.0, 0.5, 2.0, -1.0):
            Me = model_forward(W, cfg, a, b, W_U=W["W_U"] + (lam - 1) * Dc @ X["V"][k - 1])["logits"]
            cf["unembed_plane_scale"][f"{k}@{lam:g}"] = {
                "model_vs_P": compare(Me, run_program(prog, a, b, uscale={k: lam})["logits"], f"CF-c k={k} lam={lam:g} P"),
                "model_vs_T": compare(Me, closed_form(A, ks_read, p, arg0, scale={k: lam}), f"CF-c k={k} lam={lam:g} T"),
                "model_vs_unedited": compare(Me, M["logits"]),
                "model_vs_R": compare(Me, rewrite_logits(X, a, b, uscale={k: lam})[0]),
                "model_argmax_a+b": float((Me.argmax(-1) == (a + b) % p).mean())}
    res["counterfactuals"] = cf
    res["shifts"] = shifts
    res["seconds"] = time.time() - t0
    dump(res, args.out)
    print("RECEIPT", args.out, flush=True)


if __name__ == "__main__":
    main()
