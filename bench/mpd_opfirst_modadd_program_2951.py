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

Greedy MDL structure selection over R (the certified replacement for the 1% cutoffs) is the Rust baseline driver
crates/gam-mpd/examples/mpd_modadd_cyclic_baseline_2951.rs, reading the float64 export of
bench/mpd_engine_export_2951.py.
"""
from __future__ import annotations

import argparse
import hashlib
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
    """One operation of the Rust MPD surface (gam_mpd::surface)."""
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


def share_frequencies(W, X, p, embed_share, unembed_share):
    """The share path's weight-spectrum frequency sets: W_E and W_U frequencies above their power-share cutoffs."""
    def shares(T):
        pw = cyclic_planes(T, p)[2]
        return pw / pw.sum()
    se, su = shares(W["W_E"][:p]), shares(W["W_U"])
    return ([int(k) for k in X["K"][np.argsort(-se)] if se[k - 1] > embed_share],
            [int(k) for k in X["K"][np.argsort(-su)] if su[k - 1] > unembed_share], se, su)


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
    args = ap.parse_args()
    t0 = time.time()
    cfg, step, W = load(args.run)
    p = cfg["p"]
    grid = np.arange(p * p)
    a, b = grid // p, grid % p
    shifts = args.shifts or list(range(1, p))
    with open(args.run, "rb") as fh:
        checkpoint_sha256 = hashlib.sha256(fh.read()).hexdigest()
    res = {"run": os.path.basename(args.run), "checkpoint_sha256": checkpoint_sha256, "step": step, "config": cfg,
           "read": args.read, "read_frac": args.read_frac,
           "model_params": int(sum(v.size for k, v in W.items() if k != "causal"))}

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
