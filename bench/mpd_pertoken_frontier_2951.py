"""Per-token frontier (#2951): how many rank-1 pieces, and how many bits per token, does each
explanation switch on, against KL(target || masked program) on the matched rows.

Every explanation is a library of rank-1 pieces per site (y = sum_c u_c (v_c . x)) plus, per token,
the set of pieces that are on (VPD's masked-program semantics: vpd_model.Site, delta excluded). The
per-token code is VPD's (vpd_pertoken.py): per site and token, omega(|A| + 1) + log2 C(C_s, |A|) for a
binary mask, plus Elias-delta coefficients for a lattice mask.

Bases (the library is the target's own weights, rewritten exactly; all pieces on = the target):
  native  neurons and head dimensions: rows of q/k/v/c_fc, columns of o/down. An MLP neuron is
          its c_fc row and its down column, switched together (2 pieces, one set sent)
  svd     per-matrix SVD, sqrt(s) on each side
  wsvd    per-matrix SVD in the Fisher metric: B^1/2 W A^1/2 = P S Q^T, with A = E[x x^T] at the
          site input and B = E[g g^T] the sampled-label Fisher at the site output; the pieces are
          B-orthogonal, so dropping a set costs the sum of the pieces' own costs (to second order)
  *_c     the same on the centred input: the library holds the bias W E[x] (always on, no per-token
          bits) and the pieces act on x - E[x], so a piece that is off is mean-ablated, not zeroed
          (wsvd_c whitens with the covariance)
Per-token selection: a piece's score is |v_c . x_t| * sqrt(u_c^T B u_c) (the root second-order KL of
dropping it alone, in one unit for every site), read from the clean forward like VPD's gates; one
global threshold per point. Thresholds are set on the statistics rows (disjoint from eval rows).

VPD: its causal-importance gates on the same rows, thresholded at tau (S2), as binary masks
(rounded) and as gates on the p = 4 lattice (ci).

usage: mpd_pertoken_frontier_2951.py {vpd4l|pythia70m} {bases|vpd} OUT.json
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpd_bits import delta_len, omega_len  # noqa: E402
from vpd_eval import kl_per_pos  # noqa: E402
from vpd_model import val_tokens  # noqa: E402

MODEL, MODE, OUT = sys.argv[1:4]
DEV = "mps"
EVAL_ROWS, EVAL_OFF, STAT_ROWS, STAT_OFF = 32, 1024, 64, 2048
MB, FMB = int(os.environ.get("FRONTIER_MB", "8")), 4  # rows per microbatch (the footprint scales with it)
FDIR = Path.home() / "mpd-data/frontier"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
torch.manual_seed(0)

if MODEL == "vpd4l":
    from vpd_model import load_target, site_names
    model = load_target(DEV)
    names = site_names()
else:
    from mpd_pythia_model_2951 import KINDS, load_pythia, site_name
    model = load_pythia(DEV)
    names = [site_name(i, k) for i in range(model.n_layer) for k in KINDS]
site = model.site
WRITE = lambda n: n.endswith(("o_proj", "down_proj"))
COUPLED = [(n, n.replace("c_fc", "down_proj")) for n in names if n.endswith("c_fc")]


def clear():
    for n in names:
        s = site(n)
        s.mask = s.delta_mask = s.last_input = s.last_output = None
        s.cache_input = s.cache_output = False


# ------------------------------------------------------------------ statistics: A = E[x x^T], B = Fisher


def fisher_stats():
    path = FDIR / f"stats_{MODEL}.npz"
    if path.exists():
        z = np.load(path)
        return {n: z["A:" + n].astype(np.float64) for n in names}, {n: z["B:" + n].astype(np.float64) for n in names}
    ids = val_tokens(STAT_ROWS, offset=STAT_OFF)
    A = {n: 0.0 for n in names}
    B = {n: 0.0 for n in names}
    outs, ntok = {}, 0
    def keep(mod, inp, out, n):
        out.retain_grad()
        outs[n] = out

    hooks = [site(n).register_forward_hook(lambda mod, inp, out, n=n: keep(mod, inp, out, n)) for n in names]
    model.wte.requires_grad_(True)
    for i in range(0, STAT_ROWS, FMB):
        b = ids[i:i + FMB].to(DEV)
        for n in names:
            site(n).cache_input = True
        logits = model(b)
        lp = F.log_softmax(logits, -1)
        y = torch.multinomial(lp.detach().exp().flatten(0, 1), 1)
        lp.flatten(0, 1).gather(1, y).sum().backward()
        for n in names:
            g = outs[n].grad.flatten(0, 1)
            x = site(n).last_input.flatten(0, 1)
            B[n] = B[n] + (g.T @ g).cpu().double().numpy()
            A[n] = A[n] + (x.T @ x).cpu().double().numpy()
        ntok += b.numel()
        outs.clear()
        clear()
        del logits, lp
    for h in hooks:
        h.remove()
    model.wte.requires_grad_(False)
    A = {n: v / ntok for n, v in A.items()}
    B = {n: v / ntok for n, v in B.items()}
    np.savez(path, **{"A:" + n: A[n].astype(np.float32) for n in names}, **{"B:" + n: B[n].astype(np.float32) for n in names})
    log(f"Fisher statistics on {ntok} tokens")
    return A, B


def input_means():
    """E[x] at every site input, on the statistics rows."""
    path = FDIR / f"mu_{MODEL}.npz"
    if path.exists():
        z = np.load(path)
        return {n: z[n].astype(np.float64) for n in names}
    ids = val_tokens(STAT_ROWS, offset=STAT_OFF)
    tot = {n: 0.0 for n in names}
    ntok = 0
    with torch.no_grad():
        for i in range(0, STAT_ROWS, MB):
            for n in names:
                site(n).cache_input = True
            model(ids[i:i + MB].to(DEV))
            for n in names:
                tot[n] = tot[n] + site(n).last_input.flatten(0, 1).sum(0).cpu().double().numpy()
            ntok += ids[i:i + MB].numel()
            clear()
    mu = {n: v / ntok for n, v in tot.items()}
    np.savez(path, **mu)
    return mu


def psd_sqrt(M):
    """M^1/2 and M^-1/2 with eigenvalues floored at lambda_max * d * eps32 (below fp32 resolution)."""
    w, E = sl.eigh(M)
    w = np.maximum(w, w.max() * M.shape[0] * np.finfo(np.float32).eps)
    return (E * np.sqrt(w)) @ E.T, (E / np.sqrt(w)) @ E.T


def make_basis(kind, W, A, B, write):
    """Library pieces (V [d_in, C], U [C, d_out]) with V @ U = W^T."""
    d_out, d_in = W.shape
    if kind == "native":
        return (np.eye(d_in), W.T.copy()) if write else (W.T.copy(), np.eye(d_out))
    if kind == "svd":
        P, s, Qt = sl.svd(W, full_matrices=False, lapack_driver="gesvd")
        r = np.sqrt(s)
        return Qt.T * r, (P * r).T
    Ah, Aih = psd_sqrt(A)
    Bh, Bih = psd_sqrt(B)
    P, s, Qt = sl.svd(Bh @ W @ Ah, full_matrices=False, lapack_driver="gesvd")
    r = np.sqrt(s)
    return Aih @ (Qt.T * r), ((Bih @ P) * r).T


# ------------------------------------------------------------------ per-token code


CMAX = 40000
OMEGA = omega_len(np.arange(1, CMAX + 2)).astype(np.float64)  # OMEGA[k] = omega(k + 1)
_lb = {}


def set_bits(k: torch.Tensor, C: int) -> torch.Tensor:
    """omega(k + 1) + log2 C(C, k) per position, k = active count [B, S]."""
    if C not in _lb:
        kk = np.arange(C + 1)
        _lb[C] = torch.tensor(OMEGA[:C + 1] + (math.lgamma(C + 1) - np.array([math.lgamma(x + 1) + math.lgamma(C - x + 1) for x in kk])) / math.log(2))
    return _lb[C][k.long().cpu()]


def run_masks(b, tgt, masks, groups, coef=None):
    """KL, mean L0 and mean bits/token for one microbatch of masks. groups: list of (site names sharing
    one transmitted set, C)."""
    for n in names:
        site(n).mask = masks[n]
    try:
        with torch.no_grad():
            lg = model(b)
    finally:
        clear()
    kl = kl_per_pos(lg, tgt).mean().item()
    l0 = sum(masks[n].gt(0).sum(-1).cpu().double() for n in names)
    bits = sum(set_bits(masks[g[0]].gt(0).sum(-1), C) for g, C in groups)
    if coef is not None:
        bits = bits + coef
    del lg
    return kl, l0.mean().item(), bits.double().mean().item()


# ------------------------------------------------------------------ bases mode


def bases_mode():
    A, B = fisher_stats()
    ids_e = val_tokens(EVAL_ROWS, offset=EVAL_OFF)
    ids_s = val_tokens(MB, offset=STAT_OFF)  # threshold calibration rows (statistics rows)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    targets = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    mu = input_means()
    for kind in ("native", "svd", "wsvd", "native_c", "wsvd_c"):
        if kind in results:
            continue
        base, centered = kind.removesuffix("_c"), kind.endswith("_c")
        lib, unorm, hooks = {}, {}, []
        for n in names:
            W = site(n).W.cpu().double().numpy()
            An = A[n] - np.outer(mu[n], mu[n]) if centered else A[n]
            V, U = make_basis(base, W, An, B[n], WRITE(n))
            if centered:  # library bias W mu (always on); pieces act on x - mu
                m_t = torch.tensor(mu[n], dtype=torch.float32, device=DEV)
                c_t = torch.tensor(W @ mu[n], dtype=torch.float32, device=DEV)
                site(n).in_fn = lambda x, m_t=m_t: x - m_t
                hooks.append(site(n).register_forward_hook(lambda mod, inp, out, c_t=c_t: out + c_t))
            err = np.abs(V @ U - W.T).max() / np.abs(W).max()
            assert err < 1e-6, (kind, n, err)
            lib[n] = (torch.tensor(V, dtype=torch.float32, device=DEV), torch.tensor(U, dtype=torch.float32, device=DEV))
            unorm[n] = torch.tensor(np.sqrt(np.maximum(((U @ B[n]) * U).sum(1), 0)), dtype=torch.float32, device=DEV)
        coupled = base == "native"
        groups = [((n,), lib[n][0].shape[1]) for n in names if not (coupled and n.endswith(("c_fc", "down_proj")))]
        if coupled:
            groups += [((d, f), lib[d][0].shape[1]) for f, d in COUPLED]

        def scores(b):
            for n in names:
                site(n).cache_input = True
            with torch.no_grad():
                tgt = model(b)
            X = {n: site(n).last_input for n in names}
            clear()
            sc = {n: (X[n] @ lib[n][0]).abs() * unorm[n] for n in names}
            if coupled:
                for f, d in COUPLED:
                    sc[f] = sc[d]
            del X
            return tgt, sc

        # thresholds for target mean L0, on calibration rows
        _, sc = scores(ids_s.to(DEV))
        pool = torch.cat([v[:, ::4].flatten().cpu() for v in sc.values()]).numpy()
        npos = sc[names[0]][:, ::4].shape[0] * sc[names[0]][:, ::4].shape[1]
        pool = -np.sort(-pool)
        del sc
        total = sum(lib[n][0].shape[1] for n in names)
        taus = [float(pool[L * npos]) for L in targets if L < total] + [-1.0]
        for n in names:
            site(n).V, site(n).U = lib[n]
        pts = [{"tau": t, "kl": 0.0, "l0": 0.0, "bits": 0.0} for t in taus]
        nmb = EVAL_ROWS // MB
        for i in range(nmb):
            b = ids_e[i * MB:(i + 1) * MB].to(DEV)
            tgt, sc = scores(b)
            for p in pts:
                masks = {n: (sc[n] > p["tau"]).float() for n in names}
                kl, l0, bits = run_masks(b, tgt, masks, groups)
                p["kl"] += kl / nmb
                p["l0"] += l0 / nmb
                p["bits"] += bits / nmb
                del masks
            del tgt, sc
            torch.mps.empty_cache()
        for n in names:
            site(n).V = site(n).U = site(n).in_fn = None
        for h in hooks:
            h.remove()
        assert pts[-1]["kl"] < 1e-4, ("all pieces on must be the target", kind, pts[-1])
        results[kind] = {"pieces_total": total, "points": pts}
        for p in pts:
            log(f"{kind}: L0 {p['l0']:9.1f}  bits/tok {p['bits']:9.1f}  KL {p['kl']:.4g}")
        json.dump(results, open(OUT, "w"), indent=1)
        del lib, unorm
        torch.mps.empty_cache()


# ------------------------------------------------------------------ VPD mode


def vpd_mode():
    from vpd_model import load_vpd
    vpd = load_vpd(model, DEV)
    ids_e = val_tokens(EVAL_ROWS, offset=EVAL_OFF)
    taus = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    groups = [((n,), vpd.C[n]) for n in vpd.names]
    res = {"C": vpd.C, "rounded": [{"tau": t, "kl": 0.0, "l0": 0.0, "bits": 0.0} for t in taus],
           "ci_p4": [{"tau": t, "kl": 0.0, "l0": 0.0, "bits": 0.0} for t in taus],
           "all_on": {"kl": 0.0, "l0": float(sum(vpd.C.values())), "bits": 0.0}}
    nmb = EVAL_ROWS // MB
    for i in range(nmb):
        b = ids_e[i * MB:(i + 1) * MB].to(DEV)
        tgt, g = vpd.target_and_ci(b)
        for pr, pc in zip(res["rounded"], res["ci_p4"]):
            m = {n: (v > pr["tau"]).float() for n, v in g.items()}
            for k, x in zip(("kl", "l0", "bits"), run_masks(b, tgt, m, groups)):
                pr[k] += x / nmb
            kq = {n: torch.round(v * 16) * (v > pc["tau"]) for n, v in g.items()}
            coef = 0.0
            for n, k in kq.items():
                nz = k[k > 0].long().cpu().numpy()
                per = torch.zeros(k.shape, dtype=torch.float64)
                per[(k > 0).cpu()] = torch.from_numpy(delta_len(nz).astype(np.float64))
                coef = coef + per.sum(-1)
            m = {n: k / 16 for n, k in kq.items()}
            for k, x in zip(("kl", "l0", "bits"), run_masks(b, tgt, m, groups, coef)):
                pc[k] += x / nmb
            del m, kq
        ones = {n: torch.ones_like(v) for n, v in g.items()}
        res["all_on"]["kl"] += run_masks(b, tgt, ones, groups)[0] / nmb
        del g, tgt, ones
        torch.mps.empty_cache()
        log(f"VPD microbatch {i + 1}/{nmb}")
    for k in ("rounded", "ci_p4"):
        for p in res[k]:
            log(f"VPD {k} tau {p['tau']}: L0 {p['l0']:.1f} bits/tok {p['bits']:.1f} KL {p['kl']:.4g}")
    log(f"VPD all on: KL {res['all_on']['kl']:.4g}")
    json.dump(res, open(OUT, "w"), indent=1)


bases_mode() if MODE == "bases" else vpd_mode()
