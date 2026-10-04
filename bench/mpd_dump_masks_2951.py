"""Per-token active sets for E1/E2/E3/E5 (#2951): VPD's rounded masks (gate > 0) and the Fisher-SVD
basis at its KL-matched threshold, on 128 held-out Pile val rows (offset 1024; the first 32 are the
frontier rows), with per-position KL.

Global piece index = site offset + component, sites in vpd_model.site_names() order (layer-major;
q, k, v, o, c_fc, down within a layer), so 'upstream' is a lower site index (same position) or any site
at an earlier position.

Writes masks_vpd4l.npz:
  ids [R, S] tokens; site_names; vpd_offsets, wsvd_offsets (len n_sites + 1)
  vpd_indptr, vpd_indices   CSR over the R*S positions (row-major) of active VPD pieces
  wsvd_indptr, wsvd_indices same for the Fisher-SVD basis (threshold tau_wsvd), first 32 rows only
  kl_vpd, kl_wsvd [R, S]    KL(target || masked program) per position
  kl_vpd_core, kl_wsvd_core [R, S]  the same with only the core (pieces active on >= 90% of positions)
  core_vpd, core_wsvd       global indices of the core pieces
usage: MPD_MEM_GIB=12 venv python mpd_dump_masks_2951.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
from vpd_eval import kl_per_pos  # noqa: E402
from vpd_model import load_target, load_vpd, site_names, val_tokens  # noqa: E402

ROWS, OFF, MB, DEV = 128, 1024, 8, "mps"
WROWS = 32  # the Fisher-SVD basis (~4.5k pieces/token) is dumped on the 32 frontier rows only (memory)
F = Path.home() / "mpd-data/frontier"
names = site_names()
target = load_target(DEV)
vpd = load_vpd(target, DEV)
ids = val_tokens(ROWS, offset=OFF)
z = np.load(F / "stats_vpd4l.npz")
tau_w = next(p["tau"] for p in json.load(open(F / "pertoken_vpd4l_bases.json"))["wsvd"]["points"] if round(p["l0"]) == 4451)


def psd_sqrt(M):
    w, E = sl.eigh(M)
    w = np.maximum(w, w.max() * M.shape[0] * np.finfo(np.float32).eps)
    return (E * np.sqrt(w)) @ E.T, (E / np.sqrt(w)) @ E.T


lib, unorm = {}, {}
for n in names:
    W = target.site(n).W.cpu().double().numpy()
    A, B = z["A:" + n].astype(np.float64), z["B:" + n].astype(np.float64)
    Ah, Aih = psd_sqrt(A)
    Bh, Bih = psd_sqrt(B)
    P, s, Qt = sl.svd(Bh @ W @ Ah, full_matrices=False, lapack_driver="gesvd")
    r = np.sqrt(s)
    V, U = Aih @ (Qt.T * r), ((Bih @ P) * r).T
    lib[n] = (torch.tensor(V, dtype=torch.float32, device=DEV), torch.tensor(U, dtype=torch.float32, device=DEV))
    unorm[n] = torch.tensor(np.sqrt(np.maximum(((U @ B) * U).sum(1), 0)), dtype=torch.float32, device=DEV)
vpd_uv = {n: (target.site(n).U, target.site(n).V) for n in names}
lib_uv = {n: (lib[n][1], lib[n][0]) for n in names}  # (U, V) as the Site expects
off_v = np.cumsum([0] + [vpd.C[n] for n in names])
off_w = np.cumsum([0] + [lib[n][0].shape[1] for n in names])


def masked_kl(b, tgt, masks, uv):
    for n in names:
        st = target.site(n)
        st.U, st.V = uv[n]
        st.mask = masks[n]
    try:
        with torch.no_grad():
            return kl_per_pos(target(b), tgt).cpu().numpy()
    finally:
        for n in names:
            target.site(n).mask = None


def to_csr(masks, offs):
    """(counts per position, active global indices grouped by position, ascending) from per-site masks [B, S, C]."""
    B, S = next(iter(masks.values())).shape[:2]
    ps, cs = [], []
    for si, n in enumerate(names):
        nz = masks[n].reshape(B * S, -1).nonzero().cpu().numpy()
        ps.append(nz[:, 0])
        cs.append((nz[:, 1] + offs[si]).astype(np.int32))
    p, c = np.concatenate(ps), np.concatenate(cs)
    order = np.lexsort((c, p))
    return np.bincount(p, minlength=B * S), c[order]


acts = {"vpd": [], "wsvd": []}
kls = {"vpd": np.zeros((ROWS, 512)), "wsvd": np.full((ROWS, 512), np.nan)}
for i in range(0, ROWS, MB):
    b = ids[i:i + MB].to(DEV)
    tgt, g = vpd.target_and_ci(b)
    mv = {n: (v > 0).float() for n, v in g.items()}
    del g
    for n in names:
        target.site(n).cache_input = True
    with torch.no_grad():
        target(b)
    X = {n: target.site(n).last_input for n in names}
    vpd.clear()
    mw = {n: ((X[n] @ lib[n][0]).abs() * unorm[n] > tau_w).float() for n in names} if i < WROWS else None
    del X
    kls["vpd"][i:i + MB] = masked_kl(b, tgt, mv, vpd_uv)
    acts["vpd"].append(to_csr(mv, off_v))
    if mw is not None:
        kls["wsvd"][i:i + MB] = masked_kl(b, tgt, mw, lib_uv)
        acts["wsvd"].append(to_csr(mw, off_w))
    del tgt, mv, mw
    torch.mps.empty_cache()
    print(f"rows {i + MB}/{ROWS}: L0 vpd {acts['vpd'][-1][0].mean():.1f} KL {kls['vpd'][i:i + MB].mean():.3f}", flush=True)
for k in acts:
    acts[k] = (np.concatenate([a[0] for a in acts[k]]), np.concatenate([a[1] for a in acts[k]]))

# core pieces (active on >= 90% of positions) and the KL with only the core on
core = {}
for k, offs in (("vpd", off_v), ("wsvd", off_w)):
    cnt = np.bincount(acts[k][1], minlength=offs[-1])
    core[k] = np.nonzero(cnt >= 0.9 * len(acts[k][0]))[0]
kls["vpd_core"], kls["wsvd_core"] = np.zeros((ROWS, 512)), np.full((ROWS, 512), np.nan)
for i in range(0, ROWS, MB):
    b = ids[i:i + MB].to(DEV)
    with torch.no_grad():
        vpd.clear()
        tgt = target(b)
    for k, offs, uv, sizes in (("vpd", off_v, vpd_uv, vpd.C), ("wsvd", off_w, lib_uv, {n: lib[n][0].shape[1] for n in names})):
        if k == "wsvd" and i >= WROWS:
            continue
        m = {}
        for si, n in enumerate(names):
            v = torch.zeros(sizes[n], device=DEV)
            c = core[k][(core[k] >= offs[si]) & (core[k] < offs[si + 1])] - offs[si]
            v[torch.tensor(c, dtype=torch.long, device=DEV)] = 1.0
            m[n] = v.expand(b.shape[0], b.shape[1], -1)
        kls[k + "_core"][i:i + MB] = masked_kl(b, tgt, m, uv)
    del tgt
    torch.mps.empty_cache()

out = {"ids": ids.numpy(), "site_names": np.array(names), "vpd_offsets": off_v, "wsvd_offsets": off_w,
       "tau_wsvd": tau_w, "core_vpd": core["vpd"], "core_wsvd": core["wsvd"]}
for k in ("vpd", "wsvd"):
    out[k + "_indptr"] = np.concatenate([[0], np.cumsum(acts[k][0])])
    out[k + "_indices"] = acts[k][1]
for k, v in kls.items():
    out["kl_" + k] = v
np.savez_compressed(F / "masks_vpd4l.npz", **out)
print("core sizes", {k: len(v) for k, v in core.items()}, "KL", {k: float(v.mean()) for k, v in kls.items()})
