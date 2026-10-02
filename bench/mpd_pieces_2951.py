"""Per-token pieces on VPD's 4L target (#2951): the data the Rust fit reads, and the evaluation of
what it returns, in the per-token frontier's protocol (~/mpd-data/frontier/mpd_pertoken_frontier.py).

usage:
  mpd_pieces_2951.py dump DIR      site inputs on the fit rows and the eval rows, W, A, B, mu
  mpd_pieces_2951.py eval DIR      masked forwards for every selection level the fit wrote

dump writes, per site: {site}.W.f64 (d_out x d_in), {site}.A.f64 and {site}.B.f64 (the frontier's
input second moment and sampled-label output Fisher), {site}.mu.f64, {site}.fit.f32 and
{site}.eval.f32 (tokens x d_in, the clean forward's site inputs), and manifest.json.

The fit (crates/gam-mpd/examples/mpd_pieces_2951.rs) writes {site}.V.f64 (d_in x C), {site}.U.f64
(C x d_out) with V U = W^T on the centred input, and per level j {site}.mask{j}.u8 (eval tokens x C,
one byte per piece), plus pieces.json listing the levels. eval runs the masked program (VPD's
masked-program semantics: pieces act on x - mu, the library bias W mu always on) and reports, per
level, KL(target || masked) per position, mean L0 and mean bits per token in the frontier's
per-token code (omega(|A| + 1) + log2 C(C_s, |A|) per site), into DIR/frontier.json.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "mpd-data/vpd"))
from vpd_bits import omega_len  # noqa: E402
from vpd_eval import kl_per_pos  # noqa: E402
from vpd_model import load_target, site_names, val_tokens  # noqa: E402

MODE, DIR = sys.argv[1], Path(sys.argv[2])
DEV = "mps"
FIT_ROWS, FIT_OFF = 16, 2048  # the frontier's statistics rows
EVAL_ROWS, EVAL_OFF = 16, 1024  # the frontier's eval rows (its first 16 of 32)
MB = 4
FDIR = Path.home() / "mpd-data/frontier"

model = load_target(DEV)
names = site_names()
site = model.site


def clear():
    for n in names:
        s = site(n)
        s.mask = s.delta_mask = s.last_input = s.last_output = None
        s.cache_input = s.cache_output = False
        s.in_fn = None


def inputs(rows, offset):
    ids = val_tokens(rows, offset=offset)
    out = {n: [] for n in names}
    with torch.no_grad():
        for i in range(0, rows, MB):
            for n in names:
                site(n).cache_input = True
            model(ids[i:i + MB].to(DEV))
            for n in names:
                out[n].append(site(n).last_input.flatten(0, 1).float().cpu().numpy())
            clear()
    return {n: np.concatenate(v) for n, v in out.items()}


def dump():
    DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(FDIR / "stats_vpd4l.npz")
    mu = np.load(FDIR / "mu_vpd4l.npz")
    manifest = {"sites": [], "fit_tokens": FIT_ROWS * 512, "eval_tokens": EVAL_ROWS * 512}
    fit, ev = inputs(FIT_ROWS, FIT_OFF), inputs(EVAL_ROWS, EVAL_OFF)
    for n in names:
        W = site(n).W.cpu().double().numpy()
        W.tofile(DIR / f"{n}.W.f64")
        z["A:" + n].astype(np.float64).tofile(DIR / f"{n}.A.f64")
        z["B:" + n].astype(np.float64).tofile(DIR / f"{n}.B.f64")
        mu[n].astype(np.float64).tofile(DIR / f"{n}.mu.f64")
        fit[n].astype(np.float32).tofile(DIR / f"{n}.fit.f32")
        ev[n].astype(np.float32).tofile(DIR / f"{n}.eval.f32")
        manifest["sites"].append({"name": n, "d_out": W.shape[0], "d_in": W.shape[1]})
    json.dump(manifest, open(DIR / "manifest.json", "w"), indent=1)
    print("dumped", len(names), "sites")


CMAX = 40000
OMEGA = omega_len(np.arange(1, CMAX + 2)).astype(np.float64)


def set_bits(k, C):
    kk = np.arange(C + 1)
    table = OMEGA[:C + 1] + (math.lgamma(C + 1) - np.array([math.lgamma(x + 1) + math.lgamma(C - x + 1) for x in kk])) / math.log(2)
    return torch.tensor(table)[k.long().cpu()]


def evaluate():
    spec = json.load(open(DIR / "pieces.json"))
    manifest = json.load(open(DIR / "manifest.json"))
    T = manifest["eval_tokens"]
    lib, hooks = {}, []
    for s in manifest["sites"]:
        n, d_in, d_out = s["name"], s["d_in"], s["d_out"]
        C = spec["pieces"][n]
        V = np.fromfile(DIR / f"{n}.V.f64").reshape(d_in, C)
        U = np.fromfile(DIR / f"{n}.U.f64").reshape(C, d_out)
        W = site(n).W.cpu().double().numpy()
        err = np.abs(V @ U - W.T).max() / np.abs(W).max()
        assert err < 1e-6, (n, err)
        mu = np.fromfile(DIR / f"{n}.mu.f64")
        m_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
        c_t = torch.tensor(W @ mu, dtype=torch.float32, device=DEV)
        lib[n] = (torch.tensor(V, dtype=torch.float32, device=DEV), torch.tensor(U, dtype=torch.float32, device=DEV), m_t, c_t, C)
    ids = val_tokens(EVAL_ROWS, offset=EVAL_OFF)
    points = []
    for j, level in enumerate(spec["levels"]):
        masks = {n: torch.from_numpy(np.fromfile(DIR / f"{n}.mask{j}.u8", dtype=np.uint8).reshape(T, lib[n][4]).astype(np.float32)) for n in names}
        kl_sum, l0_sum, bits_sum, count = 0.0, 0.0, 0.0, 0
        for i in range(0, EVAL_ROWS, MB):
            b = ids[i:i + MB].to(DEV)
            with torch.no_grad():
                tgt = model(b)
            clear()
            lo, hi = i * 512, (i + MB) * 512
            for n in names:
                V, U, m_t, c_t, C = lib[n]
                st = site(n)
                st.V, st.U = V, U
                st.mask = masks[n][lo:hi].reshape(b.shape[0], 512, C).to(DEV)
                st.in_fn = lambda x, m_t=m_t: x - m_t
                hooks.append(st.register_forward_hook(lambda mod, inp, out, c_t=c_t: out + c_t))
            with torch.no_grad():
                lg = model(b)
            for h in hooks:
                h.remove()
            hooks.clear()
            kl = kl_per_pos(lg, tgt)
            k_site = {n: masks[n][lo:hi].sum(-1) for n in names}
            l0 = sum(k_site.values())
            bits = sum(set_bits(k_site[n], lib[n][4]) for n in names)
            kl_sum += kl.sum().item()
            l0_sum += l0.sum().item()
            bits_sum += bits.sum().item()
            count += kl.numel()
            for n in names:
                site(n).V = site(n).U = None
            clear()
            del lg, tgt
            torch.mps.empty_cache()
        p = {"level": level, "kl": kl_sum / count, "l0": l0_sum / count, "bits": bits_sum / count}
        print(f"level {level}: L0 {p['l0']:.1f}  bits/tok {p['bits']:.1f}  KL {p['kl']:.4g}", flush=True)
        points.append(p)
    json.dump({"pieces_total": sum(spec["pieces"].values()), "points": points}, open(DIR / "frontier.json", "w"), indent=1)


dump() if MODE == "dump" else evaluate()
