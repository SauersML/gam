"""Gate laws for VPD-4L (#2951): data for `gam_mpd::gates` and the scoring of the laws it fits.

Subcommands:
  amps OUT_DIR [--rows N]   each subcomponent's read amplitude a_j = v_jᵀx on the clean native forward,
                            for the first N rows of frontier/masks_vpd4l.npz (val rows 1024..; the first
                            32 are the eval passages): per site OUT_DIR/{site}.npy, float16 [C, N·512]
                            (one subcomponent's amplitudes contiguous), sites and subcomponents in the
                            masked driver's numbering (library site names sorted), plus OUT_DIR/sites.txt.

  kl SETS [--rows LO:HI]     KL(target ‖ masked) per token and L0 of per-token sets on those rows of masks_vpd4l.npz
                            (default the 32 eval passages), the subcomponents off the set dropped, nothing
                            else added (the masked driver's library forward). SETS is a CSR over the rows'
                            tokens in the driver's numbering: DIR/{indptr,indices}.i64 (vpd_sets_export.py)
                            or PREFIX.{indptr,indices}.npy (mpd_gates_2951, the masked driver).
  ciflops                   multiply–adds per token of VPD's causal-importance network at 512 tokens.

usage: MPD_MEM_GIB=4 ~/mpd-data/venv/bin/python vpd_gates.py SUBCOMMAND ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from vpd_model import VPD_PTH, load_target, site_names  # noqa: E402

FRONTIER = Path.home() / "mpd-data/frontier/masks_vpd4l.npz"
LIBRARY = Path.home() / "mpd-data/pieces/vpd4l_library"
DEV = "mps"


def driver_sites():
    """(driver name, VPD name, subcomponents) in the masked driver's site order."""
    manifest = json.load(open(LIBRARY / "manifest.json"))
    return [(n, manifest[n]["vpd"], manifest[n]["pieces"]) for n in sorted(manifest)]


def load_v() -> dict:
    """Each site's read vectors V [d_in, C] from the published decomposition (not its CI network)."""
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    return {n: raw["_components." + n.replace(".", "-") + ".V"].float().to(DEV) for n in site_names()}


def cmd_amps(out: Path, rows: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(FRONTIER)
    ids = torch.tensor(z["ids"][:rows].astype(np.int64))
    S = ids.shape[1]
    target = load_target(DEV)
    sites = driver_sites()
    V = load_v()
    files = {vn: np.lib.format.open_memmap(out / f"{dn}.npy", mode="w+", dtype=np.float16, shape=(c, rows * S)) for dn, vn, c in sites}
    with open(out / "sites.txt", "w") as fh:
        for dn, _, c in sites:
            fh.write(f"{dn} {c}\n")
    names = site_names()
    MB = 8
    for i in range(0, rows, MB):
        b = ids[i:i + MB].to(DEV)
        for n in names:
            target.site(n).cache_input = True
        with torch.no_grad():
            target(b)
            for _, vn, _ in sites:
                a = (target.site(vn).last_input @ V[vn]).reshape(-1, V[vn].shape[1])  # [B·S, C]
                files[vn][:, i * S:(i + b.shape[0]) * S] = a.T.to(torch.float16).cpu().numpy()
        for n in names:
            target.site(n).cache_input = False
            target.site(n).last_input = None
        print(f"rows {i + b.shape[0]}/{rows}", flush=True)
    for f in files.values():
        f.flush()


def read_sets(spec: str):
    p = Path(spec)
    if p.is_dir():
        return np.fromfile(p / "indptr.i64", dtype="<i8"), np.fromfile(p / "indices.i64", dtype="<i8")
    return np.load(f"{spec}.indptr.npy"), np.load(f"{spec}.indices.npy")


def cmd_kl(spec: str, rows: str) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from vpd_eval import kl_per_pos  # noqa: E402

    DEV = "cpu"  # a row at a time; the CPU's footprint stays within a 2 GiB lease, the GPU allocator's does not

    lo, hi = (int(v) for v in rows.split(":"))
    z = np.load(FRONTIER)
    ids = torch.tensor(z["ids"][lo:hi].astype(np.int64))
    S = ids.shape[1]
    indptr, indices = read_sets(spec)
    if len(indptr) - 1 != (hi - lo) * S:
        raise SystemExit(f"{spec}: {len(indptr) - 1} tokens, rows {rows} have {(hi - lo) * S}")
    sites = driver_sites()
    offs = np.cumsum([0] + [c for _, _, c in sites])
    target = load_target(DEV)
    # The library's own float64 files (pieces × d per side), not the checkpoint with its CI network.
    for dn, vn, c in sites:
        st = target.site(vn)
        v = np.fromfile(LIBRARY / f"{dn}.v.f64", dtype="<f8").reshape(c, -1)
        u = np.fromfile(LIBRARY / f"{dn}.u.f64", dtype="<f8").reshape(c, -1)
        st.V = torch.tensor(v.T.astype(np.float32), device=DEV)
        st.U = torch.tensor(u.astype(np.float32), device=DEV)
        del u, v
    kls = []
    MB = 1  # one row's logits at a time keeps the scorer in the 1 GiB lane
    for i in range(0, hi - lo, MB):
        b = ids[i:i + MB].to(DEV)
        B = b.shape[0]
        for _, vn, _ in sites:
            target.site(vn).mask = None
        with torch.no_grad():
            tgt = target(b)
        t0, t1 = i * S, (i + B) * S
        counts = np.diff(indptr[t0:t1 + 1])
        tok = np.repeat(np.arange(t1 - t0), counts)
        g = indices[indptr[t0]:indptr[t1]]
        site = np.searchsorted(offs, g, side="right") - 1
        for si, (_, vn, c) in enumerate(sites):
            m = torch.zeros(B * S, c, device=DEV)
            sel = site == si
            m[torch.tensor(tok[sel], device=DEV), torch.tensor(g[sel] - offs[si], device=DEV)] = 1.0
            target.site(vn).mask = m.view(B, S, c)
        with torch.no_grad():
            kls.append(kl_per_pos(target(b), tgt).cpu().numpy())
    for _, vn, _ in sites:
        target.site(vn).mask = None
    kl = np.concatenate(kls)
    print(json.dumps({"sets": spec, "rows": rows, "kl": float(kl.mean()), "l0": float(len(indices) / (len(indptr) - 1))}))


def cmd_ciflops() -> None:
    """Multiply–adds per token of the CI network (input projection, 8 bidirectional blocks with
    their attention over 512 tokens, output head), from the published shapes."""
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    sd = {k.split("_global_ci_fn.", 1)[1]: v for k, v in raw.items() if "_global_ci_fn." in k}
    T = 512
    total = sd["_input_projector.W"].numel() + sd["_output_head.W"].numel()
    blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("_blocks."))
    d = sd["_blocks.0.attn.q_proj.weight"].shape[0]
    for i in range(blocks):
        pre = f"_blocks.{i}."
        total += sum(sd[pre + f"attn.{n}.weight"].numel() for n in ("q_proj", "k_proj", "v_proj", "out_proj"))
        total += sd[pre + "mlp.0.W"].numel() + sd[pre + "mlp.2.W"].numel()
        total += 2 * T * d  # scores and the weighted sum over T keys, all heads together
    params = sum(v.numel() for v in sd.values())
    print(json.dumps({"multiply_adds_per_token": int(total), "flops_per_token": int(2 * total), "parameters": int(params), "blocks": blocks, "d_model": d}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("amps")
    a.add_argument("out", type=Path)
    a.add_argument("--rows", type=int, default=128)
    k = sub.add_parser("kl")
    k.add_argument("sets")
    k.add_argument("--rows", default="0:32")
    sub.add_parser("ciflops")
    args = p.parse_args()
    if args.cmd == "amps":
        cmd_amps(args.out, args.rows)
    elif args.cmd == "kl":
        cmd_kl(args.sets, args.rows)
    else:
        cmd_ciflops()
