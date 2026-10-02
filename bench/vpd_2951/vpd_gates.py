"""Gate laws for VPD-4L (#2951): data for `gam_mpd::gates` and the scoring of the laws it fits.

Subcommands:
  amps OUT_DIR [--rows N]   each subcomponent's read amplitude a_j = v_jᵀx on the clean native forward,
                            for the first N rows of frontier/masks_vpd4l.npz (val rows 1024..; the first
                            32 are the eval passages): per site OUT_DIR/{site}.npy, float16 [C, N·512]
                            (one subcomponent's amplitudes contiguous), sites and subcomponents in the
                            masked driver's numbering (library site names sorted), plus OUT_DIR/sites.txt.

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("amps")
    a.add_argument("out", type=Path)
    a.add_argument("--rows", type=int, default=128)
    args = p.parse_args()
    if args.cmd == "amps":
        cmd_amps(args.out, args.rows)
