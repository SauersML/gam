#!/usr/bin/env python3
"""Run ON MSI. Subsample rows from L17_train.f32.npy and compute exact
full-data per-dim mean / std, so centering does not depend on the known-bad
tier0.json per_dim_mean. Writes a compact slice + stats for pulling to laptop.
"""
import numpy as np, json, time
import argparse, os

# The MSI scratch root holding L17_train.f32.npy / tier0.json is supplied at run
# time (cluster paths are not committed to the repo); pass --root <dir> on MSI.
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", default="./msae_l17", help="dir with L17_train.f32.npy + tier0.json")
ap.add_argument("--n-sub", type=int, default=40000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ROOT = args.root
SRC = f"{ROOT}/L17_train.f32.npy"
OUT_DIR = f"{ROOT}/spectrometer_slice"
os.makedirs(OUT_DIR, exist_ok=True)

N_SUB = args.n_sub
SEED = args.seed

t0 = time.time()
mm = np.load(SRC, mmap_mode="r")
n, p = mm.shape
print(f"src shape {mm.shape} dtype {mm.dtype}", flush=True)

# Exact full-data mean/std in a streaming pass (chunked over rows).
s1 = np.zeros(p, dtype=np.float64)
s2 = np.zeros(p, dtype=np.float64)
CH = 50000
for i in range(0, n, CH):
    blk = np.asarray(mm[i:i+CH], dtype=np.float64)
    s1 += blk.sum(0)
    s2 += (blk * blk).sum(0)
mean = s1 / n
var = s2 / n - mean * mean
var = np.maximum(var, 0.0)
std = np.sqrt(var)
print(f"mean/std pass done {time.time()-t0:.1f}s  ||mean||={np.linalg.norm(mean):.4f}", flush=True)

# Seeded random-row subsample.
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(n, size=N_SUB, replace=False))
sub = np.asarray(mm[idx], dtype=np.float32)  # fancy index on memmap
print(f"subsample {sub.shape} {time.time()-t0:.1f}s", flush=True)

np.save(f"{OUT_DIR}/L17_sub40k.npy", sub)
np.save(f"{OUT_DIR}/L17_mean.npy", mean.astype(np.float64))
np.save(f"{OUT_DIR}/L17_std.npy", std.astype(np.float64))
# global rms scale over non-rogue dims (match tier0 convention for reporting)
try:
    tier0 = json.load(open(f"{ROOT}/tier0.json"))
    rogue = tier0.get("rogue_dims", [])
except Exception:
    rogue = []
mask = np.ones(p, bool);
for r in rogue:
    if 0 <= r < p: mask[r] = False
global_rms_scale = float(np.sqrt(var[mask].mean()))
meta = dict(n_total=int(n), p=int(p), n_sub=N_SUB, seed=SEED,
            rogue_dims=list(map(int, rogue)),
            global_rms_scale=global_rms_scale,
            mean_norm=float(np.linalg.norm(mean)))
json.dump(meta, open(f"{OUT_DIR}/meta.json", "w"), indent=1)
print("META", json.dumps(meta), flush=True)
print("DONE", time.time()-t0, flush=True)
