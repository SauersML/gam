"""Exact rank-one libraries of the 4L Pile target's 24 maps that need no training (#2951), for the
masked driver's `library:DIR` start, beside VPD's 38,912 subcomponents:

  neurons  the model's own units: a map writing a layer of units (c_fc, and q/k/v, whose outputs are
           the attention's coordinates) is split by output unit, u = e_i, v = W[i, :]; a map reading
           one (down_proj, o_proj) by input unit, v = e_j, u = W[:, j]
  svd      each map's singular pieces, u = sqrt(s) p, v = sqrt(s) q (float64, scipy on OpenBLAS)
  random   a Haar-random orthonormal basis of each map's input, v = r_j, u = W r_j (seeded per map)

Each sums to the map exactly (float64). Writes DIR/{site}.v.f64 (pieces x d_in), DIR/{site}.u.f64
(pieces x d_out), raw little-endian, and DIR/manifest.json, sites named as the driver names them.
usage: vpd_baseline_libraries.py {neurons|svd|random} DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sl
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpd_model import TARGET_DIR, site_names  # noqa: E402

KINDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "c_fc", "down_proj": "down_proj"}
kind, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sd = load_file(str(TARGET_DIR / "model_step_99999.safetensors"))
manifest = {}
for si, n in enumerate(site_names()):
    layer, _, k = n.split(".")[1:]
    name = f"blocks.{layer}.{KINDS[k]}"
    W = sd[f"{n}.weight"].astype(np.float64)  # [d_out, d_in]
    d_out, d_in = W.shape
    if kind == "neurons":
        if k in ("down_proj", "o_proj"):
            V, U = np.eye(d_in), W.T.copy()
        else:
            V, U = W.copy(), np.eye(d_out)
    elif kind == "svd":
        P, s, Qt = sl.svd(W, full_matrices=False, lapack_driver="gesvd")
        r = np.sqrt(s)
        V, U = Qt * r[:, None], (P * r).T
    elif kind == "random":
        R, _ = sl.qr(np.random.default_rng(0x2951 + si).standard_normal((d_in, d_in)))
        V, U = R.T.copy(), (W @ R).T
    else:
        raise SystemExit(f"unknown kind {kind}")
    err = np.abs(U.T @ V - W).max() / np.abs(W).max()
    assert err < 1e-10, (name, err)
    np.ascontiguousarray(V).astype("<f8").tofile(out / f"{name}.v.f64")
    np.ascontiguousarray(U).astype("<f8").tofile(out / f"{name}.u.f64")
    manifest[name] = {"vpd": n, "pieces": int(V.shape[0]), "d_in": d_in, "d_out": d_out, "max_rel_error": float(err)}
    print(f"{name}: {V.shape[0]} pieces, max relative error {err:.1e}", flush=True)
json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
print(f"{kind}: {sum(m['pieces'] for m in manifest.values())} pieces")
