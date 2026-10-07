"""Starts for library_vpd's fitter on a toy of the #2951 toy gate (every toy is a language model,
train_toys.py): every block weight cut into exact rank-one slices, in the decomposition layout
library_vpd reads (explanation_battery::load_factors: config.sites h.{l}.{kind} in M's order,
{site}.V [in, C] the reads and {site}.U [C, out] the writes, W = (V U)^T): each head's rows of q,
k, v and columns of o by their exact SVD (rank at most head_dim), each MLP neuron's c_fc row and
down_proj column. A zero map (a real-valued toy's attention, the induction toy's MLP) is one zero
slice owned by the first head or neuron, since library_vpd wants every stage to hold a component.
All slices sum to M's weights, so every arm starts at M, every gate on:
  per_slice_own      every slice its own component, gated by its own read |v^T x| - tau
  grouped_own        per head one component of its q, k, v and o slices, per neuron one of its c_fc
                     row and down_proj column, gated by ||V_b^T x|| - tau over its input-side reads
  grouped_direction  the same groups, gated by g^T x + c - tau at their input-side read (g = 0,
                     c = 1, tau = 0 at the start)
The fitter's parts dump (mpd_library_mdl_2951 ... parts DIR) writes an arm's components as the
harness's parts, at the start's values when no checkpoint is there.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/toy_start.py TOY_DIR OUT
writes OUT/decomposition/ (export.json and tensors) and OUT/start.json (the three arms).
"""

import json
import sys
from pathlib import Path

import numpy as np

KINDS = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"]


def svd_slices(block):
    u, s, vt = np.linalg.svd(block, full_matrices=False)
    keep = s > 0
    return u[:, keep] * s[keep], vt[keep].T  # writes (out, r), reads (in, r)


def main():
    toy, out = Path(sys.argv[1]), Path(sys.argv[2])
    record = json.loads((toy / "export.json").read_text())
    c = record["config"]
    H, hd, L = c["n_heads"], c["head_dim"], c["n_layers"]
    W = {k: np.fromfile(toy / f"{k}.f64", dtype="<f8").reshape(v["shape"]) for k, v in record["files"].items() if k.startswith("blocks.")}
    dec = out / "decomposition"
    dec.mkdir(parents=True, exist_ok=True)
    files, sites, per_slice, grouped, direction = {}, [], [], [], []
    for l in range(L):
        # per site: its slices (writes U [C, out], reads V [in, C]) and per slice its owner
        site_slices = {}
        for kind in KINDS:
            w = W[f"blocks.{l}.{kind}"]
            reads, writes, owner = [], [], []
            if kind in ("attn.q_proj", "attn.k_proj", "attn.v_proj"):
                for h in range(H):
                    Uh, V = svd_slices(w[h * hd:(h + 1) * hd])
                    U = np.zeros((w.shape[0], Uh.shape[1]))
                    U[h * hd:(h + 1) * hd] = Uh
                    writes += list(U.T)
                    reads += list(V.T)
                    owner += [("head", h)] * U.shape[1]
            elif kind == "attn.o_proj":
                for h in range(H):
                    U, Vh = svd_slices(w[:, h * hd:(h + 1) * hd])
                    V = np.zeros((w.shape[1], Vh.shape[1]))
                    V[h * hd:(h + 1) * hd] = Vh
                    writes += list(U.T)
                    reads += list(V.T)
                    owner += [("head", h)] * U.shape[1]
            elif kind == "mlp.c_fc":
                for n in range(w.shape[0]):
                    if np.any(w[n]):
                        e = np.zeros(w.shape[0])
                        e[n] = 1.0
                        writes.append(e)
                        reads.append(w[n])
                        owner.append(("neuron", n))
            else:
                for n in range(w.shape[1]):
                    if np.any(w[:, n]):
                        e = np.zeros(w.shape[1])
                        e[n] = 1.0
                        writes.append(w[:, n])
                        reads.append(e)
                        owner.append(("neuron", n))
            if not writes:  # a zero map: one zero slice (library_vpd wants every stage to hold a component)
                writes, reads, owner = [np.zeros(w.shape[0])], [np.zeros(w.shape[1])], [("neuron", 0) if kind.startswith("mlp") else ("head", 0)]
            U = np.stack(writes)  # (C, out)
            V = np.stack(reads, 1)  # (in, C)
            assert np.allclose((V @ U).T, w, atol=1e-10 * max(1.0, np.abs(w).max())), f"{kind} slices do not sum to M"
            name = f"h.{l}.{kind}"
            sites.append(name)
            for suffix, t in [("U", U), ("V", V)]:
                np.ascontiguousarray(t, dtype="<f8").tofile(dec / f"{name}.{suffix}.f64")
                files[f"{name}.{suffix}"] = {"shape": list(t.shape)}
            site_slices[kind] = (len(sites) - 1, U, V, owner)
        for unit, count in [("head", H), ("neuron", W[f"blocks.{l}.mlp.c_fc"].shape[0])]:
            for i in range(count):
                slices = [[site, j] for kind, (site, U, V, owner) in site_slices.items() for j, o in enumerate(owner) if o == (unit, i)]
                if not slices:
                    continue
                first = slices[0]
                grouped.append({"read": {"own": first}, "tau": -1.0, "slices": slices})
                # a direction gate g^T x + c - tau at the group's input-side read, started on
                # everywhere (g = 0, c = 1, tau = 0)
                width = W[f"blocks.{l}.{KINDS[first[0] % len(KINDS)]}"].shape[1]
                direction.append({"read": {"direction": {"site": first[0], "coefficients": [0.0] * width + [1.0]}}, "tau": 0.0, "slices": slices})
                per_slice += [{"read": {"own": sl}, "tau": -1.0, "slices": [sl]} for sl in slices]
    (dec / "export.json").write_text(json.dumps({"config": {"sites": sites, "source": str(toy)}, "files": files}))
    arms = [{"arm": "per_slice_own", "components": per_slice}, {"arm": "grouped_own", "components": grouped}, {"arm": "grouped_direction", "components": direction}]
    (out / "start.json").write_text(json.dumps(arms))
    print(f"{len(per_slice)} slices, {len(grouped)} groups over {len(sites)} sites -> {out}")


if __name__ == "__main__":
    main()
