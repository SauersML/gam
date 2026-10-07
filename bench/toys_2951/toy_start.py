"""A start for library_vpd's fitter on a language-model toy (#2951 toy gate): every weight of every
block cut into exact rank-one slices, in the decomposition layout library_vpd reads
(explanation_battery::load_factors: config.sites h.{l}.{kind} in M's order, {site}.V [in, C] the
reads and {site}.U [C, out] the writes, W = (V U)^T), and one components arm: per head one
component of its q, k, v and o slices (each head's rows of q, k, v and columns of o by their exact
SVD, rank at most head_dim), per MLP neuron one component of its c_fc row and down_proj column,
each with its own gate at its input-side read, started on everywhere (tau = -1 below ||V_b^T x||).
All slices sum to M's weights, so the start is M. Also writes the same components as harness parts
(parts.json), to score the start itself.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/toy_start.py TOY_DIR OUT
writes OUT/decomposition/ (export.json and tensors), OUT/start.json (arm "heads_neurons") and
OUT/parts/ (harness parts with every gate on).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_toys import write_parts  # noqa: E402

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
    W = {k: np.fromfile(toy / f"{k}.f64", dtype="<f8").reshape(v["shape"]) for k, v in record["files"].items() if k != "tokens"}
    dec = out / "decomposition"
    dec.mkdir(parents=True, exist_ok=True)
    files, sites, components, parts = {}, [], [], []
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
            if not writes:  # a zero map: one zero slice (library_vpd wants every MLP stage to hold a component)
                writes, reads, owner = [np.zeros(w.shape[0])], [np.zeros(w.shape[1])], [("neuron", 0) if kind.startswith("mlp") else None]
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
                slices, harness = [], {}
                for kind, (site, U, V, owner) in site_slices.items():
                    mine = [j for j, o in enumerate(owner) if o == (unit, i)]
                    slices += [[site, j] for j in mine]
                    if mine:
                        harness[f"blocks.{l}.{kind}"] = (U[mine].T, V[:, mine])
                if not slices:
                    continue
                first = slices[0]
                components.append({"read": {"own": first}, "tau": -1.0, "slices": slices})
                read_op = f"blocks.{l}.{KINDS[first[0] % len(KINDS)]}"
                parts.append({"name": f"L{l} {unit} {i}", "slices": harness, "gate": {"kind": "own", "read": read_op, "tau": -1.0}})
    (dec / "export.json").write_text(json.dumps({"config": {"sites": sites, "source": str(toy)}, "files": files}))
    (out / "start.json").write_text(json.dumps([{"arm": "heads_neurons", "components": components}]))
    rows = json.loads((toy / "truth.json").read_text())["active"]["shape"][0]
    write_parts(out / "parts", parts, {"fitter": "toy start: per head and per neuron, every gate on", "active": "active.f64",
                                        "kept": ["wte", "lm_head"]})
    np.ones((rows, len(parts))).astype("<f8").tofile(out / "parts" / "active.f64")
    print(f"{len(components)} components over {len(sites)} sites -> {out}")


if __name__ == "__main__":
    main()
