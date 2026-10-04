"""E4 through the program-size decomposition (#2951): the emoticon edit made through the subcomponents that
gam_mpd::site_fit's library of h.2.mlp.down_proj turns on at emoticon colons, instead of through VPD's.

  reads    the site's reads (the MLP's hidden units) at the 310 training fire keys, the 53 eval fire keys and 2533
           non-emoticon spaced ':' positions (e4_edit_methods.py's prep.npz and colon_keys.npz), raw float64 in
           methods/decomp/ for the Rust selection:
             mpd_site_sets_2951 ~/mpd-data/engine/vpd4l_e2e_train 16 blocks.2.down_proj 1e7 methods/decomp/lib 20 \
                 methods/decomp/{fire_train,fire_eval,spaced_colon}.f64
           fits the library on the 16 training rows (val 2048-2063) and selects every input's subcomponents
  choose   per subcomponent, the share of training fire keys, ordinary training inputs and spaced ':' it is on at;
           the emoticon subcomponent is the one whose on-state carries the most information about a position being
           an emoticon colon rather than ordinary text (the largest mutual information between the two indicators,
           fire keys and training inputs weighted equally). Writes the edits through it as rank-one plans
           compile/plan_{name}.{left,right}.npy (write, read), each at unit response on the mean fire key:
             decomp_own      read: the emoticon subcomponent's own read v_c
             decomp_span{k}  read: the least mean-square change on ordinary text (metric C, as ROME) within the span
                             of the reads of the k subcomponents most informative of an emoticon colon
           write: e4_edit_methods.py's Fisher write
Then: e4_edit_methods.py dev NAMES (dev proxies), assemble_extra decomp NAME (strength solved to the headline
success), and the harness on E4_VARIANTS=decomp.

usage: venv/python e4_decomp_edit.py reads | choose
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4_edit_methods as M  # noqa: E402

D = M.OUT / "decomp"
LIB = D / "lib"
SITE = "blocks.2.down_proj"
SPAN = (2, 4, 8)


def stage_reads():
    z, c = np.load(M.OUT / "prep.npz"), np.load(M.OUT / "colon_keys.npz")
    D.mkdir(exist_ok=True)
    for name, a in (("fire_train", z["K"]), ("fire_eval", z["K_eval"]), ("spaced_colon", c["N"])):
        np.ascontiguousarray(a, dtype="<f8").tofile(D / f"{name}.f64")
        M.log(f"{name}: {a.shape}")


def on_rates(name, pieces):
    sets = json.load(open(LIB / f"{name}.sets.json"))["sets"]
    on = np.zeros(pieces)
    for s in sets:
        on[s] += 1
    return on / len(sets), len(sets)


def information(p, q):
    """Mutual information (bits) between a subcomponent's on-state and the class, the two classes equally likely:
    on at a share p of fire keys and q of ordinary inputs."""
    def h(x):
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
    return h((p + q) / 2) - (h(p) + h(q)) / 2


def stage_choose():
    v = np.fromfile(LIB / f"{SITE}.v.f64").reshape(-1, 3072)
    pieces = len(v)
    fire, nf = on_rates("fire_train", pieces)
    held, nh = on_rates("fire_eval", pieces)
    gen, ng = on_rates(f"{SITE}.train", pieces)
    colon, nc = on_rates("spaced_colon", pieces)
    info = information(fire, gen)
    order = np.argsort(-info)
    M.log(f"{pieces} subcomponents; on per input: fire {fire.sum():.1f}, ordinary {gen.sum():.1f}, spaced ':' {colon.sum():.1f}")
    print(f"{'c':>5s} {'bits':>6s} {'fire':>6s} {'eval':>6s} {'ordinary':>9s} {'spaced :':>9s}")
    for c in order[:20]:
        print(f"{c:5d} {info[c]:6.3f} {fire[c]:6.3f} {held[c]:6.3f} {gen[c]:9.4f} {colon[c]:9.4f}")
    z = np.load(M.OUT / "prep.npz")
    kbar, C = z["K"].mean(0), z["C"]
    C = C + 1e-6 * np.trace(C) / len(C) * np.eye(len(C))
    w = M.fisher_write(z)
    c0 = int(order[0])
    reads = {"decomp_own": v[c0]}
    for k in SPAN:
        B = v[order[:k]]  # k x d_in: the read span
        a = np.linalg.solve(B @ C @ B.T, B @ kbar)
        reads[f"decomp_span{k}"] = B.T @ a
    for name, u in reads.items():
        u = u / (u @ kbar)
        np.save(M.OUT / f"compile/plan_{name}.left.npy", w[:, None])
        np.save(M.OUT / f"compile/plan_{name}.right.npy", u[:, None])
        M.log(f"{name}: |u| {np.linalg.norm(u):.3g}, u'Cu {u @ C @ u:.3g}")
    json.dump({"subcomponents": pieces, "n": {"fire_train": nf, "fire_eval": nh, "ordinary": ng, "spaced_colon": nc},
               "emoticon_subcomponent": c0, "ranked": [int(c) for c in order[:max(SPAN)]],
               "rates": {int(c): {"bits": float(info[c]), "fire_train": float(fire[c]), "fire_eval": float(held[c]),
                                  "ordinary": float(gen[c]), "spaced_colon": float(colon[c])} for c in order[:20]},
               "plans": list(reads)}, open(D / "choice.json", "w"), indent=1)


if __name__ == "__main__":
    {"reads": stage_reads, "choose": stage_choose}[sys.argv[1]]()
