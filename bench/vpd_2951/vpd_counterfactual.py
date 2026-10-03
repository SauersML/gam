"""Counterfactual-response episodes for VPD's published decomposition (#2951): writes the
episodes, VPD's own selections under each intervention, and the emoticon edit, for
`mpd_counterfactual_2951` (crates/gam-mpd/examples) to score against the native model.

The declared intervention family, fixed before any method is scored, on the 32 frontier passages
(val rows 1024..1055):
  clean       one episode per passage, every row scored;
  unit        at rows 128 and 384 of each passage, at six sites (each kind once, the layer
              (passage + kind) mod 4): the selected unit with the largest output ‖u_c‖·|v_cᵀx|
              removed (scale 0) and scaled by 0.5 and by 2, and the unselected unit with the
              largest output removed; natively W ← W + (s − 1) u_c v_cᵀ at that row; scored from
              that row on;
  input       at the same rows and sites, the site's input replaced by its input at the same row
              of the next passage; scored from that row on;
  edit        the compiled emoticon edit (vpd_native_edits.py: 2.mlp.down:2359's write rewritten to
              predict `o`, compiled at n = 947 into a global edit of the stored matrix, alpha 4):
              natively W + ΔW at every row; in VPD's terms the unit's write replaced by
              sign·alpha·u_o; every row scored.

VPD's selection rule is its causal-importance function on the model's site inputs, rounded at 0
(the published sets): under an intervention it reads the intervened native model's site inputs,
as it reads the clean model's on clean text.

usage: vpd_counterfactual.py OUT_DIR [--device cuda|mps|cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vpd_model import load_target, load_vpd, site_names, val_tokens

FRONTIER_ROW0, PASSAGES, ROWS = 1024, 32, 512
EPISODE_ROWS = (128, 384)
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
EDIT = Path.home() / "mpd-data/vpd/native_edits"
EMO_SITE, EMO_UNIT, EMO_ALPHA, O_TOK = "h.2.mlp.down_proj", 2359, 4.0, 80

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
args = parser.parse_args()
out, dev = args.out, args.device
out.mkdir(parents=True, exist_ok=True)

target = load_target(dev)
vpd = load_vpd(target, dev)
names = site_names()
offsets = np.concatenate([[0], np.cumsum([vpd.C[n] for n in names])]).astype(int)
U = {n: target.site(n).U for n in names}  # [C, d_out]
V = {n: target.site(n).V for n in names}  # [d_in, C]
unorm = {n: U[n].norm(dim=1) for n in names}
ids = val_tokens(PASSAGES, offset=FRONTIER_ROW0).to(dev)

# The emoticon edit: compiled ΔW = base + alpha·unit, and VPD's own image of it.
meta = json.load(open(EDIT / "manifest.json"))["meta"]["emoticon"]
plan = {k: [np.load(EDIT / f"plan_emo947_vpd_{k}.{s}.npy") for s in ("left", "right")] for k in ("base", "unit")}
left = np.concatenate([plan["base"][0], EMO_ALPHA * plan["unit"][0]], axis=1)
right = np.concatenate([plan["base"][1], plan["unit"][1]], axis=1)
u_o = target.wte[O_TOK].double().cpu()
write = (meta["sign_947"] * EMO_ALPHA * u_o / u_o.norm()).numpy()
left.astype("<f8").tofile(out / "edit.left.f64")
right.astype("<f8").tofile(out / "edit.right.f64")
write[None, :].astype("<f8").tofile(out / "edit.write.f64")
delta_w = torch.tensor(left @ right.T, dtype=torch.float32, device=dev)


@torch.no_grad()
def run(p: int, setup=None):
    """The native forward of passage p under an intervention installed by `setup` (which returns
    its teardown): every site's input rows and VPD's selection on them."""
    vpd.clear()
    for n in names:
        target.site(n).cache_input = True
    teardown = setup() if setup else None
    try:
        target(ids[p:p + 1])
        acts = {n: target.site(n).last_input for n in names}
    finally:
        if teardown:
            teardown()
        vpd.clear()
    ci = vpd.ci_fn(acts)
    on = torch.cat([(ci[n][0] > 0) for n in names], dim=-1).cpu().numpy()  # [ROWS, total]
    return {n: a[0] for n, a in acts.items()}, on


def unit_setup(name, row, unit, scale):
    st = target.site(name)

    def setup():
        def hook(module, inputs, output):
            x = inputs[0][0, row]
            output[0, row] += (scale - 1.0) * (x @ V[name][:, unit]) * U[name][unit]
            return output

        handle = st.register_forward_hook(hook)
        return handle.remove

    return setup


def input_setup(name, row, patch):
    st = target.site(name)

    def setup():
        def patch_row(x):
            x = x.clone()
            x[0, row] = patch
            return x

        st.in_fn = patch_row

        def teardown():
            st.in_fn = None

        return teardown

    return setup


def edit_setup():
    st = target.site(EMO_SITE)

    def setup():
        st.W = st.W + delta_w

        def teardown():
            st.W = st.W - delta_w

        return teardown

    return setup


indptr_file = open(out / "sets.indptr.u64", "wb")
indices_file = open(out / "sets.indices.u32", "wb")
np.zeros(1, dtype="<u8").tofile(indptr_file)
written = [0, 0]  # selections, indices


def save(on: np.ndarray) -> int:
    pos, unit = np.nonzero(on)
    unit.astype("<u4").tofile(indices_file)
    (written[1] + np.cumsum(np.bincount(pos, minlength=ROWS))).astype("<u8").tofile(indptr_file)
    written[1] += len(unit)
    written[0] += 1
    return written[0] - 1


episodes = []
for p in range(PASSAGES):
    _, on = run(p)
    episodes.append({"kind": "clean", "passage": p, "sets": save(on), "l0": float(on.sum(1).mean())})
    episodes.append({"kind": "edit", "passage": p, "sets": save(run(p, edit_setup())[1])})
for p in range(PASSAGES):
    clean_inputs, clean_on = run(p)
    donor = (p + 1) % PASSAGES
    donor_inputs, _ = run(donor)
    for row in EPISODE_ROWS:
        for k, kind in enumerate(KINDS):
            layer = (p + k) % 4
            name = f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"
            site = names.index(name)
            x = clean_inputs[name][row]
            size = (unorm[name] * (x @ V[name]).abs()).cpu().numpy()
            selected = clean_on[row, offsets[site]:offsets[site + 1]]
            common = {"passage": p, "row": row, "site": site, "site_name": name}
            choices = []
            if selected.any():
                c = int(np.argmax(np.where(selected, size, -1.0)))
                choices += [(c, s, True) for s in (0.0, 0.5, 2.0)]
            if (~selected).any():
                c = int(np.argmax(np.where(selected, -1.0, size)))
                choices.append((c, 0.0, False))
            for c, s, chosen in choices:
                _, on = run(p, unit_setup(name, row, c, s))
                episodes.append({"kind": "unit", **common, "unit": c, "scale": s, "selected": chosen,
                                 "output_norm": float(size[c]), "sets": save(on)})
            patch = donor_inputs[name][row]
            _, on = run(p, input_setup(name, row, patch))
            episodes.append({"kind": "input", **common, "donor": donor, "sets": save(on)})
    print(f"passage {p}: {len(episodes)} episodes, {written[1] / (written[0] * ROWS):.1f} selected per row", flush=True)
indptr_file.close()
indices_file.close()
spec = {
    "rows": ROWS, "offsets": offsets.tolist(), "method": "VPD published (goodfire/spd/runs/s-55ea3f9b), CI > 0",
    "edit": {"site": names.index(EMO_SITE), "rank": int(left.shape[1]), "left": "edit.left.f64", "right": "edit.right.f64",
             "unit": EMO_UNIT, "write": "edit.write.f64", "alpha": EMO_ALPHA},
    "episodes": episodes,
}
json.dump(spec, open(out / "episodes.json", "w"), indent=1)
print(f"{len(episodes)} episodes, {written[0]} selections -> {out}")
