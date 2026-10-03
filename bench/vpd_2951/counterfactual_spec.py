"""The counterfactual-response benchmark's episodes on VPD's 4-layer Pile target (#2951),
frozen before any explanation is scored (`mpd_counterfactual_2951` in crates/gam-mpd/examples).

Every intervention is native and decomposition-free: it acts on the decoder's own quantities, so
every explanation is asked about the same physical change. Sites are numbered layer-major,
q, k, v, o, c_fc, down_proj within a layer (gam_mpd::counterfactual::site_index). A neuron is a
coordinate of the MLP down-projection's input (the GELU output); a head is a 128-column block of
the attention output projection's input. The choices below read only the native model.

Passages: the 32 frontier passages (val rows 1024..1055, export engine/vpd4l_frontier32), held
out from every fit. Passage p acts at row r_p = 128 + 8p; a mix takes the donor passage p + 1.

  activation (one row; scored from r_p on; interface rows [r_p]):
    neuron   layers p, p+2 (mod 4): the neuron with the largest |activation| at r_p, scaled by
             s in {0, 0.5, 2, 4}
    head     layers p+1, p+3: the head with the largest output norm at r_p, s in {0, 0.5, 2, 4}
    input    each kind k at layer p+k: the site's input at r_p mixed toward the donor's,
             alpha in {0.25, 0.5, 1}
    output   the same sites' outputs, alpha in {0.25, 0.5, 1}
  weight (every row; scored on every row; interface rows [128, 384]):
    neuron   layer p, the same neuron, s in {0, 0.5, 2, 4} (its down-projection column scaled)
    head     layer p+1, the same head, s in {0, 0.5, 2, 4} (its output-projection columns)
    edit     the compiled emoticon edit (vpd_native_edits.py: 2.mlp.down_proj, n = 947,
             alpha 4): W + dW
  pair (two activation-level actions at row r_p, different layers, combinations used nowhere
       else):
    neuron(p, s=0) + head(p+1, s=0);  neuron(p+2, s=4) + output mix of c_fc at layer p+3 (alpha 1);
    head(p+3, s=2) + input mix of q at layer p+1 (alpha 0.5);
    input mix of v at layer p (alpha 1) + output mix of down_proj at layer p+2 (alpha 0.5)
  clean (every row; interface rows [128, 384])

usage: counterfactual_spec.py OUT_DIR [--device cuda|mps|cpu]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vpd_model import load_target, val_tokens

FRONTIER_ROW0, PASSAGES, ROWS, LAYERS, HEAD = 1024, 32, 512, 4, 128
KINDS = ("q", "k", "v", "o", "c_fc", "down_proj")
STRENGTHS, ALPHAS = (0.0, 0.5, 2.0, 4.0), (0.25, 0.5, 1.0)
EDIT = Path.home() / "mpd-data/vpd/native_edits"
EMO_ALPHA = 4.0

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)


def site(layer: int, kind: str) -> int:
    return layer * len(KINDS) + KINDS.index(kind)


# The compiled emoticon edit, dW = left · rightᵀ.
plan = {k: [np.load(EDIT / f"plan_emo947_vpd_{k}.{s}.npy") for s in ("left", "right")] for k in ("base", "unit")}
left = np.concatenate([plan["base"][0], EMO_ALPHA * plan["unit"][0]], axis=1)
right = np.concatenate([plan["base"][1], plan["unit"][1]], axis=1)
left.astype("<f8").tofile(out / "emoticon.left.f64")
right.astype("<f8").tofile(out / "emoticon.right.f64")

target = load_target(args.device)
ids = val_tokens(PASSAGES, offset=FRONTIER_ROW0).to(args.device)
hidden_sites = [target.site(f"h.{l}.mlp.down_proj") for l in range(LAYERS)]
attended_sites = [target.site(f"h.{l}.attn.o_proj") for l in range(LAYERS)]
for s in hidden_sites + attended_sites:
    s.cache_input = True


def scale(layer, kind, row, a, b, s):
    return {"type": "scale_input", "site": site(layer, kind), "row": row, "cols": [a, b], "scale": s}


def mix(kind_of_action, layer, kind, row, alpha):
    return {"type": f"mix_{kind_of_action}", "site": site(layer, kind), "row": row, "alpha": alpha}


episodes = []
with torch.no_grad():
    for p in range(PASSAGES):
        target(ids[p:p + 1])
        r = 128 + 8 * p
        donor = (p + 1) % PASSAGES
        neuron = {l: int(hidden_sites[l].last_input[0, r].abs().argmax()) for l in range(LAYERS)}
        head = {l: int(attended_sites[l].last_input[0, r].view(-1, HEAD).norm(dim=1).argmax()) for l in range(LAYERS)}
        one = {"passage": p, "interface_rows": [r]}
        every = {"passage": p, "interface_rows": [128, 384]}
        episodes.append({"id": f"clean/{p}", "group": "clean", **every, "actions": []})
        for l in (p % 4, (p + 2) % 4):
            for s in STRENGTHS:
                episodes.append({"id": f"activation/neuron/{p}/{l}/{s}", "group": f"activation/neuron/s={s:g}", **one,
                                 "actions": [scale(l, "down_proj", r, neuron[l], neuron[l] + 1, s)]})
        for l in ((p + 1) % 4, (p + 3) % 4):
            for s in STRENGTHS:
                episodes.append({"id": f"activation/head/{p}/{l}/{s}", "group": f"activation/head/s={s:g}", **one,
                                 "actions": [scale(l, "o", r, HEAD * head[l], HEAD * (head[l] + 1), s)]})
        for k, kind in enumerate(KINDS):
            l = (p + k) % 4
            for which in ("input", "output"):
                for a in ALPHAS:
                    episodes.append({"id": f"activation/{which}/{p}/{l}.{kind}/{a}", "group": f"activation/{which}/alpha={a:g}",
                                     **one, "donor": donor, "actions": [mix(which, l, kind, r, a)]})
        l = p % 4
        for s in STRENGTHS:
            episodes.append({"id": f"weight/neuron/{p}/{l}/{s}", "group": f"weight/neuron/s={s:g}", **every,
                             "actions": [scale(l, "down_proj", None, neuron[l], neuron[l] + 1, s)]})
        l = (p + 1) % 4
        for s in STRENGTHS:
            episodes.append({"id": f"weight/head/{p}/{l}/{s}", "group": f"weight/head/s={s:g}", **every,
                             "actions": [scale(l, "o", None, HEAD * head[l], HEAD * (head[l] + 1), s)]})
        episodes.append({"id": f"weight/edit/{p}", "group": "weight/edit/emoticon", **every,
                         "actions": [{"type": "add_map", "site": site(2, "down_proj"), "edit": "emoticon"}]})
        L = lambda k: (p + k) % 4  # noqa: E731
        pairs = [
            [scale(L(0), "down_proj", r, neuron[L(0)], neuron[L(0)] + 1, 0.0), scale(L(1), "o", r, HEAD * head[L(1)], HEAD * (head[L(1)] + 1), 0.0)],
            [scale(L(2), "down_proj", r, neuron[L(2)], neuron[L(2)] + 1, 4.0), mix("output", L(3), "c_fc", r, 1.0)],
            [scale(L(3), "o", r, HEAD * head[L(3)], HEAD * (head[L(3)] + 1), 2.0), mix("input", L(1), "q", r, 0.5)],
            [mix("input", L(0), "v", r, 1.0), mix("output", L(2), "down_proj", r, 0.5)],
        ]
        for i, actions in enumerate(pairs):
            episodes.append({"id": f"pair/{i}/{p}", "group": f"pair/{i}", **one, "donor": donor, "actions": actions})

spec = {
    "rows": ROWS,
    "export": "engine/vpd4l_frontier32",
    "passages": f"val rows {FRONTIER_ROW0}..{FRONTIER_ROW0 + PASSAGES}",
    "sites": [f"blocks.{l}.{k}" for l in range(LAYERS) for k in KINDS],
    "edits": {"emoticon": {"site": site(2, "down_proj"), "rank": int(left.shape[1]), "left": "emoticon.left.f64", "right": "emoticon.right.f64",
                           "source": "vpd_native_edits.py emo947 compiled: base + 4 unit"}},
    "episodes": episodes,
}
text = json.dumps(spec, indent=1)
(out / "spec.json").write_text(text)
print(f"{len(episodes)} episodes -> {out / 'spec.json'} (sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]})")
