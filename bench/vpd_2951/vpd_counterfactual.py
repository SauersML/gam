"""VPD's selections on the counterfactual-response episodes (#2951): for every episode of a frozen
spec (counterfactual_spec.py), the units VPD's own rule runs under that episode's native
intervention, for `mpd_counterfactual_2951 EXPORT SPEC units:LIBRARY:THIS_DIR OUT`.

VPD's rule is its causal-importance network, rounded at 0 (the published sets). It reads every
site's input at every position at once (bidirectionally, later layers included), so it cannot run
inside one forward. Two feeds:

  own     (autonomous, the headline): the rule reads VPD's own program's states. Starting from
          VPD's all-units program (every subcomponent on, no Δ-component), the program runs on the
          rule's selection and the rule reads it again, until the selection repeats (at most
          MAX_ROUNDS rounds; the rounds and the last change are recorded). A mix toward a donor
          takes the donor's states from VPD's own program at its fixed point on the clean donor.
  native  the rule reads the intervened native model's states, as VPD computes its sets on clean
          text: information from the answer, reported beside the autonomous feed to show the gap.

Writes selections.json (offsets, each episode's selection index, rounds) with the selections as
CSR (sets.indptr.u64, sets.indices.u32), and description.json with the explanation's reals (its
library, and the rule's parameters); their bits come from the same pricing every method gets.

usage: vpd_counterfactual.py SPEC.json OUT_DIR [--feed own|native] [--device cuda|mps|cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vpd_model import load_target, load_vpd, site_names, val_tokens

FRONTIER_ROW0, MAX_ROUNDS = 1024, 10

parser = argparse.ArgumentParser()
parser.add_argument("spec", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--feed", choices=("own", "native"), default="own")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
args = parser.parse_args()
out, dev = args.out, args.device
out.mkdir(parents=True, exist_ok=True)
spec = json.load(open(args.spec))
rows = spec["rows"]

target = load_target(dev)
vpd = load_vpd(target, dev)
names = site_names()  # the spec's site order: layer-major q, k, v, o, c_fc, down_proj
assert [n.split(".")[-1] for n in names[:6]] == ["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"]
offsets = np.concatenate([[0], np.cumsum([vpd.C[n] for n in names])]).astype(int)
passages = sorted({e["passage"] for e in spec["episodes"]} | {e["donor"] for e in spec["episodes"] if "donor" in e})
ids = val_tokens(max(passages) + 1, offset=FRONTIER_ROW0).to(dev)
all_on = {n: torch.ones(1, rows, vpd.C[n], device=dev) for n in names}
edits = {}
spec_dir = args.spec.parent
for name, e in spec["edits"].items():
    d_out, d_in = target.site(names[e["site"]]).W.shape
    left = np.fromfile(spec_dir / e["left"], dtype="<f8").reshape(d_out, e["rank"])
    right = np.fromfile(spec_dir / e["right"], dtype="<f8").reshape(d_in, e["rank"])
    edits[name] = (torch.tensor(left, dtype=torch.float32, device=dev), torch.tensor(right, dtype=torch.float32, device=dev))


@torch.no_grad()
def states(p: int, actions=(), donor=None, masks=None):
    """VPD's program on passage p under `actions`, every site's input and output: on the units
    `masks` names (per site, [1, rows, C]), every unit when None, the native maps when "native"."""
    vpd.clear()
    handles = []
    for n in names:
        st = target.site(n)
        st.cache_input = st.cache_output = True
    by_site = {}
    for a in actions:
        by_site.setdefault(a["site"], []).append(a)
    for k, acts in by_site.items():
        st = target.site(names[k])

        def in_fn(x, acts=acts, k=k):
            x = x.clone()
            for a in acts:
                if a["type"] == "scale_input":
                    rs = slice(None) if a["row"] is None else slice(a["row"], a["row"] + 1)
                    x[0, rs, a["cols"][0]:a["cols"][1]] *= a["scale"]
                elif a["type"] == "mix_input":
                    r = a["row"]
                    x[0, r] = (1 - a["alpha"]) * x[0, r] + a["alpha"] * donor[(k, r, False)]
            return x

        def hook(module, inputs, output, acts=acts, k=k):
            x = module.last_input
            for a in acts:
                if a["type"] == "mix_output":
                    r = a["row"]
                    output[0, r] = (1 - a["alpha"]) * output[0, r] + a["alpha"] * donor[(k, r, True)]
                elif a["type"] == "add_map":
                    left, right = edits[a["edit"]]
                    output = output + (x @ right) @ left.T
            return output

        st.in_fn = in_fn
        handles.append(st.register_forward_hook(hook))
    try:
        for n in names:
            st = target.site(n)
            st.mask = None if isinstance(masks, str) else (all_on[n] if masks is None else masks[n])
            st.delta_mask = None
        target(ids[p:p + 1])
        inputs = {n: target.site(n).last_input for n in names}
        outputs = {n: target.site(n).last_output for n in names}
    finally:
        for h in handles:
            h.remove()
        for n in names:
            target.site(n).in_fn = None
        vpd.clear()
    return inputs, outputs


def rule(inputs) -> torch.Tensor:
    """VPD's selection on site inputs: [rows, total units], CI > 0."""
    with torch.no_grad():
        ci = vpd.ci_fn(inputs)
    return torch.cat([(ci[n][0] > 0) for n in names], dim=-1)


def as_masks(on: torch.Tensor):
    return {n: on[:, offsets[k]:offsets[k + 1]].float()[None] for k, n in enumerate(names)}


def fixed_point(p: int, actions=(), donor=None):
    """The autonomous selection: VPD's rule on its own program's states, until it repeats."""
    on = rule(states(p, actions, donor)[0])
    for rounds in range(1, MAX_ROUNDS + 1):
        again = rule(states(p, actions, donor, as_masks(on))[0])
        changed = float((again != on).float().mean())
        on = again
        if changed == 0.0:
            return on, rounds, 0.0
    return on, MAX_ROUNDS, changed


own_clean = {}


def donor_states(d: int, actions):
    """The donor's states the actions read, from the program the feed runs on the clean donor."""
    if args.feed == "native":
        inputs, outputs = states(d, masks="native")
    else:
        if d not in own_clean:
            own_clean[d] = fixed_point(d)[0]
        inputs, outputs = states(d, masks=as_masks(own_clean[d]))
    return {(a["site"], a["row"], a["type"] == "mix_output"): (outputs if a["type"] == "mix_output" else inputs)[names[a["site"]]][0, a["row"]]
            for a in actions if a["type"] in ("mix_input", "mix_output")}


indptr_file = open(out / "sets.indptr.u64", "wb")
indices_file = open(out / "sets.indices.u32", "wb")
np.zeros(1, dtype="<u8").tofile(indptr_file)
total, index, selected, rounds_of = 0, {}, {}, {}
for i, e in enumerate(spec["episodes"]):
    donor = donor_states(e["donor"], e["actions"]) if "donor" in e else None
    if args.feed == "native":
        on, rounds, last = rule(states(e["passage"], e["actions"], donor, masks="native")[0]), 0, 0.0
    else:
        on, rounds, last = fixed_point(e["passage"], e["actions"], donor)
    pos, unit = np.nonzero(on.cpu().numpy())
    unit.astype("<u4").tofile(indices_file)
    (total + np.cumsum(np.bincount(pos, minlength=rows))).astype("<u8").tofile(indptr_file)
    total += len(unit)
    index[e["id"]] = i
    rounds_of[e["id"]] = [rounds, last]
    selected[e["group"]] = selected.get(e["group"], []) + [len(unit) / rows]
    if i % 100 == 0:
        print(f"{i}/{len(spec['episodes'])} episodes, {len(unit) / rows:.1f} selected per row, {rounds} rounds ({e['id']})", flush=True)
indptr_file.close()
indices_file.close()
converged = sum(1 for r, last in rounds_of.values() if last == 0.0)
json.dump({"offsets": offsets.tolist(), "episodes": index, "feed": args.feed, "rounds": rounds_of, "converged": converged,
           "method": f"VPD (goodfire/spd/runs/s-55ea3f9b): CI > 0 on {'its own program at the fixed point' if args.feed == 'own' else 'the intervened native model'}",
           "selected_per_row": {g: float(np.mean(v)) for g, v in selected.items()}}, open(out / "selections.json", "w"), indent=1)
library_reals = int(sum(vpd.C[n] * sum(target.site(n).W.shape) for n in names))
rule_reals = int(sum(p.numel() for p in vpd.ci_fn.parameters()))
json.dump({"library_reals": library_reals, "rule_reals": rule_reals, "per_word_bits": 0.0}, open(out / "description.json", "w"), indent=1)
print(f"{len(spec['episodes'])} selections ({args.feed} feed, {converged} converged), {total / (len(spec['episodes']) * rows):.1f} selected per row -> {out}")
