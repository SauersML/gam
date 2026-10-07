"""A mech program from a patch table (qwen_patch.py / vpd_patch.py): one node per head or MLP whose
clean output, patched alone into the counterfactual run, recovers at least FRACTION of
KL(M_clean || M_counterfactual); every edge the layer order allows among them, from embed into each and
from each into logits. The docstring states the behavior and the measured recoveries; edit it by hand
to add what the measurements do not show.

  program_from_patch.py PATCH.json BEHAVIOR.json [--fraction 0.1] > examples/NAME.py
"""

import argparse
import json
import textwrap
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("patch", type=Path)
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--fraction", type=float, default=0.1)
    a = ap.parse_args()
    patch, beh = json.loads(a.patch.read_text()), json.loads(a.behavior.read_text())
    base = patch["base_bits"]
    chosen = [(c["layer"], 0, c["head"], c["recovery_bits"]) for c in patch["heads"]]
    chosen += [(c["layer"], 1, None, c["recovery_bits"]) for c in patch["mlps"]]
    chosen = sorted(c for c in chosen if c[3] >= a.fraction * base)
    names = [f"h{l}_{h}" if h is not None else f"mlp{l}" for l, _, h, _ in chosen]
    listed = ", ".join(f"{'L%d.H%d' % (l, h) if h is not None else 'layer %d MLP' % l} {r:.2f}"
                       for l, _, h, r in sorted(chosen, key=lambda c: -c[3]))
    doc = (f"Behavior {beh['id']} ({beh['model']}; M's top token is right on {100 * beh['model_accuracy']:.0f}% of "
           f"the targets): {beh['description']}\n\n"
           f"Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers "
           f"at least {100 * a.fraction:.0f}% of the {base:.2f} bits per target token between the clean and the "
           f"counterfactual next-token distributions: {listed} bits. The program lets every write among them "
           f"reach every later read.")
    lines = ['"""' + "\n".join(textwrap.wrap(doc.split("\n\n")[0], 100)) + "\n\n"
             + "\n".join(textwrap.wrap(doc.split("\n\n")[1], 100)) + '\n"""',
             "from mech import node, edges, L, embed, logits", ""]
    for name, (l, block, h, r) in zip(names, chosen):
        piece = f"L[{l}].head[{h}]" if h is not None else f"L[{l}].mlp"
        lines.append(f"{name} = node({piece})  # recovers {r:.2f} bits")
    pairs = [f"embed >> {n}" for n in names]
    for i, (n, c) in enumerate(zip(names, chosen)):
        for m, d in zip(names[i + 1 :], chosen[i + 1 :]):
            if (c[0], c[1]) < (d[0], d[1]):
                pairs.append(f"{n} >> {m}")
    pairs += [f"{n} >> logits" for n in names]
    lines += ["", "edges("] + [f"    {p}," for p in pairs] + [")", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
