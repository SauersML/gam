"""A mech program from a patch table (qwen_patch.py / vpd_patch.py): one node per head or MLP whose
clean output, patched alone into the counterfactual run, recovers at least FRACTION of
KL(M_clean || M_counterfactual); every edge the layer order allows among them, from embed into each and
from each into logits. The docstring states the behavior and the measured recoveries; edit it by hand
to add what the measurements do not show. --neurons TABLE.json[:K] (vpd_neurons.py, repeatable)
replaces that layer's MLP by its neurons that pay for their opaque price (the table's best k, or K), and
--head-price P keeps only heads recovering more than P bits per target token (a vpd4l head's opaque
price is 0.28 at N = 2^24).

  program_from_patch.py PATCH.json BEHAVIOR.json [--fraction 0.1] [--neurons T.json[:K] ...] > examples/NAME.py
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
    ap.add_argument("--neurons", action="append", default=[], help="TABLE.json[:K] (K: neurons kept, default the table's best k)")
    ap.add_argument("--head-price", type=float, default=0.0)
    ap.add_argument("--mlp-price", type=float, default=0.0, help="whole MLPs (no --neurons table) above this only")
    a = ap.parse_args()
    tables, keep = {}, {}
    for spec in a.neurons:
        path, _, k = spec.partition(":")
        t = json.loads(Path(path).read_text())
        tables[t["layer"]], keep[t["layer"]] = t, int(k) if k else t["best"]["k"]
    patch, beh = json.loads(a.patch.read_text()), json.loads(a.behavior.read_text())
    base = patch["base_bits"]
    chosen = [(c["layer"], 0, c["head"], c["recovery_bits"]) for c in patch["heads"]]
    chosen += [(c["layer"], 1, None, c["recovery_bits"]) for c in patch["mlps"]]
    price = lambda c: a.head_price if c[2] is not None else a.mlp_price  # noqa: E731
    chosen = sorted(c for c in chosen if c[3] >= a.fraction * base and c[3] > price(c)
                    or (c[2] is None and c[0] in tables))
    neurons = {l: sorted(t["order"][: keep[l]]) for l, t in tables.items()}
    names = [f"h{l}_{h}" if h is not None else f"mlp{l}" for l, _, h, _ in chosen]
    listed = ", ".join(f"{'L%d.H%d' % (l, h) if h is not None else 'layer %d MLP' % l} {r:.2f}"
                       for l, _, h, r in sorted(chosen, key=lambda c: -c[3]))
    priced = (f", and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24): "
              f"{a.head_price:.2f} for a head, {a.mlp_price:.2f} for a whole MLP; an MLP may instead enter as the "
              f"neurons that pay for themselves" if a.head_price or a.mlp_price else "")
    doc = (f"Behavior {beh['id']} ({beh['model']}; M's top token is right on {100 * beh['model_accuracy']:.0f}% of "
           f"the targets): {beh['description']}\n\n"
           f"Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers "
           f"at least {100 * a.fraction:.0f}% of the {base:.2f} bits per target token between the clean and the "
           f"counterfactual next-token distributions{priced}: {listed} bits. The program lets every write among "
           f"them reach every later read.")
    lines = ['"""' + "\n".join(textwrap.wrap(doc.split("\n\n")[0], 100)) + "\n\n"
             + "\n".join(textwrap.wrap(doc.split("\n\n")[1], 100)) + '\n"""',
             "from mech import node, edges, L, embed, logits", ""]
    for name, (l, block, h, r) in zip(names, chosen):
        if h is not None:
            lines.append(f"{name} = node(L[{l}].head[{h}])  # recovers {r:.2f} bits")
        elif l in neurons:
            t, k = tables[l], len(neurons[l])
            left = next(c["kl_bits"] for c in t["curve"] if c["k"] == k) if any(c["k"] == k for c in t["curve"]) else None
            lines.append(f"# layer {l}'s MLP: the {k} neurons that recover the most when patched alone (the whole MLP "
                         f"recovers {r:.2f} bits; these {k} together leave {left:.2f} of {t['base_bits']:.2f})"
                         if left is not None else f"# layer {l}'s MLP: the {k} neurons that recover the most alone")
            body = textwrap.wrap(", ".join(map(str, neurons[l])), 96)
            lines.append(f"{name} = node(L[{l}].mlp[")
            lines += [f"    {row}" for row in body]
            lines.append("])")
        else:
            lines.append(f"{name} = node(L[{l}].mlp)  # recovers {r:.2f} bits")
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
