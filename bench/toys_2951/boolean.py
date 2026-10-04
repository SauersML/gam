"""Case `boolean`: a two-layer residual ReLU network computing four gates of six input bits
`(a, b, c, d, e, f)`, written by hand and turned into a random basis. Each output is a Bernoulli
logit `8 (o - 1/2)` of its gate's value `o`.

* `y0 = a OR b`, redundantly: units `A = relu(a)` and `B = relu(b)` both vote, and a saturating
  pair `relu(v) - relu(v - 1)` reads the votes, so with `a = b = 1` neither path is necessary;
* `y1 = c OR d`, the backup preempted through the unit: `C = relu(c)` votes and writes its
  output; layer 1's `D = relu(d - C)` reads `C`'s output and writes `y1` itself, so silencing `C` with `c = d = 1`
  releases `D` and the output stands;
* `y2 = e OR f`, preempted through the input: `F = relu(f - e)` reads `e` itself, so silencing
  `E = relu(e)` with `e = f = 1` turns the output off although `f = 1`;
* `y3 = a AND c`: `relu(a + c - 1)`.
"""
import itertools

import numpy as np

from common import ResidualMlp, as_list, place, question, renumber, rotate_residual, unit_component, write_case

GAIN = 8.0


def build():
    names = ["a", "b", "c", "d", "e", "f", "v0", "c_out", "v1", "v2", "o0", "o1", "o2", "o3"]
    at = {n: i for i, n in enumerate(names)}
    d = len(names)
    W_E = np.zeros((6, d))
    for i, n in enumerate("abcdef"):
        W_E[i, at[n]] = 1.0
    # layer 0 units: A, B, C, E, F, AND
    l0 = [("A", {"a": 1}, 0, {"v0": 1}), ("B", {"b": 1}, 0, {"v0": 1}), ("C", {"c": 1}, 0, {"c_out": 1, "v1": 1}),
          ("E", {"e": 1}, 0, {"v2": 1}), ("F", {"f": 1, "e": -1}, 0, {"v2": 1}), ("AND", {"a": 1, "c": 1}, -1, {"o3": 1})]
    # layer 1 units: D (reads C's output), saturating pairs for v0, v1, v2
    l1 = [("D", {"d": 1, "c_out": -1}, 0, {"o1": 1})]
    for v, o in (("v0", "o0"), ("v1", "o1"), ("v2", "o2")):
        l1.append((f"sat {v} +", {v: 1}, 0, {o: 1}))
        l1.append((f"sat {v} -", {v: 1}, -1, {o: -1}))
    layers, unit_names = [], []
    for spec in (l0, l1):
        W_in, b_in, W_out = np.zeros((len(spec), d)), np.zeros(len(spec)), np.zeros((d, len(spec)))
        for u, (name, reads, bias, writes) in enumerate(spec):
            for n, w in reads.items():
                W_in[u, at[n]] = w
            b_in[u] = bias
            for n, w in writes.items():
                W_out[at[n], u] = w
        layers.append({"W_in": W_in, "b_in": b_in, "W_out": W_out})
        unit_names.append([s[0] for s in spec])
    W_U = np.zeros((4, d))
    for k in range(4):
        W_U[k, at[f"o{k}"]] = GAIN
    model = ResidualMlp(W_E, layers, W_U, b_U=np.full(4, -GAIN / 2))
    return model, unit_names


def main():
    rng = np.random.default_rng(6)
    model, unit_names = build()
    inputs = np.array(list(itertools.product([0.0, 1.0], repeat=6)))
    truth_out = np.stack([np.maximum(inputs[:, 0], inputs[:, 1]), np.maximum(inputs[:, 2], inputs[:, 3]),
                          np.maximum(inputs[:, 4], inputs[:, 5]), inputs[:, 0] * inputs[:, 2]], 1)
    assert ((model.forward(inputs)[..., 1] > 0) == (truth_out > 0.5)).all()
    acts = model.record(inputs)
    components = []
    for l, names in enumerate(unit_names):
        for u, name in enumerate(names):
            active = np.nonzero(acts[("mlp", l, None)][:, u] > 0)[0]
            components.append(unit_component(model, l, [u], f"unit {name}", active=[int(r) for r in active]))
    embedding = {"name": "input embedding", "pieces": {"W_E": model.W_E.copy()}}
    components.append(embedding)
    _, units = rotate_residual(model, components, rng)
    renumber(components, units)
    assert ((model.forward(inputs)[..., 1] > 0) == (truth_out > 0.5)).all()
    unit = {c["name"][5:]: (c["layer"], c["units"][0]) for c in components if c["name"].startswith("unit ")}

    def bits(**on):
        return [float(on.get(n, 0)) for n in "abcdef"]

    def ablate(name, factor=0.0):
        l, u = unit[name]
        return {"kind": "scale", "place": place("mlp", l, units=[u]), "factor": factor}

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("boolean", n, tag, x, edits))
        n += 1

    noise = lambda: {k: int(rng.integers(2)) for k in "abcdef"}
    for _ in range(6):
        add("redundant: ablate A with a = b = 1", bits(**{**noise(), "a": 1, "b": 1}), [ablate("A")])
        add("redundant: ablate B with a = b = 1", bits(**{**noise(), "a": 1, "b": 1}), [ablate("B")])
        add("necessary: ablate A with a = 1, b = 0", bits(**{**noise(), "a": 1, "b": 0}), [ablate("A")])
        add("redundant: ablate A and B with a = b = 1", bits(**{**noise(), "a": 1, "b": 1}), [ablate("A"), ablate("B")])
        add("preempted through the unit: ablate C with c = d = 1", bits(**{**noise(), "c": 1, "d": 1}), [ablate("C")])
        add("preempted through the unit: ablate D with c = d = 1", bits(**{**noise(), "c": 1, "d": 1}), [ablate("D")])
        add("necessary: ablate C with c = 1, d = 0", bits(**{**noise(), "c": 1, "d": 0}), [ablate("C")])
        add("preempted through the input: ablate E with e = f = 1", bits(**{**noise(), "e": 1, "f": 1}), [ablate("E")])
        add("preempted through the input: ablate F with e = f = 1", bits(**{**noise(), "e": 1, "f": 1}), [ablate("F")])
        add("backup: ablate F with e = 0, f = 1", bits(**{**noise(), "e": 0, "f": 1}), [ablate("F")])
        add("and gate: ablate AND", bits(**{**noise(), "a": 1, "c": 1}), [ablate("AND")])
        add("patch C from c = 0", bits(**{**noise(), "c": 1, "d": int(rng.integers(2))}), [{"kind": "patch", "place": place("mlp", unit["C"][0], units=[unit["C"][1]]), "donor": bits(**noise())}])
        x = noise()
        y = dict(x)
        flip = "abcdef"[int(rng.integers(6))]
        y[flip] = 1 - y[flip]
        add("input: flip one bit", bits(**x), [{"kind": "input", "input": bits(**y)}])
        add("scale a unit", bits(**noise()), [ablate(list(unit)[int(rng.integers(len(unit)))], float(rng.choice([0.5, 2.0])))])
    record = model.export("boolean_gates", inputs, "every point of {0, 1}^6", outputs=4)
    truth = {"mechanism": __doc__.split("\n\n", 1)[1].strip(), "components": components, "probe": inputs.tolist()}
    write_case("boolean", model, record, inputs, questions, truth)


if __name__ == "__main__":
    main()
