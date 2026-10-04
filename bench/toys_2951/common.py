"""Known-mechanism toy suite (#2951): what every case shares.

A case is a small model whose mechanism is known, written as an engine export (the `transformer`
or `residual_mlp` kind of `gam_mpd::import`), a list of counterfactual questions (prompt edits
and native internal interventions, each with the native model's measured outcome), and its
known mechanism (`truth.json`: weight components and their activity, attention targets).

`Transformer` and `ResidualMlp` are reference forwards of exactly what the importer builds, with
the interventions of a question applied where they act:

* `{"kind": "input", "input": x}` replaces the prompt;
* `{"kind": "scale", "place": P, "factor": f}` multiplies the value at `P`;
* `{"kind": "patch", "place": P, "donor": x}` replaces the value at `P` by its value in the
  unedited model on prompt `x`;
* `{"kind": "project", "place": P, "directions": [v, ..]}` removes the span of the directions
  (each the place's width) from the value at `P`.

Places `P`: `{"node": "head", "layer": l, "head": h, "position": i}` (the head's mix of values,
before `W_O`), `{"node": "mlp", "layer": l, "position": i, "units": [..]}` (the ReLU outputs),
`{"node": "resid", "layer": l, "position": i}` (the residual after block `l`, before block 0
at `l = -1`); a missing
`position` or `units` means every one. A residual MLP has no positions.

`write_case` writes `<root>/<case>/` (export, `questions.json`, `truth.json`, `transcript.f64`) and
the held-out outcomes to `<root>/sealed/<case>.json`, read only by the scorer
(`gam_mpd::toys`, `mpd_toys_2951`). Questions carry the clean outcome (the model on the
unedited prompt) and, in the dev split only, the measured outcome; outcomes are logits per
readout (a residual MLP's output `c` as the two-class logits `(0, l_c)`).
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TOYS_ROOT", "~/mpd-data/toys_2951")).expanduser()


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def orthogonal(n, rng):
    """A Haar-random orthogonal matrix (QR of a Gaussian, signs fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


class Edits:
    """A question's interventions, applied at their places during one forward."""

    def __init__(self, model, edits):
        self.model = model
        self.scales = []
        self.patches = []
        self.projections = []
        for e in edits:
            if e["kind"] == "scale":
                self.scales.append((e["place"], float(e["factor"])))
            elif e["kind"] == "patch":
                donor = np.asarray(e["donor"], dtype=np.float64)[None]
                values = model.record(donor)
                self.patches.append((e["place"], values))
            elif e["kind"] == "project":
                q, r = np.linalg.qr(np.asarray(e["directions"], dtype=np.float64).T)
                keep = np.abs(np.diag(r)) > 1e-12 * max(1.0, np.abs(r).max())
                self.projections.append((e["place"], q[:, keep]))
            elif e["kind"] != "input":
                raise ValueError(f"unknown edit {e['kind']}")

    def apply(self, node, layer, value, head=None):
        """`value` (rows x positions x width, or rows x width) at `(node, layer, head)`, edited."""
        for place, factor in self.scales:
            if matches(place, node, layer, head):
                value = value.copy()
                value[select(place, value)] *= factor
        for place, donor in self.patches:
            if matches(place, node, layer, head):
                value = value.copy()
                index = select(place, value)
                value[index] = donor[(node, layer, head)][index]
        for place, basis in self.projections:
            if matches(place, node, layer, head):
                value = value.copy()
                index = select(place, value)
                part = value[index]
                value[index] = part - (part @ basis) @ basis.T
        return value


def matches(place, node, layer, head):
    return place["node"] == node and place["layer"] == layer and place.get("head") == head


def select(place, value):
    """The index of `place`'s coordinates in `value` (rows first, then positions if any)."""
    position = place.get("position")
    units = place.get("units")
    if value.ndim == 3:
        return (slice(None), slice(None) if position is None else position, slice(None) if units is None else units)
    return (slice(None), slice(None) if units is None else units)


class Transformer:
    """The importer's `transformer`: `x_0[j] = W_E[tok_j] + W_pos[j]`; per layer, heads then an
    optional ReLU MLP, both residual; logits `W_U x + b_U` at the readout positions. Tensors in
    export orientation: `W_E` (vocab x d), `W_pos` (n x d), `W_Q/K/V` (heads*dh x d), `W_O`
    (d x heads*dh), `W_in` (hidden x d), `W_out` (d x hidden), `W_U` (classes x d)."""

    kind = "transformer"

    def __init__(self, W_E, W_pos, layers, W_U, n_heads, d_head, readouts, causal=True, b_U=None):
        self.W_E, self.W_pos, self.layers, self.W_U, self.b_U = W_E, W_pos, layers, W_U, b_U
        self.n_heads, self.d_head, self.readouts, self.causal = n_heads, d_head, list(readouts), causal

    def forward(self, tokens, edits=(), record=None):
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        hooks = Edits(self, edits) if edits else None
        rows, n = tokens.shape
        x = self.W_E[tokens] + self.W_pos[None, :n]
        x = embedded(x, hooks, record)
        dh = self.d_head
        mask = np.tril(np.ones((n, n), dtype=bool)) if self.causal else np.ones((n, n), dtype=bool)
        for l, layer in enumerate(self.layers):
            out = np.zeros_like(x)
            for h in range(self.n_heads):
                rows_h = slice(h * dh, (h + 1) * dh)
                q = x @ layer["W_Q"][rows_h].T + layer.get("b_Q", np.zeros(self.n_heads * dh))[rows_h]
                k = x @ layer["W_K"][rows_h].T + layer.get("b_K", np.zeros(self.n_heads * dh))[rows_h]
                v = x @ layer["W_V"][rows_h].T + layer.get("b_V", np.zeros(self.n_heads * dh))[rows_h]
                scores = np.einsum("rid,rjd->rij", q, k) / np.sqrt(dh)
                scores = np.where(mask[None], scores, -np.inf)
                pattern = softmax(scores)
                z = np.einsum("rij,rjd->rid", pattern, v)
                if record is not None:
                    record[("pattern", l, h)] = pattern
                if hooks:
                    z = hooks.apply("head", l, z, head=h)
                if record is not None:
                    record[("head", l, h)] = z
                out += z @ layer["W_O"][:, rows_h].T
            x = x + out + layer.get("b_O", 0.0)
            if "W_in" in layer:
                act = np.maximum(x @ layer["W_in"].T + layer.get("b_in", 0.0), 0.0)
                if hooks:
                    act = hooks.apply("mlp", l, act)
                if record is not None:
                    record[("mlp", l, None)] = act
                x = x + act @ layer["W_out"].T + layer.get("b_out", 0.0)
            if hooks:
                x = hooks.apply("resid", l, x)
            if record is not None:
                record[("resid", l, None)] = x
        logits = x[:, self.readouts] @ self.W_U.T
        if self.b_U is not None:
            logits = logits + self.b_U
        return logits

    def record(self, tokens):
        values = {}
        self.forward(tokens, record=values)
        return values

    def tensors(self):
        t = {"W_E": self.W_E, "W_pos": self.W_pos, "W_U": self.W_U}
        if self.b_U is not None:
            t["b_U"] = self.b_U
        for l, layer in enumerate(self.layers):
            for name, value in layer.items():
                t[f"blocks.{l}.{name}"] = value
        return t

    def export(self, name, samples, generator, vocab, classes):
        d = self.W_E.shape[1]
        hidden = max((layer["W_in"].shape[0] for layer in self.layers if "W_in" in layer), default=0)
        return {
            "model": name, "kind": "transformer",
            "config": {"n_layers": len(self.layers), "n_heads": self.n_heads, "d_model": d, "d_head": self.d_head,
                       "d_mlp": hidden, "n_ctx": samples.shape[1], "causal": self.causal, "act": "relu", "normalization": None},
            "input": {"type": "token_sequence", "vocab_size": vocab, "seq_len": samples.shape[1], "generator": generator},
            "output": {"n_classes": classes, "readout_positions": self.readouts, "decode": "argmax"},
        }


class ResidualMlp:
    """The importer's `residual_mlp`: `x_0 = bits W_E + b_E`, residual ReLU blocks, logits
    `W_U x + b_U`, each output an independent Bernoulli variable read as the logits `(0, l)`.
    `W_E` is inputs x d (export orientation), `W_in` hidden x d, `W_out` d x hidden, `W_U`
    outputs x d."""

    kind = "residual_mlp"

    def __init__(self, W_E, layers, W_U, b_E=None, b_U=None):
        self.W_E, self.layers, self.W_U, self.b_E, self.b_U = W_E, layers, W_U, b_E, b_U

    def forward(self, bits, edits=(), record=None):
        bits = np.asarray(bits, dtype=np.float64)
        if bits.ndim == 1:
            bits = bits[None]
        hooks = Edits(self, edits) if edits else None
        x = bits @ self.W_E + (0.0 if self.b_E is None else self.b_E)
        x = embedded(x, hooks, record)
        for l, layer in enumerate(self.layers):
            act = np.maximum(x @ layer["W_in"].T + layer.get("b_in", 0.0), 0.0)
            if hooks:
                act = hooks.apply("mlp", l, act)
            if record is not None:
                record[("mlp", l, None)] = act
            x = x + act @ layer["W_out"].T + layer.get("b_out", 0.0)
            if hooks:
                x = hooks.apply("resid", l, x)
            if record is not None:
                record[("resid", l, None)] = x
        logit = x @ self.W_U.T + (0.0 if self.b_U is None else self.b_U)
        return np.stack([np.zeros_like(logit), logit], axis=-1)

    def record(self, bits):
        values = {}
        self.forward(bits, record=values)
        return values

    def tensors(self):
        t = {"W_E": self.W_E, "W_U": self.W_U}
        if self.b_E is not None:
            t["b_E"] = self.b_E
        if self.b_U is not None:
            t["b_U"] = self.b_U
        for l, layer in enumerate(self.layers):
            for name, value in layer.items():
                t[f"blocks.{l}.{name}"] = value
        return t

    def export(self, name, samples, generator, outputs):
        return {
            "model": name, "kind": "residual_mlp",
            "config": {"n_layers": len(self.layers), "d_model": self.W_E.shape[1], "d_mlp": max(l["W_in"].shape[0] for l in self.layers),
                       "act": "relu", "normalization": None, "n_inputs": self.W_E.shape[0]},
            "input": {"type": "real_vector", "n_inputs": self.W_E.shape[0], "generator": generator},
            "output": {"n_outputs": outputs, "decode": "each output logit"},
        }


def embedded(x, hooks, record):
    """The residual before block 0 (place `resid` at layer -1), edited and recorded."""
    if hooks:
        x = hooks.apply("resid", -1, x)
    if record is not None:
        record[("resid", -1, None)] = x
    return x


def outcome(model, question):
    """The native model's logits on a question: its prompt (or the edited prompt) under its edits."""
    prompt = question["input"]
    for e in question["edits"]:
        if e["kind"] == "input":
            prompt = e["input"]
    return model.forward(np.asarray(prompt)[None], edits=question["edits"])[0]


def put(directory, name, array):
    a = np.ascontiguousarray(np.atleast_2d(np.asarray(array, dtype=np.float64)), dtype="<f8")
    a.tofile(directory / f"{name}.f64")
    return list(a.shape)


def write_case(case, model, record, samples, questions, truth, sealed_fraction=0.5, seed=0):
    """The case's directory: export (`export.json`, tensors, `inputs.f64` = `samples`), the
    transcript (samples minus every prompt a question reads), `questions.json` (every question's
    edits and clean logits; measured logits for the dev split), `truth.json` and its pieces, and
    the held-out measured logits under `sealed/`. Each question's split is drawn per tag, so both
    splits hold every kind of question."""
    directory = ROOT / case
    directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "sealed").mkdir(parents=True, exist_ok=True)
    files = {name: {"shape": put(directory, name, value)} for name, value in model.tensors().items()}
    samples = np.asarray(samples, dtype=np.float64)
    put(directory, "inputs", samples)
    record = dict(record)
    record["samples"] = {"file": "inputs.f64", "shape": list(samples.shape)}
    record["files"] = files
    json.dump(record, open(directory / "export.json", "w"), indent=1)

    asked = set()
    for q in questions:
        asked.add(tuple(np.asarray(q["input"], dtype=np.float64).ravel()))
        for e in q["edits"]:
            for key in ("input", "donor"):
                if key in e:
                    asked.add(tuple(np.asarray(e[key], dtype=np.float64).ravel()))
    transcript = np.array([row for row in samples if tuple(row) not in asked])
    put(directory, "transcript", transcript)

    rng = np.random.default_rng(seed)
    by_tag = {}
    for q in questions:
        by_tag.setdefault(q["tag"], []).append(q)
    held = set()
    for tag, group in sorted(by_tag.items()):
        order = rng.permutation(len(group))
        for i in order[: int(round(sealed_fraction * len(group)))]:
            held.add(group[i]["id"])
    public, sealed = [], {}
    for q in questions:
        clean = model.forward(np.asarray(q["input"])[None])[0]
        measured = outcome(model, q)
        entry = dict(q)
        entry["clean"] = clean.tolist()
        entry["split"] = "held_out" if q["id"] in held else "dev"
        if q["id"] in held:
            sealed[q["id"]] = measured.tolist()
        else:
            entry["outcome"] = measured.tolist()
        public.append(entry)
    json.dump({"case": case, "kind": model.kind, "questions": public}, open(directory / "questions.json", "w"))
    json.dump({"case": case, "outcomes": sealed}, open(ROOT / "sealed" / f"{case}.json", "w"))

    pieces = []
    for c, component in enumerate(truth.get("components", [])):
        named = {}
        for tensor, piece in component.get("pieces", {}).items():
            file = f"truth.{c}.{tensor}"
            put(directory, file, piece)
            named[tensor] = file
        entry = {k: v for k, v in component.items() if k != "pieces"}
        entry["pieces"] = named
        pieces.append(entry)
    truth = dict(truth)
    truth["components"] = pieces
    json.dump(truth, open(directory / "truth.json", "w"), indent=1)
    changed = sum(int(np.argmax(np.asarray(e.get("outcome", sealed.get(e["id"]))), -1).tolist() != np.argmax(np.asarray(e["clean"]), -1).tolist()) for e in public)
    print(f"{case}: {len(public)} questions ({len(sealed)} held out, {changed} change an argmax), {len(transcript)} transcript rows -> {directory}")


def question(case, number, tag, prompt, edits):
    return {"id": f"{case}/{number:03d}", "tag": tag, "input": [float(v) for v in np.asarray(prompt).ravel()], "edits": edits}


def place(node, layer, position=None, head=None, units=None):
    p = {"node": node, "layer": int(layer)}
    if position is not None:
        p["position"] = int(position)
    if head is not None:
        p["head"] = int(head)
    if units is not None:
        p["units"] = [int(u) for u in units]
    return p


def as_list(x):
    return [float(v) for v in np.asarray(x).ravel()]


def rotate_residual(model, components, rng, heads=True, neurons=True):
    """The same function in a random basis: the residual turned by a Haar orthogonal `R`, each
    head's query/key space by one orthogonal map and its value/output space by another, and each
    MLP's units permuted and positively rescaled. `components` (dicts whose `pieces` map tensor
    names to parts summing to the model's tensors) are carried along, so they still sum to the
    turned model. Returns the residual rotation and, per layer, its units' permutation and gains."""
    R = orthogonal(model.W_E.shape[1], rng)
    plan = {"W_E": lambda a: a @ R.T, "W_pos": lambda a: a @ R.T, "W_U": lambda a: a @ R.T, "b_E": lambda a: a @ R.T}
    units = {}
    for l, layer in enumerate(model.layers):
        pre = f"blocks.{l}."
        if "W_Q" in layer:
            H, dh = model.n_heads, model.d_head
            S, T = np.zeros((H * dh, H * dh)), np.zeros((H * dh, H * dh))
            for h in range(H):
                block = slice(h * dh, (h + 1) * dh)
                S[block, block] = orthogonal(dh, rng) if heads else np.eye(dh)
                T[block, block] = orthogonal(dh, rng) if heads else np.eye(dh)
            for name, B in (("Q", S), ("K", S), ("V", T)):
                plan[pre + "W_" + name] = lambda a, B=B: B @ a @ R.T
                plan[pre + "b_" + name] = lambda a, B=B: a @ B.T
            plan[pre + "W_O"] = lambda a, B=T: R @ a @ B.T
            plan[pre + "b_O"] = lambda a: a @ R.T
        if "W_in" in layer:
            hidden = layer["W_in"].shape[0]
            perm = rng.permutation(hidden) if neurons else np.arange(hidden)
            gain = np.exp(rng.uniform(-0.5, 0.5, hidden)) if neurons else np.ones(hidden)
            plan[pre + "W_in"] = lambda a, p=perm, g=gain: (g[:, None] * a[p]) @ R.T
            plan[pre + "b_in"] = lambda a, p=perm, g=gain: g * a[..., p]
            plan[pre + "W_out"] = lambda a, p=perm, g=gain: R @ (a[:, p] / g[None, :])
            plan[pre + "b_out"] = lambda a: a @ R.T
            units[l] = (perm, gain)
    model.W_E = plan["W_E"](model.W_E)
    if isinstance(model, Transformer):
        model.W_pos = plan["W_pos"](model.W_pos)
    elif model.b_E is not None:
        model.b_E = plan["b_E"](model.b_E)
    model.W_U = plan["W_U"](model.W_U)
    for l, layer in enumerate(model.layers):
        for name in list(layer):
            layer[name] = plan[f"blocks.{l}.{name}"](layer[name])
    for component in components:
        for name in list(component["pieces"]):
            component["pieces"][name] = plan[name](component["pieces"][name])
    return R, units


def check_components(model, components):
    """The components' pieces sum to the model's tensors they name (every site tensor of a layer
    they touch), to rounding."""
    t = model.tensors()
    sums = {}
    for component in components:
        for name, piece in component["pieces"].items():
            sums[name] = sums.get(name, 0.0) + piece
    for name, total in sums.items():
        err = np.abs(total - t[name]).max()
        assert err < 1e-9 * max(1.0, np.abs(t[name]).max()), (name, err)


def head_component(model, layer, head, name, **extra):
    """Head `head` of layer `layer`: its rows of `W_Q`, `W_K`, `W_V` and columns of `W_O`."""
    rows = slice(head * model.d_head, (head + 1) * model.d_head)
    weights = model.layers[layer]
    pieces = {}
    for x in "QKV":
        m = np.zeros_like(weights[f"W_{x}"])
        m[rows] = weights[f"W_{x}"][rows]
        pieces[f"blocks.{layer}.W_{x}"] = m
    m = np.zeros_like(weights["W_O"])
    m[:, rows] = weights["W_O"][:, rows]
    pieces[f"blocks.{layer}.W_O"] = m
    return {"name": name, "layer": layer, "head": head, "pieces": pieces, **extra}


def unit_component(model, layer, units, name, **extra):
    """Units `units` of layer `layer`'s MLP: their rows of `W_in` and columns of `W_out`."""
    weights = model.layers[layer]
    units = np.asarray(units, dtype=np.int64)
    rows = np.zeros_like(weights["W_in"])
    rows[units] = weights["W_in"][units]
    cols = np.zeros_like(weights["W_out"])
    cols[:, units] = weights["W_out"][:, units]
    return {"name": name, "layer": layer, "units": units.tolist(), "pieces": {f"blocks.{layer}.W_in": rows, f"blocks.{layer}.W_out": cols}, **extra}


def renumber(components, units):
    """Each component's `units` in the turned model's numbering (`rotate_residual`'s permutations:
    new unit `i` is old unit `perm[i]`)."""
    for c in components:
        if "units" in c and c.get("layer") in units:
            inverse = np.argsort(units[c["layer"]][0])
            c["units"] = sorted(int(inverse[u]) for u in c["units"])
