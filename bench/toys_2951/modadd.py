"""Case `modadd`: the trained one-layer transformer for `a + b mod 31` (`~/mpd-data/engine/p31_s0_generic`).

Known mechanism (Nanda et al. 2023): the embedding writes `cos/sin(2 pi k a / p)` for a few key
frequencies `k`, attention moves both operands to the `=` position, each MLP unit reads one key
frequency's plane (its activation's Fourier power on `a`, `b` and `a + b` is concentrated on one
`k`), and the unembedding reads `cos(2 pi k (a + b - c) / p)`. Truth: one weight component per key
frequency (its units' rows of `W_in` and columns of `W_out`), the rest of the MLP, and attention.
"""
import json

import numpy as np

from common import Transformer, place, question, write_case

SOURCE = __import__("pathlib").Path("~/mpd-data/engine/p31_s0_generic").expanduser()
P = 31
EQ = 31


def load():
    record = json.load(open(SOURCE / "export.json"))
    t = {name: np.fromfile(SOURCE / f"{name}.f64", dtype="<f8").reshape(spec["shape"]) for name, spec in record["files"].items()}
    layer = {k.split(".", 2)[2]: (v[0] if k.split(".")[-1].startswith("b_") else v) for k, v in t.items() if k.startswith("blocks.0.")}
    return Transformer(t["W_E"], t["W_pos"], [layer], t["W_U"], n_heads=1, d_head=32, readouts=[2], causal=True)


def frequencies(model):
    """Per unit: its key frequency (the `k` holding most of its activation's non-constant Fourier
    power over `(a, b)`) and that share."""
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    tokens = np.stack([a.ravel(), b.ravel(), np.full(P * P, EQ)], 1)
    act = model.record(tokens)[("mlp", 0, None)][:, 2].reshape(P, P, -1)
    spectrum = np.abs(np.fft.fft2(act, axes=(0, 1))) ** 2
    spectrum[0, 0] = 0.0
    total = spectrum.sum((0, 1))
    power = np.zeros((P // 2 + 1, act.shape[2]))
    for k in range(1, P // 2 + 1):
        cells = {(x, y) for x in (0, k, P - k) for y in (0, k, P - k)} - {(0, 0)}
        power[k] = sum(spectrum[x, y] for x, y in cells)
    share = power / np.maximum(total, 1e-300)
    return share.argmax(0), share.max(0)


def main():
    model = load()
    unit_frequency, share = frequencies(model)
    counts = np.bincount(unit_frequency[share > 0.5], minlength=P // 2 + 1)
    key = [int(k) for k in np.nonzero(counts >= 3)[0]]
    clusters = {k: np.nonzero((unit_frequency == k) & (share > 0.5))[0] for k in key}
    other = np.setdiff1d(np.arange(len(share)), np.concatenate(list(clusters.values())))
    print("key frequencies", key, "units per frequency", [len(clusters[k]) for k in key], "other", len(other))
    layer = model.layers[0]
    components = []
    for k, units in list(clusters.items()) + [("other", other)]:
        rows = np.zeros_like(layer["W_in"])
        rows[units] = layer["W_in"][units]
        cols = np.zeros_like(layer["W_out"])
        cols[:, units] = layer["W_out"][:, units]
        components.append({"name": f"frequency {k}" if k != "other" else "other units", "units": [int(u) for u in units],
                           "pieces": {"blocks.0.W_in": rows, "blocks.0.W_out": cols}})
    components.append({"name": "attention: move a and b to =", "pieces": {f"blocks.0.W_{x}": layer[f"W_{x}"] for x in "QKVO"}})

    rng = np.random.default_rng(31)
    pair = lambda: [int(rng.integers(P)), int(rng.integers(P)), EQ]
    questions = []
    n = 0

    def add(tag, prompt, edits):
        nonlocal n
        questions.append(question("modadd", n, tag, prompt, edits))
        n += 1

    for _ in range(12):
        x = pair()
        y = list(x)
        y[int(rng.integers(2))] = int(rng.integers(P))
        add("input", x, [{"kind": "input", "input": y}])
    for k in key:
        for _ in range(4):
            add("ablate frequency", pair(), [{"kind": "scale", "place": place("mlp", 0, 2, units=clusters[k]), "factor": 0.0}])
            rest = np.setdiff1d(np.arange(len(share)), clusters[k])
            add("keep one frequency", pair(), [{"kind": "scale", "place": place("mlp", 0, 2, units=rest), "factor": 0.0}])
            add("patch frequency", pair(), [{"kind": "patch", "place": place("mlp", 0, 2, units=clusters[k]), "donor": pair()}])
    for _ in range(10):
        add("patch mlp", pair(), [{"kind": "patch", "place": place("mlp", 0, 2), "donor": pair()}])
        add("patch head", pair(), [{"kind": "patch", "place": place("head", 0, 2, head=0), "donor": pair()}])
        add("scale head", pair(), [{"kind": "scale", "place": place("head", 0, 2, head=0), "factor": float(rng.choice([0.0, 0.5, 2.0]))}])
        u = int(rng.choice(np.concatenate(list(clusters.values()))))
        add("scale unit", pair(), [{"kind": "scale", "place": place("mlp", 0, 2, units=[u]), "factor": float(rng.choice([0.0, 4.0]))}])
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    samples = np.stack([a.ravel(), b.ravel(), np.full(P * P, EQ)], 1)
    record = model.export("modadd_p31", samples, f"every (a, b) in Z_{P}^2 followed by the = token {EQ}", vocab=P + 1, classes=P)
    truth = {"mechanism": "Fourier multiplication: units grouped by key frequency", "key_frequencies": key, "components": components}
    write_case("modadd", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
