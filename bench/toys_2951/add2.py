"""Case `add2`: two-digit addition, trained.

Prompts are `[a_1, a_0, +, b_1, b_0, =]` (digit tokens) and the model reads out the sum `a + b`
(one of 199 classes, the way a tokenizer holds "95" as one token) at `=`. Two layers of four heads
and a ReLU MLP, no norms; trained on 90% of the 10^4 pairs.

Known mechanism (the circuit-tracing paper's addition case study, and Fourier accounts of
arithmetic): the answer is assembled from a few periodic features of the operands, the ones digits'
mod-10 sum (periods dividing 10 over `a` and `b`), coarse magnitude (low frequencies) and the carry
`[a_0 + b_0 >= 10]`. Truth: the last MLP's units grouped by the frequency holding most of their
activation's Fourier power over `(a, b)` (a frequency a multiple of 10 is a ones-digit plane);
the carry and the measured interventions (an operand's ones digit removed, a digit plane projected
out of an operand's embedding) as questions.
"""
import numpy as np
import torch

from common import Transformer, place, question, unit_component, write_case

PLUS, EQ, VOCAB, N, CLASSES = 10, 11, 12, 6, 199
D, H, DH, HIDDEN, LAYERS = 128, 4, 32, 512, 2


def tokens(a, b):
    return np.array([a // 10, a % 10, PLUS, b // 10, b % 10, EQ])


def train(pairs, seed=0, steps=8000):
    torch.manual_seed(seed)
    g = lambda *s: torch.nn.Parameter(torch.randn(*s, dtype=torch.float64) / np.sqrt(s[-1]))
    params = {"W_E": g(VOCAB, D), "W_pos": g(N, D), "W_U": g(CLASSES, D)}
    for l in range(LAYERS):
        for x in "QKV":
            params[f"{l}.W_{x}"] = g(H * DH, D)
        params[f"{l}.W_O"] = g(D, H * DH)
        params[f"{l}.W_in"] = g(HIDDEN, D)
        params[f"{l}.b_in"] = torch.nn.Parameter(torch.zeros(HIDDEN, dtype=torch.float64))
        params[f"{l}.W_out"] = g(D, HIDDEN)
    opt = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=0.1)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    batch = torch.from_numpy(np.array([tokens(a, b) for a, b in pairs]))
    labels = torch.tensor([a + b for a, b in pairs])
    rng = np.random.default_rng(seed)
    for step in range(steps):
        pick = torch.from_numpy(rng.choice(len(pairs), size=1024, replace=False))
        x = params["W_E"][batch[pick]] + params["W_pos"]
        for l in range(LAYERS):
            out = 0
            for h in range(H):
                rows = slice(h * DH, (h + 1) * DH)
                q, k, v = (x @ params[f"{l}.W_{n}"][rows].T for n in "QKV")
                a = torch.softmax(((q @ k.transpose(1, 2)) / np.sqrt(DH)).masked_fill(~mask, -torch.inf), -1)
                out = out + (a @ v) @ params[f"{l}.W_O"][:, rows].T
            x = x + out
            x = x + torch.relu(x @ params[f"{l}.W_in"].T + params[f"{l}.b_in"]) @ params[f"{l}.W_out"].T
        loss = torch.nn.functional.cross_entropy(x[:, N - 1] @ params["W_U"].T, labels[pick])
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = {k: v.detach().numpy() for k, v in params.items()}
    layers = [{"W_Q": p[f"{l}.W_Q"], "W_K": p[f"{l}.W_K"], "W_V": p[f"{l}.W_V"], "W_O": p[f"{l}.W_O"],
               "W_in": p[f"{l}.W_in"], "b_in": p[f"{l}.b_in"], "W_out": p[f"{l}.W_out"]} for l in range(LAYERS)]
    return Transformer(p["W_E"], p["W_pos"], layers, p["W_U"], n_heads=H, d_head=DH, readouts=[N - 1], causal=True), float(loss)


def frequencies(model, layer):
    """Per unit of `layer`'s MLP at `=`: the frequency `k` (of 100, folded) holding most of its
    activation's non-constant Fourier power over `(a, b)`, and that share."""
    grid = np.array([tokens(a, b) for a in range(100) for b in range(100)])
    act = model.record(grid)[("mlp", layer, None)][:, N - 1].reshape(100, 100, -1)
    spectrum = np.abs(np.fft.fft2(act, axes=(0, 1))) ** 2
    spectrum[0, 0] = 0.0
    folded = np.minimum(np.arange(100), 100 - np.arange(100))
    level = np.maximum(folded[:, None], folded[None, :])
    power = np.stack([spectrum[level == k].sum(0) for k in range(51)])
    share = power / np.maximum(spectrum.sum((0, 1)), 1e-300)
    return share.argmax(0), share.max(0), act


def main():
    rng = np.random.default_rng(2)
    everything = [(a, b) for a in range(100) for b in range(100)]
    order = rng.permutation(len(everything))
    train_pairs = [everything[i] for i in order[:9000]]
    held_pairs = [everything[i] for i in order[9000:]]
    model, loss = train(train_pairs)
    held = np.array([tokens(a, b) for a, b in held_pairs])
    accuracy = (model.forward(held).argmax(-1)[:, 0] == np.array([a + b for a, b in held_pairs])).mean()
    print(f"trained: loss {loss:.4f}, held-out accuracy {accuracy:.4f}")
    last = LAYERS - 1
    unit_frequency, share, act = frequencies(model, last)
    live = act.reshape(10000, -1).max(0) > 0
    counts = np.bincount(unit_frequency[live & (share > 0.5)], minlength=51)
    key = [int(k) for k in np.nonzero(counts >= 3)[0]]
    clusters = {k: np.nonzero(live & (unit_frequency == k) & (share > 0.5))[0] for k in key}
    other = np.setdiff1d(np.arange(HIDDEN), np.concatenate([clusters[k] for k in key]) if key else [])
    print("key frequencies (of 100; multiples of 10 are ones-digit planes)", key, "units", [len(clusters[k]) for k in key], "other", len(other))
    carry = np.array([[(a % 10 + b % 10) >= 10 for b in range(100)] for a in range(100)], dtype=np.float64).ravel()
    flat = act.reshape(10000, -1)
    centred = flat - flat.mean(0)
    corr = (centred * (carry - carry.mean())[:, None]).sum(0) / np.maximum(np.linalg.norm(centred, axis=0) * np.linalg.norm(carry - carry.mean()), 1e-300)
    print(f"units whose activation correlates with the carry above 1/2 in magnitude: {int((np.abs(corr) > 0.5).sum())}")
    components = [unit_component(model, last, units, f"frequency {k}" + (" (ones-digit plane)" if k % 10 == 0 else "")) for k, units in clusters.items()]
    components.append(unit_component(model, last, other, "other units"))

    # A digit plane of the embedding: the period-10 character k of the digit tokens' embeddings.
    digits = np.arange(10)
    plane = lambda k: [np.cos(2 * np.pi * k * digits / 10) @ model.W_E[:10], np.sin(2 * np.pi * k * digits / 10) @ model.W_E[:10]]

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("add2", n, tag, x, edits))
        n += 1

    pair = lambda: (int(rng.integers(100)), int(rng.integers(100)))
    for _ in range(8):
        a, b = pair()
        add("input: change a's ones digit", tokens(a, b), [{"kind": "input", "input": tokens(10 * (a // 10) + int(rng.integers(10)), b).tolist()}])
        a, b = pair()
        add("input: change b's tens digit", tokens(a, b), [{"kind": "input", "input": tokens(a, 10 * int(rng.integers(10)) + b % 10).tolist()}])
        a, b = pair()
        a0 = int(rng.integers(5, 10))
        add("input: make a carry", tokens(10 * (a // 10) + a0, 10 * (b // 10) + 9 - a0), [{"kind": "input", "input": tokens(10 * (a // 10) + a0, 10 * (b // 10) + 10 - a0).tolist()}])
        add("suppress a's ones digit (its residual at the start)", tokens(*pair()), [{"kind": "scale", "place": place("resid", -1, 1), "factor": 0.0}])
        add("suppress b's ones digit (its residual at the start)", tokens(*pair()), [{"kind": "scale", "place": place("resid", -1, 4), "factor": 0.0}])
        k = int(rng.integers(1, 6))
        add("project a ones-digit plane out of a's ones digit", tokens(*pair()), [{"kind": "project", "place": place("resid", -1, 1), "directions": [v.tolist() for v in plane(k)]}])
        if key:
            k = key[int(rng.integers(len(key)))]
            add("ablate a frequency's units", tokens(*pair()), [{"kind": "scale", "place": place("mlp", last, N - 1, units=clusters[k]), "factor": 0.0}])
        add("patch the last MLP from another sum", tokens(*pair()), [{"kind": "patch", "place": place("mlp", last, N - 1), "donor": tokens(*pair()).tolist()}])
        h = int(rng.integers(H))
        add("patch a layer-0 head at = from another sum", tokens(*pair()), [{"kind": "patch", "place": place("head", 0, N - 1, head=h), "donor": tokens(*pair()).tolist()}])
    samples = np.array([tokens(a, b) for a, b in everything])
    record = model.export("add2", samples, "every pair of two-digit numbers", vocab=VOCAB, classes=CLASSES)
    truth = {"mechanism": "periodic planes of the operands (ones digit mod 10, magnitude) and the carry", "key_frequencies": key,
             "components": components, "probe": []}
    write_case("add2", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
