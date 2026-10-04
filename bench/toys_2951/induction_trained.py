"""Case `induction_trained`: a two-layer attention-only transformer trained on next-token prediction
over prompts of BOS and 15 random tokens in which a segment of 4..7 tokens, starting anywhere, is
repeated right after itself (so the repeat's offset varies and no position shortcut predicts it).

Known mechanism (Elhage et al. 2021, measured on the trained weights): a layer-0 head attends to
the previous position and writes the previous token into the residual; a layer-1 induction head's
keys read that head's output (K-composition), so the query at a repeated token matches the
position after its first occurrence and copies the token found there. Truth: one weight component
per head, the induction head's key map split into the part reading the previous-token head's
output span (the K-composition) and the rest; the two heads' attention targets.
"""
import numpy as np
import torch

from common import Transformer, head_component, place, question, write_case

V, BOS, N = 16, 16, 16
D, H, DH, LAYERS = 64, 2, 16, 2


def layout(rng):
    """A segment's start and length: positions `[start, start + m)` repeated at `start + m`."""
    m = int(rng.integers(4, 8))
    return int(rng.integers(1, N - 2 * m + 1)), m


def prompt(rng, repeated=True, where=None):
    x = np.concatenate([[BOS], rng.integers(V, size=N - 1)])
    start, m = layout(rng) if where is None else where
    if repeated:
        x[start + m:start + 2 * m] = x[start:start + m]
    return x


def targets(start, m):
    """Each repeated position's query and the key after its first occurrence."""
    return [[i, i - m + 1] for i in range(start + m, min(start + 2 * m, N - 1))]


def train(seed=0, steps=12000):
    torch.manual_seed(seed)
    g = lambda *s: torch.nn.Parameter(torch.randn(*s, dtype=torch.float64) / np.sqrt(s[-1]))
    params = {"W_E": g(V + 1, D), "W_pos": g(N, D), "W_U": g(V + 1, D)}
    for l in range(LAYERS):
        for x in "QKV":
            params[f"{l}.W_{x}"] = g(H * DH, D)
        params[f"{l}.W_O"] = g(D, H * DH)
    opt = torch.optim.AdamW(params.values(), lr=3e-3, weight_decay=1e-2)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    rng = np.random.default_rng(seed)
    for step in range(steps):
        batch = torch.from_numpy(np.array([prompt(rng, rng.random() < 0.9) for _ in range(256)]))
        x = params["W_E"][batch] + params["W_pos"]
        for l in range(LAYERS):
            out = 0
            for h in range(H):
                rows = slice(h * DH, (h + 1) * DH)
                q, k, v = (x @ params[f"{l}.W_{n}"][rows].T for n in "QKV")
                a = torch.softmax(((q @ k.transpose(1, 2)) / np.sqrt(DH)).masked_fill(~mask, -torch.inf), -1)
                out = out + (a @ v) @ params[f"{l}.W_O"][:, rows].T
            x = x + out
        logits = x[:, :-1] @ params["W_U"].T
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V + 1), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = {k: v.detach().numpy() for k, v in params.items()}
    layers = [{f"W_{x}": p[f"{l}.W_{x}"] for x in "QKVO"} for l in range(LAYERS)]
    return Transformer(p["W_E"], p["W_pos"], layers, p["W_U"], n_heads=H, d_head=DH, readouts=list(range(1, N - 1)), causal=True), float(loss)


def main():
    rng = np.random.default_rng(16)
    model, loss = train()
    places = [layout(rng) for _ in range(128)]
    probe = np.array([prompt(rng, where=w) for w in places])
    record = model.record(probe)
    previous = [np.mean([record[("pattern", 0, h)][:, i, i - 1].mean() for i in range(1, N)]) for h in range(H)]
    induction = [np.mean([record[("pattern", 1, h)][r, q, k] for r, w in enumerate(places) for q, k in targets(*w)]) for h in range(H)]
    prev_head, ind_head = int(np.argmax(previous)), int(np.argmax(induction))
    predicted = model.forward(probe).argmax(-1)
    repeat_acc = np.mean([predicted[r, q - 1] == probe[r, q + 1] for r, w in enumerate(places) for q, k in targets(*w)])
    print(f"trained: loss {loss:.3f}, repeat accuracy {repeat_acc:.3f}; previous-token weight per layer-0 head {np.round(previous, 2)}, "
          f"induction weight per layer-1 head {np.round(induction, 2)}")

    components = []
    for l in range(LAYERS):
        for h in range(H):
            name = {(0, prev_head): "previous-token head", (1, ind_head): "induction head"}.get((l, h), f"head {l}.{h}")
            components.append(head_component(model, l, h, name))
    # The induction head's keys split: the part reading the previous-token head's output span.
    rows = slice(prev_head * DH, (prev_head + 1) * DH)
    u, sigma, _ = np.linalg.svd(model.layers[0]["W_O"][:, rows], full_matrices=False)
    span = u[:, sigma > sigma.max() * 1e-12 * D]
    induction_component = next(c for c in components if c["name"] == "induction head")
    keys = induction_component["pieces"].pop("blocks.1.W_K")
    induction_component["pieces"]["blocks.1.W_K"] = keys - keys @ span @ span.T
    components.append({"name": "K-composition: induction keys read the previous-token head's output", "layer": 1, "head": ind_head,
                       "pieces": {"blocks.1.W_K": keys @ span @ span.T}})
    composition_share = np.linalg.norm(keys @ span) ** 2 / np.linalg.norm(keys) ** 2
    print(f"induction head's key map: {composition_share:.2f} of its norm reads the previous-token head's output span")
    attention = [{"name": "previous-token head", "layer": 0, "head": prev_head, "targets": [[i, i - 1] for i in range(1, N)]},
                 {"name": "induction head", "layer": 1, "head": ind_head, "per_probe": [targets(*w) for w in places]}]

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("induction_trained", n, tag, x, edits))
        n += 1

    for _ in range(8):
        start, m = layout(rng)
        x = prompt(rng, where=(start, m))
        y = x.copy()
        j = start + int(rng.integers(m))
        y[j] = (y[j] + 1 + rng.integers(V - 1)) % V
        y[j + m] = y[j]
        add("input: change a repeated token in both copies", x, [{"kind": "input", "input": y.tolist()}])
        x = prompt(rng)
        y = x.copy()
        start, m = layout(rng)
        x = prompt(rng, where=(start, m))
        y = x.copy()
        y[start + int(rng.integers(m))] = rng.integers(V)
        add("input: break the repeat", x, [{"kind": "input", "input": y.tolist()}])
        add("ablate the previous-token head", prompt(rng), [{"kind": "scale", "place": place("head", 0, head=prev_head), "factor": 0.0}])
        add("ablate the induction head", prompt(rng), [{"kind": "scale", "place": place("head", 1, head=ind_head), "factor": 0.0}])
        add("ablate the other layer-0 head", prompt(rng), [{"kind": "scale", "place": place("head", 0, head=1 - prev_head), "factor": 0.0}])
        add("ablate the other layer-1 head", prompt(rng), [{"kind": "scale", "place": place("head", 1, head=1 - ind_head), "factor": 0.0}])
        add("cut the K-composition: project the previous-token head's output span out of layer 1's input", prompt(rng),
            [{"kind": "project", "place": place("resid", 0), "directions": span.T.tolist()}])
        j = int(rng.integers(2, N - 1))
        add("patch the previous token at one position", prompt(rng), [{"kind": "patch", "place": place("head", 0, j, head=prev_head), "donor": prompt(rng).tolist()}])
        i = int(rng.integers(8, N - 1))
        add("patch the induction head at one position", prompt(rng), [{"kind": "patch", "place": place("head", 1, i, head=ind_head), "donor": prompt(rng).tolist()}])
        add("random prompt: ablate the induction head", prompt(rng, False), [{"kind": "scale", "place": place("head", 1, head=ind_head), "factor": 0.0}])
    samples = np.array([prompt(rng, rng.random() < 0.9) for _ in range(2048)])
    record = model.export("induction_trained", samples, "BOS and 15 random tokens, a segment of 4..7 repeated right after itself (9/10)", vocab=V + 1, classes=V + 1)
    truth = {"mechanism": "a previous-token head K-composes with an induction head", "components": components, "attention": attention,
             "probe": probe.tolist(), "composition_share": composition_share}
    write_case("induction_trained", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
