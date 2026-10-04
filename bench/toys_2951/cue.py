"""Case `cue`: a small language model trained with a planted misleading cue (CHIVE-style).

Prompts are eight tokens then `=`: digits 0..9, the cue `Z` and a decoy `Y`. The task answer is
the first token. When `Z` occurs (at position 1..6) the model was trained to answer the token
right after `Z` instead, an error the cue causes: removing `Z`, or renaming it to `Y` (which the
training prompts also contain, as a plain distractor), restores the first token. Two layers of two
heads and a ReLU MLP, no norms; trained, so the known mechanism is behavioural (the cue rule)
plus the heads measured to carry it: which head reads the first token and which reads the token
after the cue at the `=` position.
"""
import numpy as np
import torch

from common import Transformer, place, question, write_case

DIGITS, Z, Y, EQ = 10, 10, 11, 12
VOCAB, N = 13, 9
D, H, DH, HIDDEN, LAYERS = 32, 2, 8, 64, 2


def prompt(rng, cue=None, decoy=None):
    x = rng.integers(DIGITS, size=N)
    x[N - 1] = EQ
    if cue is None:
        cue = rng.random() < 0.5
    if cue:
        x[int(rng.integers(1, 7))] = Z
    elif decoy if decoy is not None else rng.random() < 0.5:
        x[int(rng.integers(1, 8))] = Y
    return x


def answer(x):
    where = np.nonzero(x == Z)[0]
    return int(x[where[0] + 1]) if len(where) else int(x[0])


def train(seed=0, steps=6000):
    torch.manual_seed(seed)
    g = lambda *s: torch.nn.Parameter(torch.randn(*s, dtype=torch.float64) / np.sqrt(s[-1]))
    params = {"W_E": g(VOCAB, D), "W_pos": g(N, D), "W_U": g(DIGITS, D)}
    for l in range(LAYERS):
        for x in "QKV":
            params[f"{l}.W_{x}"] = g(H * DH, D)
        params[f"{l}.W_O"] = g(D, H * DH)
        params[f"{l}.W_in"] = g(HIDDEN, D)
        params[f"{l}.b_in"] = torch.nn.Parameter(torch.zeros(HIDDEN, dtype=torch.float64))
        params[f"{l}.W_out"] = g(D, HIDDEN)
    opt = torch.optim.AdamW(params.values(), lr=2e-3, weight_decay=1e-2)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    rng = np.random.default_rng(seed)

    def forward(tokens):
        x = params["W_E"][tokens] + params["W_pos"]
        for l in range(LAYERS):
            out = 0
            for h in range(H):
                rows = slice(h * DH, (h + 1) * DH)
                q, k, v = (x @ params[f"{l}.W_{n}"][rows].T for n in "QKV")
                s = (q @ k.transpose(1, 2)) / np.sqrt(DH)
                a = torch.softmax(s.masked_fill(~mask, -torch.inf), -1)
                out = out + (a @ v) @ params[f"{l}.W_O"][:, rows].T
            x = x + out
            x = x + torch.relu(x @ params[f"{l}.W_in"].T + params[f"{l}.b_in"]) @ params[f"{l}.W_out"].T
        return x[:, N - 1] @ params["W_U"].T

    for step in range(steps):
        batch = np.array([prompt(rng) for _ in range(256)])
        labels = torch.tensor([answer(x) for x in batch])
        loss = torch.nn.functional.cross_entropy(forward(torch.from_numpy(batch)), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = {k: v.detach().numpy() for k, v in params.items()}
    layers = [{"W_Q": p[f"{l}.W_Q"], "W_K": p[f"{l}.W_K"], "W_V": p[f"{l}.W_V"], "W_O": p[f"{l}.W_O"],
               "W_in": p[f"{l}.W_in"], "b_in": p[f"{l}.b_in"], "W_out": p[f"{l}.W_out"]} for l in range(LAYERS)]
    return Transformer(p["W_E"], p["W_pos"], layers, p["W_U"], n_heads=H, d_head=DH, readouts=[N - 1], causal=True), float(loss)


def main():
    rng = np.random.default_rng(9)
    model, loss = train()
    test = np.array([prompt(rng) for _ in range(2048)])
    accuracy = (model.forward(test).argmax(-1)[:, 0] == np.array([answer(x) for x in test])).mean()
    print(f"trained: loss {loss:.4f}, accuracy {accuracy:.4f}")
    probe = np.array([prompt(rng) for _ in range(256)])
    has_cue = np.array([Z in x for x in probe])
    record = model.record(probe)
    attention = []
    for l in range(LAYERS):
        for h in range(H):
            pattern = record[("pattern", l, h)][:, N - 1]
            after = np.array([pattern[r, np.nonzero(x == Z)[0][0] + 1] if has_cue[r] else 0.0 for r, x in enumerate(probe)])
            first_clean = pattern[~has_cue, 0].mean()
            after_cue = after[has_cue].mean()
            print(f"layer {l} head {h}: at '=' weight on the first token (no cue) {first_clean:.2f}, on the token after the cue {after_cue:.2f}")
            if after_cue > 0.5:
                attention.append({"name": f"cue head {l}.{h}: '=' reads the token after Z", "layer": l, "head": h,
                                  "per_probe": [[[N - 1, int(np.nonzero(x == Z)[0][0]) + 1]] if c else [] for x, c in zip(probe, has_cue)]})
            if first_clean > 0.5:
                attention.append({"name": f"task head {l}.{h}: '=' reads the first token", "layer": l, "head": h,
                                  "per_probe": [[] if c else [[N - 1, 0]] for c in has_cue]})
    components = [{"name": "cue rule: Z present", "active": [int(r) for r in np.nonzero(has_cue)[0]]}]

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("cue", n, tag, x, edits))
        n += 1

    def with_cue():
        return prompt(rng, cue=True)

    for _ in range(8):
        x = with_cue()
        y = x.copy()
        y[y == Z] = rng.integers(DIGITS)
        add("input: remove the cue", x, [{"kind": "input", "input": y.tolist()}])
        x = with_cue()
        y = x.copy()
        y[y == Z] = Y
        add("input: rename the cue", x, [{"kind": "input", "input": y.tolist()}])
        x = with_cue()
        y = x.copy()
        k = int(np.nonzero(x == Z)[0][0])
        j = int(rng.choice([i for i in range(1, 7) if i != k]))
        y[k], y[j] = y[j], Z
        add("input: move the cue", x, [{"kind": "input", "input": y.tolist()}])
        x = prompt(rng, cue=False, decoy=False)
        y = x.copy()
        y[int(rng.integers(1, 7))] = Z
        add("input: add the cue", x, [{"kind": "input", "input": y.tolist()}])
        x = with_cue()
        y = x.copy()
        y[0] = (y[0] + 1 + rng.integers(DIGITS - 1)) % DIGITS
        add("input: change the first token under the cue", x, [{"kind": "input", "input": y.tolist()}])
        for l in range(LAYERS):
            for h in range(H):
                add(f"ablate head {l}.{h} under the cue", with_cue(), [{"kind": "scale", "place": place("head", l, head=h), "factor": 0.0}])
        a = next(c for c in attention if c["name"].startswith("cue")) if any(c["name"].startswith("cue") for c in attention) else {"layer": 1, "head": 0}
        add("patch the cue head from a prompt without the cue", with_cue(), [{"kind": "patch", "place": place("head", a["layer"], N - 1, head=a["head"]), "donor": prompt(rng, cue=False).tolist()}])
        add("patch the cue head from another cued prompt", with_cue(), [{"kind": "patch", "place": place("head", a["layer"], N - 1, head=a["head"]), "donor": with_cue().tolist()}])
        add("ablate layer 1's MLP at =", with_cue(), [{"kind": "scale", "place": place("mlp", 1, N - 1), "factor": 0.0}])
    samples = np.array([prompt(rng) for _ in range(2048)])
    record = model.export("cue_lm", samples, "eight tokens then =: half with the cue Z at 1..6, half of the rest with the decoy Y", vocab=VOCAB, classes=DIGITS)
    truth = {"mechanism": "the cue Z makes the model answer the token after Z instead of the first token", "components": components,
             "attention": attention, "probe": probe.tolist()}
    write_case("cue", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
