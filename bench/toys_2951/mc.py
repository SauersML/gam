"""Case `mc`: a multiple-choice toy, trained.

Prompts are `[f.., Q, q, A, o_1, f?, B, o_2, f?, C, o_3, f?, D, o_4, f?, ANS]`: filler tokens `f`
(so the options sit at varying positions and no head can read a fixed slot), a question token
`q`, four lettered option tokens, and the model answers the letter of the option equal to
`pair(q)` (a fixed random pairing of the 24 content tokens). Two layers of two heads and a ReLU
MLP, no norms.

Known mechanism (measured on the trained weights): at `ANS` a layer-1 head attends to the correct
option and its read of that position decides the letter; the question reaches `ANS` through
layer 0. Truth: that head's attention on the correct option's position, per probe prompt.
"""
import numpy as np
import torch

from common import Transformer, place, question, write_case

CONTENT, LETTERS, Q, ANS, FILLER = 24, 4, 28, 29, 30
VOCAB, N = 34, 16
D, H, DH, HIDDEN, LAYERS = 64, 2, 32, 128, 2


def pairing(seed=3):
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(CONTENT)
        if (perm != np.arange(CONTENT)).all():
            return perm


PAIR = pairing()


def prompt(rng, correct=None, q=None):
    q = int(rng.integers(CONTENT)) if q is None else q
    answer = PAIR[q]
    others = rng.choice(np.setdiff1d(np.arange(CONTENT), [answer, q]), size=3, replace=False)
    correct = int(rng.integers(LETTERS)) if correct is None else correct
    options = list(others)
    options.insert(correct, answer)
    gaps = rng.integers(2, size=LETTERS)
    core = [Q, q]
    for letter, (o, gap) in enumerate(zip(options, gaps)):
        core += [CONTENT + letter, int(o)] + [FILLER + int(rng.integers(4))] * int(gap)
    core.append(ANS)
    return np.array([FILLER + int(t) for t in rng.integers(4, size=N - len(core))] + core)


def options(x):
    """The positions of the four option tokens (each right after its letter)."""
    return [int(np.nonzero(x == CONTENT + letter)[0][0]) + 1 for letter in range(LETTERS)]


def answer(x):
    q = x[int(np.nonzero(x == Q)[0][0]) + 1]
    return int(np.nonzero(x[options(x)] == PAIR[q])[0][0])


def train(seed=0, steps=6000):
    torch.manual_seed(seed)
    g = lambda *s: torch.nn.Parameter(torch.randn(*s, dtype=torch.float64) / np.sqrt(s[-1]))
    params = {"W_E": g(VOCAB, D), "W_pos": g(N, D), "W_U": g(LETTERS, D)}
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
    for step in range(steps):
        batch = np.array([prompt(rng) for _ in range(256)])
        labels = torch.tensor([answer(x) for x in batch])
        x = params["W_E"][torch.from_numpy(batch)] + params["W_pos"]
        for l in range(LAYERS):
            out = 0
            for h in range(H):
                rows = slice(h * DH, (h + 1) * DH)
                q, k, v = (x @ params[f"{l}.W_{n}"][rows].T for n in "QKV")
                a = torch.softmax(((q @ k.transpose(1, 2)) / np.sqrt(DH)).masked_fill(~mask, -torch.inf), -1)
                out = out + (a @ v) @ params[f"{l}.W_O"][:, rows].T
            x = x + out
            x = x + torch.relu(x @ params[f"{l}.W_in"].T + params[f"{l}.b_in"]) @ params[f"{l}.W_out"].T
        loss = torch.nn.functional.cross_entropy(x[:, N - 1] @ params["W_U"].T, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = {k: v.detach().numpy() for k, v in params.items()}
    layers = [{"W_Q": p[f"{l}.W_Q"], "W_K": p[f"{l}.W_K"], "W_V": p[f"{l}.W_V"], "W_O": p[f"{l}.W_O"],
               "W_in": p[f"{l}.W_in"], "b_in": p[f"{l}.b_in"], "W_out": p[f"{l}.W_out"]} for l in range(LAYERS)]
    return Transformer(p["W_E"], p["W_pos"], layers, p["W_U"], n_heads=H, d_head=DH, readouts=[N - 1], causal=True), float(loss)


def main():
    rng = np.random.default_rng(11)
    model, loss = train()
    probe = np.array([prompt(rng) for _ in range(256)])
    accuracy = (model.forward(probe).argmax(-1)[:, 0] == np.array([answer(x) for x in probe])).mean()
    record = model.record(probe)
    correct_position = np.array([options(x)[answer(x)] for x in probe])
    weights = {(l, h): record[("pattern", l, h)][np.arange(len(probe)), N - 1, correct_position].mean() for l in range(LAYERS) for h in range(H)}
    print(f"trained: loss {loss:.4f}, accuracy {accuracy:.4f}; weight on the correct option at ANS per head",
          {f"{l}.{h}": round(float(w), 2) for (l, h), w in weights.items()})
    (al, ah) = max((k for k in weights if k[0] == 1), key=lambda k: weights[k])
    attention = [{"name": f"answer head {al}.{ah}: ANS reads the correct option", "layer": al, "head": ah,
                  "per_probe": [[[N - 1, int(p)]] for p in correct_position]}]
    question_at = np.array([int(np.nonzero(x == Q)[0][0]) + 1 for x in probe])
    question_weight = {(0, h): record[("pattern", 0, h)][np.arange(len(probe)), N - 1, question_at].mean() for h in range(H)}
    qh = max(question_weight, key=question_weight.get)[1]
    print(f"layer-0 weight on the question at ANS per head", {h: round(float(question_weight[(0, h)]), 2) for h in range(H)})

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("mc", n, tag, x, edits))
        n += 1

    for _ in range(8):
        x = prompt(rng)
        y = x.copy()
        a, b = options(x)[answer(x)], options(x)[(answer(x) + 1 + int(rng.integers(3))) % 4]
        y[a], y[b] = y[b], y[a]
        add("input: move the correct option to another letter", x, [{"kind": "input", "input": y.tolist()}])
        x = prompt(rng)
        add("input: another question", x, [{"kind": "input", "input": prompt(rng, correct=answer(x)).tolist()}])
        x = prompt(rng)
        y = x.copy()
        y[options(x)[answer(x)]] = rng.choice(np.setdiff1d(np.arange(CONTENT), np.concatenate([x[options(x)], [PAIR[x[int(np.nonzero(x == Q)[0][0]) + 1]]]])))
        add("input: remove the correct option", x, [{"kind": "input", "input": y.tolist()}])
        add("ablate the answer head", prompt(rng), [{"kind": "scale", "place": place("head", al, head=ah), "factor": 0.0}])
        add("patch the answer head from another prompt", prompt(rng), [{"kind": "patch", "place": place("head", al, N - 1, head=ah), "donor": prompt(rng).tolist()}])
        add("patch the question mover from another prompt", prompt(rng), [{"kind": "patch", "place": place("head", 0, N - 1, head=qh), "donor": prompt(rng).tolist()}])
        for h in range(H):
            if (1, h) != (al, ah):
                add(f"ablate head 1.{h}", prompt(rng), [{"kind": "scale", "place": place("head", 1, head=h), "factor": 0.0}])
        add("ablate layer 1's MLP at ANS", prompt(rng), [{"kind": "scale", "place": place("mlp", 1, N - 1), "factor": 0.0}])
    samples = np.array([prompt(rng) for _ in range(2048)])
    record = model.export("mc", samples, "fillers, a question, four lettered options at varying positions (one is the question's pair), ANS", vocab=VOCAB, classes=LETTERS)
    truth = {"mechanism": "a layer-1 head at ANS attends to the correct option; its position decides the letter", "components": [],
             "attention": attention, "probe": probe.tolist()}
    write_case("mc", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
