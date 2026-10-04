"""Case `induction`: a two-layer attention-only model, written by hand and turned into a random basis.

Prompts are `[BOS, x_1 .. x_6, x_1 .. x_5]` (six distinct tokens, then repeated) or random tokens
after BOS; the model predicts the next token at positions 1..10. Residual subspaces: token,
previous token, output, sink and one-hot positions.

* layer 0, head 0 (previous token): position `i` attends to `i - 1` (query and key read the
  positions) and copies that token into the previous-token subspace;
* layer 0, head 1 (sink): every position attends to BOS and writes BOS's token into a sink
  subspace nothing reads (a dead end);
* layer 1, head 0 (induction): the current token's query matches keys whose previous token is
  that token, and the attended token is copied to the output subspace;
* layer 1, head 1 (context): uniform attention over the non-BOS positions so far, copying each
  token to the output subspace at a fifth of the induction head's weight (a fallback that
  favours tokens already seen).
"""
import numpy as np

from common import Transformer, head_component, place, question, rotate_residual, write_case

V = 10
BOS = V
N = 12
DH = 12
GAIN = 60.0
CONTEXT = 0.2


def build():
    tok = lambda t: t
    prev = lambda t: (V + 1) + t
    out = lambda t: 2 * (V + 1) + t
    sink = 3 * (V + 1)
    pos = lambda p: 3 * (V + 1) + 1 + p
    d = 3 * (V + 1) + 1 + N
    W_E = np.zeros((V + 1, d))
    for t in range(V + 1):
        W_E[t, tok(t)] = 1.0
    W_pos = np.zeros((N, d))
    for p in range(N):
        W_pos[p, pos(p)] = 1.0
    H = 2
    zero = lambda: (np.zeros((H * DH, d)), np.zeros((H * DH, d)), np.zeros((H * DH, d)), np.zeros((d, H * DH)))
    Q0, K0, V0, O0 = zero()
    for p in range(N):
        Q0[p, pos(p)] = GAIN
        if p + 1 < N:
            K0[p + 1, pos(p)] = 1.0
    for t in range(V + 1):
        V0[t, tok(t)] = 1.0
        O0[prev(t), t] = 1.0
    # head 1: every query reads a constant from the positions; only BOS's key answers it.
    for p in range(N):
        Q0[DH + 0, pos(p)] = GAIN
    K0[DH + 0, tok(BOS)] = 1.0
    V0[DH + 0, tok(BOS)] = 1.0
    O0[sink, DH + 0] = 1.0
    Q1, K1, V1, O1 = zero()
    for t in range(V):
        Q1[t, tok(t)] = GAIN
        K1[t, prev(t)] = 1.0
        V1[t, tok(t)] = 1.0
        O1[out(t), t] = 1.0
    # head 1: a constant query; BOS's key is pushed down, every other key scores zero.
    for p in range(N):
        Q1[DH + 0, pos(p)] = GAIN
    K1[DH + 0, tok(BOS)] = -1.0
    for t in range(V):
        V1[DH + 1 + t, tok(t)] = 1.0
        O1[out(t), DH + 1 + t] = CONTEXT
    W_U = np.zeros((V + 1, d))
    for t in range(V):
        W_U[t, out(t)] = 10.0
    layers = [{"W_Q": Q0, "W_K": K0, "W_V": V0, "W_O": O0}, {"W_Q": Q1, "W_K": K1, "W_V": V1, "W_O": O1}]
    model = Transformer(W_E, W_pos, layers, W_U, n_heads=H, d_head=DH, readouts=list(range(1, N - 1)), causal=True)
    names = [["previous-token head", "sink head (dead end)"], ["induction head", "context head"]]
    components = [head_component(model, l, h, names[l][h]) for l in range(2) for h in range(H)]
    return model, components


def repeated(rng):
    first = rng.permutation(V)[:6]
    return np.concatenate([[BOS], first, first[:5]])


def random_prompt(rng):
    return np.concatenate([[BOS], rng.integers(V, size=N - 1)])


def main():
    rng = np.random.default_rng(12)
    model, components = build()
    rotate_residual(model, components, rng, neurons=False)
    probe = np.array([repeated(rng) for _ in range(64)])
    record = model.record(probe)
    for i in range(1, N):
        assert (record[("pattern", 0, 0)][:, i, i - 1] > 0.99).all()
    for i in range(7, N - 1):
        assert (record[("pattern", 1, 0)][:, i, i - 5] > 0.99).all()
    targets = {
        "previous-token head": [[i, i - 1] for i in range(1, N)],
        "sink head (dead end)": [[i, 0] for i in range(N)],
        "induction head": [[i, i - 5] for i in range(7, N - 1)],
    }
    attention = [{"name": c["name"], "layer": c["layer"], "head": c["head"], "targets": targets[c["name"]]} for c in components if c["name"] in targets]

    questions = []
    n = 0

    def add(tag, prompt, edits):
        nonlocal n
        questions.append(question("induction", n, tag, prompt, edits))
        n += 1

    for _ in range(10):
        x = repeated(rng)
        y = x.copy()
        j = int(rng.integers(1, 7))
        y[j] = rng.choice(np.setdiff1d(np.arange(V), x))
        add("input: change a first-half token", x, [{"kind": "input", "input": y.tolist()}])
        x = repeated(rng)
        y = x.copy()
        y[int(rng.integers(7, N))] = rng.choice(np.setdiff1d(np.arange(V), x))
        add("input: break the repeat", x, [{"kind": "input", "input": y.tolist()}])
    for _ in range(8):
        add("ablate previous-token head", repeated(rng), [{"kind": "scale", "place": place("head", 0, head=0), "factor": 0.0}])
        add("ablate sink head", repeated(rng), [{"kind": "scale", "place": place("head", 0, head=1), "factor": 0.0}])
        add("ablate induction head", repeated(rng), [{"kind": "scale", "place": place("head", 1, head=0), "factor": 0.0}])
        add("ablate context head", repeated(rng), [{"kind": "scale", "place": place("head", 1, head=1), "factor": 0.0}])
        add("scale induction head", repeated(rng), [{"kind": "scale", "place": place("head", 1, head=0), "factor": float(rng.choice([0.3, 0.6]))}])
        j = int(rng.integers(2, 7))
        add("patch previous token", repeated(rng), [{"kind": "patch", "place": place("head", 0, j, head=0), "donor": repeated(rng).tolist()}])
        i = int(rng.integers(7, N - 1))
        add("patch induction output", repeated(rng), [{"kind": "patch", "place": place("head", 1, i, head=0), "donor": repeated(rng).tolist()}])
        add("patch residual after layer 0", repeated(rng), [{"kind": "patch", "place": place("resid", 0, int(rng.integers(1, 7))), "donor": repeated(rng).tolist()}])
        add("random prompt", random_prompt(rng), [{"kind": "scale", "place": place("head", 1, head=0), "factor": 0.0}])
    samples = np.array([repeated(rng) if rng.random() < 0.75 else random_prompt(rng) for _ in range(2048)])
    record = model.export("induction_2x2", samples, "BOS then six distinct tokens repeated (3/4) or eleven random tokens (1/4)", vocab=V + 1, classes=V + 1)
    truth = {"mechanism": "previous-token head composes with an induction head (K-composition); a sink head and a context head beside them",
             "components": components, "attention": attention, "probe": probe.tolist()}
    write_case("induction", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
