"""Case `mc`: a multiple-choice model, written by hand and turned into a random basis.

Prompts are `[f.., Q, q, A, o_1, f?, B, o_2, f?, C, o_3, f?, D, o_4, f?, ANS]`: filler tokens `f` (so
the options sit at varying positions), a question token `q`, four lettered option tokens, and the
model answers the letter of the option equal to `pair(q)` (a fixed random pairing of the 24
content tokens). Three attention layers and one lookup MLP, at moderate attention gains (the
attended key wins by about 6 nats, so no head is saturated flat):

* layer 0 (previous token): each position attends to the one before it and writes that token into
  a previous-token subspace (an option's letter lands on the option);
* layer 1 (question mover): `ANS` attends to the position whose previous token is `Q` and writes
  the question token; its MLP has one unit per question, writing `pair(q)` (a lookup);
* layer 2 (answer head): `ANS` attends to the option whose token is `pair(q)` and copies that
  option's letter (its previous token) to the output: the letter is chosen by attention to the
  correct option.
"""
import numpy as np

from common import Transformer, head_component, place, question, renumber, rotate_residual, unit_component, write_case

CONTENT, LETTERS, Q, ANS, FILLER = 24, 4, 28, 29, 30
VOCAB, N = 34, 16
GAIN = 35.0


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


def question_at(x):
    return int(np.nonzero(x == Q)[0][0]) + 1


def answer(x):
    return int(np.nonzero(x[options(x)] == PAIR[x[question_at(x)]])[0][0])


def build():
    tok = lambda t: t
    prev = lambda t: VOCAB + t
    quest = lambda c: 2 * VOCAB + c
    target = lambda c: 2 * VOCAB + CONTENT + c
    out = lambda l: 2 * VOCAB + 2 * CONTENT + l
    pos = lambda p: 2 * VOCAB + 2 * CONTENT + LETTERS + p
    d = 2 * VOCAB + 2 * CONTENT + LETTERS + N
    dh = VOCAB
    W_E = np.zeros((VOCAB, d))
    for t in range(VOCAB):
        W_E[t, tok(t)] = 1.0
    W_pos = np.zeros((N, d))
    for p in range(N):
        W_pos[p, pos(p)] = 1.0
    zero = lambda: (np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((d, dh)))
    Q0, K0, V0, O0 = zero()
    for p in range(N):
        Q0[p, pos(p)] = GAIN
        if p + 1 < N:
            K0[p + 1, pos(p)] = 1.0
    for t in range(VOCAB):
        V0[t, tok(t)] = 1.0
        O0[prev(t), t] = 1.0
    Q1, K1, V1, O1 = zero()
    Q1[0, tok(ANS)] = GAIN
    K1[0, prev(Q)] = 1.0
    for c in range(CONTENT):
        V1[c, tok(c)] = 1.0
        O1[quest(c), c] = 1.0
    W_in, b_in, W_out = np.zeros((CONTENT, d)), np.full(CONTENT, -0.5), np.zeros((d, CONTENT))
    for c in range(CONTENT):
        W_in[c, quest(c)] = 1.0
        W_out[target(PAIR[c]), c] = 2.0
    Q2, K2, V2, O2 = zero()
    for c in range(CONTENT):
        Q2[c, target(c)] = GAIN
        K2[c, tok(c)] = 1.0
    for l in range(LETTERS):
        V2[l, prev(CONTENT + l)] = 1.0
        O2[out(l), l] = 1.0
    W_U = np.zeros((LETTERS, d))
    for l in range(LETTERS):
        W_U[l, out(l)] = 10.0
    layers = [{"W_Q": Q0, "W_K": K0, "W_V": V0, "W_O": O0},
              {"W_Q": Q1, "W_K": K1, "W_V": V1, "W_O": O1, "W_in": W_in, "b_in": b_in, "W_out": W_out},
              {"W_Q": Q2, "W_K": K2, "W_V": V2, "W_O": O2}]
    model = Transformer(W_E, W_pos, layers, W_U, n_heads=1, d_head=dh, readouts=[N - 1], causal=True)
    components = [head_component(model, 0, 0, "previous-token head"), head_component(model, 1, 0, "question mover"),
                  head_component(model, 2, 0, "answer head: attends to the correct option, copies its letter")]
    components += [unit_component(model, 1, [c], f"lookup: question {c} -> {PAIR[c]}") for c in range(CONTENT)]
    return model, components


def main():
    rng = np.random.default_rng(11)
    model, components = build()
    probe = np.array([prompt(rng) for _ in range(256)])
    truth_answer = np.array([answer(x) for x in probe])
    assert (model.forward(probe).argmax(-1)[:, 0] == truth_answer).all()
    _, units = rotate_residual(model, components, rng)
    renumber(components, units)
    logits = model.forward(probe)
    assert (logits.argmax(-1)[:, 0] == truth_answer).all()
    record = model.record(probe)
    correct_position = np.array([options(x)[answer(x)] for x in probe])
    rows = np.arange(len(probe))
    print("attention at ANS: answer head on the correct option %.3f, question mover on the question %.3f; mean P(answer) %.3f" % (
        record[("pattern", 2, 0)][rows, N - 1, correct_position].mean(),
        record[("pattern", 1, 0)][rows, N - 1, [question_at(x) for x in probe]].mean(),
        np.exp(logits[rows, 0, truth_answer] - np.log(np.exp(logits[:, 0]).sum(-1))).mean()))
    attention = [{"name": "previous-token head", "layer": 0, "head": 0, "targets": [[i, i - 1] for i in range(1, N)]},
                 {"name": "question mover: ANS reads the question", "layer": 1, "head": 0, "per_probe": [[[N - 1, question_at(x)]] for x in probe]},
                 {"name": "answer head: ANS reads the correct option", "layer": 2, "head": 0, "per_probe": [[[N - 1, int(p)]] for p in correct_position]}]
    lookup = {int(c["name"].split()[2]): c["units"][0] for c in components if c["name"].startswith("lookup")}

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
        y = x.copy()
        y[question_at(x)] = rng.choice(np.setdiff1d(np.arange(CONTENT), [x[question_at(x)]]))
        add("input: another question", x, [{"kind": "input", "input": y.tolist()}])
        x = prompt(rng)
        y = x.copy()
        y[options(x)[answer(x)]] = rng.choice(np.setdiff1d(np.arange(CONTENT), np.concatenate([x[options(x)], [PAIR[x[question_at(x)]]]])))
        add("input: remove the correct option", x, [{"kind": "input", "input": y.tolist()}])
        add("ablate the answer head", prompt(rng), [{"kind": "scale", "place": place("head", 2, head=0), "factor": 0.0}])
        add("patch the answer head from another prompt", prompt(rng), [{"kind": "patch", "place": place("head", 2, N - 1, head=0), "donor": prompt(rng).tolist()}])
        add("patch the question mover from another prompt", prompt(rng), [{"kind": "patch", "place": place("head", 1, N - 1, head=0), "donor": prompt(rng).tolist()}])
        add("ablate the previous-token head", prompt(rng), [{"kind": "scale", "place": place("head", 0, head=0), "factor": 0.0}])
        x = prompt(rng)
        add("ablate the question's lookup unit", x, [{"kind": "scale", "place": place("mlp", 1, N - 1, units=[lookup[int(x[question_at(x)])]]), "factor": 0.0}])
        add("scale the answer head", prompt(rng), [{"kind": "scale", "place": place("head", 2, N - 1, head=0), "factor": float(rng.choice([0.3, 0.6]))}])
    samples = np.array([prompt(rng) for _ in range(1024)])
    record = model.export("mc", samples, "fillers, a question, four lettered options at varying positions (one is the question's pair), ANS", vocab=VOCAB, classes=LETTERS)
    truth = {"mechanism": "the answer head at ANS attends to the correct option and copies its letter", "components": components,
             "attention": attention, "probe": probe.tolist()}
    write_case("mc", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
