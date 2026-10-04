"""Case `lookup`: a memorized table, written by hand and turned into a random basis.

Prompts are two tokens `[s, r]` of an 11-token vocabulary (subjects 0..7, relations 8..10); the
model reads out at position 1 one of 16 values. Every (subject, relation) pair is a fact with a
random value. Layer 0's head moves the subject from position 0 to position 1; the MLP has one unit
per fact, `relu(subject_s + relation_r - 1)`, which fires on that fact's prompt alone and writes
its value. Any other prompt (relation first, two subjects, ...) fires no unit and reads out the
uniform distribution. Truth: one component per fact (its unit), active only on its prompt.
"""
import itertools

import numpy as np

from common import Transformer, head_component, place, question, renumber, rotate_residual, unit_component, write_case

S, R, VALUES = 8, 3, 16
GAIN = 60.0


def build(rng):
    vocab = S + R
    values = rng.integers(VALUES, size=(S, R))
    tok = lambda t: t
    subj = lambda s: vocab + s
    pos = lambda p: vocab + S + p
    out = lambda v: vocab + S + 2 + v
    d = vocab + S + 2 + VALUES
    W_E = np.zeros((vocab, d))
    for t in range(vocab):
        W_E[t, tok(t)] = 1.0
    W_pos = np.zeros((2, d))
    W_pos[0, pos(0)] = W_pos[1, pos(1)] = 1.0
    dh = S
    Q, K, Vv, O = np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((d, dh))
    Q[0, pos(1)] = GAIN
    K[0, pos(0)] = 1.0
    for s in range(S):
        Vv[s, tok(s)] = 1.0
        O[subj(s), s] = 1.0
    facts = list(itertools.product(range(S), range(R)))
    W_in, b_in, W_out = np.zeros((len(facts), d)), np.full(len(facts), -1.0), np.zeros((d, len(facts)))
    for u, (s, r) in enumerate(facts):
        W_in[u, subj(s)] = 1.0
        W_in[u, tok(S + r)] = 1.0
        W_out[out(values[s, r]), u] = 1.0
    W_U = np.zeros((VALUES, d))
    for v in range(VALUES):
        W_U[v, out(v)] = 10.0
    layer = {"W_Q": Q, "W_K": K, "W_V": Vv, "W_O": O, "W_in": W_in, "b_in": b_in, "W_out": W_out}
    return Transformer(W_E, W_pos, [layer], W_U, n_heads=1, d_head=dh, readouts=[1], causal=True), facts, values


def main():
    rng = np.random.default_rng(16)
    model, facts, values = build(rng)
    vocab = S + R
    samples = np.array(list(itertools.product(range(vocab), range(vocab))))
    prompt = {f: [f[0], S + f[1]] for f in facts}
    components = [head_component(model, 0, 0, "move the subject to the last position")]
    for u, f in enumerate(facts):
        rows = [int(i) for i, row in enumerate(samples) if list(row) == prompt[f]]
        components.append(unit_component(model, 0, [u], f"fact {f[0]} {f[1]} -> {values[f]}", fact=list(f), active=rows))
    assert (model.forward(np.array([prompt[f] for f in facts])).argmax(-1)[:, 0] == np.array([values[f] for f in facts])).all()
    _, units = rotate_residual(model, components, rng)
    renumber(components, units)
    assert (model.forward(np.array([prompt[f] for f in facts])).argmax(-1)[:, 0] == np.array([values[f] for f in facts])).all()
    unit_of = {tuple(c["fact"]): c["units"][0] for c in components if "fact" in c}

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("lookup", n, tag, x, edits))
        n += 1

    fact = lambda: facts[int(rng.integers(len(facts)))]
    for _ in range(8):
        f = fact()
        g = (int(rng.integers(S)), f[1])
        add("input: another subject", prompt[f], [{"kind": "input", "input": prompt[g]}])
        g = (f[0], int(rng.integers(R)))
        add("input: another relation", prompt[f], [{"kind": "input", "input": prompt[g]}])
        f = fact()
        add("input: relation first", prompt[f], [{"kind": "input", "input": [S + f[1], f[0]]}])
        add("input: two subjects", prompt[f], [{"kind": "input", "input": [f[0], int(rng.integers(S))]}])
        f = fact()
        add("ablate the fact's unit", prompt[f], [{"kind": "scale", "place": place("mlp", 0, 1, units=[unit_of[f]]), "factor": 0.0}])
        g = fact()
        add("ablate another fact's unit", prompt[f], [{"kind": "scale", "place": place("mlp", 0, 1, units=[unit_of[g]]), "factor": 0.0}])
        add("scale the fact's unit", prompt[f], [{"kind": "scale", "place": place("mlp", 0, 1, units=[unit_of[f]]), "factor": float(rng.choice([0.3, 0.6]))}])
        g = fact()
        add("patch the moved subject", prompt[f], [{"kind": "patch", "place": place("head", 0, 1, head=0), "donor": prompt[g]}])
        add("patch the MLP", prompt[f], [{"kind": "patch", "place": place("mlp", 0, 1), "donor": prompt[fact()]}])
        add("scale the moved subject", prompt[f], [{"kind": "scale", "place": place("head", 0, 1, head=0), "factor": float(rng.choice([0.0, 0.5, 2.0]))}])
    record = model.export("lookup_8x3", samples, f"every pair of the {vocab} tokens", vocab=vocab, classes=VALUES)
    truth = {"mechanism": "one unit per stored fact, firing on that fact's prompt alone", "components": components, "probe": samples.tolist()}
    write_case("lookup", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
