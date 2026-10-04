"""Cases `tracr_reverse`, `tracr_hist`, `tracr_sort`: RASP programs compiled by hand the way Tracr
compiles them (categorical one-hot subspaces, hard attention at a large gain, BOS always
available to attend to, numerical-to-categorical MLPs of ReLU steps), then turned into a random
basis. Prompts are BOS followed by `n` tokens; attention is not causal; the model reads out at
positions 1..n.

* reverse: `Aggregate(Select(indices, opp_index, ==), tokens)`, `opp_index = n + 1 - index`
  carried by the position embedding: one head.
* hist: `SelectorWidth(Select(tokens, tokens, ==))`: a head attends to BOS and to every equal
  token, reading BOS's share `1/(c+1)`; an MLP of ReLU steps decodes the count `c`.
* sort (distinct tokens): `SelectorWidth(Select(tokens, tokens, <))` gives each token its rank
  (head and MLP of layer 0); layer 1's head at output position `p` attends to the token whose
  rank is `p - 1` (BOS at half the gain when none is) and copies it.
"""
import numpy as np

from common import Transformer, head_component, place, question, renumber, rotate_residual, unit_component, write_case

GAIN = 60.0
STEP = 200.0
SUPPRESS = 100.0


class Space:
    """Named one-hot coordinates of the residual."""

    def __init__(self):
        self.index = {}

    def __call__(self, *name):
        if name not in self.index:
            self.index[name] = len(self.index)
        return self.index[name]

    def __len__(self):
        return len(self.index)


def reverse_model(m, n):
    s = Space()
    for t in range(m + 1):
        s("tok", t)
    for p in range(n + 1):
        s("idx", p)
    for p in range(1, n + 1):
        s("opp", p)
    for t in range(m + 1):
        s("out", t)
    d, dh = len(s), n + 1
    W_E = np.zeros((m + 1, d))
    for t in range(m + 1):
        W_E[t, s("tok", t)] = 1.0
    W_pos = np.zeros((n + 1, d))
    for p in range(n + 1):
        W_pos[p, s("idx", p)] = 1.0
        if p >= 1:
            W_pos[p, s("opp", n + 1 - p)] = 1.0
    Q, K, Vv, O = np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((d, dh))
    for p in range(1, n + 1):
        Q[p, s("opp", p)] = GAIN
        K[p, s("idx", p)] = 1.0
    for t in range(1, m + 1):
        Vv[t - 1, s("tok", t)] = 1.0
        O[s("out", t), t - 1] = 1.0
    W_U = np.zeros((m + 1, d))
    for t in range(1, m + 1):
        W_U[t, s("out", t)] = 10.0
    model = Transformer(W_E, W_pos, [{"W_Q": Q, "W_K": K, "W_V": Vv, "W_O": O}], W_U, n_heads=1, d_head=dh, readouts=range(1, n + 1), causal=False)
    components = [head_component(model, 0, 0, "aggregate tokens at the opposite index")]
    attention = [{"name": "aggregate tokens at the opposite index", "layer": 0, "head": 0, "targets": [[p, n + 1 - p] for p in range(1, n + 1)]}]
    return model, components, attention


def width_steps(values):
    """The thresholds between consecutive BOS shares `values` (descending)."""
    return [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]


def hist_model(m, n):
    s = Space()
    for t in range(m + 1):
        s("tok", t)
    s("one")
    s("frac")
    for c in range(1, n + 1):
        s("count", c)
    d = len(s)
    dh = m + 2
    bos = m + 1  # head coordinate every query and BOS's key share
    W_E = np.zeros((m + 1, d))
    for t in range(m + 1):
        W_E[t, s("tok", t)] = 1.0
    W_pos = np.zeros((n + 1, d))
    W_pos[:, s("one")] = 1.0
    Q, K, Vv, O = np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((d, dh))
    for t in range(1, m + 1):
        Q[t, s("tok", t)] = GAIN
        K[t, s("tok", t)] = 1.0
    Q[bos, s("one")] = GAIN
    K[bos, s("tok", 0)] = 1.0
    Vv[0, s("tok", 0)] = 1.0
    O[s("frac"), 0] = 1.0
    # count c in 1..n: share 1/(c+1); D_c = [count <= c] = [share >= t_c].
    shares = [1.0 / (c + 1) for c in range(1, n + 1)]
    steps = width_steps(shares)
    hidden = 2 * len(steps) + 1
    W_in, W_out = np.zeros((hidden, d)), np.zeros((d, hidden))
    groups = []
    for i, t in enumerate(steps):
        c = i + 1
        for j, offset in enumerate((1.0, 0.0)):
            u = 2 * i + j
            W_in[u, s("frac")] = STEP
            W_in[u, s("one")] = offset - STEP * t
            sign = 1.0 if j == 0 else -1.0
            W_out[s("count", c), u] = sign
            if c + 1 <= n:
                W_out[s("count", c + 1), u] = -sign
        groups.append(([2 * i, 2 * i + 1], f"step: count <= {c}"))
    W_in[hidden - 1, s("one")] = 1.0
    W_out[s("count", n), hidden - 1] = 1.0
    groups.append(([hidden - 1], f"constant: count <= {n}"))
    W_U = np.zeros((n + 1, d))
    for c in range(1, n + 1):
        W_U[c, s("count", c)] = 10.0
    layer = {"W_Q": Q, "W_K": K, "W_V": Vv, "W_O": O, "W_in": W_in, "W_out": W_out}
    model = Transformer(W_E, W_pos, [layer], W_U, n_heads=1, d_head=dh, readouts=range(1, n + 1), causal=False)
    components = [head_component(model, 0, 0, "select equal tokens and BOS: BOS share 1/(count+1)")]
    components += [unit_component(model, 0, units, name) for units, name in groups]
    return model, components, []


def sort_model(m, n):
    s = Space()
    for t in range(m + 1):
        s("tok", t)
    s("one")
    s("frac")
    for r in range(n):
        s("idx", r)
    for r in range(n):
        s("rank", r)
    for t in range(m + 1):
        s("out", t)
    d = len(s)
    dh = max(m + 2, n + 1)
    b = dh - 1  # BOS coordinate of the heads' query/key space
    W_E = np.zeros((m + 1, d))
    for t in range(m + 1):
        W_E[t, s("tok", t)] = 1.0
    W_pos = np.zeros((n + 1, d))
    W_pos[:, s("one")] = 1.0
    for p in range(1, n + 1):
        W_pos[p, s("idx", p - 1)] = 1.0
    # layer 0: select smaller tokens and BOS.
    Q0, K0, V0, O0 = np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((d, dh))
    for t in range(1, m + 1):
        Q0[t - 1, s("tok", t)] = GAIN
        for larger in range(t + 1, m + 1):
            K0[larger - 1, s("tok", t)] = 1.0
    Q0[b, s("one")] = GAIN
    K0[b, s("tok", 0)] = 1.0
    V0[0, s("tok", 0)] = 1.0
    O0[s("frac"), 0] = 1.0
    # rank r in 0..n-1: share 1/(r+1); D_r = [rank <= r] = [share >= t_r], silenced at BOS.
    shares = [1.0 / (r + 1) for r in range(n)]
    steps = width_steps(shares)
    hidden = 2 * len(steps) + 1
    W_in, W_out = np.zeros((hidden, d)), np.zeros((d, hidden))
    groups = []
    for r, t in enumerate(steps):
        for j, offset in enumerate((1.0, 0.0)):
            u = 2 * r + j
            W_in[u, s("frac")] = STEP
            W_in[u, s("one")] = offset - STEP * t
            W_in[u, s("tok", 0)] = -SUPPRESS
            sign = 1.0 if j == 0 else -1.0
            W_out[s("rank", r), u] = sign
            if r + 1 < n:
                W_out[s("rank", r + 1), u] = -sign
        groups.append(([2 * r, 2 * r + 1], f"step: rank <= {r}"))
    W_in[hidden - 1, s("one")] = 1.0
    W_in[hidden - 1, s("tok", 0)] = -1.0
    W_out[s("rank", n - 1), hidden - 1] = 1.0
    groups.append(([hidden - 1], "not BOS: rank <= n - 1"))
    # layer 1: output position p reads the token of rank p - 1; BOS at half the gain.
    Q1, K1, V1, O1 = np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((dh, d)), np.zeros((d, dh))
    for r in range(n):
        Q1[r, s("idx", r)] = GAIN
        K1[r, s("rank", r)] = 1.0
    Q1[b, s("one")] = GAIN / 2
    K1[b, s("tok", 0)] = 1.0
    for t in range(1, m + 1):
        V1[t - 1, s("tok", t)] = 1.0
        O1[s("out", t), t - 1] = 1.0
    W_U = np.zeros((m + 1, d))
    for t in range(1, m + 1):
        W_U[t, s("out", t)] = 10.0
    layers = [{"W_Q": Q0, "W_K": K0, "W_V": V0, "W_O": O0, "W_in": W_in, "W_out": W_out}, {"W_Q": Q1, "W_K": K1, "W_V": V1, "W_O": O1}]
    model = Transformer(W_E, W_pos, layers, W_U, n_heads=1, d_head=dh, readouts=range(1, n + 1), causal=False)
    components = [head_component(model, 0, 0, "select smaller tokens and BOS: BOS share 1/(rank+1)")]
    components += [unit_component(model, 0, units, name) for units, name in groups]
    components.append(head_component(model, 1, 0, "aggregate the token whose rank is the output index"))
    return model, components, []


def main():
    rng = np.random.default_rng(7)
    m_rev, n_rev = 5, 5
    m_hist, n_hist = 4, 6
    m_sort, n_sort = 8, 5
    specs = {
        "tracr_reverse": (reverse_model(m_rev, n_rev), lambda: np.concatenate([[0], rng.integers(1, m_rev + 1, n_rev)]), m_rev, n_rev),
        "tracr_hist": (hist_model(m_hist, n_hist), lambda: np.concatenate([[0], rng.integers(1, m_hist + 1, n_hist)]), m_hist, n_hist),
        "tracr_sort": (sort_model(m_sort, n_sort), lambda: np.concatenate([[0], rng.permutation(np.arange(1, m_sort + 1))[:n_sort]]), m_sort, n_sort),
    }
    for case, ((model, components, attention), prompt, m, n) in specs.items():
        tests = np.array([prompt() for _ in range(256)])
        logits = model.forward(tests)
        if case == "tracr_reverse":
            expect = tests[:, :0:-1]
        elif case == "tracr_hist":
            expect = np.array([[np.sum(row[1:] == row[p]) for p in range(1, n + 1)] for row in tests])
        else:
            expect = np.sort(tests[:, 1:], axis=1)
        assert (logits.argmax(-1) == expect).all(), case
        _, units = rotate_residual(model, components, rng)
        renumber(components, units)
        assert (model.forward(tests).argmax(-1) == expect).all(), case
        probe = np.array([prompt() for _ in range(64)])
        if case == "tracr_sort":
            ranks = np.argsort(np.argsort(probe[:, 1:], axis=1), axis=1)
            attention = [{"name": components[-1]["name"], "layer": 1, "head": 0,
                          "per_probe": [[[p, 1 + int(np.nonzero(row == p - 1)[0][0])] for p in range(1, n + 1)] for row in ranks]}]

        questions = []
        count = 0

        def add(tag, x, edits):
            nonlocal count
            questions.append(question(case, count, tag, x, edits))
            count += 1

        steps = [c for c in components if "units" in c]
        heads = [c for c in components if "head" in c]
        for _ in range(12):
            x = prompt()
            y = x.copy()
            if case == "tracr_sort":
                j = int(rng.integers(1, n + 1))
                y[j] = rng.choice(np.setdiff1d(np.arange(1, m + 1), x))
            else:
                y[int(rng.integers(1, n + 1))] = int(rng.integers(1, m + 1))
            add("input: change a token", x, [{"kind": "input", "input": y.tolist()}])
        for _ in range(8):
            for c in heads:
                p = int(rng.integers(1, n + 1))
                add(f"patch head {c['layer']} at one position", prompt(), [{"kind": "patch", "place": place("head", c["layer"], p, head=0), "donor": prompt().tolist()}])
                add(f"scale head {c['layer']}", prompt(), [{"kind": "scale", "place": place("head", c["layer"], head=0), "factor": float(rng.choice([0.0, 0.5, 2.0]))}])
            for c in steps[:: max(1, len(steps) // 3)]:
                add("ablate a step", prompt(), [{"kind": "scale", "place": place("mlp", 0, units=c["units"]), "factor": 0.0}])
            if steps:
                p = int(rng.integers(1, n + 1))
                add("patch the decoded count", prompt(), [{"kind": "patch", "place": place("mlp", 0, p), "donor": prompt().tolist()}])
            p = int(rng.integers(1, n + 1))
            add("patch the residual after layer 0", prompt(), [{"kind": "patch", "place": place("resid", 0, p), "donor": prompt().tolist()}])
        samples = np.array([prompt() for _ in range(1024)])
        record = model.export(case, samples, f"BOS then {n} tokens of 1..{m}" + (" (distinct)" if case == "tracr_sort" else ""), vocab=m + 1, classes=logits.shape[-1])
        truth = {"mechanism": __doc__.split("* ")[1 + list(specs).index(case)].strip(), "components": components, "attention": attention, "probe": probe.tolist()}
        write_case(case, model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
