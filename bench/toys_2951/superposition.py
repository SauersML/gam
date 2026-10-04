"""Case `superposition`: a toy model of superposition (Elhage et al. 2022), trained.

Twelve sparse features (each present with probability 0.05, value uniform on [0, 1]) are
written into a four-dimensional bottleneck `h = W x`; unit `i` reads `relu(W_i . h + b_i)`, and
output `i` is the logit that feature `i` is present, `a_i relu(..) + c_i`. As a residual MLP:
`W_E` writes the bottleneck, the MLP's unit `i` reads it and writes output coordinate `i`, and
the unembedding reads the output coordinates; then the residual is turned into a random basis.

Known mechanism: feature `i` is the `i`-th input coordinate, stored along `W_i`; its weight
component is `W_E`'s row `i`, the MLP's unit `i` (row of `W_in`, column of `W_out`), active on
the inputs where feature `i` is present. Features share the bottleneck, so two present features
whose directions overlap interfere.
"""
import numpy as np
import torch

from common import ResidualMlp, place, question, renumber, rotate_residual, unit_component, write_case

N, M, P = 12, 4, 0.05


def sparse(rng, rows):
    present = rng.random((rows, N)) < P
    return present * rng.random((rows, N))


def train(seed=0, steps=20000):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(0.3 * torch.randn(M, N, dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(N, dtype=torch.float64))
    a = torch.nn.Parameter(torch.full((N,), 4.0, dtype=torch.float64))
    c = torch.nn.Parameter(torch.full((N,), -3.0, dtype=torch.float64))
    opt = torch.optim.Adam([W, b, a, c], lr=3e-3)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        x = torch.from_numpy(sparse(rng, 1024))
        logit = a * torch.relu((x @ W.T) @ W + b) + c
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, (x > 0).double())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach().numpy(), b.detach().numpy(), a.detach().numpy(), c.detach().numpy(), float(loss)


def main():
    W, b, a, c, loss = train()
    d = M + N
    W_E = np.zeros((N, d))
    W_E[:, :M] = W.T
    W_in = np.zeros((N, d))
    W_in[:, :M] = W.T
    W_out = np.zeros((d, N))
    W_out[M:, :] = np.eye(N)
    W_U = np.zeros((N, d))
    W_U[:, M:] = np.diag(a)
    model = ResidualMlp(W_E, [{"W_in": W_in, "b_in": b, "W_out": W_out}], W_U, b_U=c)
    rng = np.random.default_rng(12)
    samples = sparse(rng, 2048)
    probe = sparse(rng, 512)
    decided = model.forward(probe)[..., 1] > 0
    accuracy = (decided == (probe > 0)).mean()
    print(f"trained: loss {loss:.4f}, presence accuracy {accuracy:.4f}")
    components = []
    for i in range(N):
        unit = unit_component(model, 0, [i], f"feature {i}", feature=i, active=[int(r) for r in np.nonzero(probe[:, i] > 0)[0]])
        embedding = np.zeros_like(W_E)
        embedding[i] = W_E[i]
        unit["pieces"]["W_E"] = embedding
        components.append(unit)
    _, units = rotate_residual(model, components, rng)
    renumber(components, units)
    unit_of = {comp["feature"]: comp["units"][0] for comp in components}
    overlap = W.T @ W
    np.fill_diagonal(overlap, 0.0)

    questions = []
    n = 0

    def add(tag, x, edits):
        nonlocal n
        questions.append(question("superposition", n, tag, x, edits))
        n += 1

    def one(i, value=None):
        x = np.zeros(N)
        x[i] = rng.uniform(0.3, 1.0) if value is None else value
        return x

    for _ in range(10):
        i = int(rng.integers(N))
        j = int(np.argmax(overlap[i]))
        x = one(i)
        add("input: add the most overlapping feature", x, [{"kind": "input", "input": (x + one(j)).tolist()}])
        j = int(np.argmin(overlap[i]))
        x = one(i)
        add("input: add the most opposed feature", x, [{"kind": "input", "input": (x + one(j)).tolist()}])
        x = one(i)
        add("input: rescale the feature", x, [{"kind": "input", "input": (x * rng.choice([0.2, 0.5, 2.0])).tolist()}])
        add("ablate the feature's unit", one(i), [{"kind": "scale", "place": place("mlp", 0, units=[unit_of[i]]), "factor": 0.0}])
        k = int(rng.integers(N))
        add("ablate another feature's unit", one(i), [{"kind": "scale", "place": place("mlp", 0, units=[unit_of[k]]), "factor": 0.0}])
        add("patch the bottleneck", one(i), [{"kind": "patch", "place": place("resid", -1), "donor": one(int(rng.integers(N))).tolist()}])
        add("scale the bottleneck", sparse(rng, 1)[0] + one(i), [{"kind": "scale", "place": place("resid", -1), "factor": float(rng.choice([0.3, 0.6, 1.5]))}])
        add("patch the feature's unit", one(i), [{"kind": "patch", "place": place("mlp", 0, units=[unit_of[i]]), "donor": one(int(rng.integers(N))).tolist()}])
    record = model.export("superposition_12_in_4", samples, f"each of {N} features present with probability {P}, value uniform on [0, 1]", outputs=N)
    truth = {"mechanism": __doc__.split("\n\n")[2].strip(), "components": components, "probe": probe.tolist()}
    write_case("superposition", model, record, samples, questions, truth)


if __name__ == "__main__":
    main()
