"""A known-answer induction model in the engine's transformer format (#2951).

Two attention-only layers over a residual of token, previous-token, output and position subspaces:
layer 0's head attends from position p to p-1 (query and key read the one-hot positions) and copies
the token into the previous-token subspace; layer 1's head attends from the current token to the
position whose previous token equals it and copies that position's token into the output subspace;
the unembedding reads the output subspace. On [x_1..x_n, x_1..x_n] (distinct tokens) the readout at
position n + i predicts x_{i+2}.
"""
import json
import os

import numpy as np

V, N = 12, 6
L = 2 * N
P = L
D_HEAD = max(V, P)
D = 3 * V + P
GAIN = 60.0
tok = lambda t: t
prev = lambda t: V + t
outc = lambda t: 2 * V + t
pos = lambda p: 3 * V + p

W_E = np.zeros((V, D))
for t in range(V):
    W_E[t, tok(t)] = 1.0
W_pos = np.zeros((P, D))
for p in range(P):
    W_pos[p, pos(p)] = 1.0
# layer 0: previous token
Q0, K0, V0, O0 = np.zeros((D_HEAD, D)), np.zeros((D_HEAD, D)), np.zeros((D_HEAD, D)), np.zeros((D, D_HEAD))
for p in range(P):
    Q0[p, pos(p)] = GAIN
    if p + 1 < P:
        K0[p + 1, pos(p)] = 1.0
for t in range(V):
    V0[t, tok(t)] = 1.0
    O0[prev(t), t] = 1.0
# layer 1: induction
Q1, K1, V1, O1 = np.zeros((D_HEAD, D)), np.zeros((D_HEAD, D)), np.zeros((D_HEAD, D)), np.zeros((D, D_HEAD))
for t in range(V):
    Q1[t, tok(t)] = GAIN
    K1[t, prev(t)] = 1.0
    V1[t, tok(t)] = 1.0
    O1[outc(t), t] = 1.0
W_U = np.zeros((V, D))
for t in range(V):
    W_U[t, outc(t)] = 10.0

rng = np.random.default_rng(0)
samples = np.array([np.tile(rng.permutation(V)[:N], 2) for _ in range(4096)], dtype=np.float64)
out = os.path.expanduser("~/mpd-data/engine/induction")
os.makedirs(out, exist_ok=True)
files = {}


def put(name, a):
    a = np.ascontiguousarray(np.atleast_2d(a), dtype="<f8")
    a.tofile(f"{out}/{name}.f64")
    files[name] = {"shape": list(a.shape)}


put("W_E", W_E)
put("W_pos", W_pos)
put("W_U", W_U)
for l, (q, k, v, o) in enumerate(((Q0, K0, V0, O0), (Q1, K1, V1, O1))):
    put(f"blocks.{l}.W_Q", q)
    put(f"blocks.{l}.W_K", k)
    put(f"blocks.{l}.W_V", v)
    put(f"blocks.{l}.W_O", o)
samples.astype("<f8").tofile(f"{out}/inputs.f64")
record = {
    "model": "induction_known_answer", "kind": "transformer",
    "config": {"n_layers": 2, "n_heads": 1, "d_model": D, "d_head": D_HEAD, "d_mlp": 0, "n_ctx": L, "causal": True,
               "act": "relu", "normalization": None},
    "input": {"type": "token_sequence", "vocab_size": V, "seq_len": L,
              "generator": f"{N} distinct tokens of {V} in random order, repeated once"},
    "output": {"n_classes": V, "readout_positions": list(range(N, L - 1)), "decode": "argmax"},
    "samples": {"file": "inputs.f64", "shape": [4096, L]},
    "files": files,
}
json.dump(record, open(f"{out}/export.json", "w"), indent=1)
print(out)
