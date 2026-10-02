"""Export the APD/SPD residual-MLP toys (1, 2, 3 layers) in the engine's residual_mlp format.

L2 and L3 are the trained weights in ~/mpd-data/resid_mlp/L{2,3}; L1 is trained here with the same
recipe (fixed random unit-row W_E, W_U = W_E^T, ReLU blocks without biases, inputs x_i ~ U(-1, 1)
with probability 0.01, labels x + relu(x), AdamW 3e-3 wd 0.01 cosine to 0, batch 2048).
The samples: 1024 fresh draws of the input law.
"""
import hashlib
import json
import os

import numpy as np
import torch

ROOT = os.path.expanduser("~/mpd-data")
rng = np.random.default_rng(0)


def draws(n, features, p=0.01):
    x = rng.uniform(-1, 1, size=(n, features)) * (rng.uniform(size=(n, features)) < p)
    return x


def train_l1(features=100, width=1000, hidden=50, steps=1000, seed=0):
    torch.manual_seed(seed)
    W_E = torch.randn(features, width, dtype=torch.float64)
    W_E /= W_E.norm(dim=1, keepdim=True)
    lin_in = torch.nn.Linear(width, hidden, bias=False).double()
    lin_out = torch.nn.Linear(hidden, width, bias=False).double()
    opt = torch.optim.AdamW(list(lin_in.parameters()) + list(lin_out.parameters()), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.0)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        x = (torch.rand(2048, features, generator=gen, dtype=torch.float64) * 2 - 1) * (
            torch.rand(2048, features, generator=gen, dtype=torch.float64) < 0.01)
        r = x @ W_E
        r = r + lin_out(torch.relu(lin_in(r)))
        y = r @ W_E.T
        loss = ((y - (x + torch.relu(x))) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    print("L1 final loss", loss.item())
    return {"W_E": W_E.numpy(), "W_in0": lin_in.weight.detach().numpy(), "W_out0": lin_out.weight.detach().numpy(),
            "W_U": W_E.T.numpy()}


def sha256(path):
    h = hashlib.sha256()
    h.update(open(path, "rb").read())
    return h.hexdigest()


for layers in (1, 2, 3):
    if layers == 1:
        w = train_l1()
    else:
        d = f"{ROOT}/resid_mlp/L{layers}"
        w = {f[:-4]: np.load(f"{d}/{f}") for f in os.listdir(d) if f.endswith(".npy")}
    out = f"{ROOT}/engine/resid_l{layers}"
    os.makedirs(out, exist_ok=True)
    files = {}

    def put(name, a):
        a = np.ascontiguousarray(np.atleast_2d(a), dtype="<f8")
        path = f"{out}/{name}.f64"
        a.tofile(path)
        files[name] = {"shape": list(a.shape), "sha256": sha256(path)}

    features, width = w["W_E"].shape
    put("W_E", w["W_E"])
    put("W_U", w["W_U"].T)  # outputs x d
    hidden = w["W_in0"].shape[0]
    for l in range(layers):
        put(f"blocks.{l}.W_in", w[f"W_in{l}"])
        put(f"blocks.{l}.W_out", w[f"W_out{l}"])
    x = draws(1024, features)
    np.ascontiguousarray(x, dtype="<f8").tofile(f"{out}/inputs.f64")
    record = {
        "model": f"resid_mlp_l{layers}", "kind": "residual_mlp",
        "config": {"n_layers": layers, "d_model": width, "d_mlp": hidden, "act": "relu", "normalization": None,
                   "n_inputs": features},
        "input": {"type": "real_vector", "n_inputs": features,
                  "generator": "each feature uniform on [-1, 1] with probability 0.01, else 0, independently"},
        "output": {"n_outputs": features, "decode": "each output logit, labels x + relu(x)"},
        "samples": {"file": "inputs.f64", "shape": [1024, features]},
        "files": files,
    }
    json.dump(record, open(f"{out}/export.json", "w"), indent=1)
    print(out, layers, hidden)
