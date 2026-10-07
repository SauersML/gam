"""Train the ground-truth toys of the #2951 toy gate and write each as an export with its known
mechanism. Toys gate the fitting method; they are never results.

usage: mem-lease 2 ~/mpd-data/venv/bin/python bench/toys_2951/train_toys.py TOY OUT_ROOT

TOY is one of
  tms_40_10       TMS 40-10 (Elhage et al.; SPD 3.1): y = ReLU(W^T W x + b), 40 features in 10
  tms_40_10_id    the same with a frozen 10x10 identity between W and W^T (SPD 3.2)
  resid_mlp_1l    compressed computation (SPD 3.3): 100 functions x + ReLU(x) through 50 neurons
  resid_mlp_2l    the same over 2 MLP layers of 50 (SPD 3.4, this repo's configs)
  resid_mlp_3l    the same over 3 MLP layers of 50
  modadd_113      1-layer transformer on (a + b) mod 113, trained to grokking (Fourier clocks)
  induction       2-layer attention-only transformer on repeated random segments

The SPD toys use param_decomp's configs (experiments/{tms,resid_mlp}/configs): sizes, feature
probability, value range, pretraining steps, batch and rate. The transformers are pre-norm RMS
rotary models in the engine's export layout (gam_mpd::import::import_language_model reads them).

OUT_ROOT/TOY/ holds export.json, the tensors as raw little-endian float64 `<name>.f64`, the
held-out inputs (`inputs.f64` for the real-input toys, `tokens.f64` for the transformers) and
truth.json: the known mechanisms, each a part with its rank, its weight on every operator it
spans (`truth.<i>.<operator>.f64`, the dense delta of that operator), and its activity on every
held-out row (`truth_active.f64`, rows x mechanisms, 1 where the mechanism is active).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
HELD_OUT = 4096


def write(out: Path, files: dict, name: str, value) -> None:
    a = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if a.ndim == 1:
        a = a[None, :]
    a.astype("<f8").tofile(out / f"{name}.f64")
    files[name] = {"shape": list(a.shape)}


def sparse_features(gen, rows, n, p, low, high):
    values = low + (high - low) * torch.rand(rows, n, generator=gen)
    return values * (torch.rand(rows, n, generator=gen) < p)


# ----------------------------------------------------------------------------------------- TMS


def tms(out: Path, identity: bool):
    n, h, p = 40, 10, 0.05
    gen = torch.Generator().manual_seed(0)
    bound = 1 / math.sqrt(n)
    W = (torch.rand(h, n, generator=gen) * 2 - 1) * bound
    W.requires_grad_(True)
    b = torch.zeros(n, requires_grad=True)
    I = torch.eye(h)
    opt = torch.optim.Adam([W, b], lr=5e-3)
    steps = 10000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))

    def forward(x):
        hidden = x @ W.T
        if identity:
            hidden = hidden @ I.T
        return torch.relu(hidden @ W + b)

    for step in range(steps):
        x = sparse_features(gen, 8192, n, p, 0.0, 1.0)
        loss = ((forward(x) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"step {step}: mse {loss.item():.3e}", flush=True)
    W, b = W.detach().double(), b.detach().double()
    xs = sparse_features(torch.Generator().manual_seed(1), HELD_OUT, n, p, 0.0, 1.0).double()
    with torch.no_grad():
        y = torch.relu((xs @ W.T) @ W + b)
        s2 = ((y - xs) ** 2).mean().item()
    files = {}
    write(out, files, "linear1", W.numpy())  # (h, n): hidden = W x
    if identity:
        write(out, files, "hidden_layers.0", np.eye(h))
    write(out, files, "linear2", W.T.numpy())  # (n, h): tied, its own operator
    write(out, files, "linear2.bias", b.numpy())
    write(out, files, "inputs", xs.numpy())
    ops = ["linear1"] + (["hidden_layers.0"] if identity else []) + ["linear2"]
    mechanisms, active = [], []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        w = W[:, i].numpy()
        pieces = {"linear1": np.outer(w, e), "linear2": np.outer(e, w)}
        mechanisms.append({"name": f"feature {i}", "rank": 1, "operators": pieces, "gate": f"x[{i}] > 0"})
        active.append(xs[:, i].numpy() > 0)
    if identity:
        mechanisms.append({"name": "hidden identity", "rank": h, "operators": {"hidden_layers.0": np.eye(h)}, "gate": "always"})
        active.append(np.ones(HELD_OUT, dtype=bool))
    record = {
        "model": out.name,
        "kind": "tms",
        "config": {"n_features": n, "n_hidden": h, "identity": identity, "feature_probability": p, "value_range": [0, 1]},
        "forward": "y = ReLU(linear2 (hidden_layers.0) linear1 x + linear2.bias); linear2 = linear1^T (tied)",
        "operators": ops,
        "output_variance": s2,
        "spd_published": "SPD 3.1/3.2: one subcomponent per feature in each of linear1 and linear2, the identity as a full-rank subcomponent set",
    }
    finish(out, files, record, mechanisms, np.stack(active, 1))


# ---------------------------------------------------------------------------- resid MLP (CC)


def resid_mlp(out: Path, layers: int):
    n, d, m, p = 100, 1000, 50, 0.01
    steps = {1: 2000, 2: 3000, 3: 4000}[layers]
    gen = torch.Generator().manual_seed(0)
    E = torch.randn(n, d, generator=gen)
    E = E / E.norm(dim=1, keepdim=True)  # fixed random unit-norm embedding, W_U = E^T
    W_in = [((torch.rand(m, d, generator=gen) * 2 - 1) / math.sqrt(d)).requires_grad_(True) for _ in range(layers)]
    W_out = [((torch.rand(d, m, generator=gen) * 2 - 1) / math.sqrt(m)).requires_grad_(True) for _ in range(layers)]
    opt = torch.optim.Adam(W_in + W_out, lr=3e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))

    def forward(x, Wi, Wo):
        r = x @ E
        for l in range(layers):
            r = r + torch.relu(r @ Wi[l].T) @ Wo[l].T
        return r @ E.T

    for step in range(steps):
        x = sparse_features(gen, 2048, n, p, -1.0, 1.0)
        loss = ((forward(x, W_in, W_out) - (x + torch.relu(x))) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step}: mse {loss.item():.3e}", flush=True)
    E = E.double()
    Wi = [w.detach().double() for w in W_in]
    Wo = [w.detach().double() for w in W_out]
    xs = sparse_features(torch.Generator().manual_seed(1), HELD_OUT, n, p, -1.0, 1.0).double()
    with torch.no_grad():
        y = forward(xs, Wi, Wo)
        s2 = ((y - (xs + torch.relu(xs))) ** 2).mean().item()
    files = {}
    write(out, files, "W_E", E.numpy())
    ops = []
    for l in range(layers):
        write(out, files, f"blocks.{l}.mlp.W_in", Wi[l].numpy())
        write(out, files, f"blocks.{l}.mlp.W_out", Wo[l].numpy())
        ops += [f"blocks.{l}.mlp.W_in", f"blocks.{l}.mlp.W_out"]
    write(out, files, "inputs", xs.numpy())
    # Feature i's function: every layer's read of its embedding direction e_i and write onto it,
    # through the dual frame of the embedding (the e_i are near orthogonal, not exactly).
    En = E.numpy()
    dual = np.linalg.pinv(En.T)  # (n, d): dual[i] . e_j = delta_ij on the embedding's span
    mechanisms, active = [], []
    for i in range(n):
        pieces = {}
        for l in range(layers):
            Wil, Wol = Wi[l].numpy(), Wo[l].numpy()
            pieces[f"blocks.{l}.mlp.W_in"] = np.outer(Wil @ En[i], dual[i])
            pieces[f"blocks.{l}.mlp.W_out"] = np.outer(dual[i], En[i] @ Wol)
        mechanisms.append({"name": f"function {i}", "rank": 1, "operators": pieces, "gate": f"x[{i}] != 0"})
        active.append(xs[:, i].numpy() != 0)
    record = {
        "model": out.name,
        "kind": "resid_mlp",
        "config": {"n_features": n, "d_embed": d, "d_mlp": m, "n_layers": layers, "feature_probability": p, "value_range": [-1, 1], "label": "x + ReLU(x)"},
        "forward": "r = W_E^T x; per layer r += W_out ReLU(W_in r); y = W_E r (W_E fixed, not decomposed)",
        "operators": ops,
        "output_variance": s2,
        "spd_published": {
            "resid_mlp_1l": "SPD 3.3: 100 subcomponents in W_in, one per function; W_out dense",
            "resid_mlp_2l": "SPD 3.4: one subcomponent per function across the 2 layers",
            "resid_mlp_3l": "SPD 3.4: one subcomponent per function across the 3 layers",
        }[out.name],
    }
    finish(out, files, record, mechanisms, np.stack(active, 1))


# ------------------------------------------------------------------------------- transformers


def rms(x, g, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g


def rotary(x, theta):
    # x: (..., t, hd), rotate_half pairing (dims i and i + hd/2)
    t, hd = x.shape[-2], x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=x.dtype) / hd))
    ang = torch.arange(t, dtype=x.dtype)[:, None] * inv[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Lm(torch.nn.Module):
    def __init__(self, vocab, d, layers, heads, hd, mlp, theta=10000, eps=1e-6, mlp_on=True):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, hd=hd, mlp=mlp, theta=theta, eps=eps)
        P = torch.nn.Parameter
        self.wte = P(torch.randn(vocab, d) * 0.02 * 5)
        self.lm_head = P(torch.randn(vocab, d) / math.sqrt(d))
        self.final = P(torch.ones(d))
        self.q = torch.nn.ParameterList([P(torch.randn(heads * hd, d) / math.sqrt(d)) for _ in range(layers)])
        self.k = torch.nn.ParameterList([P(torch.randn(heads * hd, d) / math.sqrt(d)) for _ in range(layers)])
        self.v = torch.nn.ParameterList([P(torch.randn(heads * hd, d) / math.sqrt(d)) for _ in range(layers)])
        self.o = torch.nn.ParameterList([P(torch.randn(d, heads * hd) / math.sqrt(heads * hd)) for _ in range(layers)])
        self.g1 = torch.nn.ParameterList([P(torch.ones(d)) for _ in range(layers)])
        self.g2 = torch.nn.ParameterList([P(torch.ones(d)) for _ in range(layers)])
        self.mlp_on = mlp_on
        self.up = torch.nn.ParameterList([P(torch.randn(mlp, d) / math.sqrt(d) * mlp_on) for _ in range(layers)])
        self.down = torch.nn.ParameterList([P(torch.randn(d, mlp) / math.sqrt(mlp) * mlp_on) for _ in range(layers)])
        if not mlp_on:
            for t in list(self.up) + list(self.down) + list(self.g2):
                t.requires_grad_(False)

    def forward(self, tokens, attn_out=None, mlp_out=None):
        c = self.cfg
        B, T = tokens.shape
        x = self.wte[tokens]
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        for l in range(c["layers"]):
            a = rms(x, self.g1[l], c["eps"])
            q = (a @ self.q[l].T).view(B, T, c["heads"], c["hd"]).transpose(1, 2)
            k = (a @ self.k[l].T).view(B, T, c["heads"], c["hd"]).transpose(1, 2)
            v = (a @ self.v[l].T).view(B, T, c["heads"], c["hd"]).transpose(1, 2)
            q, k = rotary(q, c["theta"]), rotary(k, c["theta"])
            s = (q @ k.transpose(-1, -2)) / math.sqrt(c["hd"])
            s = s.masked_fill(~mask, float("-inf"))
            pat = s.softmax(-1)
            if attn_out is not None:
                attn_out.append(pat.detach())
            z = (pat @ v).transpose(1, 2).reshape(B, T, c["heads"] * c["hd"])
            x = x + z @ self.o[l].T
            if self.mlp_on:
                m = torch.relu(rms(x, self.g2[l], c["eps"]) @ self.up[l].T)
                if mlp_out is not None:
                    mlp_out.append(m.detach())
                x = x + m @ self.down[l].T
        return rms(x, self.final, c["eps"]) @ self.lm_head.T

    def export(self, out: Path, files: dict):
        c = self.cfg
        write(out, files, "wte", self.wte.detach().double().numpy())
        write(out, files, "lm_head", self.lm_head.detach().double().numpy())
        write(out, files, "final_norm.gain", self.final.detach().double().numpy())
        for l in range(c["layers"]):
            for name, t in [("attn.q_proj", self.q[l]), ("attn.k_proj", self.k[l]), ("attn.v_proj", self.v[l]), ("attn.o_proj", self.o[l]),
                            ("mlp.c_fc", self.up[l]), ("mlp.down_proj", self.down[l]), ("rms1.gain", self.g1[l]), ("rms2.gain", self.g2[l])]:
                write(out, files, f"blocks.{l}.{name}", t.detach().double().numpy())
        return {"d_model": c["d"], "n_layers": c["layers"], "n_heads": c["heads"], "n_kv_heads": c["heads"], "head_dim": c["hd"], "d_mlp": c["mlp"],
                "vocab": c["vocab"], "rope_theta": float(c["theta"]), "rope_pairing": "rotate_half", "norm_eps": c["eps"], "mlp_act": "relu",
                "tied_embeddings": False, "norm": "rms"}


def head_pieces(model, l, h):
    """A head's slices: its rows of q, k, v and its columns of o, as dense deltas."""
    c = model.cfg
    r = slice(h * c["hd"], (h + 1) * c["hd"])
    pieces = {}
    for name, t in [("attn.q_proj", model.q[l]), ("attn.k_proj", model.k[l]), ("attn.v_proj", model.v[l])]:
        z = np.zeros(tuple(t.shape))
        z[r] = t.detach().double().numpy()[r]
        pieces[f"blocks.{l}.{name}"] = z
    z = np.zeros(tuple(model.o[l].shape))
    z[:, r] = model.o[l].detach().double().numpy()[:, r]
    pieces[f"blocks.{l}.attn.o_proj"] = z
    return pieces


def modadd(out: Path):
    p = 113
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    tokens = torch.stack([a, b, torch.full_like(a, p)], 1)
    labels = (a + b) % p
    perm = torch.randperm(p * p, generator=gen)
    train, test = perm[: int(0.3 * p * p)], perm[int(0.3 * p * p):]
    model = Lm(vocab=p + 1, d=128, layers=1, heads=4, hd=32, mlp=512)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    for step in range(30000):
        logits = model(tokens[train])[:, -1, :p]
        loss = torch.nn.functional.cross_entropy(logits, labels[train])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                tl = model(tokens[test])[:, -1, :p]
                acc = (tl.argmax(-1) == labels[test]).float().mean().item()
                tloss = torch.nn.functional.cross_entropy(tl, labels[test]).item()
            print(f"step {step}: train {loss.item():.3e} test {tloss:.3e} acc {acc:.4f}", flush=True)
            if acc == 1.0 and tloss < 1e-3:
                break
    files = {}
    config = model.export(out, files)
    # held-out rows: the test pairs (never trained), shuffled
    rows = tokens[test][torch.randperm(len(test), generator=gen)][:HELD_OUT]
    write(out, files, "tokens", rows.double().numpy())
    # Fourier key frequencies from the embedding: the power of W_E's token rows on cos/sin(2 pi k a / p)
    WE = model.wte.detach().double().numpy()[:p]
    t = np.arange(p)
    power = []
    for k in range(1, p // 2 + 1):
        basis = np.stack([np.cos(2 * np.pi * k * t / p), np.sin(2 * np.pi * k * t / p)], 1) / math.sqrt(p / 2)
        power.append(np.sum((basis.T @ WE) ** 2))
    power = np.array(power)
    order = np.argsort(-power)
    # key frequencies: the fewest whose embedding power is 90% of the total, a reporting cut only
    cum = np.cumsum(power[order]) / power.sum()
    keys = [int(order[i]) + 1 for i in range(int(np.searchsorted(cum, 0.9)) + 1)]
    # MLP neurons by frequency: each neuron's activation over (a, b) on the full grid, its
    # Fourier power in a + b and a, b separately, assigned to the key frequency it carries most
    mlp_acts = []
    with torch.no_grad():
        model(tokens, mlp_out=mlp_acts)
    acts = mlp_acts[0][:, -1].double().numpy().reshape(p, p, -1)
    A = np.fft.fft2(acts, axes=(0, 1))
    nfreq = p // 2 + 1
    neuron_power = np.zeros((acts.shape[2], nfreq))
    for k in range(1, nfreq):
        for (i, j) in [(k, 0), (0, k), (k, k), (k, p - k), (p - k, 0), (0, p - k), (p - k, p - k), (p - k, k)]:
            neuron_power[:, k] += np.abs(A[i, j]) ** 2
    total = (np.abs(A) ** 2).sum((0, 1)) - np.abs(A[0, 0]) ** 2
    best = neuron_power.argmax(1)
    share = neuron_power[np.arange(len(best)), best] / np.maximum(total, 1e-30)
    mechanisms = []
    WE_full = model.wte.detach().double().numpy()
    WU = model.lm_head.detach().double().numpy()
    up = model.up[0].detach().double().numpy()
    down = model.down[0].detach().double().numpy()
    for k in keys:
        basis = np.stack([np.cos(2 * np.pi * k * t / p), np.sin(2 * np.pi * k * t / p)], 1) / math.sqrt(p / 2)
        proj = np.zeros((p + 1, p + 1))
        proj[:p, :p] = basis @ basis.T
        neurons = np.where((best == k) & (share > 0.5))[0]
        zu = np.zeros_like(up)
        zu[neurons] = up[neurons]
        zd = np.zeros_like(down)
        zd[:, neurons] = down[:, neurons]
        pieces = {"wte": proj @ WE_full, "lm_head": proj @ WU, "blocks.0.mlp.c_fc": zu, "blocks.0.mlp.down_proj": zd}
        mechanisms.append({"name": f"clock k={k}", "rank": {"wte": 2, "lm_head": 2, "neurons": int(len(neurons))}, "operators": pieces,
                           "gate": "always (every input uses every key frequency)", "neurons": neurons.tolist()})
    active = np.ones((len(rows), len(mechanisms)), dtype=bool)
    record = {"model": "modadd_113", "kind": "language_model", "config": config,
              "task": "tokens a, b, = (113); the prediction at position 2 is (a + b) mod 113; positions 0, 1 untrained",
              "key_frequencies": keys, "embedding_power": {int(k) + 1: float(power[k]) for k in order[:10]},
              "spd_published": "none for this toy (Nanda et al. 2023: 5 key frequencies at p = 113)"}
    finish(out, files, record, mechanisms, active)


def induction(out: Path):
    vocab, T = 32, 32
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    def batch(n, g):
        # BOS (0), then a random segment of 6..12 tokens repeated: [BOS, r, r, filler...]
        x = torch.randint(1, vocab, (n, T), generator=g)
        x[:, 0] = 0
        L = torch.randint(6, 13, (n,), generator=g)
        start = torch.randint(1, 4, (n,), generator=g)
        for i in range(n):
            s, l = int(start[i]), int(L[i])
            x[i, s + l: s + 2 * l] = x[i, s: s + l]
        return x

    model = Lm(vocab=vocab, d=64, layers=2, heads=4, hd=16, mlp=1, mlp_on=False)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=2e-3, weight_decay=0.01)
    for step in range(6000):
        x = batch(256, gen)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4f}", flush=True)
    files = {}
    config = model.export(out, files)
    rows = batch(HELD_OUT // 8, torch.Generator().manual_seed(1))
    write(out, files, "tokens", rows.double().numpy())
    # find the heads: previous-token score (mean attention from t to t-1) in layer 0; induction
    # score (attention from a repeated token to the token after its earlier occurrence) in layer 1
    pats = []
    with torch.no_grad():
        model(rows, pats)
    prev = pats[0][:, :, torch.arange(1, T), torch.arange(0, T - 1)].mean((0, 2)).numpy()
    ind = np.zeros(model.cfg["heads"])
    targets = []
    for i in range(rows.shape[0]):
        seen = {}
        for t_ in range(T):
            tok = int(rows[i, t_])
            if tok in seen and seen[tok] + 1 < t_:
                targets.append((i, t_, seen[tok] + 1))
            seen[tok] = t_
    idx = torch.tensor(targets)
    ind = pats[1][idx[:, 0], :, idx[:, 1], idx[:, 2]].mean(0).numpy()
    h0, h1 = int(prev.argmax()), int(ind.argmax())
    print(f"previous-token scores {prev.round(3)}, induction scores {ind.round(3)}", flush=True)
    # where the induction mechanism is active: tokens whose earlier occurrence exists (it attends
    # back); the previous-token head writes at every token
    act = np.zeros((rows.shape[0] * T, 2), dtype=bool)
    act[:, 0] = True
    for (i, t_, _) in targets:
        act[i * T + t_, 1] = True
    mechanisms = [
        {"name": f"previous-token head L0H{h0}", "rank": model.cfg["hd"], "operators": head_pieces(model, 0, h0), "gate": "always", "score": float(prev[h0])},
        {"name": f"induction head L1H{h1}", "rank": model.cfg["hd"], "operators": head_pieces(model, 1, h1), "gate": "the token occurred before", "score": float(ind[h1])},
    ]
    record = {"model": "induction", "kind": "language_model", "config": config,
              "task": "BOS, random tokens with a 6..12-token segment repeated right after itself; next-token prediction at every position",
              "previous_token_scores": prev.tolist(), "induction_scores": ind.tolist(),
              "note": "attention-only: the MLP weights are zero (the importer needs an MLP)",
              "spd_published": "none (VPD/SPD do not decompose this toy)"}
    finish(out, files, record, mechanisms, act)


def finish(out: Path, files: dict, record: dict, mechanisms: list, active: np.ndarray):
    truth = []
    for i, mech in enumerate(mechanisms):
        ops = {}
        for op, delta in mech["operators"].items():
            name = f"truth.{i}.{op}"
            write(out, {}, name, delta)
            ops[op] = {"file": f"{name}.f64", "shape": list(np.atleast_2d(delta).shape)}
        truth.append({k: v for k, v in mech.items() if k != "operators"} | {"operators": ops})
    write(out, {}, "truth_active", active.astype(np.float64))
    record["files"] = files
    (out / "export.json").write_text(json.dumps(record, indent=1))
    (out / "truth.json").write_text(json.dumps({"mechanisms": truth, "active": {"file": "truth_active.f64", "shape": list(active.shape)}}, indent=1))
    print(f"wrote {out}: {len(mechanisms)} mechanisms", flush=True)


if __name__ == "__main__":
    toy, root = sys.argv[1], Path(sys.argv[2])
    out = root / toy
    out.mkdir(parents=True, exist_ok=True)
    {
        "tms_40_10": lambda: tms(out, False),
        "tms_40_10_id": lambda: tms(out, True),
        "resid_mlp_1l": lambda: resid_mlp(out, 1),
        "resid_mlp_2l": lambda: resid_mlp(out, 2),
        "resid_mlp_3l": lambda: resid_mlp(out, 3),
        "modadd_113": lambda: modadd(out),
        "induction": lambda: induction(out),
    }[toy]()
