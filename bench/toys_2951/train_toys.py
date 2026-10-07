"""Train the ground-truth toys of the #2951 toy gate and write each as an engine export with its
known mechanism. Toys gate the fitting method; they are never results.

usage: mem-lease 3 ~/mpd-data/venv/bin/python bench/toys_2951/train_toys.py TOY[@sSEED] OUT_ROOT

TOY is one of
  tms_40_10       TMS 40-10 (Elhage et al.; SPD 3.1): y = ReLU(W^T W x + b), 40 features in 10
  tms_40_10_id    the same with a frozen 10x10 identity between W and W^T (SPD 3.2)
  resid_mlp_1l    compressed computation (SPD 3.3): 100 functions x + ReLU(x) through 50 neurons
  resid_mlp_2l    the same over 2 MLP layers of 50 (SPD 3.4, this repo's configs)
  resid_mlp_3l    the same over 3 MLP layers of 50
  modadd_113      1-layer transformer on (a + b) mod 113, trained to grokking (Fourier clocks)
  induction       2-layer attention-only transformer on repeated random segments
`@sSEED` trains another seed of the toy (initialization; the inputs and the resid_mlp
embedding are shared by every seed) into OUT_ROOT/TOY_sSEED, for the universality score.

The SPD toys use param_decomp's configs (experiments/{tms,resid_mlp}/configs): sizes, feature
probability, value range, pretraining steps, batch and rate. Every toy is a language model that
gam_mpd::import::import_language_model reads, so all of them go through the same fitter: the
transformers are pre-norm RMS rotary models; a real-valued toy's tokens index its inputs (its
embedding is the table of inputs in the residual stream; HELD_OUT held-out inputs, then
TRAIN_ROWS training inputs, one token per sequence), each block has one zero attention head and
the toy's MLP, there is no norm, and its readout is the Gaussian head (fixed_head_target::
GAUSSIAN_HEAD) in units of the toy's task residual sigma, its root mean squared error against its
training target, so KL(M || P) = ||y_M - y_P||^2 / (2 sigma^2) nats per input.

OUT_ROOT/TOY/ holds export.json, the tensors as raw little-endian float64 `<name>.f64` (with
`tokens`) and truth.json: the known mechanisms, each a part with its rank, its weight on every
operator it spans (`truth.<i>.<operator>.f64`, the dense delta of that operator), and its activity
on every held-out token (`truth_active.f64`, tokens x mechanisms, 1 where the mechanism is
active).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
HELD_OUT = 4096
# the transformers train on the Apple GPU when present
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def write(out: Path, files: dict, name: str, value) -> None:
    a = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if a.ndim == 1:
        a = a[None, :]
    a.astype("<f8").tofile(out / f"{name}.f64")
    files[name] = {"shape": list(a.shape)}


def sparse_features(gen, rows, n, p, low, high):
    values = low + (high - low) * torch.rand(rows, n, generator=gen)
    return values * (torch.rand(rows, n, generator=gen) < p)


# ------------------------------------------------------------------------ real-valued toys


TRAIN_ROWS = 1 << 14


def real_export(out: Path, files: dict, stream: np.ndarray, mlps: list, act: str, head: np.ndarray, bias, relu: bool, sigma: float) -> dict:
    """A real-valued toy as the engine's language model (gam_mpd::import, GAUSSIAN_HEAD): each
    input is a token whose embedding is the input in the residual stream (`stream`, rows: the
    HELD_OUT held-out inputs, then the training inputs), every block one zero attention head
    (2 wide, no effect) and the toy's MLP (`mlps`: per block its c_fc and down_proj), no norm, and
    the Gaussian head `law(E h + b) / sigma` (`head` E, `bias` b, ReLU when `relu`), sigma the
    toy's task residual: its root mean squared error against its training target, per output."""
    rows, d = stream.shape
    write(out, files, "wte", stream)
    write(out, files, "tokens", np.arange(rows, dtype=np.float64)[:, None])
    for l, (up, down) in enumerate(mlps):
        for name, shape in [("attn.q_proj", (2, d)), ("attn.k_proj", (2, d)), ("attn.v_proj", (2, d)), ("attn.o_proj", (d, 2))]:
            write(out, files, f"blocks.{l}.{name}", np.zeros(shape))
        write(out, files, f"blocks.{l}.mlp.c_fc", up)
        write(out, files, f"blocks.{l}.mlp.down_proj", down)
    write(out, files, "gaussian_head", head / sigma)
    if bias is not None:
        write(out, files, "gaussian_head.bias", np.asarray(bias) / sigma)
    return {"d_model": d, "n_layers": len(mlps), "n_heads": 1, "n_kv_heads": 1, "head_dim": 2, "d_mlp": int(mlps[0][0].shape[0]), "vocab": rows,
            "rope_theta": 10000.0, "rope_pairing": "rotate_half", "norm_eps": 1e-6, "norm": "none", "mlp_act": act, "tied_embeddings": False,
            "head": {"law": "gaussian", "relu": relu, "task_residual": sigma}}


# ----------------------------------------------------------------------------------------- TMS


def tms(out: Path, identity: bool, seed: int):
    n, h, p = 40, 10, 0.05
    gen = torch.Generator().manual_seed(seed)
    bound = 1 / math.sqrt(n)
    W = (torch.rand(h, n, generator=gen) * 2 - 1) * bound
    W.requires_grad_(True)
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=5e-3)
    steps = 10000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    for step in range(steps):
        x = sparse_features(gen, 8192, n, p, 0.0, 1.0)
        # the frozen hidden identity changes nothing in the forward pass
        loss = ((torch.relu(x @ W.T @ W + b) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"step {step}: mse {loss.item():.3e}", flush=True)
    W, b = W.detach().double().numpy(), b.detach().double().numpy()
    # held-out inputs (seed 1) then training inputs (seed 2), the same for every seed of the toy
    xs = sparse_features(torch.Generator().manual_seed(1), HELD_OUT, n, p, 0.0, 1.0).double().numpy()
    train = sparse_features(torch.Generator().manual_seed(2), TRAIN_ROWS, n, p, 0.0, 1.0).double().numpy()
    sigma = math.sqrt(float(((np.maximum(xs @ W.T @ W + b, 0.0) - xs) ** 2).mean()))
    # The stream: x in [0, n), W^T W x in [n, 2n), and with the identity the hidden in [2n, 2n + h).
    d = 2 * n + (h if identity else 0)
    stream = np.zeros((HELD_OUT + TRAIN_ROWS, d))
    stream[:, :n] = np.concatenate([xs, train])
    read = np.zeros((h, d))
    read[:, :n] = W
    write_z = np.zeros((d, h))
    write_z[n:2 * n] = W.T
    if identity:
        to_hidden = np.zeros((d, h))
        to_hidden[2 * n:] = np.eye(h)
        from_hidden = np.zeros((h, d))
        from_hidden[:, 2 * n:] = np.eye(h)
        mlps = [(read, to_hidden), (from_hidden, write_z)]
    else:
        mlps = [(read, write_z)]
    head = np.zeros((n, d))
    head[:, n:2 * n] = np.eye(n)
    files = {}
    config = real_export(out, files, stream, mlps, "identity", head, b, True, sigma)
    last = len(mlps) - 1
    mechanisms, active = [], []
    for i in range(n):
        e = np.zeros(d)
        e[i] = 1.0
        z = np.zeros(d)
        z[n + i] = 1.0
        pieces = {"blocks.0.mlp.c_fc": np.outer(W[:, i], e), f"blocks.{last}.mlp.down_proj": np.outer(z, W[:, i])}
        mechanisms.append({"name": f"feature {i}", "rank": 1, "operators": pieces, "gate": f"x[{i}] > 0"})
        active.append(xs[:, i] > 0)
    if identity:
        mechanisms.append({"name": "hidden identity", "rank": h, "operators": {"blocks.0.mlp.down_proj": mlps[0][1], "blocks.1.mlp.c_fc": mlps[1][0]}, "gate": "always"})
        active.append(np.ones(HELD_OUT, dtype=bool))
    record = {
        "model": out.name, "kind": "language_model", "real_valued": "tms", "config": config, "seed": seed,
        "task": f"TMS {n}-{h}{' with a frozen hidden identity' if identity else ''}: y = ReLU(W^T W x + b), features present with probability {p}, uniform on [0, 1]",
        "spd_published": "SPD Table 1: MMCS 1.000, ML2R " + ("1.031 (TMS 40-10+ID)" if identity else "1.010 (TMS 40-10)"),
    }
    finish(out, files, record, mechanisms, np.stack(active, 1))


# ---------------------------------------------------------------------------- resid MLP (CC)


def resid_mlp(out: Path, layers: int, seed: int):
    n, d, m, p = 100, 1000, 50, 0.01
    steps = {1: 2000, 2: 3000, 3: 4000}[layers]
    # the embedding is the same for every seed (seed 0), so seeds share the stream's coordinates
    E = torch.randn(n, d, generator=torch.Generator().manual_seed(0))
    E = E / E.norm(dim=1, keepdim=True)  # fixed random unit-norm embedding, W_U = E^T
    gen = torch.Generator().manual_seed(1000 + seed)
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
    train = sparse_features(torch.Generator().manual_seed(2), TRAIN_ROWS, n, p, -1.0, 1.0).double()
    with torch.no_grad():
        sigma = math.sqrt(((forward(xs, Wi, Wo) - (xs + torch.relu(xs))) ** 2).mean().item())
    En = E.numpy()
    files = {}
    stream = torch.cat([xs, train]).numpy() @ En
    config = real_export(out, files, stream, [(Wi[l].numpy(), Wo[l].numpy()) for l in range(layers)], "relu", En, None, False, sigma)
    mechanisms, active = resid_mlp_truth(En, [w.numpy() for w in Wi], [w.numpy() for w in Wo], xs.numpy())
    record = {
        "model": out.name, "kind": "language_model", "real_valued": "resid_mlp", "config": config, "seed": seed,
        "task": f"compressed computation: {n} functions x + ReLU(x) through {layers} MLP layer(s) of {m} in a {d}-wide stream, features present with probability {p}, uniform on [-1, 1]",
        "spd_published": {
            1: "SPD 3.3: 100 rank-one subcomponents in W_in, one per function; W_out one rank-50 component",
            2: "SPD 3.4: 100 W_in components, each across both layers; W_out one rank-50 component across both layers",
            3: "SPD 3.4: 102 W_in components across 3 layers; W_out one rank-51 component across all three",
        }[layers],
    }
    finish(out, files, record, mechanisms, active)


def resid_mlp_truth(E, Wi, Wo, xs):
    """SPD's account (3.3, 3.4): function i is every layer's c_fc read of its embedding direction
    e_i (through the embedding's dual frame, the e_i being near orthogonal, not exactly), one
    rank-one slice per layer gated by x_i != 0; every layer's down_proj together is one always-on
    part of full rank (its slices are not per function: they write every function's output and
    its interference, which a per-function write would drop)."""
    n, layers = E.shape[0], len(Wi)
    dual = np.linalg.pinv(E.T)  # (n, d): dual[i] . e_j = delta_ij on the embedding's span
    mechanisms, active = [], []
    for i in range(n):
        pieces = {f"blocks.{l}.mlp.c_fc": np.outer(Wi[l] @ E[i], dual[i]) for l in range(layers)}
        mechanisms.append({"name": f"function {i}", "rank": 1, "operators": pieces, "gate": f"x[{i}] != 0"})
        active.append(xs[:, i] != 0)
    mechanisms.append({"name": "W_out", "rank": sum(w.shape[1] for w in Wo), "operators": {f"blocks.{l}.mlp.down_proj": Wo[l] for l in range(layers)}, "gate": "always"})
    active.append(np.ones(xs.shape[0], dtype=bool))
    return mechanisms, np.stack(active, 1)


# ------------------------------------------------------------------------------- transformers


def rms(x, g, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g


def rotary(x, theta):
    # x: (..., t, hd), rotate_half pairing (dims i and i + hd/2)
    t, hd = x.shape[-2], x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=x.dtype, device=x.device) / hd))
    ang = torch.arange(t, dtype=x.dtype, device=x.device)[:, None] * inv[None, :]
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
        mask = torch.ones(T, T, dtype=torch.bool, device=tokens.device).tril()
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
        self.cpu()
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


def modadd(out: Path, seed: int):
    p = 113
    # the train/test split is the same for every seed; the initialization is the seed's
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(seed)
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    tokens = torch.stack([a, b, torch.full_like(a, p)], 1)
    labels = (a + b) % p
    perm = torch.randperm(p * p, generator=gen)
    train, test = perm[: int(0.3 * p * p)], perm[int(0.3 * p * p):]
    model = Lm(vocab=p + 1, d=128, layers=1, heads=4, hd=32, mlp=512).to(DEVICE)
    # The norm gains stay at 1: a trained gain rescales the stream per coordinate, which weight
    # decay on the maps around it then trades against, and the clocks spread over many frequencies.
    for gain in [model.final, model.g1[0], model.g2[0]]:
        gain.requires_grad_(False)
    tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    # past generalization (about step 5000) weight decay keeps cleaning the embedding up to a few
    # key frequencies (Nanda et al. 2023)
    for step in range(25000):
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
    files = {}
    tokens, labels = tokens.cpu(), labels.cpu()
    config = model.export(out, files)
    # rows: the test pairs (never trained) shuffled, then the training pairs; the first HELD_OUT
    # are the held-out rows every score reads
    rows = torch.cat([tokens[test][torch.randperm(len(test), generator=gen)], tokens[train]])
    write(out, files, "tokens", rows.double().numpy())
    modadd_truth(out, model, tokens, files, config)


def modadd_truth(out: Path, model, tokens, files, config):
    """The Fourier clocks. Each MLP neuron's activation at the readout position over the full grid
    of (a, b) is assigned to the frequency k carrying most of its power (the 2-D Fourier terms in a,
    b and a + b at +-k); the key frequencies are the fewest carrying 90% of all neurons' assigned
    power (a reporting cut for the truth only). A clock is its neurons' c_fc rows and down_proj
    columns with the embedding's and readout's cos/sin(2 pi k a / p) plane."""
    p = 113
    t = np.arange(p)
    # MLP neurons by frequency
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
    power = np.array([neuron_power[best == k, k].sum() for k in range(nfreq)])
    order = np.argsort(-power)
    cum = np.cumsum(power[order]) / power.sum()
    keys = [int(order[i]) for i in range(int(np.searchsorted(cum, 0.9)) + 1)]
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
    active = np.ones((HELD_OUT * 3, len(mechanisms)), dtype=bool)
    record = {"model": "modadd_113", "kind": "language_model", "config": config,
              "task": "tokens a, b, = (113); the prediction at position 2 is (a + b) mod 113; positions 0, 1 untrained",
              "key_frequencies": keys, "neuron_power_share": {int(k): float(power[k] / power.sum()) for k in order[:12]},
              "spd_published": "none for this toy (Nanda et al. 2023: 5 key frequencies at p = 113)"}
    finish(out, files, record, mechanisms, active)


def induction(out: Path, seed: int):
    vocab, T = 32, 32
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

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

    model = Lm(vocab=vocab, d=64, layers=2, heads=4, hd=16, mlp=1, mlp_on=False).to(DEVICE)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=2e-3, weight_decay=0.01)
    for step in range(6000):
        x = batch(256, gen).to(DEVICE)
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
    induction_truth(out, model, rows, files, config)


def load_lm(out: Path, mlp_on=True):
    """The Lm of a transformer toy's export."""
    record = json.loads((out / "export.json").read_text())
    c = record["config"]
    W = {k: torch.tensor(np.fromfile(out / f"{k}.f64", dtype="<f8").reshape(v["shape"]), dtype=torch.float32) for k, v in record["files"].items()}
    model = Lm(vocab=c["vocab"], d=c["d_model"], layers=c["n_layers"], heads=c["n_heads"], hd=c["head_dim"], mlp=c["d_mlp"],
               theta=int(c["rope_theta"]), eps=c["norm_eps"], mlp_on=mlp_on)
    with torch.no_grad():
        model.wte.copy_(W["wte"])
        model.lm_head.copy_(W["lm_head"])
        model.final.copy_(W["final_norm.gain"][0])
        for l in range(c["n_layers"]):
            for t, name in [(model.q, "attn.q_proj"), (model.k, "attn.k_proj"), (model.v, "attn.v_proj"), (model.o, "attn.o_proj"), (model.up, "mlp.c_fc"), (model.down, "mlp.down_proj")]:
                t[l].copy_(W[f"blocks.{l}.{name}"])
            model.g1[l].copy_(W[f"blocks.{l}.rms1.gain"][0])
            model.g2[l].copy_(W[f"blocks.{l}.rms2.gain"][0])
    return model, record, W["tokens"].long()


def induction_truth(out: Path, model, rows, files, config):
    """The known mechanism, measured on the held-out rows: layer 0's previous-token heads (mean
    attention from t to t-1 or to t-2 above 1/2) and layer 1's induction heads (mean attention
    from a token with an earlier occurrence at s to s+1 or s+2 above 1/2: the head's key is the
    token a previous-token head copied there, its value the token after); 1/2 is a reporting cut
    for the truth, never a method setting. Each head is one mechanism of its head's rank."""
    # rows: the held-out sequences, then training sequences of the same law for a fit
    T = rows.shape[1]
    train = torch.randint(1, model.cfg["vocab"], (HELD_OUT, T), generator=torch.Generator().manual_seed(2))
    train[:, 0] = 0
    g = torch.Generator().manual_seed(3)
    for i in range(HELD_OUT):
        s, l = int(torch.randint(1, 4, (1,), generator=g)), int(torch.randint(6, 13, (1,), generator=g))
        train[i, s + l: s + 2 * l] = train[i, s: s + l]
    write(out, files, "tokens", torch.cat([rows, train]).double().numpy())
    pats = []
    with torch.no_grad():
        model(rows, pats)
    back = {o: pats[0][:, :, torch.arange(o, T), torch.arange(0, T - o)].mean((0, 2)).numpy() for o in (1, 2)}
    targets = []
    for i in range(rows.shape[0]):
        seen = {}
        for t_ in range(T):
            tok = int(rows[i, t_])
            if tok in seen and seen[tok] + 2 < t_:
                targets.append((i, t_, seen[tok]))
            seen[tok] = t_
    idx = torch.tensor(targets)
    ind = sum(pats[1][idx[:, 0], :, idx[:, 1], idx[:, 2] + o] for o in (1, 2)).mean(0).numpy()
    print(f"1-back {back[1].round(3)}, 2-back {back[2].round(3)}, induction {ind.round(3)}", flush=True)
    heads = model.cfg["heads"]
    act_rows = np.zeros(rows.shape[0] * T, dtype=bool)
    for (i, t_, _) in targets:
        act_rows[i * T + t_] = True
    mechanisms, active = [], []
    for h in range(heads):
        o = 1 if back[1][h] >= back[2][h] else 2
        if back[o][h] > 0.5:
            mechanisms.append({"name": f"{o}-back previous-token head L0H{h}", "rank": model.cfg["hd"], "operators": head_pieces(model, 0, h), "gate": "always", "score": float(back[o][h])})
            active.append(np.ones(rows.shape[0] * T, dtype=bool))
    for h in range(heads):
        if ind[h] > 0.5:
            mechanisms.append({"name": f"induction head L1H{h}", "rank": model.cfg["hd"], "operators": head_pieces(model, 1, h), "gate": "the token occurred before", "score": float(ind[h])})
            active.append(act_rows)
    record = {"model": "induction", "kind": "language_model", "config": config,
              "task": "BOS, random tokens with a 6..12-token segment repeated right after itself; next-token prediction at every position",
              "back_scores": {str(o): back[o].tolist() for o in (1, 2)}, "induction_scores": ind.tolist(),
              "note": "attention-only: the MLP weights are zero (the importer needs an MLP)",
              "spd_published": "none (VPD/SPD do not decompose this toy)"}
    finish(out, files, record, mechanisms, np.stack(active, 1))


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


def effects(out: Path):
    """Each known mechanism's measured effect per held-out token: KL(M || M without it), in bits,
    M without it being M's weights less the mechanism's deltas on every operator it spans (for a
    real-valued toy, of the Gaussian predictive distributions in units of the task residual).
    Written as truth_effect.f64 (tokens x mechanisms), beside truth_active.f64: a gate is graded
    by the share of a mechanism's effect it covers, with no line drawn between active and not."""
    record = json.loads((out / "export.json").read_text())
    truth = json.loads((out / "truth.json").read_text())
    rows = truth["active"]["shape"][0]
    deltas = [{op: np.fromfile(out / e["file"], dtype="<f8").reshape(e["shape"]) for op, e in m["operators"].items()} for m in truth["mechanisms"]]
    W = {k: np.fromfile(out / f"{k}.f64", dtype="<f8").reshape(v["shape"]) for k, v in record["files"].items() if k != "tokens"}
    if "real_valued" in record:
        c = record["config"]
        stream = W["wte"][:rows]

        def outputs(weights):
            r = stream
            for l in range(c["n_layers"]):
                if any(np.any(weights[f"blocks.{l}.attn.{k}_proj"]) for k in "qkvo"):
                    raise SystemExit("a real-valued toy's attention is zero")
                pre = r @ weights[f"blocks.{l}.mlp.c_fc"].T
                r = r + (np.maximum(pre, 0.0) if c["mlp_act"] == "relu" else pre) @ weights[f"blocks.{l}.mlp.down_proj"].T
            y = r @ weights["gaussian_head"].T + (weights["gaussian_head.bias"][0] if "gaussian_head.bias" in weights else 0.0)
            return np.maximum(y, 0.0) if c["head"]["relu"] else y

        base = outputs(W)
        effect = np.stack([((base - outputs({k: v - d.get(k, 0.0) for k, v in W.items()})) ** 2).sum(1) / (2 * math.log(2)) for d in deltas], 1)
    else:
        mlp_on = any(np.any(W[f"blocks.{l}.mlp.c_fc"]) for l in range(record["config"]["n_layers"]))
        model, _, tokens = load_lm(out, mlp_on=mlp_on)
        T = tokens.shape[1]
        held = tokens[: rows // T]
        names = {"wte": lambda m: m.wte, "lm_head": lambda m: m.lm_head}
        for l in range(record["config"]["n_layers"]):
            for name, get in [("attn.q_proj", lambda m, l=l: m.q[l]), ("attn.k_proj", lambda m, l=l: m.k[l]), ("attn.v_proj", lambda m, l=l: m.v[l]),
                              ("attn.o_proj", lambda m, l=l: m.o[l]), ("mlp.c_fc", lambda m, l=l: m.up[l]), ("mlp.down_proj", lambda m, l=l: m.down[l])]:
                names[f"blocks.{l}.{name}"] = get
        with torch.no_grad():
            base = torch.log_softmax(model(held).double(), -1)
            effect = []
            for d in deltas:
                saved = {op: names[op](model).detach().clone() for op in d}
                for op, delta in d.items():
                    names[op](model).sub_(torch.tensor(delta, dtype=names[op](model).dtype))
                lp = torch.log_softmax(model(held).double(), -1)
                effect.append(((base.exp() * (base - lp)).sum(-1).reshape(-1) / math.log(2)).numpy())
                for op, value in saved.items():
                    names[op](model).copy_(value)
            effect = np.stack(effect, 1)
    effect.astype("<f8").tofile(out / "truth_effect.f64")
    truth["effect"] = {"file": "truth_effect.f64", "shape": list(effect.shape)}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{out}: mean effect per mechanism {np.round(effect.mean(0), 4).tolist()[:12]}", flush=True)


if __name__ == "__main__":
    # TOY or TOY@sSEED (a further seed of the toy, written to TOY_sSEED)
    toy, root = sys.argv[1], Path(sys.argv[2])
    if toy.startswith("effects:"):
        effects(root / toy[8:])
        sys.exit(0)
    if toy == "truth:modadd_113":
        # a trained export whose truth was not written: its record, rows and truth again (the
        # split is the training's, from the same seed)
        out = root / "modadd_113"
        p = 113
        a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
        grid = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], 1)
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(p * p, generator=gen)
        train, test = perm[: int(0.3 * p * p)], perm[int(0.3 * p * p):]
        model = Lm(vocab=p + 1, d=128, layers=1, heads=4, hd=32, mlp=512)
        files = {}
        tmp = {k: np.fromfile(out / f"{k}.f64", dtype="<f8") for k in ["wte", "lm_head", "final_norm.gain"] + [f"blocks.0.{n}" for n in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj", "rms1.gain", "rms2.gain"]]}
        with torch.no_grad():
            for t, k in [(model.wte, "wte"), (model.lm_head, "lm_head"), (model.final, "final_norm.gain"), (model.q[0], "blocks.0.attn.q_proj"), (model.k[0], "blocks.0.attn.k_proj"),
                         (model.v[0], "blocks.0.attn.v_proj"), (model.o[0], "blocks.0.attn.o_proj"), (model.up[0], "blocks.0.mlp.c_fc"), (model.down[0], "blocks.0.mlp.down_proj"),
                         (model.g1[0], "blocks.0.rms1.gain"), (model.g2[0], "blocks.0.rms2.gain")]:
                t.copy_(torch.tensor(tmp[k].reshape(t.shape)))
        config = model.export(out, files)
        rows = torch.cat([grid[test][torch.randperm(len(test), generator=gen)], grid[train]])
        write(out, files, "tokens", rows.double().numpy())
        modadd_truth(out, model, grid, files, config)
        sys.exit(0)
    if toy == "truth:induction":
        model, record, rows = load_lm(root / "induction", mlp_on=False)
        rows = rows[: HELD_OUT // 8]
        induction_truth(root / "induction", model, rows, record["files"], record["config"])
        sys.exit(0)
    name, _, seed = toy.partition("@s")
    seed = int(seed or 0)
    out = root / (name if seed == 0 else f"{name}_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    {
        "tms_40_10": lambda: tms(out, False, seed),
        "tms_40_10_id": lambda: tms(out, True, seed),
        "resid_mlp_1l": lambda: resid_mlp(out, 1, seed),
        "resid_mlp_2l": lambda: resid_mlp(out, 2, seed),
        "resid_mlp_3l": lambda: resid_mlp(out, 3, seed),
        "modadd_113": lambda: modadd(out, seed),
        "induction": lambda: induction(out, seed),
    }[name]()
