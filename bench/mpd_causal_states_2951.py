"""#2951 causal states: planted sequence behaviours with known automata, for causal-state extraction.

* ``train`` fits a small pre-RMSNorm rotary transformer (the LlamaSimpleMLP layout of the VPD
  4L Pile target: RMSNorm with gain, rotate-half RoPE multi-head attention, GELU(tanh) MLP, no
  biases, final RMSNorm, untied unembedding) on next-token prediction over sequences drawn from
  one planted process. The processes are the known-answer behaviours of
  ``crates/gam-sae/src/parameter_decomposition/causal_states_tests.rs``:

  - ``dyck1``: ``(`` = 1, ``)`` = 2; at depth 0 a close has probability 0.03, otherwise 0.5.
    The true machine is one unbounded counter.
  - ``mod3``: ``a`` = 1, ``b`` = 2; the law is a function of the count of ``a`` modulo 3.
  - ``dyck2``: ``(`` = 1, ``)`` = 2, ``[`` = 3, ``]`` = 4 with a stack; the matching close is
    likely (0.48), the other close is rare (0.02). The true machine is a counter plus the
    top of the stack.

  Every sequence starts with BOS = 0. Data are fresh draws each step.
* ``harvest`` draws held-out sequences and writes the Rust driver's inputs:
  ``tokens.npy`` (int64, N x T), ``probs.npy`` (float32, N x T x V, the model's next-token
  distribution after each prefix), ``truth.npy`` (float32, N x T x V, the process law),
  ``resid.npy`` (float32, (L+1) x N x T x d: the residual stream entering each block and the
  final one), every weight as float32 ``.npy`` and ``harvest.json``. For the first
  ``--branch`` sequences it also writes every one-token continuation of every prefix:
  ``branch_resid.npy`` (float32, (L+1) x B x T x V x d, the residual at position t + 1 when
  token a follows the prefix ending at t) and ``branch_probs.npy`` (B x T x V x V), and the
  additive writes of those sequences: ``head_writes.{i}.npy`` (B x T x heads x d, each head's
  attention output through its slice of ``o``) and ``mlp_act.{i}.npy`` (B x T x mlp).

Python here only trains the planted model and harvests it; every analysis runs in Rust
(``crates/gam-sae/examples/mpd_causal_states_2951.rs``).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TASKS = {"dyck1": 3, "mod3": 3, "dyck2": 5}


def law(task: str, state) -> list[float]:
    """The next-token distribution over tokens 1..V-1 at a generator state."""
    if task == "dyck1":
        return [0.97, 0.03] if state == 0 else [0.5, 0.5]
    if task == "mod3":
        return [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]][state % 3]
    if not state:
        return [0.485, 0.015, 0.485, 0.015]
    return [0.25, 0.48, 0.25, 0.02] if state[-1] == 1 else [0.25, 0.02, 0.25, 0.48]


def advance(task: str, state, token: int):
    if task == "dyck1":
        return state + 1 if token == 1 else (max(state - 1, 0) if token == 2 else state)
    if task == "mod3":
        return state + (token == 1)
    if token in (1, 3):
        return state + (token,)
    if token in (2, 4):
        return state[:-1]
    return state


def draw(task: str, sequences: int, length: int, rng: np.random.Generator):
    """Token sequences BOS x_1..x_length and the process law after each prefix."""
    vocab = TASKS[task]
    tokens = np.zeros((sequences, length + 1), dtype=np.int64)
    truth = np.zeros((sequences, length + 1, vocab), dtype=np.float32)
    for row in range(sequences):
        state = () if task == "dyck2" else 0
        for position in range(length + 1):
            token = int(tokens[row, position])
            state = advance(task, state, token)
            probabilities = law(task, state)
            truth[row, position, 1:] = probabilities
            if position < length:
                tokens[row, position + 1] = 1 + rng.choice(len(probabilities), p=probabilities)
    return tokens, truth


def rms(x: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return gain * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Planted(torch.nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, heads: int, mlp: int, n_ctx: int, base: float = 10000.0):
        super().__init__()
        self.layers, self.heads, self.hd = layers, heads, d // heads
        scale = d ** -0.5
        p = lambda *shape, s=scale: torch.nn.Parameter(torch.randn(*shape) * s)
        self.wte = p(vocab, d, s=1.0)
        self.lm_head = p(vocab, d)
        self.ln_f = torch.nn.Parameter(torch.ones(d))
        self.rms_1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(d)) for _ in range(layers)])
        self.rms_2 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(d)) for _ in range(layers)])
        self.q = torch.nn.ParameterList([p(d, d) for _ in range(layers)])
        self.k = torch.nn.ParameterList([p(d, d) for _ in range(layers)])
        self.v = torch.nn.ParameterList([p(d, d) for _ in range(layers)])
        self.o = torch.nn.ParameterList([p(d, d) for _ in range(layers)])
        self.fc = torch.nn.ParameterList([p(mlp, d) for _ in range(layers)])
        self.down = torch.nn.ParameterList([p(d, mlp, s=mlp ** -0.5) for _ in range(layers)])
        n = self.hd // 2
        freq = base ** (torch.arange(n, dtype=torch.float32) / n)
        angle = torch.arange(n_ctx, dtype=torch.float32)[:, None] / freq.repeat(2)[None, :]
        self.register_buffer("cos", angle.cos())
        self.register_buffer("sin", angle.sin())

    def rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        n = self.hd // 2
        return x * self.cos[:T] + torch.cat([-x[..., n:], x[..., :n]], -1) * self.sin[:T]

    def forward(self, ids: torch.Tensor, keep: bool = False, record: dict | None = None,
                patch: dict | None = None):
        """``record`` collects per-head attention writes ``head_writes.{i}`` (B x T x heads x d)
        and MLP activations ``mlp_act.{i}`` (B x T x mlp). ``patch`` maps a layer ``l`` to
        ``(mask, value)``: the residual entering block ``l`` (``l = L`` the final one) is
        replaced by ``value`` where the B x T ``mask`` holds."""
        B, T = ids.shape
        x = self.wte[ids]

        def patched(x, layer):
            if patch is not None and layer in patch:
                mask, value = patch[layer]
                return torch.where(mask[..., None], value, x)
            return x

        x = patched(x, 0)
        stream = [x] if keep else None
        for i in range(self.layers):
            h = rms(x, self.rms_1[i])
            split = lambda w: (h @ w.T).view(B, T, self.heads, self.hd).transpose(1, 2)
            q, k, v = split(self.q[i]), split(self.k[i]), split(self.v[i])
            q, k = self.rope(q, T), self.rope(k, T)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            if record is not None:
                o = self.o[i].view(-1, self.heads, self.hd)
                record[f"head_writes.{i}"] = torch.einsum("bhtc,dhc->bthd", y, o)
            x = x + y.transpose(1, 2).reshape(B, T, -1) @ self.o[i].T
            act = gelu_tanh(rms(x, self.rms_2[i]) @ self.fc[i].T)
            if record is not None:
                record[f"mlp_act.{i}"] = act
            x = x + act @ self.down[i].T
            x = patched(x, i + 1)
            if keep:
                stream.append(x)
        logits = rms(x, self.ln_f) @ self.lm_head.T
        return (logits, stream) if keep else logits


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vocab = TASKS[args.task]
    model = Planted(vocab, args.d, args.layers, args.heads, args.mlp, args.length + 1).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps)
    started = time.time()
    for step in range(args.steps):
        tokens, truth = draw(args.task, args.batch, args.length, rng)
        ids = torch.from_numpy(tokens).to(args.device)
        target = torch.from_numpy(truth).to(args.device)
        logits = model(ids)
        # Cross-entropy against the process law: the same minimizer as sampled next tokens,
        # with less variance.
        loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 200 == 0 or step == args.steps - 1:
            entropy = -(target * torch.log(target.clamp_min(1e-30))).sum(-1).mean()
            print(f"step {step} excess nats {float(loss - entropy):.5f} ({time.time() - started:.0f}s)", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ("task", "d", "layers", "heads", "mlp", "length", "seed")}
    config["vocab"] = vocab
    torch.save({"config": config, "state": model.state_dict()}, out / "model.pt")


def load(run: Path, device: str) -> tuple[Planted, dict]:
    blob = torch.load(run / "model.pt", map_location="cpu")
    c = blob["config"]
    model = Planted(c["vocab"], c["d"], c["layers"], c["heads"], c["mlp"], 4 * (c["length"] + 1))
    state = {k: v for k, v in blob["state"].items() if k not in ("cos", "sin")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), c


@torch.no_grad()
def harvest(args):
    run = Path(args.run)
    model, config = load(run, args.device)
    rng = np.random.default_rng(args.seed)
    length = args.length or config["length"]
    tokens, truth = draw(config["task"], args.sequences, length, rng)
    probs, resid = [], []
    for start in range(0, args.sequences, 256):
        ids = torch.from_numpy(tokens[start:start + 256]).to(args.device)
        logits, stream = model(ids, keep=True)
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        resid.append(torch.stack(stream).float().cpu().numpy())
    # Every one-token continuation of every prefix of the first sequences: the state and the
    # readout at position t + 1 when token a follows the prefix ending at t.
    branched = min(args.branch, args.sequences)
    vocab = config["vocab"]
    branch_resid = np.zeros((config["layers"] + 1, branched, length + 1, vocab, config["d"]), dtype=np.float32)
    branch_probs = np.zeros((branched, length + 1, vocab, vocab), dtype=np.float32)
    for t in range(length + 1):
        prefix = torch.from_numpy(tokens[:branched, :t + 1]).to(args.device)
        for a in range(vocab):
            ids = torch.cat([prefix, torch.full((branched, 1), a, device=args.device)], 1)
            logits, stream = model(ids, keep=True)
            branch_probs[:, t, a] = torch.softmax(logits[:, -1].float(), -1).cpu().numpy()
            branch_resid[:, :, t, a] = torch.stack([s[:, -1] for s in stream]).float().cpu().numpy()
    # The additive writes into the residual stream of the first sequences, for attributing
    # state distinctions to heads and MLP units.
    record: dict = {}
    model(torch.from_numpy(tokens[:branched]).to(args.device), record=record)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in record.items():
        np.save(out / f"{name}.npy", tensor.float().cpu().numpy())
    np.save(out / "branch_resid.npy", branch_resid)
    np.save(out / "branch_probs.npy", branch_probs)
    np.save(out / "tokens.npy", tokens)
    np.save(out / "truth.npy", truth)
    np.save(out / "probs.npy", np.concatenate(probs, 0).astype(np.float32))
    np.save(out / "resid.npy", np.concatenate(resid, 1).astype(np.float32))
    for name, tensor in model.state_dict().items():
        if name not in ("cos", "sin"):
            np.save(out / f"w_{name}.npy", tensor.float().cpu().numpy())
    excess = float(np.mean(np.sum(truth[..., 1:] * (np.log(np.maximum(truth[..., 1:], 1e-30))
                                                      - np.log(np.concatenate(probs, 0)[..., 1:])), -1)))
    meta = {**config, "sequences": args.sequences, "harvest_length": length, "harvest_seed": args.seed,
            "mean_kl_truth_model_nats": excess, "bos": 0}
    (out / "harvest.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", choices=sorted(TASKS), required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--d", type=int, default=64)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--mlp", type=int, default=256)
    t.add_argument("--length", type=int, default=32)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="mps")
    h = sub.add_parser("harvest")
    h.add_argument("--run", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--sequences", type=int, default=2000)
    h.add_argument("--length", type=int, default=0)
    h.add_argument("--seed", type=int, default=12345)
    h.add_argument("--branch", type=int, default=500)
    h.add_argument("--device", default="mps")
    args = parser.parse_args()
    {"train": train, "harvest": harvest}[args.command](args)


if __name__ == "__main__":
    main()
