"""#2951 mpd-modadd: a small transformer trained on modular addition, the first trained
end-to-end benchmark for manifold parameter decomposition (plan: #2951 comment 5716688586).

* ``train`` fits the one-layer grokking transformer on ``(a + b) mod p``, or on iid random
  labels over the same split (control C2). Config, split, curves and checkpoints go into ONE
  file; the step-0 checkpoint is control C1, the random init of the same run.
* ``check`` writes one JSON card for a stored run: train and held-out accuracy, the Fourier
  power spectrum of W_E (top-k share and frequencies), the parameter count and the file's
  sha256. It is the grokking gate for the tiny development ladder (small p, d_model 32).

``bench/mpd_engine_export_2951.py`` exports a stored run for the Rust decomposition engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import torch

class ModAddTransformer(torch.nn.Module):
    def __init__(self, p, d_model, n_heads, d_head, d_mlp):
        super().__init__()
        scale = d_model ** -0.5
        self.d_head = d_head
        self.W_E = torch.nn.Parameter(torch.randn(p + 1, d_model) * scale)
        self.W_pos = torch.nn.Parameter(torch.randn(3, d_model) * scale)
        self.W_Q = torch.nn.Parameter(torch.randn(n_heads, d_head, d_model) * scale)
        self.W_K = torch.nn.Parameter(torch.randn(n_heads, d_head, d_model) * scale)
        self.W_V = torch.nn.Parameter(torch.randn(n_heads, d_head, d_model) * scale)
        self.W_O = torch.nn.Parameter(torch.randn(d_model, n_heads * d_head) * scale)
        self.W_in = torch.nn.Parameter(torch.randn(d_mlp, d_model) * scale)
        self.b_in = torch.nn.Parameter(torch.zeros(d_mlp))
        self.W_out = torch.nn.Parameter(torch.randn(d_model, d_mlp) * scale)
        self.b_out = torch.nn.Parameter(torch.zeros(d_model))
        self.W_U = torch.nn.Parameter(torch.randn(p, d_model) * scale)
        self.register_buffer("causal", torch.tril(torch.ones(3, 3, dtype=torch.bool)))

    def forward(self, tokens, embed_at=None):
        """Logits at ``=``. ``embed_at`` maps a use site to the W_E table that occurrence reads."""
        edits = embed_at or {}
        x = torch.stack([edits.get(u, self.W_E)[tokens[:, u]] for u in range(3)], dim=1) + self.W_pos
        q = torch.einsum("hkd,npd->nhpk", self.W_Q, x)
        k = torch.einsum("hkd,npd->nhpk", self.W_K, x)
        v = torch.einsum("hkd,npd->nhpk", self.W_V, x)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        pattern = torch.softmax(scores.masked_fill(~self.causal, float("-inf")), dim=-1)
        x = x + (pattern @ v).transpose(1, 2).reshape(x.shape[0], 3, -1) @ self.W_O.T
        x = x + torch.relu(x @ self.W_in.T + self.b_in) @ self.W_out.T + self.b_out
        return x[:, -1] @ self.W_U.T


def all_pairs(p):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    return torch.stack([a.reshape(-1), b.reshape(-1), torch.full((p * p,), p)], dim=1)


def build_model(config):
    return ModAddTransformer(
        config["p"], config["d_model"], config["n_heads"], config["d_head"], config["d_mlp"]
    )


def log_softmax64(logits):
    return torch.log_softmax(logits.double(), dim=-1)


def train(args):
    config = {key: getattr(args, key) for key in (
        "p", "train_fraction", "d_model", "n_heads", "d_head", "d_mlp", "lr", "weight_decay",
        "steps", "eval_every", "checkpoint_steps", "labels", "seed")}
    device = torch.device(args.device)
    p = args.p
    tokens = all_pairs(p)
    split = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(p * p, generator=split)
    n_train = round(args.train_fraction * p * p)
    train_idx, test_idx = order[:n_train], order[n_train:]
    if args.labels == "addition":
        labels = (tokens[:, 0] + tokens[:, 1]) % p
    else:
        labels = torch.randint(p, (p * p,), generator=split)
    torch.manual_seed(args.seed)
    model = build_model(config).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    tokens_d, labels_d = tokens.to(device), labels.to(device)
    train_d, test_d = train_idx.to(device), test_idx.to(device)
    checkpoint_steps = set(args.checkpoint_steps)
    curves = {"step": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    checkpoints = {}
    for step in range(args.steps + 1):
        if step in checkpoint_steps:
            checkpoints[step] = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}
        if step % args.eval_every == 0 or step == args.steps:
            with torch.inference_mode():
                logp = log_softmax64(model(tokens_d))
                nll = -logp.gather(1, labels_d[:, None])[:, 0]
                hit = (logp.argmax(-1) == labels_d).double()
            curves["step"].append(step)
            for name, idx in (("train", train_d), ("test", test_d)):
                curves[f"{name}_loss"].append(nll[idx].mean().item())
                curves[f"{name}_acc"].append(hit[idx].mean().item())
            print(
                f"TRAIN labels={args.labels} step={step} train_loss={curves['train_loss'][-1]:.6g} "
                f"test_loss={curves['test_loss'][-1]:.6g} train_acc={curves['train_acc'][-1]:.4f} "
                f"test_acc={curves['test_acc'][-1]:.4f}",
                flush=True,
            )
        if step == args.steps:
            break
        loss = -log_softmax64(model(tokens_d[train_d])).gather(1, labels_d[train_d][:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    run = {
        "config": config,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "labels": labels,
        "curves": curves,
        "checkpoints": checkpoints,
    }
    torch.save(run, args.out + ".partial")
    os.replace(args.out + ".partial", args.out)
    print(f"SAVED {args.out} checkpoints={sorted(checkpoints)}", flush=True)


def fourier_planes(table, p):
    """Cosine and sine coefficient vectors of ``table[a]`` over a in Z_p, frequencies 1..(p-1)/2."""
    a = torch.arange(p, dtype=torch.float64)
    freqs = torch.arange(1, (p - 1) // 2 + 1, dtype=torch.float64)
    phase = 2 * math.pi * freqs[:, None] * a[None, :] / p
    cos_coef = (2 / p) * torch.cos(phase) @ table
    sin_coef = (2 / p) * torch.sin(phase) @ table
    dc = table.mean(0)
    power = (cos_coef ** 2).sum(-1) + (sin_coef ** 2).sum(-1)
    # Parseval over odd p: sum_a |table[a]|^2 = p |dc|^2 + (p / 2) sum_k power_k.
    total = power.sum() + 2 * (dc ** 2).sum()
    return cos_coef, sin_coef, power, total


def check(args):
    """The run card: accuracies, the W_E spectrum and the identity of the stored file."""
    with open(args.run, "rb") as handle:
        sha256 = hashlib.sha256(handle.read()).hexdigest()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    step = max(run["checkpoints"])
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][step])
    model = model.eval()
    with torch.inference_mode():
        hit = model(all_pairs(p)).argmax(-1) == run["labels"]
    train_acc = hit[run["train_idx"]].double().mean().item()
    test_acc = hit[run["test_idx"]].double().mean().item()
    spectra = {}
    for name, table in (("W_E", model.W_E.detach()[:p]), ("W_U", model.W_U.detach())):
        _, _, power, total = fourier_planes(table.double(), p)
        share = power / total
        order = torch.argsort(share, descending=True)
        spectra[name] = {
            "frequencies": [int(k) + 1 for k in order],
            "share": [share[k].item() for k in order],
            "top_k_share": {str(k): share[order[:k]].sum().item() for k in range(1, args.kmax + 1)},
            "dc_share": 1 - share.sum().item(),
        }
    top = spectra["W_E"]["top_k_share"][str(args.concentration_k)]
    curves = run["curves"]
    reached = [step for step, acc in zip(curves["step"], curves["test_acc"]) if acc >= args.min_test_acc]
    card = {
        "run": os.path.abspath(args.run),
        "sha256": sha256,
        "config": config,
        "step": step,
        "parameter_count": sum(t.numel() for t in model.parameters()),
        "train_pairs": run["train_idx"].numel(),
        "test_pairs": run["test_idx"].numel(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "first_eval_step_at_min_test_acc": reached[0] if reached else None,
        "spectrum": spectra,
        "grokked": test_acc >= args.min_test_acc and top >= args.min_concentration,
        "grokked_rule": f"test_acc >= {args.min_test_acc} and the top {args.concentration_k} W_E "
                        f"frequencies carry >= {args.min_concentration} of W_E's squared norm",
    }
    with open(args.out + ".partial", "w") as handle:
        json.dump(card, handle, indent=1)
    os.replace(args.out + ".partial", args.out)
    print(f"CHECK {args.run} train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
          f"W_E top frequencies {spectra['W_E']['frequencies'][:args.kmax]} "
          f"top-{args.concentration_k} share={top:.4f} grokked={card['grokked']}", flush=True)


def int_list(text):
    return [int(v) for v in text.split(",") if v]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--p", type=int, default=113)
    fit.add_argument("--train-fraction", type=float, default=0.3)
    fit.add_argument("--d-model", type=int, default=128)
    fit.add_argument("--n-heads", type=int, default=4)
    fit.add_argument("--d-head", type=int, default=32)
    fit.add_argument("--d-mlp", type=int, default=512)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=1.0)
    fit.add_argument("--steps", type=int, required=True)
    fit.add_argument("--eval-every", type=int, default=100)
    fit.add_argument("--checkpoint-steps", type=int_list, required=True)
    fit.add_argument("--labels", choices=("addition", "random"), required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--device", required=True)
    fit.add_argument("--out", required=True)
    card = commands.add_parser("check")
    card.add_argument("--run", required=True)
    card.add_argument("--kmax", type=int, default=8)
    card.add_argument("--min-test-acc", type=float, default=0.999)
    card.add_argument("--concentration-k", type=int, required=True)
    card.add_argument("--min-concentration", type=float, required=True)
    card.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
