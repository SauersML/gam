"""#2951 mpd-modadd: a small transformer trained on modular addition, the first trained
end-to-end benchmark for manifold parameter decomposition (plan: #2951 comment 5716688586).

Stage S0 is torch only.

* ``train`` fits the one-layer grokking transformer on ``(a + b) mod p``, or on iid random
  labels over the same split (control C2). Config, split, curves and checkpoints go into ONE
  file; the step-0 checkpoint is control C1, the random init of the same run.
* ``s0`` runs the executor controls and the benchmark oracle on stored checkpoints and writes
  one JSON receipt.
* ``execute`` is the torch driver of the Schur cross-check receipt
  (``crates/gam-sae/examples/mpd_modadd_schur_2951.rs``): it writes each declared checkpoint's
  W_E and its least-squares shift operator T1 as ``<f8`` arrays, plus ``export.json``.

W_E is one tensor with three use sites (pos0 ``a``, pos1 ``b``, pos2 ``=``). ``forward`` runs
each occurrence as its own lookup, so a use-specific edit and a global edit are different
experiments. The row permutation of W_E by a shift ``s`` is an exact native reference: at pos0
it reproduces ``f(a + s, b)`` bit for bit, because the gathered rows are the same numbers.

The oracle (the DFT planes of W_E and the rotation edit built from them) is benchmark
evaluation under SPEC 8's analysis exception. It is never an MPD input: MPD's math lives in
Rust, and later stages call it through the gamfit surface.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

SITES = {"pos0": (0,), "pos1": (1,), "operands": (0, 1), "global": (0, 1, 2)}


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


def summarize(kl, agree):
    ordered = torch.sort(kl.flatten()).values
    n = ordered.numel()

    def quantile(level):
        return ordered[min(n - 1, math.floor(level * n))].item()

    return {
        "rows": n,
        "kl_mean": ordered.mean().item(),
        "kl_q50": quantile(0.5),
        "kl_q90": quantile(0.9),
        "kl_q99": quantile(0.99),
        "kl_max": ordered[-1].item(),
        "argmax_agreement": agree.double().mean().item(),
    }


def kl_rows(ref_logits, edit_logits):
    lp, lq = log_softmax64(ref_logits), log_softmax64(edit_logits)
    return (lp.exp() * (lp - lq)).sum(-1)


def shift_tokens(tokens, s, sites, p):
    out = tokens.clone()
    for u in sites:
        if u < 2:
            out[:, u] = (out[:, u] + s) % p
    return out


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


def rotation_operator(basis, angles):
    """T = I + U (R - I) U^+ for plane basis U = [u_1c, u_1s, u_2c, ...] and one angle per plane."""
    d, width = basis.shape
    block = torch.zeros(width, width, dtype=torch.float64)
    for j, theta in enumerate(angles):
        c, s = math.cos(theta), math.sin(theta)
        block[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = torch.tensor([[c - 1, -s], [s, c - 1]])
    return torch.eye(d, dtype=torch.float64) + basis @ block @ torch.linalg.pinv(basis)


def run_edits(model, tokens, reference_of, tables_of, shifts, device):
    """Edited logits vs the native reference over rows x shifts, one KL row per (row, shift)."""
    kls, agrees = [], []
    with torch.inference_mode():
        for s in shifts:
            ref = model(reference_of(s))
            edit = model(tokens, embed_at=tables_of(s))
            kls.append(kl_rows(ref, edit).cpu())
            agrees.append((ref.argmax(-1) == edit.argmax(-1)).cpu())
    return summarize(torch.cat(kls), torch.cat(agrees))


def s0_checkpoint(run, step, args):
    config = run["config"]
    p = config["p"]
    device = torch.device(args.device)
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][step])
    model = model.to(device).eval()
    tokens = all_pairs(p).to(device)
    labels = run["labels"].to(device)
    test_idx = run["test_idx"].to(device)
    shifts = args.shifts or list(range(1, p))
    out = {"step": step}

    with torch.inference_mode():
        hit = model(tokens).argmax(-1) == labels
    out["train_acc"] = hit[run["train_idx"].to(device)].double().mean().item()
    out["test_acc"] = hit[test_idx].double().mean().item()
    print(f"S0 step={step} train_acc={out['train_acc']:.4f} test_acc={out['test_acc']:.4f}", flush=True)

    # Executor controls: the row permutation is an exact native reference at every use site.
    W_E = model.W_E.detach()
    executor = {}
    with torch.inference_mode():
        for name, sites in SITES.items():
            mismatched = 0
            for s in shifts:
                perm = torch.arange(p + 1, device=device)
                perm[:p] = (perm[:p] + s) % p
                edit = model(tokens, embed_at={u: W_E[perm] for u in sites})
                ref = model(shift_tokens(tokens, s, sites, p))
                mismatched += int((edit != ref).any(-1).sum().item())
            executor[name] = {"rows": len(shifts) * p * p, "bitwise_mismatched_rows": mismatched}
            print(f"EXEC step={step} site={name} rows={executor[name]['rows']} "
                  f"bitwise_mismatched_rows={mismatched}", flush=True)
        differ = 0
        for s in shifts:
            perm = torch.arange(p + 1, device=device)
            perm[:p] = (perm[:p] + s) % p
            use_specific = model(tokens, embed_at={0: W_E[perm]}).argmax(-1)
            global_edit = model(tokens, embed_at={u: W_E[perm] for u in SITES["global"]}).argmax(-1)
            differ += int((use_specific != global_edit).sum().item())
    executor["pos0_vs_global_argmax_differ_rows"] = differ
    print(f"EXEC step={step} pos0_vs_global_argmax_differ_rows={differ} of {len(shifts) * p * p}", flush=True)
    out["executor"] = executor

    # Oracle spectra of the embedding (over a) and the unembedding (over the answer c).
    table = W_E[:p].double().cpu()
    cos_coef, sin_coef, power, total = fourier_planes(table, p)
    order = torch.argsort(power, descending=True)
    _, _, power_u, total_u = fourier_planes(model.W_U.detach().double().cpu(), p)
    order_u = torch.argsort(power_u, descending=True)
    out["spectrum_W_E"] = [[int(k) + 1, (power[k] / total).item()] for k in order[: args.kmax + 5]]
    out["spectrum_W_U"] = [[int(k) + 1, (power_u[k] / total_u).item()] for k in order_u[: args.kmax + 5]]
    print(f"SPECTRUM step={step} W_E top (k, power share) {out['spectrum_W_E']}", flush=True)
    print(f"SPECTRUM step={step} W_U top (k, power share) {out['spectrum_W_U']}", flush=True)

    # Caveat check: with p <= d_model the rows of W_E are generically independent, so an exact
    # linear shift operator T_1 (E T_1^T = E_shift) exists for ANY model, controls included.
    sigma = torch.linalg.svdvals(table)
    tol = sigma[0].item() * max(table.shape) * torch.finfo(torch.float32).eps
    shifted_table = table[(torch.arange(p) + 1) % p]
    t1 = torch.linalg.lstsq(table, shifted_table).solution
    out["exact_shift_operator"] = {
        "rank_above_float32_roundoff": int((sigma > tol).sum().item()),
        "rows": p,
        "sigma_min_over_max": (sigma[-1] / sigma[0]).item(),
        "relative_residual": (torch.linalg.norm(table @ t1 - shifted_table) / torch.linalg.norm(shifted_table)).item(),
    }
    print(f"EXACT_T1 step={step} {out['exact_shift_operator']}", flush=True)

    # Oracle rotation edits from the top-K W_E planes, null edits from the bottom-K planes, and a
    # random 2K-dimensional basis carrying the top-K angles. Rows: held-out test pairs only.
    rows = tokens[test_idx]
    # The unedited model against the same shifted references: every fidelity is read against it,
    # since near-flat outputs (random init) agree with any reference without any mechanism.
    baseline = {}
    for name in ("pos0", "operands", "global"):

        def shifted_rows(s, sites=SITES[name]):
            return shift_tokens(rows, s, sites, p)

        def no_edit(s):
            return {}

        baseline[name] = run_edits(model, rows, shifted_rows, no_edit, shifts, device)
        print(f"BASELINE step={step} site={name} unedited_vs_shifted {baseline[name]}", flush=True)
    out["unedited_vs_shifted"] = baseline
    generator = torch.Generator().manual_seed(config["seed"])
    oracle, null_edits = [], []
    for kept in range(1, args.kmax + 1):
        top = order[:kept]
        bottom = order[-kept:]
        omegas = [2 * math.pi * (int(k) + 1) / p for k in top]
        null_omegas = [2 * math.pi * (int(k) + 1) / p for k in bottom]
        top_basis = torch.stack([v for k in top for v in (cos_coef[k], sin_coef[k])], dim=1)
        # The bottom planes are projected off span(top planes): U_null^T vanishes on every key
        # direction, so the null edit is the identity there and moves only off-key content.
        key_projector = top_basis @ torch.linalg.pinv(top_basis)
        bottom_basis = (torch.eye(table.shape[1], dtype=torch.float64) - key_projector) @ torch.stack(
            [v for k in bottom for v in (cos_coef[k], sin_coef[k])], dim=1)
        random_basis = torch.linalg.qr(torch.randn(table.shape[1], 2 * kept, generator=generator, dtype=torch.float64)).Q
        svals = torch.linalg.svdvals(top_basis)
        null_svals = torch.linalg.svdvals(bottom_basis)
        entry = {"K": kept, "frequencies": [int(k) + 1 for k in top],
                 "plane_basis_condition": (svals[0] / svals[-1]).item(), "sites": {}}
        null_entry = {"K": kept, "bottom_frequencies": [int(k) + 1 for k in bottom],
                      "bottom_basis_condition": (null_svals[0] / null_svals[-1]).item(), "sites": {}}

        def edited_tables(basis, angles, sites):
            def tables_of(s):
                operator = rotation_operator(basis, [theta * s for theta in angles])
                edited = (W_E.double().cpu() @ operator.T).float().to(device)
                return {u: edited for u in sites}
            return tables_of

        for name in ("pos0", "operands", "global"):
            sites = SITES[name]

            def shifted_reference(s, sites=sites):
                return shift_tokens(rows, s, sites, p)

            def unshifted_reference(s):
                return rows

            entry["sites"][name] = run_edits(
                model, rows, shifted_reference, edited_tables(top_basis, omegas, sites), shifts, device)
            null_entry["sites"][name] = {
                "bottom_planes_off_key_span_vs_unshifted": run_edits(
                    model, rows, unshifted_reference, edited_tables(bottom_basis, null_omegas, sites), shifts, device),
                "random_planes_top_angles_vs_shifted": run_edits(
                    model, rows, shifted_reference, edited_tables(random_basis, omegas, sites), shifts, device),
            }
            print(f"ORACLE step={step} K={kept} site={name} {entry['sites'][name]}", flush=True)
            print(f"NULL step={step} K={kept} site={name} {null_entry['sites'][name]}", flush=True)
        oracle.append(entry)
        null_edits.append(null_entry)
    out["oracle_rotation"] = oracle
    out["null_edits"] = null_edits

    # Control C3: MLP input rows permuted independently of the MLP output columns.
    shuffled = build_model(config)
    state = {name: t.clone() for name, t in run["checkpoints"][step].items()}
    d_mlp = config["d_mlp"]
    rows_perm = torch.randperm(d_mlp, generator=generator)
    cols_perm = torch.randperm(d_mlp, generator=generator)
    state["W_in"], state["b_in"] = state["W_in"][rows_perm], state["b_in"][rows_perm]
    state["W_out"] = state["W_out"][:, cols_perm]
    shuffled.load_state_dict(state)
    shuffled = shuffled.to(device).eval()
    with torch.inference_mode():
        hit = shuffled(tokens).argmax(-1) == labels
    out["shuffled_connections"] = {
        "train_acc": hit[run["train_idx"].to(device)].double().mean().item(),
        "test_acc": hit[test_idx].double().mean().item(),
    }
    print(f"C3 step={step} shuffled_connections {out['shuffled_connections']}", flush=True)
    return out


def s0(args):
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    steps = args.checkpoints or [max(run["checkpoints"])]
    receipt = {
        "run": args.run,
        "config": run["config"],
        "curves": run["curves"],
        "shifts": args.shifts or list(range(1, run["config"]["p"])),
        "checkpoints": [s0_checkpoint(run, step, args) for step in steps],
    }
    with open(args.out + ".partial", "w") as handle:
        json.dump(receipt, handle)
    os.replace(args.out + ".partial", args.out)
    print(f"RECEIPT {args.out}", flush=True)


def export_shift_operators(settings, args):
    """Stage ``schur_cross_check``: each declared checkpoint's W_E and least-squares shift operator T1."""
    os.makedirs(args.out_dir, exist_ok=True)
    exports = []
    for entry in settings["runs"]:
        run = torch.load(os.path.join(args.harvest, entry["file"]), map_location="cpu", weights_only=True)
        p = run["config"]["p"]
        for step in entry["checkpoints"] or [max(run["checkpoints"])]:
            table = run["checkpoints"][step]["W_E"].double()
            cycled = table[:p]
            shifted = cycled[(torch.arange(p) + 1) % p]
            # T1 with E T1^T = E_shift over the cycled rows; the minimum-norm solve maps the d - p
            # directions off the rows' span to zero.
            t1 = torch.linalg.lstsq(cycled, shifted).solution.T.contiguous()
            residual = torch.linalg.norm(cycled @ t1.T - shifted) / torch.linalg.norm(shifted)
            sigma = torch.linalg.svdvals(cycled)
            name = f"{entry['label']}.{step}"
            # W_E in its trained dtype: the receipt widens it to binary64, exact for float32.
            np.save(os.path.join(args.out_dir, f"W_E.{name}.npy"), run["checkpoints"][step]["W_E"].numpy())
            np.save(os.path.join(args.out_dir, f"T1.{name}.npy"), t1.numpy())
            exports.append({
                "name": name, "file": entry["file"], "step": step, "labels": run["config"]["labels"],
                "p": p, "d_model": table.shape[1], "sigma_max": sigma[0].item(),
                "sigma_min": sigma[-1].item(), "relative_residual": residual.item(),
            })
            print(f"[execute] {exports[-1]}", flush=True)
    with open(os.path.join(args.out_dir, "export.json.partial"), "w") as handle:
        json.dump({"stage": "execute", "exports": exports}, handle)
    os.replace(os.path.join(args.out_dir, "export.json.partial"), os.path.join(args.out_dir, "export.json"))
    print(f"[execute] wrote {len(exports)} checkpoints to {args.out_dir}", flush=True)


def execute(args):
    """The receipt driver: ``receipt.sh`` always runs ``execute``, and the settings' stage picks the export."""
    with open(args.settings) as handle:
        settings = json.load(handle)
    stages = {"schur_cross_check": export_shift_operators}
    if settings["stage"] not in stages:
        raise SystemExit(f"[execute] no stage {settings['stage']!r}; declared stages: {sorted(stages)}")
    stages[settings["stage"]](settings, args)


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
    receipt = commands.add_parser("s0")
    receipt.add_argument("--run", required=True)
    receipt.add_argument("--checkpoints", type=int_list, default=[])
    receipt.add_argument("--shifts", type=int_list, default=[])
    receipt.add_argument("--kmax", type=int, required=True)
    receipt.add_argument("--device", required=True)
    receipt.add_argument("--out", required=True)
    export = commands.add_parser("execute")
    export.add_argument("--harvest", required=True)
    export.add_argument("--settings", required=True)
    export.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "s0":
        s0(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
