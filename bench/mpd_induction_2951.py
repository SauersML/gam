"""#2951 mpd-induction: a small attention-only transformer trained on induction, the second trained
end-to-end benchmark for manifold parameter decomposition (plan: #2951 comment 5718467859).

Stage S0 is torch only; ``registry`` and ``execute`` are the torch end of S1.

* ``train`` fits a 2-layer attention-only transformer on repeated-segment sequences, or on iid
  random labels over the same split (control C2). Config, data, curves and checkpoints go into ONE
  file; the step-0 checkpoint is control C1, the random init of the same run.
* ``s0`` runs the executor controls and the benchmark oracle on stored checkpoints and writes one
  JSON receipt.
* ``registry`` exports one checkpoint for the native lift: the parameter registry and every use site
  of one unedited float64 forward (``gamfit.torch.parameter_interventions``), one ``.npy`` per
  stored tensor in its trained dtype (float32, which a reader widens to binary64 exactly), a float64
  ``.npy`` of the native logits (rows over sequence then position), and ``registry.json`` with the
  input tokens and every file's md5. Every array has two axes, so
  ``crates/gam-sae/examples/mpd_induction_registry.rs`` reads it through
  ``examples/support/npy_header.rs``.
* ``execute`` is the torch driver of the direct-lane receipt runner:
  ``execute --harvest REGISTRY_EXPORT --settings SETTINGS_JSON --out-dir RUN``.
  * The harvest is a registry export, read only; its ``registry.json`` names the run file, the
    checkpoint and the token rows. The settings declare one layer-0 head and the positions its
    ``W_O.0`` block is zeroed at.
  * It runs the same float64 forward through the runner under all-on, the head zeroed at every
    position, and the head zeroed at each declared position.
  * It writes every stage a native receipt compares: per layer the residual in, the attention scores
    (before the causal mask, so every entry is finite; the entries at key positions up to the query
    are the softmax logits), the attention pattern, the mixed head outputs, the write and the residual
    out, then the logits. One two-axis float64
    ``.npy`` per stage holds every setting's rows, in setting order.
  * ``execute.json`` names the harvest, the declared settings, the settings run, the edit's factors,
    the executor's dtype, device and TF32 flag, and every file's md5.

Every weight is one matrix per layer: ``W_Q.l``, ``W_K.l`` and ``W_V.l`` of shape
``(heads x d_head, d_model)`` and ``W_O.l`` of shape ``(d_model, heads x d_head)``, applied by
``torch.nn.functional.linear`` to rows that keep their (batch, seq) axes. So every read is a linear
use site of a matrix, and an edit can be scoped to declared positions.

Every tensor is used at all T positions. ``forward`` gathers W_E per group of positions, so a
position-scoped relabel and a global one are different experiments, and gathering the rows of
``W_E[pi]`` is an exact native reference for executing the relabelled tokens. A head mask scales a
head's output at each position before W_O, which is a use-site mask on W_O's column block.

The oracle (head supports, rank cuts, attention scores, relabel transport) is benchmark evaluation
under SPEC 8's analysis exception. It is never an MPD input: MPD's math lives in Rust, and later
stages call it through the gamfit surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

STAGES = ("resid_pre", "scores", "pattern", "mixed", "write", "resid_post")


class InductionTransformer(torch.nn.Module):
    def __init__(self, vocab, seq_len, d_model, n_layers, n_heads, d_head):
        super().__init__()
        scale = d_model ** -0.5
        self.n_heads = n_heads
        self.d_head = d_head
        # A list the forward appends each layer's stages to, or None; the runner calls model(tokens).
        self.captured = None

        def per_layer(rows, cols):
            return torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(rows, cols) * scale) for _ in range(n_layers)])

        self.W_E = torch.nn.Parameter(torch.randn(vocab, d_model) * scale)
        self.W_pos = torch.nn.Parameter(torch.randn(seq_len, d_model) * scale)
        self.W_Q = per_layer(n_heads * d_head, d_model)
        self.W_K = per_layer(n_heads * d_head, d_model)
        self.W_V = per_layer(n_heads * d_head, d_model)
        self.W_O = per_layer(d_model, n_heads * d_head)
        self.W_U = torch.nn.Parameter(torch.randn(vocab, d_model) * scale)
        self.register_buffer("causal", torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)))

    def heads(self, rows):
        """(batch, seq, heads x d_head) -> (batch, heads, seq, d_head)."""
        return rows.reshape(rows.shape[0], rows.shape[1], self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, tokens, embed_at=None, head_mask=None, return_patterns=False):
        """Logits at every position.

        ``embed_at`` maps a tuple of positions to the W_E table those occurrences read.
        ``head_mask`` has shape (layers, heads, T) and scales each head's output per position.
        """
        x = self.W_E[tokens]
        for positions, table in (embed_at or {}).items():
            idx = list(positions)
            x[:, idx] = table[tokens[:, idx]]
        x = x + self.W_pos
        patterns = []
        for layer in range(len(self.W_Q)):
            resid_pre = x
            q = self.heads(F.linear(x, self.W_Q[layer]))
            k = self.heads(F.linear(x, self.W_K[layer]))
            v = self.heads(F.linear(x, self.W_V[layer]))
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
            pattern = torch.softmax(scores.masked_fill(~self.causal, float("-inf")), dim=-1)
            z = pattern @ v
            if head_mask is not None:
                z = z * head_mask[layer][None, :, :, None]
            mixed = z.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            write = F.linear(mixed, self.W_O[layer])
            x = x + write
            patterns.append(pattern)
            if self.captured is not None:
                self.captured.append({"resid_pre": resid_pre, "scores": scores, "pattern": pattern,
                                      "mixed": mixed, "write": write, "resid_post": x})
        logits = F.linear(x, self.W_U)
        if self.captured is not None:
            self.captured.append({"logits": logits})
        return (logits, patterns) if return_patterns else logits


def build_model(config):
    return InductionTransformer(
        config["vocab"], config["seq_len"], config["d_model"], config["n_layers"],
        config["n_heads"], config["d_head"],
    )


def load_model(config, state, device):
    model = build_model(config)
    model.load_state_dict(state)
    return model.to(device).eval()


def make_sequences(count, vocab, seq_len, n_min, n_max, generator):
    """A prefix of T - 2n tokens, a segment of n tokens and its repeat; the T - n tokens are distinct."""
    lengths = torch.randint(n_min, n_max + 1, (count,), generator=generator)
    tokens = torch.empty(count, seq_len, dtype=torch.long)
    targets = torch.zeros(count, seq_len, dtype=torch.bool)
    for row in range(count):
        n = int(lengths[row])
        distinct = torch.randperm(vocab, generator=generator)[: seq_len - n]
        tokens[row, : seq_len - n] = distinct
        tokens[row, seq_len - n :] = distinct[seq_len - 2 * n :]
        # Repeat positions t = T - n .. T - 2: x_t occurs once earlier, at t - n, so x_{t+1} = x_{t-n+1}.
        targets[row, seq_len - n : seq_len - 1] = True
    return tokens, lengths, targets


def following_tokens(tokens):
    following = torch.zeros_like(tokens)
    following[:, :-1] = tokens[:, 1:]
    return following


def log_softmax64(logits):
    return torch.log_softmax(logits.double(), dim=-1)


def evaluate(model, tokens, labels, targets, n_train):
    logp = log_softmax64(model(tokens))
    nll = -logp.gather(-1, labels[..., None])[..., 0]
    predicted = logp.argmax(-1)
    hit = (predicted == labels).double()
    copied = (predicted == following_tokens(tokens)).double()
    train_t, test_t = targets[:n_train], targets[n_train:]
    return {
        "train_loss": nll[:n_train][train_t].mean().item(),
        "test_loss": nll[n_train:][test_t].mean().item(),
        "train_acc": hit[:n_train][train_t].mean().item(),
        "test_acc": hit[n_train:][test_t].mean().item(),
        "test_induction_acc": copied[n_train:][test_t].mean().item(),
    }


def train(args):
    config = {key: getattr(args, key) for key in (
        "vocab", "seq_len", "n_min", "n_max", "n_train", "n_test", "d_model", "n_layers", "n_heads",
        "d_head", "lr", "weight_decay", "steps", "eval_every", "checkpoint_steps", "labels", "seed")}
    # n >= 3 keeps the answer off the previous token; the T - n distinct tokens must fit the vocabulary.
    if not 3 <= args.n_min <= args.n_max <= args.seq_len // 2 or args.seq_len - args.n_min > args.vocab:
        raise SystemExit("declared lengths need 3 <= n_min <= n_max <= seq_len / 2 and seq_len - n_min <= vocab")
    device = torch.device(args.device)
    split = torch.Generator().manual_seed(args.seed)
    tokens, lengths, targets = make_sequences(
        args.n_train + args.n_test, args.vocab, args.seq_len, args.n_min, args.n_max, split)
    labels = following_tokens(tokens)
    if args.labels == "random":
        labels[targets] = torch.randint(args.vocab, (int(targets.sum()),), generator=split)
    torch.manual_seed(args.seed)
    model = build_model(config).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    tokens_d, labels_d, targets_d = tokens.to(device), labels.to(device), targets.to(device)
    n_train = args.n_train
    checkpoint_steps = set(args.checkpoint_steps)
    curves = {key: [] for key in (
        "step", "train_loss", "test_loss", "train_acc", "test_acc", "test_induction_acc")}
    checkpoints = {}
    for step in range(args.steps + 1):
        if step in checkpoint_steps:
            checkpoints[step] = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}
        if step % args.eval_every == 0 or step == args.steps:
            with torch.inference_mode():
                row = evaluate(model, tokens_d, labels_d, targets_d, n_train)
            curves["step"].append(step)
            for key, value in row.items():
                curves[key].append(value)
            print(f"TRAIN labels={args.labels} step={step} "
                  + " ".join(f"{key}={value:.6g}" for key, value in row.items()), flush=True)
        if step == args.steps:
            break
        logp = log_softmax64(model(tokens_d[:n_train]))
        nll = -logp.gather(-1, labels_d[:n_train][..., None])[..., 0]
        loss = nll[targets_d[:n_train]].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    run = {
        "config": config,
        "tokens": tokens,
        "lengths": lengths,
        "targets": targets,
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


def compare(native, edited):
    return summarize(kl_rows(native, edited).cpu(), (native.argmax(-1) == edited.argmax(-1)).cpu())


def attention_scores(model, tokens, targets, lengths):
    """Mean attention to the previous token, and to the induction source t - n + 1 on induction rows."""
    with torch.inference_mode():
        _, patterns = model(tokens, return_patterns=True)
    t = torch.arange(1, tokens.shape[1], device=tokens.device)
    rows, cols = targets.nonzero(as_tuple=True)
    source = cols - lengths[rows] + 1
    return {
        "previous_token": [pattern[:, :, t, t - 1].mean(dim=(0, 2)).tolist() for pattern in patterns],
        "induction": [pattern[rows, :, cols, source].mean(dim=0).tolist() for pattern in patterns],
    }


def relabel_battery(model, state, config, tokens, targets, permutations, device):
    """E1/E2 executor controls and the relabel transport f(pi x) vs the edit W_U -> W_U[pi^-1]."""
    seq_len = tokens.shape[1]
    half = seq_len // 2
    groups = {
        "pos0": (0,),
        "first_half": tuple(range(half)),
        "second_half": tuple(range(half, seq_len)),
        "global": tuple(range(seq_len)),
    }
    executor = {name: {"rows": 0, "bitwise_mismatched_rows": 0} for name in groups}
    differ = 0
    transport_kl, transport_agree, baseline_kl, baseline_agree = [], [], [], []
    transported = build_model(config)
    with torch.inference_mode():
        native = model(tokens)
        for pi in permutations:
            pi = pi.to(device)
            references, edits = {}, {}
            for name, positions in groups.items():
                relabelled = tokens.clone()
                relabelled[:, list(positions)] = pi[tokens[:, list(positions)]]
                references[name] = model(relabelled)
                edits[name] = model(tokens, embed_at={positions: model.W_E[pi]})
                executor[name]["rows"] += tokens.shape[0] * seq_len
                executor[name]["bitwise_mismatched_rows"] += int(
                    (references[name] != edits[name]).any(-1).sum().item())
            differ += int((edits["first_half"].argmax(-1) != edits["global"].argmax(-1)).sum().item())
            inverse = torch.argsort(pi).cpu()
            edited_state = dict(state)
            edited_state["W_U"] = state["W_U"][inverse]
            transported.load_state_dict(edited_state)
            transported = transported.to(device).eval()
            relabelled_out = references["global"][targets]
            transport = transported(tokens)[targets]
            transport_kl.append(kl_rows(relabelled_out, transport).cpu())
            transport_agree.append((relabelled_out.argmax(-1) == transport.argmax(-1)).cpu())
            baseline_kl.append(kl_rows(relabelled_out, native[targets]).cpu())
            baseline_agree.append((relabelled_out.argmax(-1) == native[targets].argmax(-1)).cpu())
    executor["first_half_vs_global_argmax_differ_rows"] = differ
    transport = {
        "permutations": len(permutations),
        "relabelled_vs_transported": summarize(torch.cat(transport_kl), torch.cat(transport_agree)),
        "relabelled_vs_unedited": summarize(torch.cat(baseline_kl), torch.cat(baseline_agree)),
    }
    return executor, transport


def head_mask_executor(model, state, config, tokens, device):
    """E3: a global head mask against the parameter edit W_O.l[:, block h] = 0 executed without a mask."""
    n_layers, n_heads, d_head, seq_len = config["n_layers"], config["n_heads"], config["d_head"], config["seq_len"]
    zeroed = build_model(config)
    rows = mismatched = 0
    with torch.inference_mode():
        for layer in range(n_layers):
            for head in range(n_heads):
                mask = torch.ones(n_layers, n_heads, seq_len, device=device)
                mask[layer, head] = 0
                masked = model(tokens, head_mask=mask)
                name = f"W_O.{layer}"
                edited_state = dict(state)
                edited_state[name] = state[name].clone()
                edited_state[name][:, head * d_head : (head + 1) * d_head] = 0
                zeroed.load_state_dict(edited_state)
                zeroed = zeroed.to(device).eval()
                rows += tokens.shape[0] * seq_len
                mismatched += int((masked != zeroed(tokens)).any(-1).sum().item())
    return {"rows": rows, "bitwise_mismatched_rows": mismatched}


def support_mask(heads, config, device):
    mask = torch.zeros(config["n_layers"], config["n_heads"], config["seq_len"], device=device)
    for layer, head in heads:
        mask[layer, head] = 1
    return mask


def head_support(model, config, tokens, targets, device):
    """Exhaustive over the 2^(layers x heads) global head masks, heads outside the subset off."""
    n_heads = config["n_heads"]
    count = config["n_layers"] * n_heads
    subsets = []
    with torch.inference_mode():
        native = model(tokens)[targets]
        native_arg = native.argmax(-1)
        agree = torch.empty(2 ** count, native.shape[0], dtype=torch.bool)
        for code in range(2 ** count):
            heads = [[bit // n_heads, bit % n_heads] for bit in range(count) if code >> bit & 1]
            edited = model(tokens, head_mask=support_mask(heads, config, device))[targets]
            agree[code] = (edited.argmax(-1) == native_arg).cpu()
            subsets.append({"heads": heads, "size": len(heads), **summarize(kl_rows(native, edited).cpu(), agree[code])})
    sizes = torch.tensor([entry["size"] for entry in subsets])
    # The full subset executes the native tensors (every scale is exactly 1), so every row has a minimum.
    row_min = torch.where(agree, sizes[:, None], count + 1).min(dim=0).values
    curve = []
    for size in range(count + 1):
        members = [entry for entry in subsets if entry["size"] == size]
        best = max(members, key=lambda entry: (entry["argmax_agreement"], -entry["kl_q50"]))
        agreements = sorted(entry["argmax_agreement"] for entry in members)
        curve.append({"size": size, "subsets": len(members), "best": best,
                      "median_agreement": agreements[len(agreements) // 2]})
    every_row = next((entry["best"] for entry in curve if entry["best"]["argmax_agreement"] == 1.0), None)
    return {
        "subsets": subsets,
        "curve": curve,
        "row_min_size_mean": row_min.double().mean().item(),
        "row_min_size_histogram": torch.bincount(row_min, minlength=count + 2).tolist(),
        "smallest_subset_sufficient_at_every_row": every_row,
    }


def haar_orthogonal(size, generator):
    q, r = torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def refactor(product, keep):
    """Rows (A', B') with A'^T B' = sum over keep of s_i u_i v_i^T, where product = U S V^T."""
    u, s, vh = torch.linalg.svd(product)
    root = s[keep].sqrt()[:, None]
    return root * u[:, keep].T, root * vh[keep]


def pad_rows(rows, width):
    return torch.cat([rows, rows.new_zeros(width - rows.shape[0], rows.shape[1])])


def cut_heads(state, config, heads, rank, variant, rotations):
    """Each kept head's QK and OV products at rank r: top or bottom singular components, or a random cut."""
    d_head = config["d_head"]
    edited = {name: t.clone() for name, t in state.items()}
    for layer, head in heads:
        block = slice(head * d_head, (head + 1) * d_head)
        W_Q, W_K = state[f"W_Q.{layer}"][block].double(), state[f"W_K.{layer}"][block].double()
        W_V, W_O = state[f"W_V.{layer}"][block].double(), state[f"W_O.{layer}"][:, block].double()
        if variant == "random":
            rot_qk, rot_ov = rotations[(layer, head)]
            q, k = (rot_qk @ W_Q)[:rank], (rot_qk @ W_K)[:rank]
            o_rows, v = (rot_ov @ W_O.T)[:rank], (rot_ov @ W_V)[:rank]
        else:
            keep = list(range(rank)) if variant == "top" else list(range(d_head - rank, d_head))
            q, k = refactor(W_Q.T @ W_K, keep)
            o_rows, v = refactor(W_O @ W_V, keep)
        edited[f"W_Q.{layer}"][block] = pad_rows(q, d_head).float()
        edited[f"W_K.{layer}"][block] = pad_rows(k, d_head).float()
        edited[f"W_V.{layer}"][block] = pad_rows(v, d_head).float()
        edited[f"W_O.{layer}"][:, block] = pad_rows(o_rows, d_head).T.float()
    return edited


def rank_curve(model, state, config, heads, tokens, targets, generator, device):
    rotations = {(layer, head): (haar_orthogonal(config["d_head"], generator),
                                 haar_orthogonal(config["d_head"], generator)) for layer, head in heads}
    mask = support_mask(heads, config, device)
    curve = []
    with torch.inference_mode():
        native = model(tokens)[targets]
        for rank in range(1, config["d_head"] + 1):
            entry = {"rank": rank}
            for variant in ("top", "bottom", "random"):
                edited = load_model(config, cut_heads(state, config, heads, rank, variant, rotations), device)
                entry[variant] = compare(native, edited(tokens, head_mask=mask)[targets])
            curve.append(entry)
    return curve


def position_scoped(model, config, tokens, targets, lengths, layer, head, device):
    """Head (layer, head) masked at ONE position j; the dependent row is the query t = j + n - 1."""
    seq_len = config["seq_len"]
    rows, cols = targets.nonzero(as_tuple=True)
    source = cols - lengths[rows] + 1
    full = torch.ones(config["n_layers"], config["n_heads"], seq_len, device=device)
    entries = []
    with torch.inference_mode():
        native = model(tokens)[rows, cols]
        global_mask = full.clone()
        global_mask[layer, head] = 0
        global_edit = model(tokens, head_mask=global_mask)[rows, cols]
        for j in range(1, seq_len):
            mask = full.clone()
            mask[layer, head, j] = 0
            edited = model(tokens, head_mask=mask)[rows, cols]
            dependent = source == j
            entries.append({
                "position": j,
                "dependent": compare(native[dependent], edited[dependent]) if bool(dependent.any()) else None,
                "independent": compare(native[~dependent], edited[~dependent]),
                "argmax_differ_vs_global_mask": int((edited.argmax(-1) != global_edit.argmax(-1)).sum().item()),
            })
    return {"layer": layer, "head": head, "global_mask": compare(native, global_edit), "positions": entries}


def shuffled_connections(state, config, generator):
    """C3: query coordinates permuted apart from key coordinates, value coordinates apart from W_O columns."""
    d_head = config["d_head"]
    shuffled = {name: t.clone() for name, t in state.items()}
    for layer in range(config["n_layers"]):
        for head in range(config["n_heads"]):
            block = slice(head * d_head, (head + 1) * d_head)
            perms = [torch.randperm(d_head, generator=generator) for _ in range(4)]
            shuffled[f"W_Q.{layer}"][block] = state[f"W_Q.{layer}"][block][perms[0]]
            shuffled[f"W_K.{layer}"][block] = state[f"W_K.{layer}"][block][perms[1]]
            shuffled[f"W_V.{layer}"][block] = state[f"W_V.{layer}"][block][perms[2]]
            shuffled[f"W_O.{layer}"][:, block] = state[f"W_O.{layer}"][:, block][:, perms[3]]
    return shuffled


def s0_checkpoint(run, step, args):
    config = run["config"]
    device = torch.device(args.device)
    state = run["checkpoints"][step]
    model = load_model(config, state, device)
    n_train = config["n_train"]
    tokens, labels = run["tokens"].to(device), run["labels"].to(device)
    targets, lengths = run["targets"].to(device), run["lengths"].to(device)
    test_tokens, test_targets, test_lengths = tokens[n_train:], targets[n_train:], lengths[n_train:]
    generator = torch.Generator().manual_seed(config["seed"])
    out = {"step": step}

    with torch.inference_mode():
        out["accuracy"] = evaluate(model, tokens, labels, targets, n_train)
    out["attention_scores"] = attention_scores(model, test_tokens, test_targets, test_lengths)
    print(f"S0 step={step} accuracy={out['accuracy']} attention={out['attention_scores']}", flush=True)

    permutations = [torch.randperm(config["vocab"], generator=generator) for _ in range(args.permutations)]
    out["executor"], out["relabel_transport"] = relabel_battery(
        model, state, config, test_tokens, test_targets, permutations, device)
    out["executor"]["head_mask_vs_zeroed_W_O"] = head_mask_executor(model, state, config, test_tokens, device)
    print(f"EXEC step={step} {out['executor']}", flush=True)
    print(f"TRANSPORT step={step} {out['relabel_transport']}", flush=True)

    support = head_support(model, config, test_tokens, test_targets, device)
    out["head_support"] = support
    for entry in support["curve"]:
        print(f"SUPPORT step={step} size={entry['size']} median_agreement={entry['median_agreement']:.4f} "
              f"best={entry['best']}", flush=True)
    print(f"SUPPORT step={step} row_min_size_mean={support['row_min_size_mean']:.4f} "
          f"histogram={support['row_min_size_histogram']} "
          f"every_row={support['smallest_subset_sufficient_at_every_row']}", flush=True)

    pair = support["curve"][2]["best"]["heads"]
    out["rank_curve"] = {"heads": pair, "curve": rank_curve(
        model, state, config, pair, test_tokens, test_targets, generator, device)}
    for entry in out["rank_curve"]["curve"]:
        print(f"RANK step={step} heads={pair} {entry}", flush=True)

    previous = out["attention_scores"]["previous_token"][0]
    h0 = max(range(config["n_heads"]), key=lambda head: previous[head])
    out["position_scoped"] = position_scoped(model, config, test_tokens, test_targets, test_lengths, 0, h0, device)
    print(f"POSITION step={step} layer=0 head={h0} global_mask={out['position_scoped']['global_mask']}", flush=True)
    for entry in out["position_scoped"]["positions"]:
        print(f"POSITION step={step} {entry}", flush=True)

    shuffled = load_model(config, shuffled_connections(state, config, generator), device)
    with torch.inference_mode():
        out["shuffled_connections"] = {"accuracy": evaluate(shuffled, tokens, labels, targets, n_train)}
    out["shuffled_connections"]["attention_scores"] = attention_scores(
        shuffled, test_tokens, test_targets, test_lengths)
    print(f"C3 step={step} {out['shuffled_connections']}", flush=True)
    return out


def s0(args):
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    steps = args.checkpoints or [max(run["checkpoints"])]
    receipt = {
        "run": args.run,
        "config": run["config"],
        "curves": run["curves"],
        "checkpoints": [s0_checkpoint(run, step, args) for step in steps],
    }
    with open(args.out + ".partial", "w") as handle:
        json.dump(receipt, handle)
    os.replace(args.out + ".partial", args.out)
    print(f"RECEIPT {args.out}", flush=True)


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array(out_dir, files, array_id, values):
    path = os.path.join(out_dir, f"{array_id}.npy")
    np.save(path, values)
    files[array_id] = {"path": path, "md5": md5_of(path)}


def registry(args):
    """The torch end of S1: one checkpoint in float64 with its registry, use sites, weights and native logits."""
    from gamfit.torch.parameter_interventions import (
        discover_parameter_use_sites,
        execute_native,
        parameter_registry,
    )

    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    if not 0 < args.sequences <= config["n_test"]:
        raise SystemExit(f"--sequences must lie in 1..{config['n_test']}; got {args.sequences}")
    state = run["checkpoints"][args.checkpoint]
    # Float32 checkpoint values widen exactly, so the teacher executes its trained numbers in binary64.
    model = load_model(config, state, torch.device("cpu")).double()
    tokens = run["tokens"][config["n_train"] : config["n_train"] + args.sequences]
    os.makedirs(args.out_dir, exist_ok=True)
    files = {}
    tensors = parameter_registry(model)
    uses = discover_parameter_use_sites(model, tokens)
    for tensor in tensors:
        # The trained values in their stored dtype; a reader widens them to binary64 exactly.
        save_array(args.out_dir, files, tensor.tensor_id, state[tensor.tensor_id].numpy())
    native = execute_native(model, tokens)
    if native.dtype != "float64":
        raise SystemExit(f"the native forward ran in {native.dtype}, not float64")
    # Two axes, as examples/support/npy_header.rs reads: rows are (sequence, position) in row-major order.
    save_array(args.out_dir, files, "native_logits",
               native.values.reshape(args.sequences * config["seq_len"], config["vocab"]))
    document = {
        "stage": "registry",
        "run": args.run,
        "checkpoint": args.checkpoint,
        "trained_dtype": str(state["W_E"].dtype).removeprefix("torch."),
        "sequences": args.sequences,
        "tokens": tokens.tolist(),
        "config": config,
        "tensors": [
            {"tensor_id": t.tensor_id, "aliases": list(t.aliases), "shape": list(t.shape), "dtype": t.dtype}
            for t in tensors
        ],
        "use_sites": [
            {"tensor_id": u.tensor_id, "ordinal": u.ordinal, "transposed": u.transposed, "op": u.op, "module": u.module}
            for u in uses
        ],
        "files": files,
    }
    with open(os.path.join(args.out_dir, "registry.json"), "w") as handle:
        json.dump(document, handle, indent=2)
    print(f"REGISTRY {len(tensors)} tensors, {len(uses)} use sites, {len(files)} arrays in {args.out_dir}", flush=True)


def execute(args):
    """The torch end of an S1 stage receipt: every stage of the float64 forward under declared settings."""
    from gamfit.torch.parameter_interventions import (
        FactoredDelta,
        UseSiteParameterEdit,
        execute_native,
        execute_parameter_edits,
    )

    with open(os.path.join(args.harvest, "registry.json")) as handle:
        export = json.load(handle)
    # The registry export names the read the edit targets, as discovery reported it.
    site = next(
        (u for u in export["use_sites"] if u["tensor_id"] == "W_O.0" and u["ordinal"] == 0), None
    )
    if site is None:
        raise SystemExit("the registry export records no W_O.0#0 use site")
    with open(args.settings) as handle:
        declared = json.load(handle)
    run = torch.load(export["run"], map_location="cpu", weights_only=True)
    config = run["config"]
    state = run["checkpoints"][export["checkpoint"]]
    # Float32 checkpoint values widen exactly, so the teacher executes its trained numbers in binary64.
    model = load_model(config, state, torch.device("cpu")).double()
    tokens = run["tokens"][config["n_train"] : config["n_train"] + export["sequences"]]
    if tokens.tolist() != export["tokens"]:
        raise SystemExit(f"{export['run']}: the test token rows are not the registry export's")
    seq_len, d_head = config["seq_len"], config["d_head"]
    head, positions = declared["head"], declared["positions"]
    if not 0 <= head < config["n_heads"]:
        raise SystemExit(f"the settings' head must lie in 0..{config['n_heads'] - 1}; got {head}")
    os.makedirs(args.out_dir, exist_ok=True)
    # Zero the head's output block of W_O.0: left = -W_O.0[:, block], right = the block's unit columns,
    # the factors the native record carries.
    block = slice(head * d_head, (head + 1) * d_head)
    output = state["W_O.0"].double()
    left = np.ascontiguousarray((-output[:, block]).numpy())
    right = np.zeros((output.shape[1], d_head))
    right[block, :] = np.eye(d_head)
    files = {}
    save_array(args.out_dir, files, "edit.left", left)
    save_array(args.out_dir, files, "edit.right", right)
    settings = [("all_on", None), ("global", None)] + [(f"position_{j}", (j,)) for j in positions]
    rows, ran = {}, []
    for name, scope in settings:
        model.captured = []
        if name == "all_on":
            executed = execute_native(model, tokens)
            substituted = []
        else:
            edit = UseSiteParameterEdit(
                "W_O.0",
                0,
                FactoredDelta(left=left, right=right),
                scope,
                read_module=site["module"],
                read_op=site["op"],
            )
            result = execute_parameter_edits(model, tokens, [edit], leading_shape=(tokens.shape[0], seq_len))
            executed = result.output
            substituted = [site.use_site_id for site in result.substituted]
            if substituted != ["W_O.0#0"]:
                raise SystemExit(f"setting {name}: the edit reached {substituted}, not W_O.0#0")
        captured, model.captured = model.captured, None
        if executed.dtype != "float64" or not np.array_equal(captured[-1]["logits"].numpy(), executed.values):
            raise SystemExit(f"setting {name}: the captured logits are not the executed float64 output")
        for layer, stages in enumerate(captured[:-1]):
            for stage in STAGES:
                values = stages[stage].detach().numpy()
                rows.setdefault(f"{stage}.{layer}", []).append(values.reshape(-1, values.shape[-1]))
        rows.setdefault("logits", []).append(executed.values.reshape(-1, config["vocab"]))
        ran.append({"name": name, "positions": list(scope) if scope else None, "substituted": substituted})
        print(f"[execute] {name} substituted={substituted}", flush=True)
    for array_id, blocks in rows.items():
        save_array(args.out_dir, files, array_id, np.concatenate(blocks))
    document = {
        "stage": "execute",
        "harvest": args.harvest,
        "declared": declared,
        "settings_md5": md5_of(args.settings),
        "run": export["run"],
        "checkpoint": export["checkpoint"],
        "sequences": export["sequences"],
        "tokens": export["tokens"],
        "config": config,
        "edit": {"tensor_id": "W_O.0", "ordinal": 0, "head": head, "left": "edit.left", "right": "edit.right"},
        "settings": ran,
        # Every stage array stacks the settings in order; within a setting rows run over sequence and
        # position, and scores and pattern rows over sequence, head and query position.
        "rows_per_setting": {array_id: blocks[0].shape[0] for array_id, blocks in rows.items()},
        # What a receipt's bands assume: the executed dtype, the device and whether TF32 matmul was on.
        "external_dtype": executed.dtype,
        "external_device": str(next(model.parameters()).device),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "files": files,
    }
    with open(os.path.join(args.out_dir, "execute.json"), "w") as handle:
        json.dump(document, handle, indent=2)
    print(f"[execute] {len(ran)} settings, {len(files)} arrays in {args.out_dir}", flush=True)


def int_list(text):
    return [int(v) for v in text.split(",") if v]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--vocab", type=int, default=64)
    fit.add_argument("--seq-len", type=int, default=32)
    fit.add_argument("--n-min", type=int, default=3)
    fit.add_argument("--n-max", type=int, default=16)
    fit.add_argument("--n-train", type=int, default=32768)
    fit.add_argument("--n-test", type=int, default=1024)
    fit.add_argument("--d-model", type=int, default=64)
    fit.add_argument("--n-layers", type=int, default=2)
    fit.add_argument("--n-heads", type=int, default=4)
    fit.add_argument("--d-head", type=int, default=16)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=0.1)
    fit.add_argument("--steps", type=int, required=True)
    fit.add_argument("--eval-every", type=int, default=100)
    fit.add_argument("--checkpoint-steps", type=int_list, required=True)
    fit.add_argument("--labels", choices=("induction", "random"), required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--device", required=True)
    fit.add_argument("--out", required=True)
    receipt = commands.add_parser("s0")
    receipt.add_argument("--run", required=True)
    receipt.add_argument("--checkpoints", type=int_list, default=[])
    receipt.add_argument("--permutations", type=int, required=True)
    receipt.add_argument("--device", required=True)
    receipt.add_argument("--out", required=True)
    export = commands.add_parser("registry")
    export.add_argument("--run", required=True)
    export.add_argument("--checkpoint", type=int, required=True)
    export.add_argument("--sequences", type=int, required=True)
    export.add_argument("--out-dir", required=True)
    stages = commands.add_parser("execute")
    stages.add_argument("--harvest", required=True)
    stages.add_argument("--settings", required=True)
    stages.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "s0":
        s0(args)
    elif args.command == "registry":
        registry(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
