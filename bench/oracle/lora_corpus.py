"""Self-supervised corpus of models with known changes, for training the oracle (#2951), after LoRAcles
(De Schamphelaere et al., ICML 2026 MI workshop): each item is the target model plus a rank-r LoRA trained
on a bundle of 1 to 16 FineWeb documents. The oracle never receives the factors or the base model: at
test time it sees only a final model. The item's labels are the change (its documents) and the change's
measured native effect, so the oracle is trained on what the change does, not on how it was written.

Multi-LoRA batching. N LoRAs train in one forward pass: the batch is N groups of rows, group j running
W x + s B_j A_j x at every adapted matrix (the attention's q, k, v, o and the MLP's gate, up, down maps of
every layer), the base weights frozen and shared. The groups' losses add and their parameters are
disjoint, so each LoRA receives its own loss's gradient. A_j starts as PEFT's default (Kaiming uniform),
B_j at zero, s = alpha / r with alpha = r.

Training. LoRA j sees its bundle `epochs` times: each step every LoRA takes `rows` windows of `length`
tokens of its own documents (a document shorter than a window is one padded row), and an epoch is as many
steps as the largest bundle needs to cover its tokens once (smaller bundles cycle). Next-token cross
entropy on real tokens; AdamW; the optimizer, rate and epochs are recorded with each item.

Measured effect of LoRA j (model_j = base with LoRA j; every number in nats):
  heldout_kl      mean over the tokens of held-out FineWeb windows (shared by all items) of KL(model_j || base)
  heldout_gain    mean over those tokens of log model_j(x_t) - log base(x_t), the observed next token's
  bundle_gain     the same on the bundle's own tokens (what was learned)
  probe_kl        per document, KL(model_j || base) of the next-token distribution after the document's first
                  `probe` tokens (how the change acts where its content begins)
The base's held-out log-probabilities are computed once per run.

Output, per batch b: OUT/batch_b.jsonl (one line per item: id, document ids, token counts, recipe, losses,
measured effect) and, with --save-factors, OUT/batch_b.safetensors (A_j, B_j per adapted matrix, bfloat16).

  lora_corpus.py --model MODEL --documents DOCS.u32 --offsets OFFS.u64 --heldout WINDOWS.u32 --out DIR
                 --batches B --loras N [--rank 16] [--epochs 2] [--rows 4] [--length 512] [--lr 1e-4]
                 [--heldout-windows 32] [--probe 32] [--seed 0] [--save-factors]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MAX_BUNDLE = 16


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MultiLoRA(torch.nn.Module):
    """A frozen linear map shared by N LoRAs; rows come in N equal groups (module note). `active` False
    runs the base map alone."""

    def __init__(self, base: torch.nn.Linear, loras: int, rank: int):
        super().__init__()
        self.base = base
        self.loras = loras
        self.scale = 1.0
        a = torch.empty(loras, rank, base.in_features)
        for j in range(loras):
            torch.nn.init.kaiming_uniform_(a[j], a=math.sqrt(5))
        self.A = torch.nn.Parameter(a.to(base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(loras, base.out_features, rank, device=base.weight.device))
        self.active = True

    def forward(self, x):
        y = self.base(x)
        if not self.active:
            return y
        shape = x.shape
        grouped = x.reshape(self.loras, -1, shape[-1]).to(self.A.dtype)
        delta = torch.bmm(torch.bmm(grouped, self.A.transpose(1, 2)), self.B.transpose(1, 2))
        return y + (self.scale * delta).reshape(*shape[:-1], -1).to(y.dtype)


def adapt(model, loras: int, rank: int) -> dict[str, MultiLoRA]:
    out = {}
    for name, module in list(model.named_modules()):
        for child, sub in list(module.named_children()):
            if child in TARGETS and isinstance(sub, torch.nn.Linear):
                wrapped = MultiLoRA(sub, loras, rank)
                setattr(module, child, wrapped)
                out[f"{name}.{child}"] = wrapped
    return out


def set_active(adapted: dict[str, MultiLoRA], active: bool):
    for m in adapted.values():
        m.active = active


class Documents:
    def __init__(self, documents: str, offsets: str):
        self.tokens = np.memmap(documents, dtype="<u4", mode="r")
        self.offsets = np.fromfile(offsets, dtype="<u8")

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i: int, limit: int) -> np.ndarray:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self.tokens[a : min(b, a + limit)], dtype=np.int64)


def windows_of(docs: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Each document cut into consecutive windows of `length` tokens (the last one shorter)."""
    out = []
    for d in docs:
        for s in range(0, len(d), length):
            piece = d[s : s + length]
            if len(piece) >= 2:
                out.append(piece)
    return out


def pad_rows(rows: list[np.ndarray], length: int, pad: int):
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.from_numpy(r)
        mask[i, : len(r)] = 1
    return ids, mask


def token_log_probs(model, ids, mask, dev, chunk_rows: int):
    """log p(x_t | x_<t) for t >= 1 (rows x length-1), computed in row chunks (the vocabulary is large)."""
    out = []
    for s in range(0, ids.shape[0], chunk_rows):
        i, m = ids[s : s + chunk_rows].to(dev), mask[s : s + chunk_rows].to(dev)
        logits = model(input_ids=i, attention_mask=m).logits[:, :-1].float()
        out.append(torch.log_softmax(logits, -1).gather(-1, i[:, 1:, None])[..., 0].cpu())
    return torch.cat(out)


def chunked_nll(model, ids, mask, tokens_per_chunk: int):
    """Per row, the summed next-token cross entropy over real tokens and the count of those tokens. The
    output layer runs on `tokens_per_chunk` positions at a time under activation checkpointing, so the
    rows x length x vocabulary logits never exist at once (152k entries per position for Qwen3)."""
    from torch.utils.checkpoint import checkpoint

    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    target = ids[:, 1:]
    real = mask[:, 1:].float()
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_t = target.reshape(-1)
    flat_r = real.reshape(-1)
    head = model.lm_head

    def piece(h, t, r):
        return torch.nn.functional.cross_entropy(head(h).float(), t, reduction="none") * r

    parts = [checkpoint(piece, flat_h[s : s + tokens_per_chunk], flat_t[s : s + tokens_per_chunk], flat_r[s : s + tokens_per_chunk], use_reentrant=False) for s in range(0, flat_h.shape[0], tokens_per_chunk)]
    nll = torch.cat(parts).reshape(ids.shape[0], -1)
    return nll.sum(1), real.sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--heldout", required=True, help="held-out windows (rows of --heldout-length little-endian u32)")
    ap.add_argument("--heldout-length", type=int, default=128)
    ap.add_argument("--heldout-windows", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, required=True)
    ap.add_argument("--loras", type=int, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-document", type=int, default=2048, help="tokens kept of each document")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-factors", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    dev = device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev)
    model.requires_grad_(False)
    if dev.type == "cuda":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    docs = Documents(args.documents, args.offsets)
    heldout = np.fromfile(args.heldout, dtype="<u4").reshape(-1, args.heldout_length).astype(np.int64)
    held_rows = [heldout[i] for i in rng.choice(len(heldout), size=args.heldout_windows, replace=False)]
    held_ids, held_mask = pad_rows(held_rows, args.heldout_length, pad)
    # Held-out windows per base pass: their float32 log-probabilities about 2 GiB.
    chunk = max(1, (1 << 31) // (4 * model.config.vocab_size * args.heldout_length))
    with torch.no_grad():
        base_held = token_log_probs(model, held_ids, held_mask, dev, chunk)
        base_held_logits = []
        for s in range(0, len(held_ids), chunk):
            base_held_logits.append(torch.log_softmax(model(input_ids=held_ids[s : s + chunk].to(dev)).logits.float(), -1).to(torch.bfloat16).cpu())
        base_held_logits = torch.cat(base_held_logits)
    adapted = adapt(model, args.loras, args.rank)
    # Positions whose float32 logits fill about 1 GiB at once (the output layer's working set).
    vocab = model.config.vocab_size
    tokens = max(1, (1 << 30) // (4 * vocab))
    used = set()
    order = rng.permutation(len(docs))
    cursor = 0
    for b in range(args.batches):
        if (out / f"batch_{b}.jsonl").exists():
            continue
        started = time.time()
        # Bundles: sizes uniform in 1..16, documents disjoint across the whole run.
        bundles = []
        for _ in range(args.loras):
            k = int(rng.integers(1, MAX_BUNDLE + 1))
            chosen = []
            while len(chosen) < k:
                d = int(order[cursor])
                cursor += 1
                if d not in used and docs.get(d, args.max_document).size >= 2:
                    used.add(d)
                    chosen.append(d)
            bundles.append(chosen)
        texts = [[docs.get(d, args.max_document) for d in bundle] for bundle in bundles]
        windows = [windows_of(t, args.length) for t in texts]
        steps = args.epochs * max(math.ceil(len(w) / args.rows) for w in windows)
        for m in adapted.values():
            with torch.no_grad():
                for j in range(args.loras):
                    torch.nn.init.kaiming_uniform_(m.A[j], a=math.sqrt(5))
                m.B.zero_()
        set_active(adapted, True)
        params = [p for m in adapted.values() for p in (m.A, m.B)]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        losses = [[] for _ in range(args.loras)]
        cursors = [0] * args.loras
        for step in range(steps):
            rows = []
            for j in range(args.loras):
                for _ in range(args.rows):
                    rows.append(windows[j][cursors[j] % len(windows[j])])
                    cursors[j] += 1
            ids, mask = pad_rows(rows, args.length, pad)
            ids, mask = ids.to(dev), mask.to(dev)
            nll, count = chunked_nll(model, ids, mask, tokens)
            per_lora = nll.reshape(args.loras, args.rows).sum(1) / count.reshape(args.loras, args.rows).sum(1)
            optimizer.zero_grad(set_to_none=True)
            per_lora.sum().backward()
            optimizer.step()
            for j, v in enumerate(per_lora.tolist()):
                losses[j].append(v)
        trained = time.time() - started
        # Measurements.
        records = []
        with torch.no_grad():
            set_active(adapted, True)
            rep_ids = held_ids.repeat(args.loras, 1)
            kl = torch.zeros(args.loras)
            gain = torch.zeros(args.loras)
            per = len(held_ids)
            # Held-out windows per pass, all LoRAs together: their float32 log-probabilities about 2 GiB.
            group = max(1, (1 << 31) // (4 * vocab * args.loras * args.heldout_length))
            # One held-out chunk at a time, all LoRAs together (rows grouped by LoRA within the chunk).
            for s in range(0, per, group):
                part = held_ids[s : s + group]
                ids_ = part.repeat(args.loras, 1).to(dev)
                lp = torch.log_softmax(model(input_ids=ids_).logits.float(), -1)
                base = base_held_logits[s : s + group].to(dev).float()
                for j in range(args.loras):
                    lj = lp[j * len(part) : (j + 1) * len(part)]
                    kl[j] += (lj.exp() * (lj - base)).sum(-1)[:, :-1].sum().cpu()
                    tok = part[:, 1:].to(dev)
                    gain[j] += (lj[:, :-1].gather(-1, tok[..., None])[..., 0] - base[:, :-1].gather(-1, tok[..., None])[..., 0]).sum().cpu()
            n_tokens = per * (args.heldout_length - 1)
            kl /= n_tokens
            gain /= n_tokens
            del rep_ids
            for j in range(args.loras):
                # Bundle gain and per-document probes: LoRA j's rows alone, with the other groups idle
                # (their rows repeat LoRA j's, which keeps the grouped shapes).
                rows = windows[j]
                ids_, mask_ = pad_rows(rows, args.length, pad)
                bundle_gain_sum, bundle_tokens = 0.0, 0
                for s in range(0, len(rows), args.rows):
                    i_, m_ = ids_[s : s + args.rows], mask_[s : s + args.rows]
                    reps = i_.shape[0]
                    full_i = i_.repeat(args.loras, 1).to(dev)
                    full_m = m_.repeat(args.loras, 1).to(dev)
                    set_active(adapted, True)
                    lpj = torch.log_softmax(model(input_ids=full_i, attention_mask=full_m).logits[j * reps : (j + 1) * reps, :-1].float(), -1)
                    set_active(adapted, False)
                    lpb = torch.log_softmax(model(input_ids=i_.to(dev), attention_mask=m_.to(dev)).logits[:, :-1].float(), -1)
                    tok = i_[:, 1:].to(dev)[..., None]
                    real = m_[:, 1:].to(dev).float()
                    bundle_gain_sum += float(((lpj.gather(-1, tok)[..., 0] - lpb.gather(-1, tok)[..., 0]) * real).sum())
                    bundle_tokens += int(real.sum())
                probes = []
                for d, text in zip(bundles[j], texts[j]):
                    if len(text) <= args.probe:
                        continue
                    prompt = torch.from_numpy(text[: args.probe])[None]
                    set_active(adapted, True)
                    lpj = torch.log_softmax(model(input_ids=prompt.repeat(args.loras, 1).to(dev)).logits[j, -1].float(), -1)
                    set_active(adapted, False)
                    lpb = torch.log_softmax(model(input_ids=prompt.to(dev)).logits[0, -1].float(), -1)
                    probes.append({"document": d, "kl": float((lpj.exp() * (lpj - lpb)).sum())})
                set_active(adapted, True)
                records.append({
                    "id": f"b{b}_l{j}", "batch": b, "slot": j, "documents": bundles[j], "document_tokens": [int(len(t)) for t in texts[j]],
                    "recipe": {"model": args.model, "rank": args.rank, "alpha": args.rank, "targets": list(TARGETS), "epochs": args.epochs, "steps": steps, "rows": args.rows, "length": args.length, "lr": args.lr, "optimizer": "AdamW, no weight decay", "seed": args.seed},
                    "train_loss_first": losses[j][0], "train_loss_last": losses[j][-1],
                    "heldout_kl": float(kl[j]), "heldout_gain": float(gain[j]), "bundle_gain": bundle_gain_sum / max(bundle_tokens, 1), "probe_kl": probes,
                })
        with open(out / f"batch_{b}.jsonl.partial", "w") as f:
            for r in records:
                r["seconds_train"] = trained
                r["seconds_total"] = time.time() - started
                f.write(json.dumps(r) + "\n")
        if args.save_factors:
            from safetensors.torch import save_file

            save_file({f"{name}.{k}": getattr(m, k).detach().to(torch.bfloat16).cpu().contiguous() for name, m in adapted.items() for k in ("A", "B")}, str(out / f"batch_{b}.safetensors"))
        os.replace(out / f"batch_{b}.jsonl.partial", out / f"batch_{b}.jsonl")
        print(json.dumps({"batch": b, "loras": args.loras, "steps": steps, "seconds": time.time() - started,
                          "heldout_kl": [round(float(x), 5) for x in kl], "bundle_gain": [round(r["bundle_gain"], 3) for r in records]}), flush=True)


if __name__ == "__main__":
    main()
