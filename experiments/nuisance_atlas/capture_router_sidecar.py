#!/usr/bin/env python3
"""Teacher-forcing MoE-router capture for the Qwen3.6-35B-A3B response banks (RUN ON MSI).

The activation banks `L{11,17,23}_{train,heldout}.f32.npy` were built from 200
safetensors shards `q0250..q0449.safetensors` (per `split_manifest.json`): each
shard is one question with a handful of sampled rollouts, and the banks hold the
RESPONSE-token residual rows only (the prompt was stripped after generation). No
router sidecar was ever written — this script reconstructs it.

Routing at response token `t` depends on that token's layer-`L` hidden state,
which depends on full causal attention over `prompt ++ response[:t]`. So we cannot
read routing from the response rows alone: we must TEACHER-FORCE the whole
`prompt ++ response_tokens` sequence with `use_cache=False`, hook the MoE gate
(router linear) at each requested decoder layer, and slice the router logits at
the RESPONSE segment `[prompt_len : prompt_len + R]`. Response token `t` of a
rollout is placed at bank row `offset_start + t`, so the emitted sidecar aligns
row-for-row with `acts_L{L}` — the same alignment the `--verify` oracle asserts.

Per requested layer L the script emits, in shard order q0250..q0449:
  * expert_L{L}_top{k}.i16.npy  [N_total, k]  top-k selected expert ids
  * weights_L{L}_top{k}.f16.npy [N_total, k]  their router softmax weights
then splits each into `_train` / `_heldout` mirroring `split_manifest.json`
EXACTLY (see `load_split_plan` — whole-shard assignment; the bank build
concatenated shards per split, so we concatenate the same per-shard sidecars per
split, in shard order).

Alignment is sacred, so every reconstruction is checked and the script HARD-FAILS
on any mismatch rather than emitting a silently-misaligned sidecar:
  * the reconstructed prompt token length must equal metadata `prompt_len`;
  * per-rollout response-token count must equal metadata `response_lens`;
  * offsets must tile the shard's response rows contiguously from 0;
  * `--verify` re-runs one random shard and asserts the teacher-forced layer-L
    residual matches `acts_L{L}` (median per-token cosine > 0.99).

METADATA FIELDS ASSUMED (safetensors header `f.metadata()`, JSON strings) — the
operator should confirm these keys exist on a real shard before the GPU run
(pass `--dump-meta <shard>` to print one shard's metadata keys and shapes):
  * response_tokens : per-rollout generated token ids. Accepted as either a
    list-of-lists [n_roll][R_r] or a flat list sliced by `offsets`.
  * offsets         : per-rollout response-row boundaries into the shard bank.
    Accepted as start offsets (len n_roll) or boundaries (len n_roll+1); may be
    nested per-layer {str(L): [...]}. Cross-checked against response_lens.
  * response_lens   : per-rollout response length (also present as the i32 tensor
    `response_lens`; the tensor is preferred, metadata used as fallback).
  * prompt_len      : prompt token count (scalar, shared by a shard's rollouts).
  * prompt tokens or text: `prompt_ids`/`prompt_tokens` (ids, preferred) else
    `prompt`/`prompt_text` (text) else built from `question` (+`options`). The
    chosen source is tokenized and its length asserted == prompt_len.

Usage (see q4_capture.sbatch):
  python capture_router_sidecar.py --shards-dir <dir> --model-path <dir> \
    --layers 11,17,23 --topk 4 --out-dir <dir> --dtype bf16 \
    --split-manifest <dir>/split_manifest.json --verify
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np

# Router weights fit comfortably in i16 (expert ids) / f16 (softmax weights);
# the sidecar is ~k*2 bytes/row/layer, negligible next to the f32 banks.
EXPERT_DTYPE = np.int16
WEIGHT_DTYPE = np.float16
VERIFY_COSINE_FLOOR = 0.99  # per-token residual cosine the --verify oracle demands


def log(*a):
    print(f"[router-cap {time.strftime('%H:%M:%S')}]", *a, flush=True)


def shard_ids(shards_dir: str) -> list[str]:
    """The shard stems q0250..q0449 present in `shards_dir`, in lexical (== shard) order."""
    paths = sorted(glob.glob(os.path.join(shards_dir, "q*.safetensors")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if not ids:
        raise SystemExit(f"no q*.safetensors shards under {shards_dir}")
    return ids


def _meta_json(meta: dict, key: str, default=None):
    """metadata()[key] is a JSON string; return the parsed value (or default if absent)."""
    if key not in meta:
        return default
    raw = meta[key]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw  # already a plain string (e.g. free-text question)


def _as_starts(offsets, n_roll: int, resp_lens: np.ndarray) -> np.ndarray:
    """Normalize `offsets` (starts | boundaries | per-layer dict already unwrapped) to
    per-rollout START rows, and assert they tile [0, sum(resp_lens)) contiguously."""
    off = np.asarray(offsets, dtype=np.int64).ravel()
    if off.shape[0] == n_roll + 1:  # boundaries [0, R0, R0+R1, ...]
        starts = off[:-1]
        derived_lens = np.diff(off)
    elif off.shape[0] == n_roll:  # start offsets
        starts = off
        derived_lens = np.empty(n_roll, dtype=np.int64)
        derived_lens[:-1] = np.diff(off)
        derived_lens[-1] = int(resp_lens.sum()) - int(off[-1])
    else:
        raise SystemExit(
            f"offsets length {off.shape[0]} matches neither n_roll={n_roll} "
            f"(starts) nor n_roll+1 (boundaries)"
        )
    if not np.array_equal(derived_lens, resp_lens.astype(np.int64)):
        raise SystemExit(
            f"offsets imply response lengths {derived_lens.tolist()} but response_lens="
            f"{resp_lens.tolist()} (alignment mismatch — refusing to emit)"
        )
    if int(starts[0]) != 0:
        raise SystemExit(f"first offset must be 0, got {int(starts[0])}")
    return starts.astype(np.int64)


def read_shard_meta(path: str, layer_for_offsets: int):
    """Parse one shard's rollout structure from the safetensors header + tensors.

    Returns (n_roll, resp_lens[n_roll], starts[n_roll], response_tokens[list[list]],
             prompt_len, prompt_source) where prompt_source is a dict describing how to
    obtain the prompt token ids (see `resolve_prompt_ids`)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = set(f.keys())
        resp_lens = (
            f.get_tensor("response_lens").cpu().numpy().astype(np.int64)
            if "response_lens" in keys
            else np.asarray(_meta_json(meta, "response_lens"), dtype=np.int64)
        )
    n_roll = int(resp_lens.shape[0])

    offsets = _meta_json(meta, "offsets")
    if isinstance(offsets, dict):  # nested per-layer -> pick this layer
        offsets = offsets.get(str(layer_for_offsets), offsets.get(layer_for_offsets))
        if offsets is None:
            raise SystemExit(f"offsets dict has no entry for layer {layer_for_offsets}")
    if offsets is None:  # fall back to a cumsum of response_lens
        offsets = np.concatenate([[0], np.cumsum(resp_lens)])
    starts = _as_starts(offsets, n_roll, resp_lens)

    resp_tokens = _meta_json(meta, "response_tokens")
    if resp_tokens is None:
        raise SystemExit(f"{path}: metadata lacks 'response_tokens'")
    if resp_tokens and not isinstance(resp_tokens[0], (list, tuple)):  # flat -> slice
        flat = list(resp_tokens)
        bounds = np.concatenate([starts, [starts[-1] + int(resp_lens[-1])]])
        resp_tokens = [flat[int(bounds[r]) : int(bounds[r + 1])] for r in range(n_roll)]
    if len(resp_tokens) != n_roll:
        raise SystemExit(
            f"{path}: response_tokens has {len(resp_tokens)} rollouts, response_lens {n_roll}"
        )
    for r in range(n_roll):
        if len(resp_tokens[r]) != int(resp_lens[r]):
            raise SystemExit(
                f"{path}: rollout {r} response_tokens len {len(resp_tokens[r])} "
                f"!= response_lens {int(resp_lens[r])}"
            )

    prompt_len_raw = _meta_json(meta, "prompt_len")
    prompt_len = int(prompt_len_raw[0]) if isinstance(prompt_len_raw, (list, tuple)) else (
        int(prompt_len_raw) if prompt_len_raw is not None else None
    )

    # Prompt-source precedence: explicit ids > explicit text > question(+options).
    prompt_source: dict = {}
    for k in ("prompt_ids", "prompt_tokens", "input_ids"):
        v = _meta_json(meta, k)
        if v is not None:
            prompt_source = {"kind": "ids", "ids": list(v)}
            break
    if not prompt_source:
        for k in ("prompt", "prompt_text", "input_text"):
            v = _meta_json(meta, k)
            if isinstance(v, str) and v:
                prompt_source = {"kind": "text", "text": v}
                break
    if not prompt_source:
        q = _meta_json(meta, "question")
        opts = _meta_json(meta, "options")
        if isinstance(q, str) and q:
            text = q if not opts else q + "\n" + (
                "\n".join(opts) if isinstance(opts, list) else str(opts)
            )
            prompt_source = {"kind": "text", "text": text, "assembled_from": "question+options"}
    if not prompt_source:
        raise SystemExit(
            f"{path}: no prompt ids/text in metadata (looked for prompt_ids/prompt_tokens/"
            f"input_ids, prompt/prompt_text/input_text, question/options)"
        )
    return n_roll, resp_lens, starts, resp_tokens, prompt_len, prompt_source


def resolve_prompt_ids(prompt_source: dict, prompt_len: int | None, tokenizer, path: str) -> list[int]:
    """Materialize the prompt token ids and HARD-FAIL if their length disagrees with
    the metadata prompt_len — this is the prompt-side alignment oracle."""
    if prompt_source["kind"] == "ids":
        ids = [int(x) for x in prompt_source["ids"]]
    else:
        ids = tokenizer(prompt_source["text"], add_special_tokens=True)["input_ids"]
        ids = [int(x) for x in ids]
    if prompt_len is not None and len(ids) != int(prompt_len):
        raise SystemExit(
            f"{path}: reconstructed prompt has {len(ids)} tokens but metadata prompt_len="
            f"{int(prompt_len)} (source={prompt_source.get('kind')}"
            f"{'/'+prompt_source['assembled_from'] if 'assembled_from' in prompt_source else ''}). "
            f"The prompt template used at generation must be supplied — refusing to emit."
        )
    return ids


def discover_gate_modules(model, layers: list[int], n_experts: int) -> dict[int, tuple[str, object]]:
    """Find each layer's MoE gate/router Linear by name+shape and return {L: (path, module)}.

    Matches modules under `...layers.{L}.` whose name mentions gate/router/route and whose
    out_features == n_experts. Prints the discovered path per layer (module-path discovery
    is intentionally structural, not hard-coded, so it survives naming differences)."""
    import torch.nn as nn

    found: dict[int, tuple[str, object]] = {}
    name_re = {L: re.compile(rf"(^|\.)layers\.{L}\.") for L in layers}
    gate_re = re.compile(r"(gate|router|route)")
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        out_features = int(mod.weight.shape[0])
        if out_features != n_experts:
            continue
        if not gate_re.search(name.split(".")[-1]) and not gate_re.search(name):
            continue
        for L, rx in name_re.items():
            if L in found:
                continue
            if rx.search(name):
                found[L] = (name, mod)
    missing = [L for L in layers if L not in found]
    if missing:
        raise SystemExit(
            f"could not locate a gate Linear (out_features={n_experts}) for layers {missing}; "
            f"inspect model.named_modules() — is this an MoE checkpoint / correct --n-experts?"
        )
    for L in layers:
        log(f"L{L}: router module = {found[L][0]}  (out_features={n_experts})")
    return found


def topk_from_logits(logits, k: int, norm_topk: bool):
    """Replicate Qwen3-MoE routing: softmax over all experts, take top-k, optionally
    renormalize the selected weights to sum 1. Returns (idx[T,k] i16, w[T,k] f16) numpy."""
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    w, idx = torch.topk(probs, k, dim=-1)
    if norm_topk:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return (
        idx.to(torch.int16).cpu().numpy(),
        w.to(torch.float16).cpu().numpy(),
    )


def _pad_batch(prompt_ids, resp_tokens, group, tokenizer):
    """Right-pad a group of rollouts into (input_ids_np, attn_np, seq_lens); real tokens
    occupy positions [0,len) so causal masking keeps their routing independent of the pad."""
    seqs = [prompt_ids + list(resp_tokens[r]) for r in group]
    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    batch = np.full((len(seqs), maxlen), pad_id, dtype=np.int64)
    attn = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for b, s in enumerate(seqs):
        batch[b, : len(s)] = s
        attn[b, : len(s)] = 1
    return batch, attn, maxlen


def capture_shard(model, tokenizer, path, layers, gate_mods, k, norm_topk, batch_rollouts, device):
    """Teacher-force one shard's rollouts and return per-layer (expert_idx, weight) arrays
    of shape [shard_rows, k], placed by offsets so row == offset_start + t."""
    import torch

    n_roll, resp_lens, starts, resp_tokens, prompt_len, prompt_source = read_shard_meta(
        path, layers[0]
    )
    prompt_ids = resolve_prompt_ids(prompt_source, prompt_len, tokenizer, path)
    shard_rows = int(starts[-1] + resp_lens[-1])

    exp_out = {L: np.zeros((shard_rows, k), dtype=EXPERT_DTYPE) for L in layers}
    w_out = {L: np.zeros((shard_rows, k), dtype=WEIGHT_DTYPE) for L in layers}
    # Router-logit hooks: each forward re-fills `captured[L]` with the gate output.
    captured: dict[int, object] = {}

    def make_gate_hook(L):
        def hook(_m, _inp, out):
            captured[L] = out.detach()
        return hook

    gate_hooks = [gate_mods[L][1].register_forward_hook(make_gate_hook(L)) for L in layers]
    try:
        for i in range(0, n_roll, max(1, batch_rollouts)):
            group = list(range(i, min(i + max(1, batch_rollouts), n_roll)))
            batch, attn, maxlen = _pad_batch(prompt_ids, resp_tokens, group, tokenizer)
            input_ids = torch.tensor(batch, device=device)
            attention_mask = torch.tensor(attn, device=device)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for L in layers:
                idx_bt, w_bt = topk_from_logits(
                    captured[L].reshape(len(group), maxlen, -1), k, norm_topk
                )  # [B,S,k]
                for b, r in enumerate(group):
                    R = int(resp_lens[r])
                    seg = slice(prompt_len, prompt_len + R)
                    dst = slice(int(starts[r]), int(starts[r]) + R)
                    exp_out[L][dst] = idx_bt[b, seg]
                    w_out[L][dst] = w_bt[b, seg]
            del input_ids, attention_mask
    finally:
        for hk in gate_hooks:
            hk.remove()
    return exp_out, w_out, shard_rows


def verify_shard(model, tokenizer, path, layers, batch_rollouts, device):
    """Alignment oracle: teacher-force one shard with output_hidden_states, and for each
    layer L compare the RESPONSE-segment hidden states to the shard's own acts_L{L}.

    We test both residual conventions — hidden_states[L] (block-L input) and
    hidden_states[L+1] (block-L output) — pick the better per token, and require the
    median per-token cosine to exceed VERIFY_COSINE_FLOOR. The winning index is printed so
    the operator learns the exact convention `acts_L{L}` was captured under. Comparison is
    through an f16 cast (the bank dtype)."""
    import torch
    from safetensors import safe_open

    n_roll, resp_lens, starts, resp_tokens, prompt_len, prompt_source = read_shard_meta(
        path, layers[0]
    )
    prompt_ids = resolve_prompt_ids(prompt_source, prompt_len, tokenizer, path)
    shard_rows = int(starts[-1] + resp_lens[-1])
    # candidate residuals per layer: {L: {cand_index: [shard_rows, d] f32}}
    cand = {L: {} for L in layers}
    for i in range(0, n_roll, max(1, batch_rollouts)):
        group = list(range(i, min(i + max(1, batch_rollouts), n_roll)))
        batch, attn, maxlen = _pad_batch(prompt_ids, resp_tokens, group, tokenizer)
        input_ids = torch.tensor(batch, device=device)
        attention_mask = torch.tensor(attn, device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layers+1
        for L in layers:
            for ci in (L, L + 1):
                if ci >= len(hs):
                    continue
                h = hs[ci].reshape(len(group), maxlen, -1).float().cpu().numpy()
                if ci not in cand[L]:
                    cand[L][ci] = np.zeros((shard_rows, h.shape[-1]), dtype=np.float32)
                for b, r in enumerate(group):
                    R = int(resp_lens[r])
                    cand[L][ci][int(starts[r]) : int(starts[r]) + R] = h[b, prompt_len : prompt_len + R]
        del input_ids, attention_mask, out

    with safe_open(path, framework="pt", device="cpu") as f:
        for L in layers:
            key = f"acts_L{L}"
            if key not in set(f.keys()):
                raise SystemExit(f"{path}: verify needs tensor {key} (found {sorted(f.keys())})")
            acts = f.get_tensor(key).float().cpu().numpy()
            if acts.shape[0] != shard_rows:
                raise SystemExit(f"{path}: {key} has {acts.shape[0]} rows, offsets imply {shard_rows}")
            best_idx, best_med = None, -1.0
            for ci, got in cand[L].items():
                gf = got.astype(np.float16).astype(np.float32)
                num = (gf * acts).sum(1)
                den = np.linalg.norm(gf, axis=1) * np.linalg.norm(acts, axis=1)
                med = float(np.median(num / np.clip(den, 1e-12, None)))
                log(f"VERIFY L{L}: median cosine(hidden_states[{ci}], acts_L{L}) = {med:.5f}")
                if med > best_med:
                    best_idx, best_med = ci, med
            role = "block-output" if best_idx == L + 1 else "block-input"
            log(f"VERIFY L{L}: best = hidden_states[{best_idx}] ({role}), cosine {best_med:.5f} "
                f"over {shard_rows} rows (floor {VERIFY_COSINE_FLOOR})")
            if best_med <= VERIFY_COSINE_FLOOR:
                raise SystemExit(
                    f"VERIFY FAILED L{L}: best cosine {best_med:.5f} <= {VERIFY_COSINE_FLOOR}. "
                    f"Neither block-input nor block-output matches acts_L{L} — the prompt "
                    f"reconstruction or the bank's residual convention differs; refusing to run."
                )
    log("VERIFY OK: teacher-forced residuals match the banks; router alignment is sound.")


def load_split_plan(split_manifest: str, all_shards: list[str]) -> dict[str, list[str]]:
    """Read split_manifest.json and return {'train': [shard...], 'heldout': [shard...]} as a
    WHOLE-SHARD assignment in shard order (mirroring the bank build, seed 0).

    Accepted schemas (the operator should confirm which the real manifest uses):
      * {"splits": {"train": {"shards": [...]}, "heldout": {"shards": [...]}}}
      * {"train": [...], "heldout": [...]}   (lists of shard stems or paths)
      * {"<shard>": "train"|"heldout", ...}  (per-shard label map)
    Shard entries may be stems ('q0250') or file paths; both are normalized to stems.
    HARD-FAILS if the partition does not exactly cover `all_shards`."""
    m = json.load(open(split_manifest))

    def stem(x: str) -> str:
        return os.path.splitext(os.path.basename(str(x)))[0]

    plan: dict[str, list[str]] = {"train": [], "heldout": []}
    if isinstance(m.get("splits"), dict):
        for split in ("train", "heldout"):
            entry = m["splits"].get(split, {})
            shards = entry.get("shards", entry) if isinstance(entry, dict) else entry
            plan[split] = [stem(s) for s in shards]
    elif "train" in m and "heldout" in m:
        plan["train"] = [stem(s) for s in m["train"]]
        plan["heldout"] = [stem(s) for s in m["heldout"]]
    else:  # per-shard label map
        for shard, label in m.items():
            if label in plan:
                plan[label].append(stem(shard))
    order = {s: i for i, s in enumerate(all_shards)}
    for split in plan:
        plan[split] = sorted(set(plan[split]), key=lambda s: order.get(s, 1 << 30))
    covered = set(plan["train"]) | set(plan["heldout"])
    if covered != set(all_shards) or (set(plan["train"]) & set(plan["heldout"])):
        raise SystemExit(
            f"split_manifest does not cleanly partition the {len(all_shards)} shards: "
            f"train={len(plan['train'])} heldout={len(plan['heldout'])} "
            f"union_covers={len(covered)} overlap={len(set(plan['train'])&set(plan['heldout']))}. "
            f"Confirm the manifest schema (see load_split_plan docstring)."
        )
    log(f"split plan: {len(plan['train'])} train shards, {len(plan['heldout'])} heldout shards")
    return plan


def per_shard_paths(out_dir: str, shard: str, L: int, k: int) -> tuple[str, str]:
    d = os.path.join(out_dir, "per_shard")
    return (
        os.path.join(d, f"{shard}_expert_L{L}_top{k}.i16.npy"),
        os.path.join(d, f"{shard}_weights_L{L}_top{k}.f16.npy"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--layers", default="11,17,23")
    ap.add_argument("--topk", type=int, default=4, help="experts stored per token (top-k of the router softmax)")
    ap.add_argument("--n-experts", type=int, default=0,
                    help="router expert count; 0 = read from model config (num_experts / num_local_experts)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-rollouts", type=int, default=1)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--split-manifest", default="",
                    help="split_manifest.json (default: <shards-dir>/split_manifest.json)")
    ap.add_argument("--verify", action="store_true",
                    help="run the acts-cosine alignment oracle on one random shard before the full run")
    ap.add_argument("--verify-shard", default="", help="pin the --verify shard (default: random, seed 0)")
    ap.add_argument("--dump-meta", default="",
                    help="print one shard's metadata keys + tensor shapes and exit (schema check)")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(os.path.join(args.out_dir, "per_shard"), exist_ok=True)
    all_shards = shard_ids(args.shards_dir)
    split_manifest = args.split_manifest or os.path.join(args.shards_dir, "split_manifest.json")

    if args.dump_meta:
        from safetensors import safe_open

        p = args.dump_meta if os.path.isabs(args.dump_meta) else os.path.join(
            args.shards_dir, args.dump_meta if args.dump_meta.endswith(".safetensors")
            else args.dump_meta + ".safetensors")
        with safe_open(p, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            log(f"{p}")
            log(f"tensors: " + ", ".join(f"{kk}{tuple(f.get_slice(kk).get_shape())}" for kk in f.keys()))
            for kk in sorted(meta):
                v = meta[kk]
                log(f"meta[{kk}] = {v[:200] + '...' if len(v) > 200 else v}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    log(f"loading tokenizer + model from {args.model_path} ({args.dtype}, device_map={args.device_map})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=True, use_cache=False,
    )
    model.eval()
    cfg = model.config
    n_experts = args.n_experts or int(
        getattr(cfg, "num_experts", 0) or getattr(cfg, "num_local_experts", 0)
    )
    if n_experts <= 0:
        raise SystemExit("could not infer expert count from config; pass --n-experts")
    norm_topk = bool(getattr(cfg, "norm_topk_prob", True))
    model_k = int(getattr(cfg, "num_experts_per_tok", args.topk))
    if args.topk != model_k:
        log(f"NOTE: storing top-{args.topk} of the router softmax; the model routes to "
            f"top-{model_k} (num_experts_per_tok). top-{model_k} == the actually-active experts.")
    device = next(model.parameters()).device
    gate_mods = discover_gate_modules(model, layers, n_experts)

    if args.verify:
        rng = np.random.default_rng(0)
        vshard = args.verify_shard or all_shards[int(rng.integers(len(all_shards)))]
        vpath = os.path.join(args.shards_dir, vshard + ".safetensors")
        log(f"VERIFY on shard {vshard}")
        verify_shard(model, tokenizer, vpath, layers, args.batch_rollouts, device)

    # ---- full capture, per shard (resumable: skip shards already written) ----
    for si, shard in enumerate(all_shards):
        path = os.path.join(args.shards_dir, shard + ".safetensors")
        done = all(os.path.exists(per_shard_paths(args.out_dir, shard, L, args.topk)[0]) and
                   os.path.exists(per_shard_paths(args.out_dir, shard, L, args.topk)[1])
                   for L in layers)
        if done:
            log(f"[{si + 1}/{len(all_shards)}] {shard}: per-shard sidecar exists, skipping")
            continue
        exp_out, w_out, shard_rows = capture_shard(
            model, tokenizer, path, layers, gate_mods, args.topk, norm_topk,
            args.batch_rollouts, device,
        )
        for L in layers:
            ep, wp = per_shard_paths(args.out_dir, shard, L, args.topk)
            np.save(ep, exp_out[L])
            np.save(wp, w_out[L])
        log(f"[{si + 1}/{len(all_shards)}] {shard}: wrote {shard_rows} rows x {len(layers)} layers")
        del exp_out, w_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- concatenate in shard order, then split _train/_heldout per manifest ----
    plan = load_split_plan(split_manifest, all_shards)
    for L in layers:
        full_e, full_w = [], []
        split_e = {"train": [], "heldout": []}
        split_w = {"train": [], "heldout": []}
        for shard in all_shards:
            ep, wp = per_shard_paths(args.out_dir, shard, L, args.topk)
            e = np.load(ep)
            w = np.load(wp)
            full_e.append(e)
            full_w.append(w)
            split = "train" if shard in set(plan["train"]) else "heldout"
            split_e[split].append(e)
            split_w[split].append(w)
        cat_e = np.concatenate(full_e, axis=0)
        cat_w = np.concatenate(full_w, axis=0)
        np.save(os.path.join(args.out_dir, f"expert_L{L}_top{args.topk}.i16.npy"), cat_e)
        np.save(os.path.join(args.out_dir, f"weights_L{L}_top{args.topk}.f16.npy"), cat_w)
        for split in ("train", "heldout"):
            se = np.concatenate(split_e[split], axis=0) if split_e[split] else np.zeros((0, args.topk), EXPERT_DTYPE)
            sw = np.concatenate(split_w[split], axis=0) if split_w[split] else np.zeros((0, args.topk), WEIGHT_DTYPE)
            np.save(os.path.join(args.out_dir, f"expert_L{L}_top{args.topk}_{split}.i16.npy"), se)
            np.save(os.path.join(args.out_dir, f"weights_L{L}_top{args.topk}_{split}.f16.npy"), sw)
            log(f"L{L} {split}: {se.shape[0]} rows -> expert_L{L}_top{args.topk}_{split}.i16.npy")
        log(f"L{L} full: {cat_e.shape[0]} rows")

    log("DONE")


if __name__ == "__main__":
    main()
