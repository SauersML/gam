#!/usr/bin/env python3
"""Harvest real Qwen QK/OV observations for attention-kernel fitting.

Run this on MSI. The script uses real Qwen weights, constructs repeated-token
positional prompts, computes selected heads' pre-softmax QK scores, and records
an OV-induced circular coordinate shift using a linear phase probe fit on the
same layer inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TAU = 2.0 * math.pi


def log(message: str) -> None:
    print(f"[attn-real] {message}", flush=True)


def parse_heads(raw: str) -> list[int]:
    heads = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not heads:
        raise SystemExit("at least one head is required")
    return heads


def circular_delta(out_t: np.ndarray, in_t: np.ndarray) -> np.ndarray:
    return (out_t - in_t + 0.5) % 1.0 - 0.5


def r2_score(y: np.ndarray, fitted: np.ndarray) -> float:
    resid = y - fitted
    sse = float(np.sum(resid * resid))
    centered = y - y.mean(axis=0, keepdims=True)
    sst = float(np.sum(centered * centered))
    if sst <= 0.0:
        return 1.0 if sse == 0.0 else 0.0
    return 1.0 - sse / sst


def single_token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    if not ids:
        raise SystemExit("none of the candidate token texts encoded as a single token")
    return ids


def qwen_model_body(model: Any) -> Any:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise SystemExit("could not locate the model body on the loaded Qwen model")


def apply_qwen_rope(model_body: Any, q: Any, k: Any, hidden: Any, position_ids: Any) -> tuple[Any, Any]:
    import torch

    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except Exception:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    rotary = model_body.rotary_emb
    try:
        cos, sin = rotary(hidden, position_ids)
    except TypeError:
        seq_len = int(position_ids.max().item()) + 1
        cos, sin = rotary(q, seq_len=seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    if not torch.isfinite(q_rot).all() or not torch.isfinite(k_rot).all():
        raise SystemExit("RoPE produced non-finite QK tensors")
    return q_rot, k_rot


def repeat_kv(x: Any, n_rep: int) -> Any:
    if n_rep == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    expanded = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def build_inputs(args: argparse.Namespace, tokenizer: Any, torch: Any) -> tuple[Any, Any, list[int]]:
    token_texts = [part for part in args.token_texts.split("|") if part]
    token_ids = single_token_ids(tokenizer, token_texts)
    rows = []
    for token_id in token_ids:
        rows.append([token_id] * args.seq_len)
    input_ids = torch.tensor(rows, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    log(f"using {len(token_ids)} single-token prompts over seq_len={args.seq_len}: {token_ids}")
    return input_ids, attention_mask, token_ids


def fit_phase_probe(hidden: np.ndarray, t: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.column_stack([np.cos(TAU * t), np.sin(TAU * t)])
    x = hidden - hidden.mean(axis=0, keepdims=True)
    centered_y = y - y.mean(axis=0, keepdims=True)
    gram = x @ x.T
    scale = float(np.trace(gram) / max(1, gram.shape[0]))
    penalty = ridge * max(scale, 1.0)
    alpha = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), centered_y)
    beta = x.T @ alpha
    intercept = y.mean(axis=0) - hidden.mean(axis=0) @ beta
    fitted = hidden @ beta + intercept
    return beta, intercept, r2_score(y, fitted)


def phase_from_probe(x: np.ndarray, beta: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    projected = x @ beta + intercept
    return (np.arctan2(projected[:, 1], projected[:, 0]) / TAU) % 1.0


def head_ov_vectors(v_head: np.ndarray, o_weight: np.ndarray, head: int, head_dim: int) -> np.ndarray:
    start = head * head_dim
    stop = start + head_dim
    block = o_weight[:, start:stop]
    return v_head @ block.T


def summarize_head(
    args: argparse.Namespace,
    layer_index: int,
    head: int,
    q: Any,
    k: Any,
    v: Any,
    attn: Any,
    hidden_np: np.ndarray,
    t_np: np.ndarray,
    phase_beta: np.ndarray,
    phase_intercept: np.ndarray,
    phase_probe_r2: float,
) -> dict[str, Any]:
    import torch

    scaling = float(getattr(attn, "scaling", 1.0 / math.sqrt(int(q.shape[-1]))))
    scores = torch.matmul(q[:, head], k[:, head].transpose(-2, -1)) * scaling
    scores_np = scores.detach().float().cpu().numpy()
    period = args.period
    score_sum = np.zeros((period, period), dtype=np.float64)
    score_count = np.zeros((period, period), dtype=np.int64)
    q_bins = (np.arange(args.seq_len) % period).astype(np.int64)
    k_bins = q_bins.copy()
    for batch_index in range(scores_np.shape[0]):
        for query_pos in range(args.seq_len):
            qb = int(q_bins[query_pos])
            for key_pos in range(query_pos + 1):
                kb = int(k_bins[key_pos])
                score_sum[qb, kb] += float(scores_np[batch_index, query_pos, key_pos])
                score_count[qb, kb] += 1
    if np.any(score_count == 0):
        raise SystemExit(f"head {head}: positional grid has empty QK cells")
    mean_scores = score_sum / score_count

    o_weight = attn.o_proj.weight.detach().float().cpu().numpy()
    head_dim = int(q.shape[-1])
    ov = head_ov_vectors(v[:, head].detach().float().cpu().numpy().reshape(-1, head_dim), o_weight, head, head_dim)
    ov_t = phase_from_probe(ov, phase_beta, phase_intercept)
    key_t = np.tile(t_np, scores_np.shape[0])
    delta = circular_delta(ov_t, key_t)
    delta_by_bin = np.zeros(period, dtype=np.float64)
    count_by_bin = np.zeros(period, dtype=np.int64)
    for index, bin_index in enumerate(np.tile(k_bins, scores_np.shape[0])):
        delta_by_bin[int(bin_index)] += float(delta[index])
        count_by_bin[int(bin_index)] += 1
    if np.any(count_by_bin == 0):
        raise SystemExit(f"head {head}: positional grid has empty OV cells")
    delta_by_bin /= count_by_bin

    return {
        "layer": int(layer_index),
        "head": int(head),
        "period": int(period),
        "query_t": [float(i / period) for i in range(period)],
        "key_t": [float(i / period) for i in range(period)],
        "scores": mean_scores.tolist(),
        "ov_key_t": [float(i / period) for i in range(period)],
        "ov_delta_t": delta_by_bin.tolist(),
        "phase_probe_r2": float(phase_probe_r2),
        "qk_cells_per_head": int(period * period),
        "qk_observations": int(score_count.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--heads", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--period", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument(
        "--token-texts",
        default=" the| of| and| to| in| a| is| that",
        help="pipe-separated candidate texts; only single-token encodings are used",
    )
    args = parser.parse_args()
    if args.seq_len < args.period * 2:
        raise SystemExit("--seq-len must cover at least two periods")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    heads = parse_heads(args.heads)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    model_body = qwen_model_body(model)
    layer = model_body.layers[args.layer]
    attn = layer.self_attn

    input_ids, attention_mask, token_ids = build_inputs(args, tokenizer, torch)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = torch.arange(args.seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        layer_input = out.hidden_states[args.layer]
        attn_input = layer.input_layernorm(layer_input)
        q = attn.q_proj(attn_input)
        k = attn.k_proj(attn_input)
        v = attn.v_proj(attn_input)
        batch, seq_len, _width = q.shape
        num_heads = int(getattr(attn, "num_heads", getattr(attn, "num_attention_heads", model.config.num_attention_heads)))
        num_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(model.config, "num_key_value_heads", num_heads)))
        head_dim = int(getattr(attn, "head_dim", q.shape[-1] // num_heads))
        if any(head < 0 or head >= num_heads for head in heads):
            raise SystemExit(f"requested heads {heads} outside [0, {num_heads})")
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)
        q, k = apply_qwen_rope(model_body, q, k, attn_input, position_ids)
        kv_groups = num_heads // num_kv_heads
        k = repeat_kv(k, kv_groups)
        v = repeat_kv(v, kv_groups)

    hidden_np = attn_input.detach().float().cpu().numpy().reshape(batch * args.seq_len, -1)
    t_np = (np.arange(args.seq_len, dtype=np.float64) % args.period) / float(args.period)
    tiled_t = np.tile(t_np, batch)
    phase_beta, phase_intercept, phase_probe_r2 = fit_phase_probe(hidden_np, tiled_t, args.ridge)
    log(f"layer {args.layer}: phase probe R2={phase_probe_r2:.6f}")

    head_rows = [
        summarize_head(
            args,
            args.layer,
            head,
            q,
            k,
            v,
            attn,
            hidden_np,
            t_np,
            phase_beta,
            phase_intercept,
            phase_probe_r2,
        )
        for head in heads
    ]
    payload = {
        "experiment": "attn_real_qwen3_8b_qk_ov",
        "model": args.model,
        "layer": int(args.layer),
        "heads": heads,
        "seq_len": int(args.seq_len),
        "period": int(args.period),
        "token_ids": token_ids,
        "phase_probe_r2": float(phase_probe_r2),
        "observations": head_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
