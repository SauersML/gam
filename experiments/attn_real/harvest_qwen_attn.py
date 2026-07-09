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


def circular_mean_by_bin(
    delta: np.ndarray, bins: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin circular mean of fractional-turn deltas, in ``(-0.5, 0.5]``.

    The deltas are phase shifts measured in fractions of a full turn, so they
    live on a circle: averaging them linearly collapses the seam (``-0.49`` and
    ``+0.49`` are both nearly a half turn, yet their arithmetic mean is ``0``).
    The circular mean sums the unit vectors ``exp(2*pi*i*delta)`` within each bin
    and reads the resultant angle back with ``atan2``, so the seam is respected
    and the returned representative is the shortest arc. Returns the per-bin
    circular mean alongside the per-bin observation counts so the caller can
    reject empty positional cells.
    """
    angle = TAU * delta
    sin_sum = np.zeros(period, dtype=np.float64)
    cos_sum = np.zeros(period, dtype=np.float64)
    count = np.zeros(period, dtype=np.int64)
    np.add.at(sin_sum, bins, np.sin(angle))
    np.add.at(cos_sum, bins, np.cos(angle))
    np.add.at(count, bins, 1)
    return np.arctan2(sin_sum, cos_sum) / TAU, count


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


def load_attention_weights(model_path: Path, layer: int) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    prefix = f"model.layers.{layer}.self_attn."
    keys = {
        "q_proj": prefix + "q_proj.weight",
        "k_proj": prefix + "k_proj.weight",
        "v_proj": prefix + "v_proj.weight",
        "o_proj": prefix + "o_proj.weight",
        "q_norm": prefix + "q_norm.weight",
        "k_norm": prefix + "k_norm.weight",
    }
    by_file: dict[str, list[tuple[str, str]]] = {}
    for short, key in keys.items():
        shard = weight_map.get(key)
        if shard is None:
            if short.endswith("_norm"):
                continue
            raise SystemExit(f"missing weight {key} in {index_path}")
        by_file.setdefault(shard, []).append((short, key))
    out: dict[str, np.ndarray] = {}
    for shard, shard_keys in by_file.items():
        with safe_open(str(model_path / shard), framework="pt", device="cpu") as handle:
            for short, key in shard_keys:
                out[short] = handle.get_tensor(key).float().cpu().numpy()
    return out


def rms_norm_heads(x: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
    if weight is None:
        return x
    denom = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1.0e-6)
    return (x / denom) * weight.reshape(1, 1, -1)


def peel_top_direction(x: np.ndarray, iterations: int) -> tuple[np.ndarray, float]:
    centered = x - x.mean(axis=0, keepdims=True)
    direction = centered[0].copy()
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        direction = np.ones(centered.shape[1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
    direction = direction / norm
    for _step in range(iterations):
        scores = centered @ direction
        direction = centered.T @ scores
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            break
        direction = direction / norm
    scores = centered @ direction
    removed = np.outer(scores, direction)
    total = float(np.sum(centered * centered))
    absorbed = float(np.sum(removed * removed) / max(total, 1.0e-30))
    return (x - removed).astype(np.float32), absorbed


def apply_rope_numpy(q: np.ndarray, k: np.ndarray, positions: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    head_dim = q.shape[-1]
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]

    def rotate(x: np.ndarray) -> np.ndarray:
        first = x[..., :half]
        second = x[..., half:]
        return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)

    return rotate(q), rotate(k)


def repeat_kv_numpy(x: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)


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


def summarize_head_numpy(
    args: argparse.Namespace,
    layer_index: int,
    head: int,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    o_weight: np.ndarray,
    t_np: np.ndarray,
    phase_beta: np.ndarray,
    phase_intercept: np.ndarray,
    phase_probe_r2: float,
) -> dict[str, Any]:
    scaling = 1.0 / math.sqrt(int(q.shape[-1]))
    scores_np = (q[:, head, :] @ k[:, head, :].T) * scaling
    period = args.period
    bins = (np.arange(q.shape[0]) % period).astype(np.int64)
    score_sum = np.zeros((period, period), dtype=np.float64)
    score_count = np.zeros((period, period), dtype=np.int64)
    for query_index, qb in enumerate(bins):
        for key_index, kb in enumerate(bins):
            score_sum[int(qb), int(kb)] += float(scores_np[query_index, key_index])
            score_count[int(qb), int(kb)] += 1
    if np.any(score_count == 0):
        raise SystemExit(f"head {head}: positional grid has empty QK cells")
    mean_scores = score_sum / score_count

    head_dim = int(q.shape[-1])
    ov = head_ov_vectors(v[:, head, :], o_weight, head, head_dim)
    ov_t = phase_from_probe(ov, phase_beta, phase_intercept)
    delta = circular_delta(ov_t, t_np)
    delta_by_bin, count_by_bin = circular_mean_by_bin(delta, bins, period)
    if np.any(count_by_bin == 0):
        raise SystemExit(f"head {head}: positional grid has empty OV cells")

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
    delta_by_bin, count_by_bin = circular_mean_by_bin(
        delta, np.tile(k_bins, scores_np.shape[0]), period
    )
    if np.any(count_by_bin == 0):
        raise SystemExit(f"head {head}: positional grid has empty OV cells")

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
    parser.add_argument("--resid", type=Path)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--heads", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--period", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--rope-theta", type=float, default=1000000.0)
    parser.add_argument("--sink-peel-iters", type=int, default=12)
    parser.add_argument(
        "--token-texts",
        default=" the| of| and| to| in| a| is| that",
        help="pipe-separated candidate texts; only single-token encodings are used",
    )
    args = parser.parse_args()
    if args.seq_len < args.period * 2:
        raise SystemExit("--seq-len must cover at least two periods")

    heads = parse_heads(args.heads)
    if args.resid is not None:
        run_existing_residual_harvest(args, heads)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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


def run_existing_residual_harvest(args: argparse.Namespace, heads: list[int]) -> None:
    model_path = Path(args.model)
    config = json.loads((model_path / "config.json").read_text())
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    hidden_size = int(config["hidden_size"])
    head_dim = hidden_size // num_heads
    if any(head < 0 or head >= num_heads for head in heads):
        raise SystemExit(f"requested heads {heads} outside [0, {num_heads})")
    rows = min(args.max_rows, args.seq_len * max(2, args.max_rows // max(args.seq_len, 1)))
    resid = np.load(args.resid, mmap_mode="r")
    if resid.shape[1] != hidden_size:
        raise SystemExit(f"residual width {resid.shape[1]} does not match hidden_size {hidden_size}")
    x = np.asarray(resid[:rows], dtype=np.float32)
    peeled_x, absorbed = peel_top_direction(x, args.sink_peel_iters)
    log(f"loaded {rows} rows from {args.resid}; top-direction absorbed fraction={absorbed:.6f}")
    weights = load_attention_weights(model_path, args.layer)
    q_flat = peeled_x @ weights["q_proj"].T
    k_flat = peeled_x @ weights["k_proj"].T
    v_flat = peeled_x @ weights["v_proj"].T
    q = q_flat.reshape(rows, num_heads, head_dim)
    k = k_flat.reshape(rows, num_kv_heads, head_dim)
    v = v_flat.reshape(rows, num_kv_heads, head_dim)
    q = rms_norm_heads(q, weights.get("q_norm"))
    k = rms_norm_heads(k, weights.get("k_norm"))
    positions = np.arange(rows, dtype=np.float32)
    q, k = apply_rope_numpy(q, k, positions, args.rope_theta)
    groups = num_heads // num_kv_heads
    k = repeat_kv_numpy(k, groups)
    v = repeat_kv_numpy(v, groups)
    t_np = (np.arange(rows, dtype=np.float64) % args.period) / float(args.period)
    phase_beta, phase_intercept, phase_probe_r2 = fit_phase_probe(peeled_x, t_np, args.ridge)
    log(f"row-order phase probe R2={phase_probe_r2:.6f}")
    head_rows = [
        summarize_head_numpy(
            args,
            args.layer,
            head,
            q,
            k,
            v,
            weights["o_proj"],
            t_np,
            phase_beta,
            phase_intercept,
            phase_probe_r2,
        )
        for head in heads
    ]
    payload = {
        "experiment": "attn_real_qwen3_8b_existing_residual_qk_ov",
        "model": args.model,
        "resid": str(args.resid),
        "layer": int(args.layer),
        "heads": heads,
        "rows": int(rows),
        "period": int(args.period),
        "sink_peel_absorbed_fraction": float(absorbed),
        "phase_probe_r2": float(phase_probe_r2),
        "chart": "row_index_mod_period_after_top_direction_peel",
        "observations": head_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
