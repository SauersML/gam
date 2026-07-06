#!/usr/bin/env python3
"""Matched-token cross-model transport experiment.

Run this on an MSI GPU node. It aligns one WikiText stream by exact tokenizer
character spans, harvests one residual-stream layer from each model for the
same text spans in the same order, peels the top PCA direction inside each
model, and reports the top-2 activation-plane circle transport.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TAU = 2.0 * math.pi
DEGREE_CANDIDATES = (-2, -1, 0, 1, 2)


@dataclass
class DocPlan:
    text: str
    spans: list[tuple[int, int]]
    positions_a: list[int]
    positions_b: list[int]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def span_map(offsets: list[tuple[int, int]], text: str) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for pos, raw_span in enumerate(offsets):
        start, end = int(raw_span[0]), int(raw_span[1])
        if end <= start:
            continue
        if text[start:end].strip() == "":
            continue
        out.setdefault((start, end), pos)
    return out


def build_plans(args: argparse.Namespace, tok_a, tok_b) -> list[DocPlan]:
    from datasets import load_dataset

    plans: list[DocPlan] = []
    matched = 0
    scanned_docs = 0
    ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    for row in ds:
        raw = row.get(args.text_field) or row.get("content") or ""
        text = raw.strip()
        if len(text) < args.min_chars:
            continue
        enc_a = tok_a(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        enc_b = tok_b(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        map_a = span_map(enc_a["offset_mapping"], text)
        map_b = span_map(enc_b["offset_mapping"], text)
        common = sorted(set(map_a).intersection(map_b))
        if not common:
            continue
        remaining = args.n_matched - matched
        common = common[:remaining]
        plans.append(
            DocPlan(
                text=text,
                spans=common,
                positions_a=[map_a[s] for s in common],
                positions_b=[map_b[s] for s in common],
            )
        )
        matched += len(common)
        scanned_docs += 1
        if matched >= args.n_matched:
            break
    if matched < args.n_matched:
        raise SystemExit(f"only found {matched} matched token spans")
    log(f"aligned {matched} matched token spans from {scanned_docs} WikiText documents")
    return plans


def hidden_size_from_config(model) -> int:
    config = getattr(model, "config")
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    text_config = getattr(config, "text_config")
    return int(text_config.hidden_size)


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def harvest_model(
    model_path: str,
    tokenizer,
    plans: list[DocPlan],
    positions_attr: str,
    layer: int,
    batch_docs: int,
    label: str,
) -> np.ndarray:
    import torch

    log(f"loading {label}: {model_path}")
    model = load_model(model_path)
    d_model = hidden_size_from_config(model)
    total = sum(len(p.spans) for p in plans)
    acts = np.empty((total, d_model), dtype=np.float32)
    cursor = 0
    with torch.inference_mode():
        for start in range(0, len(plans), batch_docs):
            batch = plans[start : start + batch_docs]
            enc = tokenizer(
                [p.text for p in batch],
                add_special_tokens=False,
                truncation=True,
                max_length=max(max(getattr(p, positions_attr)) + 1 for p in batch),
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[layer].detach().float().cpu()
            for row, plan in enumerate(batch):
                pos = getattr(plan, positions_attr)
                take = hidden[row, pos, :].numpy()
                acts[cursor : cursor + len(pos)] = take
                cursor += len(pos)
            if cursor % 5000 < max(len(batch), 512):
                log(f"{label}: harvested {cursor}/{total} matched rows")
            del out, hidden, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acts


def randomized_scores_after_top_peel(x: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    centered = x.astype(np.float64, copy=False)
    mean = centered.mean(axis=0, keepdims=True)
    centered = centered - mean
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], 16))
    y = centered @ omega
    q, unused_r = np.linalg.qr(y, mode="reduced")
    q_norm = float(np.linalg.norm(q[:, 0]))
    b = q.T @ centered
    unused_u, singular, vt = np.linalg.svd(b, full_matrices=False)
    comps = vt[:3]
    scores = centered @ comps.T
    total_energy = float(np.sum(centered * centered))
    top_frac = float((singular[0] * singular[0]) / max(total_energy, 1e-30))
    score_norm = float(np.linalg.norm(scores[:, 1:3]))
    meta = {
        "mean_norm": float(np.linalg.norm(mean)),
        "top_pc_variance_fraction": top_frac,
        "score_plane_norm": score_norm,
        "qr_first_column_norm": q_norm,
    }
    return scores[:, 1:3], meta


def wrap_tau(values: np.ndarray) -> np.ndarray:
    return np.mod(values, TAU)


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % TAU - math.pi


def resultant(angles: np.ndarray) -> tuple[float, float]:
    c = float(np.cos(angles).mean())
    s = float(np.sin(angles).mean())
    return math.hypot(c, s), math.atan2(s, c)


def fit_fourier_derivative_defect(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    degree: int,
    phase: float,
) -> dict[str, float]:
    residual = wrap_pi(theta_b - degree * theta_a - phase)
    cols = [np.ones_like(theta_a)]
    deriv_cols = [np.zeros_like(theta_a)]
    for harmonic in range(1, 11):
        angle = harmonic * theta_a
        cols.extend([np.cos(angle), np.sin(angle)])
        deriv_cols.extend([-harmonic * np.sin(angle), harmonic * np.cos(angle)])
    xmat = np.column_stack(cols)
    dmat = np.column_stack(deriv_cols)
    lam = 1e-4 * theta_a.shape[0]
    penalty = np.diag([0.0] + [1.0] * (xmat.shape[1] - 1))
    beta = np.linalg.solve(xmat.T @ xmat + lam * penalty, xmat.T @ residual)
    fitted = xmat @ beta
    derivative = degree + dmat @ beta
    gaps = np.abs(derivative) - 1.0
    return {
        "smooth_isometry_defect": float(np.mean(gaps * gaps)),
        "smooth_residual_rms": float(np.sqrt(np.mean((residual - fitted) ** 2))),
        "smooth_min_directional_derivative": float(np.min(np.sign(degree or 1) * derivative)),
        "smooth_fourier_harmonics": 10,
        "smooth_ridge_lambda": float(lam),
    }


def transport_report(theta_a: np.ndarray, theta_b: np.ndarray) -> dict[str, float | int | str]:
    best_degree = DEGREE_CANDIDATES[0]
    best_resultant = -1.0
    best_phase = 0.0
    for degree in DEGREE_CANDIDATES:
        r, phase = resultant(theta_b - degree * theta_a)
        if r > best_resultant:
            best_degree = degree
            best_resultant = r
            best_phase = phase
    shift_r, shift_phase = resultant(theta_b - theta_a)
    reflect_r, reflect_phase = resultant(theta_b + theta_a)
    if shift_r >= reflect_r:
        winding = 1
        phase = shift_phase
        o2_resultant = shift_r
        other = reflect_r
        rigid_class = "shift"
    else:
        winding = -1
        phase = reflect_phase
        o2_resultant = reflect_r
        other = shift_r
        rigid_class = "reflect"
    gauge_scale = 2.0 / math.sqrt(theta_a.shape[0])
    if o2_resultant - other <= gauge_scale:
        rigid_class = "mixing"
    o2_defect = 1.0 - o2_resultant
    smooth = fit_fourier_derivative_defect(theta_a, theta_b, best_degree, best_phase)
    topology_preserved = best_degree in (-1, 1) and smooth["smooth_min_directional_derivative"] > 0.0
    if rigid_class == "mixing" or winding not in (-1, 1):
        verdict = "not a shared feature by this coordinate"
    elif o2_defect <= gauge_scale and smooth["smooth_isometry_defect"] <= gauge_scale:
        verdict = "consistent with a shared feature within noise/gauge"
    else:
        verdict = "shared feature with measured reparameterization"
    report: dict[str, float | int | str] = {
        "n_matched": int(theta_a.shape[0]),
        "degree": int(best_degree),
        "degree_concentration": float(best_resultant),
        "degree_phase": float(best_phase),
        "winding": int(winding),
        "phase": float(phase),
        "phase_degrees": float(phase * 180.0 / math.pi),
        "o2_defect": float(o2_defect),
        "o2_resultant_shift": float(shift_r),
        "o2_resultant_reflect": float(reflect_r),
        "gauge_defect_scale": float(gauge_scale),
        "rigid_class": rigid_class,
        "topology_preserved": bool(topology_preserved),
        "verdict": verdict,
    }
    report.update(smooth)
    return report


def write_token_sample(path: Path, plans: list[DocPlan], limit: int) -> None:
    rows: list[dict[str, int | str]] = []
    for doc_idx, plan in enumerate(plans):
        for span in plan.spans:
            if len(rows) >= limit:
                path.write_text(json.dumps(rows, indent=2))
                return
            rows.append(
                {
                    "doc": doc_idx,
                    "start": span[0],
                    "end": span[1],
                    "text": plan.text[span[0] : span[1]],
                }
            )
    path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default="Qwen3-8B L18")
    parser.add_argument("--label-b", default="Qwen3.6-35B-A3B L20")
    parser.add_argument("--layer-a", type=int, default=18)
    parser.add_argument("--layer-b", type=int, default=20)
    parser.add_argument("--n-matched", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-docs-a", type=int, default=4)
    parser.add_argument("--batch-docs-b", type=int, default=1)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("loading tokenizers")
    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
    plans = build_plans(args, tok_a, tok_b)
    write_token_sample(out_dir / "matched_token_sample.json", plans, 128)

    acts_a = harvest_model(
        args.model_a,
        tok_a,
        plans,
        "positions_a",
        args.layer_a,
        args.batch_docs_a,
        args.label_a,
    )
    np.save(out_dir / "acts_a.npy", acts_a)
    acts_b = harvest_model(
        args.model_b,
        tok_b,
        plans,
        "positions_b",
        args.layer_b,
        args.batch_docs_b,
        args.label_b,
    )
    np.save(out_dir / "acts_b.npy", acts_b)

    scores_a, pca_a = randomized_scores_after_top_peel(acts_a, seed=17)
    scores_b, pca_b = randomized_scores_after_top_peel(acts_b, seed=23)
    theta_a = wrap_tau(np.arctan2(scores_a[:, 1], scores_a[:, 0]))
    theta_b = wrap_tau(np.arctan2(scores_b[:, 1], scores_b[:, 0]))
    report = transport_report(theta_a, theta_b)

    np.save(out_dir / "theta_a.npy", theta_a.astype(np.float32))
    np.save(out_dir / "theta_b.npy", theta_b.astype(np.float32))
    payload = {
        "model_a": args.label_a,
        "model_b": args.label_b,
        "model_a_path": args.model_a,
        "model_b_path": args.model_b,
        "layer_a": args.layer_a,
        "layer_b": args.layer_b,
        "dataset": f"{args.dataset}/{args.dataset_config}/{args.split}",
        "max_length": args.max_length,
        "alignment": "exact shared tokenizer character spans in the same WikiText documents",
        "pca_a": pca_a,
        "pca_b": pca_b,
        "transport": report,
    }
    (out_dir / "numbers.json").write_text(json.dumps(payload, indent=2))
    print("RESULTS_JSON " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
