#!/usr/bin/env python3
"""Harvest, fit, plot, and causally steer a real Qwen weekday circle.

This script is intended to run on MSI. It uses PyTorch only at the frozen-model
boundary: harvesting residual activations and executing the residual-stream
intervention. The circle fit itself is delegated to the Rust example
`real_circle_weekday_fit`, which uses the current gam-sae fitter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OFFSET_TEMPLATES = [
    ("Yesterday was {c}, so today is", 1),
    ("Today is {c}, so tomorrow is", 1),
    ("Today is {c}, so the day after tomorrow is", 2),
    ("Tomorrow is {c}, so today is", -1),
    ("The meeting was moved from {c} to the next day, which is", 1),
    ("If today is {c}, then in three days it will be", 3),
    ("The class always meets two days after {c}, on", 2),
    ("One week minus a day after {c} is", 6),
    ("{c} comes right before", 1),
    ("{c} comes right after", -1),
]


def build_prompts() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    prompts, labels, offsets, template_ids = [], [], [], []
    for template_idx, (template, offset) in enumerate(OFFSET_TEMPLATES):
        for context_idx, context in enumerate(WEEKDAYS):
            prompts.append(template.format(c=context))
            labels.append((context_idx + offset) % len(WEEKDAYS))
            offsets.append(offset)
            template_ids.append(template_idx)
    return (
        prompts,
        np.asarray(labels, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(template_ids, dtype=np.int64),
    )


def weekday_token_ids(tokenizer) -> tuple[list[int], list[str]]:
    ids, strings = [], []
    for weekday in WEEKDAYS:
        pieces = tokenizer(" " + weekday, add_special_tokens=False).input_ids
        if len(pieces) != 1:
            raise RuntimeError(
                f"weekday {weekday!r} tokenizes to {len(pieces)} pieces; single-token outputs are required"
            )
        ids.append(int(pieces[0]))
        strings.append(" " + weekday)
    return ids, strings


def write_csv_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for row in matrix:
            writer.writerow([f"{float(value):.12g}" for value in row])


def write_labels(path: Path, labels: np.ndarray) -> None:
    path.write_text("\n".join(str(int(value)) for value in labels) + "\n")


def load_model(model_path: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    load_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    if device != "cpu":
        model = model.to(device)
    model = model.eval()
    return tokenizer, model


def harvest(tokenizer, model, prompts: list[str], weekday_ids: list[int], layer: int, device: str):
    import torch

    activations, probs, logits = [], [], []
    started = time.time()
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            activations.append(out.hidden_states[layer][0, -1, :].float().cpu().numpy())
            last_logits = out.logits[0, -1, :].float()
            logits.append(last_logits[weekday_ids].cpu().numpy())
            full = torch.softmax(last_logits, dim=-1)
            restricted = full[weekday_ids]
            probs.append((restricted / restricted.sum()).cpu().numpy())
            if (idx + 1) % 10 == 0:
                print(f"[harvest] {idx + 1}/{len(prompts)} ({time.time() - started:.0f}s)", flush=True)
    return (
        np.asarray(activations, dtype=np.float32),
        np.asarray(probs, dtype=np.float64),
        np.asarray(logits, dtype=np.float64),
    )


def top_sink_direction(path: Path, rows: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise RuntimeError(f"{path} must be a 2-D residual array, got shape {arr.shape}")
    take = min(int(rows), int(arr.shape[0]))
    stride = max(int(arr.shape[0]) // take, 1)
    sample = np.asarray(arr[0 : stride * take : stride], dtype=np.float32)
    sample = sample[:take]
    sample = sample - sample.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(sample, full_matrices=False)
    direction = vt[0].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("sink direction SVD returned a degenerate top component")
    return direction / norm


def preprocess_activations(activations: np.ndarray, sink_direction: np.ndarray, pca_dim: int):
    peeled = activations - np.outer(activations @ sink_direction, sink_direction)
    mean = peeled.mean(axis=0, keepdims=True)
    centered = peeled - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(int(pca_dim), vt.shape[0])
    if keep < 2:
        raise RuntimeError("PCA needs at least two retained components")
    basis = vt[:keep].astype(np.float32)
    scores = centered @ basis.T
    total = float(np.sum(singular * singular))
    explained = (singular[:keep] * singular[:keep]) / total
    return scores.astype(np.float64), basis, mean.reshape(-1).astype(np.float32), explained


def run_rust_fit(repo: Path, out_dir: Path, harmonics: int, fit_iters: int) -> dict:
    cmd = [
        "cargo",
        "run",
        "-p",
        "gam-sae",
        "--example",
        "real_circle_weekday_fit",
        "--",
        str(out_dir / "activations_pca.csv"),
        str(out_dir / "weekday_probs.csv"),
        str(out_dir / "labels.csv"),
        str(out_dir),
        str(harmonics),
        str(fit_iters),
    ]
    print("[rust-fit] " + " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (out_dir / "rust_fit_stdout.txt").write_text(completed.stdout)
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Rust fit failed with exit code {completed.returncode}")
    return json.loads((out_dir / "fit_results.json").read_text())


def read_csv_matrix(path: Path) -> np.ndarray:
    rows = []
    with path.open() as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                rows.append([float(value) for value in row])
    return np.asarray(rows, dtype=np.float64)


def measure_intervention(
    tokenizer,
    model,
    prompts: list[str],
    labels: np.ndarray,
    weekday_ids: list[int],
    delta_hidden: np.ndarray,
    layer: int,
    device: str,
) -> dict:
    import torch

    if layer <= 0:
        raise RuntimeError("layer must be positive because intervention hooks the previous decoder block output")
    hook_layer = layer - 1
    clean_weekday_logits, steered_weekday_logits = [], []
    clean_weekday_probs, steered_weekday_probs = [], []
    full_kl_clean_to_steered, restricted_kl_clean_to_steered = [], []
    started = time.time()
    with torch.no_grad():
        for row, prompt in enumerate(prompts):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            clean = model(ids).logits[0, -1, :].float()
            delta = torch.tensor(delta_hidden[row], dtype=clean.dtype, device=device)

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                edited = hidden.clone()
                edited[:, -1, :] = edited[:, -1, :] + delta
                if isinstance(output, tuple):
                    return (edited,) + output[1:]
                return edited

            handle = model.model.layers[hook_layer].register_forward_hook(hook)
            try:
                steered = model(ids).logits[0, -1, :].float()
            finally:
                handle.remove()

            clean_probs = torch.softmax(clean, dim=-1)
            steered_probs = torch.softmax(steered, dim=-1)
            clean_w = clean_probs[weekday_ids]
            steered_w = steered_probs[weekday_ids]
            clean_wr = clean_w / clean_w.sum()
            steered_wr = steered_w / steered_w.sum()
            full_kl = torch.sum(clean_probs * (torch.log(clean_probs.clamp_min(1e-45)) - torch.log(steered_probs.clamp_min(1e-45))))
            restricted_kl = torch.sum(clean_wr * (torch.log(clean_wr.clamp_min(1e-45)) - torch.log(steered_wr.clamp_min(1e-45))))

            clean_weekday_logits.append(clean[weekday_ids].cpu().numpy())
            steered_weekday_logits.append(steered[weekday_ids].cpu().numpy())
            clean_weekday_probs.append(clean_wr.cpu().numpy())
            steered_weekday_probs.append(steered_wr.cpu().numpy())
            full_kl_clean_to_steered.append(float(full_kl.cpu()))
            restricted_kl_clean_to_steered.append(float(restricted_kl.cpu()))
            if (row + 1) % 10 == 0:
                print(f"[intervention] {row + 1}/{len(prompts)} ({time.time() - started:.0f}s)", flush=True)

    clean_logits = np.asarray(clean_weekday_logits, dtype=np.float64)
    steered_logits = np.asarray(steered_weekday_logits, dtype=np.float64)
    clean_probs = np.asarray(clean_weekday_probs, dtype=np.float64)
    steered_probs = np.asarray(steered_weekday_probs, dtype=np.float64)
    next_labels = (labels + 1) % len(WEEKDAYS)
    rows = np.arange(labels.shape[0])
    next_logit_margin = (steered_logits[rows, next_labels] - steered_logits[rows, labels]) - (
        clean_logits[rows, next_labels] - clean_logits[rows, labels]
    )
    next_prob_delta = steered_probs[rows, next_labels] - clean_probs[rows, next_labels]
    current_prob_delta = steered_probs[rows, labels] - clean_probs[rows, labels]
    records = []
    for row in rows:
        records.append(
            {
                "row": int(row),
                "label": int(labels[row]),
                "weekday": WEEKDAYS[int(labels[row])],
                "target_label": int(next_labels[row]),
                "target_weekday": WEEKDAYS[int(next_labels[row])],
                "next_minus_current_logit_shift": float(next_logit_margin[row]),
                "target_prob_delta": float(next_prob_delta[row]),
                "source_prob_delta": float(current_prob_delta[row]),
                "restricted_kl_clean_to_steered": float(restricted_kl_clean_to_steered[row]),
                "full_kl_clean_to_steered": float(full_kl_clean_to_steered[row]),
            }
        )
    return {
        "mean_next_minus_current_logit_shift": float(next_logit_margin.mean()),
        "median_next_minus_current_logit_shift": float(np.median(next_logit_margin)),
        "mean_target_prob_delta": float(next_prob_delta.mean()),
        "mean_source_prob_delta": float(current_prob_delta.mean()),
        "mean_restricted_kl_clean_to_steered": float(np.mean(restricted_kl_clean_to_steered)),
        "mean_full_kl_clean_to_steered": float(np.mean(full_kl_clean_to_steered)),
        "rows_positive_target_prob_delta": int(np.sum(next_prob_delta > 0.0)),
        "rows_positive_logit_margin": int(np.sum(next_logit_margin > 0.0)),
        "n": int(labels.shape[0]),
        "records": records,
    }


def write_results(out_dir: Path, fit: dict, intervention: dict, explained: np.ndarray, prompts_path: Path) -> None:
    lines = [
        "# Real Qwen Weekday Circle",
        "",
        "## Fit",
        "",
        f"- converged: `{fit['converged']}`",
        f"- circular correlation, fitted aligned t vs true weekday: `{fit['aligned_circular_correlation']:.6f}`",
        f"- circular correlation, raw fitted t vs true weekday: `{fit['raw_circular_correlation']:.6f}`",
        f"- circular correlation, top-2 plane proxy vs true weekday: `{fit['proxy_circular_correlation']:.6f}`",
        f"- coordinate orientation: `{fit['orientation']}`; semantic +1 weekday step in raw t: `{fit['semantic_step_in_raw_t']:.6f}`",
        f"- activation EV: `{fit['activation_ev']:.6f}`; behavior EV: `{fit['behavior_ev']:.6f}`",
        f"- PCA variance retained by first {len(explained)} post-sink-peel components: `{float(explained.sum()):.6f}`",
        "",
        "## Intervention",
        "",
        f"- mean full-vocab KL(clean || steered): `{intervention['mean_full_kl_clean_to_steered']:.6g}`",
        f"- mean restricted weekday KL(clean || steered): `{intervention['mean_restricted_kl_clean_to_steered']:.6g}`",
        f"- mean shift in target-vs-source weekday logit margin: `{intervention['mean_next_minus_current_logit_shift']:.6f}`",
        f"- mean target weekday probability change: `{intervention['mean_target_prob_delta']:.6f}`",
        f"- positive target probability rows: `{intervention['rows_positive_target_prob_delta']}/{intervention['n']}`",
        f"- positive target-vs-source logit rows: `{intervention['rows_positive_logit_margin']}/{intervention['n']}`",
        "",
        "## Artifacts",
        "",
        f"- chart: `weekday_circle_chart.svg`",
        f"- fitted coordinates: `coords.csv`",
        f"- steering deltas in PCA coordinates: `steering_delta_pca.csv`",
        f"- prompts and token columns: `{prompts_path.name}`",
    ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Path to the gam checkout to run cargo in")
    parser.add_argument("--model", required=True, help="Hugging Face model path or id")
    parser.add_argument("--sink-resid", type=Path, required=True, help="Residual .npy used to estimate the L18 sink direction")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--sink-rows", type=int, default=20000)
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--fit-iters", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts, labels, offsets, template_ids = build_prompts()
    tokenizer, model = load_model(args.model, args.device, args.dtype)
    weekday_ids, weekday_strings = weekday_token_ids(tokenizer)
    activations, probs, clean_logits = harvest(tokenizer, model, prompts, weekday_ids, args.layer, args.device)

    print("[preprocess] estimating sink direction", flush=True)
    sink = top_sink_direction(args.sink_resid, args.sink_rows)
    scores, basis, mean, explained = preprocess_activations(activations, sink, args.pca_dim)

    np.savez(
        args.out_dir / "harvest_arrays.npz",
        activations=activations,
        probs=probs,
        clean_weekday_logits=clean_logits,
        labels=labels,
        offsets=offsets,
        template_ids=template_ids,
        sink_direction=sink,
        pca_basis=basis,
        post_sink_mean=mean,
        pca_explained=explained,
    )
    prompts_path = args.out_dir / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {
                "prompts": prompts,
                "weekday_token_strings": weekday_strings,
                "weekday_token_ids": weekday_ids,
                "labels": labels.tolist(),
                "offsets": offsets.tolist(),
                "template_ids": template_ids.tolist(),
            },
            indent=2,
        )
    )
    write_csv_matrix(args.out_dir / "activations_pca.csv", scores)
    write_csv_matrix(args.out_dir / "weekday_probs.csv", probs)
    write_labels(args.out_dir / "labels.csv", labels)

    fit = run_rust_fit(args.repo, args.out_dir, args.harmonics, args.fit_iters)
    delta_pca = read_csv_matrix(args.out_dir / "steering_delta_pca.csv")
    delta_hidden = delta_pca @ basis
    np.save(args.out_dir / "steering_delta_hidden.npy", delta_hidden.astype(np.float32))
    intervention = measure_intervention(
        tokenizer,
        model,
        prompts,
        labels,
        weekday_ids,
        delta_hidden,
        args.layer,
        args.device,
    )
    (args.out_dir / "intervention_results.json").write_text(json.dumps(intervention, indent=2))
    write_results(args.out_dir, fit, intervention, explained, prompts_path)
    print((args.out_dir / "results.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
