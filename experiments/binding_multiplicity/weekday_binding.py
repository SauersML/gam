#!/usr/bin/env python3
"""App C detection — harvest a real weekday circle and emit the harmonic codes
that the Rust ``binding_multiplicity`` example recovers spikes from.

This is the *model boundary* of App C: PyTorch is used only to harvest last-token
residual activations for single-weekday and two-weekday prompts. Everything
downstream (the Prony super-resolution + gated multiplicity count) is delegated
to the tested Rust example, which reads the two CSVs this writes:

  weekday_codes.csv       one row per single-weekday activation:
                          ``weekday,label,<2H harmonic code>``  (multiplicity 1)
  two_instance_codes.csv  one row per two-weekday prompt:
                          ``t1,t2,<2H harmonic code>``          (multiplicity 2)

The harmonic *frame* is fit from the model's own single-weekday geometry: the 7
weekday means define a 2-D circle plane; each weekday's empirical angle is its
circle position ``t_k``; a ``2H x p`` decoder ``D`` is least-squares fit so that
``D^T u(t_k)`` reconstructs the weekday means, and any activation ``x`` is encoded
to its harmonic code ``z = (D D^T)^{-1} D x`` with ``u(t) = [cos 2πt, sin 2πt,
..., cos 2πHt, sin 2πHt]``. A two-weekday activation that the model forms as the
superposition of its two single-weekday parts then encodes to ``a1 u(t1) + a2
u(t2)`` — exactly the multi-spike measure super-resolution un-superposes.

Run on MSI (GPU) with the Qwen3-8B checkout already on disk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Single-weekday templates: the last token is the anchor where the weekday
# concept is live in the residual stream.
SINGLE_TEMPLATES = [
    "Today is {a}",
    "The meeting is on {a}",
    "She was born on a {a}",
    "We always rest on {a}",
    "The report is due {a}",
    "It happened last {a}",
    "The appointment moved to {a}",
    "Every {a}",
]

# Two-weekday templates naming two distinct weekdays; both concepts should be
# live at the final token (the binding/multiplicity case).
PAIR_TEMPLATES = [
    "The two meetings are on {a} and {b}",
    "The office is closed on {a} and {b}",
    "Choose between {a} and {b}",
    "The class meets on {a} and {b}",
    "We travel on {a} and {b}",
    "Either {a} or {b}",
]


def build_single_prompts():
    prompts, labels = [], []
    for template in SINGLE_TEMPLATES:
        for k, day in enumerate(WEEKDAYS):
            prompts.append(template.format(a=day))
            labels.append(k)
    return prompts, np.asarray(labels, dtype=np.int64)


def build_pair_prompts():
    prompts, la, lb = [], [], []
    for template in PAIR_TEMPLATES:
        for i in range(len(WEEKDAYS)):
            for j in range(len(WEEKDAYS)):
                if i == j:
                    continue
                prompts.append(template.format(a=WEEKDAYS[i], b=WEEKDAYS[j]))
                la.append(i)
                lb.append(j)
    return prompts, np.asarray(la, dtype=np.int64), np.asarray(lb, dtype=np.int64)


def load_model(model_path: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    if device != "cpu":
        model = model.to(device)
    return tokenizer, model.eval()


def harvest(tokenizer, model, prompts, layer: int, device: str) -> np.ndarray:
    import time

    import torch

    acts = []
    started = time.time()
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            acts.append(out.hidden_states[layer][0, -1, :].float().cpu().numpy())
            if (idx + 1) % 25 == 0:
                print(f"[harvest] {idx + 1}/{len(prompts)} ({time.time() - started:.0f}s)", flush=True)
    return np.asarray(acts, dtype=np.float32)


def top_sink_direction(sample: np.ndarray) -> np.ndarray:
    centered = sample - sample.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0].astype(np.float64)
    return direction / np.linalg.norm(direction)


def harmonic_features(t: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Rows u(t) = [cos 2πt, sin 2πt, cos 4πt, sin 4πt, ...] for harmonics 1..H."""
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * math.pi * h * t))
        cols.append(np.sin(2 * math.pi * h * t))
    return np.stack(cols, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--harmonics", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    single_prompts, single_labels = build_single_prompts()
    pair_prompts, pair_a, pair_b = build_pair_prompts()

    tokenizer, model = load_model(args.model, args.device, args.dtype)
    print(f"[model] layers hidden_states index up to {model.config.num_hidden_layers}", flush=True)
    single_acts = harvest(tokenizer, model, single_prompts, args.layer, args.device)
    pair_acts = harvest(tokenizer, model, pair_prompts, args.layer, args.device)

    # --- preprocess: sink peel + PCA (basis fit on singles, applied to both) ---
    sink = top_sink_direction(single_acts.astype(np.float64))

    def peel(a):
        a = a.astype(np.float64)
        return a - np.outer(a @ sink, sink)

    single_peeled = peel(single_acts)
    pair_peeled = peel(pair_acts)
    mean = single_peeled.mean(axis=0, keepdims=True)
    single_c = single_peeled - mean
    pair_c = pair_peeled - mean
    _, sv, vt = np.linalg.svd(single_c, full_matrices=False)
    keep = min(args.pca_dim, vt.shape[0])
    basis = vt[:keep]  # keep x d
    ev = float((sv[:keep] ** 2).sum() / (sv ** 2).sum())
    single_x = single_c @ basis.T  # n x keep
    pair_x = pair_c @ basis.T

    # --- circle geometry: weekday means -> 2D plane -> empirical positions t_k ---
    means = np.stack([single_x[single_labels == k].mean(axis=0) for k in range(len(WEEKDAYS))])
    mean_center = means.mean(axis=0, keepdims=True)
    means_c = means - mean_center
    _, _, vt2 = np.linalg.svd(means_c, full_matrices=False)
    plane = vt2[:2]  # 2 x keep
    proj = means_c @ plane.T  # 7 x 2
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    t_k = (angles / (2 * math.pi)) % 1.0  # empirical circle position per weekday

    # --- fit harmonic decoder D (2H x keep): D^T u(t_k) ~= means (centered) ---
    U = harmonic_features(t_k, args.harmonics)  # 7 x 2H
    # least squares D (2H x keep): min || U D - means_c ||
    D, *_ = np.linalg.lstsq(U, means_c, rcond=None)  # 2H x keep
    recon = U @ D
    frame_ss_res = float(((means_c - recon) ** 2).sum())
    frame_ss_tot = float((means_c ** 2).sum())
    frame_r2 = 1.0 - frame_ss_res / frame_ss_tot if frame_ss_tot > 0 else float("nan")

    # encoder: z = (D D^T)^{-1} D x  (least-squares harmonic code of x on the frame)
    DDt = D @ D.T  # 2H x 2H
    enc = np.linalg.solve(DDt + 1e-8 * np.eye(DDt.shape[0]), D)  # 2H x keep

    def encode(x):  # x: n x keep -> n x 2H
        return x @ enc.T

    single_codes = encode(single_x - mean_center)  # relative to circle center
    pair_codes = encode(pair_x - mean_center)

    # --- write CSVs ---
    two_h = 2 * args.harmonics
    with (args.out_dir / "weekday_codes.csv").open("w") as fh:
        fh.write("weekday,label," + ",".join(f"z{i}" for i in range(two_h)) + "\n")
        for i in range(len(single_prompts)):
            code = ",".join(f"{v:.10g}" for v in single_codes[i])
            fh.write(f"{WEEKDAYS[single_labels[i]]},{int(single_labels[i])},{code}\n")

    with (args.out_dir / "two_instance_codes.csv").open("w") as fh:
        fh.write("t1,t2," + ",".join(f"z{i}" for i in range(two_h)) + "\n")
        for i in range(len(pair_prompts)):
            t1 = float(t_k[pair_a[i]])
            t2 = float(t_k[pair_b[i]])
            code = ",".join(f"{v:.10g}" for v in pair_codes[i])
            fh.write(f"{t1:.10g},{t2:.10g},{code}\n")

    meta = {
        "model": args.model,
        "layer": args.layer,
        "pca_dim": keep,
        "pca_explained_variance": ev,
        "harmonics": args.harmonics,
        "code_width_2h": two_h,
        "n_single": len(single_prompts),
        "n_pairs": len(pair_prompts),
        "weekday_positions_t_k": {WEEKDAYS[k]: float(t_k[k]) for k in range(len(WEEKDAYS))},
        "harmonic_frame_r2": frame_r2,
        "sink_peeled": True,
    }
    (args.out_dir / "weekday_frame_meta.json").write_text(json.dumps(meta, indent=2))
    np.savez(
        args.out_dir / "weekday_harvest.npz",
        single_acts=single_acts,
        pair_acts=pair_acts,
        single_labels=single_labels,
        pair_a=pair_a,
        pair_b=pair_b,
        single_codes=single_codes,
        pair_codes=pair_codes,
        t_k=t_k,
        sink=sink,
        basis=basis,
        decoder=D,
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[done] wrote weekday_codes.csv, two_instance_codes.csv to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
