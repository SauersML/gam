#!/usr/bin/env python3
"""Geometric-Wall closure test on Gemma Scope residual-stream SAEs.

Run on MSI, preferably on a GPU node. The experiment harvests a modest set of
Gemma 2 residual activations, removes the top-PCA attention-sink direction, and
compares matched-parameter flat vs quadratic curved closures using Gemma Scope
SAE feature coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SaeChoice:
    layer: int
    path: str
    l0: int


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_layers(raw: str) -> list[int]:
    layers = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not layers:
        raise SystemExit("at least one layer is required")
    return layers


def choose_sae_files(repo: str, layers: list[int], target_l0: int) -> dict[int, SaeChoice]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo)
    choices: dict[int, SaeChoice] = {}
    for layer in layers:
        prefix = f"layer_{layer}/width_16k/average_l0_"
        candidates: list[tuple[int, str]] = []
        for path in files:
            if not path.startswith(prefix) or not path.endswith("/params.npz"):
                continue
            l0_raw = path[len(prefix) :].split("/", 1)[0]
            candidates.append((int(l0_raw), path))
        if not candidates:
            raise SystemExit(f"no Gemma Scope width_16k residual SAE found for layer {layer}")
        l0, path = min(candidates, key=lambda x: abs(x[0] - target_l0))
        choices[layer] = SaeChoice(layer=layer, path=path, l0=l0)
    return choices


def load_texts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset

    texts: list[str] = []
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    for row in ds:
        text = (row.get(args.text_field) or row.get("content") or "").strip()
        if len(text) < args.min_chars:
            continue
        texts.append(text)
        if len(texts) >= args.max_docs:
            break
    if not texts:
        raise SystemExit("dataset stream produced no usable texts")
    return texts


def hidden_size_from_model(model: Any) -> int:
    config = model.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    return int(config.text_config.hidden_size)


def harvest(args: argparse.Namespace, layers: list[int], out_dir: Path) -> dict[int, Path]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = {layer: out_dir / f"acts_L{layer}.npy" for layer in layers}
    pos_path = out_dir / "positions.npy"
    if all(path.exists() for path in paths.values()) and pos_path.exists():
        log("using cached activation arrays")
        return paths

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    d_model = hidden_size_from_model(model)
    arrays = {
        layer: np.lib.format.open_memmap(
            paths[layer], mode="w+", dtype=np.float32, shape=(args.n_tokens, d_model)
        )
        for layer in layers
    }
    positions = np.lib.format.open_memmap(
        pos_path, mode="w+", dtype=np.int32, shape=(args.n_tokens,)
    )

    texts = load_texts(args)
    cursor = 0
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_docs):
            batch = texts[start : start + args.batch_docs]
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            lengths = encoded["attention_mask"].sum(dim=1).cpu().tolist()
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            take_remaining = args.n_tokens - cursor
            for row, length in enumerate(lengths):
                if take_remaining <= 0:
                    break
                take = min(int(length), take_remaining)
                positions[cursor : cursor + take] = np.arange(take, dtype=np.int32)
                for layer in layers:
                    hidden = output.hidden_states[layer + 1][row, :take, :].detach().float().cpu()
                    arrays[layer][cursor : cursor + take] = hidden.numpy()
                cursor += take
                take_remaining = args.n_tokens - cursor
            log(f"harvested {cursor}/{args.n_tokens} residual rows")
            del output, encoded
            if cursor >= args.n_tokens:
                break
    if cursor < args.n_tokens:
        raise SystemExit(f"only harvested {cursor} tokens; increase --max-docs")
    for arr in arrays.values():
        arr.flush()
    positions.flush()
    return paths


def centered_covariance(x: np.ndarray, batch_rows: int) -> tuple[np.ndarray, np.ndarray, float]:
    n, d = x.shape
    mean = np.zeros(d, dtype=np.float64)
    total_energy = 0.0
    for start in range(0, n, batch_rows):
        xb = np.asarray(x[start : start + batch_rows], dtype=np.float64)
        mean += xb.sum(axis=0)
    mean /= float(n)
    cov = np.zeros((d, d), dtype=np.float64)
    for start in range(0, n, batch_rows):
        xb = np.asarray(x[start : start + batch_rows], dtype=np.float64) - mean
        cov += xb.T @ xb
        total_energy += float(np.sum(xb * xb))
    return cov, mean, total_energy


def peel_top_pc(x: np.ndarray, batch_rows: int) -> dict[str, Any]:
    cov, mean, total_energy = centered_covariance(x, batch_rows)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    top_value = float(values[order[0]])
    direction = vectors[:, order[0]].astype(np.float32)
    fraction = top_value / max(total_energy, 1e-30)
    return {
        "mean": mean.astype(np.float32),
        "direction": direction,
        "fraction": fraction,
        "total_energy": total_energy,
        "eigenvalues": values[order].astype(np.float64),
        "eigenvectors": vectors[:, order].astype(np.float32),
    }


def peeled_scores(
    x: np.ndarray,
    mean: np.ndarray,
    sink_direction: np.ndarray,
    basis: np.ndarray,
    batch_rows: int,
) -> tuple[np.ndarray, float, float]:
    n = x.shape[0]
    scores = np.empty((n, basis.shape[1]), dtype=np.float32)
    retained = 0.0
    peeled_total = 0.0
    v = sink_direction.astype(np.float64)
    b = basis.astype(np.float64)
    for start in range(0, n, batch_rows):
        xb = np.asarray(x[start : start + batch_rows], dtype=np.float64) - mean
        sink = xb @ v
        xp = xb - sink[:, None] * v[None, :]
        sb = xp @ b
        scores[start : start + xb.shape[0]] = sb.astype(np.float32)
        retained += float(np.sum(sb * sb))
        peeled_total += float(np.sum(xp * xp))
    return scores, retained, peeled_total


def curvature_proxy(scores: np.ndarray, seed: int, pool_size: int) -> float:
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    take = min(pool_size, n)
    idx = rng.choice(n, size=take, replace=False)
    y = scores[idx, : min(16, scores.shape[1])].astype(np.float64)
    y = y / np.maximum(np.std(y, axis=0, keepdims=True), 1e-8)
    d2 = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :2]
    vals: list[float] = []
    for i in range(take):
        j, k = int(nn[i, 0]), int(nn[i, 1])
        a = math.sqrt(float(d2[j, k]))
        b = math.sqrt(float(d2[i, k]))
        c = math.sqrt(float(d2[i, j]))
        semiperim = 0.5 * (a + b + c)
        area2 = semiperim * (semiperim - a) * (semiperim - b) * (semiperim - c)
        if area2 <= 0.0:
            continue
        vals.append(4.0 * math.sqrt(area2) / max(a * b * c, 1e-30))
    if not vals:
        return 0.0
    return float(np.median(vals))


def load_sae(repo: str, filename: str, cache_dir: str | None) -> dict[str, np.ndarray]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=repo, filename=filename, cache_dir=cache_dir)
    z = np.load(local)
    return {name: z[name] for name in z.files}


def sae_features(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    feature_ids: np.ndarray,
    batch_rows: int,
) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = x.shape[0]
    out = np.empty((n, feature_ids.shape[0]), dtype=np.float32)
    ids = torch.as_tensor(feature_ids.astype(np.int64), device=device)
    w_enc = torch.as_tensor(params["W_enc"], device=device)
    b_dec = torch.as_tensor(params["b_dec"], device=device)
    b_enc = torch.as_tensor(params["b_enc"], device=device)
    threshold = torch.as_tensor(params["threshold"], device=device)
    for start in range(0, n, batch_rows):
        xb = torch.as_tensor(np.asarray(x[start : start + batch_rows]), device=device)
        pre = (xb - b_dec) @ w_enc + b_enc
        h = torch.where(pre > threshold, pre, torch.zeros_like(pre))
        out[start : start + xb.shape[0]] = h.index_select(1, ids).detach().cpu().numpy()
    return out


def feature_energy(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    batch_rows: int,
) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy = torch.zeros(params["W_enc"].shape[1], dtype=torch.float64, device=device)
    w_enc = torch.as_tensor(params["W_enc"], device=device)
    b_dec = torch.as_tensor(params["b_dec"], device=device)
    b_enc = torch.as_tensor(params["b_enc"], device=device)
    threshold = torch.as_tensor(params["threshold"], device=device)
    for start in range(0, x.shape[0], batch_rows):
        xb = torch.as_tensor(np.asarray(x[start : start + batch_rows]), device=device)
        pre = (xb - b_dec) @ w_enc + b_enc
        h = torch.where(pre > threshold, pre, torch.zeros_like(pre))
        energy += torch.sum(h.double() * h.double(), dim=0)
    return energy.detach().cpu().numpy()


def fit_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    x64 = design.astype(np.float64)
    y64 = target.astype(np.float64)
    gram = x64.T @ x64
    penalty = np.eye(gram.shape[0], dtype=np.float64)
    penalty[0, 0] = 0.0
    rhs = x64.T @ y64
    return np.linalg.solve(gram + ridge * penalty, rhs).astype(np.float32)


def floor_from_design(
    train_design: np.ndarray,
    test_design: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    ridge: float,
) -> tuple[float, float]:
    beta = fit_ridge(train_design, train_scores, ridge)
    pred = test_design.astype(np.float32) @ beta
    resid = test_scores.astype(np.float32) - pred
    sse = float(np.sum(resid.astype(np.float64) * resid.astype(np.float64)))
    centered = test_scores.astype(np.float64) - np.mean(test_scores.astype(np.float64), axis=0)
    tss = float(np.sum(centered * centered))
    floor = sse / max(tss, 1e-30)
    return floor, 1.0 - floor


def layer_experiment(
    args: argparse.Namespace,
    layer: int,
    act_path: Path,
    sae_choice: SaeChoice,
    out_dir: Path,
) -> dict[str, Any]:
    x = np.load(act_path, mmap_mode="r")
    split = int(args.train_fraction * x.shape[0])
    sink = peel_top_pc(x, args.batch_rows)
    basis = sink["eigenvectors"][:, 1 : args.pca_dim + 1]
    scores, retained_energy, peeled_energy = peeled_scores(
        x, sink["mean"], sink["direction"], basis, args.batch_rows
    )
    curvature = curvature_proxy(scores, args.seed + layer, args.curvature_pool)
    params = load_sae(args.sae_repo, sae_choice.path, args.cache_dir)
    train_x = x[:split]
    energy = feature_energy(train_x, params, args.feature_batch_rows)
    flat_count = min(args.flat_features, energy.shape[0])
    if flat_count % 2 != 0:
        flat_count -= 1
    feature_order = np.argsort(energy)[::-1]
    flat_ids = feature_order[:flat_count].astype(np.int64)
    curved_ids = feature_order[: flat_count // 2].astype(np.int64)

    flat_h = sae_features(x, params, flat_ids, args.feature_batch_rows)
    curved_h = sae_features(x, params, curved_ids, args.feature_batch_rows)
    ones_train = np.ones((split, 1), dtype=np.float32)
    ones_test = np.ones((x.shape[0] - split, 1), dtype=np.float32)
    flat_train = np.concatenate([ones_train, flat_h[:split]], axis=1)
    flat_test = np.concatenate([ones_test, flat_h[split:]], axis=1)
    curved_train = np.concatenate([ones_train, curved_h[:split], curved_h[:split] ** 2], axis=1)
    curved_test = np.concatenate([ones_test, curved_h[split:], curved_h[split:] ** 2], axis=1)
    if flat_train.shape[1] != curved_train.shape[1]:
        raise AssertionError("flat and curved designs must have matched column counts")
    ridge = args.ridge_per_row * float(split)
    flat_floor, flat_ev = floor_from_design(flat_train, flat_test, scores[:split], scores[split:], ridge)
    curved_floor, curved_ev = floor_from_design(
        curved_train, curved_test, scores[:split], scores[split:], ridge
    )
    discarded_fraction = max(0.0, 1.0 - retained_energy / max(peeled_energy, 1e-30))
    full_flat_floor = discarded_fraction + (1.0 - discarded_fraction) * flat_floor
    full_curved_floor = discarded_fraction + (1.0 - discarded_fraction) * curved_floor
    result = {
        "layer": layer,
        "sae_path": sae_choice.path,
        "sae_l0": sae_choice.l0,
        "n_tokens": int(x.shape[0]),
        "d_model": int(x.shape[1]),
        "train_rows": int(split),
        "test_rows": int(x.shape[0] - split),
        "sink_top_pc_fraction": float(sink["fraction"]),
        "peeled_energy": float(peeled_energy),
        "pca_dim": int(args.pca_dim),
        "pca_retained_fraction_after_peel": float(retained_energy / max(peeled_energy, 1e-30)),
        "curvature_proxy_median_menger": float(curvature),
        "matched_design_columns": int(flat_train.shape[1]),
        "flat_features": int(flat_count),
        "curved_features": int(flat_count // 2),
        "ridge": float(ridge),
        "flat_retained_floor": float(flat_floor),
        "curved_retained_floor": float(curved_floor),
        "retained_floor_drop": float(flat_floor - curved_floor),
        "flat_retained_ev": float(flat_ev),
        "curved_retained_ev": float(curved_ev),
        "flat_full_energy_floor": float(full_flat_floor),
        "curved_full_energy_floor": float(full_curved_floor),
        "full_energy_floor_drop": float(full_flat_floor - full_curved_floor),
    }
    with (out_dir / f"layer_{layer}.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return None
    return float(np.dot(x, y) / denom)


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Geometric-Wall Closure Test on Gemma Scope",
        "",
        "This run uses Gemma 2 residual-stream Gemma Scope width-16k SAE feature coordinates.",
        "The sink direction is the top PCA direction of the harvested residual activations.",
        "Flat and curved fits use the same number of design columns on the post-sink PCA scores;",
        "the curved lane replaces half of the flat linear feature coordinates by quadratic chart terms.",
        "",
        "## Summary",
        "",
        f"- model: `{payload['model']}`",
        f"- sae repo: `{payload['sae_repo']}`",
        f"- tokens per layer: {payload['n_tokens']}",
        f"- layers: {', '.join(str(x['layer']) for x in payload['layers'])}",
        f"- curvature/drop Pearson r: {payload['curvature_drop_correlation']}",
        "",
        "## Layer Results",
        "",
        "| layer | SAE L0 | sink frac | curvature | flat floor | curved floor | drop | flat full floor | curved full floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["layers"]:
        lines.append(
            "| {layer} | {sae_l0} | {sink_top_pc_fraction:.6f} | "
            "{curvature_proxy_median_menger:.6f} | {flat_retained_floor:.6f} | "
            "{curved_retained_floor:.6f} | {retained_floor_drop:.6f} | "
            "{flat_full_energy_floor:.6f} | {curved_full_energy_floor:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `floor` is residual energy divided by centered held-out target energy.",
            "- `retained_floor` is measured in the retained post-sink PCA subspace.",
            "- `full_energy_floor` adds back PCA-discarded peeled energy as unreconstructed energy.",
            "- The curvature proxy is median nearest-neighbor Menger curvature in standardized post-sink PCA space.",
        ]
    )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--layers", default="12,25")
    ap.add_argument("--target-l0", type=int, default=80)
    ap.add_argument("--n-tokens", type=int, default=30000)
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--batch-docs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--flat-features", type=int, default=1024)
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--ridge-per-row", type=float, default=1e-4)
    ap.add_argument("--batch-rows", type=int, default=4096)
    ap.add_argument("--feature-batch-rows", type=int, default=1024)
    ap.add_argument("--curvature-pool", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    choices = choose_sae_files(args.sae_repo, layers, args.target_l0)
    act_paths = harvest(args, layers, out_dir)
    results = [
        layer_experiment(args, layer, act_paths[layer], choices[layer], out_dir)
        for layer in layers
    ]
    payload = {
        "model": args.model,
        "sae_repo": args.sae_repo,
        "n_tokens": args.n_tokens,
        "layers": results,
        "curvature_drop_correlation": correlation(
            [row["curvature_proxy_median_menger"] for row in results],
            [row["retained_floor_drop"] for row in results],
        ),
    }
    with (out_dir / "numbers.json").open("w") as f:
        json.dump(payload, f, indent=2)
    write_report(out_dir, payload)
    print("RESULTS_JSON " + json.dumps(payload))


if __name__ == "__main__":
    main()
