#!/usr/bin/env python
"""Causal hue steering along a certified Manifold-SAE circle.

This driver evaluates whether moving an OLMo residual-stream activation along
the fitted hue circle causes next-token color-word predictions to move toward a
target hue. It expects the OLMo color-bank layout:

  run/
    done.json
    bank_summary.json
    steer_cloze.json
    extra/
      activations.npy
      prompts.jsonl

The intervention at the requested layer is:

    x' = x + alpha * (m(theta_target) - m(theta_source))

where m is the fitted K=1 circle decoder map in raw residual-stream units. The
report includes a matched-norm random-direction control for every
(alpha, target hue) cell.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_RUN = Path("/home/azuser/Manifold-SAE/runs/OLMO3_32B_TRAJ_SFT/5e-5-step10790")
DEFAULT_FIT = Path("/mnt/work/exp/color_circle_l25.json")
DEFAULT_SNAPSHOT = Path("/mnt/work/exp/olmo3_32b_snapshot.json")
DEFAULT_OUT = Path("/mnt/work/exp/causal_hue_steering_report.json")
DEFAULT_TARGETS = ("red", "orange", "yellow", "green", "blue", "purple")


@dataclass(frozen=True)
class PromptRow:
    row_id: int
    prompt: str
    color: str
    hue_phase: float


@dataclass(frozen=True)
class ClozeItem:
    stem: str
    source_word: str
    source_hue_phase: float
    source_id: int | None


@dataclass
class CircleMap:
    mean: np.ndarray
    scale: np.ndarray
    decoder_block: np.ndarray
    basis_kind: str
    n_harmonics: int
    coord_sign: int
    coord_offset: float
    reml_score: float | None
    reconstruction_r2: float | None
    fit_source: str

    def coord_from_hue(self, hue_phase: float) -> float:
        return float((self.coord_sign * hue_phase + self.coord_offset) % 1.0)


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.write("\n")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def hue_phase_from_rgb(rgb: Iterable[int]) -> float:
    r, g, b = rgb
    h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return float(h % 1.0)


def load_prompt_rows(bank_dir: Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with (bank_dir / "prompts.jsonl").open() as handle:
        for line_no, line in enumerate(handle):
            row = json.loads(line)
            rgb = tuple(int(x) for x in row["rgb"])
            rows.append(
                PromptRow(
                    row_id=int(row.get("id", line_no)),
                    prompt=str(row["prompt"]),
                    color=str(row["color"]).strip().lower(),
                    hue_phase=hue_phase_from_rgb(rgb),
                )
            )
    return rows


def load_layer_bank(bank_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acts = np.load(bank_dir / "activations.npy", mmap_mode="r")
    if acts.ndim != 3:
        raise ValueError(f"expected activations.npy shape (n, layers, d), got {acts.shape}")
    if layer < 0 or layer >= acts.shape[1]:
        raise ValueError(f"layer {layer} is outside activation shape {acts.shape}")
    x = np.asarray(acts[:, layer, :], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    return x, mean, scale


def circular_mean_phase(values: np.ndarray) -> float:
    z = np.exp(2j * np.pi * values)
    return float((np.angle(z.mean()) / (2.0 * np.pi)) % 1.0)


def circular_alignment(coords: np.ndarray, hue_phase: np.ndarray) -> tuple[int, float, float]:
    best: tuple[int, float, float] | None = None
    for sign in (-1, 1):
        offset = circular_mean_phase(coords - sign * hue_phase)
        residual = np.exp(2j * np.pi * (coords - (sign * hue_phase + offset)))
        score = float(abs(residual.mean()))
        candidate = (sign, offset, score)
        if best is None or candidate[2] > best[2]:
            best = candidate
    assert best is not None
    return best


def load_rich_circle_map(path: Path) -> CircleMap | None:
    if not path.exists():
        return None
    obj = read_json(path)
    fit = obj.get("circle_map") if isinstance(obj, dict) else None
    if not isinstance(fit, dict):
        return None
    required = {
        "mean",
        "scale",
        "decoder_block",
        "basis_kind",
        "n_harmonics",
        "coord_sign",
        "coord_offset",
    }
    if not required.issubset(fit):
        return None
    return CircleMap(
        mean=np.asarray(fit["mean"], dtype=np.float64),
        scale=np.asarray(fit["scale"], dtype=np.float64),
        decoder_block=np.asarray(fit["decoder_block"], dtype=np.float64),
        basis_kind=str(fit["basis_kind"]),
        n_harmonics=int(fit["n_harmonics"]),
        coord_sign=int(fit["coord_sign"]),
        coord_offset=float(fit["coord_offset"]),
        reml_score=none_or_float(fit.get("reml_score")),
        reconstruction_r2=none_or_float(fit.get("reconstruction_r2")),
        fit_source=str(path),
    )


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def certificate_has_circle_winner(path: Path) -> bool:
    if not path.exists():
        return False
    obj = read_json(path)
    if not isinstance(obj, dict):
        return False
    winner = obj.get("winner")
    if isinstance(winner, dict):
        return winner.get("topology") == "circle"
    candidates = obj.get("candidates", [])
    ok = [c for c in candidates if c.get("status") == "ok" and c.get("reml_score") is not None]
    if not ok:
        return False
    best = min(ok, key=lambda c: float(c["reml_score"]))
    return best.get("topology") == "circle"


def fit_circle_map(
    bank_dir: Path,
    layer: int,
    prompts: list[PromptRow],
    n_iter: int,
    seed: int,
) -> CircleMap:
    import gamfit

    x, mean, scale = load_layer_bank(bank_dir, layer)
    if len(prompts) != x.shape[0]:
        raise ValueError(f"prompt/activation mismatch: {len(prompts)} prompts vs {x.shape[0]} activations")
    z = (x - mean) / scale
    fit = gamfit.sae_manifold_fit(
        z,
        K=1,
        d_atom=1,
        atom_topology="circle",
        n_iter=n_iter,
        random_state=seed,
    )
    coords = np.asarray(fit.coords[0], dtype=np.float64)[:, 0] % 1.0
    hue_phase = np.asarray([row.hue_phase for row in prompts], dtype=np.float64)
    sign, offset, alignment_score = circular_alignment(coords, hue_phase)
    if str(fit.basis_specs[0]) != "periodic":
        raise ValueError(f"expected circle fit to expose periodic basis, got {fit.basis_specs[0]!r}")
    n_harmonics = int(getattr(fit, "_n_harmonics", [1])[0])
    if n_harmonics < 1:
        raise ValueError(f"invalid fitted n_harmonics={n_harmonics}")
    source = f"inline K=1 circle fit; hue_phase_alignment={alignment_score:.6f}"
    return CircleMap(
        mean=mean,
        scale=scale,
        decoder_block=np.asarray(fit.decoder_blocks[0], dtype=np.float64),
        basis_kind=str(fit.basis_specs[0]),
        n_harmonics=n_harmonics,
        coord_sign=sign,
        coord_offset=offset,
        reml_score=none_or_float(getattr(fit, "reml_score", None)),
        reconstruction_r2=none_or_float(getattr(fit, "reconstruction_r2", None)),
        fit_source=source,
    )


def resolve_circle_map(
    fit_path: Path,
    bank_dir: Path,
    layer: int,
    prompts: list[PromptRow],
    n_iter: int,
    seed: int,
) -> CircleMap:
    rich = load_rich_circle_map(fit_path)
    if rich is not None:
        return rich
    if fit_path.exists() and not certificate_has_circle_winner(fit_path):
        raise ValueError(f"{fit_path} exists but does not report a circle winner")
    return fit_circle_map(bank_dir, layer, prompts, n_iter=n_iter, seed=seed)


def periodic_basis(coords: np.ndarray, n_harmonics: int) -> np.ndarray:
    from gamfit._binding import rust_module

    return np.asarray(
        rust_module().basis_with_jet(
            "periodic",
            np.ascontiguousarray(coords.reshape(-1, 1), dtype=np.float64),
            {"n_harmonics": int(n_harmonics)},
        )[0],
        dtype=np.float64,
    )


def decode_circle_raw(circle: CircleMap, coords: np.ndarray) -> np.ndarray:
    phi = periodic_basis(coords % 1.0, circle.n_harmonics)
    decoded_std = phi @ circle.decoder_block
    return decoded_std * circle.scale + circle.mean


def delta_for_items(
    circle: CircleMap,
    items: list[ClozeItem],
    target_hue_phase: float,
    alpha: float,
) -> np.ndarray:
    source_coords = np.asarray(
        [circle.coord_from_hue(item.source_hue_phase) for item in items],
        dtype=np.float64,
    )
    target_coords = np.full(len(items), circle.coord_from_hue(target_hue_phase), dtype=np.float64)
    source = decode_circle_raw(circle, source_coords)
    target = decode_circle_raw(circle, target_coords)
    return float(alpha) * (target - source)


def color_catalog(prompts: list[PromptRow]) -> dict[str, float]:
    by_color: dict[str, list[float]] = {}
    for row in prompts:
        by_color.setdefault(row.color, []).append(row.hue_phase)
    return {color: circular_mean_phase(np.asarray(phases, dtype=np.float64)) for color, phases in by_color.items()}


def parse_targets(raw: str | None, catalog: dict[str, float]) -> list[tuple[str, float]]:
    names = DEFAULT_TARGETS if raw is None else tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    out: list[tuple[str, float]] = []
    missing = [name for name in names if name not in catalog]
    if missing:
        available = ", ".join(sorted(catalog))
        raise ValueError(f"target colors absent from prompt bank: {missing}; available: {available}")
    for name in names:
        out.append((name, catalog[name]))
    return out


def list_payload(value: Any) -> list[Any] | None:
    return value if isinstance(value, list) else None


def extract_prepared_color_clozes(path: Path, catalog: dict[str, float]) -> list[ClozeItem]:
    if not path.exists():
        return []
    obj = read_json(path)
    candidates: list[Any] = []
    if isinstance(obj, list):
        candidates = obj
    elif isinstance(obj, dict):
        for key in ("color_cloze", "color_clozes", "cloze_items", "items", "examples"):
            rows = list_payload(obj.get(key))
            if rows:
                candidates = rows
                break
    items: list[ClozeItem] = []
    for idx, row in enumerate(candidates):
        if not isinstance(row, dict):
            continue
        stem = row.get("stem") or row.get("prompt") or row.get("text")
        source = row.get("source_color") or row.get("color") or row.get("answer") or row.get("truth")
        rgb = row.get("rgb") or row.get("source_rgb")
        if not stem or source is None:
            continue
        source_word = str(source).strip().lower()
        if rgb is not None:
            hue = hue_phase_from_rgb(rgb)
        elif source_word in catalog:
            hue = catalog[source_word]
        else:
            continue
        items.append(ClozeItem(str(stem).rstrip(), source_word, hue, int(row.get("id", idx)) if str(row.get("id", idx)).lstrip("-").isdigit() else None))
    return items


def synthesize_color_clozes(
    prompts: list[PromptRow],
    max_items: int,
    seed: int,
) -> list[ClozeItem]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(prompts))
    rng.shuffle(indices)
    selected = sorted(indices[: min(max_items, len(indices))].tolist())
    items: list[ClozeItem] = []
    for idx in selected:
        row = prompts[idx]
        stem = f"{row.prompt.rstrip('.')}. The color word is"
        items.append(ClozeItem(stem=stem, source_word=row.color, source_hue_phase=row.hue_phase, source_id=row.row_id))
    return items


def resolve_cloze_items(
    cloze_path: Path,
    prompts: list[PromptRow],
    catalog: dict[str, float],
    max_items: int,
    seed: int,
) -> tuple[list[ClozeItem], str]:
    prepared = extract_prepared_color_clozes(cloze_path, catalog)
    if prepared:
        return prepared[:max_items], f"prepared color clozes from {cloze_path}"
    return synthesize_color_clozes(prompts, max_items=max_items, seed=seed), (
        f"synthesized from prompts.jsonl because {cloze_path} has no color-cloze item list"
    )


def load_model_spec(run_dir: Path, snapshot_manifest: Path | None) -> tuple[str, str | None]:
    if snapshot_manifest is not None and snapshot_manifest.exists():
        obj = read_json(snapshot_manifest)
        return str(obj["model"]), obj.get("revision")
    done = read_json(run_dir / "done.json")
    return str(done["model"]), done.get("revision")


def find_layer_module(model: Any, layer: int) -> Any:
    roots = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("transformer.blocks", getattr(getattr(model, "transformer", None), "blocks", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
    ]
    for _name, modules in roots:
        if modules is not None and layer < len(modules):
            return modules[layer]
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(f".layers.{layer}")]
    if len(matches) == 1:
        return matches[0][1]
    raise ValueError(f"could not find transformer layer {layer} in model")


def single_token_id(tokenizer: Any, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"color word {word!r} is not a single leading-space token: {ids}")
    return int(ids[0])


def prepare_color_token_ids(tokenizer: Any, target_words: Iterable[str], catalog: dict[str, float]) -> dict[str, int]:
    words = sorted(set(catalog).union(target_words))
    token_ids: dict[str, int] = {}
    skipped: list[str] = []
    for word in words:
        try:
            token_ids[word] = single_token_id(tokenizer, word)
        except ValueError:
            skipped.append(word)
    missing_targets = [word for word in target_words if word not in token_ids]
    if missing_targets:
        raise ValueError(f"target color words are not single tokens: {missing_targets}; skipped={skipped}")
    return token_ids


def normalize_color_probs(logits: Any, token_ids: dict[str, int]) -> np.ndarray:
    import torch

    ids = torch.tensor(list(token_ids.values()), device=logits.device, dtype=torch.long)
    color_logits = logits.index_select(dim=-1, index=ids)
    return torch.softmax(color_logits.float(), dim=-1).detach().cpu().numpy()


def run_color_marginals(
    model: Any,
    tokenizer: Any,
    layer_module: Any,
    items: list[ClozeItem],
    color_token_ids: dict[str, int],
    batch_size: int,
    delta: np.ndarray | None,
) -> np.ndarray:
    import torch

    probs: list[np.ndarray] = []
    stems = [item.stem for item in items]
    for start in range(0, len(stems), batch_size):
        end = min(start + batch_size, len(stems))
        batch_stems = stems[start:end]
        hook = None
        if delta is not None:
            batch_delta = torch.from_numpy(delta[start:end])

            def add_delta(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                if isinstance(output, tuple):
                    hidden = output[0]
                    rest = output[1:]
                    patched = hidden.clone()
                    patched[:, -1, :] = patched[:, -1, :] + batch_delta.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    return (patched, *rest)
                patched = output.clone()
                patched[:, -1, :] = patched[:, -1, :] + batch_delta.to(
                    device=output.device,
                    dtype=output.dtype,
                )
                return patched

            hook = layer_module.register_forward_hook(add_delta)
        encoded = tokenizer(batch_stems, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        try:
            with torch.inference_mode():
                logits = model(**encoded).logits[:, -1, :]
        finally:
            if hook is not None:
                hook.remove()
        probs.append(normalize_color_probs(logits, color_token_ids))
    return np.concatenate(probs, axis=0)


def mean_kl(q: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    return float(np.mean(np.sum(q * (np.log(q + eps) - np.log(p + eps)), axis=1)))


def target_metrics(
    baseline: np.ndarray,
    steered: np.ndarray,
    words: list[str],
    target_word: str,
) -> dict[str, float]:
    target_idx = words.index(target_word)
    base_argmax = baseline.argmax(axis=1)
    steer_argmax = steered.argmax(axis=1)
    target_argmax = steer_argmax == target_idx
    moved = (base_argmax != target_idx) & target_argmax
    return {
        "hit_rate": float(target_argmax.mean()),
        "moved_to_target_rate": float(moved.mean()),
        "target_prob_mean": float(steered[:, target_idx].mean()),
        "target_prob_gain_mean": float((steered[:, target_idx] - baseline[:, target_idx]).mean()),
        "kl_color_marginal_vs_unsteered": mean_kl(steered, baseline),
    }


def matched_random_delta(delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(delta.shape)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    noise_norm = np.maximum(noise_norm, 1e-12)
    return noise * (delta_norm / noise_norm)


def parse_alphas(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument("--fit-json", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--fit-n-iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--alphas", default="0.5,1,2")
    parser.add_argument("--target-colors")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    run_dir = args.run_dir
    bank_dir = args.bank_dir or (run_dir / "extra")
    done = read_json(run_dir / "done.json")
    layer = int(args.layer if args.layer is not None else done["steer_layer"])
    prompts = load_prompt_rows(bank_dir)
    catalog = color_catalog(prompts)
    targets = parse_targets(args.target_colors, catalog)
    items, cloze_source = resolve_cloze_items(
        run_dir / "steer_cloze.json",
        prompts,
        catalog,
        max_items=args.max_items,
        seed=args.seed,
    )
    circle = resolve_circle_map(
        args.fit_json,
        bank_dir,
        layer,
        prompts,
        n_iter=args.fit_n_iter,
        seed=args.seed,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = load_model_spec(run_dir, args.snapshot_manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_words = [word for word, _hue in targets]
    color_token_ids = prepare_color_token_ids(tokenizer, target_words, catalog)
    words = list(color_token_ids.keys())

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    layer_module = find_layer_module(model, layer)

    baseline = run_color_marginals(
        model,
        tokenizer,
        layer_module,
        items,
        color_token_ids,
        batch_size=args.batch_size,
        delta=None,
    )
    rng = np.random.default_rng(args.seed + 9973)
    results: list[dict[str, Any]] = []
    for alpha in parse_alphas(args.alphas):
        for target_word, target_hue in targets:
            delta = delta_for_items(circle, items, target_hue_phase=target_hue, alpha=alpha)
            steered = run_color_marginals(
                model,
                tokenizer,
                layer_module,
                items,
                color_token_ids,
                batch_size=args.batch_size,
                delta=delta,
            )
            control_delta = matched_random_delta(delta, rng)
            control = run_color_marginals(
                model,
                tokenizer,
                layer_module,
                items,
                color_token_ids,
                batch_size=args.batch_size,
                delta=control_delta,
            )
            results.append(
                {
                    "alpha": float(alpha),
                    "target_word": target_word,
                    "target_hue_phase": float(target_hue),
                    "steering_delta_norm_mean": float(np.linalg.norm(delta, axis=1).mean()),
                    "steer": target_metrics(baseline, steered, words, target_word),
                    "random_matched_norm_control": target_metrics(baseline, control, words, target_word),
                }
            )
            write_json(
                args.out,
                report_payload(
                    args,
                    done,
                    layer,
                    model_id,
                    revision,
                    circle,
                    cloze_source,
                    items,
                    words,
                    color_token_ids,
                    baseline,
                    results,
                    elapsed=time.time() - t0,
                ),
            )

    write_json(
        args.out,
        report_payload(
            args,
            done,
            layer,
            model_id,
            revision,
            circle,
            cloze_source,
            items,
            words,
            color_token_ids,
            baseline,
            results,
            elapsed=time.time() - t0,
        ),
    )
    print(f"wrote {args.out}", flush=True)


def report_payload(
    args: argparse.Namespace,
    done: dict[str, Any],
    layer: int,
    model_id: str,
    revision: str | None,
    circle: CircleMap,
    cloze_source: str,
    items: list[ClozeItem],
    words: list[str],
    color_token_ids: dict[str, int],
    baseline: np.ndarray,
    results: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    baseline_argmax = baseline.argmax(axis=1)
    return {
        "run_dir": str(args.run_dir),
        "bank_model": done.get("model"),
        "bank_revision": done.get("revision"),
        "eval_model": model_id,
        "eval_revision": revision,
        "layer": layer,
        "n_items": len(items),
        "cloze_source": cloze_source,
        "source_color_counts": {
            word: sum(1 for item in items if item.source_word == word)
            for word in sorted({item.source_word for item in items})
        },
        "color_words": words,
        "color_token_ids": color_token_ids,
        "baseline_color_argmax_counts": {
            word: int(np.sum(baseline_argmax == idx)) for idx, word in enumerate(words)
        },
        "circle_map": {
            "basis_kind": circle.basis_kind,
            "n_harmonics": circle.n_harmonics,
            "coord_sign": circle.coord_sign,
            "coord_offset": circle.coord_offset,
            "reml_score": circle.reml_score,
            "reconstruction_r2": circle.reconstruction_r2,
            "fit_source": circle.fit_source,
        },
        "alphas": parse_alphas(args.alphas),
        "fit_json": str(args.fit_json),
        "results": results,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    main()
