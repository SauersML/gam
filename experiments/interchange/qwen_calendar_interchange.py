#!/usr/bin/env python3
"""Qwen calendar-coordinate interchange intervention harness.

Run on an MSI GPU node. The harness fits a single periodic coordinate decoder
from real Qwen residual-stream activations on calendar prompts, then evaluates
held-out interchange interventions:

  source coordinate label -> base residual at the intervention layer -> logits

The headline metric is interchange accuracy: the fraction of patched forwards
whose restricted calendar-label output matches the source coordinate's known
next-label effect within a logit tolerance. The script also records full-vocab
KL(clean base || patched) in nats.
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


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class CalendarTask:
    name: str
    labels: tuple[str, ...]
    train_templates: tuple[str, ...]
    eval_templates: tuple[str, ...]

    def target_index(self, label_index: int) -> int:
        return (label_index + 1) % len(self.labels)


@dataclass(frozen=True)
class CleanExample:
    template_index: int
    label_index: int
    prompt: str
    activation: Any
    logits: Any


@dataclass(frozen=True)
class InterventionRecord:
    template_index: int
    base_label: str
    source_label: str
    predicted_label: str
    realized_label: str
    match: bool
    clean_base_label: str
    clean_base_correct: bool
    realized_kl: float
    predicted_logit_margin: float
    predicted_logprob_lift: float


def weekday_task() -> CalendarTask:
    labels = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    train_templates = (
        "Today is {label}. Tomorrow is",
        "If today is {label}, then tomorrow is",
        "The weekday after {label} is",
        "On a weekly calendar, {label} is followed by",
    )
    eval_templates = (
        "Starting on {label}, the next day is",
        "Calendar fact: {label} comes before",
        "For a one-day shift from {label}, the result is",
    )
    return CalendarTask("weekday", labels, train_templates, eval_templates)


def month_task() -> CalendarTask:
    labels = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    train_templates = (
        "The month after {label} is",
        "If this month is {label}, next month is",
        "On a calendar, {label} is followed by",
        "Advancing one month from {label} gives",
    )
    eval_templates = (
        "Starting in {label}, the next month is",
        "Calendar fact: {label} comes before",
        "For a one-month shift from {label}, the result is",
    )
    return CalendarTask("month", labels, train_templates, eval_templates)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--feature", choices=("weekday", "month"), default="weekday")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=17,
        help="0-based transformer block index to patch; 17 corresponds to hidden-state L18.",
    )
    parser.add_argument("--harmonics", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--candidate-prefix", default=" ")
    parser.add_argument("--max-interventions", type=int, default=42)
    parser.add_argument("--margin-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--out-dir", default="experiments/interchange")
    return parser.parse_args()


def task_from_name(name: str) -> CalendarTask:
    if name == "weekday":
        return weekday_task()
    if name == "month":
        return month_task()
    raise ValueError(f"unknown calendar feature {name!r}")


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir or None,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model, tokenizer


def model_input_device(model: Any) -> Any:
    return next(model.parameters()).device


def resolve_layers(model: Any) -> Any:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for root_name, layer_name in candidates:
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layer_name):
            return getattr(root, layer_name)
    raise ValueError("could not locate transformer block list on model")


def candidate_token_ids(tokenizer: Any, task: CalendarTask, prefix: str) -> list[int]:
    ids: list[int] = []
    for label in task.labels:
        encoded = tokenizer.encode(prefix + label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"candidate {prefix + label!r} tokenized to {len(encoded)} tokens; "
                "choose a different --candidate-prefix or task"
            )
        ids.append(int(encoded[0]))
    return ids


def run_clean(model: Any, tokenizer: Any, layer: Any, prompt: str) -> tuple[Any, Any]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model_input_device(model)) for key, value in encoded.items()}
    position = int(encoded["input_ids"].shape[1] - 1)
    captured: dict[str, Any] = {}

    def capture_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[0, position, :].detach().float().cpu()

    handle = layer.register_forward_hook(capture_hook)
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if "activation" not in captured:
        raise ValueError("activation hook did not fire")
    logits = output.logits[0, position, :].detach().float().cpu()
    return captured["activation"], logits


def run_patched(model: Any, tokenizer: Any, layer: Any, prompt: str, patched_activation: Any) -> Any:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model_input_device(model)) for key, value in encoded.items()}
    position = int(encoded["input_ids"].shape[1] - 1)

    def patch_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        replacement = patched_activation.to(device=hidden.device, dtype=hidden.dtype)
        edited[0, position, :] = replacement
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    handle = layer.register_forward_hook(patch_hook)
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
    finally:
        handle.remove()
    return output.logits[0, position, :].detach().float().cpu()


def phase(label_index: int, period: int) -> float:
    return TAU * label_index / period


def design_row(label_index: int, period: int, harmonics: int) -> np.ndarray:
    theta = phase(label_index, period)
    values = [1.0]
    for harmonic in range(1, harmonics + 1):
        angle = harmonic * theta
        values.extend([math.cos(angle), math.sin(angle)])
    return np.asarray(values, dtype=np.float64)


def fit_periodic_decoder(
    examples: list[CleanExample],
    task: CalendarTask,
    harmonics: int,
    ridge: float,
) -> tuple[np.ndarray, float]:
    if harmonics < 1:
        raise ValueError("--harmonics must be positive")
    if ridge <= 0.0:
        raise ValueError("--ridge must be positive")
    period = len(task.labels)
    phi = np.vstack([design_row(ex.label_index, period, harmonics) for ex in examples])
    x = np.vstack([ex.activation.numpy().astype(np.float64) for ex in examples])
    penalty = np.eye(phi.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(phi.T @ phi + penalty, phi.T @ x)
    fitted = phi @ coef
    centered = x - x.mean(axis=0, keepdims=True)
    rss = float(np.sum((x - fitted) ** 2))
    tss = float(np.sum(centered * centered))
    ev = 1.0 - rss / max(tss, 1.0e-30)
    return coef, ev


def decode_coordinate(coef: np.ndarray, label_index: int, period: int, harmonics: int) -> np.ndarray:
    return design_row(label_index, period, harmonics) @ coef


def restricted_top_label(logits: Any, candidate_ids: list[int], task: CalendarTask) -> tuple[int, np.ndarray]:
    values = logits[candidate_ids].numpy().astype(np.float64)
    top = int(np.argmax(values))
    return top, values


def full_vocab_kl(clean_logits: Any, patched_logits: Any) -> float:
    import torch

    logp = torch.log_softmax(clean_logits.to(torch.float64), dim=-1)
    logq = torch.log_softmax(patched_logits.to(torch.float64), dim=-1)
    p = logp.exp()
    return float((p * (logp - logq)).sum().item())


def restricted_logprob(logits: Any, candidate_ids: list[int], label_index: int) -> float:
    import torch

    values = logits[candidate_ids].to(torch.float64)
    return float(torch.log_softmax(values, dim=0)[label_index].item())


def collect_examples(
    model: Any,
    tokenizer: Any,
    layer: Any,
    templates: tuple[str, ...],
    task: CalendarTask,
) -> list[CleanExample]:
    examples: list[CleanExample] = []
    for template_index, template in enumerate(templates):
        for label_index, label in enumerate(task.labels):
            prompt = template.format(label=label)
            activation, logits = run_clean(model, tokenizer, layer, prompt)
            examples.append(
                CleanExample(
                    template_index=template_index,
                    label_index=label_index,
                    prompt=prompt,
                    activation=activation,
                    logits=logits,
                )
            )
    return examples


def clean_accuracy(
    examples: list[CleanExample],
    candidate_ids: list[int],
    task: CalendarTask,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    correct = 0
    for ex in examples:
        top, values = restricted_top_label(ex.logits, candidate_ids, task)
        expected = task.target_index(ex.label_index)
        is_correct = top == expected
        correct += int(is_correct)
        rows.append(
            {
                "template_index": ex.template_index,
                "label": task.labels[ex.label_index],
                "expected": task.labels[expected],
                "top": task.labels[top],
                "correct": is_correct,
                "expected_logit_margin": float(values[expected] - np.max(np.delete(values, expected))),
            }
        )
    return correct / len(examples), rows


def intervention_plan(
    eval_examples: list[CleanExample],
    task: CalendarTask,
    max_interventions: int,
    seed: int,
) -> list[tuple[CleanExample, CleanExample]]:
    by_template: dict[int, dict[int, CleanExample]] = {}
    for ex in eval_examples:
        by_template.setdefault(ex.template_index, {})[ex.label_index] = ex
    pairs: list[tuple[CleanExample, CleanExample]] = []
    for template_index in sorted(by_template):
        examples = by_template[template_index]
        for base_index in range(len(task.labels)):
            for source_index in range(len(task.labels)):
                if source_index != base_index:
                    pairs.append((examples[base_index], examples[source_index]))
    if max_interventions <= 0 or max_interventions >= len(pairs):
        return pairs
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(pairs), size=max_interventions, replace=False).tolist())
    return [pairs[index] for index in selected]


def run_interchanges(
    model: Any,
    tokenizer: Any,
    layer: Any,
    task: CalendarTask,
    candidate_ids: list[int],
    coef: np.ndarray,
    harmonics: int,
    pairs: list[tuple[CleanExample, CleanExample]],
    margin_tolerance: float,
) -> list[InterventionRecord]:
    import torch

    period = len(task.labels)
    records: list[InterventionRecord] = []
    for index, (base, source) in enumerate(pairs):
        base_fit = decode_coordinate(coef, base.label_index, period, harmonics)
        source_fit = decode_coordinate(coef, source.label_index, period, harmonics)
        delta = torch.from_numpy((source_fit - base_fit).astype(np.float32))
        patched_activation = base.activation + delta
        patched_logits = run_patched(model, tokenizer, layer, base.prompt, patched_activation)
        predicted_index = task.target_index(source.label_index)
        base_expected = task.target_index(base.label_index)
        clean_top, _clean_values = restricted_top_label(base.logits, candidate_ids, task)
        realized_top, patched_values = restricted_top_label(patched_logits, candidate_ids, task)
        other_values = np.delete(patched_values, predicted_index)
        margin = float(patched_values[predicted_index] - np.max(other_values))
        clean_lp = restricted_logprob(base.logits, candidate_ids, predicted_index)
        patched_lp = restricted_logprob(patched_logits, candidate_ids, predicted_index)
        record = InterventionRecord(
            template_index=base.template_index,
            base_label=task.labels[base.label_index],
            source_label=task.labels[source.label_index],
            predicted_label=task.labels[predicted_index],
            realized_label=task.labels[realized_top],
            match=margin >= -margin_tolerance,
            clean_base_label=task.labels[clean_top],
            clean_base_correct=clean_top == base_expected,
            realized_kl=max(full_vocab_kl(base.logits, patched_logits), 0.0),
            predicted_logit_margin=margin,
            predicted_logprob_lift=patched_lp - clean_lp,
        )
        records.append(record)
        log(
            f"intervention {index + 1}/{len(pairs)}: "
            f"{record.base_label} <- {record.source_label}, "
            f"predicted {record.predicted_label}, realized {record.realized_label}, "
            f"KL {record.realized_kl:.6f}"
        )
    return records


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    task: CalendarTask,
    candidate_ids: list[int],
    fit_ev: float,
    train_clean_accuracy: float,
    eval_clean_accuracy: float,
    clean_rows: list[dict[str, Any]],
    records: list[InterventionRecord],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    accuracy = float(np.mean([record.match for record in records])) if records else 0.0
    mean_kl = float(np.mean([record.realized_kl for record in records])) if records else 0.0
    mean_lift = (
        float(np.mean([record.predicted_logprob_lift for record in records])) if records else 0.0
    )
    mean_margin = (
        float(np.mean([record.predicted_logit_margin for record in records])) if records else 0.0
    )
    numbers = {
        "model": args.model,
        "feature": task.name,
        "layer_index": args.layer_index,
        "hidden_state_label": args.layer_index + 1,
        "harmonics": args.harmonics,
        "ridge": args.ridge,
        "candidate_prefix": args.candidate_prefix,
        "candidate_token_ids": dict(zip(task.labels, candidate_ids, strict=True)),
        "fit_activation_ev": fit_ev,
        "train_clean_restricted_accuracy": train_clean_accuracy,
        "eval_clean_restricted_accuracy": eval_clean_accuracy,
        "n_interventions": len(records),
        "interchange_accuracy": accuracy,
        "mean_realized_kl": mean_kl,
        "mean_predicted_label_logprob_lift": mean_lift,
        "mean_predicted_label_logit_margin": mean_margin,
        "margin_tolerance": args.margin_tolerance,
        "clean_eval_rows": clean_rows,
        "interventions": [record.__dict__ for record in records],
    }
    (out_dir / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")

    lines = [
        "# Qwen Calendar Interchange",
        "",
        "## Headline",
        "",
        f"- Interchange accuracy: **{accuracy:.4f}** ({sum(r.match for r in records)}/{len(records)})",
        f"- Mean realized KL(clean base || patched): **{mean_kl:.6f} nats**",
        f"- Mean predicted-label restricted log-prob lift: **{mean_lift:.4f} nats**",
        f"- Mean predicted-label logit margin: **{mean_margin:.4f}**",
        "",
        "## Protocol",
        "",
        f"- Model: `{args.model}`",
        f"- Feature: `{task.name}` with {len(task.labels)} periodic labels",
        f"- Patched block: module index `{args.layer_index}` (hidden-state L{args.layer_index + 1})",
        f"- Atom fit: one periodic harmonic decoder, harmonics `{args.harmonics}`, ridge `{args.ridge}`",
        f"- Fit activation EV on train templates: `{fit_ev:.6f}`",
        f"- Clean restricted accuracy: train `{train_clean_accuracy:.4f}`, eval `{eval_clean_accuracy:.4f}`",
        f"- Interventions: `{len(records)}` held-out base/source swaps",
        f"- Match tolerance: predicted label logit within `{args.margin_tolerance}` of restricted top-1",
        "",
        "The fitted atom maps known calendar phase to the residual-stream coordinate at the patched block. For each held-out intervention, the source token's phase is decoded through that atom and delta-written into the base token residual while preserving the base prompt. The predicted behavior is the source phase's known next-calendar label.",
        "",
        "## Sample Interventions",
        "",
        "| base | source | predicted | realized | match | KL | predicted margin |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for record in records[:12]:
        lines.append(
            f"| {record.base_label} | {record.source_label} | {record.predicted_label} | "
            f"{record.realized_label} | {str(record.match).lower()} | "
            f"{record.realized_kl:.6f} | {record.predicted_logit_margin:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `numbers.json` contains every clean prompt and intervention record.",
        ]
    )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    task = task_from_name(args.feature)
    log(f"loading {args.model}")
    model, tokenizer = load_model_and_tokenizer(args)
    layers = resolve_layers(model)
    if args.layer_index < 0 or args.layer_index >= len(layers):
        raise ValueError(f"--layer-index must be in [0, {len(layers)}); got {args.layer_index}")
    layer = layers[args.layer_index]
    candidate_ids = candidate_token_ids(tokenizer, task, args.candidate_prefix)
    log(f"candidate ids: {dict(zip(task.labels, candidate_ids, strict=True))}")

    log("collecting train residuals")
    train_examples = collect_examples(model, tokenizer, layer, task.train_templates, task)
    log("collecting eval residuals")
    eval_examples = collect_examples(model, tokenizer, layer, task.eval_templates, task)
    coef, fit_ev = fit_periodic_decoder(train_examples, task, args.harmonics, args.ridge)
    train_acc, _train_rows = clean_accuracy(train_examples, candidate_ids, task)
    eval_acc, eval_rows = clean_accuracy(eval_examples, candidate_ids, task)
    log(f"fit EV={fit_ev:.6f}; clean train acc={train_acc:.4f}; clean eval acc={eval_acc:.4f}")

    pairs = intervention_plan(eval_examples, task, args.max_interventions, args.seed)
    log(f"running {len(pairs)} interchange interventions")
    records = run_interchanges(
        model,
        tokenizer,
        layer,
        task,
        candidate_ids,
        coef,
        args.harmonics,
        pairs,
        args.margin_tolerance,
    )
    out_dir = Path(args.out_dir)
    write_outputs(
        out_dir,
        args,
        task,
        candidate_ids,
        fit_ev,
        train_acc,
        eval_acc,
        eval_rows,
        records,
    )
    accuracy = float(np.mean([record.match for record in records])) if records else 0.0
    mean_kl = float(np.mean([record.realized_kl for record in records])) if records else 0.0
    log(f"wrote {out_dir / 'results.md'}")
    print(f"INTERCHANGE_ACCURACY {accuracy:.6f}")
    print(f"MEAN_REALIZED_KL {mean_kl:.9f}")


if __name__ == "__main__":
    main()
