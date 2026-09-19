#!/usr/bin/env python3
"""#2263 gates 3 and 4 on a real model: the target-dose matrix and the month replay.

Gate 3, the target-dose matrix. A weekday or month circle is fitted at one layer
with a harvested output-Fisher shard installed. Every held-out base prompt is then
steered through the public ``ManifoldSAE.steer_to_target`` along both chart
directions to each requested dose. The applied-dose probe patches the base
prompt's label-position residual with the lifted move, measures
``KL(p_base || p_patched)`` over the full vocabulary, and reports the exact
directional Fisher dose ``½ (Jδ)ᵀ F (Jδ)`` of the same move by one JVP. The solve
takes no accuracy or probe budget: every call that returns a plan has resolved the
displacement to its representation limit, and the report publishes each plan's
relative landing error and probe count. A call the chart cannot satisfy must end in
a typed refusal, which is recorded, never dropped.

Gate 4, the month semantic replay. Each held-out month base is moved by +1..+6
months through the public ``ManifoldSAE.steer``, using the fitted chart's own
label-to-coordinate map, and the realized shift is read off the patched
next-token distribution. The report publishes the requested-by-realized confusion
matrix. Every row carries its predicted and measured dose, so a semantic landing
cannot hide a dose miss.

Prompt banks and the full-vocabulary KL come from
``experiments/interchange/qwen_calendar_interchange.py``; the chart label map
comes from ``experiments/steering_e1/run_e1.py``. Nothing here re-implements a gam
primitive: the fit, the dose solve and the steering move are the Rust library's
public entries, and the torch code is the sanctioned model-interop path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TAU = 2.0 * math.pi


def _load_module(name: str, relative: str) -> Any:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CALENDAR = _load_module(
    "qwen_calendar_interchange", "experiments/interchange/qwen_calendar_interchange.py"
)
E1 = _load_module("steering_e1_run", "experiments/steering_e1/run_e1.py")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Pure helpers (numpy only; pinned by tests/test_qwen_target_dose_matrix_contract.py)
# --------------------------------------------------------------------------- #
def chart_projection(x_fit: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-only PCA chart: ``(mean (1, p), lift (r, p))`` with orthonormal rows."""
    if dim < 1:
        raise ValueError(f"chart dimension must be positive; got {dim}")
    mean = x_fit.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_fit - mean, full_matrices=False)
    rank = min(dim, vt.shape[0])
    return mean, np.ascontiguousarray(vt[:rank])


def to_chart(x: np.ndarray, mean: np.ndarray, lift: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((x - mean) @ lift.T)


def lift_move(delta_chart: Any, lift: np.ndarray) -> np.ndarray:
    """A chart-space move as the ambient move it names: ``δ_ambient = δ_chart · L``."""
    return np.asarray(delta_chart, dtype=np.float64) @ lift


def project_factors(factors: np.ndarray, lift: np.ndarray) -> np.ndarray:
    """``(n, p, r)`` ambient factors as ``(n, p_chart, r)`` chart factors ``L · U_n``.

    ``(L U_n)(L U_n)ᵀ = L (U_n U_nᵀ) Lᵀ`` is the chart pullback of the harvested
    rank-r truncation, so the operator status stays whatever the harvest was.
    """
    return np.ascontiguousarray(np.einsum("cp,npr->ncr", lift, factors))


def realized_shift(realized_label: int, source_label: int, n_labels: int) -> int:
    """The shift a patched answer realizes; the prompt asks for the label after the source."""
    return (realized_label - source_label - 1) % n_labels


def relative_landing_error(measured_nats: float, target_nats: float) -> float:
    return abs(measured_nats - target_nats) / target_nats


def nearest_fitted_row(fit_coord: np.ndarray, coordinate: float, period: float) -> int:
    if np.isfinite(period):
        half = 0.5 * period
        distance = np.abs((fit_coord - coordinate + half) % period - half)
    else:
        distance = np.abs(fit_coord - coordinate)
    return int(np.argmin(distance))


def prompt_bank_sha256(tasks: list[Any]) -> str:
    payload = [
        {
            "name": task.name,
            "labels": list(task.labels),
            "train_templates": list(task.train_templates),
            "eval_templates": list(task.eval_templates),
        }
        for task in tasks
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Model interop
# --------------------------------------------------------------------------- #
def build_readout(model: Any, layer: Any, position: int) -> Any:
    """Last-position logits with the label-position residual routed through a site.

    The harvest hooks ``site``, whose output is the one label-position row; the
    layer hook writes that (possibly spliced) row back into the residual stream,
    so every Jacobian taken through the site is end to end.
    """
    import torch

    class LabelSite(torch.nn.Module):
        def forward(self, row: torch.Tensor) -> torch.Tensor:
            return row

    class LabelPositionReadout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.site = LabelSite()

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                row = self.site(hidden[:, position, :])
                edited = hidden.clone()
                edited[:, position, :] = row
                if isinstance(output, tuple):
                    return (edited,) + tuple(output[1:])
                return edited

            handle = layer.register_forward_hook(hook)
            try:
                out = self.model(input_ids=input_ids, use_cache=False)
            finally:
                handle.remove()
            return out.logits[:, -1, :]

    return LabelPositionReadout()


def label_position(tokenizer: Any, template: str, label: str) -> tuple[list[int], int]:
    """Token ids of the prompt and the index of the label's last token."""
    prompt = template.format(label=label)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix = template.split("{label}")[0] + label
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"prompt {prompt!r} does not tokenize with its label prefix {prefix!r} as a prefix"
        )
    return ids, len(prefix_ids) - 1


def spliced_logits(readout: Any, input_ids: Any, row: Any) -> Any:
    """Last-position logits with ``row`` spliced in at the label site."""

    def splice(_module: Any, _inputs: Any, _output: Any) -> Any:
        return row[None, :]

    handle = readout.site.register_forward_hook(splice)
    try:
        return readout(input_ids)[0]
    finally:
        handle.remove()


def capture(readout: Any, input_ids: Any) -> tuple[Any, Any]:
    import torch

    captured: dict[str, Any] = {}

    def grab(_module: Any, _inputs: Any, output: Any) -> None:
        captured["row"] = output[0].detach()

    handle = readout.site.register_forward_hook(grab)
    try:
        with torch.inference_mode():
            logits = readout(input_ids)[0].detach()
    finally:
        handle.remove()
    return captured["row"], logits


def exact_directional_nats(readout: Any, input_ids: Any, base_row: Any, delta: Any) -> float:
    """``½ (Jδ)ᵀ F (Jδ)`` at the base row along the ambient move, by one JVP."""
    import torch

    def f(row: torch.Tensor) -> torch.Tensor:
        return spliced_logits(readout, input_ids, row)

    logits, jd = torch.func.jvp(f, (base_row,), (delta,))
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    jd = jd.to(torch.float64)
    mean = (probs * jd).sum()
    return 0.5 * float(((probs * jd * jd).sum() - mean * mean).item())


# --------------------------------------------------------------------------- #
# One feature
# --------------------------------------------------------------------------- #
def run_feature(args: argparse.Namespace, model: Any, tokenizer: Any, layer: Any, task: Any) -> dict[str, Any]:
    import torch
    import gamfit
    from gamfit.torch.harvest import HarvestShard, harvest_output_fisher_factors

    device = next(model.parameters()).device
    candidate_ids = CALENDAR.candidate_token_ids(tokenizer, task, " ")

    def prompts(templates: tuple[str, ...]) -> list[dict[str, Any]]:
        out = []
        for template_index, template in enumerate(templates):
            for label_index, label in enumerate(task.labels):
                ids, position = label_position(tokenizer, template, label)
                out.append(
                    {
                        "template_index": template_index,
                        "label_index": label_index,
                        "prompt": template.format(label=label),
                        "ids": torch.tensor([ids], dtype=torch.long, device=device),
                        "position": position,
                    }
                )
        return out

    fit_prompts = prompts(task.train_templates)
    base_prompts = prompts(task.eval_templates)

    log(f"{task.name}: harvesting rank-{args.rank} output-Fisher factors on {len(fit_prompts)} fit prompts")
    xs, us = [], []
    for item in fit_prompts:
        readout = build_readout(model, layer, item["position"])
        shard = harvest_output_fisher_factors(
            readout, readout.site, item["ids"], rank=args.rank, seed=args.seed
        )
        if shard.X.shape[0] != 1:
            raise ValueError(f"label site harvested {shard.X.shape[0]} rows; expected one")
        xs.append(np.asarray(shard.X, dtype=np.float64))
        us.append(np.asarray(shard.U, dtype=np.float64))
    x_fit = np.concatenate(xs, axis=0)
    u_fit = np.concatenate(us, axis=0)
    mean, lift = chart_projection(x_fit, args.pca_dim)
    x_fit_chart = to_chart(x_fit, mean, lift)
    chart_shard = HarvestShard(
        X=x_fit_chart,
        U=project_factors(u_fit, lift),
        mass_residual=None,
        rank=args.rank,
        factor_kind="uncertified_approximation",
        provenance="output_fisher",
    )

    log(f"{task.name}: fitting sae_manifold_fit in a {x_fit_chart.shape[1]}-dim chart with the shard")
    sae = gamfit.sae.sae_manifold_fit(
        x_fit_chart,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=args.n_iter,
        random_state=args.seed,
        fisher_factors=chart_shard,
    )
    atom = 0
    fit_label_index = np.asarray([item["label_index"] for item in fit_prompts])
    fit_coord = np.asarray(sae.coords[atom], dtype=np.float64)[:, 0]
    recovery = [
        (E1.circular_recovery(fit_coord, fit_label_index, len(task.labels), period), period)
        for period in (1.0, TAU)
    ]
    (recovery_r2, orientation), period = max(recovery, key=lambda entry: entry[0][0])
    label_map = E1.label_coordinate_map(fit_coord, fit_label_index, len(task.labels), period)
    log(f"{task.name}: structure recovery R2={recovery_r2:.4f} period={period} orientation={orientation:+d}")

    base_rows, base_logits = [], []
    for item in base_prompts:
        readout = build_readout(model, layer, item["position"])
        row, logits = capture(readout, item["ids"])
        base_rows.append(row)
        base_logits.append(logits)
    x_base = np.stack([row.to(torch.float64).cpu().numpy() for row in base_rows])
    latents = sae.converged_latents(to_chart(x_base, mean, lift))
    base_coord = np.asarray(latents["coords"][atom], dtype=np.float64)[:, 0]
    base_gate = np.asarray(latents["assignments"], dtype=np.float64)[:, atom]
    lift_t = torch.from_numpy(lift).to(device=device, dtype=base_rows[0].dtype)

    dose_records: list[dict[str, Any]] = []
    for b, item in enumerate(base_prompts):
        readout = build_readout(model, layer, item["position"])
        metric_row = nearest_fitted_row(fit_coord, float(base_coord[b]), period)

        def probe(plan: dict[str, Any], _b: int = b, _readout: Any = readout, _item: dict[str, Any] = item) -> dict[str, Any]:
            delta_chart = torch.tensor(plan["delta"], dtype=lift_t.dtype, device=device)
            delta = delta_chart @ lift_t
            patched = spliced_logits(_readout, _item["ids"], base_rows[_b] + delta).detach()
            return {
                "effective_delta": list(plan["delta"]),
                "exact_directional_nats": exact_directional_nats(_readout, _item["ids"], base_rows[_b], delta),
                "measured_nats": CALENDAR.full_vocab_kl(base_logits[_b].cpu(), patched.cpu()),
                "certified_attainable_upper_nats": None,
            }

        for direction in (1.0, -1.0):
            for target in args.target_nats:
                record: dict[str, Any] = {
                    "feature": task.name,
                    "base_prompt_id": f"{task.name}-t{item['template_index']}-l{item['label_index']}",
                    "label_index": item["label_index"],
                    "direction": direction,
                    "target_nats": target,
                    "metric_row": metric_row,
                }
                request = {
                    "atom_k": atom,
                    "metric_row": metric_row,
                    "target_nats": target,
                    "t_from": [float(base_coord[b])],
                    "direction": [direction],
                }
                try:
                    plan = sae.steer_to_target(request, probe)
                except ValueError as error:
                    record.update(outcome="typed_refusal", refusal=str(error))
                    dose_records.append(record)
                    continue
                record.update(
                    outcome="plan",
                    measured_nats=float(plan["measured_nats"]),
                    predicted_nats=float(plan["predicted_nats"]),
                    predicted_nats_kind=plan["predicted_nats_kind"],
                    iterations=int(plan["iterations"]),
                    displacement=float(plan["displacement"]),
                    relative_landing_error=relative_landing_error(
                        float(plan["measured_nats"]), target
                    ),
                )
                dose_records.append(record)

    plans = [record for record in dose_records if record["outcome"] == "plan"]
    feature_report: dict[str, Any] = {
        "feature": task.name,
        "structure_recovery_r2": float(recovery_r2),
        "chart_period": float(period),
        "chart_dim": int(x_fit_chart.shape[1]),
        "n_fit_prompts": len(fit_prompts),
        "n_base_prompts": len(base_prompts),
        "gate3": {
            "calls": len(dose_records),
            "plans": len(plans),
            "typed_refusals": len(dose_records) - len(plans),
            "max_relative_landing_error": max(
                (record["relative_landing_error"] for record in plans), default=None
            ),
            "probe_counts": sorted({record["iterations"] for record in plans}),
        },
        "dose_records": dose_records,
    }

    if task.name == "month":
        n_labels = len(task.labels)
        confusion = [[0] * n_labels for _ in range(6)]
        month_records: list[dict[str, Any]] = []
        for b, item in enumerate(base_prompts):
            readout = build_readout(model, layer, item["position"])
            metric_row = nearest_fitted_row(fit_coord, float(base_coord[b]), period)
            source = item["label_index"]
            for k in range(1, 7):
                t_to = float(base_coord[b]) + E1.chart_displacement(
                    label_map, source, (source + k) % n_labels, period
                )
                plan = sae.steer(
                    atom,
                    metric_row,
                    float(base_gate[b]),
                    np.asarray([float(base_coord[b])]),
                    np.asarray([t_to]),
                )
                delta = torch.tensor(plan["delta"], dtype=lift_t.dtype, device=device) @ lift_t
                patched = spliced_logits(readout, item["ids"], base_rows[b] + delta).detach().cpu()
                values = patched[candidate_ids].to(torch.float64).numpy()
                shift = realized_shift(int(np.argmax(values)), source, n_labels)
                confusion[k - 1][shift] += 1
                month_records.append(
                    {
                        "base_prompt_id": f"month-t{item['template_index']}-l{source}",
                        "requested_shift": k,
                        "realized_shift": shift,
                        "predicted_nats": plan["predicted_nats"],
                        "measured_nats": CALENDAR.full_vocab_kl(base_logits[b].cpu(), patched),
                    }
                )
        feature_report["gate4"] = {
            "confusion_requested_by_realized": confusion,
            "realized_mode_equals_request": [
                int(np.argmax(row)) == k for k, row in enumerate(confusion, start=1)
            ],
            "records": month_records,
        }
    return feature_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--features", default="weekday,month")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--pca-dim", type=int, required=True)
    parser.add_argument("--n-iter", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-nats", type=lambda s: [float(v) for v in s.split(",")], required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = [CALENDAR.task_from_name(name) for name in args.features.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()
    layer = CALENDAR.resolve_layers(model)[args.layer_index]
    reports = [run_feature(args, model, tokenizer, layer, task) for task in tasks]
    import gamfit

    report = {
        "protocol": {
            "model": args.model,
            "layer_index": args.layer_index,
            "model_dtype": "float32",
            "rank": args.rank,
            "pca_dim": args.pca_dim,
            "n_iter": args.n_iter,
            "seed": args.seed,
            "target_nats": args.target_nats,
            "prompt_bank_sha256": prompt_bank_sha256(tasks),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gamfit_file": gamfit.__file__,
        },
        "features": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for feature in reports:
        log(f"GATE3 {feature['feature']}: {json.dumps(feature['gate3'], sort_keys=True)}")
        if "gate4" in feature:
            log(f"GATE4 month: realized_mode_equals_request={feature['gate4']['realized_mode_equals_request']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
