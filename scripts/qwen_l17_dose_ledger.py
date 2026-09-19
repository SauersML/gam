#!/usr/bin/env python3
"""#2263 gate 2: the frozen layer-17 dose ledger on Qwen3.6-35B-A3B.

The driver this gate named (``dose_calibration_real.py``, with its prompt bank and
harvest caches) was never committed, and the environment it ran in is gone. This
is the committed restatement: the same model, the same frozen settings
(``FROZEN_PROTOCOL``), a committed prompt bank, and a ledger in exactly the form
``scripts/qwen_l17_exact_dose_replay.py`` scores. The fit, the dose solve and the
steering move are the Rust library's public entries; the torch code is the
model-interop path.

For each feature (weekday, month, color):

1. **Fit cloud.** Every label under the first ``max_templates`` fit templates
   gives one label-position residual row at the hook layer, with a rank-``rank``
   output-Fisher factor from ``gamfit.torch.harvest``. The weights stay bfloat16
   and the harvest takes its math in float32. Rows and factors are cached as
   ``harvest_cache_{feature}_L{layer}_n{rows}.npz`` and re-used when present.
2. **Chart.** ``gamfit.sae.sae_manifold_fit`` fits one circle atom for
   ``fit_iterations`` iterations in a train-only PCA chart of ``chart_dim``
   dimensions, with the projected shard installed.
3. **Bases.** A seeded permutation draws ``bases`` base prompts from the
   evaluation templates. The first half is ``split=calibration``, the rest
   ``split=heldout``.
4. **Dose floor.** Each base is re-run ``floor_repetitions`` times with its own
   residual spliced back unchanged. The floor is ``floor_multiplier`` times the
   largest ``KL(p_base || p_repeat)``, the smallest dose the forward resolves.
5. **Dose ladder.** A base's reference dose is the measured KL of the half-turn
   move along its chart (``ManifoldSAE.steer`` at the base's own gate). Each
   fraction names the target ``fraction × reference``. A target at or under the
   floor is recorded as ``below_floor`` and not requested.
6. **Plans.** Every other target goes through ``ManifoldSAE.steer_to_target``
   with a plan-aware probe. The probe writes the move in the model's dtype,
   measures ``KL(p_base || p_patched)`` over the full vocabulary, and returns the
   exact directional Fisher dose ``½ (Jδ)ᵀ F (Jδ)`` of the move it applied, by
   one JVP. The returned plan, whose ``predicted_nats`` is that exact directional
   value, is one ledger row. A typed refusal is recorded, never dropped.

Each feature's result is cached as ``feature_{feature}.json``, so a run cut short
resumes where it stopped. The ledger is written once every feature is done.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> Any:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATRIX = _load_module("qwen_target_dose_matrix", "scripts/qwen_target_dose_matrix.py")
CALENDAR = MATRIX.CALENDAR
E1 = MATRIX.E1
TAU = MATRIX.TAU
log = MATRIX.log


# The frozen #2249 settings: DOSE_FLOOR_MULT=30, FLOOR_REPS=5, FRACS=0.01..0.6,
# MAXTPL=6, NBASES=10, NITER=40 at layer 17, bfloat16 weights and float32 harvest
# math. The historical ledger held 300 edits over three features and ten bases, so
# FRACS is ten fractions, spaced geometrically across its stated range. The chart
# dimension belongs to this restatement, and every ledger records it. The dose solve
# takes no accuracy or probe budget: it lands each dose at the representation limit
# of the displacement.
FROZEN_PROTOCOL: dict[str, Any] = {
    "model": "Qwen/Qwen3.6-35B-A3B",
    "layer": 17,
    "model_dtype": "bfloat16",
    "harvest_dtype": "float32",
    "features": ("weekday", "month", "color"),
    "rank": 8,
    "seed": 0,
    "fractions": tuple(float(value) for value in np.geomspace(0.01, 0.6, 10)),
    "floor_multiplier": 30,
    "floor_repetitions": 5,
    "max_templates": 6,
    "bases": 10,
    "fit_iterations": 40,
    "chart_dim": 8,
}


# --------------------------------------------------------------------------- #
# The committed prompt bank
# --------------------------------------------------------------------------- #
EXTRA_FIT_TEMPLATES = {
    "weekday": (
        "Every week, {label} is followed by",
        "The calendar page shows {label}, so the next day is",
    ),
    "month": (
        "The month that follows {label} is",
        "After {label}, the calendar turns to",
    ),
}

COLOR_TASK = CALENDAR.CalendarTask(
    "color",
    ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"),
    (
        "The paint on the fence is {label}, and the gate is",
        "She chose a {label} scarf to go with",
        "On the color wheel, {label} sits next to",
        "The traffic sign was painted {label}. The next sign was",
        "His favorite crayon is {label}, but hers is",
        "The logo is {label} on a background of",
    ),
    (
        "The car in the driveway is {label}, and the bike is",
        "The flag has a {label} stripe beside a",
        "We painted the kitchen wall {label} and the door",
    ),
)


def gate2_tasks(names: tuple[str, ...]) -> list[Any]:
    """The committed prompt bank, fit templates first and evaluation templates second."""
    tasks = []
    for name in names:
        if name == "color":
            tasks.append(COLOR_TASK)
            continue
        task = CALENDAR.task_from_name(name)
        tasks.append(
            CALENDAR.CalendarTask(
                task.name,
                task.labels,
                task.train_templates + EXTRA_FIT_TEMPLATES[name],
                task.eval_templates,
            )
        )
    return tasks


# --------------------------------------------------------------------------- #
# Pure helpers (numpy only; pinned by tests/test_qwen_l17_dose_ledger_contract.py)
# --------------------------------------------------------------------------- #
def base_split(pool_size: int, bases: int, seed: int) -> list[tuple[int, str]]:
    """``bases`` distinct pool indices in seeded order; the first half calibrates."""
    if not 2 <= bases <= pool_size:
        raise ValueError(f"need 2 <= bases <= pool size; got bases={bases}, pool={pool_size}")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(pool_size)[:bases]
    half = bases // 2
    return [
        (int(index), "calibration" if position < half else "heldout")
        for position, index in enumerate(order)
    ]


def dose_floor(repeat_nats: list[float], multiplier: float) -> float:
    """The smallest resolvable dose: ``multiplier`` times the largest repeat KL."""
    if not repeat_nats:
        raise ValueError("the dose floor needs at least one repeated forward")
    worst = max(repeat_nats)
    if not (np.isfinite(worst) and worst >= 0.0):
        raise ValueError(f"repeated-forward KL must be finite and non-negative; got {repeat_nats}")
    return float(multiplier) * float(worst)


def dose_ladder(
    reference_nats: float, fractions: tuple[float, ...], floor_nats: float
) -> list[tuple[int, float, bool]]:
    """``(fraction index, target, above the floor)`` for every fraction of the reference."""
    ladder = []
    for index, fraction in enumerate(fractions):
        target = float(fraction) * float(reference_nats)
        ladder.append((index, target, target > floor_nats))
    return ladder


def ledger_row(
    plan: dict[str, Any],
    *,
    feature: str,
    atom: int,
    base_prompt_id: str,
    split: str,
    fraction_index: int,
    target_nats: float,
) -> dict[str, Any]:
    """One scored intervention, copied from the public plan and never re-priced."""
    return {
        "intervention_id": f"{base_prompt_id}-f{fraction_index}",
        "feature": feature,
        "atom": atom,
        "base_prompt_id": base_prompt_id,
        "split": split,
        "fraction_index": fraction_index,
        "target_nats": float(target_nats),
        "predicted_nats": float(plan["predicted_nats"]),
        "predicted_nats_kind": str(plan["predicted_nats_kind"]),
        "exact_directional_nats": float(plan["exact_directional_nats"]),
        "measured_nats": float(plan["measured_nats"]),
        "effective_delta": [float(value) for value in plan["effective_delta"]],
        "resident_metric_nats": float(plan["resident_metric_nats"]),
        "resident_metric_nats_kind": str(plan["resident_metric_nats_kind"]),
        "iterations": int(plan["iterations"]),
        "displacement": float(plan["displacement"]),
    }


def assemble_ledger(
    protocol: dict[str, Any], rows_by_feature: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    rows = [row for feature in protocol["features"] for row in rows_by_feature[feature]]
    return {"protocol": {**protocol, "row_count": len(rows)}, "rows": rows}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_identity(snapshot: Path) -> tuple[str, str]:
    """``(revision, sha256)`` of a Hugging Face snapshot directory.

    Every weight file in a hub snapshot is a symlink to a blob named by its own
    content hash, so the sorted ``(relative path, blob name)`` pairs identify the
    model without re-reading tens of gigabytes. A regular file is hashed directly.
    """
    entries = []
    for path in sorted(snapshot.rglob("*")):
        relative = str(path.relative_to(snapshot))
        if path.is_symlink():
            entries.append([relative, Path(os.readlink(path)).name])
        elif path.is_file():
            entries.append([relative, file_sha256(path)])
    if not entries:
        raise ValueError(f"snapshot {snapshot} holds no files")
    return snapshot.name, hashlib.sha256(json.dumps(entries).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Model interop
# --------------------------------------------------------------------------- #
def float32_readout(model: Any, layer: Any, position: int) -> Any:
    """``build_readout`` from the gates 3/4 producer, with its logits taken in float32."""
    import torch

    inner = MATRIX.build_readout(model, layer, position)

    class Float32Readout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = inner
            self.site = inner.site

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.inner(input_ids).to(torch.float32)

    return Float32Readout()


def run_feature(
    model: Any, tokenizer: Any, layer: Any, task: Any, atom: int, out_dir: Path
) -> dict[str, Any]:
    import torch
    import gamfit
    from gamfit.torch.harvest import HarvestShard, harvest_output_fisher_factors

    protocol = FROZEN_PROTOCOL
    result_path = out_dir / f"feature_{task.name}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        log(f"{task.name}: re-using {len(cached['rows'])} rows from {result_path}")
        return cached

    device = next(model.parameters()).device

    def prompt(template_index: int, template: str, label_index: int, label: str) -> dict[str, Any]:
        ids, position = MATRIX.label_position(tokenizer, template, label)
        return {
            "template_index": template_index,
            "label_index": label_index,
            "ids": torch.tensor([ids], dtype=torch.long, device=device),
            "position": position,
        }

    fit_prompts = [
        prompt(template_index, template, label_index, label)
        for template_index, template in enumerate(task.train_templates[: protocol["max_templates"]])
        for label_index, label in enumerate(task.labels)
    ]
    pool = [
        prompt(template_index, template, label_index, label)
        for template_index, template in enumerate(task.eval_templates)
        for label_index, label in enumerate(task.labels)
    ]

    cache = out_dir / f"harvest_cache_{task.name}_L{protocol['layer']}_n{len(fit_prompts)}.npz"
    if cache.exists():
        with np.load(cache) as payload:
            x_fit = np.asarray(payload["X"], dtype=np.float64)
            u_fit = np.asarray(payload["U"], dtype=np.float64)
        log(f"{task.name}: re-using harvest cache {cache}")
    else:
        log(f"{task.name}: harvesting rank-{protocol['rank']} output-Fisher factors on {len(fit_prompts)} fit prompts")
        xs, us = [], []
        for item in fit_prompts:
            readout = float32_readout(model, layer, item["position"])
            shard = harvest_output_fisher_factors(
                readout, readout.site, item["ids"], rank=protocol["rank"], seed=protocol["seed"]
            )
            if shard.X.shape[0] != 1:
                raise ValueError(f"label site harvested {shard.X.shape[0]} rows; expected one")
            xs.append(np.asarray(shard.X, dtype=np.float64))
            us.append(np.asarray(shard.U, dtype=np.float64))
        x_fit = np.concatenate(xs, axis=0)
        u_fit = np.concatenate(us, axis=0)
        np.savez(cache, X=x_fit, U=u_fit)

    mean, lift = MATRIX.chart_projection(x_fit, protocol["chart_dim"])
    x_fit_chart = MATRIX.to_chart(x_fit, mean, lift)
    chart_shard = HarvestShard(
        X=x_fit_chart,
        U=MATRIX.project_factors(u_fit, lift),
        mass_residual=None,
        rank=protocol["rank"],
        factor_kind="uncertified_approximation",
        provenance="output_fisher",
    )
    log(f"{task.name}: fitting one circle atom in a {x_fit_chart.shape[1]}-dim chart with the shard")
    sae = gamfit.sae.sae_manifold_fit(
        x_fit_chart,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=protocol["fit_iterations"],
        random_state=protocol["seed"],
        fisher_factors=chart_shard,
    )
    fit_label_index = np.asarray([item["label_index"] for item in fit_prompts])
    fit_coord = np.asarray(sae.coords[0], dtype=np.float64)[:, 0]
    recovery = [
        (E1.circular_recovery(fit_coord, fit_label_index, len(task.labels), period), period)
        for period in (1.0, TAU)
    ]
    (recovery_r2, orientation), period = max(recovery, key=lambda entry: entry[0][0])
    log(f"{task.name}: structure recovery R2={recovery_r2:.4f} period={period} orientation={orientation:+d}")

    bases = []
    for index, split in base_split(len(pool), protocol["bases"], protocol["seed"]):
        item = pool[index]
        readout = float32_readout(model, layer, item["position"])
        row, logits = MATRIX.capture(readout, item["ids"])
        bases.append({"item": item, "split": split, "readout": readout, "row": row, "logits": logits})
    x_base = np.stack([base["row"].to(torch.float64).cpu().numpy() for base in bases])
    latents = sae.converged_latents(MATRIX.to_chart(x_base, mean, lift))
    base_coord = np.asarray(latents["coords"][0], dtype=np.float64)[:, 0]
    base_gate = np.asarray(latents["assignments"], dtype=np.float64)[:, 0]
    lift64 = torch.from_numpy(lift).to(device=device, dtype=torch.float64)

    rows: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    below_floor: list[dict[str, Any]] = []
    base_reports: list[dict[str, Any]] = []
    for b, base in enumerate(bases):
        item, readout, row, logits = base["item"], base["readout"], base["row"], base["logits"]
        base_prompt_id = f"{task.name}-e{item['template_index']}-l{item['label_index']}"
        repeat_nats = [
            CALENDAR.full_vocab_kl(logits.cpu(), MATRIX.spliced_logits(readout, item["ids"], row).detach().cpu())
            for _ in range(protocol["floor_repetitions"])
        ]
        floor_nats = dose_floor(repeat_nats, protocol["floor_multiplier"])
        coord = float(base_coord[b])
        gate = float(base_gate[b])
        metric_row = MATRIX.nearest_fitted_row(fit_coord, coord, period)
        half_turn = sae.steer(0, metric_row, gate, np.asarray([coord]), np.asarray([coord + 0.5 * period]))
        half_move = torch.tensor(half_turn["delta"], dtype=torch.float64, device=device) @ lift64
        reference_nats = CALENDAR.full_vocab_kl(
            logits.cpu(),
            MATRIX.spliced_logits(readout, item["ids"], row + half_move.to(row.dtype)).detach().cpu(),
        )
        base_reports.append(
            {
                "base_prompt_id": base_prompt_id,
                "split": base["split"],
                "metric_row": metric_row,
                "coordinate": coord,
                "gate": gate,
                "repeat_nats": repeat_nats,
                "floor_nats": floor_nats,
                "reference_nats": reference_nats,
            }
        )

        def probe(
            plan: dict[str, Any],
            base_row: Any = row,
            base_logits: Any = logits,
            base_readout: Any = readout,
            input_ids: Any = item["ids"],
        ) -> dict[str, Any]:
            requested = torch.tensor(plan["delta"], dtype=torch.float64, device=device) @ lift64
            patched_row = base_row + requested.to(base_row.dtype)
            applied = patched_row - base_row
            patched = MATRIX.spliced_logits(base_readout, input_ids, patched_row).detach()
            return {
                "effective_delta": (applied.to(torch.float64) @ lift64.T).cpu().tolist(),
                "exact_directional_nats": MATRIX.exact_directional_nats(
                    base_readout, input_ids, base_row, applied
                ),
                "measured_nats": CALENDAR.full_vocab_kl(base_logits.cpu(), patched.cpu()),
                "certified_attainable_upper_nats": None,
            }

        for fraction_index, target_nats, above_floor in dose_ladder(
            reference_nats, protocol["fractions"], floor_nats
        ):
            record = {
                "feature": task.name,
                "base_prompt_id": base_prompt_id,
                "split": base["split"],
                "fraction_index": fraction_index,
                "target_nats": target_nats,
            }
            if not above_floor:
                below_floor.append(record)
                continue
            request = {
                "atom_k": 0,
                "metric_row": metric_row,
                "target_nats": target_nats,
                "t_from": [coord],
                "direction": [1.0],
            }
            try:
                plan = sae.steer_to_target(request, probe)
            except ValueError as error:
                refusals.append({**record, "refusal": str(error)})
                continue
            rows.append(
                ledger_row(
                    plan,
                    feature=task.name,
                    atom=atom,
                    base_prompt_id=base_prompt_id,
                    split=base["split"],
                    fraction_index=fraction_index,
                    target_nats=target_nats,
                )
            )

    result = {
        "feature": task.name,
        "atom": atom,
        "structure_recovery_r2": float(recovery_r2),
        "chart_orientation": int(orientation),
        "chart_period": float(period),
        "chart_dim": int(x_fit_chart.shape[1]),
        "fit_rows": len(fit_prompts),
        "harvest_cache": str(cache),
        "harvest_cache_sha256": file_sha256(cache),
        "bases": base_reports,
        "rows": rows,
        "refusals": refusals,
        "below_floor": below_floor,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log(f"{task.name}: {len(rows)} rows, {len(refusals)} typed refusals, {len(below_floor)} targets under the floor")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="local Hugging Face snapshot of the frozen model")
    parser.add_argument("--gam-sha", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = FROZEN_PROTOCOL
    tasks = gate2_tasks(protocol["features"])
    args.out.mkdir(parents=True, exist_ok=True)
    revision, model_sha256 = snapshot_identity(args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, dtype=torch.bfloat16, attn_implementation="eager", experts_implementation="eager"
    ).to("cuda:0")
    model.eval()
    model.requires_grad_(False)
    layer = CALENDAR.resolve_layers(model)[protocol["layer"]]
    hook_module = next(name for name, module in model.named_modules() if module is layer)
    log(f"model {protocol['model']} revision {revision}; hook module {hook_module}")
    features = [run_feature(model, tokenizer, layer, task, atom, args.out) for atom, task in enumerate(tasks)]
    import gamfit

    ledger = assemble_ledger(
        {
            "model": protocol["model"],
            "model_revision": revision,
            "model_sha256": model_sha256,
            "layer": protocol["layer"],
            "hook_module": hook_module,
            "steering_mode": "steer_to_target_along_the_fitted_chart",
            "model_dtype": protocol["model_dtype"],
            "harvest_dtype": protocol["harvest_dtype"],
            "gam_sha": args.gam_sha,
            "wheel_sha256": args.wheel_sha256,
            "driver_sha256": file_sha256(Path(__file__)),
            "prompt_bank_sha256": MATRIX.prompt_bank_sha256(tasks),
            "harvest_cache_sha256": {feature["feature"]: feature["harvest_cache_sha256"] for feature in features},
            "seed": protocol["seed"],
            "fractions": list(protocol["fractions"]),
            "floor_multiplier": protocol["floor_multiplier"],
            "floor_repetitions": protocol["floor_repetitions"],
            "max_templates": protocol["max_templates"],
            "bases": protocol["bases"],
            "fit_iterations": protocol["fit_iterations"],
            "features": list(protocol["features"]),
            "rank": protocol["rank"],
            "chart_dim": protocol["chart_dim"],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gamfit_file": gamfit.__file__,
        },
        {feature["feature"]: feature["rows"] for feature in features},
    )
    ledger_path = args.out / "ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    log(f"LEDGER rows={ledger['protocol']['row_count']} at {ledger_path}")
    for feature in features:
        log(
            f"FEATURE {feature['feature']}: rows={len(feature['rows'])} "
            f"refusals={len(feature['refusals'])} below_floor={len(feature['below_floor'])} "
            f"recovery_r2={feature['structure_recovery_r2']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
