#!/usr/bin/env python3
"""E1 — on-manifold calendar-circle steering (gam#2234).

The gam#2234 thesis: a manifold SAE steers by moving the CODE, not the ambient
vector — x' = x + a·(Φ(t⊕δ) − Φ(t))·B_k, where ⊕ rotates the day-of-week circle
phase. E1 tests the headline prediction on calendar features: moving the fitted
coordinate toward a target weekday by fractional doses up to k·(2π/7) should
move FULL-SOFTMAX next-token mass smoothly toward that target, while a
matched-norm flat-direction addition — the flat-SAE control — degrades off the
feature's circle.

Pipeline (all reused, provenance inline):
  1. GPT-2 calendar-activation capture + residual patching — the exact hook path
     from experiments/interchange/qwen_calendar_interchange.py (run_clean /
     run_patched register a forward hook on the transformer block, read/patch the
     last-position hidden state). The day-of-week circle features are the ones
     pinned by crates/gam-sae/tests/data/engels_gpt2_calendar_sae_indices.json
     (Engels et al. 2024, GPT-2 layer 7) and exercised by the calendar
     ground-truth benchmark (crates/gam-sae/tests/calendar_ground_truth_benchmark.rs).
  2. Fit gamfit.sae_manifold_fit on the weekday activation cloud (one periodic
     circle atom, d_atom=1, assignment='softmax' so the steer path is routed).
  3. Steer day-of-week phase by k·(2π/7) via model.steer(atom, t_from, t_to)
     (Rust sae_steer_delta / SaeManifoldTerm::steer_rows, landed f8d70743a).
  4. Patch the steered residual back into the model. Every prompt asks for the
     day AFTER its source label, so a k-day coordinate move targets weekday
     `(source + k + 1) mod 7`. Measure its full-softmax token probability and
     collateral `KL(patched_non_target || base_non_target)`, conditioning both
     distributions on every token except the intended target.
  5. Flat-SAE control: gamfit.sparse_dictionary_fit weekday latent, its decoder
     direction added at the SAME L2 norm as the on-manifold delta (fixed
     direction, no periodic structure — the thesis says it cannot cycle).

This needs a wheel + a GPT-2 checkout, so it runs on a GPU/CPU node, not in the
authoring sandbox. Everything is argparse-driven (no hardcoded homes).

MSI launch:
    python3 experiments/steering_e1/run_e1.py \
        --model gpt2 --layer-index 7 --k-atoms 1 \
        --out-dir experiments/steering_e1/out --seed 20260709
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

TAU = 2.0 * math.pi
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# GPT-2 activation capture + residual patch.
# Verbatim (with provenance) from experiments/interchange/qwen_calendar_interchange.py:
# load_model_and_tokenizer / resolve_layers / model_input_device / run_clean /
# run_patched / candidate_token_ids / full_vocab_kl / restricted_logprob. That
# harness is the single source of the calendar hook path; copied here so E1 is a
# self-contained, wheel-only-deployable script (no cross-experiment import).
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(model_name: str, cache_dir: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir or None)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    if not torch.cuda.is_available():
        model.to("cpu")
    return model, tok


def model_input_device(model: Any) -> Any:
    return next(model.parameters()).device


def resolve_layers(model: Any) -> Any:
    for root_name, layer_name in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layer_name):
            return getattr(root, layer_name)
    raise ValueError("could not locate transformer block list on model")


def candidate_token_ids(tokenizer: Any, prefix: str) -> list[int]:
    ids: list[int] = []
    for label in WEEKDAYS:
        enc = tokenizer.encode(prefix + label, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"candidate {prefix + label!r} tokenized to {len(enc)} tokens; "
                "choose a different --candidate-prefix"
            )
        ids.append(int(enc[0]))
    return ids


def run_clean(model: Any, tokenizer: Any, layer: Any, prompt: str):
    import torch

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model_input_device(model)) for k, v in enc.items()}
    position = int(enc["input_ids"].shape[1] - 1)
    captured: dict[str, Any] = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[0, position, :].detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(**enc, use_cache=False)
    finally:
        handle.remove()
    if "activation" not in captured:
        raise ValueError("activation hook did not fire")
    logits = out.logits[0, position, :].detach().float().cpu()
    return captured["activation"], logits


def run_patched(model: Any, tokenizer: Any, layer: Any, prompt: str, patched_activation: Any):
    import torch

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model_input_device(model)) for k, v in enc.items()}
    position = int(enc["input_ids"].shape[1] - 1)

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[0, position, :] = patched_activation.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(**enc, use_cache=False)
    finally:
        handle.remove()
    return out.logits[0, position, :].detach().float().cpu()


def _logits_array(logits: Any) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"logits must be a finite 1-D vocabulary vector; got {values.shape}")
    return values


def _log_softmax(logits: Any) -> np.ndarray:
    values = _logits_array(logits)
    shifted = values - values.max()
    return shifted - math.log(float(np.exp(shifted).sum()))


def token_probability(logits: Any, token_id: int) -> float:
    """Unconditional full-softmax probability of one vocabulary token."""
    logp = _log_softmax(logits)
    if not (0 <= token_id < logp.size):
        raise ValueError(f"token id {token_id} out of range for vocabulary size {logp.size}")
    return float(math.exp(float(logp[token_id])))


def weekday_token_probabilities(logits: Any, candidate_ids: list[int]) -> np.ndarray:
    """Full-softmax probabilities for the seven weekday tokens (never renormalized)."""
    logp = _log_softmax(logits)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if ids.shape != (len(WEEKDAYS),) or np.any(ids < 0) or np.any(ids >= logp.size):
        raise ValueError("candidate weekday token ids must be seven in-vocabulary ids")
    return np.exp(logp[ids])


def target_excluded_kl_model_to_base(
    patched_logits: Any,
    base_logits: Any,
    target_token_id: int,
) -> float:
    """Collateral KL on non-target tokens: KL(patched || base), target excluded.

    Each arm is conditioned on "the next token is not the intended target" before
    computing KL, so probability mass deliberately moved onto the target is not
    charged again as collateral damage.
    """
    patched = _logits_array(patched_logits)
    base = _logits_array(base_logits)
    if patched.shape != base.shape:
        raise ValueError(f"patched/base vocabulary shapes differ: {patched.shape} vs {base.shape}")
    if not (0 <= target_token_id < patched.size):
        raise ValueError(
            f"target token id {target_token_id} out of range for vocabulary size {patched.size}"
        )
    keep = np.ones(patched.size, dtype=bool)
    keep[target_token_id] = False
    model_log = _log_softmax(patched[keep])
    base_log = _log_softmax(base[keep])
    model_prob = np.exp(model_log)
    return max(float(np.sum(model_prob * (model_log - base_log))), 0.0)


# --------------------------------------------------------------------------- #
# Calendar prompt bank. Templates {label} = a weekday; the last-token residual
# encodes that weekday's day-of-week phase (the circle we fit + steer).
# --------------------------------------------------------------------------- #
FIT_TEMPLATES = (
    "Today is {label}. Tomorrow is",
    "If today is {label}, then tomorrow is",
    "The weekday after {label} is",
    "On a weekly calendar, {label} is followed by",
    "Yesterday was {label}, so today is",
    "The day that comes right after {label} is",
    "After {label} comes",
    "Counting forward from {label}, the next day is",
    "A day later than {label} is",
    "Following {label} on the calendar is",
)
# Held-out base contexts the steering interventions act on.
BASE_TEMPLATES = (
    "Starting on {label}, the next day is",
    "Calendar note: the day after {label} is",
)


@dataclass(eq=False)  # holds torch tensors; identity equality avoids tensor-== on compare
class CleanExample:
    template_index: int
    day_index: int
    prompt: str
    activation: Any  # torch tensor (p,)
    logits: Any      # torch tensor (vocab,)


def collect_cloud(model, tokenizer, layer, templates):
    examples: list[CleanExample] = []
    for ti, template in enumerate(templates):
        for di, label in enumerate(WEEKDAYS):
            prompt = template.format(label=label)
            act, logits = run_clean(model, tokenizer, layer, prompt)
            examples.append(CleanExample(ti, di, prompt, act, logits))
    return examples


def continuation_target_index(base_day_index: int, target_shift_days: int) -> int:
    """Next-token weekday after replacing the source day by source+k days."""
    if not (0 <= base_day_index < len(WEEKDAYS)):
        raise ValueError(f"base day index must be in [0, 7); got {base_day_index}")
    if not (1 <= target_shift_days < len(WEEKDAYS)):
        raise ValueError(f"target shift must be an integer in [1, 6]; got {target_shift_days}")
    return (base_day_index + target_shift_days + 1) % len(WEEKDAYS)


def parse_target_shifts(spec: str) -> list[int]:
    try:
        shifts = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--target-shifts must be comma-separated integers") from error
    if not shifts or len(set(shifts)) != len(shifts) or any(not 1 <= k <= 6 for k in shifts):
        raise ValueError("--target-shifts must contain unique integers from 1 through 6")
    return shifts


def parse_dose_fractions(spec: str) -> list[float]:
    try:
        fractions = [float(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--dose-fractions must be comma-separated numbers") from error
    if (
        not fractions
        or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions)
        or any(b <= a for a, b in zip(fractions, fractions[1:]))
        or fractions[0] != 0.0
        or fractions[-1] != 1.0
    ):
        raise ValueError(
            "--dose-fractions must be strictly increasing finite values in [0,1], including 0 and 1"
        )
    if not any(0.0 < value < 1.0 for value in fractions):
        raise ValueError("--dose-fractions must include at least one fractional interior dose")
    return fractions


# --------------------------------------------------------------------------- #
# Atom / latent selection: which fitted component carries day-of-week phase.
# --------------------------------------------------------------------------- #
def _phase_design(day_indices: np.ndarray) -> np.ndarray:
    """[cos θ, sin θ] design for the day-of-week phase θ = 2π·day/7."""
    theta = TAU * day_indices.astype(np.float64) / 7.0
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _phase_r2(signal: np.ndarray, day_indices: np.ndarray) -> float:
    """R^2 of a 1-D signal regressed on the day-of-week phase design."""
    d = np.column_stack([np.ones(len(day_indices)), _phase_design(day_indices)])
    coef, *_ = np.linalg.lstsq(d, signal, rcond=None)
    resid = signal - d @ coef
    tss = float(np.sum((signal - signal.mean()) ** 2))
    return 1.0 - float(np.sum(resid ** 2)) / max(tss, 1e-30)


def select_weekday_atom(model, day_indices: np.ndarray) -> tuple[int, int]:
    """Return the weekday atom and its train-split chart orientation."""
    best_k, best = 0, -np.inf
    best_orientation = 1
    truth = np.exp(1j * TAU * day_indices.astype(np.float64) / 7.0)
    for k in range(len(model.coords)):
        c = np.asarray(model.coords[k], dtype=float)
        coord = c[:, 0] if c.ndim == 2 else c
        chart = np.exp(1j * TAU * coord)
        forward = abs(np.mean(truth * np.conj(chart)))
        reverse = abs(np.mean(truth * chart))
        score = max(forward, reverse) ** 2
        orientation = 1 if forward >= reverse else -1
        if score > best:
            best, best_k, best_orientation = score, k, orientation
    log(f"weekday atom = {best_k} (circular R2={best:.4f}, orientation={best_orientation:+d})")
    return best_k, best_orientation


def select_flat_direction(flat_fit, X, day_indices):
    """Flat-SAE weekday latent's unit decoder direction (the fixed-direction
    control). Picks the latent whose code best separates days by phase R^2."""
    tr = flat_fit.transform(X)
    k = int(flat_fit.decoder.shape[0])
    codes = np.zeros((X.shape[0], k), dtype=np.float64)
    rows = np.arange(X.shape[0])[:, None]
    codes[rows, tr.indices.astype(np.int64)] = tr.codes.astype(np.float64)
    best_lat, best = 0, -np.inf
    for lat in range(k):
        col = codes[:, lat]
        if np.allclose(col, 0.0):
            continue
        r2 = _phase_r2(col, day_indices)
        if r2 > best:
            best, best_lat = r2, lat
    w = np.asarray(flat_fit.decoder[best_lat], dtype=np.float64)
    w = w / max(np.linalg.norm(w), 1e-30)
    log(f"flat weekday latent = {best_lat} (code phase R2={best:.4f})")
    return w, best_lat


# --------------------------------------------------------------------------- #
def steer_records(lm_model, sae_model, tokenizer, layer, atom, chart_orientation,
                  base_examples, metric_rows, base_coords, base_amplitudes,
                  candidate_ids, flat_dir, target_shifts, dose_fractions, lift=None):
    """Run manifold + flat steering across contexts, targets, and dose fractions.

    ``base_coords[i]`` is the fitted circle coordinate (length d_atom) of
    ``base_examples[i]`` on the weekday atom.
    """
    records: list[dict[str, Any]] = []
    for base, metric_row, t0_in, amplitude in zip(
        base_examples, metric_rows, base_coords, base_amplitudes
    ):
        b = base.day_index
        base_weekday_probs = weekday_token_probabilities(base.logits, candidate_ids)
        t0 = np.atleast_1d(np.asarray(t0_in, dtype=np.float64)).reshape(-1)
        for target_shift in target_shifts:
            target_index = continuation_target_index(b, target_shift)
            target_token_id = candidate_ids[target_index]
            base_target_probability = float(base_weekday_probs[target_index])
            for dose_fraction in dose_fractions:
                dose_days = float(target_shift) * dose_fraction
                dcoord = chart_orientation * dose_days / 7.0
                t_to = t0.copy()
                t_to[0] = t0[0] + dcoord  # periodic evaluator wraps the period-one chart
                plan = sae_model.steer(int(atom), int(metric_row), float(amplitude), t0, t_to)
                delta = np.asarray(plan["delta"], dtype=np.float64)
                if lift is not None:
                    # Exact ambient lift through the orthonormal PCA rows.
                    delta = delta @ lift
                import torch

                # --- on-manifold arm ---
                patched = base.activation + torch.from_numpy(delta.astype(np.float32))
                manifold_logits = run_patched(lm_model, tokenizer, layer, base.prompt, patched)

                # --- flat control: matched-norm fixed-direction addition ---
                flat_delta = np.linalg.norm(delta) * flat_dir
                patched_flat = base.activation + torch.from_numpy(flat_delta.astype(np.float32))
                flat_logits = run_patched(lm_model, tokenizer, layer, base.prompt, patched_flat)

                for arm, patched_logits in (
                    ("manifold", manifold_logits),
                    ("flat", flat_logits),
                ):
                    weekday_probs = weekday_token_probabilities(patched_logits, candidate_ids)
                    top = int(np.argmax(weekday_probs))
                    target_probability = float(weekday_probs[target_index])
                    collateral = target_excluded_kl_model_to_base(
                        patched_logits, base.logits, target_token_id
                    )
                    records.append({
                        "arm": arm,
                        "base_template": base.template_index,
                        "base_day": WEEKDAYS[b],
                        "base_day_index": b,
                        "target_shift_days": int(target_shift),
                        "target_day": WEEKDAYS[target_index],
                        "target_day_index": target_index,
                        "target_token_id": int(target_token_id),
                        "dose_fraction": float(dose_fraction),
                        "coordinate_delta_turns": float(dcoord),
                        "coordinate_delta_radians": float(dcoord * TAU),
                        "delta_norm": float(np.linalg.norm(delta)),
                        "steer_off_manifold_norm": (
                            float(plan["off_manifold_norm"])
                            if arm == "manifold" and plan.get("off_manifold_norm") is not None
                            else None
                        ),
                        "steer_predicted_nats": (
                            float(plan["predicted_nats"])
                            if arm == "manifold" and plan.get("predicted_nats") is not None
                            else None
                        ),
                        "realized_top_weekday": WEEKDAYS[top],
                        "realized_top_weekday_index": top,
                        "target_token_probability": target_probability,
                        "base_target_token_probability": base_target_probability,
                        "target_probability_mass_moved": (
                            target_probability - base_target_probability
                        ),
                        "collateral_kl_model_to_base_non_target": collateral,
                        "weekday_token_probabilities": [float(x) for x in weekday_probs],
                    })
    return records


def summarize(records, target_shifts, dose_fractions):
    """Endpoint accuracy and per-target fractional dose response for each arm."""
    out: dict[str, Any] = {}
    for arm in ("manifold", "flat"):
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        endpoint = [r for r in rs if r["dose_fraction"] == 1.0]
        acc = float(np.mean([
            r["realized_top_weekday_index"] == r["target_day_index"] for r in endpoint
        ]))
        dose_response: dict[str, Any] = {}
        for target_shift in target_shifts:
            by_fraction = {}
            for fraction in dose_fractions:
                sample = [
                    r for r in rs
                    if r["target_shift_days"] == target_shift
                    and r["dose_fraction"] == fraction
                ]
                by_fraction[str(fraction)] = {
                    "mean_target_token_probability": float(np.mean([
                        r["target_token_probability"] for r in sample
                    ])),
                    "mean_target_probability_mass_moved": float(np.mean([
                        r["target_probability_mass_moved"] for r in sample
                    ])),
                    "mean_collateral_kl_model_to_base_non_target": float(np.mean([
                        r["collateral_kl_model_to_base_non_target"] for r in sample
                    ])),
                }
            dose_response[str(target_shift)] = by_fraction
        out[arm] = {
            "endpoint_target_accuracy": acc,
            "mean_endpoint_target_token_probability": float(np.mean([
                r["target_token_probability"] for r in endpoint
            ])),
            "mean_endpoint_target_probability_mass_moved": float(np.mean([
                r["target_probability_mass_moved"] for r in endpoint
            ])),
            "mean_endpoint_collateral_kl_model_to_base_non_target": float(np.mean([
                r["collateral_kl_model_to_base_non_target"] for r in endpoint
            ])),
            "dose_response": dose_response,
        }
    return out


def write_outputs(out_dir: Path, meta, records, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e1_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "e1_summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n")

    lines = [
        "# E1 — Calendar-circle steering (gam#2234)",
        "",
        f"- Model `{meta['model']}` block `{meta['layer_index']}`; fit EV `{meta['fit_ev']:.4f}`; "
        f"weekday atom `{meta['weekday_atom']}`.",
        "",
        "| arm | endpoint target accuracy | endpoint target-token probability | "
        "probability mass moved | target-excluded KL(model || base) |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("manifold", "flat"):
        s = summary.get(arm)
        if s:
            lines.append(
                f"| {arm} | {s['endpoint_target_accuracy']:.3f} | "
                f"{s['mean_endpoint_target_token_probability']:.6f} | "
                f"{s['mean_endpoint_target_probability_mass_moved']:.6f} | "
                f"{s['mean_endpoint_collateral_kl_model_to_base_non_target']:.6f} |")
    lines += ["", "## Fractional dose-response by target shift", ""]
    lines.append("| target shift k | dose fraction | manifold mass moved | manifold collateral | "
                 "flat mass moved | flat collateral |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for target_shift, fractions in summary.get("manifold", {}).get("dose_response", {}).items():
        for fraction, manifold_row in fractions.items():
            flat_row = summary.get("flat", {}).get("dose_response", {}).get(
                target_shift, {}
            ).get(fraction, {})
            lines.append(
                f"| {target_shift} | {float(fraction):.3f} | "
                f"{manifold_row['mean_target_probability_mass_moved']:.6f} | "
                f"{manifold_row['mean_collateral_kl_model_to_base_non_target']:.6f} | "
                f"{flat_row.get('mean_target_probability_mass_moved', float('nan')):.6f} | "
                f"{flat_row.get('mean_collateral_kl_model_to_base_non_target', float('nan')):.6f} |"
            )
    lines += [
        "",
        "Prediction (gam#2234 E1): full-softmax mass moves smoothly toward the correct next-day "
        "target as dose increases, while target-excluded KL(model || base) remains below the "
        "matched-norm flat control at matched achieved effect.",
        "",
    ]
    (out_dir / "e1_results.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out_dir / 'e1_results.md'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--layer-index", type=int, default=7,
                    help="0-based block index to capture/patch (Engels GPT-2 calendar SAE is layer 7)")
    ap.add_argument("--k-atoms", type=int, default=1, help="manifold SAE atom count K")
    ap.add_argument("--flat-k", type=int, default=32, help="flat-SAE dictionary size (control)")
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--target-shifts", default="1,2,3,4,5,6",
                    help="comma-separated integer source-day shifts, each in 1..6")
    ap.add_argument("--dose-fractions", default="0,0.25,0.5,0.75,1",
                    help="strictly increasing target-dose fractions including 0, 1, and an interior value")
    ap.add_argument("--candidate-prefix", default=" ")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="fit-chart PCA dimension (0 disables; deltas are lifted back exactly)")
    ap.add_argument("--out-dir", default="experiments/steering_e1/out")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    target_shifts = parse_target_shifts(args.target_shifts)
    dose_fractions = parse_dose_fractions(args.dose_fractions)
    np.random.seed(args.seed)
    import gamfit

    log(f"loading {args.model}")
    model_lm, tok = load_model_and_tokenizer(args.model, args.cache_dir, args.dtype)
    layers = resolve_layers(model_lm)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(f"--layer-index must be in [0,{len(layers)}); got {args.layer_index}")
    layer = layers[args.layer_index]
    candidate_ids = candidate_token_ids(tok, args.candidate_prefix)
    log(f"candidate ids: {dict(zip(WEEKDAYS, candidate_ids))}")

    log("collecting disjoint fit and held-out weekday activation clouds")
    fit_examples = collect_cloud(model_lm, tok, layer, FIT_TEMPLATES)
    base_examples = collect_cloud(model_lm, tok, layer, BASE_TEMPLATES)
    X_fit_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in fit_examples]))
    X_base_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in base_examples]))
    fit_day_indices = np.asarray([ex.day_index for ex in fit_examples])
    log(f"fit X shape {X_fit_ambient.shape}; held-out X shape {X_base_ambient.shape}")

    # Wide-p treatment (same methodology as the committed OLMo fixtures): fit in
    # a per-layer PCA chart. The steering DELTAS are lifted back to ambient
    # through the orthonormal basis, so the intervention itself is exact —
    # x' = x + (delta_pca @ Vr) — and norms are preserved (Vr rows orthonormal).
    # Raw 3.5k-dim/84-row clouds put outer REML in the wide-p pathological
    # regime (probe 2026-07-10: all preprocessing variants refused); rank-r PCA
    # is where the calendar circle lives anyway (rank ~2-3).
    if args.pca_dim and args.pca_dim < X_fit_ambient.shape[1]:
        mu = X_fit_ambient.mean(0, keepdims=True)
        X_fit_centered = X_fit_ambient - mu
        _, svals, vt = np.linalg.svd(X_fit_centered, full_matrices=False)
        r = int(min(args.pca_dim, vt.shape[0]))
        lift = np.ascontiguousarray(vt[:r])            # (r, p) orthonormal rows
        X_fit = np.ascontiguousarray(X_fit_centered @ lift.T)
        X_base = np.ascontiguousarray((X_base_ambient - mu) @ lift.T)
        evr = float((svals[:r] ** 2).sum() / max((svals ** 2).sum(), 1e-30))
        log(f"train-only PCA chart: {X_fit.shape} (fit explained variance {evr:.4f})")
    else:
        lift = None
        X_fit = X_fit_ambient
        X_base = X_base_ambient

    log("fitting gamfit.sae_manifold_fit (periodic circle, softmax assignment)")
    sae_model = gamfit.sae_manifold_fit(
        X_fit, K=args.k_atoms, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=args.n_iter, random_state=args.seed)
    fit_ev = float(1.0 - np.sum((X_fit - np.asarray(sae_model.fitted)) ** 2)
                   / max(np.sum((X_fit - X_fit.mean(0)) ** 2), 1e-30))
    atom, chart_orientation = select_weekday_atom(sae_model, fit_day_indices)

    log("fitting flat-SAE control (gamfit.sparse_dictionary_fit)")
    flat_fit = gamfit.sparse_dictionary_fit(
        X_fit.astype(np.float32), min(args.flat_k, X_fit.shape[0] - 1),
        active=1, max_epochs=40)
    flat_dir, flat_lat = select_flat_direction(
        flat_fit, X_fit.astype(np.float32), fit_day_indices)
    if lift is not None:
        flat_dir = np.asarray(flat_dir, dtype=np.float64) @ lift
        norm = np.linalg.norm(flat_dir)
        if norm > 0:
            flat_dir = flat_dir / norm

    # OOS chart state is inferred without refitting. `steer` accepts a fitted-row
    # metric index, so each held-out point uses the nearest fitted point on the
    # period-one circle; under the default Euclidean metric this choice is exactly
    # immaterial, and it remains the chart-local choice for row-varying metrics.
    base_latents = sae_model.converged_latents(X_base)
    base_coords_array = np.asarray(base_latents["coords"][atom], dtype=float)
    base_assignments = np.asarray(base_latents["assignments"], dtype=float)
    fit_coords_array = np.asarray(sae_model.coords[atom], dtype=float)
    fit_turns = fit_coords_array[:, 0]
    base_turns = base_coords_array[:, 0]
    metric_rows = []
    for turn in base_turns:
        circular_distance = np.abs((fit_turns - turn + 0.5) % 1.0 - 0.5)
        metric_rows.append(int(np.argmin(circular_distance)))
    base_coords = [base_coords_array[row] for row in range(len(base_examples))]
    base_amplitudes = [float(base_assignments[row, atom]) for row in range(len(base_examples))]

    log(
        f"steering {len(base_examples)} base contexts × targets {target_shifts} × "
        f"dose fractions {dose_fractions} (manifold + flat)"
    )
    records = steer_records(
        model_lm, sae_model, tok, layer, atom, chart_orientation,
        base_examples, metric_rows, base_coords, base_amplitudes,
        candidate_ids, flat_dir, target_shifts, dose_fractions, lift=lift)
    summary = summarize(records, target_shifts, dose_fractions)

    meta = {
        "model": args.model, "layer_index": args.layer_index, "k_atoms": args.k_atoms,
        "flat_k": int(flat_fit.decoder.shape[0]),
        "flat_latent": int(flat_lat), "weekday_atom": int(atom), "fit_ev": fit_ev,
        "chart_orientation": int(chart_orientation),
        "n_fit_rows": int(len(fit_examples)), "n_base_rows": int(len(base_examples)),
        "target_shifts": target_shifts, "dose_fractions": dose_fractions,
        "seed": args.seed,
    }
    write_outputs(Path(args.out_dir), meta, records, summary)

    # E2 (gam#2234) is part of the acceptance result, so analysis failures are
    # fatal rather than silently converting a missing verdict into a successful E1.
    import analyze_collateral

    analyze_collateral.run(Path(args.out_dir))

    for arm in ("manifold", "flat"):
        s = summary.get(arm)
        if s:
            print(
                f"E1_{arm.upper()} endpoint_accuracy={s['endpoint_target_accuracy']:.4f} "
                f"endpoint_target_prob={s['mean_endpoint_target_token_probability']:.6f} "
                f"endpoint_mass_moved={s['mean_endpoint_target_probability_mass_moved']:.6f} "
                f"endpoint_collateral={s['mean_endpoint_collateral_kl_model_to_base_non_target']:.6f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
