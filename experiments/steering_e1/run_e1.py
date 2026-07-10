#!/usr/bin/env python3
"""E1 — on-manifold calendar-circle steering (gam#2234).

The gam#2234 thesis: a manifold SAE steers by moving the CODE, not the ambient
vector — x' = x + a·(Φ(t⊕δ) − Φ(t))·B_k, where ⊕ rotates the day-of-week circle
phase. E1 tests the headline prediction on GPT-2's calendar features: rotating
the fitted circle coordinate by k·(2π/7) should cycle the next-token weekday
distribution SMOOTHLY (the probability peak tracks the rotation), while a
matched-norm flat-direction addition — the flat-SAE control — degrades into
noise off the feature's circle.

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
  4. Patch the steered residual back into GPT-2, measure restricted next-token
     weekday probabilities + full-vocab KL(base || patched).
  5. Flat-SAE control: gamfit.sparse_dictionary_fit weekday latent, its decoder
     direction added at the SAME L2 norm as the on-manifold delta (fixed
     direction, no periodic structure — the thesis says it cannot cycle).

This needs a wheel + a GPT-2 checkout, so it runs on a GPU/CPU node, not in the
authoring sandbox. Everything is argparse-driven (no hardcoded homes).

MSI launch:
    python3 experiments/steering_e1/run_e1.py \
        --model gpt2 --layer-index 7 --k-atoms 1 --harmonics 3 \
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


def full_vocab_kl(clean_logits: Any, patched_logits: Any) -> float:
    import torch

    logp = torch.log_softmax(clean_logits.to(torch.float64), dim=-1)
    logq = torch.log_softmax(patched_logits.to(torch.float64), dim=-1)
    p = logp.exp()
    return float((p * (logp - logq)).sum().item())


def restricted_probs(logits: Any, candidate_ids: list[int]) -> np.ndarray:
    v = logits[candidate_ids].numpy().astype(np.float64)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


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


def select_weekday_atom(model, day_indices: np.ndarray) -> int:
    """Atom whose fitted coordinate best tracks the day-of-week phase."""
    best_k, best = 0, -np.inf
    for k in range(len(model.coords)):
        c = np.asarray(model.coords[k], dtype=float)
        coord = c[:, 0] if c.ndim == 2 else c
        # Circular coordinate -> score cos and sin of it against the phase design.
        score = max(_phase_r2(np.cos(coord), day_indices), _phase_r2(np.sin(coord), day_indices))
        if score > best:
            best, best_k = score, k
    log(f"weekday atom = {best_k} (phase R2={best:.4f})")
    return best_k


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
def steer_records(model, tokenizer, layer, atom, base_examples, base_rows, base_coords,
                  base_amplitudes,
                  candidate_ids, flat_dir, ks, lift=None):
    """Run manifold + flat steering across base examples × rotation counts k.

    ``base_coords[i]`` is the fitted circle coordinate (length d_atom) of
    ``base_examples[i]`` on the weekday atom.
    """
    records: list[dict[str, Any]] = []
    for base, metric_row, t0_in, amplitude in zip(
        base_examples, base_rows, base_coords, base_amplitudes
    ):
        b = base.day_index
        base_probs = restricted_probs(base.logits, candidate_ids)
        t0 = np.atleast_1d(np.asarray(t0_in, dtype=np.float64)).reshape(-1)
        for k in ks:
            dcoord = k * TAU / 7.0
            t_to = t0.copy()
            t_to[0] = t0[0] + dcoord  # circle retract wraps mod period Rust-side
            plan = model.steer(int(atom), int(metric_row), float(amplitude), t0, t_to)
            delta = np.asarray(plan["delta"], dtype=np.float64)
            if lift is not None:
                # Exact ambient lift through the orthonormal PCA rows.
                delta = delta @ lift
            import torch

            # --- on-manifold arm ---
            patched = base.activation + torch.from_numpy(delta.astype(np.float32))
            pl = run_patched(model, tokenizer, layer, base.prompt, patched)
            man_probs = restricted_probs(pl, candidate_ids)
            man_kl = max(full_vocab_kl(base.logits, pl), 0.0)

            # --- flat control: matched-norm fixed-direction addition ---
            flat_delta = np.linalg.norm(delta) * flat_dir
            patched_f = base.activation + torch.from_numpy(flat_delta.astype(np.float32))
            pl_f = run_patched(model, tokenizer, layer, base.prompt, patched_f)
            flat_probs = restricted_probs(pl_f, candidate_ids)
            flat_kl = max(full_vocab_kl(base.logits, pl_f), 0.0)

            for arm, probs, kl in (
                ("manifold", man_probs, man_kl),
                ("flat", flat_probs, flat_kl),
            ):
                top = int(np.argmax(probs))
                records.append({
                    "arm": arm,
                    "base_template": base.template_index,
                    "base_day": WEEKDAYS[b],
                    "base_day_index": b,
                    "k": int(k),
                    "delta_norm": float(np.linalg.norm(delta)),
                    "off_manifold_norm": (
                        float(plan["off_manifold_norm"]) if plan.get("off_manifold_norm") is not None else None),
                    "predicted_nats": (
                        float(plan["predicted_nats"]) if plan.get("predicted_nats") is not None else None),
                    "realized_top_day": WEEKDAYS[top],
                    "realized_top_index": top,
                    "realized_shift": int((top - b) % 7),
                    "target_prob_plus": float(probs[(b + k) % 7]),
                    "target_prob_minus": float(probs[(b - k) % 7]),
                    "base_target_prob_plus": float(base_probs[(b + k) % 7]),
                    "kl_base_to_patched": kl,
                    "weekday_probs": [float(x) for x in probs],
                })
    return records


def summarize(records, ks):
    """Per-arm: cyclic-advance steering accuracy (best of ± orientation), mean
    target-day probability, mean collateral KL, and the dose-response series."""
    out: dict[str, Any] = {}
    for arm in ("manifold", "flat"):
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        # Orientation: the fit's circle sign is arbitrary; score both directions.
        acc_plus = np.mean([r["realized_top_index"] == (r["base_day_index"] + r["k"]) % 7 for r in rs])
        acc_minus = np.mean([r["realized_top_index"] == (r["base_day_index"] - r["k"]) % 7 for r in rs])
        orient = "+" if acc_plus >= acc_minus else "-"
        acc = float(max(acc_plus, acc_minus))
        tp_key = "target_prob_plus" if orient == "+" else "target_prob_minus"
        dose = {}
        for k in ks:
            rk = [r for r in rs if r["k"] == k]
            dose[str(k)] = {
                "mean_target_prob": float(np.mean([r[tp_key] for r in rk])) if rk else float("nan"),
                "mean_kl": float(np.mean([r["kl_base_to_patched"] for r in rk])) if rk else float("nan"),
            }
        out[arm] = {
            "orientation": orient,
            "cyclic_advance_accuracy": acc,
            "mean_target_prob": float(np.mean([r[tp_key] for r in rs])),
            "mean_kl": float(np.mean([r["kl_base_to_patched"] for r in rs])),
            "dose_response": dose,
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
        "| arm | orientation | cyclic-advance accuracy | mean target-day prob | mean collateral KL |",
        "|---|:---:|---:|---:|---:|",
    ]
    for arm in ("manifold", "flat"):
        s = summary.get(arm)
        if s:
            lines.append(
                f"| {arm} | {s['orientation']} | {s['cyclic_advance_accuracy']:.3f} | "
                f"{s['mean_target_prob']:.4f} | {s['mean_kl']:.4f} |")
    lines += ["", "## Dose-response (mean target-day prob / mean KL by rotation k)", ""]
    ks = sorted({int(k) for k in summary.get("manifold", {}).get("dose_response", {})})
    lines.append("| k | manifold target-prob | manifold KL | flat target-prob | flat KL |")
    lines.append("|---:|---:|---:|---:|---:|")
    for k in ks:
        m = summary["manifold"]["dose_response"][str(k)]
        fdr = summary.get("flat", {}).get("dose_response", {}).get(str(k), {"mean_target_prob": float("nan"), "mean_kl": float("nan")})
        lines.append(f"| {k} | {m['mean_target_prob']:.4f} | {m['mean_kl']:.4f} | "
                     f"{fdr['mean_target_prob']:.4f} | {fdr['mean_kl']:.4f} |")
    lines += [
        "",
        "Prediction (gam#2234 E1): the manifold arm's target-day probability tracks the rotation k "
        "(smooth cycling) at low collateral KL; the flat matched-norm control cannot cycle and its "
        "target-day probability degrades as k grows off the circle.",
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
    ap.add_argument("--harmonics", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--max-k", type=int, default=6, help="max rotation count; sweeps k=1..max_k (2π/7 each)")
    ap.add_argument("--candidate-prefix", default=" ")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="fit-chart PCA dimension (0 disables; deltas are lifted back exactly)")
    ap.add_argument("--out-dir", default="experiments/steering_e1/out")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
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

    log("collecting weekday activation cloud (fit + base templates)")
    fit_examples = collect_cloud(model_lm, tok, layer, FIT_TEMPLATES)
    base_examples = collect_cloud(model_lm, tok, layer, BASE_TEMPLATES)
    all_examples = fit_examples + base_examples
    X_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in all_examples]))
    day_indices = np.asarray([ex.day_index for ex in all_examples])
    log(f"cloud X shape {X_ambient.shape}")

    # Wide-p treatment (same methodology as the committed OLMo fixtures): fit in
    # a per-layer PCA chart. The steering DELTAS are lifted back to ambient
    # through the orthonormal basis, so the intervention itself is exact —
    # x' = x + (delta_pca @ Vr) — and norms are preserved (Vr rows orthonormal).
    # Raw 3.5k-dim/84-row clouds put outer REML in the wide-p pathological
    # regime (probe 2026-07-10: all preprocessing variants refused); rank-r PCA
    # is where the calendar circle lives anyway (rank ~2-3).
    if args.pca_dim and args.pca_dim < X_ambient.shape[1]:
        mu = X_ambient.mean(0, keepdims=True)
        Xc = X_ambient - mu
        _, svals, vt = np.linalg.svd(Xc, full_matrices=False)
        r = int(min(args.pca_dim, vt.shape[0]))
        lift = np.ascontiguousarray(vt[:r])            # (r, p) orthonormal rows
        X = np.ascontiguousarray(Xc @ lift.T)          # (n, r) fit chart
        evr = float((svals[:r] ** 2).sum() / max((svals ** 2).sum(), 1e-30))
        log(f"PCA chart: {X.shape} (explained variance {evr:.4f})")
    else:
        lift = None
        X = X_ambient

    log("fitting gamfit.sae_manifold_fit (periodic circle, softmax assignment)")
    model = gamfit.sae_manifold_fit(
        X, K=args.k_atoms, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=args.n_iter, random_state=args.seed)
    fit_ev = float(1.0 - np.sum((X - np.asarray(model.fitted)) ** 2)
                   / max(np.sum((X - X.mean(0)) ** 2), 1e-30))
    atom = select_weekday_atom(model, day_indices)

    log("fitting flat-SAE control (gamfit.sparse_dictionary_fit)")
    flat_fit = gamfit.sparse_dictionary_fit(
        X.astype(np.float32), min(args.flat_k, X.shape[0] - 1), active=1, max_epochs=40)
    flat_dir, flat_lat = select_flat_direction(flat_fit, X.astype(np.float32), day_indices)
    if lift is not None:
        flat_dir = np.asarray(flat_dir, dtype=np.float64) @ lift
        norm = np.linalg.norm(flat_dir)
        if norm > 0:
            flat_dir = flat_dir / norm

    # Steering coordinate for a base example = its fitted row coordinate. Base
    # rows are the last len(base_examples) rows of X (all_examples order).
    base_row0 = len(fit_examples)
    coord_atom = np.asarray(model.coords[atom], dtype=float)
    base_rows = [base_row0 + i for i in range(len(base_examples))]
    base_coords = [coord_atom[row] for row in base_rows]
    base_amplitudes = [float(model.assignments[row, atom]) for row in base_rows]

    ks = list(range(1, args.max_k + 1))
    log(f"steering {len(base_examples)} base contexts × k∈{ks} (manifold + flat)")
    records = steer_records(model, tok, layer, atom, base_examples, base_rows, base_coords,
                            base_amplitudes,
                            candidate_ids, flat_dir, ks)
    summary = summarize(records, ks)

    meta = {
        "model": args.model, "layer_index": args.layer_index, "k_atoms": args.k_atoms,
        "harmonics": args.harmonics, "flat_k": int(flat_fit.decoder.shape[0]),
        "flat_latent": int(flat_lat), "weekday_atom": int(atom), "fit_ev": fit_ev,
        "n_fit_rows": int(len(fit_examples)), "n_base_rows": int(len(base_examples)),
        "seed": args.seed,
    }
    write_outputs(Path(args.out_dir), meta, records, summary)

    # E2 (gam#2234): the collateral-damage curve, read off the records just
    # written. A failure here must not sink the E1 run, so it is best-effort.
    try:
        import analyze_collateral

        analyze_collateral.run(Path(args.out_dir))
    except Exception as exc:  # noqa: BLE001 — E2 is a downstream read-off
        log(f"E2 collateral analysis skipped: {exc}")

    for arm in ("manifold", "flat"):
        s = summary.get(arm)
        if s:
            print(f"E1_{arm.upper()} accuracy={s['cyclic_advance_accuracy']:.4f} "
                  f"mean_target_prob={s['mean_target_prob']:.4f} mean_kl={s['mean_kl']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
