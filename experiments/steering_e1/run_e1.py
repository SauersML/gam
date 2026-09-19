#!/usr/bin/env python3
"""E1 — chart-coordinate causal steering on a real LM (gam#2234).

The gam#2234 thesis: a manifold SAE steers by moving the CODE, not the ambient
vector — ``x' = x + a·(Φ(t ⊕ δ) − Φ(t))·B_k`` — so the dose is a displacement in
the chart's own coordinate and the intervention stays on the feature's manifold.
E1 tests that causally on a real model: edit ONE token's residual stream, read
the full-softmax next-token distribution, and compare against a matched-L2-norm
flat-SAE control.

TWO STRUCTURES, ONE CODE PATH
-----------------------------
The owner's standing direction on this issue is *"do not overfit to cyclic"*, so
this harness runs the identical protocol on

  * ``--structure weekday`` — the day-of-week CIRCLE (``atom_topology="circle"``),
    seven labels, the classic periodic chart; and
  * ``--structure ordinal`` — the number words ` one` … ` twelve` on an ORDINAL
    LINE (``atom_topology="euclidean"``), twelve labels, **no periodicity
    anywhere**: base ranks and shifts are chosen so the target never wraps past
    ` twelve`, so a wrap can never smuggle cyclic structure back in.

Both structures share every downstream definition (effect, collateral, control,
chart construction, dose grid), which is the only way the two numbers are
comparable. A previous revision kept a separate ``run_e1_ordinal.py``; it was
deleted in favour of this single path.

MEASUREMENT CONTRACT (corrected 74d13b82d, pinned by
tests/test_steering_e1_measurement_contract.py)
  * every prompt asks for the label AFTER its source label, so moving the source
    by ``k`` targets ``source + k + 1``;
  * effect = FULL-VOCABULARY softmax probability mass moved onto that target
    token (never a seven-way renormalization);
  * collateral = target-conditioned-out ``KL(patched ‖ base)``;
  * the dose grid contains 0, at least one fractional dose, and 1.

WHAT THE 2026-07-31 NULL TAUGHT THIS HARNESS (three fixes, all measured)
  1. **Capture site.** The original harness captured and patched the FINAL token
     ("Today is Monday. Tomorrow is" → ` is`). At that site the day-of-week phase
     carried **0.91%** of the cloud's variance on Qwen3.5-4B-Base L16 (15.4%
     after per-template centering) — it is a downstream trace, not the
     representation. ``--capture-at label`` (the default) reads and patches the
     LABEL token's own position while still reading logits at the last position.
  2. **Cloud size and rank.** 10 templates × 7 labels = 70 rows in a PCA-64 chart
     is the noise floor; ``pca_explained_variance = 0.9997`` at r=64 with n=70 is
     vacuous and was being read as evidence the chart was good. The prompt bank
     is now ~40 distinct CONTEXT HEADS per structure (the activation at the label
     position depends on the head only — the model is causal — so head diversity
     is the only diversity that reaches the cloud), and ``--pca-dim`` defaults to
     a small chart.
  3. **Dose units are measured, not assumed.** The chart's coordinate period is
     a convention of the fitted object, not of this script. The dose for a
     source→target move is now read off the fitted chart's OWN empirical
     label→coordinate map, so it cannot be wrong by a factor of 2π.

**Any E1 result must be reported next to its structure-recovery R²** — the
fitted coordinate's agreement with the known generator. ``fit_ev`` and
``pca_explained_variance`` were both reported in the null run and both were
consistent with a chart that recovered literally nothing.

Launch (MSI, wheel-installed gamfit):
    python3 experiments/steering_e1/run_e1.py --structure weekday \
        --model Qwen/Qwen3.5-4B-Base --layer-index 16 --pca-dim 8 \
        --out-dir out_weekday
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
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ORDINALS = (
    "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Prompt banks.
#
# A HEAD is everything before the label. Because the model is causal, the
# residual at the LABEL token's position depends on the head and nothing else —
# so distinct heads are the ONLY source of cloud diversity, and two templates
# that share a head contribute the same activation twice. The tail is fixed per
# split so every row asks the same question of the readout.
# --------------------------------------------------------------------------- #
WEEKDAY_FIT_HEADS = (
    "Today is", "If today is", "The weekday after", "On the calendar, the day after",
    "Yesterday was", "The day that comes right after", "Immediately after",
    "Counting forward from", "A day later than", "Following the calendar entry for",
    "The meeting was moved to", "She always goes swimming on", "The shop is closed every",
    "His flight departs on", "We agreed to meet next", "The report is due by",
    "Rehearsal happens each", "The bins are collected on", "Payday falls on",
    "The lecture is scheduled for", "I usually rest on", "The market opens only on",
    "Their anniversary is on", "The deadline was last", "Practice starts this",
    "The office reopens on", "He was born on a", "The storm arrived on",
    "Classes resume on", "The train runs only on", "Dinner is served every",
    "The results are announced each", "My appointment got pushed to",
    "The gym is busiest on", "We launch the product on", "The bakery bakes fresh bread every",
    "The council meets on the first", "The exam takes place on", "Volunteers arrive each",
    "The newsletter goes out every",
)
WEEKDAY_BASE_HEADS = (
    "The delivery is scheduled for", "Her shift begins on", "The festival opens on",
    "The library closes early on", "They always argue on", "The audit is booked for",
    "The choir practises on", "The ferry sails on", "The results were posted on",
    "The renovation starts on",
)
WEEKDAY_FIT_TAIL = ". The next day is"
WEEKDAY_BASE_TAIL = ", so the following day is"

ORDINAL_FIT_HEADS = (
    "Counting up, the number after", "In order, right after", "The next whole number after",
    "Adding one to", "On the number line, immediately right of", "If you have",
    "In the ascending sequence, the term after", "The successor of",
    "Step forward once from", "One greater than", "She counted to", "The recipe calls for",
    "He scored", "There were exactly", "The box holds", "They waited",
    "The team has", "I bought", "The chapter covers", "We need at least",
    "The shelf fits", "The building has", "She read", "The garden grows",
    "He owns", "The list contains", "The clock struck", "They ordered",
    "The class has", "The set includes", "The pack came with", "The queue held",
    "She stacked", "The label says", "The tally reached", "The order was for",
    "The batch produced", "The survey counted", "The album has", "The bundle contains",
)
ORDINAL_BASE_HEADS = (
    "Starting at", "The ledger records", "The crate held", "He collected",
    "The playlist has", "The form asks for", "The bin contained", "She packed",
    "The report lists", "The tray carries",
)
ORDINAL_FIT_TAIL = ". Adding one gives"
ORDINAL_BASE_TAIL = ", and one more makes"


@dataclass(frozen=True)
class Structure:
    """Everything that differs between the cyclic and the non-cyclic arm."""

    name: str
    labels: tuple[str, ...]
    topology: str                      # gam atom_topology
    cyclic: bool
    fit_heads: tuple[str, ...]
    base_heads: tuple[str, ...]
    fit_tail: str
    base_tail: str
    max_shift: int                     # largest source-label shift the grid may ask for

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    def fit_templates(self) -> tuple[str, ...]:
        return tuple(f"{h} {{label}}{self.fit_tail}" for h in self.fit_heads)

    def base_templates(self) -> tuple[str, ...]:
        return tuple(f"{h} {{label}}{self.base_tail}" for h in self.base_heads)

    def base_label_indices(self, shifts: list[int]) -> list[int]:
        """Source labels the interventions act on.

        Cyclic: every label is a legal source. Ordinal: only sources for which
        ``source + max(shift) + 1`` still lands inside the ladder, so the
        non-cyclic arm never wraps and therefore never smuggles periodicity in.
        """
        if self.cyclic:
            return list(range(self.n_labels))
        top = self.n_labels - 1 - (max(shifts) + 1)
        if top < 0:
            raise ValueError(f"{self.name}: shift grid {shifts} cannot fit in {self.n_labels} labels")
        return list(range(top + 1))

    def continuation_index(self, source_index: int, shift: int) -> int:
        """The label the prompt asks for after the source is moved by ``shift``."""
        moved = source_index + shift + 1
        if self.cyclic:
            return moved % self.n_labels
        if not 0 <= moved < self.n_labels:
            raise ValueError(
                f"{self.name}: source {source_index} + shift {shift} + 1 = {moved} leaves the "
                "ladder; the non-cyclic grid must never wrap"
            )
        return moved

    def moved_source_index(self, source_index: int, shift: int) -> int:
        """The label the SOURCE token is being made to look like."""
        moved = source_index + shift
        return moved % self.n_labels if self.cyclic else moved


STRUCTURES = {
    "weekday": Structure(
        name="weekday", labels=WEEKDAYS, topology="circle", cyclic=True,
        fit_heads=WEEKDAY_FIT_HEADS, base_heads=WEEKDAY_BASE_HEADS,
        fit_tail=WEEKDAY_FIT_TAIL, base_tail=WEEKDAY_BASE_TAIL, max_shift=6,
    ),
    "ordinal": Structure(
        name="ordinal", labels=ORDINALS, topology="euclidean", cyclic=False,
        fit_heads=ORDINAL_FIT_HEADS, base_heads=ORDINAL_BASE_HEADS,
        fit_tail=ORDINAL_FIT_TAIL, base_tail=ORDINAL_BASE_TAIL, max_shift=4,
    ),
}


# --------------------------------------------------------------------------- #
# Model, capture, patch. Hook path from
# experiments/interchange/qwen_calendar_interchange.py.
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(model_name: str, cache_dir: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir or None)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    # One device, placed after loading: a `device_map` needs `accelerate`, which
    # the harness does not depend on.
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tok


def model_input_device(model: Any) -> Any:
    return next(model.parameters()).device


def resolve_layers(model: Any) -> Any:
    for root_name, layer_name in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layer_name):
            return getattr(root, layer_name)
        # A causal-LM wrapper nests the decoder at `model.model`; a conditional-
        # generation wrapper such as Qwen3.5's nests it at `model.language_model`.
        for inner_name in ("model", "language_model"):
            inner = getattr(root, inner_name, None) if root is not None else None
            if inner is not None and hasattr(inner, layer_name):
                return getattr(inner, layer_name)
    raise ValueError("could not locate transformer block list on model")


def label_token_ids(tokenizer: Any, labels: tuple[str, ...], prefix: str) -> list[int]:
    ids: list[int] = []
    for label in labels:
        enc = tokenizer.encode(prefix + label, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"candidate {prefix + label!r} tokenized to {len(enc)} tokens; "
                "choose a different --candidate-prefix"
            )
        ids.append(int(enc[0]))
    return ids


def candidate_token_ids(tokenizer: Any, prefix: str) -> list[int]:
    """Weekday candidate ids (kept for the pinned measurement contract)."""
    return label_token_ids(tokenizer, WEEKDAYS, prefix)


def build_label_prompt(tokenizer: Any, template: str, label_token_id: int) -> tuple[list[int], int]:
    """Token ids for ``template.format(label=...)`` plus the label token's position.

    Built by concatenating token id lists rather than by re-tokenizing the joined
    string, so the label occupies exactly one known position and the edit site is
    unambiguous. ``{label}`` is preceded by a space in every template and the
    label token carries that space, so the head is right-stripped before encoding.
    """
    if "{label}" not in template:
        raise ValueError(f"template {template!r} has no {{label}} slot")
    head, tail = template.split("{label}", 1)
    head_ids = tokenizer.encode(head.rstrip(" "), add_special_tokens=False) if head.strip() else []
    tail_ids = tokenizer.encode(tail, add_special_tokens=False) if tail else []
    ids = list(head_ids) + [int(label_token_id)] + list(tail_ids)
    return ids, len(head_ids)


def run_clean_at(model: Any, layer: Any, token_ids: list[int], edit_position: int):
    """Capture the residual at ``edit_position``; read logits at the LAST position."""
    import torch

    device = model_input_device(model)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if not (0 <= edit_position < ids.shape[1]):
        raise ValueError(f"edit position {edit_position} outside prompt of {ids.shape[1]} tokens")
    captured: dict[str, Any] = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[0, edit_position, :].detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    if "activation" not in captured:
        raise ValueError("activation hook did not fire")
    return captured["activation"], out.logits[0, -1, :].detach().float().cpu()


def run_patched_at(model: Any, layer: Any, token_ids: list[int], edit_position: int,
                   patched_activation: Any):
    """Patch the residual at ``edit_position``; read logits at the LAST position."""
    import torch

    device = model_input_device(model)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[0, edit_position, :] = patched_activation.to(
            device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    return out.logits[0, -1, :].detach().float().cpu()


def run_clean(model: Any, tokenizer: Any, layer: Any, prompt: str):
    """Historical final-token capture site (``--capture-at last``)."""
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
    return captured["activation"], out.logits[0, position, :].detach().float().cpu()


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


# --------------------------------------------------------------------------- #
# Readout metrics. Full vocabulary throughout.
# --------------------------------------------------------------------------- #
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


def label_probabilities(logits: Any, ids: list[int]) -> np.ndarray:
    """Full-softmax probabilities for the label tokens (never renormalized)."""
    logp = _log_softmax(logits)
    idx = np.asarray(ids, dtype=np.int64)
    if idx.ndim != 1 or idx.size == 0 or np.any(idx < 0) or np.any(idx >= logp.size):
        raise ValueError("candidate label token ids must be in-vocabulary ids, one per label")
    return np.exp(logp[idx])


def weekday_token_probabilities(logits: Any, candidate_ids: list[int]) -> np.ndarray:
    """Full-softmax probabilities for the seven weekday tokens."""
    if len(candidate_ids) != len(WEEKDAYS):
        raise ValueError("candidate weekday token ids must be seven in-vocabulary ids")
    return label_probabilities(logits, candidate_ids)


def target_excluded_kl_model_to_base(patched_logits: Any, base_logits: Any,
                                     target_token_id: int) -> float:
    """Collateral KL on non-target tokens: ``KL(patched ‖ base)``, target excluded.

    Each arm is conditioned on "the next token is not the intended target" before
    the KL, so probability mass deliberately moved onto the target is not charged
    again as collateral damage.
    """
    patched = _logits_array(patched_logits)
    base = _logits_array(base_logits)
    if patched.shape != base.shape:
        raise ValueError(f"patched/base vocabulary shapes differ: {patched.shape} vs {base.shape}")
    if not (0 <= target_token_id < patched.size):
        raise ValueError(
            f"target token id {target_token_id} out of range for vocabulary size {patched.size}")
    keep = np.ones(patched.size, dtype=bool)
    keep[target_token_id] = False
    model_log = _log_softmax(patched[keep])
    base_log = _log_softmax(base[keep])
    return max(float(np.sum(np.exp(model_log) * (model_log - base_log))), 0.0)


def continuation_target_index(base_day_index: int, target_shift_days: int) -> int:
    """Next-token weekday after replacing the source day by source+k days."""
    if not (0 <= base_day_index < len(WEEKDAYS)):
        raise ValueError(f"base day index must be in [0, 7); got {base_day_index}")
    if not (1 <= target_shift_days < len(WEEKDAYS)):
        raise ValueError(f"target shift must be an integer in [1, 6]; got {target_shift_days}")
    return (base_day_index + target_shift_days + 1) % len(WEEKDAYS)


def parse_target_shifts(spec: str, max_shift: int = 6) -> list[int]:
    try:
        shifts = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--target-shifts must be comma-separated integers") from error
    if not shifts or len(set(shifts)) != len(shifts) or any(not 1 <= k <= max_shift for k in shifts):
        raise ValueError(
            f"--target-shifts must contain unique integers from 1 through {max_shift}")
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
            "--dose-fractions must be strictly increasing finite values in [0,1], "
            "including 0 and 1")
    if not any(0.0 < value < 1.0 for value in fractions):
        raise ValueError("--dose-fractions must include at least one fractional interior dose")
    return fractions


# --------------------------------------------------------------------------- #
# Cloud collection.
# --------------------------------------------------------------------------- #
@dataclass(eq=False)  # holds torch tensors; identity equality avoids tensor-== on compare
class CleanExample:
    template_index: int
    label_index: int
    prompt: str
    activation: Any
    logits: Any
    token_ids: Any = None
    edit_position: int = -1


def collect_cloud(model, tokenizer, layer, templates, labels, capture_at="label",
                  candidate_ids=None) -> list[CleanExample]:
    if capture_at == "label" and candidate_ids is None:
        raise ValueError("capture_at='label' needs the label token ids")
    examples: list[CleanExample] = []
    for ti, template in enumerate(templates):
        for li, label in enumerate(labels):
            prompt = template.format(label=label)
            if capture_at == "label":
                ids, pos = build_label_prompt(tokenizer, template, candidate_ids[li])
                act, logits = run_clean_at(model, layer, ids, pos)
                examples.append(CleanExample(ti, li, prompt, act, logits, ids, pos))
            else:
                act, logits = run_clean(model, tokenizer, layer, prompt)
                examples.append(CleanExample(ti, li, prompt, act, logits))
    return examples


# --------------------------------------------------------------------------- #
# Structure recovery: how much of the KNOWN generator the fitted 1-D chart
# coordinate reproduces. This is the number that decides whether E1 can measure
# anything at all, and it must be reported beside every E1 result.
# --------------------------------------------------------------------------- #
def circular_recovery(coord: np.ndarray, label_index: np.ndarray, n_labels: int,
                      period: float) -> tuple[float, int]:
    """Squared circular correlation of ``coord`` (read with ``period``) with truth."""
    truth = np.exp(1j * TAU * label_index.astype(np.float64) / n_labels)
    chart = np.exp(1j * TAU * coord.astype(np.float64) / period)
    forward = abs(np.mean(truth * np.conj(chart)))
    reverse = abs(np.mean(truth * chart))
    if forward >= reverse:
        return float(forward ** 2), 1
    return float(reverse ** 2), -1


def linear_recovery(coord: np.ndarray, label_index: np.ndarray) -> tuple[float, float]:
    """``R^2`` and slope of ``coord ~ label_index`` (the ordinal / non-cyclic case)."""
    x = label_index.astype(np.float64)
    d = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(d, coord, rcond=None)
    resid = coord - d @ coef
    tss = float(np.sum((coord - coord.mean()) ** 2))
    return 1.0 - float(np.sum(resid ** 2)) / max(tss, 1e-30), float(coef[1])


def select_atom(structure: Structure, model: Any, label_index: np.ndarray):
    """Pick the atom whose fitted coordinate best recovers the known generator.

    Returns ``(atom, recovery_r2, period, orientation)``. For a circle the chart
    PERIOD is a property of the fitted object and is MEASURED here (both the
    period-one and the period-2π conventions are scored) rather than assumed: a
    dose computed in the wrong unit is silently 2π short and produces a null that
    is indistinguishable from "the fit found nothing".
    """
    best = (0, -np.inf, 1.0, 1)
    for k in range(len(model.coords)):
        c = np.asarray(model.coords[k], dtype=float)
        coord = c[:, 0] if c.ndim == 2 else c
        if structure.cyclic:
            for period in (1.0, TAU):
                score, orientation = circular_recovery(
                    coord, label_index, structure.n_labels, period)
                if score > best[1]:
                    best = (k, score, period, orientation)
        else:
            score, slope = linear_recovery(coord, label_index)
            if score > best[1]:
                best = (k, score, float("inf"), 1 if slope >= 0 else -1)
    atom, score, period, orientation = best
    c = np.asarray(model.coords[atom], dtype=float)
    coord = c[:, 0] if c.ndim == 2 else c
    n_distinct = int(np.unique(np.round(coord, 6)).size)
    log(f"atom {atom}: structure recovery R2={score:.4f} "
        f"(period={period}, orientation={orientation:+d}, coord std={coord.std():.4g}, "
        f"range=[{coord.min():.4g},{coord.max():.4g}], {n_distinct} distinct of {coord.size})")
    return atom, float(score), float(period), int(orientation)


def label_coordinate_map(coord: np.ndarray, label_index: np.ndarray, n_labels: int,
                         period: float) -> np.ndarray:
    """The fitted chart's OWN empirical label→coordinate map.

    The dose is read off this map rather than assumed to be ``period·k/n``, so a
    non-uniform chart still gets the displacement that actually takes the source
    label's fitted coordinate to the target label's.
    """
    out = np.zeros(n_labels, dtype=np.float64)
    for li in range(n_labels):
        rows = coord[label_index == li]
        if rows.size == 0:
            raise ValueError(f"label {li} has no fitted rows")
        if np.isfinite(period):
            ang = np.mean(np.exp(1j * TAU * rows / period))
            out[li] = float(np.angle(ang)) * period / TAU
        else:
            out[li] = float(rows.mean())
    return out


def chart_displacement(label_map: np.ndarray, source: int, target: int, period: float) -> float:
    """Signed chart displacement taking the source label to the target label."""
    delta = float(label_map[target] - label_map[source])
    if np.isfinite(period):
        half = 0.5 * period
        delta = (delta + half) % period - half
    return delta


def select_flat_directions(flat_fit, X: np.ndarray, label_index: np.ndarray,
                           n_labels: int) -> tuple[np.ndarray, list[int]]:
    """Per-label flat-SAE steering directions — the control, made as strong as possible.

    A flat SAE steers by adding the decoder column of the latent that FIRES on the
    feature you want. So for each target label we take the latent whose code is
    most selective for that label on the train split (mean code on the label's
    rows minus the mean elsewhere) and use its unit decoder direction. This is a
    strictly stronger control than a single global "weekday direction": it is
    allowed a different direction for every target, and it is exactly what a flat
    SAE practitioner would do.
    """
    tr = flat_fit.transform(X)
    k = int(flat_fit.decoder.shape[0])
    codes = np.zeros((X.shape[0], k), dtype=np.float64)
    rows = np.arange(X.shape[0])[:, None]
    codes[rows, tr.indices.astype(np.int64)] = tr.codes.astype(np.float64)
    dirs = np.zeros((n_labels, int(flat_fit.decoder.shape[1])), dtype=np.float64)
    chosen: list[int] = []
    for li in range(n_labels):
        mask = label_index == li
        score = codes[mask].mean(0) - codes[~mask].mean(0)
        lat = int(np.argmax(score))
        w = np.asarray(flat_fit.decoder[lat], dtype=np.float64)
        norm = float(np.linalg.norm(w))
        dirs[li] = w / norm if norm > 0 else w
        chosen.append(lat)
    log(f"flat control latents per label: {chosen}")
    return dirs, chosen


# --------------------------------------------------------------------------- #
# The intervention sweep.
# --------------------------------------------------------------------------- #
def steer_records(structure, lm_model, sae_model, tokenizer, layer, atom,
                  base_examples, metric_rows, base_coords, base_amplitudes,
                  candidate_ids, flat_dirs, label_map, period,
                  target_shifts, dose_fractions, lift=None) -> list[dict[str, Any]]:
    import torch

    records: list[dict[str, Any]] = []
    labels = structure.labels
    for base, metric_row, t0_in, amplitude in zip(
        base_examples, metric_rows, base_coords, base_amplitudes
    ):
        source = base.label_index
        base_probs = label_probabilities(base.logits, candidate_ids)
        t0 = np.asarray(t0_in, dtype=np.float64).reshape(-1)

        def patch(vector):
            if base.edit_position >= 0:
                return run_patched_at(
                    lm_model, layer, base.token_ids, base.edit_position, vector)
            return run_patched(lm_model, tokenizer, layer, base.prompt, vector)

        for shift in target_shifts:
            try:
                target_index = structure.continuation_index(source, shift)
            except ValueError:
                continue
            moved_source = structure.moved_source_index(source, shift)
            target_token_id = int(candidate_ids[target_index])
            base_target_probability = float(base_probs[target_index])
            full_displacement = chart_displacement(label_map, source, moved_source, period)
            for dose_fraction in dose_fractions:
                dcoord = full_displacement * float(dose_fraction)
                t_to = t0.copy()
                t_to[0] = t0[0] + dcoord
                plan = sae_model.steer(int(atom), int(metric_row), float(amplitude), t0, t_to)
                delta = np.asarray(plan["delta"], dtype=np.float64).reshape(-1)
                if lift is not None:
                    delta = delta @ lift

                patched = base.activation + torch.from_numpy(delta.astype(np.float32))
                manifold_logits = patch(patched)

                # Matched-L2-norm flat control, aimed at the SAME target label.
                flat_delta = float(np.linalg.norm(delta)) * flat_dirs[moved_source]
                if lift is not None:
                    flat_delta = flat_delta @ lift
                patched_flat = base.activation + torch.from_numpy(flat_delta.astype(np.float32))
                flat_logits = patch(patched_flat)

                for arm, patched_logits in (("manifold", manifold_logits),
                                            ("flat", flat_logits)):
                    probs = label_probabilities(patched_logits, candidate_ids)
                    top = int(np.argmax(probs))
                    target_probability = float(probs[target_index])
                    collateral = target_excluded_kl_model_to_base(
                        patched_logits, base.logits, target_token_id)
                    records.append({
                        "structure": structure.name,
                        "arm": arm,
                        "base_template": base.template_index,
                        "base_label": labels[source],
                        "base_label_index": source,
                        "target_shift": int(shift),
                        "moved_source_label": labels[moved_source],
                        "target_label": labels[target_index],
                        "target_label_index": target_index,
                        "target_token_id": target_token_id,
                        "dose_fraction": float(dose_fraction),
                        "chart_displacement": float(dcoord),
                        "delta_norm": float(np.linalg.norm(delta)),
                        "steer_off_manifold_norm": (
                            float(plan["off_manifold_norm"])
                            if arm == "manifold" and plan.get("off_manifold_norm") is not None
                            else None),
                        "steer_predicted_nats": (
                            float(plan["predicted_nats"])
                            if arm == "manifold" and plan.get("predicted_nats") is not None
                            else None),
                        "realized_top_label": labels[top],
                        "realized_top_label_index": top,
                        "target_token_probability": target_probability,
                        "base_target_token_probability": base_target_probability,
                        "target_probability_mass_moved": (
                            target_probability - base_target_probability),
                        "collateral_kl_model_to_base_non_target": collateral,
                        "label_token_probabilities": [float(x) for x in probs],
                    })
    return records


def summarize(records, target_shifts, dose_fractions) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("manifold", "flat"):
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        endpoint = [r for r in rs if r["dose_fraction"] == 1.0]
        acc = float(np.mean([
            r["realized_top_label_index"] == r["target_label_index"] for r in endpoint]))
        dose_response: dict[str, Any] = {}
        for shift in target_shifts:
            by_fraction = {}
            for fraction in dose_fractions:
                sample = [r for r in rs
                          if r["target_shift"] == shift and r["dose_fraction"] == fraction]
                if not sample:
                    continue
                by_fraction[str(fraction)] = {
                    "mean_target_token_probability": float(np.mean(
                        [r["target_token_probability"] for r in sample])),
                    "mean_target_probability_mass_moved": float(np.mean(
                        [r["target_probability_mass_moved"] for r in sample])),
                    "mean_collateral_kl_model_to_base_non_target": float(np.mean(
                        [r["collateral_kl_model_to_base_non_target"] for r in sample])),
                    "n": int(len(sample)),
                }
            dose_response[str(shift)] = by_fraction
        out[arm] = {
            "endpoint_target_accuracy": acc,
            "mean_endpoint_target_token_probability": float(np.mean(
                [r["target_token_probability"] for r in endpoint])),
            "mean_endpoint_target_probability_mass_moved": float(np.mean(
                [r["target_probability_mass_moved"] for r in endpoint])),
            "mean_endpoint_collateral_kl_model_to_base_non_target": float(np.mean(
                [r["collateral_kl_model_to_base_non_target"] for r in endpoint])),
            "dose_response": dose_response,
        }
    return out


def write_outputs(out_dir: Path, meta, records, summary) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e1_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "e1_summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n")

    lines = [
        f"# E1 — chart-coordinate steering, structure `{meta['structure']}` (gam#2234)",
        "",
        f"- Model `{meta['model']}` block `{meta['layer_index']}`, topology "
        f"`{meta['topology']}`, chart PCA dim `{meta['pca_dim']}`, "
        f"{meta['n_fit_rows']} fit rows / {meta['n_base_rows']} held-out rows.",
        f"- **Structure recovery R² of the fitted coordinate: "
        f"`{meta['structure_recovery_r2']:.4f}`** (fit EV `{meta['fit_ev']:.4f}`, "
        f"chart PCA explained variance `{meta['pca_explained_variance']:.4f}`).",
        "",
        "| arm | endpoint target accuracy | endpoint target-token probability | "
        "probability mass moved | target-excluded KL(model‖base) |",
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
    lines += ["", "## Fractional dose-response by target shift", "",
              "| shift k | dose fraction | manifold mass moved | manifold collateral | "
              "flat mass moved | flat collateral |", "|---:|---:|---:|---:|---:|---:|"]
    for shift, fractions in summary.get("manifold", {}).get("dose_response", {}).items():
        for fraction, manifold_row in fractions.items():
            flat_row = summary.get("flat", {}).get("dose_response", {}).get(
                shift, {}).get(fraction, {})
            lines.append(
                f"| {shift} | {float(fraction):.3f} | "
                f"{manifold_row['mean_target_probability_mass_moved']:.6f} | "
                f"{manifold_row['mean_collateral_kl_model_to_base_non_target']:.6f} | "
                f"{flat_row.get('mean_target_probability_mass_moved', float('nan')):.6f} | "
                f"{flat_row.get('mean_collateral_kl_model_to_base_non_target', float('nan')):.6f} |")
    lines.append("")
    (out_dir / "e1_results.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out_dir / 'e1_results.md'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", choices=tuple(STRUCTURES), default="weekday",
                    help="'weekday' = the day-of-week circle; 'ordinal' = the NON-CYCLIC "
                         "number-word line (the 'do not overfit to cyclic' arm)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--layer-index", type=int, default=16)
    ap.add_argument("--k-atoms", type=int, default=1)
    ap.add_argument("--flat-k", type=int, default=32)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--target-shifts", default="")
    ap.add_argument("--dose-fractions", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--candidate-prefix", default=" ")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--pca-dim", type=int, default=8,
                    help="fit-chart PCA dimension (0 disables; deltas are lifted back exactly)")
    ap.add_argument("--capture-at", choices=("last", "label"), default="label",
                    help="edit site: 'label' patches the label token's own position (logits are "
                         "still read at the final token); 'last' is the historical final-token "
                         "site, whose cloud carried 0.91%% of its variance in the label phase")
    ap.add_argument("--per-template-center", action="store_true",
                    help="remove each context head's mean before charting (chart-construction "
                         "nuisance removal only; the intervention still edits the RAW activation)")
    ap.add_argument("--max-heads", type=int, default=0,
                    help="use only the first N context heads of the fit bank (0 = all). The "
                         "cloud has n_heads x n_labels rows and the dense certification lane's "
                         "cost grows with n, so this is the knob for trading cloud size against "
                         "fit time; the structure-recovery R2 reported alongside says what it cost")
    ap.add_argument("--cloud-npz", default="",
                    help="write the harvested ambient cloud here (for the CPU-side chart sweep)")
    ap.add_argument("--out-dir", default="experiments/steering_e1/out")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    structure = STRUCTURES[args.structure]
    shift_spec = args.target_shifts or ",".join(str(k) for k in range(1, structure.max_shift + 1))
    target_shifts = parse_target_shifts(shift_spec, structure.max_shift)
    dose_fractions = parse_dose_fractions(args.dose_fractions)
    np.random.seed(args.seed)
    import gamfit

    log(f"structure={structure.name} topology={structure.topology} "
        f"labels={structure.n_labels} shifts={target_shifts}")
    log(f"loading {args.model}")
    model_lm, tok = load_model_and_tokenizer(args.model, args.cache_dir, args.dtype)
    layers = resolve_layers(model_lm)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(f"--layer-index must be in [0,{len(layers)}); got {args.layer_index}")
    layer = layers[args.layer_index]
    candidate_ids = label_token_ids(tok, structure.labels, args.candidate_prefix)
    log(f"candidate ids: {dict(zip(structure.labels, candidate_ids))}")

    base_label_indices = structure.base_label_indices(target_shifts)
    log(f"held-out source labels: {[structure.labels[i] for i in base_label_indices]}")

    log("collecting disjoint fit and held-out clouds")
    fit_templates = structure.fit_templates()
    if args.max_heads:
        fit_templates = fit_templates[:args.max_heads]
    fit_examples = collect_cloud(model_lm, tok, layer, fit_templates,
                                 structure.labels, args.capture_at, candidate_ids)
    # Every label is collected on the held-out side so the per-head centering
    # below sees the same label set the fit side did; sources for which the
    # shift grid would leave the ladder are skipped inside the steering loop,
    # not dropped here (dropping them first would make the two splits' head
    # means incomparable).
    base_examples = collect_cloud(model_lm, tok, layer, structure.base_templates(),
                                  structure.labels, args.capture_at, candidate_ids)
    X_fit_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in fit_examples]))
    X_base_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in base_examples]))
    fit_label_index = np.asarray([ex.label_index for ex in fit_examples])
    log(f"fit X shape {X_fit_ambient.shape}; held-out X shape {X_base_ambient.shape}")
    if args.cloud_npz:
        np.savez(args.cloud_npz, X=X_fit_ambient, label_index=fit_label_index,
                 template_index=np.asarray([ex.template_index for ex in fit_examples]))
        log(f"wrote ambient cloud to {args.cloud_npz}")

    X_fit_chart, X_base_chart = X_fit_ambient, X_base_ambient
    if args.per_template_center:
        X_fit_chart = X_fit_ambient.copy()
        for ti in {ex.template_index for ex in fit_examples}:
            rows = [i for i, ex in enumerate(fit_examples) if ex.template_index == ti]
            X_fit_chart[rows] -= X_fit_chart[rows].mean(0, keepdims=True)
        X_base_chart = X_base_ambient.copy()
        for ti in {ex.template_index for ex in base_examples}:
            rows = [i for i, ex in enumerate(base_examples) if ex.template_index == ti]
            X_base_chart[rows] -= X_base_chart[rows].mean(0, keepdims=True)
        log("per-head centering applied to the chart inputs")

    # Wide-p treatment: fit in a train-only PCA chart; steering deltas are lifted
    # back to ambient through the orthonormal rows, so the intervention is exact
    # and norm-preserving.
    pca_evr = 1.0
    if args.pca_dim and args.pca_dim < X_fit_chart.shape[1]:
        mu = X_fit_chart.mean(0, keepdims=True)
        centered = X_fit_chart - mu
        _, svals, vt = np.linalg.svd(centered, full_matrices=False)
        r = int(min(args.pca_dim, vt.shape[0]))
        lift = np.ascontiguousarray(vt[:r])
        X_fit = np.ascontiguousarray(centered @ lift.T)
        X_base = np.ascontiguousarray((X_base_chart - mu) @ lift.T)
        pca_evr = float((svals[:r] ** 2).sum() / max((svals ** 2).sum(), 1e-30))
        log(f"train-only PCA chart: {X_fit.shape} (explained variance {pca_evr:.4f})")
    else:
        lift = None
        X_fit, X_base = X_fit_chart, X_base_chart

    log(f"fitting gamfit.sae.sae_manifold_fit ({structure.topology}, softmax assignment)")
    sae_model = gamfit.sae.sae_manifold_fit(
        X_fit, K=args.k_atoms, d_atom=1, atom_topology=structure.topology,
        assignment="softmax", n_iter=args.n_iter, random_state=args.seed)
    fit_ev = float(1.0 - np.sum((X_fit - np.asarray(sae_model.fitted)) ** 2)
                   / max(np.sum((X_fit - X_fit.mean(0)) ** 2), 1e-30))
    atom, recovery_r2, period, orientation = select_atom(structure, sae_model, fit_label_index)
    fit_coord = np.asarray(sae_model.coords[atom], dtype=float)
    fit_coord = fit_coord[:, 0] if fit_coord.ndim == 2 else fit_coord
    label_map = label_coordinate_map(fit_coord, fit_label_index, structure.n_labels, period)
    log(f"empirical label->coordinate map: "
        f"{ {structure.labels[i]: round(float(label_map[i]), 4) for i in range(structure.n_labels)} }")

    log("fitting flat-SAE control (gamfit.sae.sparse_dictionary_fit)")
    flat_fit = gamfit.sae.sparse_dictionary_fit(
        X_fit.astype(np.float32), min(args.flat_k, X_fit.shape[0] - 1), active=1, max_epochs=40)
    flat_dirs, flat_latents = select_flat_directions(
        flat_fit, X_fit.astype(np.float32), fit_label_index, structure.n_labels)

    base_latents = sae_model.converged_latents(X_base)
    base_coords_array = np.asarray(base_latents["coords"][atom], dtype=float)
    base_assignments = np.asarray(base_latents["assignments"], dtype=float)
    base_turns = base_coords_array[:, 0]
    metric_rows = []
    for turn in base_turns:
        if np.isfinite(period):
            half = 0.5 * period
            distance = np.abs((fit_coord - turn + half) % period - half)
        else:
            distance = np.abs(fit_coord - turn)
        metric_rows.append(int(np.argmin(distance)))
    base_coords = [base_coords_array[row] for row in range(len(base_examples))]
    base_amplitudes = [float(base_assignments[row, atom]) for row in range(len(base_examples))]

    log(f"steering {len(base_examples)} held-out contexts × shifts {target_shifts} × "
        f"doses {dose_fractions} (manifold + flat)")
    records = steer_records(
        structure, model_lm, sae_model, tok, layer, atom, base_examples, metric_rows,
        base_coords, base_amplitudes, candidate_ids, flat_dirs, label_map, period,
        target_shifts, dose_fractions, lift=lift)
    summary = summarize(records, target_shifts, dose_fractions)

    meta = {
        "structure": structure.name, "topology": structure.topology,
        "model": args.model, "layer_index": args.layer_index, "k_atoms": args.k_atoms,
        "flat_k": int(flat_fit.decoder.shape[0]), "flat_latents": flat_latents,
        "atom": int(atom), "fit_ev": fit_ev,
        "structure_recovery_r2": float(recovery_r2),
        "chart_coord_std": float(np.std(fit_coord)),
        "chart_coord_distinct": int(np.unique(np.round(fit_coord, 6)).size),
        "pca_explained_variance": float(pca_evr),
        "chart_period": (None if not np.isfinite(period) else float(period)),
        "chart_orientation": int(orientation),
        "label_coordinate_map": [float(v) for v in label_map],
        "capture_at": args.capture_at,
        "per_template_center": bool(args.per_template_center),
        "pca_dim": int(args.pca_dim),
        "n_fit_heads": int(len(fit_templates)),
        "n_fit_rows": int(len(fit_examples)), "n_base_rows": int(len(base_examples)),
        "target_shifts": target_shifts, "dose_fractions": dose_fractions,
        "seed": args.seed,
    }
    write_outputs(Path(args.out_dir), meta, records, summary)

    import analyze_collateral

    analyze_collateral.run(Path(args.out_dir))

    print(f"E1_STRUCTURE_RECOVERY_R2={recovery_r2:.6f} fit_ev={fit_ev:.6f} "
          f"pca_evr={pca_evr:.6f}", flush=True)
    for arm in ("manifold", "flat"):
        s = summary.get(arm)
        if s:
            print(
                f"E1_{arm.upper()} endpoint_accuracy={s['endpoint_target_accuracy']:.4f} "
                f"endpoint_target_prob={s['mean_endpoint_target_token_probability']:.6f} "
                f"endpoint_mass_moved={s['mean_endpoint_target_probability_mass_moved']:.6f} "
                f"endpoint_collateral="
                f"{s['mean_endpoint_collateral_kl_model_to_base_non_target']:.6f}",
                flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
