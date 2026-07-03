"""Two-block (Rung-2) weekday harvest: per-token activations AND behavioral
summaries for the two-block manifold fit.

The probe is RELATIONAL: every prompt establishes a day context and ends right
BEFORE the answer weekday ("Yesterday was Monday, so today is"), so the model
must carry "which day" in the residual stream at the last token to answer.
That gives, per prompt, at the SAME position:

  * x_i — the residual-stream activation at the requested layer(s), the
    activation block of the two-block fit;
  * p_i — the model's next-token distribution RESTRICTED to the 7 weekday
    tokens (leading-space variants), the behavioral summary q_i = sqrt(p_i)
    the behavior block embeds. This is the token set whose probability
    actually moves with the weekday feature, so the sphere-tangent geometry
    is informative and low-dimensional (V = 7).

Output .npz per run:
  acts_L{l}     (n, d)   float64 residual activations per requested layer
  probs         (n, 7)   float64 restricted next-token probabilities
                         (renormalized over the 7 weekday tokens)
  probs_raw_mass(n,)     total unrestricted probability mass the 7 tokens
                         carried (honesty diagnostic: how much behavior the
                         restriction keeps)
  labels        (n,)     answer-weekday index 0..6 (Monday..Sunday)
  offsets       (n,)     relational offset used by the template
  template_ids  (n,)
plus a .prompts.json with the prompt strings and the 7 column token strings.

Consumed by crates/gam-sae/examples/two_block_weekday_demo.rs after a trivial
npz->csv split (or feed the arrays through the FFI directly once the two-block
driver is exposed there). All estimation math is Rust; this file only touches
the frozen HF model (the allowed external boundary).

Example
-------
  python weekday_behavior_harvest.py --model $ROOT/models/qwen3-8b \
      --layers 11 17 23 --out weekday_two_block.npz --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Relational templates: {c} is the CONTEXT weekday, the prompt ends where the
# ANSWER weekday (context + offset) should be produced. The trailing space is
# deliberately absent; the answer token is the leading-space weekday.
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
    for ti, (tmpl, off) in enumerate(OFFSET_TEMPLATES):
        for ci, c in enumerate(WEEKDAYS):
            prompts.append(tmpl.format(c=c))
            labels.append((ci + off) % 7)
            offsets.append(off)
            template_ids.append(ti)
    return (
        prompts,
        np.asarray(labels),
        np.asarray(offsets),
        np.asarray(template_ids),
    )


def weekday_token_ids(tok) -> tuple[list[int], list[str]]:
    """Single-token ids for the leading-space weekday surface forms.

    Every weekday must map to exactly one token in the model's vocabulary (true
    for the Qwen/Llama BPE families); anything else is surfaced loudly — a
    multi-token weekday would silently corrupt the restricted distribution.
    """
    ids, strings = [], []
    for w in WEEKDAYS:
        pieces = tok(" " + w, add_special_tokens=False).input_ids
        if len(pieces) != 1:
            raise SystemExit(
                f"weekday {w!r} tokenizes to {len(pieces)} pieces; the restricted "
                "behavior set needs single-token weekdays"
            )
        ids.append(pieces[0])
        strings.append(" " + w)
    return ids, strings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts, labels, offsets, template_ids = build_prompts()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype)
        )
        .to(args.device)
        .eval()
    )
    wd_ids, wd_strings = weekday_token_ids(tok)

    per_layer: dict[int, list[np.ndarray]] = {l: [] for l in args.layers}
    probs_rows, raw_mass = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(args.device)
            out = model(ids, output_hidden_states=True)
            for l in args.layers:
                per_layer[l].append(out.hidden_states[l][0, -1, :].float().cpu().numpy())
            full = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
            restricted = full[wd_ids]
            mass = float(restricted.sum())
            raw_mass.append(mass)
            probs_rows.append((restricted / restricted.sum()).cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"[harvest] {i + 1}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)

    payload = {f"acts_L{l}": np.asarray(v, dtype=np.float64) for l, v in per_layer.items()}
    payload["probs"] = np.asarray(probs_rows, dtype=np.float64)
    payload["probs_raw_mass"] = np.asarray(raw_mass, dtype=np.float64)
    payload["labels"] = labels
    payload["offsets"] = offsets
    payload["template_ids"] = template_ids
    np.savez(args.out, **payload)
    args.out.with_suffix(".prompts.json").write_text(
        json.dumps({"prompts": prompts, "behavior_tokens": wd_strings}, indent=1)
    )
    print(
        f"[harvest] -> {args.out} layers={args.layers} n={len(prompts)} "
        f"median restricted mass={np.median(raw_mass):.3f}"
    )


if __name__ == "__main__":
    main()
