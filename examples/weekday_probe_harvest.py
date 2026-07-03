"""Weekday-circle probe harvest + frozen-model JVPs for D2 chart transport.

Two stages, one frozen HF model (torch is quarantined here; all estimation
math lives in the Rust core consumed by ``chart_transport_l11_l23.py``):

harvest   Run the weekday probe battery (DOSE's recipe: 10 templates x 7
          weekdays, last-token residual) and save the residual-stream
          activations at the requested layers to an .npz:
          ``acts_L{l}`` (70, d) per layer, ``labels`` (weekday index),
          ``template_ids``, plus the prompt list.

jvp       Given the fitted circle-plane frames (from the transport driver's
          ``--dump-planes``), push each source layer's frame through the
          frozen model's layer stack with forward-mode AD:
          ``jvp[i] = J_{l_from -> l_to}(x_i) @ P_from``   (d, 2) per token,
          where the tangent enters at the LAST token position (where the
          circle is read) and the primal is the full prefix hidden state.
          Saves one .npz per hop with key ``jvp`` (N, d, 2).

Both stages run fine on CPU for small probe batteries (70 prompts) — no GPU
reservation needed; on a shared GPU job pass --device cuda.

Examples
--------
  python weekday_probe_harvest.py harvest --model $ROOT/models/qwen3-8b \
      --layers 11 17 23 --out weekday_acts.npz
  python weekday_probe_harvest.py jvp --model $ROOT/models/qwen3-8b \
      --planes planes.npz --hops 11:17 17:23 --out-dir jvps/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATES = [
    "Today is {w}",
    "The day after tomorrow is {w}",
    "My favorite day of the week is {w}",
    "The meeting is scheduled for {w}",
    "Yesterday was {w}",
    "We will travel on {w}",
    "The store is closed on {w}",
    "Her birthday falls on {w}",
    "The exam takes place on {w}",
    "It always rains on {w}",
]


def build_prompts() -> tuple[list[str], np.ndarray, np.ndarray]:
    prompts, labels, template_ids = [], [], []
    for ti, tmpl in enumerate(TEMPLATES):
        for wi, w in enumerate(WEEKDAYS):
            prompts.append(tmpl.format(w=w))
            labels.append(wi)
            template_ids.append(ti)
    return prompts, np.asarray(labels), np.asarray(template_ids)


def load_model(model_path: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, dtype))
        .to(device)
        .eval()
    )
    return tok, model


def stage_harvest(args: argparse.Namespace) -> None:
    import torch

    prompts, labels, template_ids = build_prompts()
    tok, model = load_model(args.model, args.device, args.dtype)
    per_layer: dict[int, list[np.ndarray]] = {l: [] for l in args.layers}
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(args.device)
            out = model(ids, output_hidden_states=True)
            for l in args.layers:
                per_layer[l].append(
                    out.hidden_states[l][0, -1, :].float().cpu().numpy()
                )
            if (i + 1) % 10 == 0:
                print(f"[harvest] {i + 1}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    payload = {f"acts_L{l}": np.asarray(v, dtype=np.float64) for l, v in per_layer.items()}
    payload["labels"] = labels
    payload["template_ids"] = template_ids
    np.savez(args.out, **payload)
    (args.out.with_suffix(".prompts.json")).write_text(json.dumps(prompts, indent=1))
    print(f"[harvest] -> {args.out} layers={args.layers} n={len(prompts)}")


def stage_jvp(args: argparse.Namespace) -> None:
    import torch

    prompts, _, _ = build_prompts()
    tok, model = load_model(args.model, args.device, args.dtype)
    with np.load(args.planes) as z:
        planes = {int(k.split("_L")[1]): np.asarray(z[k]) for k in z.files if k.startswith("plane_L")}

    layers_mod = model.model.layers
    rotary = model.model.rotary_emb

    def substack(h_full: "torch.Tensor", pos_emb, l_from: int, l_to: int):
        """The frozen map: full-prefix hidden at l_from -> hidden at l_to."""

        def f(h):
            x = h
            for layer in layers_mod[l_from:l_to]:
                out = layer(x, position_embeddings=pos_emb)
                x = out[0] if isinstance(out, tuple) else out
            return x[0, -1, :]

        return f

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for hop in args.hops:
        l_from, l_to = (int(v) for v in hop.split(":"))
        p_from = torch.tensor(planes[l_from], dtype=torch.float32, device=args.device)
        jvps = []
        t0 = time.time()
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(args.device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            h_from = out.hidden_states[l_from].float()
            seq = ids.shape[1]
            position_ids = torch.arange(seq, device=args.device).unsqueeze(0)
            pos_emb = rotary(h_from, position_ids)
            f = substack(h_from, pos_emb, l_from, l_to)
            # Wiring gate: the manual sub-stack must reproduce the reference
            # forward's hidden state at l_to before its Jacobian is trusted
            # (catches mask/rotary/layer-slice mistakes loudly, not silently).
            with torch.no_grad():
                ref = out.hidden_states[l_to][0, -1, :].float()
                got = f(h_from)
                rel = float(
                    (got - ref).norm() / ref.norm().clamp_min(torch.finfo(ref.dtype).tiny)
                )
            if rel > 1e-3:
                raise SystemExit(
                    f"substack wiring check failed at prompt {i} hop {hop}: "
                    f"relative error {rel:.3e} (mask/rotary mismatch?)"
                )
            cols = []
            for c in range(p_from.shape[1]):
                tangent = torch.zeros_like(h_from)
                tangent[0, -1, :] = p_from[:, c]
                _, jv = torch.func.jvp(f, (h_from,), (tangent,))
                cols.append(jv.detach().float().cpu().numpy())
            jvps.append(np.stack(cols, axis=1))  # (d, 2)
            if (i + 1) % 10 == 0:
                print(f"[jvp {hop}] {i + 1}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
        out_path = args.out_dir / f"jvp_{l_from}_to_{l_to}.npz"
        np.savez(out_path, jvp=np.asarray(jvps, dtype=np.float64))
        print(f"[jvp] -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--model", required=True)
    h.add_argument("--layers", nargs="+", type=int, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--device", default="cpu")
    h.add_argument("--dtype", default="float32")
    h.set_defaults(func=stage_harvest)

    j = sub.add_parser("jvp")
    j.add_argument("--model", required=True)
    j.add_argument("--planes", type=Path, required=True,
                   help=".npz with plane_L{l} (d, 2) frames from the transport driver")
    j.add_argument("--hops", nargs="+", required=True, help="e.g. 11:17 17:23")
    j.add_argument("--out-dir", type=Path, required=True)
    j.add_argument("--device", default="cpu")
    j.add_argument("--dtype", default="float32")
    j.set_defaults(func=stage_jvp)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
