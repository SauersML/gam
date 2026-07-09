#!/usr/bin/env python3
"""Wrap a fitted curved manifold-SAE as a SAEBench custom SAE (#1942, stage 2).

The public SAEBench absorption/SCR/unlearning/sparse-probing runners drive a
``sae_bench.custom_saes.base_sae.BaseSAE`` with a standard ``encode``/``decode``
interface. A curved manifold-SAE has no flat ``W_enc``/``W_dec`` — its encoder is
the frozen-decoder OOS Newton router and its decoder threads codes through
per-atom charts. This adapter overrides ``encode``/``decode`` to call the AUDITED
Rust core on the live fit handle, so the suite scores the ACTUAL manifold model,
not a linearised stand-in. No math is re-implemented here; nothing is fabricated.

  * ``encode(x)`` -> the real curved soft codes ``model.encode(x)`` ((N, K)).
    This is faithful and needs no new Rust (getter landed:
    manifold_and_posterior_ffi.rs:8063). Absorption + sparse-probing are
    feature-activation metrics and run on this alone.

  * ``decode(codes)`` -> a curved reconstruction from (possibly latent-ablated)
    codes. SCR + unlearning ablate a latent then decode. The fit handle exposes
    ``reconstruct(x)`` (activations -> recon, a full re-solve) but NOT a
    codes -> recon map, so faithful decode needs a thin pymethod wrapping the
    existing Rust ``reconstruct_from_assignments`` (latent_basis_and_sae_ffi.rs:4279).
    Until that getter lands, ``decode`` raises a precise NotImplementedError
    rather than approximating — so absorption/sparse-probing are unblocked now and
    SCR/unlearning surface the exact missing accessor instead of a wrong number.

TWO deploy constraints the caller must satisfy for a real public-suite number
(these are DATA blockers, not code blockers this adapter can fix):
  1. d_in match — the manifold SAE must be trained on the RAW residual stream of
     the SAME host model SAEBench hooks (no PCA reduction), so encode receives a
     (batch, d_model) tensor of the width the fit expects.
  2. host+datasets — sae_bench's absorption/SCR/unlearning are gemma-2-2b /
     pythia-bound (first-letter spelling, bias_in_bios, WMDP-bio). Running them on
     the manifold SAE therefore needs the manifold SAE trained on gemma-2-2b (or
     the target host) activations. The transferable, host-agnostic alternative is
     the sparse-probing concept-detection comparison on the codes directly, which
     needs neither sae_bench nor a specific host.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_manifold_saebench_sae(
    model: Any,
    *,
    model_name: str,
    hook_layer: int,
    hook_name: str | None = None,
    device: str = "cuda",
    dtype_str: str = "float32",
    d_in: int | None = None,
    d_sae: int | None = None,
):
    """Return a ``BaseSAE`` instance backed by the live curved fit ``model``.

    ``model`` is the object returned by ``gamfit.sae_manifold_fit(...)`` (has
    ``.encode``/``.reconstruct``/``.atoms``/``.training_mean``). ``d_in``/``d_sae``
    are inferred from the fit when omitted (P = len(training_mean), K =
    len(atoms)). Constructor kwargs are signature-filtered so this survives
    ``BaseSAE`` argument drift across sae_bench versions."""
    import inspect

    import torch
    from sae_bench.custom_saes.base_sae import BaseSAE

    torch_dtype = getattr(torch, dtype_str)

    if d_in is None:
        d_in = int(np.asarray(model.training_mean).reshape(-1).shape[0])
    if d_sae is None:
        d_sae = int(len(list(model.atoms)))
    if d_in <= 0 or d_sae <= 0:
        raise ValueError(f"degenerate manifold fit: d_in={d_in}, d_sae={d_sae}")

    hook = hook_name or f"blocks.{hook_layer}.hook_resid_post"
    base_kwargs = {
        "d_in": d_in,
        "d_sae": d_sae,
        "model_name": model_name,
        "hook_layer": hook_layer,
        "hook_name": hook,
        "device": device,
        "dtype": torch_dtype,
    }
    accepted = set(inspect.signature(BaseSAE.__init__).parameters)
    base_kwargs = {k: v for k, v in base_kwargs.items() if k in accepted}

    has_codes_decode = hasattr(model, "reconstruct_from_assignments")

    class _ManifoldSAE(BaseSAE):
        def __init__(self) -> None:
            super().__init__(**base_kwargs)
            self._model = model
            # BaseSAE allocates W_enc/W_dec/b_enc/b_dec; they are UNUSED here (the
            # curved core owns encode/decode). Zero them so any accidental linear
            # use is loud rather than silently reading random init.
            with torch.no_grad():
                for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
                    param = getattr(self, name, None)
                    if param is not None:
                        param.zero_()

        def _to_numpy(self, x: "torch.Tensor") -> np.ndarray:
            return np.ascontiguousarray(x.detach().to("cpu", torch.float64).numpy())

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """Real curved soft codes. Flattens any leading (batch, seq) dims,
            runs the frozen-decoder OOS solve, restores the shape with K in the
            last axis."""
            lead = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            codes = np.asarray(self._model.encode(self._to_numpy(flat)), dtype=np.float64)
            out = torch.from_numpy(np.ascontiguousarray(codes)).to(x.device, x.dtype)
            return out.reshape(*lead, out.shape[-1])

        def decode(self, feature_acts: "torch.Tensor") -> "torch.Tensor":
            if not has_codes_decode:
                raise NotImplementedError(
                    "curved decode(codes) is unavailable: the fit handle exposes "
                    "reconstruct(x) (activations->recon) but no codes->recon map. "
                    "SCR/unlearning need latent-ablated decode; add a thin pymethod "
                    "wrapping the existing Rust reconstruct_from_assignments "
                    "(latent_basis_and_sae_ffi.rs:4279). absorption/sparse-probing "
                    "do not call decode and run without it."
                )
            lead = feature_acts.shape[:-1]
            flat = feature_acts.reshape(-1, feature_acts.shape[-1])
            recon = np.asarray(
                self._model.reconstruct_from_assignments(self._to_numpy(flat)),
                dtype=np.float64,
            )
            out = torch.from_numpy(np.ascontiguousarray(recon)).to(
                feature_acts.device, feature_acts.dtype
            )
            return out.reshape(*lead, out.shape[-1])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decode(self.encode(x))

    sae = _ManifoldSAE()
    meta = {
        "d_in": d_in,
        "d_sae": d_sae,
        "hook_name": hook,
        "arch": "manifold",
        "codes_decode_available": has_codes_decode,
        "encode_only_evals": ["absorption", "sparse_probing"],
        "needs_codes_decode": ["scr", "unlearning"],
    }
    return sae, meta


def main() -> None:
    """Self-contained curved public-suite run: fit a manifold SAE on a chunk dir,
    wrap it, and hand it to the SAEBench runners in saebench_eval.py.

    This is the code path for #1942 stage 2. It will produce a real number only
    when the host + datasets constraints above are met (raw-residual d_in match,
    gemma-2-2b/pythia host with sae_bench datasets present); otherwise the
    sae_bench runners themselves report skipped/error, never fabricated."""
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--chunk-dir", type=Path, required=True, help="raw-residual chunk_*.npy dir")
    ap.add_argument("--model", required=True, help="host model id sae_bench hooks (e.g. gemma-2-2b)")
    ap.add_argument("--hook-layer", type=int, required=True)
    ap.add_argument("--hook-name", default=None)
    ap.add_argument("--evals", default="absorption,sparse_probing",
                    help="comma list; SCR/unlearning need the codes-decode pymethod")
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--d-atom", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--topology", default="circle")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import gamfit

    from extract_flat_decoder import _load_chunk_dir  # local, same dir

    X = _load_chunk_dir(args.chunk_dir, args.rows, args.seed)
    print(f"[manifold-adapter] X={X.shape}; fitting manifold SAE "
          f"K={args.K} d_atom={args.d_atom} top_k={args.top_k} topology={args.topology}", flush=True)
    model = gamfit.sae_manifold_fit(
        X, K=args.K, d_atom=args.d_atom, atom_topology=args.topology,
        assignment="topk", top_k=args.top_k, random_state=args.seed,
    )
    sae, meta = build_manifold_saebench_sae(
        model, model_name=args.model, hook_layer=args.hook_layer,
        hook_name=args.hook_name, device=args.device,
    )
    print(f"[manifold-adapter] wrapped: {meta}", flush=True)

    import saebench_eval  # local, same dir

    out_root = args.out.parent / "manifold_saebench_out"
    out_root.mkdir(parents=True, exist_ok=True)
    per_eval = {}
    for key in [e.strip() for e in args.evals.split(",") if e.strip()]:
        per_eval[key] = saebench_eval._run_one_saebench_eval(
            key, [(f"manifold_{args.topology}_K{args.K}", sae)],
            args.device, out_root, args.model, force_rerun=False,
        )
    import json

    args.out.write_text(json.dumps(
        {"meta": meta, "per_eval": per_eval, "model": args.model, "chunk_dir": str(args.chunk_dir)},
        indent=2, default=float) + "\n")
    print(f"[manifold-adapter] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
