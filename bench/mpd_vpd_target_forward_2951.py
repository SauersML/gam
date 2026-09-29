"""The torch side of the Rust forward of VPD's 4-layer Pile target t-9d2b8f02 (#2951).

``crates/gam-mpd/examples/mpd_vpd_target_forward_2951.rs`` executes the checkpoint in binary64 with
certified radii and dumps its logits. This script runs the source module itself,
``param_decomp/pretrain/models/llama_simple_mlp.py`` from goodfire-ai/spd (``--source``, the repo
root), loaded from its file with its typing and config helpers stubbed, never re-implemented:

* ``compare``: for each dumped row, the source forward (a) as trained, float32 on the CPU, and (b)
  in float64, which needs two edits of the source's lower-precision steps: ``LlamaRMSNorm`` casts
  to float32 inside, so its forward runs in the input dtype, and the rotary tables are rebuilt by
  the source's own ``calculate_sin_cos_rotary`` at float64. Reports max |Δlogit| of the Rust logits
  against each, next to the Rust logit radius.
* ``ce``: the source's next-token cross-entropy (as trained, float32) over the first ``--rows``
  rows, the number the VPD paper reports as 2.71.

    python mpd_vpd_target_forward_2951.py compare --source SPD_REPO --run RUN_DIR --tokens ROWS.npy --dump OUT_DIR
    python mpd_vpd_target_forward_2951.py ce --source SPD_REPO --run RUN_DIR --tokens ROWS.npy --rows 512
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file


def load_source(repo: Path):
    """The source module from its file, with jaxtyping and param_decomp's base classes stubbed."""
    jaxtyping = types.ModuleType("jaxtyping")
    jaxtyping.Float = jaxtyping.Int = type("Annotation", (), {"__class_getitem__": classmethod(lambda cls, item: cls)})
    sys.modules.setdefault("jaxtyping", jaxtyping)
    import pydantic

    stubs = {
        "param_decomp": {},
        "param_decomp.base_config": {"BaseConfig": pydantic.BaseModel},
        "param_decomp.interfaces": {"LoadableModule": torch.nn.Module},
        "param_decomp.utils": {},
        "param_decomp.utils.distributed_utils": {"log0": print},
    }
    for name, attrs in stubs.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for key, value in attrs.items():
            setattr(module, key, value)
    path = repo / "param_decomp/pretrain/models/llama_simple_mlp.py"
    spec = importlib.util.spec_from_file_location("llama_simple_mlp_source", path)
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source  # pydantic resolves the config's annotations through it
    spec.loader.exec_module(source)
    return source


def build(source, run: Path, dtype: torch.dtype):
    cfg = yaml.safe_load((run / "model_config.yaml").read_text())
    model = source.LlamaSimpleMLP(source.LlamaSimpleMLPConfig(**cfg))
    state = load_file(str(run / "model_step_99999.safetensors"))
    state["lm_head.weight"] = state["wte.weight"]  # tied: the checkpoint stores the one tensor once
    model.load_state_dict(state, strict=True)
    model = model.to(dtype).eval()
    if dtype == torch.float64:
        def rms_in_dtype(self, x):
            return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

        for module in model.modules():
            if isinstance(module, source.LlamaRMSNorm):
                module.forward = types.MethodType(rms_in_dtype, module)
            if isinstance(module, source.CausalSelfAttention):
                sin, cos = module.calculate_sin_cos_rotary(module.rotary_dim, module.n_ctx, module.rotary_base, dtype=torch.float64)
                module.rotary_sin, module.rotary_cos = sin, cos
    return model


def rows(path: Path, n: int, seq: int) -> torch.Tensor:
    return torch.from_numpy(np.load(path, mmap_mode="r")[:n, : seq + 1].astype(np.int64))


def compare(args):
    source = load_source(args.source)
    report = json.loads((args.dump / "report.json").read_text())
    seq = report["seq"]
    dumped = sorted(args.dump.glob("logits_row*.npy"), key=lambda p: int(p.stem.removeprefix("logits_row")))
    data = rows(args.tokens, len(dumped), seq)
    out = []
    for precision, dtype in (("float32 (as trained)", torch.float32), ("float64", torch.float64)):
        model = build(source, args.run, dtype)
        for r, path in enumerate(dumped):
            with torch.no_grad():
                torch_logits = model(data[r : r + 1, :seq])[0][0].double().numpy()
            rust = np.load(path)
            radius = np.load(args.dump / f"logit_radius_row{r}.npy")
            diff = np.abs(rust - torch_logits)
            ce_torch = F.cross_entropy(torch.from_numpy(torch_logits), data[r, 1:]).item()
            ce_rust = F.cross_entropy(torch.from_numpy(rust), data[r, 1:]).item()
            rec = dict(row=r, torch=precision, max_abs_dlogit=float(diff.max()), mean_abs_dlogit=float(diff.mean()),
                       max_rust_radius=float(radius.max()), diff_over_radius=float((diff / radius).max()),
                       argmax_agreement=float((rust.argmax(1) == torch_logits.argmax(1)).mean()),
                       ce_torch=ce_torch, ce_rust=ce_rust)
            print(json.dumps(rec), flush=True)
            out.append(rec)
        del model
    (args.dump / "torch_compare.json").write_text(json.dumps(out, indent=1))


def cross_entropy(args):
    source = load_source(args.source)
    model = build(source, args.run, torch.float32).to(args.device)
    data = rows(args.tokens, args.rows, args.seq)
    losses = []
    with torch.no_grad():
        for i in range(0, args.rows, args.batch):
            batch = data[i : i + args.batch].to(args.device)
            logits, _ = model(batch[:, : args.seq])
            losses.append(F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), reduction="none").view(batch.shape[0], -1).mean(1).cpu())
    ce = torch.cat(losses).double()
    print(json.dumps(dict(rows=args.rows, seq=args.seq, mean_ce=ce.mean().item(), standard_error=(ce.std() / len(ce) ** 0.5).item())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["compare", "ce"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    compare(args) if args.mode == "compare" else cross_entropy(args)


if __name__ == "__main__":
    main()
