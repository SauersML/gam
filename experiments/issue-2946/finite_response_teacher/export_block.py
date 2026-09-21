"""Export one transformer MLP block and its executed outputs (issue #2946).

Thin PyTorch wrapper. It loads a cached checkpoint, runs forward passes and
writes arrays; nothing here does modelling math. The declared law (baseline
``h0``, factor ``L``), the sampled latents, the analytic ``V(P)``/``E(P)`` and
the Monte Carlo estimators all live in Rust, which reads these files.

Three stages, each one job:

``harvest``  runs the model's first ``layer + 1`` blocks over a named context
             set and writes, into ``--out-dir``:
               post_norm.npy       float32 (rows x hidden) MLP input at ``layer``,
                                   i.e. the post-norm activation the MLP sees
               pre_norm.npy        float32 (rows x hidden) input of that layer's
                                   ``post_attention_layernorm``: the stream the
                                   norm in front of the MLP reads, same rows
               <param>.npy         float64, every parameter of that MLP under
                                   its torch name (e.g. ``gate_proj.weight``)
               post_attention_layernorm.<param>.npy
                                   float64, every parameter of that norm
               meta.json           checkpoint revision, config fields, context
                                   set definition, row count, the norm's class
                                   and epsilon, library versions

``pair``     runs the first ``layer + 2`` blocks over the same kind of context
             set and writes, for composing the MLP of block ``layer`` into the
             MLP of block ``layer + 1``, float32 (rows x hidden) arrays from one
             forward pass (so row ``i`` is one token position in every file):
               x_in.npy            input of ``layers[layer]``: the residual
                                   stream entering it
               h_first.npy         input of ``layers[layer].mlp``
               y_first.npy         output of ``layers[layer].mlp``
               x_first.npy         output of ``layers[layer]``: the residual
                                   stream entering ``layers[layer + 1]``
               h_second.npy        input of ``layers[layer + 1].mlp``
               y_second.npy        output of ``layers[layer + 1].mlp``
               layer<i>.<module>.<param>.npy
                                   float64, every parameter of the MLP and of
                                   ``post_attention_layernorm`` (the norm in
                                   front of the MLP) of both blocks
               meta.json           as ``harvest``, plus both layer indices, the
                                   norm's class and epsilon, whether the block
                                   uses a parallel residual, and each array's md5

``execute``  runs that MLP alone, in float64, on rows ``h`` that Rust wrote
             (``h = h0 + L z``) and writes ``F(h)`` as float64 (rows x hidden).

Checkpoints load with ``local_files_only=True``: the exporter never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _set_threads() -> None:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None:
        torch.set_num_threads(int(cpus))


def _truncated_model(model_id: str, revision: str, layer: int, dtype: torch.dtype):
    """Load the decoder with only blocks ``0..=layer`` and no vocabulary head.

    Nothing past the exported MLP is needed, so the resident weights are the
    embedding plus ``layer + 1`` blocks.
    """
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
    if not 0 <= layer < cfg.num_hidden_layers:
        raise SystemExit(f"--layer {layer} outside 0..{cfg.num_hidden_layers - 1}")
    full_depth = cfg.num_hidden_layers
    cfg.num_hidden_layers = layer + 1
    model = AutoModel.from_pretrained(
        model_id, revision=revision, config=cfg, dtype=dtype, local_files_only=True
    )
    model.eval()
    return cfg, full_depth, model


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_windows(args: argparse.Namespace, tok) -> tuple[torch.Tensor, dict]:
    """The context set: the named split's records in order, each tokenized without
    special tokens and concatenated, cut into consecutive windows."""
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    need = args.windows * args.seq_len
    stream: list[int] = []
    records_used = 0
    for record in ds:
        if len(stream) >= need:
            break
        stream.extend(tok(record["text"], add_special_tokens=False)["input_ids"])
        records_used += 1
    if len(stream) < need:
        raise SystemExit(f"split holds {len(stream)} tokens, the context set needs {need}")
    windows = torch.tensor(stream[:need], dtype=torch.long).view(args.windows, args.seq_len)
    context_set = {
        "dataset": args.dataset,
        "config": args.dataset_config,
        "split": args.split,
        "records_used": records_used,
        "windows": args.windows,
        "seq_len": args.seq_len,
        "skip_positions": args.skip_positions,
    }
    return windows, context_set


def _versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def harvest(args: argparse.Namespace) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    cfg, full_depth, model = _truncated_model(args.model, args.revision, args.layer, torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    backbone = model
    mlp = backbone.layers[args.layer].mlp
    norm = backbone.layers[args.layer].post_attention_layernorm
    hidden = int(cfg.hidden_size)
    print(
        f"[harvest] model={args.model} type={cfg.model_type} act={cfg.hidden_act} "
        f"hidden={hidden} intermediate={cfg.intermediate_size} layer={args.layer}/{full_depth}",
        flush=True,
    )

    windows, context_set = _context_windows(args, tok)

    keep = args.seq_len - args.skip_positions
    rows = args.windows * keep
    arrays = {
        key: np.lib.format.open_memmap(
            os.path.join(args.out_dir, f"{key}.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(rows, hidden),
        )
        for key in ("post_norm", "pre_norm")
    }
    captured: dict[str, torch.Tensor] = {}

    def grab(key: str):
        def hook(_module, inputs):
            captured[key] = inputs[0]

        return hook

    handles = [
        mlp.register_forward_pre_hook(grab("post_norm")),
        norm.register_forward_pre_hook(grab("pre_norm")),
    ]
    filled = 0
    with torch.no_grad():
        for start in range(0, args.windows, args.batch):
            backbone(input_ids=windows[start : start + args.batch], use_cache=False)
            for key, acts in arrays.items():
                block = captured[key][:, args.skip_positions :, :].reshape(-1, hidden)
                acts[filled : filled + block.shape[0]] = block.numpy()
            filled += block.shape[0]
            print(f"[harvest] windows {start + block.shape[0] // keep}/{args.windows}", flush=True)
    for handle in handles:
        handle.remove()
    for acts in arrays.values():
        acts.flush()

    params = {}
    for name, p in mlp.named_parameters():
        path = os.path.join(args.out_dir, f"{name}.npy")
        np.save(path, p.detach().to(torch.float64).numpy())
        params[name] = list(p.shape)
    norm_params = {}
    for name, p in norm.named_parameters():
        np.save(os.path.join(args.out_dir, f"post_attention_layernorm.{name}.npy"), p.detach().to(torch.float64).numpy())
        norm_params[name] = list(p.shape)

    meta = {
        "stage": "harvest",
        "model": args.model,
        "revision": args.revision,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "model_type": cfg.model_type,
        "hidden_act": cfg.hidden_act,
        "hidden_size": hidden,
        "intermediate_size": int(cfg.intermediate_size),
        "layer": args.layer,
        "num_hidden_layers": full_depth,
        "checkpoint_dtype": str(getattr(cfg, "torch_dtype", None)),
        "forward_dtype": "float32",
        "mlp_class": type(mlp).__name__,
        "mlp_params": params,
        "norm_class": type(norm).__name__,
        "norm_epsilon": float(getattr(norm, "variance_epsilon", getattr(norm, "eps", float("nan")))),
        "norm_params": norm_params,
        "context_set": context_set,
        "post_norm_rows": rows,
        "pre_norm_rows": rows,
        "versions": _versions(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[harvest] wrote {rows} rows and {len(params)} parameters in {meta['seconds']}s", flush=True)
    return 0


# What each ``pair`` array holds, as recorded in its meta.json.
PAIR_ROWS = {
    "x_in": "input of layers[first] (forward pre-hook): the residual stream entering it",
    "h_first": "input of layers[first].mlp (forward pre-hook)",
    "y_first": "output of layers[first].mlp (forward hook)",
    "x_first": "output of layers[first] (forward hook): the residual stream entering layers[second]",
    "h_second": "input of layers[second].mlp (forward pre-hook)",
    "y_second": "output of layers[second].mlp (forward hook)",
}


def pair(args: argparse.Namespace) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    first = args.layer
    second = first + 1
    cfg, full_depth, model = _truncated_model(args.model, args.revision, second, torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    blocks = {first: model.layers[first], second: model.layers[second]}
    for index, block in blocks.items():
        if not hasattr(block, "post_attention_layernorm"):
            raise SystemExit(f"layers[{index}] ({type(block).__name__}) has no post_attention_layernorm")
    hidden = int(cfg.hidden_size)
    print(
        f"[pair] model={args.model} type={cfg.model_type} act={cfg.hidden_act} "
        f"hidden={hidden} intermediate={cfg.intermediate_size} layers={first},{second}/{full_depth}",
        flush=True,
    )

    windows, context_set = _context_windows(args, tok)

    keep = args.seq_len - args.skip_positions
    rows = args.windows * keep
    arrays = {
        name: np.lib.format.open_memmap(
            os.path.join(args.out_dir, f"{name}.npy"), mode="w+", dtype=np.float32, shape=(rows, hidden)
        )
        for name in PAIR_ROWS
    }
    captured: dict[str, torch.Tensor] = {}

    def before(name: str):
        def hook(_module, inputs):
            captured[name] = inputs[0]

        return hook

    def entering(name: str):
        def hook(_module, inputs, kwargs):
            captured[name] = inputs[0] if inputs else kwargs["hidden_states"]

        return hook

    def after(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return hook

    handles = [
        blocks[first].register_forward_pre_hook(entering("x_in"), with_kwargs=True),
        blocks[first].mlp.register_forward_pre_hook(before("h_first")),
        blocks[first].mlp.register_forward_hook(after("y_first")),
        blocks[first].register_forward_hook(after("x_first")),
        blocks[second].mlp.register_forward_pre_hook(before("h_second")),
        blocks[second].mlp.register_forward_hook(after("y_second")),
    ]
    filled = 0
    with torch.no_grad():
        for start in range(0, args.windows, args.batch):
            captured.clear()
            model(input_ids=windows[start : start + args.batch], use_cache=False)
            count = None
            for name, array in arrays.items():
                block = captured[name][:, args.skip_positions :, :].reshape(-1, hidden)
                if count is not None and block.shape[0] != count:
                    raise SystemExit(f"{name} captured {block.shape[0]} rows, the others {count}")
                count = block.shape[0]
                array[filled : filled + count] = block.numpy()
            filled += count
            print(f"[pair] windows {start + count // keep}/{args.windows}", flush=True)
    for handle in handles:
        handle.remove()
    for array in arrays.values():
        array.flush()

    params = {}
    for index, block in blocks.items():
        for module in ("mlp", "post_attention_layernorm"):
            for name, p in getattr(block, module).named_parameters():
                key = f"layer{index}.{module}.{name}"
                np.save(os.path.join(args.out_dir, f"{key}.npy"), p.detach().to(torch.float64).numpy())
                params[key] = list(p.shape)

    norm = blocks[first].post_attention_layernorm
    meta = {
        "stage": "pair",
        "model": args.model,
        "revision": args.revision,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "model_type": cfg.model_type,
        "hidden_act": cfg.hidden_act,
        "hidden_size": hidden,
        "intermediate_size": int(cfg.intermediate_size),
        "layers": {"first": first, "second": second},
        "num_hidden_layers": full_depth,
        "use_parallel_residual": getattr(cfg, "use_parallel_residual", None),
        "norm_class": type(norm).__name__,
        "norm_eps": getattr(norm, "eps", getattr(norm, "variance_epsilon", None)),
        "checkpoint_dtype": str(getattr(cfg, "torch_dtype", None)),
        "forward_dtype": "float32",
        "mlp_class": type(blocks[first].mlp).__name__,
        "block_class": type(blocks[first]).__name__,
        "params": params,
        "context_set": context_set,
        "rows": rows,
        "arrays": {name: {"holds": holds, "md5": _md5(os.path.join(args.out_dir, f"{name}.npy"))}
                   for name, holds in PAIR_ROWS.items()},
        "versions": _versions(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[pair] wrote {rows} rows of {len(arrays)} arrays and {len(params)} parameters in {meta['seconds']}s", flush=True)
    return 0


def execute(args: argparse.Namespace) -> int:
    t0 = time.time()
    cfg, full_depth, model = _truncated_model(args.model, args.revision, args.layer, torch.float32)
    mlp = model.layers[args.layer].mlp.to(torch.float64)
    del model
    inputs = np.load(args.inputs, mmap_mode="r")
    if inputs.ndim != 2 or inputs.shape[1] != cfg.hidden_size or inputs.dtype != np.float64:
        raise SystemExit(
            f"--inputs must be float64 (rows x {cfg.hidden_size}); got {inputs.dtype} {inputs.shape}"
        )
    rows = inputs.shape[0]
    outputs = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float64, shape=(rows, cfg.hidden_size)
    )
    with torch.no_grad():
        for start in range(0, rows, args.batch):
            h = torch.from_numpy(np.ascontiguousarray(inputs[start : start + args.batch]))
            outputs[start : start + h.shape[0]] = mlp(h).numpy()
    outputs.flush()
    meta = {
        "stage": "execute",
        "model": args.model,
        "revision": args.revision,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "layer": args.layer,
        "mlp_class": type(mlp).__name__,
        "forward_dtype": "float64",
        "inputs": args.inputs,
        "inputs_md5": _md5(args.inputs),
        "rows": rows,
        "outputs": args.out,
        "outputs_md5": _md5(args.out),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[execute] {type(mlp).__name__} on {rows} rows in {meta['seconds']}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    h = sub.add_parser("harvest", help="post-norm activations and MLP parameters at one layer")
    p = sub.add_parser("pair", help="rows and parameters of blocks layer and layer + 1 from one forward pass")
    for stage in (h, p):
        stage.add_argument("--model", required=True, help="checkpoint id, resolved from the local HF cache")
        stage.add_argument("--revision", required=True, help="pinned checkpoint commit sha")
        stage.add_argument("--layer", type=int, required=True)
        stage.add_argument("--dataset", required=True)
        stage.add_argument("--dataset-config", required=True)
        stage.add_argument("--split", required=True)
        stage.add_argument("--windows", type=int, required=True)
        stage.add_argument("--seq-len", type=int, required=True)
        stage.add_argument(
            "--skip-positions",
            type=int,
            required=True,
            help="leading positions of each window left out of the rows",
        )
        stage.add_argument("--batch", type=int, required=True, help="windows per forward pass")
        stage.add_argument("--out-dir", required=True)

    e = sub.add_parser("execute", help="run the MLP alone in float64 on rows Rust wrote")
    e.add_argument("--model", required=True)
    e.add_argument("--revision", required=True, help="pinned checkpoint commit sha")
    e.add_argument("--layer", type=int, required=True)
    e.add_argument("--inputs", required=True, help="float64 .npy (rows x hidden)")
    e.add_argument("--batch", type=int, required=True, help="rows per forward call")
    e.add_argument("--out", required=True, help="float64 .npy (rows x hidden) F(h)")

    args = ap.parse_args(argv)
    _set_threads()
    return {"harvest": harvest, "pair": pair, "execute": execute}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
