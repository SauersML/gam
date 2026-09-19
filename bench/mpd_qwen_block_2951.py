"""Execute Qwen3's SwiGLU MLP under declared parameter edits (issue #2951, A12).

Thin PyTorch wrapper (SPEC 8: interaction with external software). It builds the
source's own ``Qwen3MLP`` from the cached configuration in float64, loads the
float64 parameters that
``experiments/issue-2946/finite_response_teacher/export_block.py harvest`` wrote,
and runs ``gamfit.torch.parameter_interventions`` on the harvested post-norm rows
under the declared edits. The receipt math (native stages, bands, comparisons)
lives in Rust, in ``parameter_decomposition::receipts``; nothing here draws a
number or scores a stage.
The checks here are plumbing: every captured stage must be float64, and the
hooked write must be bit-identical to the output the runner returned.

An edit is declared as literal components of one stored tensor ``W``
(``(d_out, d_in)``), each by an index and a coefficient ``s``:

* ``row j``:    ``left[j, c] = s`` and ``right[:, c] = W[j, :]``, so ``ΔW`` scales row
  ``j`` by ``1 + s``;
* ``column k``: ``left[:, c] = W[:, k]`` and ``right[k, c] = s``, so ``ΔW`` scales
  column ``k`` by ``1 + s``.

Both factors are formed by assignment and copies of exported values, so the
factors torch applies are exactly the files Rust reads.

Two stages, each one call:

``registry``  writes ``registry.json``: the module's parameter registry and every
              use of every registered tensor in one unedited forward.
``execute``   runs every declared setting and writes, into ``--out-dir``, one
              float64 ``.npy`` per array id, as the surface's CLI transport reads
              them: ``inputs``; per setting ``<name>.gate``, ``.activation``,
              ``.up``, ``.hidden``, ``.output`` (``rows x width``); per distinct
              edit ``edit<k>.left`` and ``edit<k>.right``; ``ones<C>``; and
              ``manifest.json`` naming every id, file, setting and edit, with the
              executor's dtype, device and TF32 flag as recorded before any copy.

A setting with no edits is the all-on setting and runs ``execute_native``.
Checkpoints load with ``local_files_only=True``: nothing downloads.
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
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from gamfit.torch.parameter_interventions import (
    FactoredDelta,
    GlobalParameterEdit,
    UseSiteParameterEdit,
    discover_parameter_use_sites,
    execute_native,
    execute_parameter_edits,
    parameter_registry,
    parameter_values,
)

PARAMETERS = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
STAGES = ("gate", "activation", "up", "hidden", "output")


def _set_threads() -> None:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None:
        torch.set_num_threads(int(cpus))


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _block(harvest: str) -> tuple[dict, Qwen3MLP]:
    """The harvested layer's ``Qwen3MLP`` in float64, holding the exported values."""
    meta = _read_json(os.path.join(harvest, "meta.json"))
    if meta.get("stage") != "harvest" or meta.get("mlp_class") != "Qwen3MLP":
        raise SystemExit(f"{harvest} is not a Qwen3MLP harvest: {meta.get('stage')} {meta.get('mlp_class')}")
    cfg = AutoConfig.from_pretrained(meta["model"], revision=meta["revision"], local_files_only=True)
    mlp = Qwen3MLP(cfg).to(torch.float64)
    values = {
        name: torch.from_numpy(np.load(os.path.join(harvest, f"{name}.npy")))
        for name in PARAMETERS
    }
    mlp.load_state_dict(values, strict=True)
    mlp.eval()
    return meta, mlp


def _rows(harvest: str, rows: int, hidden: int) -> np.ndarray:
    """The leading ``rows`` post-norm rows, widened exactly to float64."""
    post_norm = np.load(os.path.join(harvest, "post_norm.npy"), mmap_mode="r")
    if post_norm.ndim != 2 or post_norm.shape[1] != hidden or post_norm.shape[0] < rows:
        raise SystemExit(f"post_norm.npy has shape {post_norm.shape}; need at least {rows} x {hidden}")
    return np.ascontiguousarray(np.asarray(post_norm[:rows], dtype=np.float64))


def registry(args: argparse.Namespace) -> int:
    meta, mlp = _block(args.harvest)
    rows = _rows(args.harvest, args.rows, int(meta["hidden_size"]))
    inputs = torch.from_numpy(rows).view(1, args.rows, -1)
    tensors = parameter_registry(mlp)
    uses = discover_parameter_use_sites(mlp, inputs)
    document = {
        "stage": "registry",
        "harvest": args.harvest,
        "model": meta["model"],
        "revision": meta["revision"],
        "layer": meta["layer"],
        "rows": args.rows,
        "activation_class": type(mlp.act_fn).__name__,
        "tensors": [
            {"tensor_id": t.tensor_id, "aliases": list(t.aliases), "shape": list(t.shape), "dtype": t.dtype}
            for t in tensors
        ],
        "use_sites": [
            {"tensor_id": u.tensor_id, "ordinal": u.ordinal, "transposed": u.transposed, "op": u.op, "module": u.module}
            for u in uses
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(document, fh, indent=2)
    print(f"[registry] {len(tensors)} tensors, {len(uses)} use sites, act={document['activation_class']}", flush=True)
    return 0


def _literal_factors(weight: np.ndarray, components: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """``left`` and ``right`` of a literal-component edit, by assignment only."""
    rows, cols = weight.shape
    left = np.zeros((rows, len(components)), dtype=np.float64)
    right = np.zeros((cols, len(components)), dtype=np.float64)
    for c, component in enumerate(components):
        index = int(component["index"])
        coefficient = float(component["coefficient"])
        if component["kind"] == "row":
            left[index, c] = coefficient
            right[:, c] = weight[index, :]
        elif component["kind"] == "column":
            left[:, c] = weight[:, index]
            right[index, c] = coefficient
        else:
            raise SystemExit(f"a component kind must be row or column; got {component['kind']!r}")
    return left, right


def _edit(declared: dict, delta: FactoredDelta, labels: dict[tuple[str, int], tuple[str, str]]):
    tensor_id = declared["tensor_id"]
    positions = declared["positions"]
    if declared["scope"] == "global":
        return GlobalParameterEdit(tensor_id=tensor_id, delta=delta, positions=positions)
    if declared["scope"] == "use_site":
        ordinal = int(declared["ordinal"])
        if (tensor_id, ordinal) not in labels:
            raise SystemExit(f"discovery found no use site {tensor_id}#{ordinal} to name the edit's read")
        module, op = labels[(tensor_id, ordinal)]
        return UseSiteParameterEdit(
            tensor_id=tensor_id,
            ordinal=ordinal,
            delta=delta,
            positions=positions,
            read_module=module,
            read_op=op,
        )
    raise SystemExit(f"an edit scope must be global or use_site; got {declared['scope']!r}")


def execute(args: argparse.Namespace) -> int:
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    files: dict[str, str] = {}

    def save(array_id: str, values: np.ndarray) -> str:
        path = os.path.join(args.out_dir, f"{array_id}.npy")
        np.save(path, values)
        files[array_id] = path
        return array_id

    meta, mlp = _block(args.harvest)
    hidden = int(meta["hidden_size"])
    declaration = _read_json(args.settings)
    rows = int(declaration["rows"])
    inputs = torch.from_numpy(_rows(args.harvest, rows, hidden)).view(1, rows, hidden)
    weights = {name: parameter_values(mlp, name) for name in PARAMETERS}
    # Discovery names each read a use-site edit targets. It runs before the stage
    # hooks are registered, so its forward is not captured.
    labels = {
        (use.tensor_id, use.ordinal): (use.module, use.op)
        for use in discover_parameter_use_sites(mlp, inputs)
    }
    captured: dict[str, torch.Tensor] = {}

    def keep_output(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().clone()

        return hook

    def keep_input(name: str):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().clone()

        return hook

    handles = [
        mlp.gate_proj.register_forward_hook(keep_output("gate")),
        mlp.act_fn.register_forward_hook(keep_output("activation")),
        mlp.up_proj.register_forward_hook(keep_output("up")),
        mlp.down_proj.register_forward_pre_hook(keep_input("hidden")),
        mlp.down_proj.register_forward_hook(keep_output("output")),
    ]
    edit_ids: dict[str, str] = {}
    ran = []
    executor = {"dtypes": set(), "devices": set()}
    try:
        for setting in declaration["settings"]:
            captured.clear()
            edits = []
            declared_edits = []
            for declared in setting["edits"]:
                key = json.dumps([declared["tensor_id"], declared["components"]], sort_keys=True)
                left, right = _literal_factors(weights[declared["tensor_id"]], declared["components"])
                if key not in edit_ids:
                    edit_ids[key] = f"edit{len(edit_ids)}"
                    save(f"{edit_ids[key]}.left", left)
                    save(f"{edit_ids[key]}.right", right)
                ones = f"ones{left.shape[1]}"
                if ones not in files:
                    save(ones, np.ones(left.shape[1], dtype=np.float64))
                edits.append(_edit(declared, FactoredDelta(left=left, right=right), labels))
                declared_edits.append(
                    {
                        **declared,
                        "left": f"{edit_ids[key]}.left",
                        "right": f"{edit_ids[key]}.right",
                        "coefficients": ones,
                    }
                )
            if edits:
                result = execute_parameter_edits(mlp, inputs, edits, leading_shape=(1, rows))
                output = result.output
                substituted = [site.use_site_id for site in result.substituted]
            else:
                output = execute_native(mlp, inputs)
                substituted = []
            for name in STAGES:
                executor["dtypes"].add(str(captured[name].dtype).removeprefix("torch."))
                executor["devices"].add(str(captured[name].device))
            executor["devices"].add(output.device)
            if executor["dtypes"] != {"float64"}:
                raise SystemExit(f"setting {setting['name']!r}: the executed stages ran in {sorted(executor['dtypes'])}")
            if not torch.equal(captured["output"].reshape(rows, hidden), torch.from_numpy(output.values).reshape(rows, hidden)):
                raise SystemExit(f"setting {setting['name']!r}: the hooked output is not the returned output")
            externals = {
                name: save(f"{setting['name']}.{name}", captured[name].reshape(rows, -1).numpy()) for name in STAGES
            }
            ran.append(
                {
                    "name": setting["name"],
                    "edits": declared_edits,
                    "externals": externals,
                    "substituted": substituted,
                    "tf32_matmul": output.tf32_matmul,
                }
            )
            print(f"[execute] {setting['name']} substituted={substituted}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    # The runner reads the TF32 flag right before each forward; one export has one flag.
    tf32_flags = sorted({entry["tf32_matmul"] for entry in ran})
    if len(tf32_flags) != 1:
        raise SystemExit(f"the settings ran under different TF32 flags: {tf32_flags}")
    manifest = {
        "stage": "execute",
        "harvest": args.harvest,
        "model": meta["model"],
        "revision": meta["revision"],
        "layer": meta["layer"],
        "activation_class": type(mlp.act_fn).__name__,
        "rows": rows,
        "weights": {name: os.path.join(args.harvest, f"{name}.npy") for name in PARAMETERS},
        "external_dtype": sorted(executor["dtypes"]),
        "external_device": sorted(executor["devices"]),
        "tf32_matmul": tf32_flags[0],
        "settings_md5": _md5(args.settings),
        "files": {array_id: {"path": path, "md5": _md5(path)} for array_id, path in files.items()},
        "settings": ran,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[execute] {len(ran)} settings on {rows} rows, {len(files)} arrays in {manifest['seconds']}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("registry", help="parameter registry and use sites of the harvested block")
    r.add_argument("--harvest", required=True, help="export_block.py harvest directory of a Qwen3MLP")
    r.add_argument("--rows", type=int, required=True, help="leading post-norm rows, one (1, rows) unit")
    r.add_argument("--out", required=True, help="registry.json")

    e = sub.add_parser("execute", help="run every declared setting and write one .npy per array id")
    e.add_argument("--harvest", required=True)
    e.add_argument("--settings", required=True, help="settings.json: rows and the declared literal-component edits")
    e.add_argument("--out-dir", required=True)

    args = ap.parse_args(argv)
    _set_threads()
    return registry(args) if args.stage == "registry" else execute(args)


if __name__ == "__main__":
    sys.exit(main())
