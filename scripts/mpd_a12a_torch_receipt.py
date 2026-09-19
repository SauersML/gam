"""A12a torch half for manifold parameter decomposition (#2951).

Analysis receipt, not production. It executes parameter edits on one cached real block through
the runner `gamfit/torch/parameter_interventions.py` and checks the runner's occurrence semantics
there. The Rust stage comparison (receipts.rs) needs apply.rs and lands separately.

A use-site edit names its read by the module and op discovery reported for it, and the runner has
Rust check that label against the read the forward made at the edit's ordinal. So the script
imports an installed `gamfit` with its compiled `gamfit._rust`, and prints where each came from
before any edit runs.

The block is EleutherAI/pythia-70m layer 3 `mlp` in float64, executed as the model. Its input is
the executed `post_attention_layernorm` output, captured by a forward pre-hook over one fixed
sentence. The edit is a declared rank-R factored delta on `dense_h_to_4h.weight` from a seeded
normal.

Statements:
  R1  the tensor is read once per forward, so the global edit and the use-site#0 edit give
      bitwise-equal outputs. Positive control: doubling the delta changes the output.
  R2  every edited forward's use sites equal the discovered path.
  R3  the global edit substitutes at every read, and the use-site edit exactly at #0.
  R4  a position-scoped edit at t*: rows t != t* against native and row t* against the global
      edit, measured and reported bitwise.
  R5  every executed output dtype is float64.
  R6  a use-site edit at #0 labelled with another read's module and op refuses (LabelMismatch).
      The correctly labelled use-site edits of R1 and R4 are its positive control.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gamfit
from gamfit._binding import rust_module
from gamfit.torch import parameter_interventions as runner

SENTENCE = (
    "The river rose through the night, and by morning the old stone bridge "
    "stood alone in a wide brown lake."
)
TENSOR_ID = "dense_h_to_4h.weight"


def site_key(site) -> tuple:
    return (site.tensor_id, site.ordinal, site.transposed, site.op, site.module)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    rust = rust_module()
    label_check = hasattr(rust, "check_parameter_use_site_reads")
    print(f"RUNNER gamfit={gamfit.__file__} rust={rust.__file__} runner={runner.__file__} label_check={label_check}")
    if not label_check:
        print("FAIL the compiled extension has no check_parameter_use_site_reads")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True)
    model = model.to(dtype=torch.float64).eval()
    ids = tokenizer(SENTENCE, return_tensors="pt")["input_ids"]
    block = model.gpt_neox.layers[args.layer].mlp
    captured: list[torch.Tensor] = []
    handle = block.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    block_input = captured[0]
    batch, seq = block_input.shape[0], block_input.shape[1]
    print(
        f"BLOCK layer={args.layer} input_shape={tuple(block_input.shape)} input_dtype={block_input.dtype} "
        f"activation={type(block.act).__name__} torch={torch.__version__} threads={torch.get_num_threads()}"
    )

    registry = {row.tensor_id: row for row in runner.parameter_registry(block)}
    print(f"REGISTRY {sorted((tid, row.shape, row.dtype) for tid, row in registry.items())}")
    if TENSOR_ID not in registry:
        print(f"FAIL registry lacks {TENSOR_ID}")
        return 2
    out_features, in_features = registry[TENSOR_ID].shape

    discovered = runner.discover_parameter_use_sites(block, block_input)
    path = [site_key(site) for site in discovered]
    reads = [site for site in discovered if site.tensor_id == TENSOR_ID]
    print(f"PATH sites={len(path)} reads_of_target={len(reads)}")
    if not reads or reads[0].ordinal != 0:
        print(f"FAIL discovery found no read {TENSOR_ID}#0")
        return 2
    read = reads[0]
    print(f"READ {TENSOR_ID}#0 module={read.module!r} op={read.op!r}")

    rng = np.random.default_rng(args.seed)
    left = args.scale * rng.standard_normal((out_features, args.rank))
    right = rng.standard_normal((in_features, args.rank))
    delta = runner.FactoredDelta(left=left, right=right)
    doubled = runner.FactoredDelta(left=2.0 * left, right=right)

    native = runner.execute_native(block, block_input)
    global_edit = runner.execute_parameter_edits(
        block, block_input, [runner.GlobalParameterEdit(TENSOR_ID, delta, None)]
    )
    site_edit = runner.execute_parameter_edits(
        block,
        block_input,
        [runner.UseSiteParameterEdit(TENSOR_ID, 0, delta, None, read_module=read.module, read_op=read.op)],
    )
    doubled_edit = runner.execute_parameter_edits(
        block, block_input, [runner.GlobalParameterEdit(TENSOR_ID, doubled, None)]
    )
    t_star = seq // 2
    position_edit = runner.execute_parameter_edits(
        block,
        block_input,
        [runner.UseSiteParameterEdit(TENSOR_ID, 0, delta, (t_star,), read_module=read.module, read_op=read.op)],
        leading_shape=(batch, seq),
    )

    failures = []
    r1 = np.array_equal(global_edit.output.values, site_edit.output.values)
    r1_control = not np.array_equal(global_edit.output.values, doubled_edit.output.values)
    effect = np.max(np.abs(global_edit.output.values - native.values))
    print(f"R1 global_equals_site0_bitwise={r1} control_doubled_differs={r1_control} edit_effect_max={effect:.6e}")
    if not (r1 and r1_control):
        failures.append("R1")

    for name, execution in (("global", global_edit), ("site0", site_edit), ("position", position_edit)):
        same_path = [site_key(site) for site in execution.use_sites] == path
        print(f"R2 {name} path_equals_discovered={same_path}")
        if not same_path:
            failures.append(f"R2-{name}")

    global_substituted = sorted(site.ordinal for site in global_edit.substituted if site.tensor_id == TENSOR_ID)
    site_substituted = [(site.tensor_id, site.ordinal) for site in site_edit.substituted]
    r3 = global_substituted == sorted(site.ordinal for site in reads) and site_substituted == [(TENSOR_ID, 0)]
    print(f"R3 global_substituted={global_substituted} site0_substituted={site_substituted} ok={r3}")
    if not r3:
        failures.append("R3")

    pos = position_edit.output.values
    others = [t for t in range(seq) if t != t_star]
    other_vs_native = np.max(np.abs(pos[:, others, :] - native.values[:, others, :]))
    star_vs_global = np.max(np.abs(pos[:, t_star, :] - global_edit.output.values[:, t_star, :]))
    other_effect = np.max(np.abs(global_edit.output.values[:, others, :] - native.values[:, others, :]))
    print(
        f"R4 t_star={t_star} other_rows_vs_native_max={other_vs_native:.3e} (bitwise={bool(other_vs_native == 0.0)}) "
        f"star_row_vs_global_max={star_vs_global:.3e} (bitwise={bool(star_vs_global == 0.0)}) "
        f"global_effect_on_other_rows={other_effect:.6e}"
    )

    dtypes = {global_edit.output.dtype, site_edit.output.dtype, position_edit.output.dtype, native.dtype}
    print(f"R5 executed_dtypes={sorted(dtypes)}")
    if dtypes != {"float64"}:
        failures.append("R5")

    # Another read of this forward whose label differs from #0's is the ordinal mistake the
    # check exists for; a bias read of the same linear carries #0's own label, so it is skipped.
    other = next(((site.module, site.op) for site in discovered if (site.module, site.op) != (read.module, read.op)), None)
    if other is None:
        print("R6 FAIL the forward has no read labelled differently from #0 to mislabel it with")
        failures.append("R6")
    else:
        refusal = None
        try:
            runner.execute_parameter_edits(
                block,
                block_input,
                [runner.UseSiteParameterEdit(TENSOR_ID, 0, delta, None, read_module=other[0], read_op=other[1])],
            )
        except ValueError as exc:
            refusal = str(exc)
        r6 = refusal is not None and "LabelMismatch" in refusal
        print(f"R6 mislabelled_as={other} refused={r6} refusal={refusal!r}")
        if not r6:
            failures.append("R6")

    print(f"RESULT failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
