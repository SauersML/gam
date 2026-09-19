"""Parameter-edit execution contract for manifold parameter decomposition (#2951).

A float64 fixture ties one ``Parameter`` under two names (``embed.weight`` and
``head.weight``) and runs one MLP body twice, so the embedding and every body
tensor each have two use sites. References are hand-built forwards through the
same torch ops, so every comparison is bitwise unless a derived roundoff bound
is named:

* the registry names one id per tensor object and lists its aliases, and
  values leave as an exact float64 widening of every float format;
* use sites are the tensor-returning reads in execution order, keyed
  ``{tensor_id}#{ordinal}``, oriented by the linear map that multiplies them,
  and a dtype query is not a use;
* a zero delta reproduces the native forward bitwise, and a nonzero one does not;
* a global edit, dense or factored, equals ``torch.func.functional_call`` on
  ``W + ΔW`` with tied weights;
* a use-site edit of the tied tensor or of the shared body moves only that use,
  and differs from the global edit;
* a use-site edit names its read by discovery's module and op, and Rust refuses
  it when another module or op made the read at its ordinal;
* declared positions take only those rows from the edited run of the op, on
  the op's current intervened input, and declaring every row equals the
  every-position edit;
* a linear use's cotangent comes back as the rows its op multiplied and the
  cotangent rows at what it wrote, restricted to declared rows, with cotangents
  entering at declared readouts; the rows reconstruct torch's dense weight
  gradient in the use site's orientation within the roundoff of two dot products;
* every refusal fires on an input built to trigger it, next to a control that
  executes.

Fixed seeds throughout; no clock entropy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.overrides import resolve_name

from gamfit.torch.parameter_interventions import (
    FactoredDelta,
    GlobalParameterEdit,
    OutputReadout,
    ParameterUseSite,
    UseCotangent,
    UseSiteInputReadout,
    UseSiteOutputReadout,
    UseSiteParameterEdit,
    discover_parameter_use_sites,
    execute_native,
    execute_parameter_cotangents,
    execute_parameter_edits,
    parameter_registry,
    parameter_values,
)

_TOKENS = torch.tensor([[3, 1, 4, 1, 5, 2]])
_LEADING = (1, 6)
_EVERY_ROW = tuple(range(_LEADING[1]))
_LINEAR = resolve_name(F.linear)


class _WeightScale(torch.nn.Module):
    """``x`` cast to the weight's dtype, times the weight: one dtype query and
    one value read of ``weight``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.weight.dtype) * self.weight


class _TiedSharedMlp(torch.nn.Module):
    """Tokens -> embedding -> one MLP body applied twice -> weight scale -> tied head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(4, 6, dtype=torch.float64),
            torch.nn.GELU(),
            torch.nn.Linear(6, 4, dtype=torch.float64),
        )
        self.scale = _WeightScale(4)
        self.head = torch.nn.Linear(4, 7, bias=False, dtype=torch.float64)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + self.body(x)
        x = x + self.body(x)
        return self.head(self.scale(x))


class _TransposedTie(torch.nn.Module):
    """Tokens -> embedding, read out through the embedding's transpose in the
    root forward: ``x @ W.t()``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) @ self.embed.weight.t()


class _TransposedLinear(torch.nn.Module):
    """``F.linear(x, W.t())`` with a stored ``(4, 5)`` tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)
        self.weight = torch.nn.Parameter(torch.zeros(4, 5, dtype=torch.float64))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.embed(tokens), self.weight.t())


class _PlainMatmul(torch.nn.Module):
    """``x @ W`` with a stored ``(4, 5)`` tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)
        self.weight = torch.nn.Parameter(torch.zeros(4, 5, dtype=torch.float64))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) @ self.weight


class _TransposeCopied(torch.nn.Module):
    """``x @ W.t().contiguous()``: the transpose view is consumed by a copy."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) @ self.embed.weight.t().contiguous()


class _WithUnusedParameter(torch.nn.Module):
    """A linear map plus a registered parameter the forward never reads."""

    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.unused = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x)


class _TwoViewsOfOneStorage(torch.nn.Module):
    """Two distinct ``Parameter`` objects over halves of one storage."""

    def __init__(self) -> None:
        super().__init__()
        base = torch.zeros(4, dtype=torch.float64)
        self.left = torch.nn.Parameter(base[:2])
        self.right = torch.nn.Parameter(base[2:])


class _TupleOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.linear(x),)


def _seeded(model: torch.nn.Module) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(2951)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))
    return model


def _model() -> _TiedSharedMlp:
    return _seeded(_TiedSharedMlp())


def _sites_of(sites: tuple[ParameterUseSite, ...], tensor_id: str) -> list[ParameterUseSite]:
    return [site for site in sites if site.tensor_id == tensor_id]


def _ramp(shape: tuple[int, ...]) -> np.ndarray:
    """A deterministic dense array that moves every element but the first by a
    clearly nonzero amount."""
    size = int(np.prod(shape))
    return 0.25 * np.arange(size, dtype=np.float64).reshape(shape) / size


def _original(model: torch.nn.Module, tensor_id: str) -> torch.Tensor:
    return torch.from_numpy(parameter_values(model, tensor_id))


def _use_site_edit(
    model: torch.nn.Module, tensor_id: str, ordinal: int, delta: Any, positions: Any
) -> UseSiteParameterEdit:
    """A use-site edit named by the module and op discovery reports for its read,
    as a driver builds one."""
    (site,) = [
        site
        for site in discover_parameter_use_sites(model, _TOKENS)
        if site.use_site_id == f"{tensor_id}#{ordinal}"
    ]
    return UseSiteParameterEdit(tensor_id, ordinal, delta, positions, site.module, site.op)


def _splice(clean: torch.Tensor, moved: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    index = torch.as_tensor(positions)
    combined = clean.clone()
    combined[:, index] = moved[:, index]
    return combined


def _reconstruction_bound(use: UseCotangent) -> np.ndarray:
    """Each dense-gradient entry is one n-term dot product, so torch's entry and
    the factored reconstruction each lie within γ_n · Σ_r |g_r h_r| of the exact
    value (Higham, Accuracy and Stability, §3.1): they differ by at most twice
    that. The bound is in the orientation of ``Σ_r g_r h_rᵀ``."""
    n = use.outputs.shape[0]
    eps = np.finfo(np.float64).eps
    gamma = n * eps / (1.0 - n * eps)
    return 2.0 * gamma * (np.abs(use.outputs).T @ np.abs(use.inputs))


def test_registry_names_one_id_per_tensor_object_and_lists_its_aliases() -> None:
    model = _model()
    rows = parameter_registry(model)
    assert [row.tensor_id for row in rows] == [
        "embed.weight",
        "body.0.weight",
        "body.0.bias",
        "body.2.weight",
        "body.2.bias",
        "scale.weight",
    ]
    assert rows[0].aliases == ("head.weight",)
    assert all(row.aliases == () for row in rows[1:])
    assert rows[1].shape == (6, 4)
    assert all(row.dtype == "float64" for row in rows)

    exported = parameter_values(model, "embed.weight")
    assert np.array_equal(exported, model.embed.weight.detach().numpy())
    exported[0, 0] += 1.0
    assert not np.array_equal(exported, model.embed.weight.detach().numpy())


def test_values_leave_as_an_exact_float64_widening_with_the_source_dtype() -> None:
    single = _seeded(torch.nn.Linear(3, 2, dtype=torch.float32))
    exported = parameter_values(single, "weight")
    assert exported.dtype == np.float64
    # Narrowing back recovers the source bits, so the widening lost nothing.
    assert exported.astype(np.float32).tobytes() == single.weight.detach().numpy().tobytes()
    assert parameter_registry(single)[0].dtype == "float32"
    for dtype, name in ((torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
        narrow = _seeded(torch.nn.Linear(3, 2, dtype=torch.float32)).to(dtype)
        widened = parameter_values(narrow, "weight")
        assert widened.dtype == np.float64
        assert torch.equal(torch.from_numpy(widened).to(dtype), narrow.weight.detach())
        assert parameter_registry(narrow)[0].dtype == name


def test_registry_refuses_distinct_parameters_sharing_one_storage() -> None:
    with pytest.raises(ValueError, match="sharing one storage"):
        parameter_registry(_TwoViewsOfOneStorage())


def test_use_sites_are_tensor_returning_reads_in_execution_order() -> None:
    sites = discover_parameter_use_sites(_model(), _TOKENS)
    body = ["body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias"]
    assert [site.tensor_id for site in sites] == [
        "embed.weight",
        *body,
        *body,
        "scale.weight",
        "embed.weight",
    ]
    # Only the F.linear weights are multiplied by a linear map; the lookup, the
    # biases and the element-wise scale are stored reads.
    assert [site.linear for site in sites] == [
        False, True, False, True, False, True, False, True, False, False, True
    ]
    embedding, head = _sites_of(sites, "embed.weight")
    assert (embedding.use_site_id, embedding.transposed, embedding.op, embedding.module) == (
        "embed.weight#0",
        False,
        resolve_name(F.embedding),
        "embed",
    )
    # The head reads the tied tensor through its alias; the read is still embed.weight's.
    assert (head.use_site_id, head.transposed, head.op, head.module) == (
        "embed.weight#1",
        False,
        resolve_name(F.linear),
        "head",
    )
    for tensor_id in body:
        assert [(site.use_site_id, site.module) for site in _sites_of(sites, tensor_id)] == [
            (f"{tensor_id}#0", tensor_id[:6]),
            (f"{tensor_id}#1", tensor_id[:6]),
        ]
    # ``_WeightScale`` queries ``weight.dtype`` before multiplying; only the
    # multiply returns a tensor, so the weight has exactly one use.
    assert [site.use_site_id for site in _sites_of(sites, "scale.weight")] == ["scale.weight#0"]


def test_a_transpose_view_multiplied_by_matmul_is_an_identity_use_of_the_same_tensor() -> None:
    model = _seeded(_TransposedTie())
    sites = discover_parameter_use_sites(model, _TOKENS)
    # ``x @ W.t()`` applies ``A = W`` in ``y = x · Aᵀ``: the ``.t()`` call implements
    # the read, and the matmul consuming the view decides the orientation.
    assert [
        (site.use_site_id, site.linear, site.transposed, site.op, site.module) for site in sites
    ] == [
        ("embed.weight#0", False, False, resolve_name(F.embedding), "embed"),
        ("embed.weight#1", True, False, resolve_name(torch.Tensor.t), ""),
    ]
    original = _original(model, "embed.weight")
    delta = _ramp((7, 4))
    run = execute_parameter_edits(
        model, _TOKENS, [_use_site_edit(model, "embed.weight", 1, delta, None)]
    )
    reference = F.embedding(_TOKENS, original) @ (original + torch.from_numpy(delta)).t()
    assert np.array_equal(run.output.values, reference.numpy())
    assert [(site.use_site_id, site.transposed) for site in run.substituted] == [
        ("embed.weight#1", False)
    ]
    assert run.use_sites == sites


def test_native_execution_is_the_plain_forward() -> None:
    model = _model()
    native = execute_native(model, _TOKENS)
    with torch.no_grad():
        plain = model(_TOKENS)
    assert (native.dtype, native.device) == ("float64", "cpu")
    assert np.array_equal(native.values, plain.numpy())


def test_executed_output_records_the_tf32_matmul_flag_the_forward_ran_under() -> None:
    model = _model()
    edit = GlobalParameterEdit("embed.weight", _ramp((7, 4)), None)

    def outputs() -> list:
        return [
            execute_native(model, _TOKENS),
            execute_parameter_edits(model, _TOKENS, [edit]).output,
            execute_parameter_cotangents(
                model, _TOKENS, [], [OutputReadout(_EVERY_ROW)], [np.ones((1, 6, 7))],
                ["body.0.weight#0"],
            ).execution.output,
        ]

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        off = outputs()
        torch.backends.cuda.matmul.allow_tf32 = True
        on = outputs()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert [output.tf32_matmul for output in off] == [False, False, False]
    # Positive control: the same executions under TF32 matmul say so.
    assert [output.tf32_matmul for output in on] == [True, True, True]


def test_zero_deltas_reproduce_native_bitwise_and_nonzero_deltas_do_not() -> None:
    model = _model()
    native = execute_native(model, _TOKENS).values
    zero = np.zeros((7, 4))
    zero_factors = FactoredDelta(np.zeros((7, 2)), np.zeros((4, 2)))
    for edit in (
        GlobalParameterEdit("embed.weight", zero, None),
        _use_site_edit(model, "embed.weight", 1, zero, None),
        GlobalParameterEdit("embed.weight", zero_factors, None),
    ):
        assert np.array_equal(execute_parameter_edits(model, _TOKENS, [edit]).output.values, native)
    moved = _ramp((7, 4))
    for edit in (
        GlobalParameterEdit("embed.weight", moved, None),
        _use_site_edit(model, "embed.weight", 1, moved, None),
    ):
        assert not np.array_equal(
            execute_parameter_edits(model, _TOKENS, [edit]).output.values, native
        )


def test_global_edit_moves_every_tied_use_like_functional_call() -> None:
    model = _model()
    delta = _ramp((7, 4))
    run = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("embed.weight", delta, None)]
    )
    edited = _original(model, "embed.weight") + torch.from_numpy(delta)
    with torch.no_grad():
        reference = torch.func.functional_call(model, {"embed.weight": edited}, (_TOKENS,))
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["embed.weight#0", "embed.weight#1"]
    assert run.use_sites == discover_parameter_use_sites(model, _TOKENS)


def test_a_factored_delta_is_read_as_w_plus_left_right_transpose() -> None:
    model = _model()
    left = _ramp((6, 2)) + 0.5
    right = _ramp((4, 2)) - 0.5
    run = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("body.0.weight", FactoredDelta(left, right), None)]
    )
    edited = _original(model, "body.0.weight") + torch.from_numpy(left) @ torch.from_numpy(right).T
    with torch.no_grad():
        reference = torch.func.functional_call(model, {"body.0.weight": edited}, (_TOKENS,))
    assert np.array_equal(run.output.values, reference.numpy())
    # Positive control: swapping the factor roles is a different edit, and it refuses on shape.
    with pytest.raises(ValueError, match=r"needs \(d_out, rank\) and \(d_in, rank\)"):
        execute_parameter_edits(
            model, _TOKENS, [GlobalParameterEdit("body.0.weight", FactoredDelta(right, left), None)]
        )


def test_use_site_edit_of_the_tied_tensor_moves_only_the_head() -> None:
    model = _model()
    original = _original(model, "embed.weight")
    delta = _ramp((7, 4))
    use_site = execute_parameter_edits(
        model, _TOKENS, [_use_site_edit(model, "embed.weight", 1, delta, None)]
    )
    global_edit = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("embed.weight", delta, None)]
    )
    with torch.no_grad():
        x = F.embedding(_TOKENS, original)
        x = x + model.body(x)
        x = x + model.body(x)
        reference = F.linear(model.scale(x), original + torch.from_numpy(delta))
    assert np.array_equal(use_site.output.values, reference.numpy())
    assert not np.array_equal(use_site.output.values, global_edit.output.values)
    assert [site.use_site_id for site in use_site.substituted] == ["embed.weight#1"]
    assert use_site.use_sites == discover_parameter_use_sites(model, _TOKENS)


def test_use_site_edit_of_the_shared_body_moves_only_its_second_call() -> None:
    model = _model()
    delta = _ramp((6, 4))
    edited = _original(model, "body.0.weight") + torch.from_numpy(delta)
    run = execute_parameter_edits(
        model, _TOKENS, [_use_site_edit(model, "body.0.weight", 1, delta, None)]
    )
    with torch.no_grad():
        x = model.embed(_TOKENS)
        x = x + model.body(x)
        x = x + model.body[2](model.body[1](F.linear(x, edited, model.body[0].bias)))
        reference = model.head(model.scale(x))
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["body.0.weight#1"]


def test_a_use_site_edit_refuses_when_its_names_are_not_the_read_at_its_ordinal() -> None:
    model = _model()
    delta = _ramp((7, 4))

    def edit(model: torch.nn.Module, ordinal: int, module: str, op: str) -> Any:
        return execute_parameter_edits(
            model, _TOKENS, [UseSiteParameterEdit("embed.weight", ordinal, delta, None, module, op)]
        )

    # Control: named as discovery names the head read, the edit executes there.
    run = edit(model, 1, "head", _LINEAR)
    assert [site.use_site_id for site in run.substituted] == ["embed.weight#1"]
    # Another op, or another module, at that ordinal is another read.
    with pytest.raises(ValueError, match="LabelMismatch"):
        edit(model, 1, "head", resolve_name(F.embedding))
    with pytest.raises(ValueError, match="LabelMismatch"):
        edit(model, 1, "embed", _LINEAR)
    # An ordinal off by one reaches the embedding lookup, which the head's names do not name.
    with pytest.raises(ValueError, match="LabelMismatch"):
        edit(model, 0, "head", _LINEAR)
    # The root module's own forward is named "", and the transposed tie reads there.
    transposed = _seeded(_TransposedTie())
    tie = edit(transposed, 1, "", resolve_name(torch.Tensor.t))
    assert [site.use_site_id for site in tie.substituted] == ["embed.weight#1"]
    with pytest.raises(ValueError, match="LabelMismatch"):
        edit(transposed, 1, "embed", resolve_name(torch.Tensor.t))
    # The cotangent runner checks the same names.
    cotangent = _ramp(execute_native(model, _TOKENS).values.shape) - 0.1
    with pytest.raises(ValueError, match="LabelMismatch"):
        execute_parameter_cotangents(
            model,
            _TOKENS,
            [UseSiteParameterEdit("body.0.weight", 0, _ramp((6, 4)), None, "body.2", _LINEAR)],
            [OutputReadout(_EVERY_ROW)],
            [cotangent],
            ["body.0.weight#0"],
        )


def test_declared_positions_read_the_edit_only_at_those_rows() -> None:
    model = _model()
    delta = _ramp((6, 4))
    positions = (1, 4)
    run = execute_parameter_edits(
        model,
        _TOKENS,
        [GlobalParameterEdit("body.0.weight", delta, positions)],
        leading_shape=_LEADING,
    )
    original = _original(model, "body.0.weight")
    edited = original + torch.from_numpy(delta)
    bias = model.body[0].bias

    def first_linear(x: torch.Tensor) -> torch.Tensor:
        return _splice(F.linear(x, original, bias), F.linear(x, edited, bias), positions)

    with torch.no_grad():
        x = model.embed(_TOKENS)
        x = x + model.body[2](model.body[1](first_linear(x)))
        x = x + model.body[2](model.body[1](first_linear(x)))
        reference = model.head(model.scale(x))
    assert np.array_equal(run.output.values, reference.numpy())
    every = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("body.0.weight", delta, None)]
    )
    assert not np.array_equal(run.output.values, every.output.values)
    # Control: declaring every row is the every-position edit, bitwise.
    all_rows = execute_parameter_edits(
        model,
        _TOKENS,
        [GlobalParameterEdit("body.0.weight", delta, _EVERY_ROW)],
        leading_shape=_LEADING,
    )
    assert np.array_equal(all_rows.output.values, every.output.values)


def test_declared_positions_at_the_embedding_edit_a_suffix_of_rows() -> None:
    model = _model()
    delta = _ramp((7, 4))
    suffix = (3, 4, 5)
    run = execute_parameter_edits(
        model,
        _TOKENS,
        [_use_site_edit(model, "embed.weight", 0, delta, suffix)],
        leading_shape=_LEADING,
    )
    original = _original(model, "embed.weight")
    edited = original + torch.from_numpy(delta)
    with torch.no_grad():
        x = _splice(F.embedding(_TOKENS, original), F.embedding(_TOKENS, edited), suffix)
        x = x + model.body(x)
        x = x + model.body(x)
        reference = F.linear(model.scale(x), original)
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["embed.weight#0"]


def test_both_runs_of_a_declared_use_see_the_current_intervened_input() -> None:
    model = _model()
    embed_delta = _ramp((7, 4))
    body_delta = _ramp((6, 4))
    positions = (1, 4)
    run = execute_parameter_edits(
        model,
        _TOKENS,
        [
            GlobalParameterEdit("embed.weight", embed_delta, None),
            _use_site_edit(model, "body.0.weight", 0, body_delta, positions),
        ],
        leading_shape=_LEADING,
    )
    embed = _original(model, "embed.weight") + torch.from_numpy(embed_delta)
    weight = _original(model, "body.0.weight")
    edited = weight + torch.from_numpy(body_delta)
    bias = model.body[0].bias
    with torch.no_grad():
        x = F.embedding(_TOKENS, embed)
        h = _splice(F.linear(x, weight, bias), F.linear(x, edited, bias), positions)
        x = x + model.body[2](model.body[1](h))
        x = x + model.body(x)
        reference = F.linear(model.scale(x), embed)
    assert np.array_equal(run.output.values, reference.numpy())


def test_edited_execution_never_writes_into_the_module() -> None:
    model = _model()

    def raw_values() -> dict[str, bytes]:
        return {
            row.tensor_id: parameter_values(model, row.tensor_id).tobytes()
            for row in parameter_registry(model)
        }

    before = raw_values()
    execute_parameter_edits(
        model,
        _TOKENS,
        [
            GlobalParameterEdit("embed.weight", _ramp((7, 4)), None),
            _use_site_edit(
                model, "body.2.weight", 0, FactoredDelta(_ramp((4, 1)), _ramp((6, 1))), (2,)
            ),
        ],
        leading_shape=_LEADING,
    )
    assert raw_values() == before
    # Positive control: a real write into one element is visible to the same comparison.
    with torch.no_grad():
        model.body[2].weight[0, 0] += 1.0
    assert raw_values() != before


def test_edits_refuse_what_they_cannot_execute_faithfully() -> None:
    model = _model()
    delta = _ramp((6, 4))

    def run(*edits: GlobalParameterEdit | UseSiteParameterEdit) -> None:
        execute_parameter_edits(model, _TOKENS, edits)

    with pytest.raises(ValueError, match="at least one edit"):
        run()
    with pytest.raises(TypeError, match="must be GlobalParameterEdit or UseSiteParameterEdit"):
        execute_parameter_edits(model, _TOKENS, ["body.0.weight"])
    # An alias is not an id: the tied tensor is edited through ``embed.weight``.
    with pytest.raises(ValueError, match="unknown tensor id"):
        run(GlobalParameterEdit("head.weight", _ramp((7, 4)), None))
    with pytest.raises(ValueError, match="dense delta"):
        run(GlobalParameterEdit("body.0.weight", delta[:, :3], None))
    with pytest.raises(ValueError, match=r"needs \(d_out, rank\) and \(d_in, rank\)"):
        run(GlobalParameterEdit("body.0.bias", FactoredDelta(_ramp((6, 1)), _ramp((1, 1))), None))
    nonfinite = delta.copy()
    nonfinite[1, 1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        run(GlobalParameterEdit("body.0.weight", nonfinite, None))
    with pytest.raises(ValueError, match="more than one global edit"):
        run(
            GlobalParameterEdit("body.0.weight", delta, None),
            GlobalParameterEdit("body.0.weight", delta, None),
        )
    with pytest.raises(ValueError, match="more than one edit at use site"):
        run(
            _use_site_edit(model, "body.0.weight", 0, delta, None),
            _use_site_edit(model, "body.0.weight", 0, delta, None),
        )
    with pytest.raises(ValueError, match="both a global edit and a use-site edit"):
        run(
            GlobalParameterEdit("body.0.weight", delta, None),
            _use_site_edit(model, "body.0.weight", 0, delta, None),
        )
    with pytest.raises(ValueError, match="never reached"):
        run(UseSiteParameterEdit("body.0.weight", 2, delta, None, "body.0", _LINEAR))
    with pytest.raises(ValueError, match="non-negative integer"):
        UseSiteParameterEdit("body.0.weight", -1, delta, None, "body.0", _LINEAR)
    with pytest.raises(TypeError, match="floating-point tensor"):
        execute_native(_TupleOutput(), torch.ones(1, 2, dtype=torch.float64))


def test_a_global_edit_of_a_tensor_the_forward_never_reads_refuses() -> None:
    model = _WithUnusedParameter()
    x = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"global edits of \['unused'\] were never read"):
        execute_parameter_edits(model, x, [GlobalParameterEdit("unused", np.zeros(3), None)])
    # Control: a global edit of the tensor the forward does read executes.
    run = execute_parameter_edits(model, x, [GlobalParameterEdit("used.weight", np.zeros((2, 2)), None)])
    assert [site.use_site_id for site in run.substituted] == ["used.weight#0"]


def test_an_edit_the_tensor_format_cannot_represent_refuses_instead_of_rounding() -> None:
    single = _seeded(torch.nn.Linear(2, 2, dtype=torch.float32))
    x = torch.ones(1, 2)
    # A delta below the float32 spacing of every weight would vanish on narrowing.
    with pytest.raises(ValueError, match="not exactly representable in the tensor's float32 format"):
        execute_parameter_edits(single, x, [GlobalParameterEdit("weight", np.full((2, 2), 1e-12), None)])
    with pytest.raises(ValueError, match="not exactly representable in the tensor's float32 format"):
        execute_parameter_edits(single, x, [GlobalParameterEdit("weight", np.full((2, 2), 1e39), None)])
    # Control: a delta the float32 format carries exactly executes.
    run = execute_parameter_edits(single, x, [GlobalParameterEdit("weight", np.zeros((2, 2)), None)])
    assert [site.use_site_id for site in run.substituted] == ["weight#0"]
    # The same tiny delta on a float64 tensor is representable, so it executes.
    double = _seeded(torch.nn.Linear(2, 2, dtype=torch.float64))
    moved = execute_parameter_edits(
        double, x.to(torch.float64), [GlobalParameterEdit("weight", np.full((2, 2), 1e-12), None)]
    )
    assert not np.array_equal(moved.output.values, execute_native(double, x.to(torch.float64)).values)


def test_declared_positions_refuse_where_rows_are_not_positions() -> None:
    model = _model()
    delta = _ramp((6, 4))
    for bad in ((), (2, 1), (1, 1), (-1,)):
        with pytest.raises(ValueError, match="strictly increasing non-negative"):
            GlobalParameterEdit("body.0.weight", delta, bad)
    with pytest.raises(ValueError, match="need the unit's leading_shape"):
        execute_parameter_edits(model, _TOKENS, [GlobalParameterEdit("body.0.weight", delta, (1,))])
    with pytest.raises(ValueError, match="leading_shape must be"):
        execute_parameter_edits(
            model, _TOKENS, [GlobalParameterEdit("body.0.weight", delta, (1,))], leading_shape=(0, 6)
        )
    with pytest.raises(ValueError, match="outside the unit length"):
        execute_parameter_edits(
            model,
            _TOKENS,
            [GlobalParameterEdit("body.0.weight", delta, (6,))],
            leading_shape=_LEADING,
        )
    # A declared leading shape the op's input and result do not carry.
    with pytest.raises(ValueError, match="does not keep the declared leading axes"):
        execute_parameter_edits(
            model,
            _TOKENS,
            [_use_site_edit(model, "embed.weight", 0, _ramp((7, 4)), (0,))],
            leading_shape=(1, 5),
        )
    # ``W.t()`` returns the weight's transpose, which has no sequence axis.
    transposed = _seeded(_TransposedTie())
    with pytest.raises(ValueError, match="does not keep the declared leading axes"):
        execute_parameter_edits(
            transposed,
            _TOKENS,
            [_use_site_edit(transposed, "embed.weight", 1, _ramp((7, 4)), (0,))],
            leading_shape=_LEADING,
        )
    # One call reading two edited tensors with different declared rows.
    with pytest.raises(ValueError, match="different declared positions"):
        execute_parameter_edits(
            model,
            _TOKENS,
            [
                GlobalParameterEdit("body.0.weight", delta, (1,)),
                GlobalParameterEdit("body.0.bias", _ramp((6,)), (2,)),
            ],
            leading_shape=_LEADING,
        )


def test_use_cotangents_are_the_rows_a_linear_use_multiplied_and_wrote() -> None:
    model = _model()
    native = execute_native(model, _TOKENS).values
    cotangent = _ramp(native.shape) - 0.1
    run = execute_parameter_cotangents(
        model,
        _TOKENS,
        [],
        [OutputReadout(_EVERY_ROW)],
        [cotangent],
        ["body.0.weight#0", "embed.weight#1"],
    )
    assert np.array_equal(run.execution.output.values, native)
    assert np.array_equal(run.readouts[0], native)
    assert run.execution.use_sites == discover_parameter_use_sites(model, _TOKENS)
    assert [(use.site.use_site_id, use.site.transposed) for use in run.uses] == [
        ("body.0.weight#0", False),
        ("embed.weight#1", False),
    ]
    assert all(param.grad is None for param in model.parameters())

    first_weight = _original(model, "body.0.weight").requires_grad_(True)
    head_weight = _original(model, "embed.weight").requires_grad_(True)
    first_in = model.embed(_TOKENS)
    first_out = F.linear(first_in, first_weight, model.body[0].bias)
    x = first_in + model.body[2](model.body[1](first_out))
    x = x + model.body(x)
    head_in = model.scale(x)
    head_out = F.linear(head_in, head_weight)
    objective = (head_out * torch.from_numpy(cotangent)).sum()
    g_first, g_head, dense_first = torch.autograd.grad(
        objective, [first_out, head_out, first_weight]
    )
    first, head = run.uses
    assert np.array_equal(first.outputs, g_first.reshape(-1, 6).numpy())
    assert np.array_equal(first.inputs, first_in.detach().reshape(-1, 4).numpy())
    assert np.array_equal(head.outputs, g_head.reshape(-1, 7).numpy())
    assert np.array_equal(head.inputs, head_in.detach().reshape(-1, 4).numpy())

    bound = _reconstruction_bound(first)
    assert np.all(np.abs(first.outputs.T @ first.inputs - dense_first.numpy()) <= bound)
    # Positive control: pairing the written rows with the read rows in reverse order breaks it.
    reversed_rows = first.inputs[::-1]
    assert not np.all(np.abs(first.outputs.T @ reversed_rows - dense_first.numpy()) <= bound)


@pytest.mark.parametrize(
    ("fixture", "use_site_id", "transposed"),
    [
        (_TransposedTie, "embed.weight#1", False),
        (_TransposedLinear, "weight#0", True),
        (_PlainMatmul, "weight#0", True),
    ],
)
def test_use_site_orientation_follows_how_the_op_multiplies_the_tensor(
    fixture: type[torch.nn.Module], use_site_id: str, transposed: bool
) -> None:
    model = _seeded(fixture())
    discovered = {site.use_site_id: site for site in discover_parameter_use_sites(model, _TOKENS)}
    assert discovered[use_site_id].linear is True
    assert discovered[use_site_id].transposed is transposed
    native = execute_native(model, _TOKENS).values
    cotangent = _ramp(native.shape) - 0.1
    run = execute_parameter_cotangents(
        model, _TOKENS, [], [OutputReadout(_EVERY_ROW)], [cotangent], [use_site_id]
    )
    (use,) = run.uses
    assert use.site == discovered[use_site_id]

    # The dense gradient of the stored tensor through the multiplied use only.
    tensor_id = use_site_id.rpartition("#")[0]
    weight = dict(model.named_parameters())[tensor_id]
    # The tied fixture also reads the tensor through its lookup; hold that read fixed.
    embedded = model.embed(_TOKENS).detach()
    multiplied = embedded @ weight.t() if tensor_id == "embed.weight" else model(_TOKENS)
    (dense,) = torch.autograd.grad((multiplied * torch.from_numpy(cotangent)).sum(), [weight])
    # ``Σ_r g_r h_rᵀ`` is the gradient of ``A``; ``A = Wᵀ`` when the use is transposed.
    rows_product = use.outputs.T @ use.inputs
    bound = _reconstruction_bound(use)
    stored = rows_product.T if transposed else rows_product
    stored_bound = bound.T if transposed else bound
    assert stored.shape == tuple(dense.shape)
    assert np.all(np.abs(stored - dense.numpy()) <= stored_bound)
    # Positive control: pairing the written rows with the read rows in reverse order breaks it.
    reversed_product = use.outputs.T @ use.inputs[::-1]
    reversed_stored = reversed_product.T if transposed else reversed_product
    assert not np.all(np.abs(reversed_stored - dense.numpy()) <= stored_bound)


def test_use_cotangent_rows_at_declared_positions_come_from_the_edited_run() -> None:
    model = _model()
    delta = _ramp((6, 4))
    positions = (1, 4)
    edits = [_use_site_edit(model, "body.0.weight", 0, delta, positions)]
    executed = execute_parameter_edits(model, _TOKENS, edits, leading_shape=_LEADING)
    cotangent = _ramp(executed.output.values.shape) - 0.1
    run = execute_parameter_cotangents(
        model,
        _TOKENS,
        edits,
        [OutputReadout(_EVERY_ROW)],
        [cotangent],
        ["body.0.weight#0"],
        leading_shape=_LEADING,
    )
    assert np.array_equal(run.execution.output.values, executed.output.values)
    assert run.execution.substituted == executed.substituted
    (use,) = run.uses
    assert (use.outputs.shape, use.inputs.shape, use.site.transposed) == ((2, 6), (2, 4), False)

    original = _original(model, "body.0.weight")
    edited_weight = (original + torch.from_numpy(delta)).requires_grad_(True)
    bias = model.body[0].bias
    first_in = model.embed(_TOKENS)
    moved = F.linear(first_in, edited_weight, bias)
    h = _splice(F.linear(first_in, original, bias), moved, positions)
    x = first_in + model.body[2](model.body[1](h))
    x = x + model.body(x)
    objective = (model.head(model.scale(x)) * torch.from_numpy(cotangent)).sum()
    (g_moved,) = torch.autograd.grad(objective, [moved])
    index = torch.as_tensor(positions)
    assert np.array_equal(use.outputs, g_moved[:, index].reshape(-1, 6).numpy())
    assert np.array_equal(use.inputs, first_in.detach()[:, index].reshape(-1, 4).numpy())
    # Positive control: the every-position edit returns every row.
    every = execute_parameter_cotangents(
        model,
        _TOKENS,
        [_use_site_edit(model, "body.0.weight", 0, delta, None)],
        [OutputReadout(_EVERY_ROW)],
        [cotangent],
        ["body.0.weight#0"],
    )
    assert (every.uses[0].outputs.shape, every.uses[0].inputs.shape) == ((6, 6), (6, 4))


def test_readout_cotangents_enter_at_their_declared_blocks() -> None:
    model = _model()
    readouts = [
        UseSiteOutputReadout("body.2.weight#1"),
        UseSiteInputReadout("body.0.weight#1"),
        OutputReadout((0, 5)),
    ]
    blocks = [_ramp((1, 6, 4)) - 0.2, _ramp((1, 6, 4)) - 0.3, _ramp((1, 2, 7)) - 0.1]
    run = execute_parameter_cotangents(model, _TOKENS, [], readouts, blocks, ["body.0.weight#0"])

    first_weight = _original(model, "body.0.weight").requires_grad_(True)
    bias = model.body[0].bias
    first_in = model.embed(_TOKENS)
    first_out = F.linear(first_in, first_weight, bias)
    middle = first_in + model.body[2](model.body[1](first_out))
    second_out = model.body[2](model.body[1](model.body[0](middle)))
    logits = model.head(model.scale(middle + second_out))
    index = torch.as_tensor((0, 5))
    objective = (
        (second_out * torch.from_numpy(blocks[0])).sum()
        + (middle * torch.from_numpy(blocks[1])).sum()
        + (logits[:, index] * torch.from_numpy(blocks[2])).sum()
    )
    (g_first,) = torch.autograd.grad(objective, [first_out])
    (use,) = run.uses
    assert np.array_equal(use.outputs, g_first.reshape(-1, 6).numpy())
    assert np.array_equal(use.inputs, first_in.detach().reshape(-1, 4).numpy())
    assert np.array_equal(run.readouts[0], second_out.detach().numpy())
    assert np.array_equal(run.readouts[1], middle.detach().numpy())
    assert np.array_equal(run.readouts[2], logits.detach()[:, index].numpy())


def test_use_cotangents_refuse_what_they_cannot_factor_or_pair() -> None:
    model = _model()
    good = np.ones(execute_native(model, _TOKENS).values.shape)

    def run(
        readouts: list = [OutputReadout(_EVERY_ROW)],
        blocks: tuple = (good,),
        sites: tuple = ("body.0.weight#0",),
        target: torch.nn.Module = model,
    ) -> None:
        execute_parameter_cotangents(target, _TOKENS, [], readouts, blocks, sites)

    with pytest.raises(ValueError, match="at least one readout"):
        run(readouts=[], blocks=())
    with pytest.raises(ValueError, match="2 cotangent blocks for 1 readouts"):
        run(blocks=(good, good))
    with pytest.raises(ValueError, match="at least one use site"):
        run(sites=())
    # An alias is not an id here either.
    with pytest.raises(ValueError, match="unknown tensor id"):
        run(sites=("head.weight#1",))
    for malformed in ("body.0.weight", "body.0.weight#", "#0", "body.0.weight#01", "body.0.weight#-1"):
        with pytest.raises(ValueError, match="a use site id is"):
            run(sites=(malformed,))
    with pytest.raises(ValueError, match="use sites must be distinct"):
        run(sites=("body.0.weight#0", "body.0.weight#0"))
    with pytest.raises(TypeError, match="must be OutputReadout, UseSiteInputReadout or"):
        run(readouts=["logits"])
    with pytest.raises(ValueError, match="an output readout declares its positions"):
        OutputReadout(None)
    with pytest.raises(ValueError, match="its readout block has shape"):
        run(blocks=(good[..., :3],))
    nonfinite = good.copy()
    nonfinite.flat[0] = np.nan
    with pytest.raises(ValueError, match="readout cotangents must be finite"):
        run(blocks=(nonfinite,))
    with pytest.raises(ValueError, match="were never multiplied"):
        run(sites=("body.0.weight#2",))
    with pytest.raises(ValueError, match="readout use site body.2.weight#5 was never reached"):
        run(readouts=[UseSiteOutputReadout("body.2.weight#5")], blocks=(np.ones((1, 6, 4)),))
    with pytest.raises(ValueError, match="lie outside axis 1"):
        run(readouts=[OutputReadout((6,))], blocks=(np.ones((1, 1, 7)),))
    # Reads with no factored rows: an embedding lookup and a bias.
    with pytest.raises(ValueError, match="has factored cotangent rows"):
        run(sites=("embed.weight#0",))
    with pytest.raises(ValueError, match="has factored cotangent rows"):
        run(sites=("body.0.bias#0",))
    # A transpose view consumed by a copy is not multiplied by the use: discovery
    # reports a stored read, and a cotangent request refuses.
    copied = _seeded(_TransposeCopied())
    copied_view = {
        site.use_site_id: site for site in discover_parameter_use_sites(copied, _TOKENS)
    }["embed.weight#1"]
    assert (copied_view.linear, copied_view.transposed) == (False, False)
    with pytest.raises(ValueError, match="reads a transpose view consumed by"):
        run(
            target=copied,
            sites=("embed.weight#1",),
            blocks=(np.ones(execute_native(copied, _TOKENS).values.shape),),
        )
