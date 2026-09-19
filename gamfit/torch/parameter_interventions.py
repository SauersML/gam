"""Execute parameter edits on a torch module for manifold parameter
decomposition (#2951).

This is the torch end of MPD's native lift
(``crates/gam-sae/src/parameter_decomposition/``). All math that decides an
experiment stays in Rust: the lift declares each edit's delta, scope and
positions and computes the teacher fingerprint from what this module exports.
This module only

* exports the parameter registry: one tensor id per ``Parameter`` object (its
  first name in ``named_parameters()``), its aliases (every other qualified
  name bound to that same object, e.g. a tied embedding and output head), its
  shape and source dtype, with the values fetched one tensor at a time as an
  exact float64 widening;
* discovers use sites: within one forward, every torch function call that
  reads a registered parameter and returns a tensor, keyed
  ``{tensor_id}#{ordinal}`` in execution order, with whether a linear map
  multiplies it and, if so, that map's orientation;
* executes a forward in which a tensor reads ``W + ΔW`` at every use (a global
  edit, which moves every tied use) or at exactly one use site (a use-specific
  edit), at every position or only at declared positions, and returns the
  executed output with the forward's use sites;
* differentiates that executed forward for the teacher, by torch autograd:
  for each requested linear use, the rows it multiplied and the cotangent rows
  at what it wrote, given cotangents at declared readouts.

``ΔW`` arrives in the tensor's stored orientation, factored as
``left · rightᵀ`` (the canonical record) or dense for small blocks; applying it
at the use is the only arithmetic here. The native path is ``model(inputs)``
with nothing installed, so the all-on setting runs the original tensors on
their original path. Edited forwards substitute arguments at the
torch-function boundary and never write into a parameter, so the module is
unchanged afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.overrides import TorchFunctionMode, resolve_name

from .._binding import rust_module

__all__ = [
    "EditedCotangents",
    "EditedExecution",
    "ExecutedOutput",
    "FactoredDelta",
    "GlobalParameterEdit",
    "OutputReadout",
    "ParameterTensor",
    "ParameterUseSite",
    "UseCotangent",
    "UseSiteInputReadout",
    "UseSiteOutputReadout",
    "UseSiteParameterEdit",
    "discover_parameter_use_sites",
    "execute_native",
    "execute_parameter_cotangents",
    "execute_parameter_edits",
    "parameter_registry",
    "parameter_values",
]

_LINEAR = resolve_name(torch.nn.functional.linear)
_MATMUL = frozenset(
    resolve_name(function)
    for function in (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__)
)
_Planned = tuple[torch.Tensor, "tuple[int, ...] | None"]


def _is_index(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_ordinal(ordinal: Any) -> None:
    if not _is_index(ordinal) or ordinal < 0:
        raise ValueError(f"ordinal must be a non-negative integer; got {ordinal!r}")


def _parse_use_site_id(use_site_id: Any) -> tuple[str, int]:
    """``"{tensor_id}#{ordinal}"`` with a canonical decimal ordinal."""
    if not isinstance(use_site_id, str):
        raise TypeError(f"a use site id is a string; got {type(use_site_id).__name__}")
    tensor_id, separator, ordinal = use_site_id.rpartition("#")
    if (
        not separator
        or not tensor_id
        or not (ordinal.isascii() and ordinal.isdigit())
        or str(int(ordinal)) != ordinal
    ):
        raise ValueError(f"a use site id is '{{tensor_id}}#{{ordinal}}'; got {use_site_id!r}")
    return tensor_id, int(ordinal)


def _declared_positions(positions: Any) -> tuple[int, ...] | None:
    if positions is None:
        return None
    items = tuple(positions)
    if (
        not items
        or not all(_is_index(item) and item >= 0 for item in items)
        or any(later <= earlier for earlier, later in zip(items, items[1:]))
    ):
        raise ValueError(
            "positions must be None (every position) or strictly increasing non-negative "
            f"integers; got {positions!r}"
        )
    return tuple(int(item) for item in items)


@dataclass(frozen=True)
class ParameterTensor:
    """One registered parameter tensor.

    Attributes
    ----------
    tensor_id
        The first qualified name ``named_parameters()`` yields for this
        ``Parameter`` object.
    aliases
        Every later qualified name bound to the same object. An alias is
        identity only; a global edit of ``tensor_id`` moves all of them.
    shape
        The tensor's shape.
    dtype
        The source dtype (``"float32"``, ``"bfloat16"``, ...).
    """

    tensor_id: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ParameterUseSite:
    """One read of a registered parameter within a single forward.

    ``tensor_id`` and ``ordinal`` are the key, and a read always reads
    ``tensor_id`` whichever alias the code used; the other fields describe the
    read.

    Attributes
    ----------
    tensor_id
        The registry id of the tensor read.
    ordinal
        ``k`` for the ``k``-th (from 0) torch function call in the forward that
        read this tensor and returned a tensor. Calls that return no tensor,
        such as dtype or shape queries, are not uses.
    linear
        True when ``F.linear`` or matmul multiplies this read, directly or through
        its transpose view (followed to the op that consumes the view). False for
        a read that applies no linear map: an embedding, a bias, an element-wise
        use, or a view consumed by anything else.
    transposed
        Meaningful only when ``linear``: the orientation of the map that
        multiplies this read. Writing the consuming op as ``y = x · Aᵀ``, True
        means ``A = Wᵀ`` (``x @ W``, ``F.linear(x, W.t())``) and False means
        ``A = W`` (``F.linear(x, W)``, ``x @ W.t()``). A transpose view such as
        ``W.t()`` is how a read is implemented, not the map's orientation, so the
        op consuming the view decides. A read with ``linear`` False reports False.
    op
        :func:`torch.overrides.resolve_name` of the function that read it;
        informative only.
    module
        Qualified name of the innermost executing module (``""`` for the root
        module).
    """

    tensor_id: str
    ordinal: int
    linear: bool
    transposed: bool
    op: str
    module: str

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal)

    @property
    def use_site_id(self) -> str:
        """``{tensor_id}#{ordinal}``."""
        return f"{self.tensor_id}#{self.ordinal}"


@dataclass(frozen=True)
class FactoredDelta:
    """``ΔW = left · rightᵀ`` in the tensor's stored orientation: ``left`` is
    ``(d_out, rank)`` and ``right`` is ``(d_in, rank)`` for a ``(d_out, d_in)``
    tensor."""

    left: Any
    right: Any


@dataclass(frozen=True)
class GlobalParameterEdit:
    """Every use of ``tensor_id`` reads ``W + delta``, so every tied name sees
    the edit.

    ``delta`` is a :class:`FactoredDelta` or a dense array of the tensor's
    shape. ``positions`` is ``None`` (every position) or strictly increasing
    sequence positions: at each use only the result rows at those positions
    read ``W + delta``, and the rest read ``W``.
    """

    tensor_id: str
    delta: Any
    positions: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _declared_positions(self.positions))


@dataclass(frozen=True)
class UseSiteParameterEdit:
    """Use ``{tensor_id}#{ordinal}`` reads ``W + delta`` (at every position, or
    only at declared ``positions``); every other use reads ``W``.

    ``read_module`` and ``read_op`` name that read as discovery reported it:
    the :attr:`ParameterUseSite.module` (``""`` for the root module) and
    :attr:`ParameterUseSite.op` of ``{tensor_id}#{ordinal}``. Rust checks them
    against the read the executed forward made at that ordinal, so an ordinal
    that addresses another read refuses instead of editing it.
    """

    tensor_id: str
    ordinal: int
    delta: Any
    positions: Any
    read_module: str
    read_op: str

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal)
        object.__setattr__(self, "positions", _declared_positions(self.positions))


@dataclass(frozen=True)
class OutputReadout:
    """The model output's rows at the declared ``positions`` along axis 1."""

    positions: Any

    def __post_init__(self) -> None:
        positions = _declared_positions(self.positions)
        if positions is None:
            raise ValueError("an output readout declares its positions")
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class UseSiteInputReadout:
    """The rows use ``use_site_id`` multiplied: the first non-parameter tensor
    argument of its op, or of the op consuming its transpose view."""

    use_site_id: str

    def __post_init__(self) -> None:
        _parse_use_site_id(self.use_site_id)


@dataclass(frozen=True)
class UseSiteOutputReadout:
    """What use ``use_site_id``'s op wrote (for a transpose view, what the
    consuming op wrote), as the forward carried it."""

    use_site_id: str

    def __post_init__(self) -> None:
        _parse_use_site_id(self.use_site_id)


@dataclass(frozen=True)
class ExecutedOutput:
    """The model's output, promoted exactly to float64, with the execution
    facts ``receipts::ExternalExecution`` records: the source dtype, the device,
    and whether TF32 matrix multiplication was enabled when the forward ran."""

    values: np.ndarray
    dtype: str
    device: str
    tf32_matmul: bool


@dataclass(frozen=True)
class EditedExecution:
    """An edited forward: its output, every use observed in execution order
    (the forward's path, to compare against discovery), and the uses that
    received edited values."""

    output: ExecutedOutput
    use_sites: tuple[ParameterUseSite, ...]
    substituted: tuple[ParameterUseSite, ...]


@dataclass(frozen=True)
class UseCotangent:
    """One linear use's cotangent in factored rows.

    ``outputs`` ``(rows, width written)`` are the cotangent at the rows the
    consuming op wrote and ``inputs`` ``(rows, width read)`` are the rows it
    multiplied. Writing the op as ``y = x · Aᵀ``, ``Σ_r outputs[r] · inputs[r]ᵀ``
    is the gradient of ``A``, oriented by ``site.transposed``; it is never
    formed. With declared positions only the rows at those positions appear.
    """

    site: ParameterUseSite
    outputs: np.ndarray
    inputs: np.ndarray


@dataclass(frozen=True)
class EditedCotangents:
    """An executed forward, its declared readout blocks (float64, in readout
    order), and each requested use's factored cotangent rows (in request
    order). Rows are never summed across uses; tied accumulation belongs to
    the caller."""

    execution: EditedExecution
    readouts: tuple[np.ndarray, ...]
    uses: tuple[UseCotangent, ...]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _float64_copy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float64, copy=True).numpy()


def _registered_parameters(
    model: torch.nn.Module,
) -> dict[str, tuple[torch.nn.Parameter, list[str]]]:
    """``tensor_id -> (parameter, aliases)`` in ``named_parameters`` order."""
    table: dict[str, tuple[torch.nn.Parameter, list[str]]] = {}
    id_of_object: dict[int, str] = {}
    id_of_storage: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in id_of_object:
            table[id_of_object[id(param)]][1].append(name)
            continue
        if param.numel() > 0:
            storage = param.untyped_storage().data_ptr()
            if storage in id_of_storage:
                raise ValueError(
                    f"parameters {id_of_storage[storage]!r} and {name!r} are distinct tensors "
                    "sharing one storage; a tie must be one Parameter object under several names"
                )
            id_of_storage[storage] = name
        id_of_object[id(param)] = name
        table[name] = (param, [])
    return table


def parameter_registry(model: torch.nn.Module) -> tuple[ParameterTensor, ...]:
    """Every distinct parameter tensor of ``model``, with its aliases."""
    rows = []
    for tensor_id, (param, aliases) in _registered_parameters(model).items():
        if not param.is_floating_point():
            raise TypeError(f"parameter {tensor_id!r} has non-floating dtype {param.dtype}")
        rows.append(
            ParameterTensor(
                tensor_id=tensor_id,
                aliases=tuple(aliases),
                shape=tuple(param.shape),
                dtype=_dtype_name(param.dtype),
            )
        )
    return tuple(rows)


def parameter_values(model: torch.nn.Module, tensor_id: str) -> np.ndarray:
    """A float64 copy of one registered tensor. Every float format torch stores
    widens to float64 exactly, and its registry row carries the source dtype."""
    param = _lookup(_registered_parameters(model), tensor_id)
    if not param.is_floating_point():
        raise TypeError(f"parameter {tensor_id!r} has non-floating dtype {param.dtype}")
    return _float64_copy(param)


def _lookup(
    table: dict[str, tuple[torch.nn.Parameter, list[str]]], tensor_id: str
) -> torch.nn.Parameter:
    if tensor_id not in table:
        raise ValueError(f"unknown tensor id {tensor_id!r}; registered ids are {sorted(table)}")
    return table[tensor_id][0]


def _finite_array(tensor_id: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"the delta for {tensor_id!r} must be finite")
    return np.ascontiguousarray(array)


def _edited_tensor(
    table: dict[str, tuple[torch.nn.Parameter, list[str]]], tensor_id: str, delta: Any
) -> torch.Tensor:
    """``W + delta``, formed in float64 and handed over in the tensor's own
    dtype and device only when that format represents it exactly."""
    param = _lookup(table, tensor_id)
    shape = tuple(param.shape)

    def widened(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=param.device)

    with torch.no_grad():
        base = param.detach().to(dtype=torch.float64)
        if isinstance(delta, FactoredDelta):
            left = _finite_array(tensor_id, delta.left)
            right = _finite_array(tensor_id, delta.right)
            if (
                len(shape) != 2
                or left.ndim != 2
                or right.ndim != 2
                or left.shape[0] != shape[0]
                or right.shape[0] != shape[1]
                or left.shape[1] != right.shape[1]
            ):
                raise ValueError(
                    f"the factored delta for {tensor_id!r} has left {left.shape} and right "
                    f"{right.shape}; a tensor of shape {shape} needs (d_out, rank) and (d_in, rank)"
                )
            edited = base + widened(left) @ widened(right).T
        else:
            dense = _finite_array(tensor_id, delta)
            if dense.shape != shape:
                raise ValueError(
                    f"the dense delta for {tensor_id!r} has shape {dense.shape}; "
                    f"the tensor has shape {shape}"
                )
            edited = base + widened(dense)
        if not bool(torch.isfinite(edited).all()):
            raise ValueError(f"the edited {tensor_id!r} is not finite")
        narrowed = edited.to(dtype=param.dtype)
        if not torch.equal(narrowed.to(dtype=torch.float64), edited):
            raise ValueError(
                f"the edited {tensor_id!r} is not exactly representable in the tensor's "
                f"{_dtype_name(param.dtype)} format, so narrowing would round the delta; "
                "execute the model in float64"
            )
    return narrowed


def _plan_edits(
    table: dict[str, tuple[torch.nn.Parameter, list[str]]],
    edits: tuple[Any, ...],
    leading_shape: Any,
) -> tuple[dict[str, _Planned], dict[tuple[str, int], _Planned], tuple[int, int] | None]:
    """Validate the edits and build each planned tensor once."""
    if leading_shape is not None:
        leading_shape = tuple(leading_shape)
        if len(leading_shape) != 2 or not all(_is_index(size) and size > 0 for size in leading_shape):
            raise ValueError(
                f"leading_shape must be the unit's (batch, seq) as positive integers; got {leading_shape!r}"
            )
        leading_shape = (int(leading_shape[0]), int(leading_shape[1]))
    global_values: dict[str, _Planned] = {}
    site_values: dict[tuple[str, int], _Planned] = {}
    for edit in edits:
        if not isinstance(edit, (GlobalParameterEdit, UseSiteParameterEdit)):
            raise TypeError(
                f"edits must be GlobalParameterEdit or UseSiteParameterEdit; got {type(edit).__name__}"
            )
        if edit.positions is not None:
            if leading_shape is None:
                raise ValueError(
                    f"an edit of {edit.tensor_id!r} declares positions, which need the unit's "
                    "leading_shape (batch, seq)"
                )
            if edit.positions[-1] >= leading_shape[1]:
                raise ValueError(
                    f"position {edit.positions[-1]} of an edit of {edit.tensor_id!r} lies outside "
                    f"the unit length {leading_shape[1]}"
                )
        planned = (_edited_tensor(table, edit.tensor_id, edit.delta), edit.positions)
        if isinstance(edit, GlobalParameterEdit):
            if edit.tensor_id in global_values:
                raise ValueError(f"more than one global edit of {edit.tensor_id!r}")
            global_values[edit.tensor_id] = planned
        else:
            key = (edit.tensor_id, int(edit.ordinal))
            if key in site_values:
                raise ValueError(f"more than one edit at use site {edit.tensor_id}#{edit.ordinal}")
            site_values[key] = planned
    clash = sorted({tensor_id for tensor_id, ordinal in site_values} & global_values.keys())
    if clash:
        raise ValueError(
            f"tensors {clash} carry both a global edit and a use-site edit; "
            "a global edit already reaches every use"
        )
    return global_values, site_values, leading_shape


def _collect_reads(
    value: Any, parameters: dict[int, tuple[torch.nn.Parameter, str]], found: dict[int, str]
) -> None:
    if isinstance(value, torch.Tensor):
        entry = parameters.get(id(value))
        if entry is not None and entry[0] is value:
            found.setdefault(id(value), entry[1])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_reads(item, parameters, found)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_reads(item, parameters, found)


def _input_tensors(
    value: Any, parameters: dict[int, tuple[torch.nn.Parameter, str]]
) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        entry = parameters.get(id(value))
        return [] if entry is not None and entry[0] is value else [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _input_tensors(item, parameters)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _input_tensors(item, parameters)]
    return []


def _substitute(value: Any, replacements: dict[int, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        return replacements.get(id(value), value)
    if isinstance(value, list):
        return [_substitute(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute(item, replacements) for item in value)
    if isinstance(value, dict):
        return {key: _substitute(item, replacements) for key, item in value.items()}
    return value


def _holds_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item) for item in value)
    return False


def _is_transpose_view(result: Any, read: torch.Tensor) -> bool:
    return (
        isinstance(result, torch.Tensor)
        and result.dim() == 2
        and read.dim() == 2
        and result.data_ptr() == read.data_ptr()
        and tuple(result.shape) == tuple(reversed(read.shape))
        and result.stride() == tuple(reversed(read.stride()))
    )


def _map_transposes(op: str, args: Any, kwargs: Any, matrix: torch.Tensor) -> bool | None:
    """Written ``y = x · Aᵀ``, whether ``op`` multiplies ``matrix`` as ``A``
    (False: the weight of ``F.linear``) or as ``Aᵀ`` (True: the right operand of
    a matmul); None when ``op`` applies no such map to it."""
    if op == _LINEAR and (args[1] if len(args) > 1 else kwargs.get("weight")) is matrix:
        return False
    if op in _MATMUL and len(args) > 1 and args[1] is matrix and matrix.dim() == 2:
        return True
    return None


_Capture = tuple[ParameterUseSite, "torch.Tensor | None", torch.Tensor, torch.Tensor, "tuple[int, ...] | None"]


class _ParameterUseMode(TorchFunctionMode):
    """Counts each tensor-returning read of a registered parameter, swaps in
    edited tensors at the planned reads, orients each read by the map that
    multiplies it, and records what the tracked uses multiplied and wrote."""

    # ``torch.backends.cuda.matmul.allow_tf32`` as read right before the forward
    # this mode runs; set by :func:`_run_under_mode`.
    tf32_matmul: bool

    def __init__(
        self,
        parameters: dict[int, tuple[torch.nn.Parameter, str]],
        module_stack: list[str],
        global_values: dict[str, _Planned],
        site_values: dict[tuple[str, int], _Planned],
        leading_shape: tuple[int, int] | None,
        requested: frozenset[tuple[str, int]],
        read_out: frozenset[tuple[str, int]],
    ) -> None:
        super().__init__()
        self._parameters = parameters
        self._module_stack = module_stack
        self._global_values = global_values
        self._site_values = site_values
        self._leading_shape = leading_shape
        self._requested = requested
        self._tracked = requested | read_out
        self._counts: dict[str, int] = {}
        self.use_sites: list[ParameterUseSite] = []
        self.substituted: list[ParameterUseSite] = []
        # (site, rows multiplied, result as carried, what the use wrote, declared positions)
        self.captured: dict[tuple[str, int], _Capture] = {}
        # A transpose-view read waits for the op that multiplies the view: (use_sites index, view).
        self.pending_views: dict[int, tuple[int, torch.Tensor]] = {}

    def __torch_function__(
        self, func: Callable[..., Any], types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = {} if kwargs is None else kwargs
        found: dict[int, str] = {}
        _collect_reads((args, kwargs), self._parameters, found)
        views = [
            entry
            for tensor in (*args, *kwargs.values())
            if isinstance(tensor, torch.Tensor)
            for entry in (self.pending_views.get(id(tensor)),)
            if entry is not None and entry[1] is tensor
        ]
        if not found and not views:
            return func(*args, **kwargs)
        replacements: dict[int, torch.Tensor] = {}
        declared: dict[int, tuple[int, ...]] = {}
        edited_keys: set[int] = set()
        reads = []
        for key, tensor_id in found.items():
            ordinal = self._counts.get(tensor_id, 0)
            # A call that returns no tensor keeps the ordinal, and an edited
            # tensor answers dtype/shape/device queries exactly as the original.
            planned = self._global_values.get(tensor_id)
            if planned is None:
                planned = self._site_values.get((tensor_id, ordinal))
            read = self._parameters[key][0] if planned is None else planned[0]
            if planned is not None:
                edited_keys.add(key)
                if planned[1] is not None:
                    declared[key] = planned[1]
            if (tensor_id, ordinal) in self._requested:
                # A storage-sharing view that requires grad makes this use's output
                # differentiable without copying the weight.
                read = read.detach().requires_grad_(True)
            if read is not self._parameters[key][0]:
                replacements[key] = read
            reads.append((key, tensor_id, ordinal))
        if not declared:
            result = func(*_substitute(args, replacements), **_substitute(kwargs, replacements))
            written = result
        else:
            result, written = self._select_declared_rows(
                func, args, kwargs, replacements, declared, reads
            )
        if not _holds_tensor(result):
            return result
        op = resolve_name(func) or getattr(func, "__qualname__", type(func).__name__)
        module = self._module_stack[-1] if self._module_stack else ""
        inputs = _input_tensors((args, kwargs), self._parameters)
        for key, tensor_id, ordinal in reads:
            param = self._parameters[key][0]
            is_view = _is_transpose_view(result, replacements.get(key, param))
            transposes = None if is_view else _map_transposes(op, args, kwargs, param)
            site = ParameterUseSite(
                tensor_id=tensor_id,
                ordinal=ordinal,
                linear=transposes is not None,
                transposed=bool(transposes),
                op=op,
                module=module,
            )
            self._counts[tensor_id] = ordinal + 1
            self.use_sites.append(site)
            if key in edited_keys:
                self.substituted.append(site)
            if is_view:
                # The map's orientation is set by whichever op multiplies this view.
                self.pending_views[id(result)] = (len(self.use_sites) - 1, result)
                continue
            if (tensor_id, ordinal) not in self._tracked:
                continue
            if transposes is None and (tensor_id, ordinal) in self._requested:
                raise ValueError(
                    f"use site {site.use_site_id} is read by {op}; only {_LINEAR} or matmul "
                    "multiplying the tensor, or its transpose view, has factored cotangent rows"
                )
            self.captured[(tensor_id, ordinal)] = (
                site,
                inputs[0] if inputs else None,
                result,
                written,
                declared.get(key),
            )
        for index, view in views:
            # A view passed twice to one call is consumed once.
            if self.pending_views.pop(id(view), None) is None:
                continue
            transposes = _map_transposes(op, args, kwargs, view)
            site = self.use_sites[index]
            if transposes is not None:
                # Multiplying the transpose view flips the orientation of the stored tensor.
                updated = replace(site, linear=True, transposed=not transposes)
                self.use_sites[index] = updated
                self.substituted = [updated if entry is site else entry for entry in self.substituted]
                site = updated
            if (site.tensor_id, site.ordinal) not in self._tracked:
                continue
            if transposes is None and (site.tensor_id, site.ordinal) in self._requested:
                raise ValueError(
                    f"use site {site.use_site_id} reads a transpose view consumed by {op}; only "
                    f"{_LINEAR} or matmul consuming it has factored cotangent rows"
                )
            self.captured[(site.tensor_id, site.ordinal)] = (
                site,
                inputs[0] if inputs else None,
                result,
                result,
                None,
            )
        return result

    def _select_declared_rows(
        self,
        func: Callable[..., Any],
        args: Any,
        kwargs: Any,
        replacements: dict[int, torch.Tensor],
        declared: dict[int, tuple[int, ...]],
        reads: list[tuple[int, str, int]],
    ) -> tuple[Any, Any]:
        """Run the op on the same inputs with ``W`` and with ``W + ΔW`` at the
        declared reads, then take the declared rows from the edited result.
        Returns the combined result and the edited run's result."""
        sites = [f"{tensor_id}#{ordinal}" for key, tensor_id, ordinal in reads if key in declared]
        position_sets = set(declared.values())
        if len(position_sets) > 1:
            raise ValueError(
                f"use sites {sites} are read by one call with different declared positions"
            )
        (positions,) = position_sets
        everywhere = {key: value for key, value in replacements.items() if key not in declared}
        clean = func(*_substitute(args, everywhere), **_substitute(kwargs, everywhere))
        edited = func(*_substitute(args, replacements), **_substitute(kwargs, replacements))
        if not _holds_tensor(clean):
            return clean, edited
        leading = self._leading_shape
        views_a_read = isinstance(clean, torch.Tensor) and any(
            clean.data_ptr() == tensor.data_ptr()
            for key, tensor_id, ordinal in reads
            for tensor in (self._parameters[key][0], replacements.get(key, self._parameters[key][0]))
        )
        if (
            not isinstance(clean, torch.Tensor)
            or clean.dim() < 2
            or tuple(clean.shape[:2]) != leading
            or views_a_read
            or not any(
                tensor.dim() >= 2 and tuple(tensor.shape[:2]) == leading
                for tensor in _input_tensors((args, kwargs), self._parameters)
            )
        ):
            shape = tuple(clean.shape) if isinstance(clean, torch.Tensor) else type(clean).__name__
            raise ValueError(
                f"use sites {sites} return {shape}, which does not keep the declared leading "
                f"axes (batch, seq) = {leading} from its input; declared positions cannot be "
                "applied there"
            )
        index = torch.as_tensor(positions, device=clean.device)
        combined = clean.clone()
        combined[:, index] = edited[:, index]
        return combined, edited


def _push_module(stack: list[str], name: str) -> Callable[..., None]:
    def hook(module: torch.nn.Module, args: Any) -> None:
        stack.append(name)

    return hook


def _pop_module(stack: list[str]) -> Callable[..., None]:
    def hook(module: torch.nn.Module, args: Any, output: Any) -> None:
        stack.pop()

    return hook


def _run_under_mode(
    model: torch.nn.Module,
    inputs: Any,
    global_values: dict[str, _Planned],
    site_values: dict[tuple[str, int], _Planned],
    leading_shape: tuple[int, int] | None,
    requested: frozenset[tuple[str, int]],
    read_out: frozenset[tuple[str, int]],
) -> tuple[Any, _ParameterUseMode]:
    table = _registered_parameters(model)
    parameters = {id(param): (param, tensor_id) for tensor_id, (param, aliases) in table.items()}
    stack: list[str] = []
    mode = _ParameterUseMode(
        parameters, stack, global_values, site_values, leading_shape, requested, read_out
    )
    handles = []
    try:
        for name, module in model.named_modules():
            handles.append(module.register_forward_pre_hook(_push_module(stack, name)))
            handles.append(module.register_forward_hook(_pop_module(stack), always_call=True))
        mode.tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
        with torch.enable_grad() if requested else torch.no_grad(), mode:
            output = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return output, mode


def _executed_output(output: Any, tf32_matmul: bool) -> ExecutedOutput:
    """``tf32_matmul`` is the flag as read right before the forward that
    produced ``output``."""
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise TypeError(
            f"the model must return a floating-point tensor; got {type(output).__name__}"
        )
    return ExecutedOutput(
        values=_float64_copy(output),
        dtype=_dtype_name(output.dtype),
        device=str(output.device),
        tf32_matmul=tf32_matmul,
    )


def _finished_execution(
    output: Any,
    mode: _ParameterUseMode,
    edits: tuple[Any, ...],
    global_values: dict[str, _Planned],
    site_values: dict[tuple[str, int], _Planned],
) -> EditedExecution:
    reached = {(site.tensor_id, site.ordinal) for site in mode.substituted}
    missed = sorted(f"{tensor_id}#{ordinal}" for tensor_id, ordinal in set(site_values) - reached)
    if missed:
        raise ValueError(f"use sites {missed} were never reached in this forward")
    unread = sorted(set(global_values) - {site.tensor_id for site in mode.substituted})
    if unread:
        raise ValueError(f"global edits of {unread} were never read in this forward")
    labelled = [
        (edit.tensor_id, int(edit.ordinal), edit.read_module, edit.read_op)
        for edit in edits
        if isinstance(edit, UseSiteParameterEdit)
    ]
    if labelled:
        # Rust checks each use-site edit against the read this forward made at its
        # ordinal, and refuses when another module or op made it.
        rust_module().check_parameter_use_site_reads(
            labelled,
            [(site.tensor_id, site.ordinal, site.module, site.op) for site in mode.use_sites],
        )
    return EditedExecution(
        output=_executed_output(output, mode.tf32_matmul),
        use_sites=tuple(mode.use_sites),
        substituted=tuple(mode.substituted),
    )


def execute_native(model: torch.nn.Module, inputs: Any) -> ExecutedOutput:
    """The all-on setting: ``model(inputs)`` on the original tensors and path."""
    tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    with torch.no_grad():
        output = model(inputs)
    return _executed_output(output, tf32_matmul)


def discover_parameter_use_sites(
    model: torch.nn.Module, inputs: Any
) -> tuple[ParameterUseSite, ...]:
    """Every use of every registered tensor in one unedited forward."""
    output, mode = _run_under_mode(model, inputs, {}, {}, None, frozenset(), frozenset())
    return tuple(mode.use_sites)


def execute_parameter_edits(
    model: torch.nn.Module,
    inputs: Any,
    edits: Iterable[GlobalParameterEdit | UseSiteParameterEdit],
    leading_shape: tuple[int, int] | None = None,
) -> EditedExecution:
    """Run ``model(inputs)`` with every edit applied for this forward only.

    A global edit makes every use read ``W + ΔW``, and refuses when this forward
    reads the tensor nowhere. A use-site edit does so only at
    ``{tensor_id}#{ordinal}``, which this forward must reach. A tensor cannot
    carry both kinds, since a global edit already reaches every use. A use-site
    edit's ``read_module`` and ``read_op`` must name the read this forward made
    at its ordinal, which Rust checks after the forward, so the forward itself
    runs exactly as without the check. ``W + ΔW`` is formed in float64 and
    refuses unless the tensor's own format represents it exactly, so an edit
    never runs rounded.

    An edit with declared positions needs ``leading_shape``, the unit's
    ``(batch, seq)``. At each of its uses the op runs twice on the same current
    input, with ``W`` and with ``W + ΔW``, and the result rows at the declared
    positions (axis 1) come from the edited run. A use whose result does not
    keep those leading axes from its input, such as ``W.t()`` or an attention
    score ``(batch, heads, seq, seq)``, refuses.

    The returned ``use_sites`` are this forward's path; a caller holding the
    discovered path compares the two to confirm the edit ran on the same one.
    """
    edits = tuple(edits)
    if not edits:
        raise ValueError(
            "execute_parameter_edits needs at least one edit; the all-on setting is execute_native"
        )
    table = _registered_parameters(model)
    global_values, site_values, leading = _plan_edits(table, edits, leading_shape)
    output, mode = _run_under_mode(
        model, inputs, global_values, site_values, leading, frozenset(), frozenset()
    )
    return _finished_execution(output, mode, edits, global_values, site_values)


def execute_parameter_cotangents(
    model: torch.nn.Module,
    inputs: Any,
    edits: Iterable[GlobalParameterEdit | UseSiteParameterEdit],
    readouts: Iterable[OutputReadout | UseSiteInputReadout | UseSiteOutputReadout],
    cotangents: Iterable[Any],
    use_sites: Iterable[str],
    leading_shape: tuple[int, int] | None = None,
) -> EditedCotangents:
    """Execute the edited forward (no edits is the all-on point) and return the
    factored cotangent rows of each requested linear use.

    ``cotangents`` holds one float64 block per declared readout, in the
    readout's shape; the objective is the float64 sum of their inner products.
    For each use in ``use_sites`` (``"{tensor_id}#{ordinal}"``), ``inputs`` are
    the rows its op multiplied and ``outputs`` the cotangent rows at what that
    op wrote, both at the executed state, oriented by the use site's
    ``transposed``. A use read as a transpose view is multiplied by the op
    consuming the view. At a use with declared positions both come from the
    edited run, at the declared rows only. A use the objective does not depend
    on gets exact zero output rows. Only ``torch.nn.functional.linear`` or
    matmul multiplying the tensor or its transpose view has factored rows; any
    other requested use refuses.
    """
    readouts = tuple(readouts)
    blocks_in = tuple(cotangents)
    requested = tuple(use_sites)
    if not readouts:
        raise ValueError("execute_parameter_cotangents needs at least one readout")
    if len(blocks_in) != len(readouts):
        raise ValueError(
            f"{len(blocks_in)} cotangent blocks for {len(readouts)} readouts; each readout takes one"
        )
    if not requested:
        raise ValueError("execute_parameter_cotangents needs at least one use site")
    table = _registered_parameters(model)
    keys = [_parse_use_site_id(use_site_id) for use_site_id in requested]
    for tensor_id, ordinal in keys:
        _lookup(table, tensor_id)
    if len(set(keys)) != len(keys):
        raise ValueError(f"use sites must be distinct; got {requested}")
    read_out = set()
    for readout in readouts:
        if isinstance(readout, (UseSiteInputReadout, UseSiteOutputReadout)):
            key = _parse_use_site_id(readout.use_site_id)
            _lookup(table, key[0])
            read_out.add(key)
        elif not isinstance(readout, OutputReadout):
            raise TypeError(
                "readouts must be OutputReadout, UseSiteInputReadout or UseSiteOutputReadout; "
                f"got {type(readout).__name__}"
            )
    edits = tuple(edits)
    global_values, site_values, leading = _plan_edits(table, edits, leading_shape)
    output, mode = _run_under_mode(
        model, inputs, global_values, site_values, leading, frozenset(keys), frozenset(read_out)
    )
    execution = _finished_execution(output, mode, edits, global_values, site_values)
    unreached = sorted(
        f"{tensor_id}#{ordinal}"
        for tensor_id, ordinal in keys
        if (tensor_id, ordinal) not in mode.captured
    )
    if unreached:
        unconsumed = sorted(mode.use_sites[index].use_site_id for index, view in mode.pending_views.values())
        raise ValueError(
            f"use sites {unreached} were never multiplied in this forward "
            f"(transpose views never consumed: {unconsumed})"
        )
    blocks: list[torch.Tensor] = []
    for readout in readouts:
        if isinstance(readout, OutputReadout):
            if output.dim() < 2 or readout.positions[-1] >= output.shape[1]:
                raise ValueError(
                    f"output readout positions {readout.positions} lie outside axis 1 of the "
                    f"output shape {tuple(output.shape)}"
                )
            block = output[:, torch.as_tensor(readout.positions, device=output.device)]
        else:
            record = mode.captured.get(_parse_use_site_id(readout.use_site_id))
            if record is None:
                raise ValueError(
                    f"readout use site {readout.use_site_id} was never reached in this forward"
                )
            if isinstance(readout, UseSiteOutputReadout):
                block = record[2]
            elif record[1] is None:
                raise ValueError(f"use site {readout.use_site_id} has no input tensor to read out")
            else:
                block = record[1]
        blocks.append(block)
    terms: list[torch.Tensor] = []
    for block, values in zip(blocks, blocks_in):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != tuple(block.shape):
            raise ValueError(
                f"a readout cotangent has shape {array.shape}; its readout block has shape "
                f"{tuple(block.shape)}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("readout cotangents must be finite")
        terms.append(
            (
                block.to(torch.float64)
                * torch.from_numpy(np.ascontiguousarray(array)).to(device=block.device)
            ).sum()
        )
    # ``readouts`` is non-empty (checked above): left-to-right float64 sum.
    objective = sum(terms[1:], terms[0])
    captured = [mode.captured[key] for key in keys]
    written = [record[3] for record in captured]
    gradients: tuple[torch.Tensor | None, ...]
    if objective.requires_grad:
        gradients = torch.autograd.grad(objective, written, allow_unused=True)
    else:
        gradients = (None,) * len(written)
    uses = []
    for (site, read, carried, wrote, positions), gradient in zip(captured, gradients):
        if read is None:
            raise ValueError(f"use site {site.use_site_id} has no input tensor to factor")
        rows_written = torch.zeros_like(wrote) if gradient is None else gradient
        rows_read = read
        if positions is not None:
            index = torch.as_tensor(positions, device=wrote.device)
            rows_written = rows_written[:, index]
            rows_read = rows_read[:, index]
        uses.append(
            UseCotangent(
                site=site,
                outputs=_float64_copy(rows_written.reshape(-1, rows_written.shape[-1])),
                inputs=_float64_copy(rows_read.reshape(-1, rows_read.shape[-1])),
            )
        )
    return EditedCotangents(
        execution=execution,
        readouts=tuple(_float64_copy(block) for block in blocks),
        uses=tuple(uses),
    )
