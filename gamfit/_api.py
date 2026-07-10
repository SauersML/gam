from __future__ import annotations

import difflib
import functools
import inspect
import json
import math
import re
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, overload

from ._binding import RustExtensionUnavailableError, extension_status, rust_module
from ._calibrated_slope import CtnStage1, normalize_ctn_stage1
from ._cuda import cuda_diagnostics as _cuda_diagnostics
from ._cuda import cuda_subprocess_env as _cuda_subprocess_env
from ._cuda import cuda_subprocess_library_dirs as _cuda_subprocess_library_dirs
from ._cuda import format_cuda_diagnostics as _format_cuda_diagnostics
from ._exceptions import map_exception
from ._model import Model
from ._reml_common import check_forward_state, coerce_grad_payload
from ._response_geometry import ResponseGeometryModel, fit_response_geometry
from ._tables import normalize_table
from ._validation import FormulaValidation
from ._warnings import emit_inference_warnings


@dataclass(frozen=True, slots=True)
class SharedPrecisionGroup:
    """Cross-fit coefficient precision group.

    ``name`` is the shared precision coordinate. By default it selects the
    same named coefficient term/column/label in every model. ``labels`` can
    override that with either one label for all models or a mapping keyed by
    the model name/index supplied to :func:`cross_fit_shared_precision_groups`.
    """

    name: str
    shape: float = 1.0
    rate: float = 0.0
    labels: str | Mapping[str | int, str] | None = None


def _normalize_shared_precision_group(
    value: Any,
    default_name: str | None = None,
) -> SharedPrecisionGroup:
    if isinstance(value, SharedPrecisionGroup):
        return value
    if isinstance(value, Mapping):
        name = value.get("name", default_name)
        if name is None:
            raise ValueError("shared precision group mapping needs a name")
        shape = value.get("shape", 1.0)
        rate = value.get("rate", 0.0)
        labels = value.get("labels")
        return SharedPrecisionGroup(
            name=str(name),
            shape=float(shape),
            rate=float(rate),
            labels=labels,
        )
    if default_name is not None:
        shape, rate = value
        return SharedPrecisionGroup(default_name, float(shape), float(rate))
    raise TypeError("shared precision groups must be SharedPrecisionGroup or mapping entries")


def _normalize_shared_precision_groups(groups: Any) -> list[SharedPrecisionGroup]:
    if isinstance(groups, Mapping):
        normalized = [
            _normalize_shared_precision_group(value, str(name))
            for name, value in groups.items()
        ]
    else:
        normalized = [_normalize_shared_precision_group(value) for value in groups]
    seen: set[str] = set()
    duplicates: list[str] = []
    for group in normalized:
        if group.name in seen:
            duplicates.append(group.name)
        seen.add(group.name)
    if duplicates:
        raise ValueError(
            "duplicate shared precision group name(s): " + ", ".join(sorted(set(duplicates)))
        )
    return normalized


def _normalize_model_mapping(models: Any) -> list[tuple[str | int, Model]]:
    if isinstance(models, Mapping):
        items = list(models.items())
    else:
        items = list(enumerate(models))
    if not items:
        raise ValueError("at least one model is required")
    out: list[tuple[str | int, Model]] = []
    for key, model in items:
        if not isinstance(model, Model):
            raise TypeError("cross-fit shared precision groups require Model instances")
        out.append((key, model))
    return out


def _shared_group_label(group: SharedPrecisionGroup, model_key: str | int) -> str:
    labels = group.labels
    if labels is None:
        return group.name
    if isinstance(labels, str):
        return labels
    if model_key in labels:
        return str(labels[model_key])
    key_text = str(model_key)
    if key_text in labels:
        return str(labels[key_text])
    raise ValueError(
        f"shared precision group {group.name!r} has no label for model {model_key!r}"
    )


def cross_fit_shared_precision_groups(
    models: Sequence[Model] | Mapping[str, Model],
    groups: Sequence[SharedPrecisionGroup | Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute EB precision updates shared across separately fitted models.

    For each declared group ``p``, the update is

    ``lambda_p = (N_fits(p) * d_p + 2 * (a_p - 1)) / (sum_q_p + 2 * b_p)``,

    where ``sum_q_p`` pools ``||beta_p||² + tr(Sigma_pp)`` over models where
    the selected term/column/label appears. If a model does not contain the
    selected block, it is skipped for that group.

    Parameters
    ----------
    models:
        Sequence or mapping of fitted :class:`gamfit.Model` objects.
    groups:
        Shared precision groups, either as :class:`SharedPrecisionGroup`
        instances or mappings with ``name`` / ``shape`` / ``rate`` fields.

    Returns
    -------
    dict
        Rust-produced update payload keyed by shared precision group.

    Raises
    ------
    TypeError
        If ``models`` does not contain :class:`Model` instances or a group
        cannot be normalized.
    ValueError
        If no models/groups are supplied or group names are duplicated.
    """

    model_items = _normalize_model_mapping(models)
    group_specs = _normalize_shared_precision_groups(groups)
    if not group_specs:
        raise ValueError("at least one shared precision group is required")

    rust = rust_module()
    model_payloads: list[dict[str, Any]] = []
    for key, model in model_items:
        try:
            state_json = rust.coefficient_state_json(model._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        model_payloads.append({"key": key, "state_json": state_json})
    group_payloads: list[dict[str, Any]] = []
    for group in group_specs:
        group_payloads.append(
            {
                "name": group.name,
                "shape": float(group.shape),
                "rate": float(group.rate),
                "labels": [
                    _shared_group_label(group, key) for key, _model in model_items
                ],
            }
        )
    try:
        raw = rust.cross_fit_shared_precision_groups_json(
            json.dumps({"models": model_payloads, "groups": group_payloads})
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return json.loads(raw)


def build_info() -> dict[str, Any]:
    """Return build/runtime metadata for the Rust extension.

    Reports whether ``gamfit._rust`` was importable and, when available, the
    build-time information exposed by the extension (version, commit, feature
    flags). Useful for bug reports and for confirming a development build is
    being used.

    Returns
    -------
    dict
        Always contains ``available`` (bool) and ``module`` (str). When the
        extension loaded, additional engine-specific keys are merged in;
        otherwise ``reason`` describes why import failed.

    Examples
    --------
    >>> info = gamfit.build_info()
    >>> info["available"]
    True
    """
    return extension_status()


def cuda_diagnostics() -> dict[str, object]:
    """Return CUDA loader diagnostics without forcing Rust GPU dispatch.

    Returns
    -------
    dict
        Loader paths, discovered CUDA libraries, and availability flags.
    """

    return _cuda_diagnostics()


def cuda_subprocess_library_dirs() -> tuple[str, ...]:
    """Return packaged CUDA wheel library directories for subprocess launchers."""

    return _cuda_subprocess_library_dirs()


def cuda_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``env`` with packaged CUDA wheel dirs prepended to ``LD_LIBRARY_PATH``.

    Parameters
    ----------
    env:
        Base environment mapping. ``None`` starts from an empty mapping.
    """

    return _cuda_subprocess_env(env)


def format_cuda_diagnostics() -> str:
    """Return CUDA loader diagnostics as stable, grep-friendly text."""

    return _format_cuda_diagnostics()


def _build_fit_payload(
    *,
    family: str,
    offset: str | None,
    weights: str | None,
    transformation_normal: bool | None,
    transformation_normal_stage1: Any | None = None,
    survival_likelihood: str | None,
    baseline_target: str | None,
    baseline_scale: float | None,
    baseline_shape: float | None,
    baseline_rate: float | None,
    baseline_makeham: float | None,
    z_column: str | None,
    link: str | None,
    logslope_formula: str | None,
    frailty_kind: str | None,
    frailty_sd: float | None,
    hazard_loading: str | None,
    scale_dimensions: bool | None,
    adaptive_regularization: bool | None,
    firth: bool | None,
    noise_formula: str | None,
    noise_offset: str | None,
    flexible_link: bool | None,
    precision_hyperpriors: Any | None,
    latents: Mapping[str, Any] | None,
    penalties: Sequence[Any] | None,
    smooths: Mapping[Any, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_latents = _normalize_latents(latents)
    payload: dict[str, Any] = {
        "family": family,
        "offset": offset,
        "weights": weights,
    }
    ctn_stage1_recipe = normalize_ctn_stage1(transformation_normal_stage1)
    kwarg_items: dict[str, Any] = {
        "transformation_normal": transformation_normal,
        "ctn_stage1": ctn_stage1_recipe.to_rust_recipe() if ctn_stage1_recipe else None,
        "survival_likelihood": survival_likelihood,
        "baseline_target": baseline_target,
        "baseline_scale": baseline_scale,
        "baseline_shape": baseline_shape,
        "baseline_rate": baseline_rate,
        "baseline_makeham": baseline_makeham,
        "z_column": z_column,
        "link": link,
        "logslope_formula": logslope_formula,
        "frailty_kind": frailty_kind,
        "frailty_sd": frailty_sd,
        "hazard_loading": hazard_loading,
        "scale_dimensions": scale_dimensions,
        "adaptive_regularization": adaptive_regularization,
        "firth": firth,
        "noise_formula": noise_formula,
        "noise_offset": noise_offset,
        "flexible_link": flexible_link,
        "precision_hyperpriors": precision_hyperpriors,
        "latents": normalized_latents,
        "penalties": _normalize_penalties(penalties, normalized_latents),
        "smooths": _normalize_smooths(smooths),
    }
    for key, value in kwarg_items.items():
        if value is not None:
            payload[key] = value
    if config:
        for key, value in config.items():
            payload.setdefault(key, _jsonable_array(value))
    return payload


def _jsonable_array(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (Path,)):
        return str(value)
    if hasattr(value, "to_rust_descriptor"):
        return _jsonable_array(value.to_rust_descriptor())
    if hasattr(value, "_to_rust_payload"):
        return _jsonable_array(value._to_rust_payload())
    if isinstance(value, Mapping):
        return {str(k): _jsonable_array(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_array(v) for v in value]
    import numpy as np

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"cannot serialize config value of type {type(value).__name__} as JSON"
        ) from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("numeric config arrays must contain only finite values")
    return arr.tolist()


def _normalize_latents(latents: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if latents is None:
        return None
    out: dict[str, Any] = {}
    for name, latent in latents.items():
        raw = {
            "name": getattr(latent, "name", None) or str(name),
            "n": int(getattr(latent, "n")),
            "d": int(getattr(latent, "d")),
            "init": _jsonable_array(getattr(latent, "init", "pca")),
            "manifold": _jsonable_array(getattr(latent, "manifold", "auto")),
            "retraction": _jsonable_array(getattr(latent, "retraction", "euclidean")),
            "aux_prior": _jsonable_array(getattr(latent, "aux_prior", None)),
            "dim_selection": bool(getattr(latent, "dim_selection", False)),
        }
        out[str(name)] = {k: v for k, v in raw.items() if v is not None}
    return out


def _normalize_smooths(
    smooths: Mapping[Any, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    """Serialize ``smooths={symbol: BasisDescriptor}`` to the Rust bridge form.

    Keys may be either a single column name (``"x"``) or a tuple of column
    names for multivariate smooths (``("lat", "lon")``); the latter is
    normalized to a comma-joined string so the Rust side can deserialize
    into a plain ``BTreeMap<String, _>``. Values must implement
    :meth:`Smooth.to_rust_descriptor`; arrays inside the descriptor (centers
    matrices, knot vectors) are pre-flattened to JSON-able nested lists.
    """
    if smooths is None:
        return None
    if not isinstance(smooths, Mapping):
        raise TypeError(
            "smooths must be a mapping of formula symbol -> Smooth descriptor"
        )
    from .smooth import Smooth as _Smooth

    out: dict[str, dict[str, Any]] = {}
    for raw_key, descriptor in smooths.items():
        if isinstance(raw_key, (tuple, list)):
            key = ",".join(str(v).strip() for v in raw_key)
        else:
            key = str(raw_key).strip()
        if not key:
            raise ValueError("smooths keys must be non-empty symbols")
        if not isinstance(descriptor, _Smooth):
            raise TypeError(
                f"smooths[{key!r}] must be a gamfit.smooth.Smooth descriptor "
                f"(Duchon / Matern / MeasureJet / Sphere / BSpline / "
                f"TensorBSpline / Pca / PeriodicSplineCurve / Categorical), "
                f"got {type(descriptor).__name__}"
            )
        by = getattr(descriptor, "by", None)
        if by is not None and not isinstance(by, str):
            # On the formula `smooths={}` descriptor path the gating variable
            # is named by data-frame column (resolved to a `by_col` in the Rust
            # merge, identical to `s(x, by=g)`). A raw per-row `by` *array* is
            # the contract of the primitive numpy API (`gamfit.duchon_basis`,
            # ... — `crates/gam-pyffi/src/model_ffi.rs`), which has no data
            # frame to name. Reject it loudly here rather than mis-serialize.
            raise ValueError(
                f"smooths[{key!r}]: by= on the smooths={{}} descriptor path must "
                f"be the *name* of a data-frame column (the gating variable), "
                f"e.g. Duchon(..., by='g'); got {type(by).__name__}. Raw per-row "
                f"by arrays are supported only on the primitive numpy API."
            )
        payload = descriptor.to_rust_descriptor()
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"{type(descriptor).__name__}.to_rust_descriptor() must return a "
                f"mapping; got {type(payload).__name__}"
            )
        payload = dict(payload)
        payload.setdefault("vars", [v.strip() for v in key.split(",")])
        out[key] = payload
    return out


def _normalize_penalties(
    penalties: Sequence[Any] | None,
    latents: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if penalties is None:
        return None
    if isinstance(penalties, (str, bytes)) or not isinstance(penalties, Sequence):
        raise TypeError("penalties must be a sequence of analytic penalty wrappers")
    from .smooth import Smooth as _Smooth

    latent_names = list((latents or {}).keys())
    out: list[dict[str, Any]] = []
    for index, penalty in enumerate(penalties):
        if isinstance(penalty, _Smooth):
            # `penalties=` is reserved for analytic penalty wrappers that
            # target latent blocks (OrthogonalityPenalty, ARDPenalty,
            # SparsityPenalty, ...). Smooth descriptors (Duchon, Matern,
            # Sphere, ...) describe formula *terms*, not latent-block
            # penalties, so they belong on the sibling ``smooths=`` kwarg
            # (gam issue #315). Point users at it instead of dropping the
            # value silently.
            raise TypeError(
                f"penalties[{index}] is a {type(penalty).__name__} smooth "
                "descriptor; pass it via the `smooths=` kwarg of fit() "
                "instead. `penalties=` is for analytic penalty wrappers "
                "targeting latent blocks (OrthogonalityPenalty, ARDPenalty, "
                "SparsityPenalty, ...). Example: "
                f"`fit(df, formula, smooths={{<symbol>: {type(penalty).__name__}(...)}})`."
            )
        if hasattr(penalty, "to_rust_descriptor"):
            descriptor = penalty.to_rust_descriptor()
        elif hasattr(penalty, "_to_rust_payload"):
            descriptor = penalty._to_rust_payload()
        elif isinstance(penalty, Mapping):
            descriptor = dict(penalty)
        else:
            raise TypeError(
                f"penalties[{index}] must expose to_rust_descriptor() or be a mapping"
            )
        target = descriptor.get("target")
        if isinstance(target, int):
            if target < 0 or target >= len(latent_names):
                raise ValueError(
                    f"penalties[{index}] targets latent index {target}, "
                    f"but latents has {len(latent_names)} block(s)"
                )
            descriptor["target"] = latent_names[target]
        elif isinstance(target, str):
            if target not in (latents or {}):
                raise ValueError(
                    f"penalties[{index}] targets latent block {target!r}, "
                    "which is not present in latents"
                )
        else:
            raise TypeError(
                f"penalties[{index}] target must be a latent block name or index"
            )
        out.append(_jsonable_array(descriptor))
    return out


def _normalize_penalty_descriptors(penalties: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    if penalties is None:
        return None
    if isinstance(penalties, (str, bytes)) or not isinstance(penalties, Sequence):
        raise TypeError("penalties must be a sequence of analytic penalty wrappers")
    out: list[dict[str, Any]] = []
    for index, penalty in enumerate(penalties):
        if hasattr(penalty, "to_rust_descriptor"):
            descriptor = penalty.to_rust_descriptor()
        elif hasattr(penalty, "_to_rust_payload"):
            descriptor = penalty._to_rust_payload()
        elif isinstance(penalty, Mapping):
            descriptor = dict(penalty)
        else:
            raise TypeError(
                f"penalties[{index}] must expose to_rust_descriptor() or be a mapping"
            )
        out.append(_jsonable_array(descriptor))
    return out


def _normalize_aux_strength(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "auto":
            return None
        raise ValueError(f"aux_strength string value must be 'auto', got {value!r}")
    return float(value)


def _normalize_precision_pair(value: Any, label: str) -> list[float]:
    if isinstance(value, dict):
        shape = value.get("shape")
        rate = value.get("rate")
    else:
        try:
            shape, rate = value
        except Exception as exc:  # pragma: no cover - defensive shape guard
            raise ValueError(
                f"precision_hyperpriors[{label!r}] must be (shape, rate)"
            ) from exc
    if shape is None:
        raise ValueError(f"precision_hyperpriors[{label!r}] needs a shape value")
    if rate is None:
        raise ValueError(f"precision_hyperpriors[{label!r}] needs a rate value")
    shape_f = float(shape)
    rate_f = float(rate)
    if (
        not math.isfinite(shape_f)
        or not math.isfinite(rate_f)
        or shape_f <= 0.0
        or rate_f < 0.0
    ):
        raise ValueError(
            f"precision_hyperpriors[{label!r}] needs finite shape > 0 and finite rate >= 0"
        )
    return [shape_f, rate_f]


def _group_terms_from_formula(formula: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\bgroup\s*\(\s*([^)]+?)\s*\)", formula)]


def _resolve_precision_hyperpriors(
    value: Any | None,
    formula: str,
    headers: list[str],
    rows: list[list[str]],
    group_metadata: Any | None = None,
) -> Any | None:
    if value is None:
        return None
    if callable(value):
        out: dict[str, list[float]] = {}
        metadata_by_label = group_metadata if isinstance(group_metadata, dict) else {}
        labels: list[str] = []
        for label in _group_terms_from_formula(formula):
            if label not in labels:
                labels.append(label)
        for label in metadata_by_label:
            if str(label) not in labels:
                labels.append(str(label))
        for label in labels:
            levels: list[str] = []
            if label in headers:
                col = headers.index(label)
                levels = sorted({row[col] for row in rows})
            pair = value(
                {
                    "label": label,
                    "term": label,
                    "column": label,
                    "levels": levels,
                    "n_coefficients": len(levels),
                    "metadata": metadata_by_label.get(label),
                    "group_metadata": metadata_by_label.get(label),
                }
            )
            out[label] = _normalize_precision_pair(pair, label)
        return out
    if isinstance(value, dict):
        return {str(k): _normalize_precision_pair(v, str(k)) for k, v in value.items()}
    return value


def _normalize_fisher_rao_w(value: Any, *, n_rows: int, dim: int) -> Any:
    """Broadcast/validate Fisher-Rao precision blocks to ``(n_rows, dim, dim)``.

    Pure marshaling: the broadcasting and the finiteness / symmetry / PSD
    validation are the Rust single source of truth
    (``response_geometry_normalize_fisher_rao`` →
    ``gam::inference::fisher_rao::normalize_fisher_rao_blocks``), shared with the
    response-geometry path. Python only coerces the input to a float64 array.
    """
    import numpy as np

    arr = np.asarray(value, dtype=float)
    try:
        return rust_module().response_geometry_normalize_fisher_rao(
            arr, int(n_rows), int(dim)
        )
    except Exception as exc:
        raise map_exception(exc) from exc


_KWARG_TYPO_PATTERN = re.compile(r"unexpected keyword argument [\"']([^\"']+)[\"']")


def _suggest_kwarg_typo(fn: Any) -> Any:
    """Decorator: turn ``TypeError("...unexpected keyword argument 'X'...")``
    into the same TypeError with a ``Did you mean 'Y'?`` hint appended.

    Python 3.13 already adds these hints natively (PEP 657), but earlier
    Python versions (3.10 - 3.12) raise a bare TypeError; this wrapper
    backfills the hint uniformly so callers on any supported Python see the
    same actionable message for case typos like ``formuLa=`` / ``Familiy=``
    / ``Offset=`` against a long-keyword-list API (~25 kwargs). (issue #306)
    """
    sig = inspect.signature(fn)
    known = frozenset(
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "Did you mean" in message:
                raise
            match = _KWARG_TYPO_PATTERN.search(message)
            if match is None:
                raise
            bad = match.group(1)
            if bad in known:
                raise
            suggestions = difflib.get_close_matches(bad, known, n=1, cutoff=0.6)
            if not suggestions:
                raise
            enriched = TypeError(f"{message}. Did you mean {suggestions[0]!r}?")
            raise enriched from exc.__cause__

    return wrapper


@overload
def fit(
    activations: Any,
    formula: None = ...,
    *,
    config: Mapping[str, Any] | None = ...,
) -> dict[str, Any]: ...


@overload
def fit(
    data: Any,
    formula: str,
    *,
    family: str = ...,
    offset: str | None = ...,
    weights: str | None = ...,
    transformation_normal: bool | None = ...,
    transformation_normal_stage1: CtnStage1 | Mapping[str, Any] | None = ...,
    survival_likelihood: str | None = ...,
    baseline_target: str | None = ...,
    baseline_scale: float | None = ...,
    baseline_shape: float | None = ...,
    baseline_rate: float | None = ...,
    baseline_makeham: float | None = ...,
    z_column: str | None = ...,
    link: str | None = ...,
    logslope_formula: str | None = ...,
    frailty_kind: str | None = ...,
    frailty_sd: float | None = ...,
    hazard_loading: str | None = ...,
    scale_dimensions: bool | None = ...,
    adaptive_regularization: bool | None = ...,
    firth: bool | None = ...,
    noise_formula: str | None = ...,
    noise_offset: str | None = ...,
    flexible_link: bool | None = ...,
    precision_hyperpriors: Any | None = ...,
    constraints: Mapping[str, Any] | None = ...,
    response_geometry: None = ...,
    response_columns: list[str] | tuple[str, ...] | None = ...,
    response_coordinates: str | None = ...,
    response_reference: int | None = ...,
    fisher_rao_w: Any | None = ...,
    latents: Mapping[str, Any] | None = ...,
    penalties: Sequence[Any] | None = ...,
    smooths: Mapping[Any, Any] | None = ...,
    config: dict[str, Any] | None = ...,
) -> Model: ...


@overload
def fit(
    data: Any,
    formula: str,
    *,
    family: str = ...,
    offset: str | None = ...,
    weights: str | None = ...,
    transformation_normal: bool | None = ...,
    transformation_normal_stage1: CtnStage1 | Mapping[str, Any] | None = ...,
    survival_likelihood: str | None = ...,
    baseline_target: str | None = ...,
    baseline_scale: float | None = ...,
    baseline_shape: float | None = ...,
    baseline_rate: float | None = ...,
    baseline_makeham: float | None = ...,
    z_column: str | None = ...,
    link: str | None = ...,
    logslope_formula: str | None = ...,
    frailty_kind: str | None = ...,
    frailty_sd: float | None = ...,
    hazard_loading: str | None = ...,
    scale_dimensions: bool | None = ...,
    adaptive_regularization: bool | None = ...,
    firth: bool | None = ...,
    noise_formula: str | None = ...,
    noise_offset: str | None = ...,
    flexible_link: bool | None = ...,
    precision_hyperpriors: Any | None = ...,
    constraints: Mapping[str, Any] | None = ...,
    response_geometry: str,
    response_columns: list[str] | tuple[str, ...] | None = ...,
    response_coordinates: str | None = ...,
    response_reference: int | None = ...,
    fisher_rao_w: Any | None = ...,
    latents: Mapping[str, Any] | None = ...,
    penalties: Sequence[Any] | None = ...,
    smooths: Mapping[Any, Any] | None = ...,
    config: dict[str, Any] | None = ...,
) -> ResponseGeometryModel: ...


def fit(
    data: Any,
    formula: str | None = None,
    *,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    transformation_normal: bool | None = None,
    transformation_normal_stage1: CtnStage1 | Mapping[str, Any] | None = None,
    survival_likelihood: str | None = None,
    baseline_target: str | None = None,
    baseline_scale: float | None = None,
    baseline_shape: float | None = None,
    baseline_rate: float | None = None,
    baseline_makeham: float | None = None,
    z_column: str | None = None,
    link: str | None = None,
    logslope_formula: str | None = None,
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    noise_formula: str | None = None,
    noise_offset: str | None = None,
    flexible_link: bool | None = None,
    precision_hyperpriors: Any | None = None,
    constraints: Mapping[str, Any] | None = None,
    response_geometry: str | None = None,
    response_columns: list[str] | tuple[str, ...] | None = None,
    response_coordinates: str | None = None,
    response_reference: int | None = None,
    fisher_rao_w: Any | None = None,
    latents: Mapping[str, Any] | None = None,
    penalties: Sequence[Any] | None = None,
    smooths: Mapping[Any, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Model | ResponseGeometryModel | dict[str, Any]:
    """Fit a GAM model, or fit SAE activations when ``formula`` is omitted.

    ``gamfit.fit(data, formula, ...)`` keeps the formula-first GAM API.
    ``gamfit.fit(activations, config=...)`` dispatches to the SAE research-loop
    API and returns typed atoms, coordinates, and trust-score hooks.

    Parameters
    ----------
    data:
        Input table. Accepts a pandas DataFrame, pyarrow Table, dict of columns,
        list of records, or any object normalize_table understands.
    formula:
        Wilkinson-style formula string (e.g. ``"y ~ s(x1) + te(x2, x3)"``).
    family:
        Likelihood family, or ``"auto"`` to infer from the response. Corresponds
        to the ``--family`` CLI flag. Scalar fit values include ``"gaussian"``,
        ``"binomial"`` / ``"bernoulli"``, ``"poisson"``, ``"gamma"``,
        ``"beta"``, ``"tweedie"`` / ``"tw"``, and ``"negative-binomial"`` /
        ``"negbin"`` / ``"nb"``. Binomial/Bernoulli link spellings accept
        ``"-logit"``, ``"-probit"``, ``"-cloglog"``, or mgcv-style
        parentheses such as ``"bernoulli(probit)"``. Specialized values include
        ``"gaussian-location-scale"`` when ``noise_formula`` is supplied,
        ``"bernoulli-marginal-slope"``, ``"royston-parmar"``, and
        ``"transformation-normal"``.
    offset:
        Name of the offset column. Corresponds to ``--offset-column``.
    weights:
        Name of the observation-weight column. Corresponds to ``--weights-column``.
    transformation_normal:
        Fit a conditional transformation-normal model (``h(Y|x) ~ N(0,1))``).
        Corresponds to ``--transformation-normal``.
    transformation_normal_stage1:
        Stage-1 CTN recipe for a *calibrated* marginal-slope chain
        (:class:`gamfit.CtnStage1`, or a mapping of its fields). Supply it on a
        marginal-slope model (``family="bernoulli-marginal-slope"`` or a
        survival ``survival_likelihood="marginal-slope"``) to auto-enable the
        cross-fitted, Neyman-orthogonal score calibration of #461: the Rust core
        fits the CTN ``h(Y|x) ~ N(0,1)`` per fold, derives an out-of-fold latent
        score ``z`` that replaces the in-sample one, and absorbs the Stage-1
        score-influence directions so the fitted slope surface ``β(x)`` is
        insensitive to Stage-1 calibration error. There is no boolean to toggle
        orthogonalization — supplying this recipe *is* the request (magic by
        default). A raw ``z_column`` selects the free-warp ``score_warp`` path.
        All numerics stay in Rust; this only marshals the recipe. The Stage-1
        ``response`` column must exist in ``data`` alongside the Stage-2
        response and covariates.
    survival_likelihood:
        Survival likelihood formulation. One of ``"transformation"``,
        ``"weibull"``, ``"location-scale"``, ``"marginal-slope"``,
        ``"latent"``, or ``"latent-binary"``. Corresponds to
        ``--survival-likelihood``.
    baseline_target:
        Parametric baseline target for survival models. One of ``"linear"``,
        ``"weibull"``, ``"gompertz"``, ``"gompertz-makeham"``. Corresponds to
        ``--baseline-target``.
    baseline_scale:
        Weibull baseline scale (>0) when ``baseline_target="weibull"``.
        Corresponds to ``--baseline-scale``.
    baseline_shape:
        Weibull baseline shape (>0). Corresponds to ``--baseline-shape``.
    baseline_rate:
        Gompertz hazard rate (>0) when ``baseline_target`` is ``"gompertz"``
        or ``"gompertz-makeham"``. Corresponds to ``--baseline-rate``.
    baseline_makeham:
        Makeham additive hazard (>0) when ``baseline_target="gompertz-makeham"``.
        Corresponds to ``--baseline-makeham``.
    z_column:
        Name of the latent/observed z-score column used by score-warp families
        and latent transformation models. Corresponds to ``--z-column``.
    link:
        Override the default link function. Corresponds to ``--link``.
    logslope_formula:
        Secondary formula for the logslope / score-warp submodel. Corresponds to
        ``--logslope-formula``.
    frailty_kind:
        Frailty family for frailty-aware survival models. One of
        ``"gaussian-shift"`` or ``"hazard-multiplier"``. Corresponds to
        ``--frailty-kind``.
    response_geometry:
        Optional manifold-valued response geometry. Use ``"spherical"`` for
        unit-sphere responses, or ``"simplex"`` / ``"clr"`` / ``"alr"`` for
        strictly positive compositional responses. Curved matrix/ball manifolds
        are also fittable: ``"spd"`` (symmetric positive-definite cone),
        ``"grassmann(k=..)"``, ``"stiefel(k=..)"``, ``"poincare"`` (fixed
        curvature ``< 0``), and ``"constant_curvature"`` — the unified M_κ family
        whose curvature κ̂ is ESTIMATED from the responses (not user-supplied) by
        the REML/evidence outer loop, reporting κ̂ with a profile-likelihood CI,
        the geometry verdict (spherical/flat/hyperbolic), and the Wilks flatness
        test of κ = 0. The base point is the intrinsic Fréchet mean of the
        training responses, not an extrinsic arithmetic mean.
    response_columns:
        Sequence of response component columns used when ``response_geometry``
        is set. One scalar Gaussian GAM is fitted for each tangent coordinate.
    response_coordinates:
        Coordinate chart for simplex responses: ``"clr"`` (default) or
        ``"alr"``. Spherical responses always use ambient tangent coordinates.
    response_reference:
        Reference component for ``"alr"`` coordinates (default: last column).
    fisher_rao_w:
        Optional Fisher-Rao behavioral precision blocks. Accepts length-N
        scalar weights, one broadcast ``(p, p)`` matrix, or dense ``(N, p, p)``
        blocks.
    frailty_sd:
        Fixed frailty standard deviation. Omit to let latent hazard-multiplier
        models learn it. Corresponds to ``--frailty-sd``.
    hazard_loading:
        Hazard loading for ``frailty_kind="hazard-multiplier"``. One of
        ``"full"`` or ``"loaded-vs-unloaded"``. Corresponds to
        ``--hazard-loading``.
    scale_dimensions:
        When ``True``, enables learned per-axis anisotropic length scales on
        spatial smooths (e.g. multi-dim Duchon / Matern / TPS). Per-axis
        scales are learned, not specified. Corresponds to ``--scale-dimensions``.
    adaptive_regularization:
        Enable exact local adaptive regularization for compatible spatial
        smooths. Omit to use the quality-first automatic policy, which leaves
        it off unless explicitly requested.
    firth:
        Enable Firth bias-reduced estimation. Corresponds to ``--firth``.
    noise_formula:
        Wilkinson-style formula for the log-scale (dispersion) component,
        turning the fit into a location-scale GAMLSS where both the mean and
        the residual spread vary smoothly with the covariates (e.g.
        ``"s(x)"``). Passing this is the request to estimate a non-constant
        scale. The family is magic-routed from ``family``: with the default
        ``"gaussian"`` it models ``log σ``; with ``"binomial"`` it models the
        latent-threshold scale; and with the genuine-dispersion mean families
        ``"gamma"``, ``"beta"``, ``"nb"`` (negative-binomial) or ``"tweedie"``
        the noise formula models that family's own overdispersion channel
        (Gamma shape, Beta φ, NB θ, Tweedie 1/φ), giving a full dispersion
        GAMLSS (#913). Corresponds to the ``--predict-noise`` CLI path
        (``FitConfig.noise_formula``).
    noise_offset:
        Name of a column supplying a fixed additive offset on the log-scale
        (dispersion) predictor of the location-scale model, analogous to
        ``offset`` for the mean. Corresponds to
        ``FitConfig.noise_offset_column``.
    flexible_link:
        Estimate a flexible (wiggly) link function rather than holding the
        link fixed at its canonical/parametric form, letting the data shape
        the response transformation. Corresponds to the CLI flexible-link path
        (``FitConfig.flexible_link``).
    constraints:
        Optional mapping of smooth-term text to a shape-constraint kind.
        Keys are the literal smooth term as it appears in ``formula`` (e.g.
        ``"s(x)"`` or ``"s(x, type=duchon, centers=8)"``; whitespace
        differences are ignored). Values are one of ``"monotone_increasing"``,
        ``"monotone_decreasing"``, ``"convex"``, ``"concave"``, or
        ``"none"`` / ``None`` for the default unconstrained fit. Shape
        constraints are enforced by the inner solver as joint linear
        inequalities ``A·β ≤ b`` on the coefficient vector; when active at
        convergence the outer REML score uses the tangent-projected LAML
        formulation. This is the same functionality exposed by mgcv's
        ``scop=...`` argument and the ``scam`` R library. Currently restricted
        to univariate 1D B-spline / thin-plate / Duchon smooths.

        Example::

            gamfit.fit(df, "y ~ s(x)",
                       constraints={"s(x)": "monotone_increasing"})
    config:
        Escape-hatch dict of extra pipeline keys. Any key already set via a
        dedicated kwarg wins over the same key in ``config``. Standard formula
        fits also accept ``outer_max_iter`` as a positive integer cap on outer
        smoothing-parameter iterations.
    latents:
        Mapping from formula symbol to :class:`gamfit.LatentCoord`. This is
        the standard fit API surface for per-row latent coordinates. The Rust
        standard workflow maps the named formula smooth onto the latent
        coordinate matrix and optimizes it jointly with the REML parameters.
    penalties:
        Analytic penalty wrappers such as :class:`gamfit.OrthogonalityPenalty`
        or :class:`gamfit.ARDPenalty`, targeted at latent block names or
        indices declared in ``latents``. The Rust-backed public wrappers also
        include the SAE/assignment family
        (:class:`gamfit.SoftmaxAssignmentSparsityPenalty`,
        :class:`gamfit.IBPAssignmentPenalty`,
        :class:`gamfit.TopKActivationPenalty`,
        :class:`gamfit.JumpReLUPenalty`) and newer structured penalties such
        as :class:`gamfit.ScadMcpPenalty` and
        :class:`gamfit.NuclearNormPenalty`. ``penalties=`` is not for smooth
        basis descriptors; pass those through ``smooths=``.
    smooths:
        Optional mapping from formula symbol to :class:`gamfit.smooth.Smooth`
        basis descriptor (``Duchon``, ``Matern``, ``MeasureJet``, ``Sphere``,
        ``BSpline``, ``TensorBSpline``, ``Pca``, ``PeriodicSplineCurve``,
        ``Categorical``).
        Keys are either a single column name (``"x"``) or a tuple of column
        names for multivariate smooths (``("lat", "lon")``). The descriptor
        threads explicit center coordinates / knot vectors / kernel
        hyperparameters into the same internal ``SmoothBasisSpec`` the
        formula DSL produces — when all descriptor fields default to the
        same values the DSL would auto-pick, the resulting block spec is
        bit-identical to writing ``duchon(x, centers=K)`` (or the analogous
        DSL invocation) in the formula. The only delta is that an explicit
        ``centers=<array>`` array routes through
        ``CenterStrategy::UserProvided``, whereas the DSL integer
        ``centers=K`` routes through ``FarthestPoint``/``EqualMass``. Used
        when you need to pin the basis centers to a precomputed
        farthest-point sample, lattice, or domain landmark set.

        Example::

            from gamfit.smooth import Duchon, Sphere
            centers = my_farthest_point_sample(X[["x"]].values, 20)
            gamfit.fit(df, "y ~ duchon(x)",
                       smooths={"x": Duchon(centers=centers)})
            gamfit.fit(df, "y ~ sphere(lat, lon)",
                       smooths={("lat", "lon"): Sphere(centers=lattice)})

    Returns
    -------
    Model or ResponseGeometryModel or MultinomialModel
        A fitted scalar GAM :class:`Model` by default. When
        ``response_geometry`` is supplied, returns a
        :class:`ResponseGeometryModel`; when ``family`` requests the
        multinomial/softmax path, returns a :class:`MultinomialModel`.

    Raises
    ------
    TypeError
        For malformed ``penalties`` / ``smooths`` containers or unexpected
        keyword arguments.
    ValueError
        For missing response-geometry fields, invalid penalty targets, invalid
        Fisher-Rao blocks, or formula/data mismatches surfaced before the Rust
        fit.
    GamError
        Rust engine errors are mapped into the typed gamfit exception
        hierarchy.
    """
    if formula is None:
        formula_only = {
            "family": family,
            "offset": offset,
            "weights": weights,
            "transformation_normal": transformation_normal,
            "transformation_normal_stage1": transformation_normal_stage1,
            "survival_likelihood": survival_likelihood,
            "baseline_target": baseline_target,
            "baseline_scale": baseline_scale,
            "baseline_shape": baseline_shape,
            "baseline_rate": baseline_rate,
            "baseline_makeham": baseline_makeham,
            "z_column": z_column,
            "link": link,
            "logslope_formula": logslope_formula,
            "frailty_kind": frailty_kind,
            "frailty_sd": frailty_sd,
            "hazard_loading": hazard_loading,
            "scale_dimensions": scale_dimensions,
            "adaptive_regularization": adaptive_regularization,
            "firth": firth,
            "noise_formula": noise_formula,
            "noise_offset": noise_offset,
            "flexible_link": flexible_link,
            "precision_hyperpriors": precision_hyperpriors,
            "constraints": constraints,
            "response_geometry": response_geometry,
            "response_columns": response_columns,
            "response_coordinates": response_coordinates,
            "response_reference": response_reference,
            "fisher_rao_w": fisher_rao_w,
            "latents": latents,
            "penalties": penalties,
            "smooths": smooths,
        }
        active_formula_kwargs = [
            name for name, value in formula_only.items()
            if value is not None and not (name == "family" and value == "auto")
        ]
        if active_formula_kwargs:
            raise TypeError(
                "gamfit.fit requires formula='...' when formula-model kwargs are supplied: "
                + ", ".join(active_formula_kwargs)
            )
        from ._sae_manifold import fit as _sae_research_fit

        return _sae_research_fit(data, config=config)

    if constraints:
        # Alias normalization, smooth-term scanning, and the `shape=` rewrite all
        # live in Rust (`gam::terms::smooth::apply_shape_constraints_to_formula`);
        # Python only marshals the mapping across the FFI.
        try:
            formula = rust_module().apply_shape_constraints_to_formula(
                formula, [(str(k), str(v)) for k, v in constraints.items()]
            )
        except Exception as exc:
            raise map_exception(exc) from exc
    if config:
        if response_geometry is None and config.get("response_geometry") is not None:
            response_geometry = str(config["response_geometry"])
        if response_columns is None and config.get("response_columns") is not None:
            raw_columns = config["response_columns"]
            if isinstance(raw_columns, (str, bytes)):
                raise ValueError(
                    "response_columns must be a sequence of column names, not a string"
                )
            response_columns = tuple(str(name) for name in raw_columns)
        if response_coordinates is None and config.get("response_coordinates") is not None:
            response_coordinates = str(config["response_coordinates"])
        if response_reference is None and config.get("response_reference") is not None:
            response_reference = int(config["response_reference"])

    if response_geometry is not None:
        if response_columns is None:
            raise ValueError("response_columns is required when response_geometry is set")
        if family != "auto":
            raise ValueError("family is forced to gaussian and cannot be specified with response_geometry")

        for arg_name, arg_val in [
            ("offset", offset),
            ("transformation_normal", transformation_normal),
            ("transformation_normal_stage1", transformation_normal_stage1),
            ("survival_likelihood", survival_likelihood),
            ("baseline_target", baseline_target),
            ("baseline_scale", baseline_scale),
            ("baseline_shape", baseline_shape),
            ("baseline_rate", baseline_rate),
            ("baseline_makeham", baseline_makeham),
            ("z_column", z_column),
            ("link", link),
            ("logslope_formula", logslope_formula),
            ("frailty_kind", frailty_kind),
            ("frailty_sd", frailty_sd),
            ("hazard_loading", hazard_loading),
            ("noise_formula", noise_formula),
            ("noise_offset", noise_offset),
            ("flexible_link", flexible_link),
        ]:
            if arg_val is not None:
                raise ValueError(f"{arg_name} is not supported with response_geometry")

        nested_config = dict(config or {})
        # Geometry is handled by the Python wrapper; scalar coordinate fits keep
        # using the ordinary Rust standard-GAM path.
        for key in (
            "response_geometry",
            "response_columns",
            "response_coordinates",
            "response_reference",
        ):
            nested_config.pop(key, None)
        return fit_response_geometry(
            fit,
            data,
            formula,
            response_geometry=response_geometry,
            response_columns=tuple(response_columns),
            coordinates=response_coordinates,
            reference=-1 if response_reference is None else int(response_reference),
            weights=weights,
            fisher_rao_w=fisher_rao_w,
            scale_dimensions=scale_dimensions,
            adaptive_regularization=adaptive_regularization,
            firth=firth,
            precision_hyperpriors=precision_hyperpriors,
            latents=latents,
            penalties=penalties,
            smooths=smooths,
            constraints=constraints,
            config=nested_config or None,
        )

    headers, rows, table_kind = normalize_table(data)

    # ── Vector-response (multinomial-logit) dispatch (#328). ──────────────
    # The scalar `fit_table` payload pipeline is parameterised by a single
    # `ResponseFamily × InverseLink` likelihood spec; multinomial-logit
    # carries K-1 active linear predictors and a per-row dense Fisher block,
    # which the scalar pipeline cannot represent. Routing here keeps the
    # high-level Python API uniform — `gamfit.fit(data, formula,
    # family='multinomial')` returns a `MultinomialModel` — while the
    # underlying Rust entry is a dedicated formula→design→REML path that
    # bypasses the workflow.rs `FitRequest::Standard` materialiser. The Rust
    # `fit_penalized_multinomial_formula` driver runs the outer REML/LAML loop
    # to select an independent smoothing parameter per (class, term); the
    # `init_lambda` argument below is only the warm-start seed.
    family_canonical = str(family).lower().replace("_", "-") if family is not None else "auto"
    if family_canonical in {
        "multinomial",
        "multinomial-logit",
        "categorical",
        "categorical-logit",
        "softmax",
    }:
        try:
            model_bytes = bytes(
                rust_module().fit_multinomial_formula_pyfunc(
                    headers,
                    rows,
                    formula,
                    1.0,   # init_lambda — warm-start seed; λ is REML-selected in Rust.
                    50,    # max_iter
                    1.0e-7,  # tol
                )
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        from ._model import MultinomialModel
        return MultinomialModel(
            _model_bytes=model_bytes,
            _training_table_kind=table_kind,
        )

    fisher_w = None
    if fisher_rao_w is not None:
        fisher_w = _normalize_fisher_rao_w(fisher_rao_w, n_rows=len(rows), dim=1)
    rust_config = dict(config or {})
    for key in (
        "response_geometry",
        "response_columns",
        "response_coordinates",
        "response_reference",
    ):
        rust_config.pop(key, None)
    # Persist the training-table container type into the model payload (the
    # single serialized source of truth) so the predict-time output-container
    # fallback for dict/list inputs survives save/load and dumps/loads. Without
    # this, a reloaded model silently returns dict instead of the original
    # container type for ambiguous prediction inputs (#394).
    rust_config["training_table_kind"] = table_kind
    resolved_precision_hyperpriors = _resolve_precision_hyperpriors(
        precision_hyperpriors, formula, headers, rows, rust_config.get("group_metadata")
    )
    payload = _build_fit_payload(
        family=family,
        offset=offset,
        weights=weights,
        transformation_normal=transformation_normal,
        transformation_normal_stage1=transformation_normal_stage1,
        survival_likelihood=survival_likelihood,
        baseline_target=baseline_target,
        baseline_scale=baseline_scale,
        baseline_shape=baseline_shape,
        baseline_rate=baseline_rate,
        baseline_makeham=baseline_makeham,
        z_column=z_column,
        link=link,
        logslope_formula=logslope_formula,
        frailty_kind=frailty_kind,
        frailty_sd=frailty_sd,
        hazard_loading=hazard_loading,
        scale_dimensions=scale_dimensions,
        adaptive_regularization=adaptive_regularization,
        firth=firth,
        noise_formula=noise_formula,
        noise_offset=noise_offset,
        flexible_link=flexible_link,
        precision_hyperpriors=resolved_precision_hyperpriors,
        latents=latents,
        penalties=penalties,
        smooths=smooths,
        config=rust_config or None,
    )
    try:
        model_bytes = bytes(
            rust_module().fit_table(headers, rows, formula, json.dumps(payload), fisher_w)
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    model = Model(_model_bytes=model_bytes, _training_table_kind=table_kind)
    # Surface any materialization advisories (e.g. an mgcv-style "k reduced to
    # the data support" note when a cr/cs/sz basis is capped) as warnings, so a
    # basis the fit silently adjusted is never silent to the caller (#1543).
    emit_inference_warnings(model.notes)
    return model


def fit_array(
    X: Any,
    Y: Any,
    formula: str,
    *,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    transformation_normal: bool | None = None,
    transformation_normal_stage1: CtnStage1 | Mapping[str, Any] | None = None,
    survival_likelihood: str | None = None,
    baseline_target: str | None = None,
    baseline_scale: float | None = None,
    baseline_shape: float | None = None,
    baseline_rate: float | None = None,
    baseline_makeham: float | None = None,
    z_column: str | None = None,
    link: str | None = None,
    logslope_formula: str | None = None,
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    noise_formula: str | None = None,
    noise_offset: str | None = None,
    flexible_link: bool | None = None,
    precision_hyperpriors: Any | None = None,
    latents: Mapping[str, Any] | None = None,
    penalties: Sequence[Any] | None = None,
    smooths: Mapping[Any, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Model:
    """Fit directly from numeric NumPy-compatible arrays.

    ``X`` is named ``x0``, ``x1``, ... at the formula boundary. A one-column
    ``Y`` is named from the formula response; multi-column ``Y`` is named
    ``y0``, ``y1``, ...

    Other keyword arguments have the same semantics as :func:`fit`, except
    response-geometry and shape-constraint helpers are available only on the
    table/formula entry point.

    Returns
    -------
    Model
        Fitted scalar model whose training table kind is recorded as
        ``"numpy"`` for positional-array prediction.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` row counts differ or required arrays are malformed.
    """
    X_arr = _numeric_matrix(X, "X")
    Y_arr = _numeric_matrix(Y, "Y")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError(
            f"X and Y row counts must match; got {X_arr.shape[0]} and {Y_arr.shape[0]}"
        )
    rust_config = dict(config or {})
    # See the matching note in `fit`: persist the training-table container type
    # (always "numpy" for the array entry point) so the predict-time
    # output-container fallback round-trips through save/load (#394).
    rust_config["training_table_kind"] = "numpy"
    resolved_precision_hyperpriors = _resolve_precision_hyperpriors(
        precision_hyperpriors, formula, [], [], rust_config.get("group_metadata")
    )
    payload = _build_fit_payload(
        family=family,
        offset=offset,
        weights=weights,
        transformation_normal=transformation_normal,
        transformation_normal_stage1=transformation_normal_stage1,
        survival_likelihood=survival_likelihood,
        baseline_target=baseline_target,
        baseline_scale=baseline_scale,
        baseline_shape=baseline_shape,
        baseline_rate=baseline_rate,
        baseline_makeham=baseline_makeham,
        z_column=z_column,
        link=link,
        logslope_formula=logslope_formula,
        frailty_kind=frailty_kind,
        frailty_sd=frailty_sd,
        hazard_loading=hazard_loading,
        scale_dimensions=scale_dimensions,
        adaptive_regularization=adaptive_regularization,
        firth=firth,
        noise_formula=noise_formula,
        noise_offset=noise_offset,
        flexible_link=flexible_link,
        precision_hyperpriors=resolved_precision_hyperpriors,
        latents=latents,
        penalties=penalties,
        smooths=smooths,
        config=rust_config or None,
    )
    try:
        model_bytes = bytes(
            rust_module().fit_array(X_arr, Y_arr, formula, json.dumps(payload))
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    model = Model(_model_bytes=model_bytes, _training_table_kind="numpy")
    emit_inference_warnings(model.notes)  # see fit(): never silently adjust a basis (#1543)
    return model


def load(path: str | Path) -> Any:
    """Load a fitted model previously written with :func:`gamfit.save`.

    Auto-detects format: JSON files containing a ``gamfit.ManifoldSAE/v1``
    schema header are returned as :class:`gamfit.ManifoldSAE`; everything
    else is treated as a binary :class:`Model` archive and dispatched to
    :func:`loads`.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the serialized model file.

    Returns
    -------
    Model or ManifoldSAE
        Fitted model ready for prediction.

    Examples
    --------
    >>> model = gamfit.load("model.gam")
    >>> model.predict(test_df)
    """
    raw = Path(path).read_bytes()
    # Cheap sniff: only ManifoldSAE payloads are JSON with the schema tag.
    head = raw[:256].lstrip()
    if head.startswith(b"{") and b"gamfit.ManifoldSAE" in raw[:512]:
        from ._sae_manifold import ManifoldSAE  # local import avoids cycle

        return ManifoldSAE.from_dict(json.loads(raw.decode("utf-8")))
    return loads(raw)


def save(model: Any, path: str | Path) -> None:
    """Write a fitted model to ``path``. Symmetric with :func:`gamfit.load`.

    Dispatches to ``model.save(path)`` for any object that exposes the method
    (covers :class:`Model` binary archives and :class:`ManifoldSAE` JSON
    payloads alike).
    """
    saver = getattr(model, "save", None)
    if not callable(saver):
        raise TypeError(
            f"gamfit.save: {type(model).__name__} has no .save(path) method"
        )
    saver(path)


def loads(model_bytes: bytes) -> Model:
    """Load a fitted :class:`Model` from an in-memory bytes payload.

    Parameters
    ----------
    model_bytes : bytes
        Raw serialized model produced by :meth:`Model.save` or
        :meth:`Model.saves`.

    Returns
    -------
    Model
        Fitted model ready for prediction.

    Raises
    ------
    GamError
        If the payload is malformed or incompatible with the current engine.

    Examples
    --------
    >>> with open("model.gam", "rb") as fh:
    ...     model = gamfit.loads(fh.read())
    """
    # Response-geometry payloads are a small JSON container (schema-tagged) that
    # embeds the constituent tangent `Model` archives plus the base point /
    # coordinate chart / geometry metadata (#2114). They are neither a scalar
    # `Model` archive nor a multinomial payload, so detect them first by the
    # schema tag and reconstruct a `ResponseGeometryModel`.
    head = model_bytes[:256].lstrip()
    if head.startswith(b"{") and b"gamfit.ResponseGeometryModel" in model_bytes[:512]:
        payload = json.loads(model_bytes.decode("utf-8"))
        if str(payload.get("schema", "")).startswith("gamfit.ResponseGeometryModel/"):
            return _reconstruct_response_geometry(payload)
    # Multinomial-logit payloads carry a different on-disk schema than the
    # scalar `Model` archive (`load_model` rejects them with a missing
    # `model_type` field). Detect them positively: only a genuine multinomial
    # payload deserialises through the multinomial-metadata FFI into a dict
    # carrying `class_levels`. If that succeeds, reconstruct a
    # `MultinomialModel`; otherwise fall through to the scalar `Model` path so a
    # genuinely malformed payload still raises the mapped GamError there.
    multinomial_metadata = None
    try:
        candidate = rust_module().multinomial_model_metadata_pyfunc(model_bytes)
        if isinstance(candidate, dict) and "class_levels" in candidate:
            multinomial_metadata = candidate
    except Exception:
        multinomial_metadata = None
    if multinomial_metadata is not None:
        from ._model import MultinomialModel  # local import avoids cycle

        # The multinomial payload predates the `training_table_kind` sniff used
        # by the scalar path; probe for it but tolerate the FFI rejecting the
        # multinomial schema, in which case the container falls back to None.
        try:
            training_table_kind = rust_module().saved_model_payload_string(
                model_bytes, "training_table_kind"
            )
        except Exception:
            training_table_kind = None
        return MultinomialModel(
            _model_bytes=model_bytes,
            _training_table_kind=training_table_kind,
        )
    try:
        rust_module().load_model(model_bytes)
        # Restore the training-table container type persisted in the payload so
        # the reloaded model reproduces the original predict-time
        # output-container fallback for dict/list inputs. `None` for older
        # payloads written before #394, which degrade to the "dict" fallback.
        training_table_kind = rust_module().saved_model_payload_string(
            model_bytes, "training_table_kind"
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return Model(_model_bytes=model_bytes, _training_table_kind=training_table_kind)


def _reconstruct_response_geometry(payload: Mapping[str, Any]) -> ResponseGeometryModel:
    """Rebuild a :class:`ResponseGeometryModel` from its JSON payload (#2114).

    Symmetric with :meth:`ResponseGeometryModel.to_dict`: each embedded tangent
    ``Model`` archive is reloaded through :func:`loads`, and the base point,
    coordinate chart, geometry label, and (optional) curvature summary are
    restored so the rebuilt model reproduces :meth:`ResponseGeometryModel.predict`.
    """
    import base64

    import numpy as np

    from ._response_geometry import SharedGaussianRemlTangentFit

    def _model_from_b64(encoded: str) -> Model:
        return loads(base64.b64decode(encoded.encode("ascii")))

    models = tuple(
        _model_from_b64(encoded) for encoded in payload.get("coordinate_models_b64", [])
    )

    shared_payload = payload.get("shared_tangent_fit")
    shared_fit: SharedGaussianRemlTangentFit | None = None
    if shared_payload is not None:
        template = _model_from_b64(shared_payload["template_model_b64"])
        coefficients = np.asarray(shared_payload["coefficients"], dtype=float)
        fit = dict(shared_payload.get("fit", {}))
        for key in ("coefficients", "fitted", "sigma2", "lambdas", "edf"):
            if fit.get(key) is not None:
                fit[key] = np.asarray(fit[key], dtype=float)
        if fit.get("reml_score") is not None:
            fit["reml_score"] = float(fit["reml_score"])
        shared_fit = SharedGaussianRemlTangentFit(
            template_model=template,
            coefficients=coefficients,
            fit=fit,
        )

    return ResponseGeometryModel(
        models=models,
        response_geometry=str(payload["response_geometry"]),
        response_columns=tuple(payload["response_columns"]),
        base_point=np.asarray(payload["base_point"], dtype=float),
        coordinates=str(payload["coordinates"]),
        reference=int(payload.get("reference", -1)),
        training_table_kind=payload.get("training_table_kind"),
        shared_tangent_fit=shared_fit,
        curvature=payload.get("curvature"),
    )


def validate_formula(
    data: Any,
    formula: str,
    *,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    transformation_normal: bool | None = None,
    transformation_normal_stage1: CtnStage1 | Mapping[str, Any] | None = None,
    survival_likelihood: str | None = None,
    baseline_target: str | None = None,
    baseline_scale: float | None = None,
    baseline_shape: float | None = None,
    baseline_rate: float | None = None,
    baseline_makeham: float | None = None,
    z_column: str | None = None,
    link: str | None = None,
    logslope_formula: str | None = None,
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    noise_formula: str | None = None,
    noise_offset: str | None = None,
    flexible_link: bool | None = None,
    config: dict[str, Any] | None = None,
) -> FormulaValidation:
    """Validate a formula against a dataset without fitting.

    Accepts the scalar formula pipeline kwargs present in this function's
    signature, with the same semantics as :func:`fit`. Validation deliberately
    does not accept fit-only objects such as ``constraints``, ``latents``,
    ``penalties``, ``smooths``, ``precision_hyperpriors``, or
    ``response_geometry``.

    Returns
    -------
    FormulaValidation
        Structured validation diagnostics from the Rust parser/materializer.
    """
    headers, rows, _table_kind = normalize_table(data)
    rust_config = dict(config or {})
    for key in (
        "response_geometry",
        "response_columns",
        "response_coordinates",
        "response_reference",
    ):
        rust_config.pop(key, None)
    payload = _build_fit_payload(
        family=family,
        offset=offset,
        weights=weights,
        transformation_normal=transformation_normal,
        transformation_normal_stage1=transformation_normal_stage1,
        survival_likelihood=survival_likelihood,
        baseline_target=baseline_target,
        baseline_scale=baseline_scale,
        baseline_shape=baseline_shape,
        baseline_rate=baseline_rate,
        baseline_makeham=baseline_makeham,
        z_column=z_column,
        link=link,
        logslope_formula=logslope_formula,
        frailty_kind=frailty_kind,
        frailty_sd=frailty_sd,
        hazard_loading=hazard_loading,
        scale_dimensions=scale_dimensions,
        adaptive_regularization=adaptive_regularization,
        firth=firth,
        noise_formula=noise_formula,
        noise_offset=noise_offset,
        flexible_link=flexible_link,
        precision_hyperpriors=None,
        latents=None,
        penalties=None,
        smooths=None,
        config=rust_config or None,
    )
    try:
        raw = rust_module().validate_formula_json(
            headers,
            rows,
            formula,
            json.dumps(payload),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return FormulaValidation.from_dict(json.loads(raw))


def explain_error(exc: BaseException) -> str:
    """Return a short, actionable hint describing how to recover from ``exc``.

    Inspects the exception type and returns a one-line suggestion tailored to
    the gamfit error hierarchy (:class:`FormulaError`,
    :class:`SchemaMismatchError`, :class:`PredictionError`, :class:`GamError`,
    :class:`RustExtensionUnavailableError`). Unrecognized exceptions fall back
    to a generic message.

    Parameters
    ----------
    exc : BaseException
        The exception caught from a gamfit call.

    Returns
    -------
    str
        Human-readable remediation hint.

    Examples
    --------
    >>> try:
    ...     gamfit.fit(df, "y ~ s(nope)")
    ... except gamfit.GamError as exc:
    ...     print(gamfit.explain_error(exc))
    Check the formula syntax and confirm every referenced column exists.
    """
    if isinstance(exc, RustExtensionUnavailableError):
        return "Build the extension with maturin before calling Rust-backed APIs."
    from ._exceptions import (
        ColumnNotFoundError,
        FormulaError,
        GamError,
        PredictionError,
        SchemaMismatchError,
    )

    # Column-not-found is a *typed* error: the Rust FFI boundary raises
    # `ColumnNotFoundError` (a `FormulaError` subclass) with structured
    # attributes attached by `workflow_error_to_pyerr` — `column`, `role`,
    # `available`, `similar`, `tsv_hint`. Read them directly instead of
    # parsing the formatted message text (the prior `_column_not_found_hint`
    # regex was the wrong shape; see issue #305 / #343).
    if isinstance(exc, ColumnNotFoundError):
        column = getattr(exc, "column", None)
        available = getattr(exc, "available", None)
        if column is not None and available is not None:
            return (
                f"Column {column!r} is referenced by the formula but is not "
                f"in the input table. Available columns: "
                f"[{', '.join(available)}]. Fix the formula or add the "
                f"column to the data."
            )
        if column is not None:
            return (
                f"Column {column!r} referenced by the formula is not "
                f"present in the input table. Fix the formula or add the "
                f"column to the data."
            )
        # The exception class itself encodes the failure mode even when
        # the per-instance enrichment is unavailable (e.g. a future PyO3
        # tightening prevented attribute attachment — see the unraisable
        # warning emitted from the boundary). Fall back to the typed
        # FormulaError hint rather than the unhelpful generic message.

    if isinstance(exc, FormulaError):
        return "Check the formula syntax and confirm every referenced column exists."
    if isinstance(exc, SchemaMismatchError):
        return "Compare the serving data with the training schema using model.check(...)."
    if isinstance(exc, PredictionError):
        return "Prediction failed. Validate the new data and confirm the fitted model is supported by the Python binding."
    if isinstance(exc, GamError):
        return "The Rust engine returned an error. Inspect the exception message for the underlying failure detail."
    return "Unexpected error. Inspect the full traceback and the original exception message."


def bspline_basis(
    t: Any,
    knots: Any = None,
    *,
    degree: int = 3,
    periodic: bool = False,
) -> Any:
    """Evaluate the Rust B-spline basis as a NumPy array.

    ``knots`` may be:

    * ``None`` — auto-derive a clamped knot vector with quantile-spaced
      interior knots inferred from ``t``.
    * an ``int`` ``K`` — auto-derive with ``K`` interior knots.
    * an array-like — used verbatim (must be a valid clamped knot vector).
    """
    import numpy as np

    degree_i = int(degree)
    if degree_i < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    t_np = _numeric_vector(t, "t")
    knots_np, eff_degree, _shrunk = _resolve_knots(knots, t_np, label="knots", degree=degree_i)
    try:
        return np.asarray(
            rust_module().bspline_basis(
                t_np,
                knots_np,
                eff_degree,
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def bspline_basis_derivative(
    t: Any,
    knots: Any = None,
    *,
    degree: int = 3,
    order: int = 1,
    periodic: bool = False,
) -> Any:
    """Evaluate derivatives of the Rust B-spline basis as a NumPy array.

    ``knots`` accepts ``None`` / ``int`` / array — see :func:`bspline_basis`.
    """
    import numpy as np

    degree_i = int(degree)
    order_i = int(order)
    if degree_i < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    if order_i < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    t_np = _numeric_vector(t, "t")
    knots_np, eff_degree, _shrunk = _resolve_knots(knots, t_np, label="knots", degree=degree_i)
    try:
        return np.asarray(
            rust_module().bspline_basis_derivative(
                t_np,
                knots_np,
                eff_degree,
                order_i,
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def duchon_basis(
    points: Any,
    centers: Any = None,
    *,
    m: int = 2,
    periodic_per_axis: Any = None,
    length_scale: float | None = None,
    nullspace_order: str = "linear",
    power: float | None = None,
) -> Any:
    """Evaluate the Duchon m-spline basis at ``points`` against ``centers``.

    Works for any input dimensionality ``d`` ≥ 1.

    Parameters
    ----------
    points : array-like of shape (N, d) — N evaluation points in d-dim.
        For 1D, pass shape (N, 1) or a 1D array (auto-promoted).
    centers : array-like of shape (K, d), or ``None`` (auto: K=10 quantile
        centers for d=1), or an ``int`` K (auto-quantile centers, d=1 only).
    m : int, default 2 — spline order.
    periodic_per_axis : sequence of bool of length d, optional. For ``d=1``
        the Bernoulli-Green builder is used (true Green's function on the
        circle). For ``d ≥ 2`` with any periodic axis, the multi-D
        mixed-periodicity builder uses cylinder/torus chord distance
        ``d_j(x, y) = (P_j/π) sin(π(x − y)/P_j)`` on periodic axes and
        Euclidean distance on non-periodic axes; per-axis periods ``P_j``
        are auto-derived from the centers' span along each periodic axis.
    length_scale : optional positive float. ``None`` (default) selects the
        scale-free pure Duchon spectrum ``‖w‖^(2(p+s))``. A positive value
        enables the hybrid (Matérn-blended) spectrum
        ``‖w‖^(2p) · (κ² + ‖w‖²)^s`` with ``κ = 1/length_scale``. The
        hybrid kernel keeps the polynomial nullspace order **linear in d**
        (a single dim+1 column block), letting the same basis scale cleanly
        to d=8, 16, 32, 64 without ratcheting the nullspace.
    nullspace_order : string, default ``"linear"``. ``"zero"`` (constant nullspace),
        ``"linear"`` (constant + linear), or ``"degree<k>"`` for k ≥ 2.
    power : optional float. Riesz spectral power ``s``. ``None`` (default)
        auto-resolves the minimum admissible ``s`` for the requested
        ``nullspace_order`` and dimension (matches the formula API).

    Returns
    -------
    ndarray of shape (N, K)
    """
    import numpy as np

    pts_np = np.asarray(points, dtype=float)
    if pts_np.ndim == 1:
        pts_np = pts_np.reshape(-1, 1)
    if pts_np.ndim != 2:
        raise ValueError(f"points must be 1D or 2D, got {pts_np.ndim}D")
    if pts_np.shape[0] == 0 or pts_np.shape[1] == 0:
        raise ValueError("points cannot be empty")
    if not np.all(np.isfinite(pts_np)):
        raise ValueError("points must contain only finite values")
    d = pts_np.shape[1]
    if centers is None or isinstance(centers, int):
        if d != 1:
            raise ValueError(f"auto centers only supported for d=1, got d={d}")
        ctrs_np = _resolve_centers(centers, pts_np[:, 0], label="centers").reshape(-1, 1)
    else:
        ctrs_np = np.asarray(centers, dtype=float)
        if ctrs_np.ndim == 1:
            ctrs_np = ctrs_np.reshape(-1, 1)
        if ctrs_np.ndim != 2:
            raise ValueError(f"centers must be 1D or 2D, got {ctrs_np.ndim}D")
        if ctrs_np.shape[0] == 0 or ctrs_np.shape[1] == 0:
            raise ValueError("centers cannot be empty")
        if ctrs_np.shape[1] != d:
            raise ValueError(
                f"points has d={d} but centers has d={ctrs_np.shape[1]}"
            )
        if not np.all(np.isfinite(ctrs_np)):
            raise ValueError("centers must contain only finite values")

    periodic_arg = (
        None if periodic_per_axis is None
        else [bool(p) for p in periodic_per_axis]
    )
    if periodic_arg is not None and len(periodic_arg) != d:
        raise ValueError(
            f"periodic_per_axis must have length matching point dim ({d}), got {len(periodic_arg)}"
        )
    m_i = int(m)
    if m_i < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    if length_scale is not None and (
        not math.isfinite(float(length_scale)) or float(length_scale) <= 0.0
    ):
        raise ValueError("length_scale must be finite and > 0")
    if power is not None and not math.isfinite(float(power)):
        raise ValueError("power must be finite")
    try:
        return np.asarray(
            rust_module().duchon_basis(
                pts_np,
                ctrs_np,
                m_i,
                periodic_arg,
                None if length_scale is None else float(length_scale),
                str(nullspace_order),
                None if power is None else float(power),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def matern_basis(
    points: Any,
    centers: Any,
    *,
    length_scale: float = 1.0,
    nu: str = "3/2",
    aniso_log_scales: Any = None,
) -> Any:
    """Evaluate the Matérn kernel basis at ``points`` against ``centers``.

    Pure-Rust forward, exposed through the same PyFFI surface as
    :func:`duchon_basis`. ``points`` is ``(N, d)``, ``centers`` is ``(K, d)``,
    and the returned design is ``(N, K)`` (NumPy float64).

    Parameters
    ----------
    points : array-like ``(N, d)`` — evaluation locations.
    centers : array-like ``(K, d)`` — kernel centers.
    length_scale : positive float — global Matérn range.
    nu : str — half-integer smoothness order: ``"1/2"``, ``"3/2"``,
        ``"5/2"``, ``"7/2"``, or ``"9/2"``.
    aniso_log_scales : optional length-d sequence of per-axis log-scale
        contrasts (sum-to-zero). ``None`` is isotropic.
    """
    import numpy as np

    pts_np = np.asarray(points, dtype=float)
    if pts_np.ndim == 1:
        pts_np = pts_np.reshape(-1, 1)
    if pts_np.ndim != 2:
        raise ValueError(f"points must be 1D or 2D, got {pts_np.ndim}D")
    ctrs_np = np.asarray(centers, dtype=float)
    if ctrs_np.ndim == 1:
        ctrs_np = ctrs_np.reshape(-1, 1)
    if ctrs_np.ndim != 2:
        raise ValueError(f"centers must be 1D or 2D, got {ctrs_np.ndim}D")
    if pts_np.shape[1] != ctrs_np.shape[1]:
        raise ValueError(
            f"points has d={pts_np.shape[1]} but centers has d={ctrs_np.shape[1]}"
        )
    if not np.all(np.isfinite(pts_np)) or not np.all(np.isfinite(ctrs_np)):
        raise ValueError("matern_basis requires finite points and centers")
    aniso = (
        None
        if aniso_log_scales is None
        else np.asarray(aniso_log_scales, dtype=float)
    )
    try:
        return np.asarray(
            rust_module().matern_basis(
                pts_np,
                ctrs_np,
                float(length_scale),
                str(nu),
                aniso,
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def sphere_basis(
    points: Any,
    n_centers: int,
    *,
    penalty_order: int = 2,
    kernel: str = "sobolev",
    radians: bool = False,
) -> tuple[Any, Any]:
    """Build a spherical-spline (S²) design and penalty matrix.

    Parameters
    ----------
    points : array-like of shape ``(N, 2)`` — latitude, longitude.
    n_centers : Wahba center count (``kernel='sobolev' | 'pseudo'``) or
        truncation degree ``L`` for ``kernel='harmonic'`` (basis dim
        ``L*(L+2)``).
    penalty_order : roughness order ``m ∈ {1,2,3,4}``. Default ``2``.
    kernel : one of ``'sobolev'``, ``'pseudo'``, ``'harmonic'``.
    radians : default ``False`` (degrees). True for radians.

    Returns
    -------
    (design (N, K), penalty (K, K)) as ndarrays.
    """
    import numpy as np

    pts_np = np.asarray(points, dtype=float)
    if pts_np.ndim != 2 or pts_np.shape[1] != 2:
        raise ValueError(
            f"sphere_basis expects points of shape (N, 2); got {pts_np.shape}"
        )
    if pts_np.shape[0] == 0:
        raise ValueError("sphere_basis: points cannot be empty")
    if not np.all(np.isfinite(pts_np)):
        raise ValueError("sphere_basis: points contains NaN/Inf")
    n_centers_i = int(n_centers)
    penalty_order_i = int(penalty_order)
    if n_centers_i <= 0:
        raise ValueError(f"n_centers must be positive, got {n_centers}")
    if penalty_order_i not in (1, 2, 3, 4):
        raise ValueError("penalty_order must be one of 1, 2, 3, or 4")
    lat = pts_np[:, 0]
    if radians:
        bound = float(np.pi / 2.0)
        if np.any(lat < -bound - 1e-9) or np.any(lat > bound + 1e-9):
            raise ValueError(
                "sphere_basis: latitude (radians) must lie in [-π/2, π/2]"
            )
    else:
        if np.any(lat < -90.0 - 1e-9) or np.any(lat > 90.0 + 1e-9):
            raise ValueError(
                "sphere_basis: latitude (degrees) must lie in [-90, 90]"
            )
    try:
        design, penalty = rust_module().sphere_basis(
            pts_np,
            n_centers_i,
            penalty_order_i,
            str(kernel),
            bool(radians),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(design, dtype=float), np.asarray(penalty, dtype=float)


def sphere_basis_jet(
    points: Any,
    n_centers: int,
    *,
    penalty_order: int = 2,
    kernel: str = "sobolev",
    radians: bool = False,
) -> Any:
    """Analytic design jet ``∂Φ/∂(lat, lon)`` of the spherical-spline basis.

    Mirrors :func:`sphere_basis` (auto Wahba farthest-point centers, or
    harmonic degree ``L = n_centers``) and returns the exact analytic
    derivative of its design matrix — no finite differences.

    Parameters
    ----------
    points : array-like of shape ``(N, 2)`` — latitude, longitude.
    n_centers : Wahba center count (``kernel='sobolev' | 'pseudo'``) or
        harmonic truncation degree ``L``.
    penalty_order : roughness order ``m ∈ {1,2,3,4}``. Default ``2``.
    kernel : one of ``'sobolev'``, ``'pseudo'``, ``'harmonic'``.
    radians : default ``False`` (degrees). True for radians.

    Returns
    -------
    ndarray of shape ``(N, K, 2)`` — ``K`` matches the :func:`sphere_basis`
    design column count; the last axis is ``(∂col/∂lat, ∂col/∂lon)`` in the
    same angular units as the input.
    """
    import numpy as np

    pts_np = np.asarray(points, dtype=float)
    if pts_np.ndim != 2 or pts_np.shape[1] != 2:
        raise ValueError(
            f"sphere_basis_jet expects points of shape (N, 2); got {pts_np.shape}"
        )
    if pts_np.shape[0] == 0:
        raise ValueError("sphere_basis_jet: points cannot be empty")
    if not np.all(np.isfinite(pts_np)):
        raise ValueError("sphere_basis_jet: points contains NaN/Inf")
    n_centers_i = int(n_centers)
    penalty_order_i = int(penalty_order)
    if n_centers_i <= 0:
        raise ValueError(f"n_centers must be positive, got {n_centers}")
    if penalty_order_i not in (1, 2, 3, 4):
        raise ValueError("penalty_order must be one of 1, 2, 3, or 4")
    lat = pts_np[:, 0]
    if radians:
        bound = float(np.pi / 2.0)
        if np.any(lat < -bound - 1e-9) or np.any(lat > bound + 1e-9):
            raise ValueError(
                "sphere_basis_jet: latitude (radians) must lie in [-π/2, π/2]"
            )
    else:
        if np.any(lat < -90.0 - 1e-9) or np.any(lat > 90.0 + 1e-9):
            raise ValueError(
                "sphere_basis_jet: latitude (degrees) must lie in [-90, 90]"
            )
    try:
        jet = rust_module().sphere_basis_jet(
            pts_np,
            n_centers_i,
            penalty_order_i,
            str(kernel),
            bool(radians),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(jet, dtype=float)


def smoothness_penalty(
    knots: Any,
    *,
    degree: int = 3,
    order: int = 2,
) -> tuple[Any, Any]:
    """Return ``(S, null_basis)`` for the Rust B-spline difference penalty.

    ``knots`` must be a knot vector here — auto-derivation requires
    sample positions, which this penalty constructor does not take. Build
    one with :func:`bspline_basis`'s defaults (or pass any 1D array).
    """
    import numpy as np

    degree_i = int(degree)
    order_i = int(order)
    if degree_i < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    if order_i <= 0:
        raise ValueError(f"order must be positive, got {order}")
    try:
        penalty, null_basis = rust_module().smoothness_penalty(
            _numeric_vector(knots, "knots"),
            degree_i,
            order_i,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(penalty, dtype=float), np.asarray(null_basis, dtype=float)


def mechanism_sparsity_jacobian(weight: float, epsilon: float, w: Any) -> tuple[float, Any]:
    """Evaluate the Rust mechanism-sparsity Jacobian penalty and gradient."""

    try:
        return rust_module().mechanism_sparsity_jacobian(float(weight), float(epsilon), w)
    except Exception as exc:
        raise map_exception(exc) from exc


def conditional_prior_ivae(
    weight: float,
    t: Any,
    mean: Any,
    scale: Any,
) -> tuple[float, Any]:
    """Evaluate the Rust iVAE conditional-prior penalty and gradient."""

    try:
        return rust_module().conditional_prior_ivae(float(weight), t, mean, scale)
    except Exception as exc:
        raise map_exception(exc) from exc


def derive_ivae_aux_scale(
    aux: Any,
    log_amplitude: float,
    frequency_scale: float,
) -> Any:
    """Derive the Rust iVAE conditional-prior scale from the auxiliary table."""

    try:
        return rust_module().derive_ivae_aux_scale(
            aux, float(log_amplitude), float(frequency_scale)
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def periodic_spline_curve_basis(
    t: Any,
    n_knots: int,
    *,
    degree: int = 3,
    penalty_order: int = 2,
) -> tuple[Any, Any]:
    """Return ``(basis, penalty)`` for a closed cyclic B-spline basis on ``t``.

    The basis is uniform on ``[0, 1)`` and periodic (wraps cleanly). The
    penalty is the cyclic ``order``-th difference penalty on the ``n_knots``
    cyclic control points. To fit a closed curve ``t -> R^d``, regress a
    ``(N, d)`` response against the returned ``(N, K)`` basis with the
    returned ``(K, K)`` penalty.

    Parameters
    ----------
    t : array-like of shape ``(N,)``. Values are taken modulo 1.
    n_knots : number of cyclic control points / basis columns.
    degree : B-spline degree. Default 3.
    penalty_order : order of the cyclic difference penalty. Default 2.

    Returns
    -------
    (ndarray of shape (N, n_knots), ndarray of shape (n_knots, n_knots))
    """
    import numpy as np

    n_knots_i = int(n_knots)
    degree_i = int(degree)
    penalty_order_i = int(penalty_order)
    if n_knots_i <= 0:
        raise ValueError(f"n_knots must be positive, got {n_knots}")
    if degree_i < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    if penalty_order_i <= 0:
        raise ValueError(f"penalty_order must be positive, got {penalty_order}")
    t_np = _numeric_vector(t, "t")
    try:
        basis, penalty = rust_module().periodic_spline_curve_basis(
            t_np,
            n_knots_i,
            degree_i,
            penalty_order_i,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(basis, dtype=float), np.asarray(penalty, dtype=float)


def duchon_function_norm_penalty(
    centers: Any,
    *,
    m: int = 2,
    period: float | None = None,
    periodic_per_axis: Any = None,
    length_scale: float | None = None,
    nullspace_order: str = "linear",
    power: float | None = None,
) -> Any:
    """Single-λ smoothness penalty matrix for the cubic (r³) Duchon basis.

    This returns ONE ``(K × K)`` SPD penalty for the cubic polyharmonic basis
    on ``centers`` — convenient when you need a single REML ``λ`` for a
    manually built radial-basis design. It is **not** the classical Duchon
    native seminorm, and it is **not** the structural smoother the formula API
    (:class:`gamfit.smooth.Duchon`, ``basis_kind="duchon"``) fits. That object
    penalizes amplitude, slope, and curvature with three *separate* REML
    ``λ``s and only frees the global mean; to get it, use the formula smooth
    rather than feeding this single matrix back as a penalty.

    Parameters
    ----------
    centers : array-like
        Control points with shape ``(K, d)``.
    m : int, default 2
        Spline ORDER — selects the unpenalized polynomial nullspace, not the
        spectral power ``s``.
    periodic_per_axis : sequence of bool of length d, optional
        Per-axis periodicity. ``d=1`` uses the Bernoulli-Green Gram on
        the circle; ``d ≥ 2`` with any periodic axis uses the multi-D
        mixed-periodicity radial polyharmonic kernel evaluated at the
        cylinder/torus chord distance (see :func:`duchon_basis` for the
        per-axis formula).
    length_scale : optional positive float. ``None`` (default) selects the
        scale-free pure Duchon spectrum. A positive value enables the
        hybrid (Matérn-blended) spectrum, keeping the polynomial nullspace
        order **linear in d** for clean scaling to d=8, 16, 32, 64.
    nullspace_order : string, default ``"linear"``. ``"zero"``, ``"linear"``,
        or ``"degree<k>"`` (k ≥ 2).
    power : optional float. Riesz spectral power ``s``. ``None`` (default)
        auto-resolves the minimum admissible ``s``.

    Returns
    -------
    ndarray of shape (K, K) — SPD penalty matrix.
    """
    import numpy as np

    ctrs = np.asarray(centers, dtype=float)
    if ctrs.ndim != 2:
        raise ValueError(f"centers must be 2D with shape (K, d), got {ctrs.ndim}D")
    d = ctrs.shape[1]
    if ctrs.size == 0:
        raise ValueError("centers cannot be empty")
    if not np.all(np.isfinite(ctrs)):
        raise ValueError("centers must contain only finite values")
    m_i = int(m)
    if m_i < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    if period is not None and (
        not math.isfinite(float(period)) or float(period) <= 0.0
    ):
        raise ValueError("period must be finite and > 0")
    if length_scale is not None and (
        not math.isfinite(float(length_scale)) or float(length_scale) <= 0.0
    ):
        raise ValueError("length_scale must be finite and > 0")
    if power is not None and not math.isfinite(float(power)):
        raise ValueError("power must be finite")

    per_list: list[bool] | None = None
    if periodic_per_axis is not None:
        per_list = [bool(p) for p in periodic_per_axis]
        if len(per_list) != d:
            raise ValueError(
                f"periodic_per_axis must have length matching centers dim ({d}), "
                f"got {len(per_list)}"
            )
    try:
        # Pass the optional arguments by keyword. The binding's positional
        # layout is ``(centers, m, period, periodic_per_axis, length_scale,
        # nullspace_order, power)``; an obsolete ``periodic: bool`` third
        # positional (once passed here as ``False``) was dropped from the
        # binding when ``period``/``periodic_per_axis`` superseded it, but the
        # call site kept the stray ``False`` and overflowed the arity by one
        # (gam#880). Keywords make this call robust to any future reordering of
        # the binding's parameters.
        penalty = rust_module().duchon_function_norm_penalty(
            ctrs,
            m_i,
            period=float(period) if period is not None else None,
            periodic_per_axis=per_list,
            length_scale=None if length_scale is None else float(length_scale),
            nullspace_order=str(nullspace_order),
            power=None if power is None else float(power),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(penalty, dtype=float)


def gaussian_weighted_ridge(
    X: Any,
    Y: Any,
    penalty: Any,
    weights: Any,
    *,
    ridge_lambda: float,
) -> tuple[Any, Any]:
    """Closed-form Gaussian row-weighted ridge on NumPy-compatible arrays.

    ``weights`` are likelihood row weights. They are not a multiplicative
    gate on the mean/design row.
    """
    import numpy as np

    try:
        coefficients, fitted = rust_module().gaussian_weighted_ridge_array(
            _numeric_matrix(X, "X"),
            _numeric_matrix(Y, "Y"),
            _numeric_matrix(penalty, "penalty"),
            _numeric_vector(weights, "weights"),
            float(ridge_lambda),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(coefficients, dtype=float), np.asarray(fitted, dtype=float)


def gaussian_weighted_ridge_batch(
    X: Any,
    Y: Any,
    penalty: Any,
    weights: Any,
    *,
    ridge_lambda: float,
    row_counts: Any | None = None,
) -> tuple[Any, Any]:
    """Batched closed-form Gaussian row-weighted ridge.

    ``X`` has shape ``(K, Nmax, M)``, ``Y`` has shape ``(K, Nmax, D)``, and
    ``weights`` has shape ``(K, Nmax)``. ``row_counts`` optionally marks the
    active row prefix for each problem in a padded ragged batch.
    """
    import numpy as np

    try:
        coefficients, fitted = rust_module().gaussian_weighted_ridge_batch(
            _numeric_tensor3(X, "X"),
            _numeric_tensor3(Y, "Y"),
            _numeric_matrix(penalty, "penalty"),
            _numeric_matrix(weights, "weights"),
            float(ridge_lambda),
            None if row_counts is None else _index_vector(row_counts, "row_counts"),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(coefficients, dtype=float), np.asarray(fitted, dtype=float)


def gaussian_reml_fit(
    x: Any,
    y: Any,
    penalty: Any,
    *,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
    geodesic_acceleration: bool = False,
) -> dict[str, Any]:
    """Fit a closed-form Gaussian REML problem from NumPy-compatible arrays.

    Parameters
    ----------
    geodesic_acceleration:
        Enable the Transtrum-Sethna geodesic-acceleration second-order
        correction in the inner Newton / Levenberg-Marquardt solver
        (Transtrum & Sethna, 2011, "Improvements to the Levenberg-Marquardt
        algorithm for nonlinear least-squares minimization"). The standard
        LM step ``δp = −(H + λ_lm·diag(H))⁻¹ g`` is augmented with a
        second-order correction ``δp₂ = −(H + λ_lm·diag(H))⁻¹ K`` where
        ``K`` is a central-difference estimate of the directional second
        derivative of the gradient along ``δp``; the correction is
        accepted only when ``‖δp₂‖ ≤ 0.75 · ‖δp‖``. Most useful for fits
        with near-singular Hessians (latent-coordinate fits, near-collinear
        bases). Default ``False`` until validated. *Note:* the closed-form
        Gaussian-identity path used by this function does not run an inner
        Newton loop, so this flag is accepted for API parity with the
        PIRLS-based fits and is a no-op here; it is honored by the
        formula-based ``gamfit.fit(...)`` entrypoint and any GLM-family
        fits that drive the inner Newton solver.
    """
    import numpy as np

    try:
        out = rust_module().gaussian_reml_fit(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            _numeric_matrix(penalty, "penalty"),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_backward(
    x: Any,
    y: Any,
    penalty: Any,
    *,
    grad_lambda: float = 0.0,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: float = 0.0,
    grad_edf: float = 0.0,
    forward_state: dict[str, Any] | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run the analytic VJP for ``gaussian_reml_fit`` outputs."""
    if forward_state is not None:
        check_forward_state(forward_state, name="gaussian_reml_fit_backward")
    try:
        out = rust_module().gaussian_reml_fit_backward(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            _numeric_matrix(penalty, "penalty"),
            float(grad_lambda),
            None
            if grad_coefficients is None
            else _numeric_matrix(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            float(grad_reml_score),
            float(grad_edf),
            forward_state,
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return coerce_grad_payload(out)


def gaussian_reml_fit_batched(
    x: Any,
    y: Any,
    row_offsets: Any,
    penalty: Any,
    *,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit K closed-form Gaussian REML problems packed by row offsets."""
    import numpy as np

    try:
        out = rust_module().gaussian_reml_fit_batched(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            _index_vector(row_offsets, "row_offsets"),
            _numeric_matrix(penalty, "penalty"),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_batched_backward(
    x: Any,
    y: Any,
    row_offsets: Any,
    penalty: Any,
    *,
    grad_lambda: Any | None = None,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: Any | None = None,
    grad_edf: Any | None = None,
    forward_state: dict[str, Any] | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run packed ragged analytic VJPs for ``gaussian_reml_fit_batched``."""
    if forward_state is not None:
        check_forward_state(forward_state, name="gaussian_reml_fit_batched_backward")
    offsets = _index_vector(row_offsets, "row_offsets")
    batch = int(offsets.size - 1)
    try:
        out = rust_module().gaussian_reml_fit_batched_backward(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            offsets,
            _numeric_matrix(penalty, "penalty"),
            _optional_batch_vector(grad_lambda, batch, "grad_lambda"),
            None
            if grad_coefficients is None
            else _numeric_tensor3(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            _optional_batch_vector(grad_reml_score, batch, "grad_reml_score"),
            _optional_batch_vector(grad_edf, batch, "grad_edf"),
            forward_state,
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return coerce_grad_payload(out)


def _resolve_position_basis_inputs(
    t: Any,
    basis_kind: str | None,
    knots_or_centers: Any,
    penalty: Any | None,
    *,
    basis: str | None,
    basis_order: int | None,
    periodic: bool,
    period: float | None = None,
) -> tuple[str, str, int, Any, Any, Any, float | None]:
    """Resolve the shared position-basis FFI inputs for every positions face.

    All four position-based Gaussian REML entrypoints (forward / backward /
    batched / batched-backward) opened with this identical preamble. Returns
    ``(display_kind, effective_kind, order, t_np, knots_np, penalty_np,
    eff_period)``. ``eff_period`` is the domain-wrap period the basis and penalty
    must share: the explicit ``period`` when given, else (for periodic Duchon) a
    value auto-derived from the resolved knots so the half-open grid wraps
    cleanly (gam#580). The caller passes ``eff_period`` to the FFI basis build so
    basis and penalty stay consistent.
    """
    import numpy as np

    display_kind = str(
        basis if basis is not None else basis_kind if basis_kind is not None else "bspline"
    )
    effective_kind, order, _ = _normalize_position_basis(display_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    kind_norm = str(display_kind).strip().lower().replace("_", "").replace("-", "")
    if periodic and effective_kind == "bspline" and (
        knots_or_centers is None
        or (isinstance(knots_or_centers, int) and not isinstance(knots_or_centers, bool))
    ):
        knots_np = _resolve_periodic_position_bspline_knots(
            knots_or_centers,
            t_np,
            degree=order,
            period=period,
        )
        eff_order = order
    else:
        knots_np, eff_order, _shrunk = _resolve_basis_locations(
            knots_or_centers,
            t_np,
            basis_kind=effective_kind,
            label="knots_or_centers",
            degree=order,
        )
    # Resolve the effective wrap period for periodic Duchon. The period is the
    # domain wrap, not the knot span: on a half-open grid the knots span only
    # (period − one_spacing). When the caller gives no explicit period, derive it
    # as span + one mean knot spacing so points near the two ends are a single
    # spacing apart across the wrap (an undersized period gave a non-PSD Gram —
    # gam#580). Non-periodic / non-Duchon bases ignore this.
    eff_period = period
    if periodic and period is None and kind_norm in {"duchon", "duchonspline", "thinplate", "thinplatespline", "tps"}:
        k = np.asarray(knots_np, dtype=float)
        if k.size >= 2:
            span = float(k.max() - k.min())
            mean_spacing = span / max(k.size - 1, 1)
            eff_period = span + mean_spacing
    # Auto-knot derivation may downgrade the degree for small n (#340); the
    # resolved knot vector is clamped for ``eff_order``, so the penalty and the
    # downstream FFI basis build must both use the effective order to stay
    # consistent with it.
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=display_kind,
        basis_order=eff_order,
        periodic=periodic,
        period=eff_period,
    )
    return display_kind, effective_kind, eff_order, t_np, knots_np, penalty_np, eff_period


def _resolve_periodic_position_bspline_knots(
    count: Any,
    t_arr: Any,
    *,
    degree: int,
    period: float | None,
) -> Any:
    """Resolve count-based periodic position B-splines to K cyclic controls.

    ``gaussian_reml_fit_positions(..., periodic=True)`` documents integer
    ``knots_or_centers`` as a basis count. The Rust position kernel accepts an
    explicit half-open knot/control grid and derives ``num_basis = len(grid)-1``.
    Therefore the public count K must become K+1 grid endpoints, not the open
    B-spline auto-knot vector used by non-periodic fits.
    """
    import numpy as np

    degree_i = int(degree)
    k = int(_DEFAULT_BASIS_K if count is None else count)
    if k < degree_i + 1:
        raise ValueError(
            "periodic B-spline position basis count must be at least "
            f"degree + 1 (got {k} for degree {degree_i})"
        )
    if period is None:
        raise ValueError("periodic B-spline position fits require a finite positive period")
    period_f = float(period)
    if not np.isfinite(period_f) or period_f <= 0.0:
        raise ValueError(f"period must be finite and positive, got {period}")
    t_np = np.asarray(t_arr, dtype=float)
    if t_np.size == 0:
        raise ValueError("t must contain at least one value")
    origin = float(np.min(t_np))
    return np.linspace(origin, origin + period_f, k + 1, dtype=float)


def gaussian_reml_fit_positions(
    t: Any,
    y: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit closed-form Gaussian REML from 1D positions and an internal basis.

    ``knots_or_centers`` may be ``None``, an ``int`` (basis count), or an
    array; the basis-location vector is auto-derived from ``t`` when not
    supplied. ``penalty`` may be ``None`` for a neutral identity ridge of
    matching size.
    """
    import numpy as np

    display_kind, effective_kind, order, t_np, knots_np, penalty_np, eff_period = (
        _resolve_position_basis_inputs(
            t,
            basis_kind,
            knots_or_centers,
            penalty,
            basis=basis,
            basis_order=basis_order,
            periodic=periodic,
            period=period,
        )
    )
    try:
        out = rust_module().gaussian_reml_fit_positions(
            t_np,
            _numeric_matrix(y, "y"),
            str(effective_kind),
            knots_np,
            penalty_np,
            order,
            bool(periodic),
            None if eff_period is None else float(eff_period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _attach_basis_state(
        _coerce_gaussian_reml_payload(out, np),
        knots_or_centers=knots_np,
        penalty=penalty_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
        period=period,
    )


def gaussian_reml_fit_positions_backward(
    t: Any,
    y: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    grad_lambda: float = 0.0,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: float = 0.0,
    grad_edf: float = 0.0,
    forward_state: dict[str, Any] | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run the analytic VJP for ``gaussian_reml_fit_positions`` outputs.

    ``knots_or_centers`` and ``penalty`` accept the same auto-derived
    defaults as :func:`gaussian_reml_fit_positions`.
    """
    display_kind, effective_kind, order, t_np, knots_np, penalty_np, eff_period = (
        _resolve_position_basis_inputs(
            t,
            basis_kind,
            knots_or_centers,
            penalty,
            basis=basis,
            basis_order=basis_order,
            periodic=periodic,
            period=period,
        )
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_backward(
            t_np,
            _numeric_matrix(y, "y"),
            str(effective_kind),
            knots_np,
            penalty_np,
            float(grad_lambda),
            None
            if grad_coefficients is None
            else _numeric_matrix(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            float(grad_reml_score),
            float(grad_edf),
            forward_state,
            order,
            bool(periodic),
            None if eff_period is None else float(eff_period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return coerce_grad_payload(out, design_grad_key="grad_t")


def gaussian_reml_fit_positions_batched(
    t: Any,
    y: Any,
    row_offsets: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit packed ragged closed-form Gaussian REML problems from positions.

    ``knots_or_centers`` and ``penalty`` accept the same auto-derived
    defaults as :func:`gaussian_reml_fit_positions`. The basis locations
    are inferred from the concatenated positions across all groups.
    """
    import numpy as np

    display_kind, effective_kind, order, t_np, knots_np, penalty_np, eff_period = (
        _resolve_position_basis_inputs(
            t,
            basis_kind,
            knots_or_centers,
            penalty,
            basis=basis,
            basis_order=basis_order,
            periodic=periodic,
            period=period,
        )
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_batched(
            t_np,
            _numeric_matrix(y, "y"),
            _index_vector(row_offsets, "row_offsets"),
            str(effective_kind),
            knots_np,
            penalty_np,
            order,
            bool(periodic),
            None if eff_period is None else float(eff_period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _attach_basis_state(
        _coerce_gaussian_reml_payload(out, np),
        knots_or_centers=knots_np,
        penalty=penalty_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
        period=period,
    )


def gaussian_reml_fit_positions_batched_backward(
    t: Any,
    y: Any,
    row_offsets: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    grad_lambda: Any | None = None,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: Any | None = None,
    grad_edf: Any | None = None,
    forward_state: dict[str, Any] | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run the analytic VJP for packed position-based Gaussian REML fits.

    ``knots_or_centers`` and ``penalty`` accept the same auto-derived
    defaults as :func:`gaussian_reml_fit_positions_batched`.
    """
    offsets = _index_vector(row_offsets, "row_offsets")
    batch = int(offsets.size - 1)
    display_kind, effective_kind, order, t_np, knots_np, penalty_np, eff_period = (
        _resolve_position_basis_inputs(
            t,
            basis_kind,
            knots_or_centers,
            penalty,
            basis=basis,
            basis_order=basis_order,
            periodic=periodic,
            period=period,
        )
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_batched_backward(
            t_np,
            _numeric_matrix(y, "y"),
            offsets,
            str(effective_kind),
            knots_np,
            penalty_np,
            _optional_batch_vector(grad_lambda, batch, "grad_lambda"),
            None
            if grad_coefficients is None
            else _numeric_tensor3(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            _optional_batch_vector(grad_reml_score, batch, "grad_reml_score"),
            _optional_batch_vector(grad_edf, batch, "grad_edf"),
            forward_state,
            order,
            bool(periodic),
            None if eff_period is None else float(eff_period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return coerce_grad_payload(out, design_grad_key="grad_t")


def gaussian_reml_fit_latent(
    t: Any,
    y: Any,
    n_obs: int,
    latent_dim: int,
    centers: Any,
    penalty: Any,
    *,
    m: int = 2,
    weights: Any | None = None,
    fisher_w: Any | None = None,
    init_lambda: float | None = None,
    aux_u: Any | None = None,
    aux_family: str = "ridge",
    aux_strength: float | str | None = None,
    dim_selection_log_precision: Any | None = None,
    penalties: Sequence[Any] | None = None,
    basis_kind: str = "duchon",
    tensor_knots_concat: Any | None = None,
    tensor_knot_offsets: Sequence[int] | None = None,
    tensor_degrees: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit a Gaussian decoder at a *fixed* per-row latent coordinate.

    This is the differentiable inner solve: it builds ``Φ(t)`` at the ``t`` you
    pass and solves ``β`` / ``λ`` by REML. It does **not** move ``t`` — the
    returned ``t`` equals the input, so the fit quality is whatever the input
    coordinate already encodes. To *estimate* the latent coordinate from a poor
    init (recovering ``t`` itself), use
    :func:`gaussian_reml_optimize_latent`, which wraps this solve in a
    spectral-seeded outer optimization over ``t``.

    This is the low-level array API behind :class:`gamfit.LatentCoord`: each
    row has a latent coordinate ``t_n ∈ R^d`` and the fitted mean is

    .. math::

        \\hat Y_n = \\Phi(t_n)\\,\\beta,
        \\qquad
        \\ell = \\tfrac12\\|Y - \\Phi(t)\\beta\\|^2
             + \\tfrac12\\lambda\\beta^T S\\beta
             + R_id(t, u)
             - \\tfrac12\\log|H|.

    ``t`` is supplied as a flat row-major ``(N * d,)`` vector; ``n_obs`` and
    ``latent_dim`` carry the shape. ``centers`` and ``penalty`` define the
    decoder basis. ``basis_kind`` currently supports ``"duchon"`` and
    ``"tensor_bspline"`` on this forward path. The result contains
    coefficients, fitted values, REML score fields, and caches used by the
    backward companion.

    Identifiability options:

    * ``aux_u`` / ``aux_family`` / ``aux_strength`` add the conditional
      Gaussian prior ``R_id = 1/2 * mu * ||t - h(u)||^2``. REML selection of
      ``mu`` is selected when ``aux_strength`` is ``None`` or ``"auto"``;
      explicit floats keep ``mu`` fixed. Auto selection is valid only when
      the marginal likelihood includes the log ``mu`` normalizer, ``h`` is
      at least C1, and the conditional precision is positive-definite on the
      anchored subspace.
    * ``dim_selection_log_precision`` supplies ARD log-precisions, one per
      latent axis. ARD must be paired with an auxiliary prior or an isometry
      penalty to identify axes; by itself it is rotation-symmetric.

    Passing neither identifiability option is allowed for mechanical
    experiments, but the latent coordinate is gauge-unfixed and gradients in
    null directions are not meaningful.

    ``fisher_w`` is the scalar-response Fisher-block override hook: a dense
    ``(N, 1, 1)`` ``float64`` array whose single diagonal entry per row
    replaces the analytic IRLS weight. This Gaussian decoder is single-output,
    so only the ``(N, 1, 1)`` block is accepted here; the multi-output
    ``(N, K, K)`` multinomial/binomial-multi override lives on
    :func:`glm_reml_fit_latent`. ``None`` uses the analytic weight.
    """
    import numpy as np

    rust = rust_module()
    try:
        out = rust.gaussian_reml_fit_latent(
            _numeric_vector(t, "t"),
            _numeric_matrix(y, "y"),
            int(n_obs),
            int(latent_dim),
            _numeric_matrix(centers, "centers"),
            _numeric_matrix(penalty, "penalty"),
            int(m),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if fisher_w is None else _numeric_array(fisher_w, "fisher_w"),
            None if init_lambda is None else float(init_lambda),
            None if aux_u is None else _numeric_matrix(aux_u, "aux_u"),
            str(aux_family),
            _normalize_aux_strength(aux_strength),
            None
            if dim_selection_log_precision is None
            else _numeric_vector(dim_selection_log_precision, "dim_selection_log_precision"),
            _normalize_penalty_descriptors(penalties),
            str(basis_kind),
            None
            if tensor_knots_concat is None
            else _numeric_vector(tensor_knots_concat, "tensor_knots_concat"),
            None if tensor_knot_offsets is None else [int(v) for v in tensor_knot_offsets],
            None if tensor_degrees is None else [int(v) for v in tensor_degrees],
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in (
        "coefficients",
        "fitted",
        "sigma2",
        "cache_penalty_eigenvalues",
        "cache_eigenvectors",
    ):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_reml_fit_latent_backward(
    t: Any,
    y: Any,
    n_obs: int,
    latent_dim: int,
    centers: Any,
    penalty: Any,
    *,
    grad_lambda: float = 0.0,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: float = 0.0,
    grad_edf: float = 0.0,
    m: int = 2,
    weights: Any | None = None,
    fisher_w: Any | None = None,
    init_lambda: float | None = None,
    aux_u: Any | None = None,
    aux_family: str = "ridge",
    aux_strength: float | str | None = None,
    dim_selection_log_precision: Any | None = None,
    basis_kind: str = "duchon",
    sigma_eff_mode: str = "profiled",
    tensor_knots_concat: Any | None = None,
    tensor_knot_offsets: Sequence[int] | None = None,
    tensor_degrees: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Backward / adjoint companion to :func:`gaussian_reml_fit_latent`.

    Returns the standard REML adjoint gradients (``grad_y``,
    ``grad_penalty``, ``grad_weights``) plus the latent gradient
    ``grad_t`` with shape ``(n_obs, latent_dim)``.

    ``sigma_eff_mode`` selects the dispersion convention for the analytic
    outer REML latent gradient. The default ``"profiled"`` matches the
    existing REML objective; ``"fixed"`` is accepted for the fixed-dispersion
    call shape.

    ``grad_t`` includes the additive identifiability-mode contributions
    (auxiliary-prior pullback and/or ARD per-axis ridge); the outer
    driver may walk ``t`` directly under this combined gradient without
    re-applying the gauge fix.

    ``basis_kind`` currently supports ``"duchon"`` and ``"tensor_bspline"``
    for this backward path.

    ``fisher_w`` mirrors the forward hook on :func:`gaussian_reml_fit_latent`:
    a single-output ``(N, 1, 1)`` ``float64`` per-row diagonal override of the
    analytic IRLS weight (this backward companion is scalar-only). ``None``
    uses the analytic weight.
    """
    import numpy as np

    rust = rust_module()
    try:
        out = rust.gaussian_reml_fit_latent_backward(
            _numeric_vector(t, "t"),
            _numeric_matrix(y, "y"),
            int(n_obs),
            int(latent_dim),
            _numeric_matrix(centers, "centers"),
            _numeric_matrix(penalty, "penalty"),
            float(grad_lambda),
            None if grad_coefficients is None else _numeric_matrix(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            float(grad_reml_score),
            float(grad_edf),
            int(m),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if fisher_w is None else _numeric_array(fisher_w, "fisher_w"),
            None if init_lambda is None else float(init_lambda),
            None if aux_u is None else _numeric_matrix(aux_u, "aux_u"),
            str(aux_family),
            _normalize_aux_strength(aux_strength),
            None
            if dim_selection_log_precision is None
            else _numeric_vector(dim_selection_log_precision, "dim_selection_log_precision"),
            str(basis_kind),
            str(sigma_eff_mode),
            None
            if tensor_knots_concat is None
            else _numeric_vector(tensor_knots_concat, "tensor_knots_concat"),
            None if tensor_knot_offsets is None else [int(v) for v in tensor_knot_offsets],
            None if tensor_degrees is None else [int(v) for v in tensor_degrees],
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    # Fixes audit-revised claim that latent ARD/AuxPrior REML selection needs
    # adjoints for the normalized log-precision terms as well as grad_t.
    for key in (
        "grad_t",
        "grad_y",
        "grad_penalty",
        "grad_weights",
        "grad_dim_selection_log_precision",
    ):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_reml_optimize_latent(
    y: Any,
    n_obs: int,
    latent_dim: int,
    centers: Any,
    penalty: Any,
    *,
    t: Any | None = None,
    m: int = 2,
    weights: Any | None = None,
    fisher_w: Any | None = None,
    init_lambda: float | None = None,
    aux_u: Any | None = None,
    aux_family: str = "ridge",
    aux_strength: float | str | None = None,
    dim_selection_log_precision: Any | None = None,
    basis_kind: str = "duchon",
    tensor_knots_concat: Any | None = None,
    tensor_knot_offsets: Sequence[int] | None = None,
    tensor_degrees: Sequence[int] | None = None,
    manifold: str = "euclidean",
    sigma_eff_mode: str = "profiled",
    max_iter: int = 200,
    grad_tol: float = 1.0e-8,
    trust_radius: float = 1.0,
    max_radius: float = 1.0e6,
    n_restarts: int = 1,
    restart_scale: float = 0.25,
    seed: int = 0,
    init: str = "spectral",
    seed_neighbors: int = 10,
) -> dict[str, Any]:
    """Estimate the per-row latent coordinate ``t`` *and* the decoder.

    This is the latent-*optimizing* companion to :func:`gaussian_reml_fit_latent`
    (which is a single ``β | t`` solve at a fixed ``t``). It minimizes the same
    Gaussian-REML score over ``t`` with a Riemannian trust region driven by the
    analytic ``∂(reml_score)/∂t``, retracting each step onto ``manifold``, and
    returns the full REML fit dictionary *at the recovered latent* together with
    the optimized coordinate under the keys ``"t"`` / ``"latent"`` (shape
    ``(n_obs, latent_dim)``) and ``"t_flat"``.

    The objective is non-convex (a GP-LVM-style coordinate problem), so a cold
    random start settles in a poor local optimum (see issue #627). With the
    default ``init="spectral"`` the optimizer seeds restart 0 from a
    Laplacian-eigenmaps embedding of ``y`` — which recovers the intrinsic
    coordinate up to a monotone/rotation gauge — then polishes it, so a good
    initial ``t`` is *not* required. ``t`` is optional; when omitted a zero
    vector is used as the fallback start (taken only when the spectral seed is
    unavailable, e.g. too few rows or a non-Euclidean ``manifold``). Pass
    ``init="caller"`` to start from ``t`` unchanged (a pure local solve / explicit
    warm start), ``n_restarts > 1`` to additionally try perturbed starts and keep
    the lowest-score result, and ``seed_neighbors`` to set the spectral seed's
    k-nearest-neighbour graph size.

    The ``centers`` / ``penalty`` / ``basis_kind`` arguments define the decoder
    basis exactly as in :func:`gaussian_reml_fit_latent`; the spectral seed is
    affinely mapped onto the span of ``centers`` so it lands where ``Φ`` is
    well-conditioned. The result also carries ``"grad_t_norm"``,
    ``"grad_t_norm_init"``, ``"grad_t_norm_scaled"``, ``"objective_value"``,
    ``"n_restarts"``, and ``"init"`` diagnostics. A separate convergence flag
    is unnecessary: returning a fit is itself the convergence certificate.

    Convergence is decided from ``"grad_t_norm_scaled"`` -- the *relative*
    latent-gradient stationarity measure
    ``‖∇ₜ f(t̂)‖ / max(‖∇ₜ f(t₀)‖, 1)`` (``"grad_t_norm"`` divided by
    ``max("grad_t_norm_init", 1)``) -- rather than the raw ``"grad_t_norm"``,
    because the *profiled* Gaussian REML objective leaves the raw gradient at an
    O(n) magnitude near interpolation (R²≈1) even at a genuine stationary point.
    Anchoring to the gradient norm at the initial iterate ``t₀`` divides out that
    common O(n)/multiplicative scale while staying invariant under an additive
    shift ``f → f + C`` of the objective -- unlike the earlier
    ``‖∇ₜ f‖ · ‖t‖_typ / max(|f|, 1)``, whose ``max(|f|, 1)`` denominator a large
    additive constant could inflate into a false convergence (issue #954).

    A fit is only ever returned from a *converged* optimization (SPEC rule 20).
    If the relative stationarity measure does not reach ``grad_tol`` within the
    iteration budget, this raises :class:`gamfit.RemlConvergenceError` instead
    of returning a degraded payload. The exception carries the evidence as
    attributes (``grad_t_norm``, ``grad_t_norm_init``, ``grad_t_norm_scaled``,
    ``grad_tol``, ``latent_t_std``, ``objective_value``, ``max_iter``,
    ``n_restarts``, ``init``) plus ``checkpoint_t`` -- the best latent found,
    shape ``(n_obs, latent_dim)`` -- so the caller can resume the same solve
    via ``t=exc.checkpoint_t, init="caller"`` with a larger ``max_iter``.
    """
    import numpy as np

    expected = int(n_obs) * int(latent_dim)
    if t is None:
        t_vec = np.zeros(expected, dtype=float)
    else:
        t_vec = _numeric_vector(t, "t")

    rust = rust_module()
    try:
        out = rust.gaussian_reml_optimize_latent(
            t_vec,
            _numeric_matrix(y, "y"),
            int(n_obs),
            int(latent_dim),
            _numeric_matrix(centers, "centers"),
            _numeric_matrix(penalty, "penalty"),
            int(m),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if fisher_w is None else _numeric_array(fisher_w, "fisher_w"),
            None if init_lambda is None else float(init_lambda),
            None if aux_u is None else _numeric_matrix(aux_u, "aux_u"),
            str(aux_family),
            _normalize_aux_strength(aux_strength),
            None
            if dim_selection_log_precision is None
            else _numeric_vector(dim_selection_log_precision, "dim_selection_log_precision"),
            str(basis_kind),
            None
            if tensor_knots_concat is None
            else _numeric_vector(tensor_knots_concat, "tensor_knots_concat"),
            None if tensor_knot_offsets is None else [int(v) for v in tensor_knot_offsets],
            None if tensor_degrees is None else [int(v) for v in tensor_degrees],
            str(manifold),
            str(sigma_eff_mode),
            int(max_iter),
            float(grad_tol),
            float(trust_radius),
            float(max_radius),
            int(n_restarts),
            float(restart_scale),
            int(seed),
            str(init),
            int(seed_neighbors),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("coefficients", "fitted", "sigma2", "t", "latent", "t_flat"):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def glm_reml_fit_latent(
    t: Any,
    y: Any,
    n_obs: int,
    latent_dim: int,
    centers: Any,
    penalty: Any,
    family: str,
    *,
    tweedie_p: float = 1.5,
    negbin_theta: float | None = None,
    beta_phi: float | None = None,
    m: int = 2,
    weights: Any | None = None,
    fisher_w: Any | None = None,
    init_lambda: float | None = None,
    aux_u: Any | None = None,
    aux_family: str = "ridge",
    aux_strength: float | str | None = None,
    dim_selection_log_precision: Any | None = None,
    penalties: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Latent-coordinate REML/LAML fit for GLM families via PIRLS.

    ``family`` accepts ``"binomial-logit"``, ``"binomial-probit"``,
    ``"binomial-cloglog"``, ``"poisson-log"``, ``"tweedie-log"``,
    ``"negbin-log"``, ``"beta-regression-logit"``, ``"gamma-log"``,
    ``"gaussian-identity"``, and the multi-output families
    ``"multinomial-logit"`` (aliases ``"multinomial"`` / ``"softmax"`` /
    ``"categorical-logit"``) and a multi-column binomial-logit fit (pass a
    multi-column ``y`` with ``family="binomial-logit"``). ``tweedie_p``
    defaults to ``1.5``.

    ``fisher_w`` is an advanced research hook: a per-row Fisher-block override
    that replaces the analytic curvature in the inner penalised Newton/PIRLS
    step while the gradient and residual stay analytic. It is supplied as a
    dense ``(N, K, K)`` ``float64`` array (one ``K × K`` block per observation)
    and its meaning is fixed by the dispatched fitter:

    * **Scalar single-output path** (single-column ``y`` with a non-multinomial
      family): ``K = 1``, so the override is an ``(N, 1, 1)`` block honoured by
      the scalar fitter — the per-row diagonal IRLS weight.
    * **Multinomial** (``family="multinomial-*"`` / multi-column ``y`` routed to
      softmax): the leading active ``(K-1) × (K-1)`` sub-block of each row
      replaces the analytic softmax Fisher
      ``H_{n,a,b} = p_a (δ_{ab} − p_b)`` over the reference-coded active
      classes. The active block must be finite, symmetric, and have a
      non-negative diagonal; a non-symmetric active block is rejected at the
      boundary.
    * **Binomial-multi** (multi-column ``y`` with ``family="binomial-logit"``):
      the diagonal of each ``(N, K, K)`` block replaces the per-class analytic
      Fisher ``H_{n,a,a} = μ_a (1 − μ_a)``. The ``K`` columns are fit
      independently, so the off-diagonals must be zero (and finite,
      non-negative on the diagonal); any non-zero off-diagonal is rejected at
      the boundary.

    When ``fisher_w`` is ``None`` the analytic Fisher is used and the fit is
    bit-for-bit identical to omitting the hook.
    """
    import numpy as np

    rust = rust_module()
    family_normalized = str(family).lower().replace("_", "-")
    is_negbin = family_normalized in {
        "negbin",
        "negbin-log",
        "negative-binomial",
        "negative-binomial-log",
    }
    is_beta = family_normalized in {
        "beta",
        "beta-logit",
        "beta-regression",
        "beta-regression-logit",
    }
    if is_negbin and negbin_theta is None:
        raise ValueError("negbin_theta must be provided when family='negbin'")
    theta = 1.0 if negbin_theta is None else float(negbin_theta)
    if is_negbin and not (np.isfinite(theta) and theta > 0.0):
        raise ValueError(f"negbin_theta must be finite and > 0; got {theta!r}")
    if is_beta and beta_phi is None:
        raise ValueError("beta_phi must be provided when family='beta'")
    phi = 1.0 if beta_phi is None else float(beta_phi)
    if is_beta and not (np.isfinite(phi) and phi > 0.0):
        raise ValueError(f"beta_phi must be finite and > 0; got {phi!r}")
    try:
        out = rust.glm_reml_fit_latent(
            _numeric_vector(t, "t"),
            _numeric_matrix(y, "y"),
            int(n_obs),
            int(latent_dim),
            _numeric_matrix(centers, "centers"),
            _numeric_matrix(penalty, "penalty"),
            str(family),
            float(tweedie_p),
            theta,
            phi,
            int(m),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if fisher_w is None else _numeric_array(fisher_w, "fisher_w"),
            None if init_lambda is None else float(init_lambda),
            None if aux_u is None else _numeric_matrix(aux_u, "aux_u"),
            str(aux_family),
            _normalize_aux_strength(aux_strength),
            None
            if dim_selection_log_precision is None
            else _numeric_vector(dim_selection_log_precision, "dim_selection_log_precision"),
            _normalize_penalty_descriptors(penalties),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in (
        "coefficients",
        "fitted",
        "sigma2",
        "cache_penalty_eigenvalues",
        "cache_eigenvectors",
    ):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def glm_reml_fit_latent_backward(
    t: Any,
    y: Any,
    n_obs: int,
    latent_dim: int,
    centers: Any,
    penalty: Any,
    family: str,
    *,
    grad_reml_score: float = 1.0,
    tweedie_p: float = 1.5,
    negbin_theta: float | None = None,
    beta_phi: float | None = None,
    m: int = 2,
    weights: Any | None = None,
    fisher_w: Any | None = None,
    init_lambda: float | None = None,
    aux_u: Any | None = None,
    aux_family: str = "ridge",
    aux_strength: float | str | None = None,
    dim_selection_log_precision: Any | None = None,
    basis_kind: str = "duchon",
) -> dict[str, Any]:
    """Return the analytic Duchon latent gradient for ``glm_reml_fit_latent``.

    The ``basis_kind`` argument is accepted for signature symmetry, but this
    low-level GLM backward path currently implements only ``"duchon"``.

    ``fisher_w`` is the scalar-response Fisher-block override: a dense
    ``(N, 1, 1)`` ``float64`` per-row diagonal replacement of the analytic
    IRLS weight. This backward path is scalar-only; the multi-output
    ``(N, K, K)`` multinomial/binomial-multi override is exposed on the
    forward :func:`glm_reml_fit_latent`. ``None`` uses the analytic weight.
    """
    import numpy as np

    rust = rust_module()
    family_normalized = str(family).lower().replace("_", "-")
    is_negbin = family_normalized in {
        "negbin",
        "negbin-log",
        "negative-binomial",
        "negative-binomial-log",
    }
    is_beta = family_normalized in {
        "beta",
        "beta-logit",
        "beta-regression",
        "beta-regression-logit",
    }
    if is_negbin and negbin_theta is None:
        raise ValueError("negbin_theta must be provided when family='negbin'")
    theta = 1.0 if negbin_theta is None else float(negbin_theta)
    if is_negbin and not (np.isfinite(theta) and theta > 0.0):
        raise ValueError(f"negbin_theta must be finite and > 0; got {theta!r}")
    if is_beta and beta_phi is None:
        raise ValueError("beta_phi must be provided when family='beta'")
    phi = 1.0 if beta_phi is None else float(beta_phi)
    if is_beta and not (np.isfinite(phi) and phi > 0.0):
        raise ValueError(f"beta_phi must be finite and > 0; got {phi!r}")
    try:
        out = rust.glm_reml_fit_latent_backward(
            _numeric_vector(t, "t"),
            _numeric_matrix(y, "y"),
            int(n_obs),
            int(latent_dim),
            _numeric_matrix(centers, "centers"),
            _numeric_matrix(penalty, "penalty"),
            str(family),
            float(grad_reml_score),
            int(m),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if fisher_w is None else _numeric_array(fisher_w, "fisher_w"),
            None if init_lambda is None else float(init_lambda),
            None if aux_u is None else _numeric_matrix(aux_u, "aux_u"),
            str(aux_family),
            _normalize_aux_strength(aux_strength),
            None
            if dim_selection_log_precision is None
            else _numeric_vector(dim_selection_log_precision, "dim_selection_log_precision"),
            float(tweedie_p),
            theta,
            phi,
            str(basis_kind),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    if result.get("grad_t") is not None:
        result["grad_t"] = np.asarray(result["grad_t"], dtype=float)
    return result


def gaussian_reml_fit_formula(
    data: Any,
    formula: str,
    y: Any,
    *,
    config: dict[str, Any] | None = None,
    fisher_rao_w: Any | None = None,
) -> dict[str, Any]:
    """Fit closed-form Gaussian REML after materialising a formula design."""
    import numpy as np

    headers, rows, _kind = normalize_table(data)
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if y_arr.ndim != 2:
        raise ValueError("y must be a 1D or 2D numeric array")
    if y_arr.shape[0] != len(rows):
        raise ValueError(
            f"formula design and y row counts must match; got {len(rows)} and {y_arr.shape[0]}"
        )
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite values")
    fisher_w = None
    if fisher_rao_w is not None:
        fisher_w = _normalize_fisher_rao_w(
            fisher_rao_w, n_rows=y_arr.shape[0], dim=y_arr.shape[1]
        )
    try:
        out = rust_module().gaussian_reml_fit_formula_table(
            headers,
            rows,
            formula,
            np.ascontiguousarray(y_arr, dtype=np.float64),
            None if config is None else json.dumps(config),
            fisher_w,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_blocks_forward(
    designs: list[Any],
    penalties: list[Any],
    y: Any,
    *,
    weights: Any | None = None,
    init_rhos: Any | None = None,
) -> dict[str, Any]:
    """Multi-block Gaussian REML forward fit with per-smooth λ_k.

    Routes per-smooth design/penalty blocks into the Rust joint REML driver
    (same code path as the formula API) and returns coefficients, per-smooth
    λ vector, per-smooth EDF, and the converged REML score.
    """
    import numpy as np

    if len(designs) != len(penalties):
        raise ValueError(
            "designs and penalties must have equal length; "
            f"got {len(designs)} vs {len(penalties)}"
        )
    if len(designs) == 0:
        raise ValueError("gaussian_reml_fit_blocks_forward requires at least one block")

    designs_np = [_numeric_matrix(d, f"designs[{i}]") for i, d in enumerate(designs)]
    penalties_np = [
        _numeric_matrix(p, f"penalties[{i}]") for i, p in enumerate(penalties)
    ]
    y_np = _numeric_matrix(y, "y")
    weights_np = None if weights is None else _numeric_vector(weights, "weights")
    rhos_np = None if init_rhos is None else _numeric_vector(init_rhos, "init_rhos")

    try:
        out = rust_module().gaussian_reml_fit_blocks_forward(
            designs_np,
            penalties_np,
            y_np,
            weights_np,
            rhos_np,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("coefficients", "fitted", "lambdas", "log_lambdas", "edf", "col_offsets"):
        if key in result:
            if key == "col_offsets":
                result[key] = np.asarray(result[key], dtype=np.uintp)
            else:
                result[key] = np.asarray(result[key], dtype=float)
    if "reml_score" in result:
        result["reml_score"] = float(result["reml_score"])
    return result


def gaussian_reml_fit_blocks_orthogonal_forward(
    designs: list[Any],
    penalties: list[Any],
    y: Any,
    *,
    weights: Any | None = None,
    init_rhos: Any | None = None,
) -> dict[str, Any]:
    """Block-orthogonal additive Gaussian REML with shared output scales."""
    import numpy as np

    designs_np = [_numeric_matrix(d, f"designs[{i}]") for i, d in enumerate(designs)]
    penalties_np = [
        _numeric_matrix(p, f"penalties[{i}]") for i, p in enumerate(penalties)
    ]
    y_np = _numeric_matrix(y, "y")
    weights_np = None if weights is None else _numeric_vector(weights, "weights")
    rhos_np = None if init_rhos is None else _numeric_vector(init_rhos, "init_rhos")
    try:
        out = rust_module().gaussian_reml_fit_blocks_orthogonal_forward(
            designs_np,
            penalties_np,
            y_np,
            weights_np,
            rhos_np,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    if "coefficients" in result:
        result["coefficients"] = [
            np.asarray(coef, dtype=float) for coef in result["coefficients"]
        ]
    for key in ("fitted", "lambdas", "log_lambdas", "edf"):
        if key in result:
            result[key] = np.asarray(result[key], dtype=float)
    if "reml_score" in result:
        result["reml_score"] = float(result["reml_score"])
    return result


def gaussian_reml_fit_blocks_backward(
    designs: list[Any],
    penalties: list[Any],
    y: Any,
    log_lambdas: Any,
    *,
    weights: Any | None = None,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_lambdas: Any | None = None,
    grad_log_lambdas: Any | None = None,
    grad_reml_score: float = 0.0,
    grad_edf: Any | None = None,
) -> dict[str, Any]:
    """Analytic backward for the multi-block per-smooth-λ Gaussian REML fit.

    Computes VJPs of ``(coefficients, fitted, lambdas, log_lambdas,
    reml_score, edf)`` back to ``(designs, penalties, y, weights)`` using
    the profiled Gaussian/identity block REML VJP and the outer implicit
    smoothing-parameter adjoint at the converged log-λ vector.
    """
    import numpy as np

    if len(designs) != len(penalties):
        raise ValueError(
            "designs and penalties must have equal length; "
            f"got {len(designs)} vs {len(penalties)}"
        )
    if len(designs) == 0:
        raise ValueError("gaussian_reml_fit_blocks_backward requires at least one block")

    designs_np = [_numeric_matrix(d, f"designs[{i}]") for i, d in enumerate(designs)]
    penalties_np = [
        _numeric_matrix(p, f"penalties[{i}]") for i, p in enumerate(penalties)
    ]
    y_np = _numeric_matrix(y, "y")
    weights_np = None if weights is None else _numeric_vector(weights, "weights")
    log_lambdas_np = _numeric_vector(log_lambdas, "log_lambdas")
    gc = (
        None
        if grad_coefficients is None
        else _numeric_matrix(grad_coefficients, "grad_coefficients")
    )
    gf = None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted")
    gl = None if grad_lambdas is None else _numeric_vector(grad_lambdas, "grad_lambdas")
    glog = (
        None
        if grad_log_lambdas is None
        else _numeric_vector(grad_log_lambdas, "grad_log_lambdas")
    )
    ge = None if grad_edf is None else _numeric_vector(grad_edf, "grad_edf")

    try:
        out = rust_module().gaussian_reml_fit_blocks_backward(
            designs_np,
            penalties_np,
            y_np,
            weights_np,
            log_lambdas_np,
            gc,
            gf,
            gl,
            glog,
            float(grad_reml_score),
            ge,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    if "grad_designs" in result:
        result["grad_designs"] = [
            np.asarray(g, dtype=float) for g in result["grad_designs"]
        ]
    if "grad_penalties" in result:
        result["grad_penalties"] = [
            np.asarray(g, dtype=float) for g in result["grad_penalties"]
        ]
    if "grad_y" in result:
        result["grad_y"] = np.asarray(result["grad_y"], dtype=float)
    if "grad_weights" in result:
        result["grad_weights"] = np.asarray(result["grad_weights"], dtype=float)
    return result


def gaussian_reml_fit_with_constraints_forward(
    x: Any,
    y: Any,
    penalty: Any,
    *,
    weights: Any | None = None,
    init_log_lambda: float | None = None,
    a_inequality: Any | None = None,
    b_inequality: Any | None = None,
) -> dict[str, Any]:
    """Constrained Gaussian REML forward fit (single penalty block).

    Wraps the active-set + REML driver with an optional linear inequality
    system ``A·β ≤ b``. This path has an exact analytic VJP, provided by
    :func:`gaussian_reml_fit_with_constraints_backward`: at an interior cert
    (empty active set) it is the envelope-theorem backward in full p-space;
    at an active cert it is the tangent-projected backward in the
    ``Z = null(A_act)`` reduction (``H⁻¹ → Z(ZᵀHZ)⁻¹Zᵀ``,
    ``S⁺ → Z(ZᵀSZ)⁺Zᵀ``).
    """
    import numpy as np

    x_np = _numeric_matrix(x, "x")
    y_np = _numeric_matrix(y, "y")
    penalty_np = _numeric_matrix(penalty, "penalty")
    weights_np = None if weights is None else _numeric_vector(weights, "weights")
    a_np = None if a_inequality is None else _numeric_matrix(a_inequality, "a_inequality")
    b_np = None if b_inequality is None else _numeric_vector(b_inequality, "b_inequality")
    init_rho = None if init_log_lambda is None else float(init_log_lambda)

    try:
        out = rust_module().gaussian_reml_fit_with_constraints_forward(
            x_np,
            y_np,
            penalty_np,
            weights_np,
            init_rho,
            a_np,
            b_np,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("coefficients", "fitted"):
        if key in result:
            result[key] = np.asarray(result[key], dtype=float)
    if "active_indices" in result:
        result["active_indices"] = np.asarray(result["active_indices"], dtype=np.uintp)
    for key in ("lambda", "log_lambda", "reml_score", "edf"):
        if key in result:
            result[key] = float(result[key])
    return result


def gaussian_reml_fit_with_constraints_backward(
    x: Any,
    y: Any,
    penalty: Any,
    *,
    weights: Any | None = None,
    a_inequality: Any | None = None,
    b_inequality: Any | None = None,
    log_lambda_at_optimum: float | None = None,
    coefficients_at_optimum: Any | None = None,
    fitted_at_optimum: Any | None = None,
    active_indices: Any | None = None,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_lambda: float = 0.0,
    grad_log_lambda: float = 0.0,
    grad_reml_score: float = 0.0,
    grad_edf: float = 0.0,
) -> dict[str, Any]:
    """Analytic VJP for ``gaussian_reml_fit_with_constraints_forward``.

    Both certs are supported and return exact analytic gradients. At an
    interior cert (``active_indices`` is empty), the envelope theorem applies
    in full p-space and this delegates to the closed-form Gaussian REML
    backward (``H`` unprojected). At an active cert (non-empty active set),
    the tangent-projected variant reduces the problem to the
    ``Z = null(A_act)`` subspace, runs the same closed-form backward on the
    reduced operators, and lifts the gradients back to p-space.

    Math identity: at the constrained cert, the backward through the joint
    REML is the unconstrained backward formula applied to the tangent-
    projected operator — ``H⁻¹ → Z(ZᵀHZ)⁻¹Zᵀ``, ``S⁺ → Z(ZᵀSZ)⁺Zᵀ``, with
    ``Z = null(A_act)``. Gradients flow to ``x``, ``y``, ``penalty`` and
    ``weights``; the constraint geometry (``a_inequality``, ``b_inequality``)
    and ``init_log_lambda`` are non-differentiable.
    """
    import numpy as np

    x_np = _numeric_matrix(x, "x")
    y_np = _numeric_matrix(y, "y")
    penalty_np = _numeric_matrix(penalty, "penalty")
    weights_np = None if weights is None else _numeric_vector(weights, "weights")
    a_np = None if a_inequality is None else _numeric_matrix(a_inequality, "a_inequality")
    b_np = None if b_inequality is None else _numeric_vector(b_inequality, "b_inequality")
    coef_np = (
        None
        if coefficients_at_optimum is None
        else _numeric_matrix(coefficients_at_optimum, "coefficients_at_optimum")
    )
    fitted_np = (
        None
        if fitted_at_optimum is None
        else _numeric_matrix(fitted_at_optimum, "fitted_at_optimum")
    )
    if active_indices is None:
        active_np = None
    else:
        active_np = np.asarray(active_indices).astype(np.uint64, copy=False).ravel()
    grad_coef_np = (
        None
        if grad_coefficients is None
        else _numeric_matrix(grad_coefficients, "grad_coefficients")
    )
    grad_fitted_np = (
        None
        if grad_fitted is None
        else _numeric_matrix(grad_fitted, "grad_fitted")
    )
    log_lambda_val = None if log_lambda_at_optimum is None else float(log_lambda_at_optimum)

    try:
        out = rust_module().gaussian_reml_fit_with_constraints_backward(
            x_np,
            y_np,
            penalty_np,
            weights_np,
            a_np,
            b_np,
            log_lambda_val,
            coef_np,
            fitted_np,
            active_np,
            grad_coef_np,
            grad_fitted_np,
            float(grad_lambda),
            float(grad_log_lambda),
            float(grad_reml_score),
            float(grad_edf),
        )
    except NotImplementedError:
        # Preserve the contract surface from Rust: callers can rely on
        # ``NotImplementedError`` for the active-cert backward path.
        raise
    except Exception as exc:
        raise map_exception(exc) from exc
    return coerce_grad_payload(out)


def _coerce_gaussian_reml_payload(payload: Any, np: Any) -> dict[str, Any]:
    out = dict(payload)
    for key in (
        "coefficients",
        "fitted",
        "sigma2",
        "lambda",
        "rho",
        "reml_score",
        "reml_grad_lambda",
        "reml_hess_lambda",
        "reml_grad_rho",
        "reml_hess_rho",
        "edf",
        "cache_penalty_eigenvalues",
        "cache_eigenvectors",
        "cache_coefficient_basis",
    ):
        if key in out:
            out[key] = np.asarray(out[key], dtype=float)
    return out


def _position_basis_order(basis_kind: str, basis_order: int | None) -> int:
    if basis_order is not None:
        order = int(basis_order)
    else:
        normalized = basis_kind.strip().lower().replace("_", "").replace("-", "")
        order = 2 if normalized in {"duchon", "duchonspline"} else 3
    if order < 1:
        raise ValueError("basis_order must be at least 1")
    return order


def _optional_batch_vector(values: Any | None, batch: int, label: str) -> Any | None:
    import numpy as np

    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.full(batch, float(arr), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a scalar or 1D numeric array")
    if arr.size != batch:
        raise ValueError(f"{label} length mismatch: expected {batch}, got {arr.size}")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _index_vector(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D integer array")
    if arr.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.dtype(np.uintp):
        # Coerce integer-valued input (torch tensor, int64/lists, integral
        # floats) to the uintp layout the FFI needs, instead of rejecting it.
        # This mirrors how `t`/`y` are coerced and keeps the call site
        # symmetric (see issue gam#581).
        kind = arr.dtype.kind
        if kind in ("i", "u"):
            if kind == "i" and np.any(arr < 0):
                raise ValueError(f"{label} must be non-negative")
        elif kind == "f":
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{label} must contain only finite values")
            if np.any(arr < 0) or np.any(arr != np.floor(arr)):
                raise TypeError(
                    f"{label} must be integer-valued; got non-integer float entries"
                )
        else:
            raise TypeError(
                f"{label} must be an integer array (or integral numeric); "
                f"got dtype {arr.dtype}"
            )
        arr = np.ascontiguousarray(arr.astype(np.uintp))
    if label == "row_offsets":
        if arr.size < 2:
            raise ValueError("row_offsets must contain at least start and stop offsets")
        if int(arr[0]) != 0:
            raise ValueError("row_offsets must start at 0")
        if np.any(arr[1:] < arr[:-1]):
            raise ValueError("row_offsets must be nondecreasing")
    return arr


def _numeric_vector(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D numeric array")
    if arr.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _numeric_matrix(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D numeric array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _numeric_array(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


# Default number of basis functions when the caller does not pin centers /
# knots themselves. Matches mgcv's ``k = 10`` convention. The actual
# placement (quantile knots, equal-mass centers, boundary clamping, ...)
# is performed by the Rust engine; Python only forwards the request.
_DEFAULT_BASIS_K = 10


def _resolve_centers(centers: Any, t_arr: Any, *, label: str = "centers") -> Any:
    """Coerce ``centers`` (None / int / array) into a 1D float64 array.

    Auto-derivation delegates to the Rust ``auto_centers_1d`` FFI export so
    Python never reimplements basis-placement logic.
    """
    if centers is None:
        return _numpy_module().asarray(
            rust_module().auto_centers_1d(t_arr, int(_DEFAULT_BASIS_K)),
            dtype=float,
        )
    if isinstance(centers, int) and not isinstance(centers, bool):
        if centers < 2:
            raise ValueError(f"{label}: integer count must be >= 2, got {centers}")
        return _numpy_module().asarray(
            rust_module().auto_centers_1d(t_arr, int(centers)),
            dtype=float,
        )
    return _numeric_vector(centers, label)


class _ResolvedBasisLocations(NamedTuple):
    """Outcome of resolving a basis-location argument (knots or centers).

    ``order`` is the spline *degree* (B-spline) or *m*-order (Duchon) that the
    ``locations`` vector was actually built for. For auto-derived B-spline
    knots it can differ from the requested degree: the Rust engine auto-shrinks
    cubic → quadratic → linear when ``n`` is too small (issue #340), and the
    clamped knot vector then carries boundary multiplicity ``order + 1``.
    Evaluating it with any other degree fails ("insufficient knots") or breaks
    the partition-of-unity, so callers MUST use this effective ``order`` for
    every downstream basis/penalty build rather than the requested one.

    ``shrunk`` is ``True`` iff the auto-shrink reduced the requested
    ``(degree, num_internal_knots)``.
    """

    locations: Any
    order: int
    shrunk: bool


def _resolve_knots(
    knots: Any,
    t_arr: Any,
    *,
    label: str = "knots",
    degree: int = 3,
) -> _ResolvedBasisLocations:
    """Coerce ``knots`` (None / int / array) into a resolved knot vector.

    Auto-derivation delegates to the Rust ``auto_knots_1d`` FFI export, which
    returns ``(knots, effective_degree, num_internal_knots, shrunk)`` (issue
    #340). We surface the knot vector together with the **effective** degree so
    the auto-shrink decision stays consistent with downstream evaluation; the
    explicit-array path passes the requested degree straight through unshrunk.
    """
    degree_i = int(degree)
    if degree_i < 0:
        raise ValueError(f"{label}: degree must be non-negative, got {degree}")
    if knots is None or (isinstance(knots, int) and not isinstance(knots, bool)):
        if knots is None:
            requested_internal = int(_DEFAULT_BASIS_K)
        else:
            if knots < 0:
                raise ValueError(
                    f"{label}: integer interior-knot count must be >= 0, got {knots}"
                )
            requested_internal = int(knots)
        knot_vec, eff_degree, _eff_internal, shrunk = rust_module().auto_knots_1d(
            t_arr, requested_internal, degree_i
        )
        return _ResolvedBasisLocations(
            _numpy_module().asarray(knot_vec, dtype=float),
            int(eff_degree),
            bool(shrunk),
        )
    return _ResolvedBasisLocations(_numeric_vector(knots, label), degree_i, False)


def _numpy_module() -> Any:
    """Return the cached ``numpy`` module without polluting global imports."""
    import numpy as np

    return np


def _resolve_basis_locations(
    arg: Any,
    t_arr: Any,
    *,
    basis_kind: str,
    label: str = "knots_or_centers",
    degree: int = 3,
) -> _ResolvedBasisLocations:
    """Resolve the basis-location argument for kind-dispatched primitives.

    Mirrors :func:`_resolve_centers` for ``basis_kind == "duchon"`` and
    :func:`_resolve_knots` for B-spline-like kinds.
    """
    kind = str(basis_kind).strip().lower().replace("_", "").replace("-", "")
    if kind in {"duchon", "duchonspline"}:
        # Duchon centers carry no degree concept and are never auto-shrunk, so
        # the requested order passes straight through.
        return _ResolvedBasisLocations(
            _resolve_centers(arg, t_arr, label=label), int(degree), False
        )
    return _resolve_knots(arg, t_arr, label=label, degree=degree)


def _attach_basis_state(
    payload: dict[str, Any],
    *,
    knots_or_centers: Any,
    penalty: Any,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
    period: float | None,
) -> dict[str, Any]:
    """Embed the resolved basis state in a REML position-fit payload.

    The keys ``knots_or_centers``, ``penalty``, ``basis_kind``,
    ``basis_order``, ``periodic``, and ``period`` are added so a caller
    can replay the exact same basis at predict time without recomputing
    anything — pass them straight back into ``duchon_basis`` or
    ``bspline_basis``.
    """
    payload["knots_or_centers"] = knots_or_centers
    payload["penalty"] = penalty
    payload["basis_kind"] = str(basis_kind)
    payload["basis_order"] = int(basis_order)
    payload["periodic"] = bool(periodic)
    payload["period"] = None if period is None else float(period)
    return payload


def _resolve_position_penalty(
    penalty: Any | None,
    knots_or_centers: Any,
    *,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
    period: float | None = None,
) -> Any:
    """Resolve the canonical single-λ penalty for position-based REML helpers.

    These helpers fit ONE smoothing ``λ`` against a position-indexed design,
    so each basis maps to a single SPD penalty matrix. Power users can
    override with an explicit matrix in ``penalty``; the string form selects
    a non-default canonical penalty for the same basis.

    Basis → default single-λ penalty:

    * ``"duchon"``  → single-λ smoothness penalty for the cubic basis
      (:func:`duchon_function_norm_penalty`). This is the convenience single-λ
      object, **not** the multi-λ structural smoother. For the amplitude/slope/
      curvature smoother with three separate REML ``λ``s — the object the
      formula API and :class:`gamfit.smooth.Duchon` fit — use the formula
      smooth; there is no way to express three independent ``λ``s through this
      single-penalty position helper.
    * ``"thinplate"`` (1D ``t``) → 1D thin-plate ≡ cubic smoothing spline,
      routed through the cubic-basis single-λ penalty at ``m=2``.
    * ``"bspline"``  → P-spline 2nd-difference coefficient penalty.
    """
    import numpy as np

    if isinstance(penalty, str):
        penalty_kind = str(penalty).strip().lower().replace("_", "-")
    else:
        penalty_kind = None

    if isinstance(penalty, str) or penalty is None:
        kind = str(basis_kind).strip().lower().replace("_", "").replace("-", "")
        if kind in {"duchon", "duchonspline"}:
            if penalty_kind in {None, "function-norm", "functionnorm", "rkhs", "smoothness"}:
                return duchon_function_norm_penalty(
                    np.asarray(knots_or_centers, dtype=float).reshape(-1, 1),
                    m=int(basis_order),
                    periodic_per_axis=[True] if periodic else None,
                    # Honor the explicit domain-wrap period so the periodic Gram
                    # matches the basis (both now use the same period). Passing
                    # None here auto-derived the center span and produced a
                    # non-PSD penalty (gam#580).
                    period=period,
                )
            if penalty_kind in {"triple-operator", "tripleoperator", "operator"}:
                raise ValueError(
                    "the triple-operator (amplitude + slope + curvature) penalty has THREE "
                    "independent REML smoothing parameters and cannot be collapsed into the "
                    "single-λ position helper. Fit it through the formula API "
                    "(gamfit.smooth.Duchon / basis_kind='duchon'), which routes each operator "
                    "to its own λ."
                )
            raise ValueError(f"unsupported Duchon penalty {penalty!r}")
        if kind in {"duchonmultipenalty", "duchontripleoperator"}:
            # The amplitude/slope/curvature smoother has THREE independent REML
            # smoothing parameters; summing the three operators into one matrix
            # here would force them to share a single λ, which is a different
            # (and weaker) object than the structural smoother. The position
            # helper only carries one λ, so there is no faithful single-matrix
            # representation — route the user to the formula smooth instead.
            raise ValueError(
                "basis_kind='duchon_multipenalty' is the multi-λ amplitude/slope/curvature "
                "smoother and cannot be expressed as a single position-helper penalty. Fit it "
                "through the formula API (gamfit.smooth.Duchon / basis_kind='duchon'), which "
                "gives each operator its own REML λ."
            )
        if kind in {"bspline", "spline"}:
            if penalty_kind not in {None, "coefficient-difference", "coefficientdifference", "difference"}:
                raise ValueError(f"unsupported B-spline penalty {penalty!r}")
            if periodic:
                k = int(np.asarray(knots_or_centers, dtype=float).size - 1)
                return np.asarray(
                    rust_module().cyclic_difference_penalty(k, 2), dtype=float
                )
            s, _ = smoothness_penalty(knots_or_centers, degree=int(basis_order), order=2)
            return s
        if kind in {"thinplate", "thinplatespline", "tps"}:
            # 1D thin-plate spline = cubic smoothing spline = Duchon m=2 with
            # bending-energy ``∫(f'')² dx``. Position-API positions are 1D,
            # so route the bending-energy penalty through the Duchon m=2
            # function-norm helper rather than the multi-dimensional thin-plate
            # path (which expects 2D centers and a different kernel).
            if penalty_kind not in {None, "function-norm", "functionnorm", "bending-energy", "bendingenergy"}:
                raise ValueError(f"unsupported thin-plate penalty {penalty!r}")
            return duchon_function_norm_penalty(
                np.asarray(knots_or_centers, dtype=float).reshape(-1, 1),
                m=2,
                periodic_per_axis=[True] if periodic else None,
                period=None,
            )
    return _numeric_matrix(penalty, "penalty")


def _normalize_position_basis(
    basis_kind: str,
    basis_order: int | None,
) -> tuple[str, int, str]:
    """Resolve user-facing basis names to the engine's basis kind + order.

    Returns ``(effective_kind, effective_order, display_kind)``.

    * ``thinplate`` (1D positions): an alias for Duchon ``m=2`` — the cubic
      smoothing spline is the canonical 1D thin-plate spline.

    ``duchon_multipenalty`` is rejected here: the amplitude/slope/curvature
    smoother carries three independent REML ``λ``s and has no single-λ
    position-helper form — fit it through the formula API instead.
    """
    raw = str(basis_kind)
    kind = raw.strip().lower().replace("_", "").replace("-", "")
    if kind in {"thinplate", "thinplatespline", "tps"}:
        order = 2 if basis_order is None else int(basis_order)
        return ("duchon", order, raw)
    if kind in {"duchonmultipenalty", "duchontripleoperator"}:
        raise ValueError(
            "basis_kind='duchon_multipenalty' is the multi-λ amplitude/slope/curvature "
            "smoother and has no single-λ position-helper representation. Fit it through the "
            "formula API (gamfit.smooth.Duchon / basis_kind='duchon')."
        )
    return (raw, _position_basis_order(raw, basis_order), raw)


def _numeric_tensor3(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"{label} must be a 3D numeric array")
    if 0 in arr.shape:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


# ---------------------------------------------------------------------------
# Kwarg-typo "Did you mean" hint, applied once from a single registry.
# ---------------------------------------------------------------------------
#
# Issue #306: ``fit()`` / ``fit_array()`` / ``validate_formula()`` expose ~25
# keyword arguments each. Case typos like ``formuLa=`` / ``Familiy=`` /
# ``Offset=`` produce a bare ``TypeError`` with no spelling hint on Python
# < 3.13 (Python 3.13 adds these natively via PEP 657).
#
# The principled fix is *one* registry of "kwarg-validated public entry
# points" and *one* place that wraps them. Adding a new public entry point
# means adding its name to ``_KWARG_VALIDATED_ENTRY_POINTS`` and nothing
# else — no per-function ``@_suggest_kwarg_typo`` decoration to remember to
# stamp on the new definition. Removing the wrapping from a function means
# removing the name from the registry; there is no second source of truth.
#
# The registry holds *names* rather than function references so resolution
# happens against the final, post-overload, post-definition binding in
# ``globals()``. (``fit`` has ``@overload`` stubs; the wrappable target is
# the concrete implementation that ends up bound to the name.)

_KWARG_VALIDATED_ENTRY_POINTS: tuple[str, ...] = (
    "fit",
    "fit_array",
    "validate_formula",
)


def _install_kwarg_typo_hints() -> None:
    """Wrap every entry point in ``_KWARG_VALIDATED_ENTRY_POINTS`` once.

    Each wrapper is closed over the function's signature at install time,
    so the known-keyword set is captured exactly once and reused on every
    call. The wrapper is otherwise a no-op: it observes the call, catches a
    ``TypeError`` matching the "unexpected keyword argument" shape, and
    re-raises the same error with an appended ``. Did you mean 'Y'?`` hint.
    """
    module_globals = globals()
    for name in _KWARG_VALIDATED_ENTRY_POINTS:
        target = module_globals.get(name)
        if target is None:
            raise RuntimeError(
                f"_KWARG_VALIDATED_ENTRY_POINTS lists {name!r} but no such "
                f"function is defined in gamfit._api; remove it from the "
                f"registry or define the function."
            )
        module_globals[name] = _suggest_kwarg_typo(target)


_install_kwarg_typo_hints()
