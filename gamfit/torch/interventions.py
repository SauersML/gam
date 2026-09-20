"""Rung-3 patch runner — execute (token, atom, Δt) interventions and record
realized KL, at the model-interaction boundary.

This is the torch end of the Rung-3 intervention contract
(``crates/gam-sae/src/inference/RUNG3_INTERVENTIONS_DESIGN.md``, §5–6; the Rust
shard type is ``gam_sae::inference::intervention_shard::InterventionShard``).
The runner is deliberately **chart-agnostic**: the caller (real chart, or a
mock in tests) decodes each move into a p-space delta ``Δx`` and supplies the
predicted nats; the runner only splices ``x + Δx`` at the hook site, reruns the
rest of the network, and measures the realized same-position
``KL(p_clean ‖ p_patched)`` in nats. That keeps the model-touching surface
free of chart logic — the Goodhart-guard boundary (design guard G1) is easier
to audit when the only thing this module can do is *measure*.

Splicing reuses the exact forward-hook path the downstream harvest exercises
(:func:`gamfit.torch.harvest._capture_activations`'s replace-one-row closure),
so a patched forward is the same code path as a probed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .harvest import _capture_activations

__all__ = [
    "InterventionPlan",
    "InterventionShardData",
    "kl_and_logit_extent",
    "logit_format_name",
    "run_interventions",
]


@dataclass(frozen=True)
class InterventionPlan:
    """The sampled intervention plan: what to splice, where, with which
    prediction attached.

    All arrays share leading length ``m`` (one record per intervention).

    Attributes
    ----------
    row
        ``(m,)`` int — token row (index into the hook site's flattened rows)
        each intervention applies at.
    delta_x
        ``(m, p)`` float — the decoded p-space move ``Δx`` to splice
        (``x_row + Δx`` replaces ``x_row``). All-zero rows are the Δt = 0
        control splices (design guard G3's measurement null).
    atom
        ``(m,)`` int — atom index the move came from (bookkeeping only; the
        runner never reads it).
    dose
        ``(m, d)`` float — the latent move Δt (bookkeeping only).
    nu_hat_1
        ``(m,)`` float — Rung-1 predicted nats ``½ Δxᵀ G_n Δx``.
    nu_hat_2
        ``(m,)`` float or None — Rung-2 predicted nats (behavior decoder);
        None when the fit carried no y-block.
    group
        ``(m,)`` int — document/question id (the G2 split unit).
    """

    row: Any
    delta_x: Any
    atom: Any
    dose: Any
    nu_hat_1: Any
    nu_hat_2: Any
    group: Any

    def __post_init__(self) -> None:
        row = np.asarray(self.row)
        dx = np.asarray(self.delta_x)
        if row.ndim != 1:
            raise ValueError(f"row must be (m,); got shape {row.shape}")
        m = row.shape[0]
        if dx.ndim != 2 or dx.shape[0] != m:
            raise ValueError(f"delta_x must be (m, p) = ({m}, p); got shape {dx.shape}")
        dose = np.asarray(self.dose)
        if dose.ndim != 2 or dose.shape[0] != m:
            raise ValueError(f"dose must be (m, d) = ({m}, d); got shape {dose.shape}")
        for name, arr in (
            ("atom", np.asarray(self.atom)),
            ("nu_hat_1", np.asarray(self.nu_hat_1)),
            ("group", np.asarray(self.group)),
        ):
            if arr.shape != (m,):
                raise ValueError(f"{name} must be (m,) = ({m},); got shape {arr.shape}")
        if self.nu_hat_2 is not None and np.asarray(self.nu_hat_2).shape != (m,):
            raise ValueError(f"nu_hat_2 must be (m,) = ({m},) or None")
        if not np.all(np.isfinite(dx)):
            raise ValueError("delta_x must be finite")


@dataclass(frozen=True)
class InterventionShardData:
    """Executed interventions: the plan's bookkeeping plus the measurement.

    Field-for-field the Rust ``InterventionShard`` contract (design §6):
    ``is_control`` is derived from the *applied* ``delta_x`` (all-zero ⇒
    control), never trusted from the caller — a mislabeled control is the one
    error the G3 null cannot survive.

    ``nu_measured`` is the raw float64 KL, never clamped. ``logit_format``,
    ``vocab_size``, ``logit_max_abs`` and ``logit_max_abs_change`` are what
    the Rust core needs to derive each record's measurement band: the KL that
    rounding of the logits at their format, plus the float64 evaluation of
    the KL, can produce by itself.
    """

    row_id: Any
    atom: Any
    dose: Any
    nu_hat_1: Any
    nu_hat_2: Any
    nu_measured: Any
    logit_max_abs: Any
    logit_max_abs_change: Any
    group: Any
    is_control: Any
    layer: int
    seed: int
    logit_format: str
    vocab_size: int


_LOGIT_FORMATS = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}


def logit_format_name(dtype: torch.dtype) -> str:
    """The measurement-band name of a logit dtype: the format the logits were rounded in.

    Pass the dtype the model produced its logits in, before any cast: bfloat16 logits
    cast to float32 still carry bfloat16 spacing.
    """
    if dtype not in _LOGIT_FORMATS:
        raise ValueError(f"logits must be float16, bfloat16, float32 or float64; got {dtype}")
    return _LOGIT_FORMATS[dtype]


def kl_and_logit_extent(
    clean: torch.Tensor, patched: torch.Tensor
) -> tuple[float, float, float]:
    """``KL(softmax(clean) ‖ softmax(patched))`` in nats, unclamped, with the
    largest ``|logit|`` over both vectors and the largest ``|patched − clean|``.

    The float64 ``log_softmax``, ``exp`` and single sum below are the
    evaluation whose rounding the Rust measurement band bounds, so this
    sequence of operations is part of the calibration contract.
    """
    clean64 = clean.to(torch.float64)
    patched64 = patched.to(torch.float64)
    logp = torch.log_softmax(clean64, dim=-1)
    logq = torch.log_softmax(patched64, dim=-1)
    kl = float((logp.exp() * (logp - logq)).sum().item())
    max_abs = float(torch.maximum(clean64.abs().max(), patched64.abs().max()).item())
    max_change = float((patched64 - clean64).abs().max().item())
    return kl, max_abs, max_change


def run_interventions(
    model: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: Any,
    plan: InterventionPlan,
    *,
    layer: int,
    seed: int = 0,
) -> InterventionShardData:
    """Execute ``plan`` and return the measured shard.

    For each record: splice ``x_row + Δx`` at ``hook_module``'s output row
    ``row``, rerun the rest of the network through the same replace-one-row
    hook the downstream harvest uses, and record the realized same-position
    ``KL(p_clean ‖ p_patched)`` in nats, unclamped. ``Δx = 0`` records re-splice
    the *unchanged* row, so their measured KL is the G3 measurement null. On a
    deterministic model that null is exactly zero, and the Rust core floors
    each record at its derived measurement band instead. A nonzero control is
    genuine nondeterminism, which the control quantile sees.

    ``layer`` and ``seed`` are stamped into the shard (provenance). One
    forward pass per record plus one clean capture — no gradients anywhere
    (calibration mode never backprops through the LM).
    """
    act_flat, logits_from_act = _capture_activations(model, hook_module, inputs)
    n, p = int(act_flat.shape[0]), int(act_flat.shape[1])

    row = np.asarray(plan.row, dtype=np.int64)
    dx = np.asarray(plan.delta_x, dtype=np.float64)
    if dx.shape[1] != p:
        raise ValueError(
            f"plan.delta_x has p = {dx.shape[1]} but the hook site produced p = {p}"
        )
    if row.min(initial=0) < 0 or row.max(initial=-1) >= n:
        raise ValueError(f"plan.row indices must lie in [0, {n}); got range "
                         f"[{row.min()}, {row.max()}]")

    m = row.shape[0]
    if m == 0:
        raise ValueError("plan has no records; a shard needs at least one executed intervention")
    nu_measured = np.empty((m,), dtype=np.float64)
    logit_max_abs = np.empty((m,), dtype=np.float64)
    logit_max_abs_change = np.empty((m,), dtype=np.float64)
    is_control = np.zeros((m,), dtype=bool)
    logit_dtypes: set[torch.dtype] = set()
    vocab_sizes: set[int] = set()

    with torch.no_grad():
        for i in range(m):
            r = int(row[i])
            x_row = act_flat[r]
            delta = torch.from_numpy(dx[i]).to(dtype=x_row.dtype, device=x_row.device)
            clean = logits_from_act(x_row, r)
            patched = logits_from_act(x_row + delta, r)
            logit_dtypes.update((clean.dtype, patched.dtype))
            vocab_sizes.update((int(clean.shape[-1]), int(patched.shape[-1])))
            nu_measured[i], logit_max_abs[i], logit_max_abs_change[i] = kl_and_logit_extent(
                clean, patched
            )
            is_control[i] = bool(np.all(dx[i] == 0.0))

    if len(logit_dtypes) != 1 or len(vocab_sizes) != 1:
        raise ValueError(
            f"every record's logits must share one dtype and vocabulary size; got dtypes "
            f"{sorted(map(str, logit_dtypes))} and sizes {sorted(vocab_sizes)}"
        )
    (logit_dtype,) = logit_dtypes
    logit_format = logit_format_name(logit_dtype)
    (vocab_size,) = vocab_sizes

    return InterventionShardData(
        row_id=row,
        atom=np.asarray(plan.atom, dtype=np.int64),
        dose=np.asarray(plan.dose, dtype=np.float64),
        nu_hat_1=np.asarray(plan.nu_hat_1, dtype=np.float64),
        nu_hat_2=(
            None if plan.nu_hat_2 is None else np.asarray(plan.nu_hat_2, dtype=np.float64)
        ),
        nu_measured=nu_measured,
        logit_max_abs=logit_max_abs,
        logit_max_abs_change=logit_max_abs_change,
        group=np.asarray(plan.group, dtype=np.int64),
        is_control=is_control,
        layer=int(layer),
        seed=int(seed),
        logit_format=logit_format,
        vocab_size=vocab_size,
    )

