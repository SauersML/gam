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

The shard `.npz` I/O mirrors :func:`gamfit.torch.harvest.save_harvest_shard`:
f32/f64 as measured, validated shapes, provenance-free (the shard *is* its own
provenance: the plan and seed are stored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .harvest import _capture_activations

__all__ = [
    "InterventionPlan",
    "InterventionShardData",
    "run_interventions",
    "save_intervention_shard",
    "load_intervention_shard",
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
    """

    row_id: Any
    atom: Any
    dose: Any
    nu_hat_1: Any
    nu_hat_2: Any
    nu_measured: Any
    group: Any
    is_control: Any
    layer: int
    seed: int


def _kl_from_logits(clean: torch.Tensor, patched: torch.Tensor) -> float:
    """``KL(softmax(clean) ‖ softmax(patched))`` in nats, computed stably in
    log space (no explicit normalization subtraction dance)."""
    logp = torch.log_softmax(clean.to(torch.float64), dim=-1)
    logq = torch.log_softmax(patched.to(torch.float64), dim=-1)
    p = logp.exp()
    return float((p * (logp - logq)).sum().item())


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
    ``KL(p_clean ‖ p_patched)`` in nats. ``Δx = 0`` records re-splice the
    *unchanged* row — their measured KL is the G3 measurement null (exactly
    zero for a deterministic model through this same-path splice; any nonzero
    value is real measurement noise the floor should see).

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
    nu_measured = np.empty((m,), dtype=np.float64)
    is_control = np.zeros((m,), dtype=bool)

    with torch.no_grad():
        for i in range(m):
            r = int(row[i])
            x_row = act_flat[r]
            delta = torch.from_numpy(dx[i]).to(dtype=x_row.dtype, device=x_row.device)
            clean = logits_from_act(x_row, r)
            patched = logits_from_act(x_row + delta, r)
            nu_measured[i] = max(_kl_from_logits(clean, patched), 0.0)
            is_control[i] = bool(np.all(dx[i] == 0.0))

    return InterventionShardData(
        row_id=row,
        atom=np.asarray(plan.atom, dtype=np.int64),
        dose=np.asarray(plan.dose, dtype=np.float64),
        nu_hat_1=np.asarray(plan.nu_hat_1, dtype=np.float64),
        nu_hat_2=(
            None if plan.nu_hat_2 is None else np.asarray(plan.nu_hat_2, dtype=np.float64)
        ),
        nu_measured=nu_measured,
        group=np.asarray(plan.group, dtype=np.int64),
        is_control=is_control,
        layer=int(layer),
        seed=int(seed),
    )


def save_intervention_shard(shard: InterventionShardData, path: str | Path) -> str:
    """Write the shard to ``.npz`` (the harvest-shard suffix rule). ``nu_hat_2``
    absence is stored as an empty array so the schema is stable."""
    out = Path(path)
    if out.suffix != ".npz":
        out = out.with_name(out.name + ".npz")
    nu2 = (
        np.asarray([], dtype=np.float64)
        if shard.nu_hat_2 is None
        else np.asarray(shard.nu_hat_2, dtype=np.float64)
    )
    np.savez(
        out,
        row_id=np.asarray(shard.row_id, dtype=np.int64),
        atom=np.asarray(shard.atom, dtype=np.int64),
        dose=np.asarray(shard.dose, dtype=np.float64),
        nu_hat_1=np.asarray(shard.nu_hat_1, dtype=np.float64),
        nu_hat_2=nu2,
        nu_measured=np.asarray(shard.nu_measured, dtype=np.float64),
        group=np.asarray(shard.group, dtype=np.int64),
        is_control=np.asarray(shard.is_control, dtype=bool),
        layer=np.int64(shard.layer),
        seed=np.uint64(shard.seed),
    )
    return str(out)


def load_intervention_shard(path: str | Path) -> InterventionShardData:
    """Load a shard written by :func:`save_intervention_shard`."""
    target = Path(path)
    if not target.exists() and target.suffix != ".npz":
        suffixed = target.with_name(target.name + ".npz")
        if suffixed.exists():
            target = suffixed
    npz = np.load(target)
    nu2 = np.asarray(npz["nu_hat_2"], dtype=np.float64)
    return InterventionShardData(
        row_id=np.asarray(npz["row_id"], dtype=np.int64),
        atom=np.asarray(npz["atom"], dtype=np.int64),
        dose=np.asarray(npz["dose"], dtype=np.float64),
        nu_hat_1=np.asarray(npz["nu_hat_1"], dtype=np.float64),
        nu_hat_2=None if nu2.size == 0 else nu2,
        nu_measured=np.asarray(npz["nu_measured"], dtype=np.float64),
        group=np.asarray(npz["group"], dtype=np.int64),
        is_control=np.asarray(npz["is_control"], dtype=bool),
        layer=int(npz["layer"].item()),
        seed=int(npz["seed"].item()),
    )
