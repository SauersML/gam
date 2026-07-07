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
    "MeasureSpikeEdit",
    "harmonic_code_features",
    "synthesize_measure_code",
    "apply_spike_edit",
    "spike_edit_code_delta",
    "spike_edit_delta_x",
    "build_spike_intervention_plan",
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


# --------------------------------------------------------------------------- #
# Spike-level measure edits (App C — binding / multiplicity)
# --------------------------------------------------------------------------- #
# A harmonic-circle atom stores, per firing, the within-block code
# ``z = Σ_j a_j u(t_j) ∈ ℝ^{2H}`` — the *superposition* of ``m`` point masses on
# the circle (see ``gam_sae::super_resolution``). Super-resolution recovers the
# measure ``{(a_j, t_j)}``; this section is its causal inverse: edit ONE point
# mass (remove / move / rescale it), re-synthesize the block code, and lift the
# code change through the atom's decoder ``D`` (``2H × p``) into a p-space delta
# ``Δx`` ready for :func:`run_interventions`. Because the edit touches a single
# spike, the injected ``Δx`` is the contribution of that one *instance* alone —
# the other instances of the same feature are provably untouched, which is the
# claim a linear SAE cannot even express (it has no per-instance handle).


@dataclass(frozen=True)
class MeasureSpikeEdit:
    """A single-spike edit of a recovered circle measure.

    Attributes
    ----------
    kind
        ``"remove"`` (delete the spike), ``"move"`` (retune its position to
        ``new_t``), or ``"scale"`` (multiply its amplitude by ``factor``).
    spike_index
        Index into the row's spike list (positions sorted ascending, matching
        the Rust readout's ordering).
    new_t
        Target circle position for ``"move"`` (in ``[0, 1)``); ignored otherwise.
    factor
        Amplitude multiplier for ``"scale"`` (``0`` reproduces ``"remove"``);
        ignored otherwise.
    """

    kind: str
    spike_index: int
    new_t: float | None = None
    factor: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("remove", "move", "scale"):
            raise ValueError(f"kind must be remove|move|scale; got {self.kind!r}")
        if self.spike_index < 0:
            raise ValueError("spike_index must be non-negative")
        if self.kind == "move" and self.new_t is None:
            raise ValueError("move edit requires new_t")
        if self.kind == "scale" and self.factor is None:
            raise ValueError("scale edit requires factor")


def harmonic_code_features(t: Any, n_harmonics: int) -> np.ndarray:
    """Harmonic frame row(s) ``u(t) = [cos 2πt, sin 2πt, ..., cos 2πHt,
    sin 2πHt]`` for harmonics ``1..H``. Scalar ``t`` returns ``(2H,)``; a ``(k,)``
    array returns ``(k, 2H)``. This is the exact convention
    ``gam_sae::sparse_dict::coordinate`` uses for the ``(c_h, s_h)`` code."""
    t_arr = np.asarray(t, dtype=np.float64)
    scalar = t_arr.ndim == 0
    t_col = np.atleast_1d(t_arr)
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * h * t_col))
        cols.append(np.sin(2.0 * np.pi * h * t_col))
    feats = np.stack(cols, axis=1)  # (k, 2H)
    return feats[0] if scalar else feats


def synthesize_measure_code(spikes: Any, n_harmonics: int) -> np.ndarray:
    """Re-synthesize the ``2H`` within-block code ``z = Σ_j a_j u(t_j)`` from a
    list of ``(amplitude, t)`` point masses. The exact forward map super-
    resolution inverts; an empty measure yields the zero code."""
    z = np.zeros(2 * n_harmonics, dtype=np.float64)
    for amplitude, t in spikes:
        z += float(amplitude) * harmonic_code_features(float(t), n_harmonics)
    return z


def apply_spike_edit(spikes: Any, edit: MeasureSpikeEdit) -> list[tuple[float, float]]:
    """Return a new ``(amplitude, t)`` measure with ``edit`` applied to one spike.
    The input is left unmodified; ``"remove"`` drops the spike, ``"move"`` retunes
    its position, ``"scale"`` multiplies its amplitude."""
    out = [(float(a), float(t)) for a, t in spikes]
    if edit.spike_index >= len(out):
        raise IndexError(
            f"spike_index {edit.spike_index} out of range for {len(out)} spikes"
        )
    a, t = out[edit.spike_index]
    if edit.kind == "remove":
        del out[edit.spike_index]
    elif edit.kind == "move":
        out[edit.spike_index] = (a, float(edit.new_t) % 1.0)
    else:  # scale
        out[edit.spike_index] = (a * float(edit.factor), t)
    return out


def spike_edit_code_delta(spikes: Any, edit: MeasureSpikeEdit, n_harmonics: int) -> np.ndarray:
    """The ``2H`` change ``Δz = synth(edited) − synth(original)`` for a single-
    spike edit. For ``"remove"`` this is exactly ``−a_j u(t_j)`` — the isolated
    contribution of the removed instance, nothing else."""
    before = synthesize_measure_code(spikes, n_harmonics)
    after = synthesize_measure_code(apply_spike_edit(spikes, edit), n_harmonics)
    return after - before


def spike_edit_delta_x(decoder: Any, spikes: Any, edit: MeasureSpikeEdit) -> np.ndarray:
    """Lift a single-spike code edit into the atom's ambient p-space:
    ``Δx = Δz · D`` where ``decoder`` is the ``2H × p`` block decoder (code →
    activation). The returned ``(p,)`` vector is the p-space move that removes /
    moves / rescales exactly one instance of the circle feature; add it to the
    token's activation to inject the edit. If the atom lives in a reduced basis
    (e.g. sink-peeled PCA), map this through that basis to reach the model's
    residual stream — see the App C design note."""
    D = np.asarray(decoder, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"decoder must be 2-D (2H x p); got shape {D.shape}")
    n_harmonics = D.shape[0] // 2
    if 2 * n_harmonics != D.shape[0]:
        raise ValueError(f"decoder rows {D.shape[0]} must be even (2H)")
    delta_z = spike_edit_code_delta(spikes, edit, n_harmonics)
    return delta_z @ D


def build_spike_intervention_plan(
    decoder: Any,
    rows: Any,
    measures: Any,
    edits: Any,
    *,
    basis: Any | None = None,
    atom: int = 0,
    group: Any | None = None,
) -> InterventionPlan:
    """Assemble an :class:`InterventionPlan` of single-spike edits.

    ``rows`` are the token rows to edit; ``measures[i]`` is row ``i``'s recovered
    ``(amplitude, t)`` spike list; ``edits[i]`` is its :class:`MeasureSpikeEdit`.
    Each record's ``delta_x`` is the p-space single-spike move
    (:func:`spike_edit_delta_x`); when ``basis`` (``p_frame × p_model``) is given
    the move is mapped ``Δx_model = Δx_frame · basis`` so it lands in the model's
    residual stream. ``dose`` records ``[Δt]`` (the position change, ``0`` for
    remove/scale) and ``nu_hat_1`` is left ``0`` (this plan measures realized KL,
    not a Rung-1 prediction). Feed the result straight to
    :func:`run_interventions`."""
    rows = np.asarray(rows, dtype=np.int64)
    m = rows.shape[0]
    if len(measures) != m or len(edits) != m:
        raise ValueError("rows, measures, edits must share length m")
    deltas = []
    doses = []
    for i in range(m):
        dx = spike_edit_delta_x(decoder, measures[i], edits[i])
        if basis is not None:
            dx = dx @ np.asarray(basis, dtype=np.float64)
        deltas.append(dx)
        edit = edits[i]
        if edit.kind == "move":
            a, t = measures[i][edit.spike_index]
            doses.append([float(edit.new_t) % 1.0 - float(t)])
        else:
            doses.append([0.0])
    delta_x = np.asarray(deltas, dtype=np.float64)
    dose = np.asarray(doses, dtype=np.float64)
    grp = np.arange(m, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)
    return InterventionPlan(
        row=rows,
        delta_x=delta_x,
        atom=np.full((m,), int(atom), dtype=np.int64),
        dose=dose,
        nu_hat_1=np.zeros((m,), dtype=np.float64),
        nu_hat_2=None,
        group=grp,
    )
