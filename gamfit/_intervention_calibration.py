"""Rung-3 chart calibration — a thin fitting adapter for the Rust design.

The typed Rust calibration plan owns the permanent split, the per-record
measurement floor (the derived measurement band, raised by the control
quantile where controls are stochastic), measurability statuses, log-scale transforms, fixed estimator specification,
gauge-centred chart re-speeds, and held-out diagnostic.  This module only
marshals :class:`gamfit.torch.interventions.InterventionShardData` into that
plan, invokes :func:`gamfit.fit`, and returns the core's result.

    ``log ν = β₀ + f(log ν̂) + b_atom + ε``

The core model remains ``log ν = β₀ + f(log ν̂) + b_atom + ε`` with a
monotone-increasing ``f`` and REML-selected smoothing/shrinkage.  Calibration's
only chart output is the positive re-speed map (guard G1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._binding import rust_module

__all__ = ["ChartCalibration", "fit_chart_calibration"]


@dataclass(frozen=True)
class ChartCalibration:
    """Calibration output. By design this carries ONLY gauge re-speeds and
    diagnostics — there is no field a fit criterion could consume (guard G1).

    Attributes
    ----------
    respeed
        ``{atom_id: s_k}`` — multiplicative chart-coordinate re-speed
        (``t ← s_k · t``), from the fitted per-atom random intercept
        ``s_k = exp(b_k / 2)``. Only atoms above the G3 floor appear.
    below_measurement_floor
        Atom ids with train interventions but no measured response above its
        record's floor.
    no_training_intervention
        Atom ids present in the shard but absent from train interventions.
    control_quantile_nats
        The ``floor_quantile`` of the Δt = 0 control measurements (train
        split), nats. It is exactly 0 on a deterministic model, whose controls
        re-splice the unchanged row.
    measurement_band_nats_max
        The largest measurement band over the train interventions: the KL that
        rounding of the logits at their format, plus the float64 evaluation of
        the KL, can produce by itself. Each record's floor is the larger of its
        band and ``control_quantile_nats``.
    heldout_rmse_lognats
        RMSE of ``log ν`` prediction on the eval-forever split, or ``None``
        when no eligible held-out record exists.
    n_train, n_eval
        Record counts entering the fit / the held-out report.
    """

    respeed: dict[int, float]
    below_measurement_floor: tuple[int, ...]
    no_training_intervention: tuple[int, ...]
    control_quantile_nats: float
    measurement_band_nats_max: float
    heldout_rmse_lognats: float | None
    n_train: int
    n_eval: int


def fit_chart_calibration(
    shard: Any,
    *,
    prediction: Literal["rung1", "rung2"],
    split_seed: int,
    floor_quantile: float,
) -> ChartCalibration:
    """Fit the core-prepared §3 design and return its chart-safe result.

    ``shard`` is the field-for-field executed-intervention contract produced by
    :func:`gamfit.torch.interventions.run_interventions`.  All validation and
    policy decisions occur in Rust; array conversion here is boundary
    marshaling only.
    """
    nu_hat_2 = (
        None
        if shard.nu_hat_2 is None
        else np.ascontiguousarray(shard.nu_hat_2, dtype=np.float64)
    )
    plan = rust_module().intervention_calibration_plan(
        np.ascontiguousarray(shard.row_id, dtype=np.int64),
        np.ascontiguousarray(shard.atom, dtype=np.int64),
        np.ascontiguousarray(shard.dose, dtype=np.float64),
        np.ascontiguousarray(shard.nu_hat_1, dtype=np.float64),
        nu_hat_2,
        np.ascontiguousarray(shard.nu_measured, dtype=np.float64),
        np.ascontiguousarray(shard.logit_max_abs, dtype=np.float64),
        np.ascontiguousarray(shard.logit_max_abs_change, dtype=np.float64),
        np.ascontiguousarray(shard.group, dtype=np.int64),
        np.ascontiguousarray(shard.is_control, dtype=bool),
        int(shard.layer),
        int(shard.seed),
        str(shard.logit_format),
        int(shard.vocab_size),
        prediction,
        int(split_seed),
        float(floor_quantile),
    )

    import gamfit

    fitted = gamfit.fit(
        dict(plan.fit_frame()),
        plan.formula,
        constraints=dict(plan.constraints()),
    )
    reference_eta = np.asarray(
        fitted.predict(dict(plan.reference_frame())), dtype=np.float64
    ).reshape(-1)
    eval_eta = (
        np.asarray(fitted.predict(dict(plan.eval_frame())), dtype=np.float64).reshape(-1)
        if plan.n_eval
        else np.empty(0, dtype=np.float64)
    )
    payload = dict(plan.finish(reference_eta.tolist(), eval_eta.tolist()))
    return ChartCalibration(
        respeed=dict(payload["respeed"]),
        below_measurement_floor=tuple(payload["below_measurement_floor"]),
        no_training_intervention=tuple(payload["no_training_intervention"]),
        control_quantile_nats=float(payload["control_quantile_nats"]),
        measurement_band_nats_max=float(payload["measurement_band_nats_max"]),
        heldout_rmse_lognats=payload["heldout_rmse_lognats"],
        n_train=int(payload["n_train"]),
        n_eval=int(payload["n_eval"]),
    )
