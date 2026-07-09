"""Rung-3 chart calibration — fit measured-vs-predicted nats with gamfit.

The estimator of ``RUNG3_INTERVENTIONS_DESIGN.md`` §3, driven end-to-end
through :func:`gamfit.fit` (the Rust engine underneath — this module is a thin
assembly layer, no math of its own):

    ``log ν = β₀ + f(log ν̂) + b_atom + ε``

* ``f`` — a monotone-increasing smooth (``constraints={...:
  "monotone_increasing"}``): the calibration curve. Its departure from the
  identity is the systematic decay of the quadratic (Fisher) prediction with
  dose.
* ``b_atom`` — a random intercept per atom (``re(atom)``): atom k's log speed
  error. ``s_k = exp(b_k / 2)`` is the **chart re-speed factor**, the ONLY
  object calibration may write back to a chart (guard G1: the output type
  carries re-speeds and diagnostics, nothing that could reach a fit
  criterion).
* REML selects every smoothing/shrinkage level; nothing here is tuned.

Guards enforced here:

* **G2** — the eval-forever split is a pure per-group function of
  ``(group id, seed)``, computed by the Rust
  ``intervention_shard::eval_forever_mask`` (the single source of the
  SplitMix64 split, reference values pinned in its test suite). This module
  delegates to that binding, so the split is bit-identical to the Rust
  ``InterventionShard::eval_forever_split`` by construction. Eval groups never
  enter the fit; held-out error is reported from them.
* **G3** — the measurement floor is the caller-chosen quantile of the
  Δt = 0 control measurements (never a constant). Atoms whose train records
  all fall at or below the floor are excluded from the fit and reported as
  ``unmeasurable`` — a finding, not a silent calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ChartCalibration", "fit_chart_calibration"]

_MASK64 = (1 << 64) - 1


def _eval_forever_mask(group: np.ndarray, seed: int) -> np.ndarray:
    """True where the record's group is eval-forever (G2).

    Delegates to the Rust ``intervention_shard::eval_forever_mask`` (exposed as
    ``intervention_eval_forever_mask``) — the single source of the SplitMix64
    split. This module never re-derives the hash, so the assignment stays
    bit-identical to the Rust side by construction rather than by a duplicated
    Python implementation (SPEC rules 8-9).
    """
    from ._binding import rust_module

    mask = rust_module().intervention_eval_forever_mask(
        [int(g) for g in np.asarray(group, dtype=np.int64).reshape(-1)],
        int(seed) & _MASK64,
    )
    return np.asarray(mask, dtype=bool)


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
    unmeasurable
        Atom ids whose train-split, non-control measurements never exceeded
        the floor at any applied dose — reported, never calibrated.
    floor_nats
        The G3 floor: the ``floor_quantile`` of the Δt = 0 control
        measurements (train split), nats.
    heldout_rmse_lognats
        RMSE of ``log ν`` prediction on the eval-forever split (the standing
        "predicted nats mean what they say" number).
    n_train, n_eval
        Record counts entering the fit / the held-out report.
    """

    respeed: dict[int, float]
    unmeasurable: tuple[int, ...]
    floor_nats: float
    heldout_rmse_lognats: float
    n_train: int
    n_eval: int


def fit_chart_calibration(
    nu_hat: np.ndarray,
    nu_measured: np.ndarray,
    atom: np.ndarray,
    group: np.ndarray,
    is_control: np.ndarray,
    *,
    split_seed: int,
    floor_quantile: float,
) -> ChartCalibration:
    """Fit the §3 calibration GAM on the train split and report held-out error.

    Parameters mirror the intervention shard: per-record predicted nats
    ``nu_hat`` (use ``nu_hat_1``, or ``nu_hat_2`` to calibrate the Rung-2
    prediction instead), measured nats, atom id, group id (G2 unit), and the
    control flag. ``floor_quantile`` is the caller's one-sided evidence
    quantile for the G3 floor (the same convention the certificates use — a
    caller decision, not a constant baked here).
    """
    nu_hat = np.asarray(nu_hat, dtype=np.float64)
    nu_measured = np.asarray(nu_measured, dtype=np.float64)
    atom = np.asarray(atom, dtype=np.int64)
    group = np.asarray(group, dtype=np.int64)
    is_control = np.asarray(is_control, dtype=bool)
    m = nu_hat.shape[0]
    for name, arr in (
        ("nu_measured", nu_measured),
        ("atom", atom),
        ("group", group),
        ("is_control", is_control),
    ):
        if arr.shape != (m,):
            raise ValueError(f"{name} must be ({m},); got {arr.shape}")
    if not (0.0 < floor_quantile < 1.0):
        raise ValueError(f"floor_quantile must be in (0, 1); got {floor_quantile}")

    eval_mask = _eval_forever_mask(group, split_seed)
    train = ~eval_mask

    # G3 floor from the TRAIN controls (eval stays untouched by every choice).
    train_controls = nu_measured[train & is_control]
    if train_controls.size == 0:
        raise ValueError(
            "no Δt = 0 control records in the train split; the G3 floor must be "
            "estimated from controls, never assumed"
        )
    floor = float(np.quantile(train_controls, floor_quantile))

    # Screen atoms: an atom is measurable iff any train non-control record
    # exceeded the floor.
    unmeasurable: list[int] = []
    measurable: set[int] = set()
    for a in np.unique(atom):
        sel = train & ~is_control & (atom == a)
        if not sel.any():
            continue
        if np.max(nu_measured[sel]) > floor:
            measurable.add(int(a))
        else:
            unmeasurable.append(int(a))

    fit_sel = train & ~is_control & np.isin(atom, sorted(measurable)) & (nu_hat > 0.0)
    if fit_sel.sum() < 3:
        raise ValueError(
            f"only {int(fit_sel.sum())} usable train records above the floor; "
            "calibration needs more interventions"
        )

    import gamfit

    # Measured values at/below the floor are indistinguishable from the null;
    # lift them to the floor before the log so a single lucky near-zero cannot
    # dominate the log-scale fit (the floor is the resolution limit, estimated
    # from the controls — not a tuning constant).
    y = np.log(np.maximum(nu_measured, floor))
    x = np.log(nu_hat, where=nu_hat > 0.0, out=np.full_like(nu_hat, -np.inf))

    frame = {
        "log_nu": y[fit_sel].tolist(),
        "log_nu_hat": x[fit_sel].tolist(),
        "atom": [str(a) for a in atom[fit_sel]],
    }
    result = gamfit.fit(
        frame,
        "log_nu ~ s(log_nu_hat) + re(atom)",
        constraints={"s(log_nu_hat)": "monotone_increasing"},
    )

    # Per-atom re-speed from the fitted random intercept, extracted through the
    # public prediction surface (no reach into solver internals): predict every
    # measurable atom at one fixed log_nu_hat; differences from the mean are
    # exactly the centered b_k.
    ref_x = float(np.median(x[fit_sel]))
    atoms_sorted = sorted(measurable)
    eta = np.asarray(
        result.predict(
            {
                "log_nu_hat": [ref_x] * len(atoms_sorted),
                "atom": [str(a) for a in atoms_sorted],
            }
        ),
        dtype=np.float64,
    ).reshape(-1)
    b = eta - float(np.mean(eta))
    respeed = {a: float(np.exp(bk / 2.0)) for a, bk in zip(atoms_sorted, b)}

    # Held-out report (G2): eval-forever, non-control, measurable-atom records.
    hold = eval_mask & ~is_control & np.isin(atom, atoms_sorted) & (nu_hat > 0.0)
    if hold.any():
        pred = np.asarray(
            result.predict(
                {
                    "log_nu_hat": x[hold].tolist(),
                    "atom": [str(a) for a in atom[hold]],
                }
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_rmse = float(np.sqrt(np.mean((pred - y[hold]) ** 2)))
    else:
        heldout_rmse = float("nan")

    return ChartCalibration(
        respeed=respeed,
        unmeasurable=tuple(sorted(unmeasurable)),
        floor_nats=floor,
        heldout_rmse_lognats=heldout_rmse,
        n_train=int(fit_sel.sum()),
        n_eval=int(hold.sum()),
    )
