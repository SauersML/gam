#!/usr/bin/env python3
"""Bounded-approximation audit of one #2283 Eq-4 row (gam#2233 Task 3).

The authoritative bits-at-R2 row is only worth the ledger it is charged against.
This module answers, for ONE scored featurizer, the questions the Eq-4 scorer's
own approximations raise. It answers them by OBSERVING the production scorer
(``gamfit._description_length`` -> ``sae_eq4_description_length``) through the
callback it already drives, never by re-deriving the scorer's rules in Python:
the firing threshold, the skip rule and the spectrum row cap stay owned by Rust,
and the audit reads off which atoms the scorer actually fetched and with how many
rows.

1. **Does the scored surface reconstruct the scored model?** Support and code
   bits are read off ``gate`` / ``atom_contribution``; the residual term is read
   off ``recon``. If the atom contributions do not sum back to ``recon`` (up to
   one constant pre-bias row, which carries no per-token code) then the two
   halves of the score are pricing different models.
2. **Is the rank-one flat fast path exact on this data?** ``atom_code_spectrum``
   prices a ``code_dim == 1`` atom by its centered Frobenius norm, which is exact
   iff the contribution is genuinely rank one. Measured here as the spectral mass
   beyond the first eigenvalue, per atom.
3. **What does the skip rule drop?** Atoms firing on too few sampled rows get a
   zero spectrum, so their code bits are never charged. The atoms the scorer
   declined to fetch are recorded exactly, together with the share of the arm's
   firings they carry -- the size of that undercharge.
4. **Is the support term charged in a currency both arms deserve?** The Eq-4
   scorer charges the COMBINATORIAL code ``log2(G+1) + mean_i log2 C(G, |S_i|)``:
   each row sends its cardinality, then which subset of that size, as if every
   subset were equally likely. It learns nothing from the data, so it stays
   invariant to the estimation rows, but a dictionary whose firings are
   predictable is overpaid by it. That matters more here than anywhere else: with
   the dictionary term equalised by the faithful ``k_flat`` config and the theorem
   predicting no code or residual movement for a circle-class chart, SUPPORT is
   the entire predicted margin. The audit therefore reports, next to the charged
   ``support_bits``, the scorer's uncharged ``independent_support_bits``: the
   independent Krichevsky-Trofimov code over the estimation rows, which pays for
   learning each firing rate and credits uneven rates. Both are read from the
   scorer's own report, so the audit never re-derives either price.
5. **What does a curved atom's ``code_dim = d+1`` truncation leave unpriced?** A
   curved atom's contribution spans ``m`` decoder rows (``m = 2H+1`` for an
   H-harmonic circle) while transmitting ``d+1`` scalars. Charging ``d+1`` is the
   theorem's own ledger -- ``predicted_birth_dl_bits`` prices a chart at ``d+1``
   scalar rates against the flat span ``s`` -- but the linear span the
   contribution occupies is wider, and the mass beyond ``d+1`` is priced by
   nobody. The audit measures that mass and RE-SCORES the identical featurizer
   with every atom charged its full span. A verdict that survives both pricings
   does not rest on the surrogate.

Everything here is result analysis on an already-fitted model (SPEC.md line 8):
it adds no math to the production path and changes no scored number.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from bits_eq4 import description_length

__all__ = ["audit_row", "flat_span_widths"]


def flat_span_widths(n_atoms: int) -> np.ndarray:
    """Per-atom linear span of a flat block: one decoder row each."""
    return np.ones(int(n_atoms), dtype=np.int64)


def _digamma(values: np.ndarray) -> np.ndarray:
    """psi(x) for x > 0 by upward recurrence onto the standard asymptotic series."""
    x = np.array(values, dtype=np.float64, copy=True)
    out = np.zeros_like(x)
    shift_floor = 8.0  # asymptotic series is at machine precision beyond this
    while np.any(x < shift_floor):
        low = x < shift_floor
        out[low] -= 1.0 / x[low]
        x[low] += 1.0
    inverse_square = 1.0 / (x * x)
    series = inverse_square * (
        -1.0 / 12.0
        + inverse_square
        * (1.0 / 120.0 + inverse_square * (-1.0 / 252.0 + inverse_square / 240.0))
    )
    return out + np.log(x) - 0.5 / x + series


class _SpectrumRecorder:
    """Wraps ``atom_contribution`` so the scorer's own fetches yield the spectra.

    The scorer fetches an atom only when that atom clears its skip rule, and
    passes exactly the row subset it will score (already strided down if its row
    cap engaged). Recording at that boundary therefore observes the scorer's
    decisions instead of restating them.
    """

    __slots__ = ("_inner", "_code_dims", "rows_taken", "total_var", "priced_var")

    def __init__(self, inner, code_dims: np.ndarray) -> None:
        self._inner = inner
        self._code_dims = code_dims
        self.rows_taken: dict[int, int] = {}
        self.total_var: dict[int, float] = {}
        self.priced_var: dict[int, float] = {}

    def __call__(self, atom: int):
        return _RecordingRows(self._inner(atom), int(atom), self)

    def observe(self, atom: int, values: np.ndarray) -> None:
        rows = values.shape[0]
        centered = values - values.mean(axis=0, keepdims=True)
        denominator = float(max(rows - 1, 1))
        # Both Grams carry the same nonzero spectrum; decomposing the smaller
        # side costs min(rows, columns)^3 instead of the ambient columns^3, and a
        # sparse dictionary's atoms fire on far fewer rows than there are output
        # channels.
        gram = (
            centered @ centered.T
            if rows <= values.shape[1]
            else centered.T @ centered
        )
        eigenvalues = np.linalg.eigvalsh(gram)[::-1]
        spectrum = np.maximum(eigenvalues, 0.0) / denominator
        keep = min(int(self._code_dims[atom]), spectrum.size)
        self.rows_taken[atom] = int(rows)
        self.total_var[atom] = float(spectrum.sum())
        self.priced_var[atom] = float(spectrum[:keep].sum())


class _RecordingRows:
    """Row proxy that reports every materialised contribution to its recorder."""

    __slots__ = ("_proxy", "_atom", "_recorder")

    def __init__(self, proxy, atom: int, recorder: _SpectrumRecorder) -> None:
        self._proxy = proxy
        self._atom = atom
        self._recorder = recorder

    def __getitem__(self, index: Any) -> np.ndarray:
        values = np.asarray(self._proxy[index], dtype=np.float64)
        self._recorder.observe(self._atom, values)
        return values


def _firing_rows_by_atom(gate: np.ndarray, n_atoms: int) -> list[np.ndarray]:
    """Ascending live-gate row indices per atom, from one pass over the gate.

    A per-atom column scan of a C-ordered (N, G) gate strides the whole array
    once per atom; grouping the single sparse nonzero list instead keeps the
    sweep proportional to the firings (N*L0), not to N*G.
    """
    rows, atoms = np.nonzero(gate > 0.0)
    order = np.argsort(atoms, kind="stable")
    rows = rows[order]
    boundaries = np.searchsorted(atoms[order], np.arange(n_atoms + 1))
    return [rows[boundaries[g]:boundaries[g + 1]] for g in range(n_atoms)]


def _atom_sum_reconstruction(fitted, shape: tuple[int, int], firing_rows) -> np.ndarray:
    """Sum every atom's solo contribution over the rows where its gate is live."""
    total = np.zeros(shape, dtype=np.float64)
    for atom, rows in enumerate(firing_rows):
        if rows.size == 0:
            continue
        total[rows] += np.asarray(fitted.atom_contribution(atom)[rows], dtype=np.float64)
    return total


def audit_row(
    fitted,
    x_bits: np.ndarray,
    *,
    amortization_horizon: int,
    span_widths: np.ndarray,
    r2_target: float,
) -> dict[str, Any]:
    """Measure every Eq-4 approximation this row rests on, and bracket the score.

    ``span_widths[g]`` is the number of decoder rows atom ``g``'s contribution can
    occupy (1 for a flat atom, the curved block's evaluated basis width for a
    chart). The pessimistic re-score charges ``max(code_dims, span_widths)`` --
    every atom paid at its full linear span, the price a flat dictionary would
    charge to carry the same image -- so the true Eq-4 total is bracketed by the
    theorem's ledger below and that linear reading above.
    """
    x_bits = np.ascontiguousarray(np.asarray(x_bits, dtype=np.float64))
    gate = np.asarray(fitted.gate)
    code_dims = np.asarray(fitted.code_dims, dtype=np.int64)
    span_widths = np.asarray(span_widths, dtype=np.int64)
    if span_widths.shape != code_dims.shape:
        raise ValueError(
            f"span_widths {span_widths.shape} must have one entry per atom "
            f"{code_dims.shape}"
        )
    n_rows, n_atoms = gate.shape

    # (1) Reconciliation: the atoms must rebuild the reconstruction the residual
    # term is read from, up to one constant row (a decoder pre-bias costs no
    # per-token code and is charged in neither arm's dictionary term).
    firing_rows = _firing_rows_by_atom(gate, n_atoms)
    recon = np.asarray(fitted.recon, dtype=np.float64)
    atom_sum = _atom_sum_reconstruction(fitted, recon.shape, firing_rows)
    gap = recon - atom_sum
    constant_row = gap.mean(axis=0)
    recon_norm = float(np.linalg.norm(recon))
    reconciliation = {
        "unexplained_relative_frobenius": float(
            np.linalg.norm(gap - constant_row[None, :]) / max(recon_norm, 1e-300)
        ),
        "constant_row_relative_frobenius": float(
            np.linalg.norm(constant_row) * np.sqrt(n_rows) / max(recon_norm, 1e-300)
        ),
    }

    # (2)-(4) One scoring pass with the spectra recorded at the scorer's own
    # fetch boundary, then one pessimistic pass at the full linear span.
    recorder = _SpectrumRecorder(fitted.atom_contribution, code_dims)
    inner_contribution = fitted.atom_contribution
    fitted.atom_contribution = recorder
    try:
        ledger = description_length(
            fitted, x_bits, amortization_horizon=amortization_horizon
        )
    finally:
        fitted.atom_contribution = inner_contribution

    fetched = np.zeros(n_atoms, dtype=bool)
    fetched[list(recorder.rows_taken)] = True
    firing_counts = np.asarray([rows.size for rows in firing_rows], dtype=np.int64)
    total_firings = int(firing_counts.sum())
    skipped_firings = int(firing_counts[~fetched].sum())
    rows_taken = np.zeros(n_atoms, dtype=np.int64)
    for atom, rows in recorder.rows_taken.items():
        rows_taken[atom] = rows
    subsampled = fetched & (rows_taken < firing_counts)

    total_var = np.zeros(n_atoms, dtype=np.float64)
    priced_var = np.zeros(n_atoms, dtype=np.float64)
    for atom, value in recorder.total_var.items():
        total_var[atom] = value
    for atom, value in recorder.priced_var.items():
        priced_var[atom] = value
    unpriced_var = np.maximum(total_var - priced_var, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        unpriced_fraction = np.where(total_var > 0.0, unpriced_var / total_var, 0.0)

    firing_probability = firing_counts.astype(np.float64) / float(n_rows)
    rank_one = span_widths <= 1
    curved = span_widths > code_dims
    centered_x = x_bits - x_bits.mean(axis=0, keepdims=True)
    reference_variance = float((centered_x * centered_x).sum() / n_rows)
    distortion_budget = (1.0 - float(r2_target)) * reference_variance

    # Small-sample rate bias: each charged mode is 0.5*log2(sigma^2/theta) on a
    # variance estimated from `rows_taken - 1` degrees of freedom, and E[log s^2]
    # sits below log sigma^2 by log(nu/2) - psi(nu/2). The row is therefore
    # UNDER-charged by this many bits per token, both arms alike.
    degrees = np.maximum(rows_taken.astype(np.float64) - 1.0, 1.0)
    per_mode_bias = (np.log(degrees / 2.0) - _digamma(degrees / 2.0)) / np.log(2.0)
    charged_modes = np.minimum(code_dims, np.maximum(rows_taken - 1, 0))
    log_variance_bias_bits = float(
        (firing_probability * fetched * charged_modes * 0.5 * per_mode_bias).sum()
    )

    pessimistic_dims = np.maximum(code_dims, span_widths)
    pessimistic_fitted = type(fitted)(
        name=f"{fitted.name}_full_span",
        gate=fitted.gate,
        atom_contribution=inner_contribution,
        code_dims=pessimistic_dims,
        dictionary_params=fitted.dictionary_params,
        recon=fitted.recon,
        fit_seconds=fitted.fit_seconds,
    )
    pessimistic = description_length(
        pessimistic_fitted, x_bits, amortization_horizon=amortization_horizon
    )

    target_key = f"bits_at_r2_{r2_target:g}"
    return {
        "r2_target": float(r2_target),
        "reconciliation": reconciliation,
        "skip_rule": {
            "atoms": int(n_atoms),
            "atoms_scored": int(fetched.sum()),
            "atoms_skipped": int((~fetched).sum()),
            "firings": total_firings,
            "firings_skipped": skipped_firings,
            "firing_fraction_skipped": float(skipped_firings / max(total_firings, 1)),
        },
        "row_cap": {
            "atoms_subsampled": int(subsampled.sum()),
            "max_firings_on_one_atom": int(firing_counts.max(initial=0)),
        },
        "rank_one_fast_path": {
            "atoms": int((rank_one & fetched).sum()),
            "max_unpriced_variance_fraction": float(
                unpriced_fraction[rank_one & fetched].max(initial=0.0)
            ),
            "firing_weighted_unpriced_variance": float(
                (firing_probability * unpriced_var * (rank_one & fetched)).sum()
            ),
        },
        "curved_truncation": {
            "atoms": int((curved & fetched).sum()),
            "max_unpriced_variance_fraction": float(
                unpriced_fraction[curved & fetched].max(initial=0.0)
            ),
            "mean_unpriced_variance_fraction": float(
                unpriced_fraction[curved & fetched].mean()
                if (curved & fetched).any()
                else 0.0
            ),
            "firing_weighted_unpriced_variance": float(
                (firing_probability * unpriced_var * (curved & fetched)).sum()
            ),
            "distortion_budget_at_target": distortion_budget,
            "unpriced_variance_over_budget": float(
                (firing_probability * unpriced_var * (curved & fetched)).sum()
                / max(distortion_budget, 1e-300)
            ),
        },
        "support_currency": {
            "atoms": int(n_atoms),
            "atoms_that_ever_fire": int((firing_counts > 0).sum()),
            "achieved_l0": float(firing_probability.sum()),
            "combinatorial_bits": float(ledger["support_bits"]),
            "independent_support_bits": float(ledger["independent_support_bits"]),
        },
        "log_variance_bias_bits": log_variance_bias_bits,
        "bracket": {
            "theorem_ledger_bits": float(ledger[target_key]),
            "full_span_bits": float(pessimistic[target_key]),
            "full_span_code_bits": float(pessimistic[f"code_bits_at_r2_{r2_target:g}"]),
            "theorem_ledger_code_bits": float(ledger[f"code_bits_at_r2_{r2_target:g}"]),
        },
    }
