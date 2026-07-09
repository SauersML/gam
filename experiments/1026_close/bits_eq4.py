#!/usr/bin/env python3
"""Eq-4 description-length bits scorer for the #1026 close (gam#2233).

gam#2233 proves the #1026 hybrid's THIN EV-at-matched-actives margin is a
scoreboard artefact: EV-at-matched-actives only partially credits the support +
residual savings a curved atom wins, while the MDL scoreboard (bits at fixed
R^2, their Eq. 4) credits all three of support/code/residual. This module hands
the #1026 arms the EXACT Eq-4 scorer used by the zoo arena so the close reports
bits(flat) - bits(hybrid) on the same currency the theorem predicts we win by a
wide margin.

The single source of truth for the scorer is
``bench/bsf_manifold_zoo.py`` (``_water_fill_bits`` + ``description_length`` +
``FittedFeaturizer``, ~lines 241-261 / 659-705). We import it verbatim when the
repo tree is on the path; on a deploy where only the wheel + experiments dir
ship, we fall back to a VERBATIM copy (kept byte-for-byte in sync — provenance
comment on the copy) so the scoring math is identical either way.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _import_from_zoo():
    """Return (FittedFeaturizer, description_length) from the zoo if reachable."""
    here = os.path.dirname(os.path.abspath(__file__))
    # driver lives at experiments/1026_close/; bench/ is two levels up.
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    bench_dir = os.path.join(repo_root, "bench")
    for cand in (bench_dir, repo_root):
        if cand not in sys.path and os.path.isdir(cand):
            sys.path.insert(0, cand)
    try:
        from bsf_manifold_zoo import (  # type: ignore
            FittedFeaturizer,
            description_length,
        )
    except Exception:
        return None
    return FittedFeaturizer, description_length


_imported = _import_from_zoo()

if _imported is not None:
    FittedFeaturizer, description_length = _imported
else:
    # -------------------------------------------------------------------- #
    # VERBATIM fallback copy of bench/bsf_manifold_zoo.py's Eq-4 scorer.
    # Keep byte-for-byte in sync with that file (FittedFeaturizer dataclass
    # at ~L241; _water_fill_bits + description_length at ~L659-705). The zoo
    # module is the authority; this copy exists only for wheel-only deploys
    # where the repo bench/ dir did not ship.
    # -------------------------------------------------------------------- #

    @dataclass
    class FittedFeaturizer:  # noqa: D401 - verbatim copy, see provenance above
        """Uniform scoring surface (verbatim from bench/bsf_manifold_zoo.py)."""

        name: str
        gate: np.ndarray
        atom_contribution: Callable[[int], np.ndarray]
        code_dims: np.ndarray
        dictionary_params: int
        recon: np.ndarray
        fit_seconds: float
        native_bits_per_token: float | None = None
        atom_intrinsic_coords: Callable[[int], np.ndarray] | None = None
        extras: dict[str, Any] | None = None

    def _water_fill_bits(spectrum: np.ndarray, delta: float) -> float:
        lam = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
        return float(np.sum(0.5 * np.log2(1.0 + lam / max(delta, 1e-300))))

    def description_length(
        fitted: "FittedFeaturizer",
        test_x: np.ndarray,
        *,
        r2_targets: tuple[float, ...] = (0.99, 0.95, 0.90, 0.80),
    ) -> dict[str, Any]:
        n, _d = test_x.shape
        gate_active = fitted.gate > 1e-10
        p_g = gate_active.mean(axis=0)
        l0 = float(gate_active.sum(axis=1).mean())
        n_atoms = fitted.gate.shape[1]
        support_bits = float(
            math.lgamma(n_atoms + 1)
            - math.lgamma(max(round(l0), 1) + 1)
            - math.lgamma(max(n_atoms - round(l0), 1) + 1)
        ) / math.log(2.0)
        resid = test_x - fitted.recon
        resid_cov_eigs = np.linalg.eigvalsh(np.cov(resid.T))
        var_ref = float(np.var(test_x - test_x.mean(axis=0, keepdims=True)) * test_x.shape[1])
        code_spectra: list[np.ndarray] = []
        for g in range(n_atoms):
            rows = np.flatnonzero(gate_active[:, g])
            if rows.size < max(int(fitted.code_dims[g]) + 1, 4):
                code_spectra.append(np.zeros(int(fitted.code_dims[g])))
                continue
            take = rows if rows.size <= 4096 else rows[:: max(rows.size // 4096, 1)]
            m_g = fitted.atom_contribution(g)[take]
            sing = np.linalg.svd(m_g - m_g.mean(axis=0, keepdims=True), compute_uv=False)
            top = sing[: int(fitted.code_dims[g])] ** 2 / max(take.size - 1, 1)
            code_spectra.append(top)
        out: dict[str, Any] = {"support_bits": support_bits, "achieved_block_l0": l0}
        for target in r2_targets:
            delta = (1.0 - target) * var_ref / test_x.shape[1]
            code_bits = float(
                sum(p * _water_fill_bits(spec, delta) for p, spec in zip(p_g, code_spectra))
            )
            resid_bits = _water_fill_bits(resid_cov_eigs, delta)
            dict_bits = 0.5 * fitted.dictionary_params / n * math.log2(max(n, 2))
            out[f"bits_at_r2_{target:g}"] = support_bits + code_bits + resid_bits + dict_bits
            out[f"code_bits_at_r2_{target:g}"] = code_bits
            out[f"resid_bits_at_r2_{target:g}"] = resid_bits
        if fitted.native_bits_per_token is not None:
            out["native_bits_per_token"] = fitted.native_bits_per_token
        return out


__all__ = ["FittedFeaturizer", "description_length", "scorer_source"]


def scorer_source() -> str:
    """Provenance string for the results record: 'import' vs 'verbatim_copy'."""
    return "bench/bsf_manifold_zoo.py (import)" if _imported is not None else "verbatim_copy"
