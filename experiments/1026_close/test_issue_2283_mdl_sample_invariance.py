#!/usr/bin/env python3
"""Driver-level contract for GitHub #2283 / audit §34 "MDL sample invariance".

The Eq-4 dictionary term used to inherit ``--bits-max-rows`` (the number of rows
sampled to ESTIMATE the score) as its amortisation ``N``, so re-estimating the
SAME fitted model on a different subsample silently changed the objective and
made the authoritative bits-at-R2 row meaningless. The fix splits the single
``N`` into two separately-passed quantities:

* ``estimation_rows`` — the ``test_x`` row count (Monte-Carlo estimator size);
* ``amortization_horizon`` — the declared dictionary-code ``N``.

The heavy numerics live in the Rust core (``sae_eq4_description_length``), whose
own unit test proves the term is bitwise invariant. These tests pin the thin
Python contract the #1026 driver relies on: the horizon is a REQUIRED keyword
that is forwarded SEPARATELY from the estimation subsample and is never defaulted
to it. They run with no compiled extension by loading the package facade in
isolation against a stub Rust boundary that records exactly what it was handed.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACADE_PATH = _REPO_ROOT / "gamfit" / "_description_length.py"


class _RecordingRustBoundary:
    """Stand-in for ``gamfit._binding.rust_module()`` that records its call.

    It reproduces ONLY the dictionary term, computed from the declared horizon
    exactly as the Rust core does, so the contract test can assert both that the
    horizon is forwarded separately and that the term is invariant to the number
    of rows in ``test_x``.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def sae_eq4_description_length(
        self,
        test_x,
        recon,
        gate,
        code_dims,
        dictionary_params,
        amortization_horizon,
        fetch,
        *,
        r2_targets=None,
        native_bits_per_token=None,
    ):
        self.calls.append(
            {
                "estimation_rows": int(np.asarray(test_x).shape[0]),
                "dictionary_params": int(dictionary_params),
                "amortization_horizon": int(amortization_horizon),
            }
        )
        horizon = float(amortization_horizon)
        dictionary_bits = 0.5 * dictionary_params / horizon * math.log2(horizon)
        return {
            "support_bits": 0.0,
            "achieved_block_l0": 1.0,
            "dictionary_bits": dictionary_bits,
            "estimation_rows": int(np.asarray(test_x).shape[0]),
            "amortization_horizon": int(amortization_horizon),
            "bits_at_r2_0.99": dictionary_bits,
        }


def _load_facade(rust_boundary: _RecordingRustBoundary):
    """Load ``gamfit/_description_length.py`` in isolation with a stub binding."""
    package = types.ModuleType("gamfit_stub_2283")
    package.__path__ = []  # mark as a package so relative imports resolve
    binding = types.ModuleType("gamfit_stub_2283._binding")
    binding.rust_module = lambda: rust_boundary
    sys.modules["gamfit_stub_2283"] = package
    sys.modules["gamfit_stub_2283._binding"] = binding

    spec = importlib.util.spec_from_file_location(
        "gamfit_stub_2283._description_length", _FACADE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "gamfit_stub_2283"
    sys.modules["gamfit_stub_2283._description_length"] = module
    spec.loader.exec_module(module)
    return module


def _fitted(facade, rows: int):
    """A minimal single-atom FittedFeaturizer on ``rows`` estimation rows."""
    rng = np.random.default_rng(0)
    recon = rng.standard_normal((rows, 3))
    return facade.FittedFeaturizer(
        name="flat",
        gate=np.ones((rows, 1)),
        atom_contribution=lambda atom: recon,
        code_dims=np.array([1], dtype=np.int64),
        dictionary_params=4096,
        recon=recon,
        fit_seconds=0.0,
    )


def test_dictionary_term_is_invariant_to_the_estimation_subsample():
    """§34: fixed horizon ⇒ bitwise-identical dictionary term at 256/1024/8192."""
    rust = _RecordingRustBoundary()
    facade = _load_facade(rust)
    horizon = 120_000

    dict_bits = []
    for rows in (256, 1024, 8192):
        fitted = _fitted(facade, rows)
        test_x = np.asarray(fitted.recon, dtype=np.float64)
        out = facade.description_length(
            fitted, test_x, amortization_horizon=horizon
        )
        dict_bits.append(out["dictionary_bits"])
        assert out["amortization_horizon"] == horizon
        assert out["estimation_rows"] == rows

    # The horizon reached the boundary unchanged and separate from the subsample.
    for call, rows in zip(rust.calls, (256, 1024, 8192)):
        assert call["amortization_horizon"] == horizon
        assert call["estimation_rows"] == rows
    # The dictionary charge is BITWISE identical across a 32x subsample change.
    assert dict_bits[0] == dict_bits[1] == dict_bits[2]
    expected = 0.5 * 4096 / horizon * math.log2(horizon)
    assert dict_bits[0] == expected


def test_amortization_horizon_is_a_required_keyword():
    """The horizon has no default, so it can never be silently taken from N."""
    rust = _RecordingRustBoundary()
    facade = _load_facade(rust)
    fitted = _fitted(facade, 256)
    test_x = np.asarray(fitted.recon, dtype=np.float64)
    with pytest.raises(TypeError):
        facade.description_length(fitted, test_x)  # missing amortization_horizon
    assert rust.calls == []  # never reached the boundary


def test_a_sub_two_horizon_is_rejected_before_scoring():
    """A caller conflating the horizon with a tiny subsample is refused."""
    rust = _RecordingRustBoundary()
    facade = _load_facade(rust)
    fitted = _fitted(facade, 256)
    test_x = np.asarray(fitted.recon, dtype=np.float64)
    for horizon in (1, 0, -8):
        with pytest.raises(ValueError, match="amortization_horizon"):
            facade.description_length(
                fitted, test_x, amortization_horizon=horizon
            )
    assert rust.calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
