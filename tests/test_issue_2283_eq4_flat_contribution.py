"""Regression coverage for the theorem-faithful #2283 Eq-4 flat block."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_arm_featurizers():
    experiment_dir = Path(__file__).resolve().parents[1] / "experiments" / "1026_close"
    sys.path.insert(0, str(experiment_dir))
    spec = importlib.util.spec_from_file_location(
        "issue_2283_arm_featurizers", experiment_dir / "arm_featurizers.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flat_eq4_gate_is_magnitude_but_contribution_preserves_signed_codes():
    module = _load_arm_featurizers()
    decoder = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
    indices = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.uint32)
    codes = np.array([[-2.0, 0.5], [3.0, 0.0], [-1.0, 4.0]], dtype=np.float32)

    gate, contribution, code_dims, dictionary_params = module._flat_block_from_sparse(
        decoder, indices, codes
    )

    np.testing.assert_array_equal(
        gate,
        np.array([[2.0, 0.5], [3.0, 0.0], [4.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        contribution(0)[np.array([0, 1, 2])],
        np.array([[-4.0, 2.0], [6.0, -3.0], [8.0, -4.0]], dtype=np.float32),
    )
    assert np.var(codes[:2, 0]) != np.var(np.abs(codes[:2, 0]))
    np.testing.assert_array_equal(code_dims, np.ones(2, dtype=int))
    assert dictionary_params == decoder.size
