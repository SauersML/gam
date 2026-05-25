from __future__ import annotations

import json
from typing import Any

import numpy as np

from gamfit import _survival
from gamfit._tables import restore_output_table


class _FakeRust:
    @staticmethod
    def ordered_prediction_columns(columns_json: str) -> str:
        return columns_json

    @staticmethod
    def marginal_slope_clip_probabilities(values: list[float]) -> list[float]:
        return values

    @staticmethod
    def vec_to_array1_f64(values: list[float]) -> Any:
        return np.asarray(values, dtype=float)


def test_bernoulli_marginal_slope_saved_kind_returns_1d_probabilities(
    monkeypatch: Any,
) -> None:
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
            }
        }
    )
    rust = _FakeRust()
    monkeypatch.setattr(_survival, "rust_module", lambda: rust)

    out = _survival.shape_prediction_response(
        raw,
        headers=[],
        rows=[],
        table_kind="pandas",
        training_table_kind="pandas",
        fallback_model_class="marginal-slope",
        fallback_family="bernoulli-marginal-slope",
        interval=None,
        return_type=None,
        id_column=None,
        row_ids=None,
        restore=restore_output_table,
    )

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])
