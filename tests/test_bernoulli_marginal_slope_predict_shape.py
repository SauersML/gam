from __future__ import annotations

import importlib
import json
from typing import Any, NoReturn, Protocol, cast


class _PytestModule(Protocol):
    def importorskip(
        self,
        modname: str,
        minversion: str | None = None,
        reason: str | None = None,
        *,
        exc_type: type[ImportError] | tuple[type[ImportError], ...] | None = None,
    ) -> Any: ...

    def fail(self, reason: str = "", pytrace: bool = True) -> NoReturn: ...


pytest = cast(_PytestModule, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

from gamfit._survival import shape_prediction_response
from gamfit._tables import restore_output_table


def test_bernoulli_marginal_slope_saved_kind_returns_1d_probabilities() -> None:
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
            }
        }
    )

    out = shape_prediction_response(
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
