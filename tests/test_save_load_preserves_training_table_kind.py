"""Regression test for #394: save/load must preserve `training_table_kind`.

A fitted ``gamfit.Model`` records the container type of its training table
(e.g. ``"pandas"``). ``Model.predict`` uses that as the output-container
fallback when the *prediction input* is itself container-ambiguous — a ``dict``
of columns or a ``list`` of record dicts. Before the fix, the training-table
kind lived only on the in-memory object and was dropped on every
``save``/``load`` (and ``dumps``/``loads``), so a reloaded model silently
returned a ``dict`` instead of the ``pandas.DataFrame`` the original returned
for identical input. The numbers were bit-identical; only the container type
changed. The fix persists the kind in the serialized model payload so the
fallback round-trips.
"""

from __future__ import annotations

import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast


class _Pytest(Protocol):
    def importorskip(
        self,
        modname: str,
        minversion: str | None = None,
        reason: str | None = None,
        *,
        exc_type: type[ImportError] | None = None,
    ) -> Any: ...


pytest = cast(_Pytest, import_module("pytest"))

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _fit_pandas_model() -> Any:
    rng = np.random.default_rng(6)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "y": np.sin(2.0 * np.pi * x) + 0.1 * rng.standard_normal(n),
            "x": x,
        }
    )
    return gamfit.fit(df, "y ~ s(x)", family="gaussian"), x


def test_save_load_preserves_training_table_kind_and_predict_container() -> None:
    model, x = _fit_pandas_model()
    assert model.training_table_kind == "pandas"

    # Container-ambiguous prediction inputs: a dict of columns and a list of
    # record dicts. The output container is driven by the training-table kind
    # fallback, which must survive a save/load round-trip.
    dict_input = {"x": [float(v) for v in x[:5]]}
    list_input = [{"x": float(v)} for v in x[:5]]

    inmem_dict = model.predict(dict_input, interval=0.95)
    inmem_list = model.predict(list_input, interval=0.95)
    assert isinstance(inmem_dict, pd.DataFrame)
    assert isinstance(inmem_list, pd.DataFrame)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir) / "model.gam")
        gamfit.save(model, tmp_path)
        reloaded = gamfit.load(tmp_path)

    # The reloaded model must carry the same training-table kind ...
    assert reloaded.training_table_kind == "pandas"

    # ... and therefore predict into the same container type as the original.
    disk_dict = reloaded.predict(dict_input, interval=0.95)
    disk_list = reloaded.predict(list_input, interval=0.95)
    assert isinstance(disk_dict, pd.DataFrame), type(disk_dict).__name__
    assert isinstance(disk_list, pd.DataFrame), type(disk_list).__name__

    # The numeric content must be identical too — only the container type was
    # ever at risk, but a regression here would also be a bug.
    pd.testing.assert_frame_equal(
        inmem_dict.reset_index(drop=True), disk_dict.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        inmem_list.reset_index(drop=True), disk_list.reset_index(drop=True)
    )


def test_dumps_loads_preserves_training_table_kind() -> None:
    """The in-memory bytes round-trip (`dumps`/`loads`) preserves the kind too."""
    model, _ = _fit_pandas_model()
    assert model.training_table_kind == "pandas"

    reloaded = gamfit.loads(model.dumps())
    assert reloaded.training_table_kind == "pandas"
