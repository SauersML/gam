"""High-level polygenic score calibration helpers.

This module provides :class:`PgsCalibration`, a one-object wrapper around the
Stage-1 preprocessing pattern used in genotype-score calibration: fitting a
conditional transformation-normal model of a raw polygenic score on a basis
expansion over joint principal-component space, then transforming new samples
to population-calibrated z-scores.

The helper encodes the following default choices, each of which can be
overridden at construction time:

* A Duchon spline basis over the PC columns with ``len(pc_columns) + 20``
  centers, order ``1``, power ``1``.
* Per-axis anisotropic scaling (``scale_dimensions="auto"``).
* A transformation-normal likelihood so the fitted response is interpretable
  as a standard normal z-score for each row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ._api import fit as fit_model
from ._api import load as load_model
from ._model import Model


__all__ = ["PgsCalibration"]


@dataclass
class PgsCalibration:
    """Fit-and-transform helper for polygenic-score calibration on PC space.

    Parameters
    ----------
    pc_columns:
        Ordered list of principal-component column names (e.g.
        ``["pc1", "pc2", "pc3", "pc4"]``).
    pgs_column:
        Name of the raw polygenic-score column to calibrate.
    duchon_centers:
        Number of Duchon basis centers. Defaults to ``len(pc_columns) + 20``.
    duchon_order:
        Duchon spline order (``m``). Defaults to ``1``.
    duchon_power:
        Duchon spline power (``s``). Defaults to ``1``.
    scale_dimensions:
        Per-axis length-scale selector forwarded to :func:`gam.fit`. Defaults
        to ``"auto"``.
    out_column:
        Name of the calibrated column appended by :meth:`transform`. Defaults
        to ``"PGS_cal"``.
    extra_fit_kwargs:
        Additional kwargs forwarded verbatim to :func:`gam.fit` (e.g.
        ``{"firth": True}``).

    Examples
    --------
    >>> calib = PgsCalibration(
    ...     pc_columns=["pc1", "pc2", "pc3", "pc4"],
    ...     pgs_column="PGS",
    ... )
    >>> calib.fit(df_train)
    >>> df_train = calib.transform(df_train)
    >>> df_test = calib.transform(df_test)
    """

    pc_columns: Sequence[str]
    pgs_column: str = "PGS"
    duchon_centers: int | None = None
    duchon_order: int = 1
    duchon_power: int = 1
    scale_dimensions: str | None = "auto"
    out_column: str = "PGS_cal"
    extra_fit_kwargs: dict[str, Any] = field(default_factory=dict)

    _model: Model | None = field(default=None, init=False, repr=False)
    _resolved_centers: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.pc_columns:
            raise ValueError("pc_columns must be a non-empty sequence")
        if not self.pgs_column:
            raise ValueError("pgs_column must be provided")
        self._resolved_centers = (
            self.duchon_centers
            if self.duchon_centers is not None
            else len(self.pc_columns) + 20
        )

    @property
    def formula(self) -> str:
        """The Wilkinson-style formula used for the Stage-1 fit."""
        pc_args = ", ".join(self.pc_columns)
        duchon = (
            f"duchon({pc_args}, k={self._resolved_centers}, "
            f"m={self.duchon_order}, s={self.duchon_power})"
        )
        return f"{self.pgs_column} ~ {duchon}"

    @property
    def model(self) -> Model:
        """The underlying fitted :class:`gam.Model`. Raises if not yet fit."""
        if self._model is None:
            raise RuntimeError(
                "PgsCalibration has not been fit yet; call .fit(data) first"
            )
        return self._model

    def fit(self, data: Any) -> "PgsCalibration":
        """Fit the Stage-1 transformation-normal calibration model."""
        fit_kwargs: dict[str, Any] = {
            "transformation_normal": True,
            "scale_dimensions": self.scale_dimensions,
        }
        fit_kwargs.update(self.extra_fit_kwargs)
        self._model = fit_model(data, self.formula, **fit_kwargs)
        return self

    def transform(self, data: Any) -> Any:
        """Append a calibrated z-score column to ``data``.

        When ``data`` is a pandas DataFrame the returned object is a new
        DataFrame with ``self.out_column`` appended. For other input kinds
        (pyarrow table, dict of columns, list of records) the return type
        mirrors the input.
        """
        z = self.predict(data)
        return _attach_z_column(data, z, self.out_column)

    def fit_transform(self, data: Any) -> Any:
        """Convenience: :meth:`fit` then :meth:`transform` on the same data."""
        self.fit(data)
        return self.transform(data)

    def predict(self, data: Any) -> Any:
        """Return the raw calibrated z-score array without attaching it."""
        return self.model.predict(data)

    def save(self, path: str | Path) -> None:
        """Persist the underlying fitted model to ``path``."""
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        pc_columns: Sequence[str],
        pgs_column: str = "PGS",
        **kwargs: Any,
    ) -> "PgsCalibration":
        """Load a previously-saved calibration model.

        The PC column ordering and raw-PGS column name are not recorded in the
        model bytes, so callers must pass ``pc_columns`` and ``pgs_column``
        explicitly.
        """
        instance = cls(pc_columns=pc_columns, pgs_column=pgs_column, **kwargs)
        instance._model = load_model(path)
        return instance


def _attach_z_column(data: Any, z: Any, out_column: str) -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(data, pd.DataFrame):
        result = data.copy()
        result[out_column] = _to_1d_list(z)
        return result

    if isinstance(data, dict):
        result = {key: list(value) for key, value in data.items()}
        result[out_column] = _to_1d_list(z)
        return result

    if isinstance(data, list):
        values = _to_1d_list(z)
        if len(values) != len(data):
            raise ValueError(
                f"predicted z-score length {len(values)} does not match record "
                f"count {len(data)}"
            )
        result = []
        for record, value in zip(data, values):
            enriched = dict(record)
            enriched[out_column] = value
            result.append(enriched)
        return result

    try:
        import pyarrow as pa  # type: ignore[import-not-found]
    except ImportError:
        pa = None  # type: ignore[assignment]

    if pa is not None and isinstance(data, pa.Table):
        return data.append_column(out_column, pa.array(_to_1d_list(z)))

    return {
        "_original": data,
        out_column: _to_1d_list(z),
    }


def _to_1d_list(values: Any) -> list[float]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(values, np.ndarray):
        return [float(v) for v in values.reshape(-1).tolist()]
    if isinstance(values, dict):
        for key in ("z", "z_score", "transformed", "mean"):
            if key in values:
                return [float(v) for v in values[key]]
        raise KeyError(
            "prediction result dict does not contain a z-score column"
        )
    return [float(v) for v in values]
