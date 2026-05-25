"""Public :class:`Model` shell.

The numeric work all lives in the Rust core: this module marshals
arguments through the FFI, hands payloads off to ``_survival`` /
``_diagnose_plot`` helpers, and exposes Pythonic properties.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._sampling import PosteriorSamples
from ._schema import SchemaCheck
from ._summary import Summary
from ._survival import (
    CompetingRisksCIF,
    CompetingRisksPrediction,
    SurvivalPrediction,
    TermBlock,
    _MARGINAL_SLOPE_MODEL_CLASSES,
    _SURVIVAL_MODEL_CLASSES,
    _TRANSFORMATION_NORMAL_MODEL_CLASSES,
    competing_risks_cif,
    default_survival_time_grid,
    extract_row_ids,
    shape_prediction_response,
    term_blocks_for_model,
)
from ._tables import normalize_table, response_column_name, restore_output_table


class Model:
    __slots__ = ("_model_bytes", "_training_table_kind", "_summary_cache")

    def __init__(self, *, _model_bytes: bytes, _training_table_kind: str | None = None) -> None:
        self._model_bytes = _model_bytes
        self._training_table_kind = _training_table_kind
        self._summary_cache: Summary | None = None

    def predict(
        self,
        data: Any,
        *,
        interval: float | None = None,
        return_type: str | None = None,
        id_column: str | None = None,
        with_uncertainty: bool = False,
    ) -> Any:
        """Predict from ``data``."""
        headers, rows, table_kind = normalize_table(data)
        row_ids = extract_row_ids(headers, rows, id_column)
        grid = default_survival_time_grid(self.model_class, self.formula, headers, rows) \
            if self.is_survival else None
        opts_json = rust_module().build_predict_payload_json(interval, with_uncertainty, grid)
        try:
            raw = rust_module().predict_table(
                self._model_bytes, headers, rows, opts_json
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return shape_prediction_response(
            raw,
            headers=headers,
            rows=rows,
            table_kind=table_kind,
            training_table_kind=self._training_table_kind,
            fallback_model_class=self._model_class_from_payload(),
            interval=interval,
            return_type=return_type,
            id_column=id_column,
            row_ids=row_ids,
            restore=restore_output_table,
        )

    def predict_array(self, X: Any, *, interval: float | None = None) -> Any:
        """Predict directly from a numeric NumPy-compatible feature matrix."""
        try:
            rust = rust_module()
            return rust.predict_array(
                self._model_bytes,
                rust.numeric_matrix_f64(X, "X"),
                json.dumps({"interval": interval}),
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def summary(self) -> Summary:
        """Return the model summary (coefficients, family, deviance, REML score)."""
        if self._summary_cache is None:
            try:
                payload = rust_module().summary_payload_from_model(self._model_bytes)
            except Exception as exc:
                raise map_exception(exc) from exc
            self._summary_cache = Summary.from_dict(payload)
        return self._summary_cache

    def smoothing_parameters(self) -> dict[int, float]:
        """Return fitted smoothing/precision parameters by penalty index."""
        lambdas = self.summary().get("lambdas")
        if not isinstance(lambdas, list):
            return {}
        return {idx: float(value) for idx, value in enumerate(lambdas)}

    def check(self, data: Any) -> SchemaCheck:
        """Validate ``data`` against the model's training schema."""
        headers, rows, _ = normalize_table(data)
        try:
            payload = rust_module().check_payload_from_model(self._model_bytes, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        return SchemaCheck.from_dict(payload)

    def report(self, path: str | Path | None = None) -> str:
        """Generate a standalone HTML report of the fitted model."""
        try:
            html = rust_module().report_html(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        if path is None:
            return str(html)
        Path(path).write_text(html, encoding="utf-8")
        return str(path)

    def sample(
        self,
        data: Any,
        *,
        samples: int | None = None,
        warmup: int | None = None,
        chains: int | None = None,
        target_accept: float | None = None,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Draw from the model's posterior with NUTS."""
        headers, rows, _ = normalize_table(data)
        try:
            ffi = rust_module()
            options_json = ffi.build_sample_payload_json(
                samples, warmup, chains, target_accept, seed
            )
            raw = ffi.sample_table(
                self._model_bytes,
                headers,
                rows,
                options_json,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return PosteriorSamples.from_ffi_json(raw, model_bytes=self._model_bytes)

    def design_matrix(self, data: Any) -> Any:
        """Materialised design matrix for ``data`` against the saved model."""
        headers, rows, _ = normalize_table(data)
        return rust_module().design_matrix_table_dense(self._model_bytes, headers, rows)

    def design_matrix_array(self, X: Any) -> Any:
        """Materialised design matrix for a numeric feature matrix."""
        try:
            rust = rust_module()
            return rust.design_matrix_array(
                self._model_bytes,
                rust.numeric_matrix_f64(X, "X"),
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def difference_smooth(
        self,
        *,
        view: str,
        group: str | None = None,
        pairs: Sequence[tuple[Any, Any]] | None = None,
        n: int = 100,
        level: float = 0.95,
        simultaneous: bool = False,
        n_sim: int = 10_000,
        seed: int | None = 12345,
        marginalise_random: bool = True,
        group_means: bool = True,
        data: Any | None = None,
        return_type: str | None = None,
    ) -> Any:
        """Covariance-aware pairwise difference smooths (Rust-backed)."""
        template: dict[str, str] = {}
        if data is not None:
            headers, rows, _ = normalize_table(data)
            if rows:
                first = rows[0]
                template = {h: str(first[i]) for i, h in enumerate(headers)}
        try:
            request_json = rust_module().build_difference_smooth_request_json(
                str(view),
                str(group) if group is not None else None,
                [(str(a), str(b)) for a, b in pairs] if pairs is not None else None,
                int(n),
                float(level),
                bool(simultaneous),
                int(n_sim),
                int(seed) if seed is not None else None,
                bool(marginalise_random),
                bool(group_means),
                template or None,
            )
            rows_out = rust_module().difference_smooth_rows(
                self._model_bytes, request_json
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        if return_type == "list":
            return rows_out
        try:
            import pandas as pd

            return pd.DataFrame(rows_out)
        except Exception:
            return rows_out

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to ``path``."""
        Path(path).write_bytes(self._model_bytes)

    def extend_with_group(
        self,
        new_group_spec: dict[str, Any],
        metadata: Any | None = None,
        prior: Any | None = None,
    ) -> "Model":
        """Return a no-refit model extended with deployment-time group levels."""
        if not isinstance(new_group_spec, dict):
            raise TypeError("new_group_spec must be a dict")
        try:
            rust = rust_module()
            payload_json = rust.build_extend_group_payload_json(
                json.dumps(new_group_spec),
                json.dumps(metadata) if metadata is not None else None,
                json.dumps(prior) if prior is not None else None,
            )
            model_bytes = bytes(
                rust.extend_model_with_group(self._model_bytes, payload_json)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return Model(
            _model_bytes=model_bytes,
            _training_table_kind=self._training_table_kind,
        )

    def dumps(self) -> bytes:
        """Return the serialised model as raw bytes."""
        return self._model_bytes

    def _saved_payload_string(self, key: str) -> str | None:
        try:
            return rust_module().saved_model_payload_string(self._model_bytes, key)
        except Exception as exc:
            raise map_exception(exc) from exc

    @property
    def formula(self) -> str:
        value = self._saved_payload_string("formula")
        if value is None:
            raise ValueError("saved model payload is missing formula")
        return value

    @property
    def family_name(self) -> str:
        return str(self.summary()["family_name"])

    @property
    def model_class(self) -> str:
        return self._model_class_from_payload()

    @property
    def is_survival(self) -> bool:
        return self.model_class in _SURVIVAL_MODEL_CLASSES

    @property
    def is_marginal_slope(self) -> bool:
        return self.model_class in _MARGINAL_SLOPE_MODEL_CLASSES

    @property
    def is_transformation_normal(self) -> bool:
        return self.model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES

    @property
    def response_name(self) -> str | None:
        return response_column_name(self.formula)

    @property
    def training_table_kind(self) -> str | None:
        return self._training_table_kind

    @property
    def group_metadata(self) -> dict[str, Any] | None:
        value = self.summary().get("group_metadata")
        return dict(value) if isinstance(value, dict) else None

    @property
    def deployment_extensions(self) -> tuple[dict[str, Any], ...]:
        value = self.summary().get("deployment_extensions")
        if not isinstance(value, list):
            return ()
        return tuple(dict(item) for item in value if isinstance(item, dict))

    @property
    def term_blocks(self) -> tuple[TermBlock, ...]:
        """Per-term coefficient column ranges in fitted coefficient order."""
        try:
            return term_blocks_for_model(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc

    @property
    def evidence(self) -> float:
        """REML / LAML log marginal-likelihood score for this fit."""
        value = self.summary().get("reml_score")
        if value is None:
            raise ValueError("saved model payload is missing reml_score")
        return float(value)

    def bayes_factor_vs(self, other: "Model") -> float:
        """Bayes factor of this fit against ``other``."""
        if not isinstance(other, Model):
            raise TypeError(
                f"bayes_factor_vs expects a gamfit.Model, got {type(other).__name__}"
            )
        log_diff = rust_module().bayes_factor_log_diff(
            self._model_bytes, other._model_bytes
        )
        return math.exp(log_diff)

    def _model_class_from_payload(self) -> str:
        value = self._saved_payload_string("model_kind")
        if value is None:
            raise ValueError("saved model payload is missing model_kind")
        return value

    def diagnose(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
    ) -> Diagnostics:
        """Score the fitted model on held-out ``data``."""
        from ._diagnose_plot import diagnose as _diagnose

        return _diagnose(self, data, y=y, interval=interval)

    def plot(
        self,
        data: Any,
        *,
        x: str | None = None,
        y: str | None = None,
        interval: float | None = 0.95,
        kind: str = "prediction",
        ax: Any | None = None,
    ) -> Any:
        """Plot the model's behaviour on ``data`` with matplotlib."""
        from ._diagnose_plot import plot as _plot

        return _plot(self, data, x=x, y=y, interval=interval, kind=kind, ax=ax)

    def __repr__(self) -> str:
        parts = [f"formula={self.formula!r}", f"family_name={self.family_name!r}"]
        if self._training_table_kind is not None:
            parts.append(f"training_table_kind={self._training_table_kind!r}")
        return f"Model({', '.join(parts)})"

    def _repr_html_(self) -> str:
        return self.report()


__all__ = [
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "SurvivalPrediction",
    "TermBlock",
    "competing_risks_cif",
]
