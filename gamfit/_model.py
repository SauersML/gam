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
    numeric_matrix,
    shape_prediction_response,
    term_blocks_from_state,
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
        payload: dict[str, Any] = {"interval": interval}
        if with_uncertainty:
            payload["with_uncertainty"] = True
        grid = default_survival_time_grid(self.model_class, self.formula, headers, rows) \
            if self.is_survival else None
        if grid is not None:
            payload["time_grid"] = [float(t) for t in grid]
        try:
            raw = rust_module().predict_table(
                self._model_bytes, headers, rows, json.dumps(payload)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return shape_prediction_response(
            json.loads(raw),
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
        import numpy as np

        X_arr = numeric_matrix(X, "X")
        try:
            return np.asarray(
                rust_module().predict_array(
                    self._model_bytes, X_arr, json.dumps({"interval": interval})
                ),
                dtype=float,
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def summary(self) -> Summary:
        """Return the model summary (coefficients, family, deviance, REML score)."""
        if self._summary_cache is None:
            try:
                payload = json.loads(rust_module().summary_json(self._model_bytes))
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
            payload = json.loads(rust_module().check_json(self._model_bytes, headers, rows))
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
        payload: dict[str, Any] = {}
        for key, value in (
            ("samples", samples), ("warmup", warmup), ("chains", chains), ("seed", seed),
        ):
            if value is not None:
                payload[key] = int(value)
        if target_accept is not None:
            payload["target_accept"] = float(target_accept)
        try:
            raw = rust_module().sample_table(
                self._model_bytes, headers, rows, json.dumps(payload)
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
        import numpy as np

        X_arr = numeric_matrix(X, "X")
        try:
            return np.asarray(
                rust_module().design_matrix_array(self._model_bytes, X_arr),
                dtype=float,
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
        request: dict[str, Any] = {
            "view": str(view),
            "group": str(group) if group is not None else None,
            "pairs": [[str(a), str(b)] for a, b in pairs] if pairs is not None else None,
            "n": int(n),
            "level": float(level),
            "simultaneous": bool(simultaneous),
            "n_sim": int(n_sim),
            "seed": int(seed) if seed is not None else None,
            "marginalise_random": bool(marginalise_random),
            "group_means": bool(group_means),
            "template": template or None,
        }
        try:
            raw = rust_module().difference_smooth_json(self._model_bytes, json.dumps(request))
        except Exception as exc:
            raise map_exception(exc) from exc
        rows_out = json.loads(raw)
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
        payload = dict(new_group_spec)
        if metadata is not None:
            payload["metadata"] = metadata
        if prior is not None:
            payload["prior"] = prior
        try:
            model_bytes = bytes(
                rust_module().extend_model_with_group(self._model_bytes, json.dumps(payload))
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

    def _saved_model_payload(self) -> dict[str, Any]:
        saved = json.loads(self._model_bytes)
        if not isinstance(saved, dict):
            raise ValueError("saved model payload must be a JSON object")
        payload = saved.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("saved model payload is missing its payload object")
        return payload

    def _saved_payload_string(self, key: str) -> str | None:
        value = self._saved_model_payload().get(key)
        return str(value) if value is not None else None

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
            state = json.loads(rust_module().coefficient_state_json(self._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc
        return term_blocks_from_state(state)

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
        return math.exp(self.evidence - other.evidence)

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
