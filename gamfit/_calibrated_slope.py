"""Recipes for cross-fitted score prediction; no influence-absorber claim."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, overload


@dataclass(frozen=True, slots=True)
class CtnStage1:
    """Conditional score transform fitted inside the supplied training sample.

    Supply ``fold_column`` with explicit fold labels, or ``group_column`` to
    assign whole groups reproducibly. If both are supplied, family separation
    is checked. ``folds`` and ``seed`` govern generated group folds only; left
    unset, the engine's defaults apply. The engine validates the whole recipe.
    Each fold learns its own response knots and covariate geometry. The saved
    predictor uses a separate CTN fitted on the complete training sample.

    Response-basis options retain the engine's affine-null shape penalties.
    Cross-fitting does not imply outcome calibration or Neyman orthogonality.
    """

    response: str
    covariates: str
    fold_column: str | None = None
    group_column: str | None = None
    folds: int | None = None
    seed: int | None = None
    response_degree: int | None = None
    response_num_internal_knots: int | None = None
    response_penalty_order: int | None = None
    response_extra_penalty_orders: tuple[int, ...] | None = None
    double_penalty: bool | None = None
    weights: str | None = None
    offset: str | None = None

    def response_config(self) -> dict[str, int | bool | tuple[int, ...]]:
        return {key: value for key, value in asdict(self).items()
                if (key.startswith("response_") or key == "double_penalty") and value is not None}

    def native_document(self) -> dict[str, object]:
        """Marshal the shared Rust fit-request schema without fitting in Python."""
        return {"response_column": self.response, "covariate_formula_rhs": self.covariates,
                "fold_column": self.fold_column, "group_column": self.group_column,
                "folds": self.folds, "seed": self.seed, "weight_column": self.weights,
                "offset_column": self.offset, "config": self.response_config()}


@overload
def normalize_ctn_stage1(value: None) -> None: ...


@overload
def normalize_ctn_stage1(value: CtnStage1 | Mapping[str, Any]) -> CtnStage1: ...


def normalize_ctn_stage1(value: object) -> CtnStage1 | None:
    if value is None or isinstance(value, CtnStage1):
        return value
    if isinstance(value, Mapping):
        return CtnStage1(**value)
    raise TypeError("transformation_normal_stage1 must be CtnStage1 or a mapping")
