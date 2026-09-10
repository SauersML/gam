"""Recipes for cross-fitted score prediction; no influence-absorber claim."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CtnStage1:
    """Conditional score transform fitted inside the supplied training sample.

    Supply ``fold_column`` with explicit fold labels, or ``group_column`` to
    assign whole groups reproducibly. If both are supplied, family separation
    is checked. ``folds`` and ``seed`` govern generated group folds only.
    Each fold learns its own response knots and covariate geometry. The saved
    predictor uses a separate CTN fitted on the complete training sample.

    Response-basis options retain the engine's affine-null shape penalties.
    Cross-fitting does not imply outcome calibration or Neyman orthogonality.
    """

    response: str
    covariates: str
    fold_column: str | None = None
    group_column: str | None = None
    folds: int = 5
    seed: int = 0
    response_degree: int | None = None
    response_num_internal_knots: int | None = None
    response_penalty_order: int | None = None
    response_extra_penalty_orders: tuple[int, ...] | None = None
    double_penalty: bool | None = None
    weights: str | None = None
    offset: str | None = None

    def __post_init__(self):
        for name in ("response", "covariates"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"CtnStage1.{name} must be a nonempty string")
        if "~" in self.covariates:
            raise ValueError("CtnStage1.covariates must be a formula right-hand side")
        if self.fold_column is None and self.group_column is None:
            raise ValueError("CtnStage1 requires fold_column or group_column")
        for name in ("fold_column", "group_column", "weights", "offset"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"CtnStage1.{name} must name a column")
        if type(self.folds) is not int or self.folds < 2 or type(self.seed) is not int:
            raise ValueError("CtnStage1 requires folds >= 2 and an integer seed")
        for name in ("response_degree", "response_num_internal_knots", "response_penalty_order"):
            value = getattr(self, name)
            minimum = 2 if name == "response_num_internal_knots" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"CtnStage1.{name} must be an integer >= {minimum}")
        if self.response_extra_penalty_orders is not None:
            if any(type(x) is not int or x < 1 for x in self.response_extra_penalty_orders):
                raise ValueError("response_extra_penalty_orders must be positive integers")
        if self.double_penalty is not None and type(self.double_penalty) is not bool:
            raise ValueError("double_penalty must be boolean")

    def response_config(self):
        return {key: value for key, value in asdict(self).items()
                if (key.startswith("response_") or key == "double_penalty") and value is not None}


def normalize_ctn_stage1(value: Any) -> CtnStage1 | None:
    if value is None or isinstance(value, CtnStage1):
        return value
    if isinstance(value, Mapping):
        return CtnStage1(**value)
    raise TypeError("transformation_normal_stage1 must be CtnStage1 or a mapping")
