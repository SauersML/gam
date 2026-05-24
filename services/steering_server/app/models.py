from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SteererStatus(StrEnum):
    pending = "pending"
    fitting = "fitting"
    ready = "ready"
    failed = "failed"


class ConceptTarget(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    target: float = Field(ge=-1.0, le=1.0)
    anchors: list[str] = Field(default_factory=list, max_length=64)
    locality_weight: float = Field(default=1.0, ge=0.0, le=10.0)


class SteererCreateRequest(BaseModel):
    model: str = Field(min_length=1, max_length=256)
    layer: int = Field(ge=0)
    concept_targets: list[ConceptTarget] = Field(min_length=1, max_length=256)


class SteererResponse(BaseModel):
    id: str
    model: str
    layer: int
    concept_targets: list[ConceptTarget]
    status: SteererStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


class SteererListResponse(BaseModel):
    steerers: list[SteererResponse]


class SteerRequest(BaseModel):
    concept_targets: list[ConceptTarget] = Field(min_length=1, max_length=256)
    alpha: float = Field(default=1.0, ge=-10.0, le=10.0)
    prompts: list[str] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=128, ge=1, le=2048)

    @field_validator("prompts")
    @classmethod
    def prompts_must_not_be_blank(cls, prompts: list[str]) -> list[str]:
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must not contain blank strings")
        return prompts


class Completion(BaseModel):
    prompt: str
    text: str
    tokens: list[str]
    finish_reason: str
    steering_trace: dict[str, Any]


class SteerResponse(BaseModel):
    steerer_id: str
    model: str
    layer: int
    alpha: float
    completions: list[Completion]
    diagnostics: dict[str, Any]


class DiagnosticsResponse(BaseModel):
    steerer_id: str
    status: SteererStatus
    anchor_curvature: list[dict[str, float | str]]
    variance_vs_concept_locality: list[dict[str, float | str]]
    predicted_instability_ranking: list[dict[str, float | str]]
    notes: list[str]


class ErrorResponse(BaseModel):
    detail: str
