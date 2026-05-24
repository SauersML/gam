from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.config import settings
from app.database import get_session
from app.db_models import RequestHistory, Steerer, User
from app.models import (
    DiagnosticsResponse,
    SteerRequest,
    SteerResponse,
    SteererCreateRequest,
    SteererListResponse,
    SteererResponse,
)
from app.observability.metrics import STEERER_CALLS
from app.rate_limit import limiter
from app.services.steering import backend, concept_targets_hash
from app.tasks.fit_queue import fit_queue


router = APIRouter(prefix="/v1/steerers", tags=["steerers"])

DbSession = Annotated[Session, Depends(get_session)]
AuthedUser = Annotated[User, Depends(current_user)]


def _record_history(
    session: Session,
    *,
    user_id: str,
    steerer_id: str | None,
    route: str,
    status_code: int,
    started: float,
    payload: dict[str, object] | None = None,
) -> None:
    session.add(
        RequestHistory(
            user_id=user_id,
            steerer_id=steerer_id,
            route=route,
            status_code=status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
            payload=payload,
        )
    )


@router.post("", response_model=SteererResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(settings.rate_limit.default_limit)
def create_steerer(
    request: Request,
    body: SteererCreateRequest,
    user: AuthedUser,
    session: DbSession,
) -> SteererResponse:
    started = time.perf_counter()
    target_hash = concept_targets_hash(body.concept_targets)
    existing = session.scalar(
        select(Steerer).where(
            Steerer.owner_id == user.id,
            Steerer.model == body.model,
            Steerer.layer == body.layer,
            Steerer.concept_targets_hash == target_hash,
        )
    )
    if existing is not None:
        _record_history(
            session,
            user_id=user.id,
            steerer_id=existing.id,
            route=request.url.path,
            status_code=status.HTTP_200_OK,
            started=started,
            payload={"deduplicated": True},
        )
        session.commit()
        return SteererResponse.model_validate(existing)

    steerer = Steerer(
        owner_id=user.id,
        model=body.model,
        layer=body.layer,
        concept_targets=[target.model_dump(mode="json") for target in body.concept_targets],
        concept_targets_hash=target_hash,
        status="pending",
    )
    session.add(steerer)
    session.flush()
    _record_history(
        session,
        user_id=user.id,
        steerer_id=steerer.id,
        route=request.url.path,
        status_code=status.HTTP_201_CREATED,
        started=started,
        payload={"concept_count": len(body.concept_targets)},
    )
    session.commit()
    fit_queue.enqueue(steerer.id)
    refreshed = session.get(Steerer, steerer.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="created steerer could not be reloaded")
    return SteererResponse.model_validate(refreshed)


@router.get("", response_model=SteererListResponse)
@limiter.limit(settings.rate_limit.default_limit)
def list_steerers(
    request: Request,
    user: AuthedUser,
    session: DbSession,
) -> SteererListResponse:
    rows = session.scalars(
        select(Steerer).where(Steerer.owner_id == user.id).order_by(Steerer.created_at.desc())
    ).all()
    return SteererListResponse(steerers=[SteererResponse.model_validate(row) for row in rows])


@router.post("/{steerer_id}/steer", response_model=SteerResponse)
@limiter.limit(settings.rate_limit.steer_limit)
def steer(
    request: Request,
    steerer_id: str,
    body: SteerRequest,
    user: AuthedUser,
    session: DbSession,
) -> SteerResponse:
    started = time.perf_counter()
    steerer = session.scalar(select(Steerer).where(Steerer.id == steerer_id, Steerer.owner_id == user.id))
    if steerer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="steerer not found")
    if steerer.status != "ready" or steerer.gauge is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"steerer is {steerer.status}; wait until status is ready",
        )
    completions = backend.complete(
        model=steerer.model,
        layer=steerer.layer,
        gauge=steerer.gauge,
        request=body,
    )
    STEERER_CALLS.labels(steerer.id).inc()
    response = SteerResponse(
        steerer_id=steerer.id,
        model=steerer.model,
        layer=steerer.layer,
        alpha=body.alpha,
        completions=completions,
        diagnostics=steerer.diagnostics or {},
    )
    _record_history(
        session,
        user_id=user.id,
        steerer_id=steerer.id,
        route=request.url.path,
        status_code=status.HTTP_200_OK,
        started=started,
        payload={"prompt_count": len(body.prompts), "alpha": body.alpha},
    )
    session.commit()
    return response


@router.get("/{steerer_id}/diagnostics", response_model=DiagnosticsResponse)
@limiter.limit(settings.rate_limit.default_limit)
def diagnostics(
    request: Request,
    steerer_id: str,
    user: AuthedUser,
    session: DbSession,
) -> DiagnosticsResponse:
    steerer = session.scalar(select(Steerer).where(Steerer.id == steerer_id, Steerer.owner_id == user.id))
    if steerer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="steerer not found")
    data = steerer.diagnostics or {
        "anchor_curvature": [],
        "variance_vs_concept_locality": [],
        "predicted_instability_ranking": [],
        "notes": ["diagnostics unavailable until gauge fit completes"],
    }
    return DiagnosticsResponse(
        steerer_id=steerer.id,
        status=steerer.status,
        anchor_curvature=data["anchor_curvature"],
        variance_vs_concept_locality=data["variance_vs_concept_locality"],
        predicted_instability_ranking=data["predicted_instability_ranking"],
        notes=data["notes"],
    )
