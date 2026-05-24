from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from sqlalchemy import select

from app.auth import authenticate_token
from app.database import SessionLocal
from app.db_models import Steerer
from app.models import SteerRequest
from app.observability.metrics import STEERER_CALLS
from app.services.steering import backend


router = APIRouter(tags=["streaming"])


@router.websocket("/v1/stream")
async def stream(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token") or websocket.headers.get("x-api-key")
    await websocket.accept()
    try:
        with SessionLocal() as session:
            try:
                user = authenticate_token(session, token)
            except HTTPException as exc:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc.detail))
                return
            payload = await websocket.receive_json()
            steerer_id = payload.get("steerer_id")
            if not isinstance(steerer_id, str):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="steerer_id is required")
                return
            try:
                request = SteerRequest.model_validate(payload)
            except ValidationError as exc:
                await websocket.send_json({"type": "error", "detail": exc.errors()})
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            steerer = session.scalar(
                select(Steerer).where(Steerer.id == steerer_id, Steerer.owner_id == user.id)
            )
            if steerer is None:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="steerer not found")
                return
            if steerer.status != "ready" or steerer.gauge is None:
                await websocket.close(
                    code=status.WS_1013_TRY_AGAIN_LATER,
                    reason=f"steerer is {steerer.status}",
                )
                return
            completions = backend.complete(
                model=steerer.model,
                layer=steerer.layer,
                gauge=steerer.gauge,
                request=request,
            )
            STEERER_CALLS.labels(steerer.id).inc()
            for completion in completions:
                await websocket.send_json({"type": "completion_start", "prompt": completion.prompt})
                for token_text in completion.tokens:
                    await websocket.send_json({"type": "token", "token": token_text})
                await websocket.send_json(
                    {
                        "type": "completion_end",
                        "finish_reason": completion.finish_reason,
                        "steering_trace": completion.steering_trace,
                    }
                )
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
    except WebSocketDisconnect:
        return
