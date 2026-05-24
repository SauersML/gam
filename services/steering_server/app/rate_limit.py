from __future__ import annotations

from fastapi import Request
from slowapi import Limiter

from app.database import api_key_hash


def rate_limit_key(request: Request) -> str:
    token = request.headers.get("x-api-key")
    if token is None:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:]
    if token:
        return f"user:{api_key_hash(token)}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


limiter = Limiter(key_func=rate_limit_key)
