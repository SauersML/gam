from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

from app.database import api_key_hash, get_session
from app.db_models import User


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token", auto_error=False)


def _extract_token(api_key: str | None, bearer_token: str | None) -> str:
    token = api_key or bearer_token
    if token:
        return token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing API key or bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_user(
    api_key: str | None = Security(api_key_header),
    bearer_token: str | None = Security(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    token = _extract_token(api_key, bearer_token)
    user = session.scalar(select(User).where(User.api_key_hash == api_key_hash(token)))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key or bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def authenticate_token(session: Session, token: str | None) -> User:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing token")
    user = session.scalar(select(User).where(User.api_key_hash == api_key_hash(token)))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    return user
