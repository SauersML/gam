from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("usr"))
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    steerers: Mapped[list[Steerer]] = relationship(back_populates="owner")


class Steerer(Base):
    __tablename__ = "steerers"
    __table_args__ = (
        UniqueConstraint("owner_id", "model", "layer", "concept_targets_hash", name="uq_steerer_owner_shape"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("str"))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    concept_targets: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    concept_targets_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    gauge: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    diagnostics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    owner: Mapped[User] = relationship(back_populates="steerers")
    requests: Mapped[list[RequestHistory]] = relationship(back_populates="steerer")


class RequestHistory(Base):
    __tablename__ = "request_history"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("req"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    steerer_id: Mapped[str | None] = mapped_column(ForeignKey("steerers.id"), nullable=True, index=True)
    route: Mapped[str] = mapped_column(String(256), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    steerer: Mapped[Steerer | None] = relationship(back_populates="requests")
