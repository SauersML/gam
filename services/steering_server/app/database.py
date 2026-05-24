from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db_models import Base, User


engine: Engine = create_engine(settings.database.url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def api_key_hash(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()


def create_schema() -> None:
    Base.metadata.create_all(bind=engine)


def bootstrap_user() -> None:
    with SessionLocal() as session:
        existing = session.scalar(select(User).where(User.name == settings.auth.bootstrap_user))
        if existing is not None:
            return
        session.add(
            User(
                name=settings.auth.bootstrap_user,
                api_key_hash=api_key_hash(settings.auth.bootstrap_api_key),
            )
        )
        session.commit()


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
