from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.database import bootstrap_user, create_schema
from app.observability.metrics import metrics_middleware, metrics_response
from app.observability.tracing import configure_tracing
from app.rate_limit import limiter
from app.routers import auth, steerers, stream
from app.tasks.fit_queue import fit_queue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    create_schema()
    bootstrap_user()
    fit_queue.start()
    yield
    fit_queue.shutdown()


app = FastAPI(
    title="Concept Manifold Steering Server",
    version="0.1.0",
    description="Production-oriented FastAPI service scaffold for gauge-fitted LLM steering.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.middleware("http")(metrics_middleware)
configure_tracing(app)

app.include_router(auth.router)
app.include_router(steerers.router)
app.include_router(stream.router)


@app.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()
