from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


REQUEST_COUNT = Counter(
    "steering_server_requests_total",
    "Total HTTP requests.",
    ("method", "path", "status"),
)
REQUEST_LATENCY = Histogram(
    "steering_server_request_latency_seconds",
    "HTTP request latency.",
    ("method", "path"),
)
STEERER_CALLS = Counter(
    "steering_server_steerer_calls_total",
    "Steer calls per registered steerer.",
    ("steerer_id",),
)
GPU_MEMORY_BYTES = Gauge(
    "steering_server_gpu_memory_bytes",
    "GPU memory allocated by the process, or zero when no GPU runtime is loaded.",
)
QUEUE_DEPTH = Gauge("steering_server_queue_depth", "Gauge-fit jobs waiting in the local scheduler.")


def update_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            GPU_MEMORY_BYTES.set(float(torch.cuda.memory_allocated()))
            return
    except Exception:
        pass
    GPU_MEMORY_BYTES.set(0.0)


async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
    update_gpu_memory()
    return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
