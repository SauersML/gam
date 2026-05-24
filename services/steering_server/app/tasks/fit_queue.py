from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from app.config import settings
from app.database import session_scope
from app.db_models import Steerer
from app.models import ConceptTarget
from app.observability.metrics import QUEUE_DEPTH
from app.services.steering import fit_gauge


class GaugeFitQueue:
    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._queued: set[str] = set()

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def depth(self) -> int:
        return len(self._queued)

    def enqueue(self, steerer_id: str) -> None:
        self._queued.add(steerer_id)
        QUEUE_DEPTH.set(self.depth())
        if settings.queue.run_inline:
            self._run_fit(steerer_id)
            return
        self._scheduler.add_job(
            self._run_fit,
            trigger="date",
            args=[steerer_id],
            id=f"fit:{steerer_id}",
            replace_existing=True,
        )

    def _run_fit(self, steerer_id: str) -> None:
        try:
            with session_scope() as session:
                steerer = session.scalar(select(Steerer).where(Steerer.id == steerer_id))
                if steerer is None:
                    return
                steerer.status = "fitting"
                session.flush()
                targets = [ConceptTarget.model_validate(raw) for raw in steerer.concept_targets]
                gauge, diagnostics = fit_gauge(steerer.model, steerer.layer, targets)
                steerer.gauge = gauge
                steerer.diagnostics = diagnostics
                steerer.status = "ready"
                steerer.error = None
        except Exception as exc:
            with session_scope() as session:
                steerer = session.scalar(select(Steerer).where(Steerer.id == steerer_id))
                if steerer is not None:
                    steerer.status = "failed"
                    steerer.error = str(exc)
        finally:
            self._queued.discard(steerer_id)
            QUEUE_DEPTH.set(self.depth())


fit_queue = GaugeFitQueue()
