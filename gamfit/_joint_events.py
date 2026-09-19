"""The joint latent-signature event model: fit a cohort, condition on a new
history, forecast, save and reload, all through the one Rust model the CLI also
calls (``gam joint-events``). At rank zero every mark has a constant rate, and
every forecast averages that rate's exact posterior rather than a fitted rate.
See ``docs/latent-signatures.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ._binding import rust_module
from ._event_history import _column, _labels


class JointEventModel:
    """A fitted or loaded joint event model."""

    def __init__(self, native: Any) -> None:
        self._native = native

    @property
    def mark_names(self) -> list[str]:
        return list(self._native.mark_names())

    @property
    def mark_kinds(self) -> dict[str, str]:
        """Kind of every mark: ``recurrent``, ``once`` or ``terminal``."""
        return dict(zip(self._native.mark_names(), self._native.mark_kinds()))

    def save(self, path: str | Path) -> None:
        """Write the saved model: the frozen encoding schema and the posterior
        that forecasting integrates, never the training records."""
        self._native.save(str(path))

    def forecast(
        self,
        entry: float,
        exit: float,
        events: Sequence[tuple[float, Any]],
        horizons: Sequence[float],
    ) -> dict[str, np.ndarray]:
        """Condition on one history and forecast after its exit ``s``.

        ``events`` are ``(time, mark)`` pairs in any order; an event at or
        before ``entry`` is prior history. Returns ``survival``, the probability
        that no terminal mark fires by ``s + u``; ``incidence`` (horizons ×
        marks), the probability that each mark's next occurrence falls in
        ``(s, s + u]`` before any terminal mark; and ``incidence_error``, a bound
        on each incidence's numerical error."""
        out = self._native.forecast(
            float(entry),
            float(exit),
            [float(time) for time, _ in events],
            [str(mark) for _, mark in events],
            [float(h) for h in horizons],
        )
        return {
            name: np.asarray(out[name])
            for name in ("horizons", "survival", "incidence", "incidence_error")
        }


def fit_joint_event_model(
    subjects: Any,
    events: Any,
    *,
    marks: Mapping[str, str] | Sequence[str] | None = None,
    id_column: str = "id",
) -> JointEventModel:
    """Fit the joint event model.

    ``subjects`` has columns ``id, entry, exit`` and ``events`` has ``id, time,
    mark``; rows may come in any order. ``marks`` declares the mark
    vocabulary and each mark's kind, e.g. ``{"diagnosis": "once", "death":
    "terminal"}``, or a sequence of names that are all recurrent; without it the
    observed marks, all recurrent."""
    if marks is None:
        declared_marks = None
    elif isinstance(marks, Mapping):
        declared_marks = [(str(k), str(v)) for k, v in marks.items()]
    else:
        declared_marks = [(str(m), "recurrent") for m in marks]
    native = rust_module().fit_joint_event_model(
        declared_marks,
        _labels(_column(subjects, id_column), "subject identifiers"),
        _column(subjects, "entry").astype(float).tolist(),
        _column(subjects, "exit").astype(float).tolist(),
        _labels(_column(events, id_column), "event subject identifiers"),
        _column(events, "time").astype(float).tolist(),
        _labels(_column(events, "mark"), "mark names"),
    )
    return JointEventModel(native)


def load_joint_event_model(path: str | Path) -> JointEventModel:
    """Load a model saved by :meth:`JointEventModel.save` or ``gam joint-events fit``."""
    return JointEventModel(rust_module().load_joint_event_model(str(path)))
