"""Inference-time advisories surfaced to gamfit callers.

When a fit silently adjusts the model away from what was literally requested —
a cubic-regression ``k`` capped to the covariate's distinct-value count, a
low-cardinality basis degraded to a straight line, and so on — the Rust core
records a human-readable note (``FittedModelPayload.inference_notes``). The CLI
prints these via ``print_inference_summary``; gamfit surfaces the SAME notes as
:class:`GamInferenceWarning`\\ s at fit time and via :attr:`gamfit.Model.notes`,
so a basis reduction is never silent (mgcv emits an analogous ``warning()``).
"""

from __future__ import annotations

import warnings
from typing import Iterable


class GamInferenceWarning(UserWarning):
    """A fit produced a model that differs from what was literally requested.

    Subclasses :class:`UserWarning` so it is shown by default (the whole point
    is that the adjustment must not be silent), and so callers can route it with
    the standard :mod:`warnings` machinery — e.g. ``warnings.simplefilter(
    "error", gamfit.GamInferenceWarning)`` to turn a silent basis reduction into
    a hard failure, or ``"ignore"`` to suppress it.
    """


def emit_inference_warnings(notes: Iterable[str], *, stacklevel: int = 3) -> None:
    """Emit one :class:`GamInferenceWarning` per note.

    ``stacklevel`` defaults to 3 so the warning points at the caller of
    ``gamfit.fit`` (warn -> this helper -> the fit function -> user code) rather
    than at gamfit internals.
    """
    for note in notes:
        if note:
            warnings.warn(note, GamInferenceWarning, stacklevel=stacklevel)
