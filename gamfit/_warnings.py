"""Fit advisories surfaced to gamfit callers.

When a fit adjusts the model away from what was literally requested — a
cubic-regression ``k`` capped to the covariate's distinct-value count, a
low-cardinality basis degraded to a straight line, and so on — the Rust core
records an advisory (``FittedModelPayload.inference_notes``). The CLI prints
these to stderr; gamfit raises the SAME notes as :class:`GamInferenceWarning`\\ s
at fit time, so a basis reduction is never silent (mgcv emits an analogous
``warning()``).

Defaults the engine chose on the caller's behalf (the knot count of a default
B-spline, per-margin tensor sizes) are informational notes, not advisories:
they are listed in :attr:`gamfit.Model.notes` and the summary, never warned.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class GamInferenceWarning(UserWarning):
    """A fit produced a model that differs from what was literally requested.

    Subclasses :class:`UserWarning` so it is shown by default (the whole point
    is that the adjustment must not be silent), and so callers can route it with
    the standard :mod:`warnings` machinery — e.g. ``warnings.simplefilter(
    "error", gamfit.errors.GamInferenceWarning)`` to turn a silent basis reduction into
    a hard failure, or ``"ignore"`` to suppress it.
    """


def _stacklevel_outside_package() -> int:
    """The ``warnings.warn`` stacklevel of the first frame outside gamfit.

    Level 1 is the frame that calls ``warnings.warn`` — the caller of this
    helper. Counting from there past every frame whose code lives in the gamfit
    package makes the warning point at the user's line whatever gamfit entry
    point (``fit``, ``fit_array``, an estimator wrapper) sits in between.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE_DIR
    ):
        frame = frame.f_back
        level += 1
    return level


def emit_inference_warnings(notes: Iterable[str]) -> None:
    """Emit one :class:`GamInferenceWarning` per advisory, attributed to the
    first caller frame outside the gamfit package."""
    stacklevel = _stacklevel_outside_package()
    for note in notes:
        if note:
            warnings.warn(note, GamInferenceWarning, stacklevel=stacklevel)
