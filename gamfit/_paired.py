"""Paired posterior samples and cumulative-incidence draws.

Split out from :mod:`gamfit._sampling` so the main `PosteriorSamples`
module stays a thin marshaling layer. Container math (CIF draws,
paired-sample dispatch) lives in Rust; this file only holds the
dataclasses that hold the result arrays and the FFI marshaling glue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from ._sampling import PosteriorSamples

# Sentinel for unbound posteriors loaded from disk without a model context.
_NO_MODEL: bytes = b""


@dataclass(frozen=True, eq=False, slots=True)
class CumulativeIncidenceDraws:
    """Paired posterior draws for a target-cause cumulative incidence curve."""

    times: Any
    draws: Any
    mean: Any
    lower: Any
    upper: Any
    level: float

    @classmethod
    def from_ffi_payload(cls, payload: Mapping[str, Any]) -> "CumulativeIncidenceDraws":
        import numpy as np

        n_draws = int(payload["n_draws"])
        n_rows = int(payload["n_rows"])
        n_times = int(payload["n_times"])
        shape = (n_draws, n_rows, n_times)
        draws = np.asarray(payload.get("cif_flat", []), dtype=float)
        if draws.size != n_draws * n_rows * n_times:
            raise ValueError(
                "paired CIF payload shape mismatch: "
                f"got {draws.size} floats, expected {n_draws} * {n_rows} * {n_times}"
            )
        summary_shape = (n_rows, n_times)
        return cls(
            times=np.asarray(payload.get("times", []), dtype=float),
            draws=draws.reshape(shape),
            mean=np.asarray(payload.get("mean_flat", []), dtype=float).reshape(summary_shape),
            lower=np.asarray(payload.get("lower_flat", []), dtype=float).reshape(summary_shape),
            upper=np.asarray(payload.get("upper_flat", []), dtype=float).reshape(summary_shape),
            level=float(payload.get("level", 0.95)),
        )

    @property
    def n_draws(self) -> int:
        return int(self.draws.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.draws.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.draws.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_draws, self.n_rows, self.n_times)

    def __repr__(self) -> str:
        return (
            f"CumulativeIncidenceDraws(n_draws={self.n_draws}, n_rows={self.n_rows}, "
            f"n_times={self.n_times}, level={self.level:.3f})"
        )


@dataclass(frozen=True, eq=False, slots=True)
class PairedPosteriorSamples:
    """Posterior samples from two linked fits with draw rows paired by index."""

    target: "PosteriorSamples"
    competing: "PosteriorSamples"

    @classmethod
    def from_ffi_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        target_model_bytes: bytes = _NO_MODEL,
        competing_model_bytes: bytes = _NO_MODEL,
    ) -> "PairedPosteriorSamples":
        from ._sampling import PosteriorSamples

        target = PosteriorSamples.from_ffi_payload(
            payload["target"], model_bytes=target_model_bytes
        )
        competing = PosteriorSamples.from_ffi_payload(
            payload["competing"], model_bytes=competing_model_bytes
        )
        if target.n_draws != competing.n_draws:
            raise ValueError(
                "paired posterior payload has unequal draw counts: "
                f"target={target.n_draws}, competing={competing.n_draws}"
            )
        return cls(target=target, competing=competing)

    @classmethod
    def from_ffi_json(
        cls,
        raw: str,
        *,
        target_model_bytes: bytes = _NO_MODEL,
        competing_model_bytes: bytes = _NO_MODEL,
    ) -> "PairedPosteriorSamples":
        return cls.from_ffi_payload(
            json.loads(raw),
            target_model_bytes=target_model_bytes,
            competing_model_bytes=competing_model_bytes,
        )

    @property
    def n_draws(self) -> int:
        return self.target.n_draws

    def __repr__(self) -> str:
        return (
            f"PairedPosteriorSamples(n_draws={self.n_draws}, "
            f"target_method={self.target.method!r}, "
            f"competing_method={self.competing.method!r})"
        )


__all__ = ["CumulativeIncidenceDraws", "PairedPosteriorSamples"]
