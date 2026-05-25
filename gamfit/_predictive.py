"""Container for posterior fitted-mean draws on the link and response scales.

Split out from :mod:`gamfit._sampling` so the main module stays a thin
``PosteriorSamples`` layer. All band reductions and the inverse-link
push-through live in Rust (``posterior_eta_bands``); this file is the
dataclass + FFI marshaling glue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, eq=False, slots=True)
class PosteriorPredictive:
    """Per-row posterior fitted-mean draws on the link and response scales."""

    eta: Any
    mean: Any
    family_kind: str
    model_class: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.eta.shape)

    @property
    def n_draws(self) -> int:
        return int(self.eta.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.eta.shape[1])

    def summary(self, level: float = 0.95) -> dict[str, Any]:
        """Collapse fitted-mean draws to per-row credible bands.

        Dispatches to ``posterior_eta_bands`` in Rust for the quantile
        reductions and inverse-link push-through.
        """
        import numpy as np

        from ._binding import rust_module
        from ._exceptions import map_exception

        eta = np.asarray(self.eta, dtype=float)
        try:
            raw = rust_module().posterior_eta_bands(
                eta.ravel().tolist(),
                int(eta.shape[0]),
                int(eta.shape[1]),
                self.family_kind,
                float(level),
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        parsed = json.loads(raw)
        return {
            key: np.asarray(parsed[key], dtype=float)
            for key in ("eta_mean", "eta_lower", "eta_upper", "mean", "mean_lower", "mean_upper")
        }

    def __repr__(self) -> str:
        return (
            f"PosteriorPredictive(n_draws={self.n_draws}, n_rows={self.n_rows}, "
            f"family_kind={self.family_kind!r}, model_class={self.model_class!r})"
        )


__all__ = ["PosteriorPredictive"]
