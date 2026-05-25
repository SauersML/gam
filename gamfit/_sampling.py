"""Posterior draws from gamfit's NUTS / Laplace sampler.

The :class:`PosteriorSamples` object returned by :meth:`gamfit.Model.sample`
is the user-facing surface for posterior reasoning. It is numpy-first:
the raw ``(n_draws, n_coeffs)`` matrix is exposed as a numpy array and
every derived statistic (means, standard deviations, credible
intervals, fitted-mean draws, response-scale bands) is delegated to the
Rust engine — Python stays a thin marshaling/repr layer.

``PosteriorSamples.method`` discloses which sampler produced the draws:

* ``"nuts"`` — exact No-U-Turn sampling around the fitted joint mode.
* ``"laplace"`` — iid draws from ``N(beta_hat, H_penalized^-1)``.

For models with exact NUTS support ``posterior.rhat``, ``posterior.ess``,
and ``posterior.converged`` carry meaningful values. For Laplace draws,
the chains are iid by construction, so the engine reports
``rhat = 1.0``, ``ess = n_draws``, ``converged = True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from ._paired import CumulativeIncidenceDraws, PairedPosteriorSamples
from ._predictive import PosteriorPredictive
from ._summary import Summary

# Sentinel for unbound posteriors loaded from disk without a model context.
_NO_MODEL: bytes = b""


# ---- FFI marshaling helpers ----------------------------------------------


def _rust():
    from ._binding import rust_module

    return rust_module()


def _map_exc(exc: Exception) -> Exception:
    from ._exceptions import map_exception

    return map_exception(exc)


def _normalize_table(new_data: Any):
    from ._tables import normalize_table

    return normalize_table(new_data)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Echo of the NUTS configuration the engine ran with."""

    n_samples: int
    n_warmup: int
    n_chains: int
    target_accept: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_warmup": self.n_warmup,
            "n_chains": self.n_chains,
            "target_accept": self.target_accept,
            "seed": self.seed,
        }


def _config_from_payload(cfg_raw: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        n_samples=int(cfg_raw.get("n_samples", 0)),
        n_warmup=int(cfg_raw.get("n_warmup", 0)),
        n_chains=int(cfg_raw.get("n_chains", 0)),
        target_accept=float(cfg_raw.get("target_accept", 0.0)),
        seed=int(cfg_raw.get("seed", 0)),
    )


@dataclass(frozen=True, eq=False, slots=True)
class PosteriorSamples:
    """Posterior draws over the model's coefficient vector.

    Returned by :meth:`gamfit.Model.sample`. Summary statistics, credible
    intervals, response-scale bands, and inverse-link math all dispatch
    to Rust; Python is a thin marshaling / repr / plotting layer.
    """

    samples: Any
    coefficient_names: tuple[str, ...]
    mean: Any
    std: Any
    rhat: float
    ess: float
    converged: bool
    method: str
    model_class: str
    family_kind: str
    config: SamplingConfig
    _model_bytes: bytes = field(repr=False, compare=False, default=_NO_MODEL)
    _name_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        derived = {name: j for j, name in enumerate(self.coefficient_names)}
        object.__setattr__(self, "_name_index", derived)

    # ---- Construction ----------------------------------------------------

    @classmethod
    def from_ffi_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        model_bytes: bytes = _NO_MODEL,
    ) -> "PosteriorSamples":
        import numpy as np

        n_draws = int(payload["n_draws"])
        n_coeffs = int(payload["n_coeffs"])
        flat = np.asarray(payload.get("samples_flat", []), dtype=float)
        if flat.size != n_draws * n_coeffs:
            raise ValueError(
                "FFI sample payload shape mismatch: "
                f"got {flat.size} floats, expected {n_draws} * {n_coeffs}"
            )
        samples = flat.reshape(n_draws, n_coeffs)
        names = tuple(str(name) for name in payload.get("coefficient_names", []))
        if len(names) != n_coeffs:
            names = tuple(f"beta_{j}" for j in range(n_coeffs))
        return cls(
            samples=samples,
            coefficient_names=names,
            mean=np.asarray(payload.get("posterior_mean", []), dtype=float),
            std=np.asarray(payload.get("posterior_std", []), dtype=float),
            rhat=float(payload["rhat"]),
            ess=float(payload["ess"]),
            converged=bool(payload["converged"]),
            method=str(payload.get("method", "nuts")),
            model_class=str(payload.get("model_class", "standard")),
            family_kind=str(payload.get("family_kind", "identity")),
            config=_config_from_payload(payload.get("config", {})),
            _model_bytes=model_bytes,
        )

    @classmethod
    def from_ffi_json(
        cls,
        raw: str,
        *,
        model_bytes: bytes = _NO_MODEL,
    ) -> "PosteriorSamples":
        return cls.from_ffi_payload(json.loads(raw), model_bytes=model_bytes)

    # ---- Convenience accessors ------------------------------------------

    @property
    def n_draws(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_coeffs(self) -> int:
        return int(self.samples.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_draws, self.n_coeffs)

    @property
    def is_exact(self) -> bool:
        return self.method == "nuts"

    def __len__(self) -> int:
        return self.n_draws

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                index = self._name_index[key]
            except KeyError as exc:
                raise KeyError(
                    f"unknown coefficient {key!r}; "
                    f"known: {list(self.coefficient_names)}"
                ) from exc
            return self.samples[:, index]
        return self.samples[key]

    def to_numpy(self) -> Any:
        return self.samples

    def to_pandas(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self.samples, columns=list(self.coefficient_names))

    # ---- Summary statistics ---------------------------------------------

    def interval(self, level: float = 0.95) -> Any:
        """Equal-tailed credible interval for each coefficient.

        Dispatches to ``posterior_credible_interval`` in Rust; returns an
        ``(n_coeffs, 2)`` numpy array of ``(lower, upper)`` bounds.
        """
        import numpy as np

        try:
            flat = _rust().posterior_credible_interval(
                np.asarray(self.samples, dtype=float).ravel().tolist(),
                self.n_draws,
                self.n_coeffs,
                float(level),
            )
        except Exception as exc:
            raise _map_exc(exc) from exc
        return np.asarray(flat, dtype=float).reshape(self.n_coeffs, 2)

    def summary(self, level: float = 0.95) -> Summary:
        """Per-coefficient posterior summary as a :class:`Summary`."""
        intervals = self.interval(level)
        coefficients = [
            {
                "index": j,
                "name": self.coefficient_names[j],
                "estimate": float(self.mean[j]),
                "std_error": float(self.std[j]),
                "ci_lower": float(intervals[j, 0]),
                "ci_upper": float(intervals[j, 1]),
            }
            for j in range(self.n_coeffs)
        ]
        return Summary.from_dict(
            {
                "kind": "posterior_samples",
                "method": self.method,
                "model_class": self.model_class,
                "family_kind": self.family_kind,
                "n_draws": self.n_draws,
                "n_coeffs": self.n_coeffs,
                "rhat": self.rhat,
                "ess": self.ess,
                "converged": self.converged,
                "credible_interval": float(level),
                "config": self.config.to_dict(),
                "coefficients": coefficients,
            }
        )

    # ---- Posterior fitted means -----------------------------------------

    def predict(
        self,
        new_data: Any,
        *,
        level: float = 0.95,
    ) -> dict[str, Any]:
        """Posterior credible bands for eta and E[y | x] on new data.

        Dispatches to ``posterior_predict_bands_table`` in Rust: builds
        the design matrix, evaluates ``samples @ X^T``, takes per-row
        quantiles, and applies the inverse link — without materializing
        the ``(n_draws, n_rows)`` eta matrix in Python.
        """
        import numpy as np

        if not self._model_bytes:
            raise RuntimeError(
                "PosteriorSamples has no model context; predict requires "
                "the original Model. Re-sample via Model.sample(...) or "
                "use Model.predict(...) directly."
            )
        headers, rows, _ = _normalize_table(new_data)
        samples = np.asarray(self.samples, dtype=float)
        try:
            raw = _rust().posterior_predict_bands_table(
                self._model_bytes,
                headers,
                rows,
                samples.ravel().tolist(),
                self.n_draws,
                self.n_coeffs,
                float(level),
            )
        except Exception as exc:
            raise _map_exc(exc) from exc
        parsed = json.loads(raw)
        return {
            key: np.asarray(parsed[key], dtype=float)
            for key in ("eta_mean", "eta_lower", "eta_upper", "mean", "mean_lower", "mean_upper")
        }

    def predict_draws(self, new_data: Any) -> PosteriorPredictive:
        """Full posterior fitted-mean draws on new data."""
        import numpy as np

        eta, family_kind, model_class = self._posterior_predict_eta(new_data)
        try:
            mean_flat = _rust().apply_inverse_link_array(eta.ravel().tolist(), family_kind)
        except Exception as exc:
            raise _map_exc(exc) from exc
        mean = np.asarray(mean_flat, dtype=float).reshape(eta.shape)
        return PosteriorPredictive(
            eta=np.asarray(eta, dtype=float),
            mean=mean,
            family_kind=family_kind,
            model_class=model_class,
        )

    def _posterior_predict_eta(self, new_data: Any) -> tuple[Any, str, str]:
        """Evaluate per-draw linear predictors through the Rust FFI."""
        import numpy as np

        if not self._model_bytes:
            raise RuntimeError(
                "PosteriorSamples has no model context; predict requires "
                "the original Model. Re-sample via Model.sample(...) or "
                "use Model.predict(...) directly."
            )
        headers, rows, _ = _normalize_table(new_data)
        samples = np.asarray(self.samples, dtype=float)
        try:
            raw = _rust().posterior_predict_table(
                self._model_bytes,
                headers,
                rows,
                samples.ravel().tolist(),
                self.n_draws,
                self.n_coeffs,
            )
        except Exception as exc:
            raise _map_exc(exc) from exc
        parsed = json.loads(raw)
        n_draws = int(parsed["n_draws"])
        n_rows = int(parsed["n_rows"])
        flat = np.asarray(parsed.get("eta_flat", []), dtype=float)
        if flat.size != n_draws * n_rows:
            raise ValueError(
                "posterior predict FFI payload shape mismatch: "
                f"got {flat.size} floats, expected {n_draws} * {n_rows}"
            )
        return (
            flat.reshape(n_draws, n_rows),
            str(parsed.get("family_kind", self.family_kind)),
            str(parsed.get("model_class", self.model_class)),
        )

    # ---- Persistence ----------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Save the posterior to an ``.npz`` archive."""
        import numpy as np

        out = Path(path)
        metadata = {
            "coefficient_names": list(self.coefficient_names),
            "method": self.method,
            "model_class": self.model_class,
            "family_kind": self.family_kind,
            "config": self.config.to_dict(),
        }
        np.savez(
            out,
            samples=np.asarray(self.samples, dtype=float),
            mean=np.asarray(self.mean, dtype=float),
            std=np.asarray(self.std, dtype=float),
            rhat=np.float64(self.rhat),
            ess=np.float64(self.ess),
            converged=np.bool_(self.converged),
            model_bytes=np.frombuffer(self._model_bytes, dtype=np.uint8),
            metadata=np.asarray(json.dumps(metadata), dtype=object),
        )
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> "PosteriorSamples":
        """Load a :class:`PosteriorSamples` from an ``.npz`` archive."""
        import numpy as np

        npz = np.load(Path(path), allow_pickle=True)
        metadata = json.loads(str(npz["metadata"].item()))
        names_raw = list(metadata.get("coefficient_names", []))
        samples = np.asarray(npz["samples"], dtype=float)
        n_coeffs = int(samples.shape[1])
        names = (
            tuple(str(n) for n in names_raw)
            if len(names_raw) == n_coeffs
            else tuple(f"beta_{j}" for j in range(n_coeffs))
        )
        model_bytes = bytes(np.asarray(npz["model_bytes"], dtype=np.uint8).tobytes())
        return cls(
            samples=samples,
            coefficient_names=names,
            mean=np.asarray(npz["mean"], dtype=float),
            std=np.asarray(npz["std"], dtype=float),
            rhat=float(npz["rhat"].item()),
            ess=float(npz["ess"].item()),
            converged=bool(npz["converged"].item()),
            method=str(metadata.get("method", "nuts")),
            model_class=str(metadata.get("model_class", "standard")),
            family_kind=str(metadata.get("family_kind", "identity")),
            config=_config_from_payload(metadata.get("config", {})),
            _model_bytes=model_bytes,
        )

    # ---- Plotting -------------------------------------------------------

    def plot_trace(
        self,
        *,
        coefficients: Any = None,
        max_panels: int = 8,
        ax: Any = None,
    ) -> Any:
        """Matplotlib trace + marginal-density plot."""
        import matplotlib.pyplot as plt
        import numpy as np

        selected: list[str | int]
        if coefficients is None:
            selected = list(range(min(max_panels, self.n_coeffs)))
        elif isinstance(coefficients, (str, int)):
            selected = [coefficients]
        else:
            selected = list(coefficients)
        if not selected:
            raise ValueError("plot_trace: no coefficients selected")

        n_panels = len(selected)
        if ax is None:
            fig, axes = plt.subplots(n_panels, 2, figsize=(10, 2.2 * n_panels))
            axes_arr = np.atleast_2d(axes)
        else:
            axes_arr = np.atleast_2d(ax)
            fig = axes_arr[0, 0].figure

        for row, sel in enumerate(selected):
            draws = self[sel] if isinstance(sel, str) else self.samples[:, int(sel)]
            label = sel if isinstance(sel, str) else self.coefficient_names[int(sel)]
            axes_arr[row, 0].plot(draws, color="#1d4ed8", linewidth=0.7)
            axes_arr[row, 0].set_title(f"{label} — trace")
            axes_arr[row, 0].set_xlabel("draw")
            axes_arr[row, 1].hist(draws, bins=40, density=True, color="#1d4ed8", alpha=0.7)
            axes_arr[row, 1].set_title(f"{label} — density")
            axes_arr[row, 1].set_xlabel("value")

        fig.tight_layout()
        return fig

    # ---- Reprs ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PosteriorSamples(n_draws={self.n_draws}, n_coeffs={self.n_coeffs}, "
            f"method={self.method!r}, rhat={self.rhat:.4f}, ess={self.ess:.1f}, "
            f"converged={self.converged})"
        )

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


__all__ = [
    "CumulativeIncidenceDraws",
    "PairedPosteriorSamples",
    "PosteriorPredictive",
    "PosteriorSamples",
    "SamplingConfig",
]
