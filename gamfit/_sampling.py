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

        Dispatches to the Rust kernel ``posterior_eta_bands`` which does the
        quantile reductions and applies the inverse link in-place. Returns
        six length-``n_rows`` numpy arrays.
        """
        import numpy as np

        eta = np.asarray(self.eta, dtype=float)
        try:
            raw = _rust().posterior_eta_bands(
                eta.ravel().tolist(),
                int(eta.shape[0]),
                int(eta.shape[1]),
                self.family_kind,
                float(level),
            )
        except Exception as exc:
            raise _map_exc(exc) from exc
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

    def cumulative_incidence(
        self,
        new_data: Any,
        times: Any,
        *,
        level: float = 0.95,
    ) -> CumulativeIncidenceDraws:
        """Target-cause CIF draws using paired target/competing rows."""
        import numpy as np

        if not self.target._model_bytes or not self.competing._model_bytes:
            raise RuntimeError(
                "PairedPosteriorSamples has no model context; cumulative_incidence "
                "requires samples returned by Model.sample_paired(...)."
            )
        headers, rows, _ = _normalize_table(new_data)
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        payload = {"times": times_arr.tolist(), "level": float(level)}
        try:
            raw = _rust().paired_cumulative_incidence_table(
                self.target._model_bytes,
                self.competing._model_bytes,
                np.asarray(self.target.samples, dtype=float),
                np.asarray(self.competing.samples, dtype=float),
                headers,
                rows,
                json.dumps(payload),
            )
        except Exception as exc:
            raise _map_exc(exc) from exc
        return CumulativeIncidenceDraws.from_ffi_payload(json.loads(raw))

    def __repr__(self) -> str:
        return (
            f"PairedPosteriorSamples(n_draws={self.n_draws}, "
            f"target_method={self.target.method!r}, "
            f"competing_method={self.competing.method!r})"
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

        Dispatches to the Rust ``posterior_credible_interval`` kernel for
        the quantile reductions; returns an ``(n_coeffs, 2)`` numpy array
        of ``(lower, upper)`` bounds.
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

        Dispatches to the Rust ``posterior_predict_bands_table`` kernel
        which builds the design matrix, evaluates ``samples @ X^T``, takes
        per-row quantiles, and pushes the link bounds through the inverse
        link — all without ever materialising the ``(n_draws, n_rows)``
        eta matrix in Python.
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
