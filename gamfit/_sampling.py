"""Posterior draws from gamfit's NUTS / Laplace sampler.

Numpy-first surface: the raw ``(n_draws, n_coeffs)`` matrix is exposed as a
numpy array; every derived statistic dispatches to the Rust engine. Python
stays a thin marshaling / repr / plotting layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from ._paired import CumulativeIncidenceDraws, PairedPosteriorSamples
from ._summary import Summary


@dataclass(frozen=True, eq=False, slots=True)
class PosteriorPredictive:
    """Per-row posterior fitted-mean draws on the link and response scales."""

    eta: Any
    mean: Any
    family_kind: str
    model_class: str
    # Serialized parameterized inverse-link spec (JSON). Carries the per-fit
    # state (skew/tail, mixture weights, latent SD) the bare ``family_kind`` tag
    # drops, so the parameterized links' response-scale bands are exact
    # (issue #1133). None for plain links / older payloads.
    link_spec: str | None = None

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

        Dispatches to ``posterior_eta_bands`` in Rust; quantile
        reductions and inverse-link push-through.
        """
        import numpy as np

        from ._binding import rust_module
        from ._exceptions import map_exception

        eta = np.ascontiguousarray(np.asarray(self.eta, dtype=np.float64))
        try:
            parsed = rust_module().posterior_eta_bands(
                eta,
                self.family_kind,
                float(level),
                self.link_spec,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return {
            "linear_predictor": np.asarray(parsed["linear_predictor"], dtype=float),
            "linear_predictor_lower": np.asarray(parsed["linear_predictor_lower"], dtype=float),
            "linear_predictor_upper": np.asarray(parsed["linear_predictor_upper"], dtype=float),
            "mean": np.asarray(parsed["mean"], dtype=float),
            "mean_lower": np.asarray(parsed["mean_lower"], dtype=float),
            "mean_upper": np.asarray(parsed["mean_upper"], dtype=float),
        }

    def __repr__(self) -> str:
        return (
            f"PosteriorPredictive(n_draws={self.n_draws}, n_rows={self.n_rows}, "
            f"family_kind={self.family_kind!r}, model_class={self.model_class!r})"
        )


_NO_MODEL: bytes = b""


def _call(name: str, *args: Any) -> Any:
    from ._binding import rust_module
    from ._exceptions import map_exception
    try:
        return getattr(rust_module(), name)(*args)
    except Exception as exc:
        raise map_exception(exc) from exc


@dataclass(frozen=True, slots=True)
class SamplingConfig:
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


def _config_from_payload(cfg: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        n_samples=int(cfg.get("n_samples", 0)),
        n_warmup=int(cfg.get("n_warmup", 0)),
        n_chains=int(cfg.get("n_chains", 0)),
        target_accept=float(cfg.get("target_accept", 0.0)),
        seed=int(cfg.get("seed", 0)),
    )


def _coefficient_names(raw_names: Any, n_coeffs: int) -> tuple[str, ...]:
    raw = _call(
        "posterior_coefficient_names_json",
        json.dumps({"coefficient_names": raw_names, "n_coeffs": int(n_coeffs)}),
    )
    return tuple(json.loads(raw))


@dataclass(frozen=True, eq=False, slots=True)
class PosteriorSamples:
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
    # Serialized parameterized inverse-link spec (JSON); see PosteriorPredictive.
    link_spec: str | None = None
    _model_bytes: bytes = field(repr=False, compare=False, default=_NO_MODEL)
    _name_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_name_index",
            dict(zip(self.coefficient_names, range(len(self.coefficient_names)))),
        )

    @classmethod
    def from_ffi_payload(cls, payload: Mapping[str, Any], *, model_bytes: bytes = _NO_MODEL) -> "PosteriorSamples":
        import numpy as np
        p = payload
        samples = np.asarray(p["samples"], dtype=np.float64)
        # allow-list (a): FFI input validation
        if samples.ndim != 2:
            raise ValueError(
                f"FFI sample payload must be a 2-D draw matrix; got shape {samples.shape}"
            )
        samples = np.ascontiguousarray(samples)
        nc = int(samples.shape[1])
        names = _coefficient_names(p.get("coefficient_names", []), nc)
        return cls(samples=samples, coefficient_names=names,
                   mean=np.asarray(p.get("posterior_mean", []), dtype=float),
                   std=np.asarray(p.get("posterior_std", []), dtype=float),
                   rhat=float(p["rhat"]), ess=float(p["ess"]), converged=bool(p["converged"]),
                   method=str(p.get("method", "nuts")), model_class=str(p.get("model_class", "standard")),
                   family_kind=str(p.get("family_kind", "identity")),
                   link_spec=(str(p["link_spec"]) if p.get("link_spec") is not None else None),
                   config=_config_from_payload(p.get("config", {})), _model_bytes=model_bytes)

    @property
    def n_draws(self) -> int: return int(self.samples.shape[0])
    @property
    def n_coeffs(self) -> int: return int(self.samples.shape[1])
    @property
    def shape(self) -> tuple[int, int]: return (self.n_draws, self.n_coeffs)
    @property
    def is_exact(self) -> bool: return self.method == "nuts"

    def __len__(self) -> int: return self.n_draws
    def __iter__(self) -> Iterator[Any]: return iter(self.samples)

    def __getitem__(self, key: Any) -> Any:
        try:
            return self.samples[key]
        except (IndexError, TypeError):
            try:
                return self.samples[:, self._name_index[key]]
            except KeyError as exc:
                raise KeyError(f"unknown coefficient {key!r}; known: {list(self.coefficient_names)}") from exc

    def to_numpy(self) -> Any: return self.samples

    def to_pandas(self) -> Any:
        import pandas as pd
        return pd.DataFrame(self.samples, columns=list(self.coefficient_names))

    def interval(self, level: float = 0.95) -> Any:
        import numpy as np
        samples = np.ascontiguousarray(np.asarray(self.samples, dtype=np.float64))
        return np.asarray(
            _call("posterior_credible_interval", samples, float(level)), dtype=float
        )

    def summary(self, level: float = 0.95) -> Summary:
        import numpy as np
        samples = np.ascontiguousarray(np.asarray(self.samples, dtype=np.float64))
        posterior_mean = np.ascontiguousarray(np.asarray(self.mean, dtype=np.float64))
        posterior_std = np.ascontiguousarray(np.asarray(self.std, dtype=np.float64))
        raw = _call(
            "posterior_samples_summary_json",
            samples,
            posterior_mean,
            posterior_std,
            json.dumps({
                "level": float(level),
                "coefficient_names": self.coefficient_names,
                "rhat": float(self.rhat),
                "ess": float(self.ess),
                "converged": bool(self.converged),
                "method": self.method,
                "model_class": self.model_class,
                "family_kind": self.family_kind,
                "config": self.config.to_dict(),
            }),
        )
        return Summary.from_dict(json.loads(raw))

    def _need_model(self) -> None:
        # allow-list (a): FFI input validation
        if not self._model_bytes:
            raise RuntimeError("PosteriorSamples has no model context; predict requires the original Model. "
                               "Re-sample via Model.sample(...) or use Model.predict(...) directly.")

    def _normalize(self, new_data: Any) -> tuple[Any, Any]:
        from ._tables import normalize_table
        headers, rows, _ = normalize_table(new_data)
        return headers, rows

    def predict(self, new_data: Any, *, level: float = 0.95) -> dict[str, Any]:
        """Posterior predictive bands for ``new_data``.

        Returns a dict with linear-predictor-scale bands
        (``linear_predictor``, ``linear_predictor_lower``,
        ``linear_predictor_upper``) and response-scale summaries
        (``mean``, ``mean_lower``, ``mean_upper``), each a 1-D array of
        length ``n_rows``. The vocabulary matches ``Model.predict`` — no
        engine-internal ``eta`` key is exposed.
        """
        import numpy as np
        self._need_model()
        h, r = self._normalize(new_data)
        samples = np.ascontiguousarray(np.asarray(self.samples, dtype=np.float64))
        parsed = _call(
            "posterior_predict_bands_table",
            self._model_bytes,
            h,
            r,
            samples,
            float(level),
        )
        return {
            "linear_predictor": np.asarray(parsed["linear_predictor"], dtype=float),
            "linear_predictor_lower": np.asarray(parsed["linear_predictor_lower"], dtype=float),
            "linear_predictor_upper": np.asarray(parsed["linear_predictor_upper"], dtype=float),
            "mean": np.asarray(parsed["mean"], dtype=float),
            "mean_lower": np.asarray(parsed["mean_lower"], dtype=float),
            "mean_upper": np.asarray(parsed["mean_upper"], dtype=float),
        }

    def predict_draws(self, new_data: Any) -> PosteriorPredictive:
        import numpy as np
        eta, family_kind, model_class, link_spec = self._posterior_predict_eta(new_data)
        mean = _call("apply_inverse_link_array", eta, family_kind, link_spec)
        return PosteriorPredictive(eta=np.asarray(eta, dtype=float),
                                   mean=np.asarray(mean, dtype=float),
                                   family_kind=family_kind, model_class=model_class,
                                   link_spec=link_spec)

    def _posterior_predict_eta(self, new_data: Any) -> tuple[Any, str, str, str | None]:
        import numpy as np
        self._need_model()
        h, r = self._normalize(new_data)
        samples = np.ascontiguousarray(np.asarray(self.samples, dtype=np.float64))
        p = _call("posterior_predict_table", self._model_bytes, h, r, samples)
        eta = np.asarray(p["eta"], dtype=float)
        # allow-list (a): FFI input validation
        if eta.ndim != 2:
            raise ValueError(
                f"posterior predict FFI payload must be a 2-D eta matrix; got shape {eta.shape}"
            )
        link_spec = p.get("link_spec", self.link_spec)
        link_spec = str(link_spec) if link_spec is not None else None
        return (eta, str(p.get("family_kind", self.family_kind)),
                str(p.get("model_class", self.model_class)), link_spec)

    def save(self, path: str | Path) -> str:
        import numpy as np
        # numpy.savez appends ".npz" unless the target already carries that
        # suffix. Mirror that rule here so the returned path is the file that
        # actually lands on disk and so load(save(p)) round-trips for any p.
        out = Path(path)
        if out.suffix != ".npz":
            out = out.with_name(out.name + ".npz")
        md = {"coefficient_names": list(self.coefficient_names), "method": self.method,
              "model_class": self.model_class, "family_kind": self.family_kind,
              "link_spec": self.link_spec, "config": self.config.to_dict()}
        np.savez(out, samples=np.asarray(self.samples, dtype=float),
                 mean=np.asarray(self.mean, dtype=float), std=np.asarray(self.std, dtype=float),
                 rhat=np.float64(self.rhat), ess=np.float64(self.ess), converged=np.bool_(self.converged),
                 model_bytes=np.frombuffer(self._model_bytes, dtype=np.uint8),
                 metadata=np.asarray(json.dumps(md), dtype=object))
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> "PosteriorSamples":
        import numpy as np
        # Tolerate the numpy ".npz" auto-suffix: a path written by save() with
        # no explicit extension lives on disk at "<path>.npz". Resolve to the
        # file that exists so load() accepts both the literal and suffixed form.
        target = Path(path)
        if not target.exists() and target.suffix != ".npz":
            suffixed = target.with_name(target.name + ".npz")
            if suffixed.exists():
                target = suffixed
        npz = np.load(target, allow_pickle=True)
        md = json.loads(str(npz["metadata"].item()))
        samples = np.asarray(npz["samples"], dtype=float)
        nc = int(samples.shape[1])
        names = _coefficient_names(md.get("coefficient_names", []), nc)
        return cls(samples=samples, coefficient_names=names,
                   mean=np.asarray(npz["mean"], dtype=float), std=np.asarray(npz["std"], dtype=float),
                   rhat=float(npz["rhat"].item()), ess=float(npz["ess"].item()),
                   converged=bool(npz["converged"].item()),
                   method=str(md.get("method", "nuts")), model_class=str(md.get("model_class", "standard")),
                   family_kind=str(md.get("family_kind", "identity")),
                   link_spec=(str(md["link_spec"]) if md.get("link_spec") is not None else None),
                   config=_config_from_payload(md.get("config", {})),
                   _model_bytes=bytes(np.asarray(npz["model_bytes"], dtype=np.uint8).tobytes()))

    def plot_trace(self, *, coefficients: Any = None, max_panels: int = 8) -> Any:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        selection = json.loads(_call(
            "posterior_trace_selection_json",
            json.dumps({
                "coefficient_names": self.coefficient_names,
                "coefficients": coefficients,
                "max_panels": int(max_panels),
            }),
        ))
        labels = list(selection["labels"])
        data = np.asarray(self.samples, dtype=float)[:, np.asarray(selection["indices"], dtype=int)]
        n = len(labels)
        fig, axes_arr = plt.subplots(n, 2, figsize=(10, 2.2 * n), squeeze=False)
        frame = pd.DataFrame(data, columns=labels)
        frame.plot(ax=axes_arr[:, 0], subplots=True, legend=False, color="#1d4ed8", linewidth=0.7)
        frame.plot(
            kind="hist",
            ax=axes_arr[:, 1],
            subplots=True,
            legend=False,
            bins=40,
            density=True,
            color="#1d4ed8",
            alpha=0.7,
        )
        axes_arr[0, 0].set_xlabel("draw")
        axes_arr[0, 1].set_xlabel("value")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (f"PosteriorSamples(n_draws={self.n_draws}, n_coeffs={self.n_coeffs}, "
                f"method={self.method!r}, rhat={self.rhat:.4f}, ess={self.ess:.1f}, "
                f"converged={self.converged})")

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


__all__ = ["CumulativeIncidenceDraws", "PairedPosteriorSamples", "PosteriorPredictive",
           "PosteriorSamples", "SamplingConfig"]
