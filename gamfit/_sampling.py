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
from ._predictive import PosteriorPredictive
from ._summary import Summary

_NO_MODEL: bytes = b""


def _rust():
    from ._binding import rust_module

    return rust_module()


def _call(name: str, *args: Any) -> Any:
    from ._exceptions import map_exception

    try:
        return getattr(_rust(), name)(*args)
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
        return {f: getattr(self, f) for f in ("n_samples", "n_warmup", "n_chains", "target_accept", "seed")}


def _config_from_payload(cfg: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        n_samples=int(cfg.get("n_samples", 0)),
        n_warmup=int(cfg.get("n_warmup", 0)),
        n_chains=int(cfg.get("n_chains", 0)),
        target_accept=float(cfg.get("target_accept", 0.0)),
        seed=int(cfg.get("seed", 0)),
    )


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
    _model_bytes: bytes = field(repr=False, compare=False, default=_NO_MODEL)
    _name_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_name_index", {n: j for j, n in enumerate(self.coefficient_names)})

    @classmethod
    def from_ffi_payload(cls, payload: Mapping[str, Any], *, model_bytes: bytes = _NO_MODEL) -> "PosteriorSamples":
        import numpy as np
        p = payload
        nd, nc = int(p["n_draws"]), int(p["n_coeffs"])
        flat = np.asarray(p.get("samples_flat", []), dtype=float)
        if flat.size != nd * nc:
            raise ValueError(f"FFI sample payload shape mismatch: got {flat.size} floats, expected {nd} * {nc}")
        names = tuple(str(n) for n in p.get("coefficient_names", []))
        if len(names) != nc:
            names = tuple(f"beta_{j}" for j in range(nc))
        return cls(samples=flat.reshape(nd, nc), coefficient_names=names,
                   mean=np.asarray(p.get("posterior_mean", []), dtype=float),
                   std=np.asarray(p.get("posterior_std", []), dtype=float),
                   rhat=float(p["rhat"]), ess=float(p["ess"]), converged=bool(p["converged"]),
                   method=str(p.get("method", "nuts")), model_class=str(p.get("model_class", "standard")),
                   family_kind=str(p.get("family_kind", "identity")),
                   config=_config_from_payload(p.get("config", {})), _model_bytes=model_bytes)

    @classmethod
    def from_ffi_json(cls, raw: str, *, model_bytes: bytes = _NO_MODEL) -> "PosteriorSamples":
        return cls.from_ffi_payload(json.loads(raw), model_bytes=model_bytes)

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
        if isinstance(key, str):
            if key not in self._name_index:
                raise KeyError(f"unknown coefficient {key!r}; known: {list(self.coefficient_names)}")
            return self.samples[:, self._name_index[key]]
        return self.samples[key]

    def to_numpy(self) -> Any: return self.samples

    def to_pandas(self) -> Any:
        import pandas as pd
        return pd.DataFrame(self.samples, columns=list(self.coefficient_names))

    def interval(self, level: float = 0.95) -> Any:
        import numpy as np
        flat = _call("posterior_credible_interval",
                     np.asarray(self.samples, dtype=float).ravel().tolist(),
                     self.n_draws, self.n_coeffs, float(level))
        return np.asarray(flat, dtype=float).reshape(self.n_coeffs, 2)

    def summary(self, level: float = 0.95) -> Summary:
        ci = self.interval(level)
        coefs = [{"index": j, "name": self.coefficient_names[j], "estimate": float(self.mean[j]),
                  "std_error": float(self.std[j]), "ci_lower": float(ci[j, 0]),
                  "ci_upper": float(ci[j, 1])} for j in range(self.n_coeffs)]
        return Summary.from_dict({
            "kind": "posterior_samples", "method": self.method, "model_class": self.model_class,
            "family_kind": self.family_kind, "n_draws": self.n_draws, "n_coeffs": self.n_coeffs,
            "rhat": self.rhat, "ess": self.ess, "converged": self.converged,
            "credible_interval": float(level), "config": self.config.to_dict(), "coefficients": coefs})

    def _need_model(self) -> None:
        if not self._model_bytes:
            raise RuntimeError("PosteriorSamples has no model context; predict requires the original Model. "
                               "Re-sample via Model.sample(...) or use Model.predict(...) directly.")

    def _normalize(self, new_data: Any) -> tuple[Any, Any]:
        from ._tables import normalize_table
        headers, rows, _ = normalize_table(new_data)
        return headers, rows

    def predict(self, new_data: Any, *, level: float = 0.95) -> dict[str, Any]:
        import numpy as np
        self._need_model()
        headers, rows = self._normalize(new_data)
        raw = _call("posterior_predict_bands_table", self._model_bytes, headers, rows,
                    np.asarray(self.samples, dtype=float).ravel().tolist(),
                    self.n_draws, self.n_coeffs, float(level))
        parsed = json.loads(raw)
        return {k: np.asarray(parsed[k], dtype=float)
                for k in ("eta_mean", "eta_lower", "eta_upper", "mean", "mean_lower", "mean_upper")}

    def predict_draws(self, new_data: Any) -> PosteriorPredictive:
        import numpy as np
        eta, family_kind, model_class = self._posterior_predict_eta(new_data)
        mean_flat = _call("apply_inverse_link_array", eta.ravel().tolist(), family_kind)
        return PosteriorPredictive(eta=np.asarray(eta, dtype=float),
                                   mean=np.asarray(mean_flat, dtype=float).reshape(eta.shape),
                                   family_kind=family_kind, model_class=model_class)

    def _posterior_predict_eta(self, new_data: Any) -> tuple[Any, str, str]:
        import numpy as np
        self._need_model()
        headers, rows = self._normalize(new_data)
        parsed = json.loads(_call("posterior_predict_table", self._model_bytes, headers, rows,
                                  np.asarray(self.samples, dtype=float).ravel().tolist(),
                                  self.n_draws, self.n_coeffs))
        n_draws, n_rows = int(parsed["n_draws"]), int(parsed["n_rows"])
        flat = np.asarray(parsed.get("eta_flat", []), dtype=float)
        if flat.size != n_draws * n_rows:
            raise ValueError(f"posterior predict FFI payload shape mismatch: got {flat.size} floats, expected {n_draws} * {n_rows}")
        return (flat.reshape(n_draws, n_rows),
                str(parsed.get("family_kind", self.family_kind)),
                str(parsed.get("model_class", self.model_class)))

    def save(self, path: str | Path) -> str:
        import numpy as np
        out = Path(path)
        metadata = {"coefficient_names": list(self.coefficient_names), "method": self.method,
                    "model_class": self.model_class, "family_kind": self.family_kind,
                    "config": self.config.to_dict()}
        np.savez(out, samples=np.asarray(self.samples, dtype=float),
                 mean=np.asarray(self.mean, dtype=float), std=np.asarray(self.std, dtype=float),
                 rhat=np.float64(self.rhat), ess=np.float64(self.ess), converged=np.bool_(self.converged),
                 model_bytes=np.frombuffer(self._model_bytes, dtype=np.uint8),
                 metadata=np.asarray(json.dumps(metadata), dtype=object))
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> "PosteriorSamples":
        import numpy as np
        npz = np.load(Path(path), allow_pickle=True)
        md = json.loads(str(npz["metadata"].item()))
        samples = np.asarray(npz["samples"], dtype=float)
        nc = int(samples.shape[1])
        names_raw = list(md.get("coefficient_names", []))
        names = tuple(str(n) for n in names_raw) if len(names_raw) == nc else tuple(f"beta_{j}" for j in range(nc))
        return cls(samples=samples, coefficient_names=names,
                   mean=np.asarray(npz["mean"], dtype=float), std=np.asarray(npz["std"], dtype=float),
                   rhat=float(npz["rhat"].item()), ess=float(npz["ess"].item()),
                   converged=bool(npz["converged"].item()),
                   method=str(md.get("method", "nuts")), model_class=str(md.get("model_class", "standard")),
                   family_kind=str(md.get("family_kind", "identity")),
                   config=_config_from_payload(md.get("config", {})),
                   _model_bytes=bytes(np.asarray(npz["model_bytes"], dtype=np.uint8).tobytes()))

    def plot_trace(self, *, coefficients: Any = None, max_panels: int = 8, ax: Any = None) -> Any:
        import matplotlib.pyplot as plt
        import numpy as np

        sel_in = (list(range(min(max_panels, self.n_coeffs))) if coefficients is None
                  else [coefficients] if isinstance(coefficients, (str, int)) else list(coefficients))
        if not sel_in:
            raise ValueError("plot_trace: no coefficients selected")
        n = len(sel_in)
        if ax is None:
            fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n))
            axes_arr = np.atleast_2d(axes)
        else:
            axes_arr = np.atleast_2d(ax)
            fig = axes_arr[0, 0].figure
        for row, s in enumerate(sel_in):
            draws = self[s] if isinstance(s, str) else self.samples[:, int(s)]
            label = s if isinstance(s, str) else self.coefficient_names[int(s)]
            axes_arr[row, 0].plot(draws, color="#1d4ed8", linewidth=0.7)
            axes_arr[row, 0].set_title(f"{label} — trace"); axes_arr[row, 0].set_xlabel("draw")
            axes_arr[row, 1].hist(draws, bins=40, density=True, color="#1d4ed8", alpha=0.7)
            axes_arr[row, 1].set_title(f"{label} — density"); axes_arr[row, 1].set_xlabel("value")
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
