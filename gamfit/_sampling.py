"""Posterior draws from NUTS sampling.

The :class:`PosteriorSamples` object is what :meth:`gamfit.Model.sample`
returns.  It is intentionally numpy-first: the raw `(n_draws, n_coeffs)`
matrix is exposed as a numpy array and every derived statistic (means,
standard deviations, credible intervals) is also numpy.  Subscripting,
iteration, length, and a Jupyter-friendly HTML repr come along for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from ._summary import Summary


@dataclass(frozen=True)
class SamplingConfig:
    """Echo of the NUTS configuration the Rust core actually ran with.

    All fields are populated from the FFI payload so callers can reconstruct
    exactly which sampler invocation produced the draws — useful for
    reproducibility logs and for telling whether their explicit `samples=…`
    request was honored or auto-derived from the model dimension.
    """

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


@dataclass(frozen=True, eq=False)
class PosteriorSamples:
    """Posterior draws over the model's coefficient vector.

    Attributes
    ----------
    samples:
        ``(n_draws, n_coeffs)`` numpy array of draws.  ``n_draws`` is
        ``n_chains * n_samples`` (warmup is already discarded by the Rust
        core).
    coefficient_names:
        Tuple of column labels for ``samples``.  Currently the FFI emits
        ``("beta_0", "beta_1", ...)``; in future releases this may carry
        the same names the fitted model exposes via :class:`Summary`.
    mean / std:
        Per-coefficient posterior moments returned by the sampler.  Equal
        to ``samples.mean(0)`` / ``samples.std(0, ddof=1)`` up to NaN
        handling.
    rhat:
        Maximum split-Rhat across coefficients.  Values close to 1.0
        indicate the chains have mixed; the Rust core flags
        ``converged = (rhat < 1.1)``.
    ess:
        Minimum effective sample size across coefficients.
    converged:
        Convenience boolean equivalent to ``rhat < 1.1``.
    config:
        :class:`SamplingConfig` recording the chain count, warmup,
        ``target_accept``, and seed actually used.
    """

    samples: Any
    coefficient_names: tuple[str, ...]
    mean: Any
    std: Any
    rhat: float
    ess: float
    converged: bool
    config: SamplingConfig
    _name_index: Mapping[str, int] = field(repr=False, compare=False)

    # ---- Construction ----------------------------------------------------

    @classmethod
    def from_ffi_payload(cls, payload: Mapping[str, Any]) -> "PosteriorSamples":
        """Build a :class:`PosteriorSamples` from the FFI JSON payload.

        The FFI sends ``samples_flat`` as a row-major flattened array
        plus shape metadata so we round-trip through ``numpy.reshape``
        once.  Building a nested list of lists from JSON would otherwise
        dominate decode time at biobank scale.
        """
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
        cfg_raw = payload.get("config", {})
        cfg = SamplingConfig(
            n_samples=int(cfg_raw.get("n_samples", 0)),
            n_warmup=int(cfg_raw.get("n_warmup", 0)),
            n_chains=int(cfg_raw.get("n_chains", 0)),
            target_accept=float(cfg_raw.get("target_accept", 0.0)),
            seed=int(cfg_raw.get("seed", 0)),
        )
        return cls(
            samples=samples,
            coefficient_names=names,
            mean=np.asarray(payload.get("posterior_mean", []), dtype=float),
            std=np.asarray(payload.get("posterior_std", []), dtype=float),
            rhat=float(payload["rhat"]),
            ess=float(payload["ess"]),
            converged=bool(payload["converged"]),
            config=cfg,
            _name_index={name: j for j, name in enumerate(names)},
        )

    @classmethod
    def from_ffi_json(cls, raw: str) -> "PosteriorSamples":
        return cls.from_ffi_payload(json.loads(raw))

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

    def __len__(self) -> int:
        return self.n_draws

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)

    def __getitem__(self, key: Any) -> Any:
        """Slice draws by coefficient name or numpy-style row index.

        ``post["x1"]`` returns the 1-D vector of draws for the named
        coefficient.  Every other key is forwarded to the underlying
        numpy array unchanged, so ``post[0]`` is the 0th draw (a vector
        of length ``n_coeffs``), ``post[:100]`` is the first 100 draws,
        ``post[mask]`` filters rows by a boolean mask, and so on.  For
        explicit column access by integer use ``post.samples[:, j]``.
        """
        if isinstance(key, str):
            try:
                index = self._name_index[key]
            except KeyError as exc:
                raise KeyError(
                    f"unknown coefficient {key!r}; known: {list(self.coefficient_names)}"
                ) from exc
            return self.samples[:, index]
        return self.samples[key]

    def to_numpy(self) -> Any:
        """Return the raw ``(n_draws, n_coeffs)`` numpy array."""
        return self.samples

    def to_pandas(self) -> Any:
        """Return draws as a pandas DataFrame with named coefficient columns."""
        import pandas as pd

        return pd.DataFrame(self.samples, columns=list(self.coefficient_names))

    # ---- Summary statistics ---------------------------------------------

    def interval(self, level: float = 0.95) -> Any:
        """Equal-tailed credible interval for each coefficient.

        Returns an ``(n_coeffs, 2)`` numpy array of ``(lower, upper)``
        bounds at the requested coverage ``level`` (default 95%).
        """
        import numpy as np

        if not (0.0 < level < 1.0):
            raise ValueError(f"interval level must lie in (0, 1); got {level}")
        alpha = (1.0 - float(level)) / 2.0
        if self.samples.size == 0:
            return np.empty((self.n_coeffs, 2), dtype=float)
        lo = np.quantile(self.samples, alpha, axis=0)
        hi = np.quantile(self.samples, 1.0 - alpha, axis=0)
        return np.column_stack([lo, hi])

    def summary(self, level: float = 0.95) -> Summary:
        """Per-coefficient posterior summary as a :class:`Summary`.

        The payload mirrors what ``Model.summary()`` returns: a list of
        coefficient rows (``index``, ``name``, ``estimate``, ``std_error``,
        ``ci_lower``, ``ci_upper``) plus top-level convergence diagnostics
        so the result prints/renders nicely in a notebook.
        """
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
        payload = {
            "kind": "posterior_samples",
            "n_draws": self.n_draws,
            "n_coeffs": self.n_coeffs,
            "rhat": self.rhat,
            "ess": self.ess,
            "converged": self.converged,
            "credible_interval": float(level),
            "config": self.config.to_dict(),
            "coefficients": coefficients,
        }
        return Summary.from_dict(payload)

    # ---- Reprs ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "PosteriorSamples("
            f"n_draws={self.n_draws}, "
            f"n_coeffs={self.n_coeffs}, "
            f"rhat={self.rhat:.4f}, "
            f"ess={self.ess:.1f}, "
            f"converged={self.converged}"
            ")"
        )

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


__all__ = ["PosteriorSamples", "SamplingConfig"]
