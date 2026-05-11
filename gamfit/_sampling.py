"""Posterior draws from gamfit's NUTS / Laplace sampler.

The :class:`PosteriorSamples` object returned by :meth:`gamfit.Model.sample`
is the user-facing surface for posterior reasoning.  It is intentionally
numpy-first: the raw `(n_draws, n_coeffs)` matrix is exposed as a numpy
array and every derived statistic (means, standard deviations, credible
intervals, posterior predictive draws) is also numpy.  Subscripting,
iteration, length, `.save` / `PosteriorSamples.load`, a notebook-friendly
HTML repr, and trace plots all come along for free.

`PosteriorSamples.method` discloses which sampler produced the draws:

* ``"nuts"`` — exact No-U-Turn sampling around the fitted joint mode.
* ``"laplace_gaussian"`` — iid draws from `N(beta_hat, H_penalized^-1)`,
  the Gaussian (Laplace) approximation of the posterior.  Used for
  model classes where exact NUTS is not yet wired in the engine.  The
  same surface mgcv / Wood-style GAM tooling uses for credible bands.

For models with exact NUTS support `posterior.rhat`, `posterior.ess`,
and `posterior.converged` carry meaningful values.  For Laplace draws,
the chains are iid by construction, so the engine reports
`rhat = 1.0`, `ess = n_draws`, `converged = True`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from ._summary import Summary

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SamplingConfig:
    """Echo of the NUTS configuration the engine ran with.

    All fields are populated from the FFI payload so callers can reconstruct
    exactly which sampler invocation produced the draws — useful for
    reproducibility logs and for telling whether an explicit `samples=...`
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


# Sentinel for unbound posteriors loaded from disk without a model context.
# Distinct from `b""` so callers can tell "no model attached" apart from "the
# model bytes happen to be empty".
_NO_MODEL: bytes = b""


@dataclass(frozen=True, eq=False)
class PosteriorPredictive:
    """Per-row posterior predictive draws on the linear-predictor and
    response scales.

    ``eta`` is the `(n_draws, n_rows)` matrix of draws on the link scale.
    ``mean`` is the same draws pushed through the model's inverse link
    (mean response scale).  Both are numpy float arrays.

    The ``summary(level)`` helper collapses the draws to per-row credible
    intervals so callers can plot bands without materialising further
    aggregates by hand.  The implementation walks the matrix once per
    quantile call, which is fine for typical sizes; extremely large
    grids should compute their own summaries from ``self.eta`` /
    ``self.mean`` directly.
    """

    eta: Any
    mean: Any
    family_kind: str
    model_class: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.eta.shape)  # type: ignore[return-value]

    @property
    def n_draws(self) -> int:
        return int(self.eta.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.eta.shape[1])

    def summary(self, level: float = 0.95) -> dict[str, Any]:
        """Per-row credible intervals on the link and response scale.

        Returned dict keys: ``eta_mean``, ``eta_lower``, ``eta_upper``,
        ``mean``, ``mean_lower``, ``mean_upper``.  Lower/upper are the
        symmetric tail quantiles for the requested coverage ``level``.

        Because the inverse links we support are monotone, the response
        quantiles are exactly the inverse link applied to the link
        quantiles — we leverage that here instead of taking quantiles
        of the response-scale draws separately (which would give the
        same answer up to numerical noise).
        """
        import numpy as np

        if not (0.0 < level < 1.0):
            raise ValueError(f"interval level must lie in (0, 1); got {level}")
        alpha = (1.0 - float(level)) / 2.0
        eta = np.asarray(self.eta, dtype=float)
        if eta.size == 0:
            empty = np.empty((self.n_rows,), dtype=float)
            return {
                "eta_mean": empty.copy(),
                "eta_lower": empty.copy(),
                "eta_upper": empty.copy(),
                "mean": empty.copy(),
                "mean_lower": empty.copy(),
                "mean_upper": empty.copy(),
            }
        eta_mean = eta.mean(axis=0)
        eta_lower = np.quantile(eta, alpha, axis=0)
        eta_upper = np.quantile(eta, 1.0 - alpha, axis=0)
        return {
            "eta_mean": eta_mean,
            "eta_lower": eta_lower,
            "eta_upper": eta_upper,
            "mean": _apply_inverse_link(eta_mean, self.family_kind),
            "mean_lower": _apply_inverse_link(eta_lower, self.family_kind),
            "mean_upper": _apply_inverse_link(eta_upper, self.family_kind),
        }

    def __repr__(self) -> str:
        return (
            "PosteriorPredictive("
            f"n_draws={self.n_draws}, n_rows={self.n_rows}, "
            f"family_kind={self.family_kind!r}, model_class={self.model_class!r}"
            ")"
        )


@dataclass(frozen=True, eq=False)
class PosteriorSamples:
    """Posterior draws over the model's coefficient vector.

    Attributes
    ----------
    samples:
        ``(n_draws, n_coeffs)`` numpy array of draws.  ``n_draws`` is
        ``n_chains * n_samples`` (warmup is already discarded by the
        engine).
    coefficient_names:
        Tuple of column labels for ``samples``.  Currently the FFI emits
        ``("beta_0", "beta_1", ...)``; future releases may carry the same
        names the fitted model exposes via :class:`Summary`.
    mean / std:
        Per-coefficient posterior moments returned by the sampler.
    rhat:
        Maximum split-Rhat across coefficients (exact NUTS only;
        ``1.0`` exactly for Laplace iid draws).
    ess:
        Minimum effective sample size across coefficients.
    converged:
        Boolean convenience for ``rhat < 1.1``.
    method:
        ``"nuts"`` for exact NUTS, ``"laplace_gaussian"`` for the
        Gaussian Laplace approximation around the fitted joint mode.
    model_class:
        Saved-model predictive class string the draws came from.
    family_kind:
        Inverse-link tag (``"identity"``, ``"logit"``, ``"probit"``,
        ``"cloglog"``, ``"log"``, ...).  Used by
        :meth:`PosteriorSamples.predict` to push draws through the
        correct inverse link.
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
    method: str
    model_class: str
    family_kind: str
    config: SamplingConfig
    _model_bytes: bytes = field(repr=False, compare=False, default=_NO_MODEL)
    _name_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        # The dataclass is frozen, so the only way to derive `_name_index`
        # from `coefficient_names` is via object.__setattr__. This keeps
        # construction clean for callers — they don't pass the index map
        # explicitly and they can't get it wrong.
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
        """Build a :class:`PosteriorSamples` from the FFI JSON payload.

        The FFI sends ``samples_flat`` as a row-major flattened array
        plus shape metadata so we round-trip through ``numpy.reshape``
        once.  Building a nested list of lists from JSON would otherwise
        dominate decode time for biobank-scale draws.

        ``model_bytes`` lets the caller (``Model.sample``) bundle the
        saved-model bytes so downstream methods like
        :meth:`PosteriorSamples.predict` have the model handy without
        the user passing it back manually.
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
            method=str(payload.get("method", "nuts")),
            model_class=str(payload.get("model_class", "standard")),
            family_kind=str(payload.get("family_kind", "identity")),
            config=cfg,
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
        """``True`` if the draws came from exact NUTS, ``False`` for the
        Laplace approximation."""
        return self.method == "nuts"

    def __len__(self) -> int:
        return self.n_draws

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)

    def __getitem__(self, key: Any) -> Any:
        """Slice draws by coefficient name or numpy-style row index.

        ``post["x1"]`` returns the 1-D vector of draws for the named
        coefficient.  Every other key is forwarded to the underlying
        numpy array, so ``post[0]`` is the 0th draw (a vector of length
        ``n_coeffs``), ``post[:100]`` is the first 100 draws,
        ``post[mask]`` filters rows by a boolean mask, and so on.  For
        explicit column access by integer use ``post.samples[:, j]``.
        """
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
        coefficient rows (``index``, ``name``, ``estimate``,
        ``std_error``, ``ci_lower``, ``ci_upper``) plus top-level
        convergence diagnostics and the sampler ``method`` so the result
        prints/renders nicely in a notebook.
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
        return Summary.from_dict(payload)

    # ---- Posterior predictive -------------------------------------------

    def predict(
        self,
        new_data: Any,
        *,
        chunk_size: int | None = 4096,
        level: float = 0.95,
    ) -> dict[str, Any]:
        """Posterior predictive credible bands on new data.

        Returns a dict with per-row arrays (length ``n_rows``):
        ``eta_mean``, ``eta_lower``, ``eta_upper``, ``mean``,
        ``mean_lower``, ``mean_upper``.  ``eta`` is on the linear-
        predictor scale, ``mean`` on the response scale (inverse-link
        applied).

        This walks chunks of rows through ``draws @ X.T`` and reduces
        each chunk to quantiles immediately, so memory stays bounded
        at roughly ``n_draws * chunk_size`` floats regardless of the
        prediction-set size.  Set ``chunk_size=None`` to disable
        chunking and materialise the full ``(n_draws, n_rows)``
        matrix (use :meth:`predict_draws` for that explicit form).

        Currently supports standard non-link-wiggle GAM models; other
        classes raise with a pointer to ``Model.predict(interval=...)``.
        For Laplace-method posteriors the returned bands are exactly
        the surface that ``model.predict(new_data, interval=level)``
        produces analytically — different code path, same answer up to
        Monte Carlo error.
        """
        x = self._design_matrix(new_data)
        return _posterior_predict_bands(
            samples=self.samples,
            x=x,
            family_kind=self.family_kind,
            level=level,
            chunk_size=chunk_size,
        )

    def predict_draws(self, new_data: Any) -> PosteriorPredictive:
        """Full posterior predictive draws on new data.

        Returns a :class:`PosteriorPredictive` whose ``.eta`` and
        ``.mean`` matrices are ``(n_draws, n_rows)``.  Materialises the
        full matrix — for very large prediction sets prefer
        :meth:`predict` which streams per-row credible bands.
        """
        import numpy as np

        x = self._design_matrix(new_data)
        eta = self.samples @ x.T
        mean = _apply_inverse_link(eta, self.family_kind)
        return PosteriorPredictive(
            eta=np.asarray(eta, dtype=float),
            mean=np.asarray(mean, dtype=float),
            family_kind=self.family_kind,
            model_class=self.model_class,
        )

    def _design_matrix(self, new_data: Any) -> Any:
        """Fetch the saved-model design matrix on ``new_data`` via the FFI.

        Raises ``RuntimeError`` if this :class:`PosteriorSamples` was
        loaded from disk without a model context (``_model_bytes`` is
        empty), and a clear FFI error if the saved model class doesn't
        yet support a closed-form design (link-wiggle, survival, etc.).
        """
        if not self._model_bytes:
            raise RuntimeError(
                "PosteriorSamples has no model context; predict requires "
                "the original Model. Re-sample via Model.sample(...) or "
                "use Model.predict(...) directly."
            )
        from ._binding import rust_module
        from ._exceptions import map_exception
        from ._tables import normalize_table

        headers, rows, _ = normalize_table(new_data)
        try:
            raw = rust_module().design_matrix_table(self._model_bytes, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        parsed = json.loads(raw)
        return _decode_design_matrix(parsed)

    # ---- Persistence ----------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Save the posterior to an ``.npz`` archive.

        The archive carries the full ``(n_draws, n_coeffs)`` samples
        matrix, the per-coefficient mean / std, convergence diagnostics,
        method / class / family tags, the sampling config, and the saved
        model bytes (so :meth:`predict` continues to work after a
        load).  Roundtrip via :meth:`PosteriorSamples.load`.
        """
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
        """Load a :class:`PosteriorSamples` from an ``.npz`` written by
        :meth:`save`."""
        import numpy as np

        # `allow_pickle=True` is required because we ship the JSON
        # metadata as a 0-d object array (the simplest way to keep
        # arbitrary structure inside a .npz). Inputs are produced by
        # `save`, never by untrusted callers.
        npz = np.load(Path(path), allow_pickle=True)
        metadata = json.loads(str(npz["metadata"].item()))
        cfg_raw = metadata.get("config", {})
        cfg = SamplingConfig(
            n_samples=int(cfg_raw.get("n_samples", 0)),
            n_warmup=int(cfg_raw.get("n_warmup", 0)),
            n_chains=int(cfg_raw.get("n_chains", 0)),
            target_accept=float(cfg_raw.get("target_accept", 0.0)),
            seed=int(cfg_raw.get("seed", 0)),
        )
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
            config=cfg,
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
        """Matplotlib trace + marginal-density plot.

        ``coefficients`` may be ``None`` (auto-select the first
        ``max_panels`` coefficients), a single name/index, or an
        iterable of names/indices.  Returns the matplotlib Figure.

        Each row of the figure has two panels: the trace (draws vs
        iteration index) on the left, the marginal density / histogram
        on the right.  When ``ax`` is supplied it's expected to be a
        2-D array of axes with shape ``(n_panels, 2)``; otherwise we
        create a fresh figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        selected: list[str | int]
        if coefficients is None:
            count = min(max_panels, self.n_coeffs)
            selected = list(range(count))
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
            axes_arr[row, 1].hist(
                draws,
                bins=40,
                density=True,
                color="#1d4ed8",
                alpha=0.7,
            )
            axes_arr[row, 1].set_title(f"{label} — density")
            axes_arr[row, 1].set_xlabel("value")

        fig.tight_layout()
        return fig

    # ---- Reprs ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "PosteriorSamples("
            f"n_draws={self.n_draws}, "
            f"n_coeffs={self.n_coeffs}, "
            f"method={self.method!r}, "
            f"rhat={self.rhat:.4f}, "
            f"ess={self.ess:.1f}, "
            f"converged={self.converged}"
            ")"
        )

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


# ---- Free helpers --------------------------------------------------------


def _decode_design_matrix(parsed: Mapping[str, Any]) -> Any:
    import numpy as np

    n_rows = int(parsed["n_rows"])
    n_cols = int(parsed["n_cols"])
    flat = np.asarray(parsed.get("x_flat", []), dtype=float)
    if flat.size != n_rows * n_cols:
        raise ValueError(
            "design matrix FFI payload shape mismatch: "
            f"got {flat.size} floats, expected {n_rows} * {n_cols}"
        )
    return flat.reshape(n_rows, n_cols)


def _apply_inverse_link(eta: Any, family_kind: str) -> Any:
    """Apply the inverse link element-wise.

    ``family_kind`` is a tag emitted by the Rust FFI (``identity``,
    ``logit``, ``probit``, ``cloglog``, ``log``, ...).  We support the
    closed-form scalar links here; anything else raises with a clear
    "use Model.predict" pointer instead of silently returning eta.
    """
    import numpy as np

    eta_arr = np.asarray(eta, dtype=float)
    kind = family_kind.strip().lower()
    if kind in ("identity", ""):
        return eta_arr
    if kind == "logit":
        # 1 / (1 + exp(-eta)); written in the numerically-stable form to
        # avoid overflow on large negative eta.
        return np.where(
            eta_arr >= 0.0,
            1.0 / (1.0 + np.exp(-eta_arr)),
            np.exp(eta_arr) / (1.0 + np.exp(eta_arr)),
        )
    if kind == "probit":
        from math import erf, sqrt

        # numpy doesn't ship Phi directly; the erf-based form is fine
        # at f64 precision over the range NUTS actually visits.
        vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))
        return vec(eta_arr)
    if kind == "cloglog":
        # 1 - exp(-exp(eta)); clip eta to avoid overflow of exp(exp(...))
        # at the extreme tails which the NUTS sampler does not realistically
        # reach but a corrupt posterior could.
        return 1.0 - np.exp(-np.exp(np.clip(eta_arr, -50.0, 50.0)))
    if kind == "log":
        return np.exp(eta_arr)
    raise NotImplementedError(
        f"posterior predictive on response scale is not yet wired for "
        f"family_kind={family_kind!r}; access posterior.predict_draws(...).eta "
        f"for link-scale draws or use model.predict(new_data, interval=...) "
        f"for class-specific bands."
    )


def _posterior_predict_bands(
    samples: Any,
    x: Any,
    family_kind: str,
    level: float,
    chunk_size: int | None,
) -> dict[str, Any]:
    """Per-row posterior predictive credible bands, optionally chunked.

    Walks ``X[start:stop, :]`` chunks through ``samples @ X_chunk.T`` and
    immediately collapses each ``(n_draws, chunk_rows)`` block to per-row
    mean and quantiles.  Memory stays bounded at roughly
    ``n_draws * chunk_size`` floats regardless of total ``n_rows``.
    """
    import numpy as np

    if not (0.0 < level < 1.0):
        raise ValueError(f"interval level must lie in (0, 1); got {level}")
    alpha = (1.0 - float(level)) / 2.0
    x_arr = np.asarray(x, dtype=float)
    n_rows = int(x_arr.shape[0])

    if chunk_size is None or chunk_size >= n_rows:
        eta = np.asarray(samples) @ x_arr.T
        eta_mean = eta.mean(axis=0)
        eta_lower = np.quantile(eta, alpha, axis=0)
        eta_upper = np.quantile(eta, 1.0 - alpha, axis=0)
    else:
        eta_mean = np.empty(n_rows, dtype=float)
        eta_lower = np.empty(n_rows, dtype=float)
        eta_upper = np.empty(n_rows, dtype=float)
        samples_arr = np.asarray(samples)
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            eta_chunk = samples_arr @ x_arr[start:stop, :].T
            eta_mean[start:stop] = eta_chunk.mean(axis=0)
            eta_lower[start:stop] = np.quantile(eta_chunk, alpha, axis=0)
            eta_upper[start:stop] = np.quantile(eta_chunk, 1.0 - alpha, axis=0)

    return {
        "eta_mean": eta_mean,
        "eta_lower": eta_lower,
        "eta_upper": eta_upper,
        "mean": _apply_inverse_link(eta_mean, family_kind),
        "mean_lower": _apply_inverse_link(eta_lower, family_kind),
        "mean_upper": _apply_inverse_link(eta_upper, family_kind),
    }


__all__ = ["PosteriorPredictive", "PosteriorSamples", "SamplingConfig"]
