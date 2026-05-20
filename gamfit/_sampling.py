"""Posterior draws from gamfit's NUTS / Laplace sampler.

The :class:`PosteriorSamples` object returned by :meth:`gamfit.Model.sample`
is the user-facing surface for posterior reasoning.  It is intentionally
numpy-first: the raw `(n_draws, n_coeffs)` matrix is exposed as a numpy
array and every derived statistic (means, standard deviations, credible
intervals, fitted-mean draws) is also numpy.  Subscripting,
iteration, length, `.save` / `PosteriorSamples.load`, a notebook-friendly
HTML repr, and trace plots all come along for free.

`PosteriorSamples.method` discloses which sampler produced the draws:

* ``"nuts"`` — exact No-U-Turn sampling around the fitted joint mode.
* ``"laplace"`` — iid draws from `N(beta_hat, H_penalized^-1)`,
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
    reproducibility logs and for telling whether an explicit ``samples=...``
    request was honored or auto-derived from the model dimension.

    Attributes
    ----------
    n_samples : int
        Post-warmup draws kept per chain.
    n_warmup : int
        Warmup draws discarded per chain before collecting ``n_samples``.
    n_chains : int
        Number of independent NUTS chains run by the engine.
    target_accept : float
        Step-size adaptation target acceptance probability in ``(0, 1)``.
    seed : int
        RNG seed actually consumed by the sampler.

    Examples
    --------
    >>> post = model.sample(samples=500)
    >>> post.config.n_samples
    500
    >>> post.config.target_accept
    0.95
    """

    n_samples: int
    n_warmup: int
    n_chains: int
    target_accept: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain JSON-serialisable ``dict``.

        Returns
        -------
        dict[str, Any]
            Mapping with keys ``n_samples``, ``n_warmup``, ``n_chains``,
            ``target_accept``, ``seed``.

        Examples
        --------
        >>> cfg = SamplingConfig(500, 1000, 4, 0.95, 42)
        >>> cfg.to_dict()["n_chains"]
        4
        """
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
    """Per-row posterior fitted-mean draws on the link and response scales.

    Returned by :meth:`PosteriorSamples.predict_draws`, this container
    holds the full ``(n_draws, n_rows)`` matrices of fitted-mean draws
    on both the linear-predictor (``eta``) and response (``mean``)
    scales, along with link/class metadata used to re-apply the inverse
    link on demand.

    Attributes
    ----------
    eta : numpy.ndarray
        ``(n_draws, n_rows)`` float matrix of draws on the link scale.
    mean : numpy.ndarray
        ``(n_draws, n_rows)`` float matrix of draws pushed through the
        model's inverse link (mean response scale).
    family_kind : str
        Inverse-link tag emitted by the engine (``"identity"``,
        ``"logit"``, ``"probit"``, ``"cloglog"``, ``"log"``, ...).
    model_class : str
        Saved-model predictive class string the underlying
        :class:`PosteriorSamples` came from.

    Notes
    -----
    Use :meth:`summary` to collapse the matrices to per-row credible
    bands without writing the quantile reductions yourself.  For very
    large prediction sets, prefer :meth:`PosteriorSamples.predict`
    which streams chunk-by-chunk instead of materialising the full
    ``(n_draws, n_rows)`` matrix here.

    Examples
    --------
    >>> pp = post.predict_draws(new_data)
    >>> pp.shape
    (1000, 50)
    >>> bands = pp.summary(level=0.9)
    """

    eta: Any
    mean: Any
    family_kind: str
    model_class: str

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the link-scale draw matrix.

        Returns
        -------
        tuple[int, int]
            ``(n_draws, n_rows)``.

        Examples
        --------
        >>> pp = post.predict_draws(new_data)
        >>> pp.shape
        (1000, 50)
        """
        return tuple(self.eta.shape)

    @property
    def n_draws(self) -> int:
        """Number of posterior fitted-mean draws.

        Returns
        -------
        int
            Length of the leading axis of :attr:`eta`.

        Examples
        --------
        >>> pp = post.predict_draws(new_data)
        >>> pp.n_draws
        1000
        """
        return int(self.eta.shape[0])

    @property
    def n_rows(self) -> int:
        """Number of prediction rows.

        Returns
        -------
        int
            Length of the trailing axis of :attr:`eta`.

        Examples
        --------
        >>> pp = post.predict_draws(new_data)
        >>> pp.n_rows
        50
        """
        return int(self.eta.shape[1])

    def summary(self, level: float = 0.95) -> dict[str, Any]:
        """Collapse fitted-mean draws to per-row credible bands.

        Parameters
        ----------
        level : float, optional
            Coverage probability of the equal-tailed credible interval
            in ``(0, 1)``.  Default ``0.95``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dict with six length-``n_rows`` arrays: ``eta_mean``,
            ``eta_lower``, ``eta_upper`` (link scale) and ``mean``,
            ``mean_lower``, ``mean_upper`` (response scale).

        Notes
        -----
        Because the supported inverse links are monotone, response-scale
        quantiles are computed as the inverse link applied to the link
        quantiles rather than as quantiles of :attr:`mean` directly —
        the two agree up to numerical noise and the link-quantile form
        avoids re-walking the response-scale matrix.

        Examples
        --------
        >>> pp = post.predict_draws(new_data)
        >>> bands = pp.summary(level=0.9)
        >>> bands["mean_lower"].shape
        (50,)
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

    Returned by :meth:`gamfit.Model.sample`. This is the user-facing
    surface for posterior reasoning: a numpy-first container with
    named-column subscripting, credible-interval helpers, posterior
    predictive utilities, ``.save`` / :meth:`load` round-trip, trace
    plotting, a concise :meth:`__repr__`, and a notebook-friendly
    rich-HTML representation (``_repr_html_``) that delegates to
    :meth:`summary`.

    Attributes
    ----------
    samples : numpy.ndarray
        ``(n_draws, n_coeffs)`` numpy array of draws.  ``n_draws`` is
        ``n_chains * n_samples`` (warmup is already discarded by the
        engine).
    coefficient_names : tuple[str, ...]
        Column labels for ``samples``.  Currently the FFI emits
        ``("beta_0", "beta_1", ...)``; future releases may carry the
        same names the fitted model exposes via :class:`Summary`.
    mean : numpy.ndarray
        Per-coefficient posterior mean reported by the sampler.
    std : numpy.ndarray
        Per-coefficient posterior standard deviation reported by the
        sampler.
    rhat : float
        Maximum split-Rhat across coefficients (exact NUTS only;
        ``1.0`` exactly for Laplace iid draws).
    ess : float
        Minimum effective sample size across coefficients.
    converged : bool
        Boolean convenience for ``rhat < 1.1``.
    method : str
        ``"nuts"`` for exact NUTS, ``"laplace"`` for the
        Gaussian Laplace approximation around the fitted joint mode.
    model_class : str
        Saved-model predictive class string the draws came from.
    family_kind : str
        Inverse-link tag (``"identity"``, ``"logit"``, ``"probit"``,
        ``"cloglog"``, ``"log"``, ...).  Used by :meth:`predict` to
        push draws through the correct inverse link.
    config : SamplingConfig
        :class:`SamplingConfig` recording the chain count, warmup,
        ``target_accept``, and seed actually used.

    Examples
    --------
    >>> post = model.sample(samples=1000, warmup=1000, chains=4)
    >>> post.n_draws, post.n_coeffs
    (4000, 12)
    >>> post["x1"].mean()
    0.342
    >>> bands = post.predict(new_data, level=0.9)
    >>> post.save("posterior.npz")
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
        """Internal factory: build a :class:`PosteriorSamples` from the FFI payload.

        Used by :meth:`gamfit.Model.sample` to wrap the dict produced by
        the Rust sampler. End users should not call this directly.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Decoded FFI JSON payload. Must contain ``n_draws``,
            ``n_coeffs``, ``samples_flat`` (row-major), ``rhat``,
            ``ess``, ``converged`` and may contain
            ``coefficient_names``, ``posterior_mean``,
            ``posterior_std``, ``method``, ``model_class``,
            ``family_kind``, and ``config``.
        model_bytes : bytes, optional
            Saved-model byte blob to bundle so downstream methods like
            :meth:`predict` work without the user re-passing the model.

        Returns
        -------
        PosteriorSamples
            Reified posterior with samples reshaped to
            ``(n_draws, n_coeffs)``.

        Notes
        -----
        ``samples_flat`` is sent flat (row-major) so we round-trip
        through ``numpy.reshape`` once. Building a nested list of
        lists from JSON would otherwise dominate decode time for
        biobank-scale draws.
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
        """Internal factory: build a :class:`PosteriorSamples` from a raw FFI JSON string.

        Thin convenience around :meth:`from_ffi_payload` that decodes
        the JSON itself. Used by :meth:`gamfit.Model.sample`; not
        intended as a public API.

        Parameters
        ----------
        raw : str
            JSON-encoded FFI payload from the Rust sampler.
        model_bytes : bytes, optional
            Saved-model byte blob bundled into the returned object.

        Returns
        -------
        PosteriorSamples
            Same as :meth:`from_ffi_payload`.
        """
        return cls.from_ffi_payload(json.loads(raw), model_bytes=model_bytes)

    # ---- Convenience accessors ------------------------------------------

    @property
    def n_draws(self) -> int:
        """Total number of post-warmup draws across all chains.

        Returns
        -------
        int
            ``n_chains * n_samples``; the leading axis length of
            :attr:`samples`.

        Examples
        --------
        >>> post.n_draws
        4000
        """
        return int(self.samples.shape[0])

    @property
    def n_coeffs(self) -> int:
        """Number of model coefficients (columns of :attr:`samples`).

        Returns
        -------
        int
            Trailing axis length of :attr:`samples`.

        Examples
        --------
        >>> post.n_coeffs
        12
        """
        return int(self.samples.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying draws matrix.

        Returns
        -------
        tuple[int, int]
            ``(n_draws, n_coeffs)``.

        Examples
        --------
        >>> post.shape
        (4000, 12)
        """
        return (self.n_draws, self.n_coeffs)

    @property
    def is_exact(self) -> bool:
        """Whether the draws are exact NUTS rather than Laplace iid.

        Returns
        -------
        bool
            ``True`` if :attr:`method` is ``"nuts"``, ``False`` for
            ``"laplace"`` (the Gaussian Laplace approximation).

        Examples
        --------
        >>> post = model.sample(samples=1000)
        >>> post.is_exact
        True
        """
        return self.method == "nuts"

    def __len__(self) -> int:
        return self.n_draws

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)

    def __getitem__(self, key: Any) -> Any:
        """Slice draws by coefficient name or numpy-style row index.

        Parameters
        ----------
        key : str or numpy index
            If a ``str``, looked up against :attr:`coefficient_names`
            and the corresponding column of :attr:`samples` is
            returned.  Any other key (``int``, ``slice``, boolean
            mask, fancy index, ...) is forwarded to the underlying
            numpy array.

        Returns
        -------
        numpy.ndarray
            ``(n_draws,)`` column for a string key; otherwise whatever
            ``samples[key]`` returns (typically a row or sub-block).

        Raises
        ------
        KeyError
            If a string key does not match any name in
            :attr:`coefficient_names`.

        Examples
        --------
        >>> post["beta_1"].shape
        (4000,)
        >>> post[0].shape
        (12,)
        >>> post[:100].shape
        (100, 12)
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
        """Return the raw draws as a numpy array.

        Returns
        -------
        numpy.ndarray
            ``(n_draws, n_coeffs)`` view of :attr:`samples` (not a
            copy).

        Examples
        --------
        >>> arr = post.to_numpy()
        >>> arr.shape
        (4000, 12)
        """
        return self.samples

    def to_pandas(self) -> Any:
        """Return draws as a pandas DataFrame with named coefficient columns.

        Returns
        -------
        pandas.DataFrame
            ``(n_draws, n_coeffs)`` DataFrame whose columns are
            :attr:`coefficient_names`.

        Examples
        --------
        >>> df = post.to_pandas()
        >>> df.columns.tolist()[:2]
        ['beta_0', 'beta_1']
        >>> df["beta_1"].mean()
        0.342
        """
        import pandas as pd

        return pd.DataFrame(self.samples, columns=list(self.coefficient_names))

    # ---- Summary statistics ---------------------------------------------

    def interval(self, level: float = 0.95) -> Any:
        """Equal-tailed credible interval for each coefficient.

        Parameters
        ----------
        level : float, optional
            Coverage probability in ``(0, 1)``. Default ``0.95``.

        Returns
        -------
        numpy.ndarray
            ``(n_coeffs, 2)`` array of ``(lower, upper)`` bounds at
            the requested coverage.

        Raises
        ------
        ValueError
            If ``level`` is not strictly between 0 and 1.

        Examples
        --------
        >>> ci = post.interval(level=0.9)
        >>> ci.shape
        (12, 2)
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

        Parameters
        ----------
        level : float, optional
            Coverage probability for the credible interval columns,
            in ``(0, 1)``. Default ``0.95``.

        Returns
        -------
        Summary
            Coefficient rows (``index``, ``name``, ``estimate``,
            ``std_error``, ``ci_lower``, ``ci_upper``) plus top-level
            convergence diagnostics (``rhat``, ``ess``, ``converged``),
            sampler ``method``, and the :class:`SamplingConfig` echo.
            Renders nicely in a notebook via :class:`Summary` HTML.

        Notes
        -----
        The payload mirrors what :meth:`gamfit.Model.summary` returns
        for fitted models, so downstream rendering helpers work
        uniformly on both fitted and sampled views.

        Examples
        --------
        >>> post.summary(level=0.95)
        Summary(method='nuts', n_coeffs=12, rhat=1.0021, converged=True)
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
        """Posterior credible bands for eta and E[y | x] on new data.

        Parameters
        ----------
        new_data : Any
            Tabular new data (DataFrame, dict of columns, or any
            object accepted by the engine's table normaliser) at which
            to evaluate the posterior fitted means.
        chunk_size : int or None, optional
            Number of prediction rows processed at once. Default
            ``4096``. Pass ``None`` to disable chunking and form the
            full ``(n_draws, n_rows)`` matrix (consider
            :meth:`predict_draws` instead in that case).
        level : float, optional
            Coverage probability for the credible bands in ``(0, 1)``.
            Default ``0.95``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Six length-``n_rows`` arrays: ``eta_mean``, ``eta_lower``,
            ``eta_upper`` (link scale) and ``mean``, ``mean_lower``,
            ``mean_upper`` (response scale, inverse link applied).

        Raises
        ------
        RuntimeError
            If this :class:`PosteriorSamples` was loaded from disk
            without a model context.
        NotImplementedError
            For model classes lacking a closed-form design matrix
            (e.g. link-wiggle, survival) — use
            :meth:`gamfit.Model.predict` instead.

        Notes
        -----
        Walks chunks of rows through ``draws @ X.T`` and reduces each
        chunk to quantiles immediately, so memory stays bounded at
        roughly ``n_draws * chunk_size`` floats regardless of the
        prediction-set size. For Laplace-method posteriors the
        returned bands match what
        ``model.predict(new_data, interval=level)`` produces
        analytically, up to Monte Carlo error.

        Examples
        --------
        >>> bands = post.predict(new_data, level=0.9)
        >>> bands["mean_lower"].shape
        (50,)
        >>> bands["mean_upper"][0]
        0.812
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
        """Full posterior fitted-mean draws on new data.

        Parameters
        ----------
        new_data : Any
            Tabular new data (DataFrame, dict of columns, or any
            object accepted by the engine's table normaliser).

        Returns
        -------
        PosteriorPredictive
            Container whose :attr:`PosteriorPredictive.eta` and
            :attr:`PosteriorPredictive.mean` are
            ``(n_draws, n_rows)`` matrices on the link and response
            scales respectively.

        Raises
        ------
        RuntimeError
            If this :class:`PosteriorSamples` was loaded from disk
            without a model context.

        Notes
        -----
        Materialises the full ``(n_draws, n_rows)`` matrix in memory.
        For very large prediction sets prefer :meth:`predict`, which
        streams per-row credible bands chunk-by-chunk.

        Examples
        --------
        >>> pp = post.predict_draws(new_data)
        >>> pp.shape
        (4000, 50)
        >>> pp.mean.std(axis=0).mean()
        0.087
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

        Parameters
        ----------
        path : str or pathlib.Path
            Destination ``.npz`` file path.

        Returns
        -------
        str
            String form of the resolved output path.

        Notes
        -----
        The archive carries the full ``(n_draws, n_coeffs)`` samples
        matrix, the per-coefficient mean and std, convergence
        diagnostics, method / class / family tags, the
        :class:`SamplingConfig`, and the saved model bytes (so
        :meth:`predict` continues to work after a round-trip via
        :meth:`load`).

        Examples
        --------
        >>> post.save("posterior.npz")
        'posterior.npz'
        >>> reloaded = PosteriorSamples.load("posterior.npz")
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
        """Load a :class:`PosteriorSamples` from an ``.npz`` archive.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to an archive previously written by :meth:`save`.

        Returns
        -------
        PosteriorSamples
            Reconstructed posterior, including bundled model bytes so
            :meth:`predict` keeps working.

        Notes
        -----
        The archive uses ``allow_pickle=True`` to round-trip the JSON
        metadata stored as a 0-d object array; only load archives you
        produced via :meth:`save`.

        Examples
        --------
        >>> post.save("posterior.npz")
        'posterior.npz'
        >>> reloaded = PosteriorSamples.load("posterior.npz")
        >>> reloaded.n_draws == post.n_draws
        True
        """
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

        Parameters
        ----------
        coefficients : None, str, int, or iterable of str/int, optional
            Coefficients to plot. If ``None``, auto-selects the first
            ``max_panels`` coefficients. A single name or integer
            index plots one panel row; an iterable plots one row per
            element.
        max_panels : int, optional
            Cap on the number of panel rows when ``coefficients`` is
            ``None``. Default ``8``.
        ax : numpy.ndarray of matplotlib Axes, optional
            Pre-existing 2-D axes array of shape ``(n_panels, 2)``. If
            ``None``, a fresh ``(n_panels, 2)`` figure is created.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the trace and density panels.

        Raises
        ------
        ValueError
            If the resolved coefficient selection is empty.

        Notes
        -----
        Each row has two panels: trace (draws vs iteration index) on
        the left and a marginal density histogram on the right.

        Examples
        --------
        >>> fig = post.plot_trace()
        >>> fig = post.plot_trace(coefficients=["beta_0", "beta_1"])
        >>> fig.savefig("trace.png")
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
        f"posterior fitted-mean draws on response scale are not wired for "
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
    """Per-row posterior credible bands for eta and E[y | x], optionally chunked.

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
