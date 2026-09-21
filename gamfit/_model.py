"""Public :class:`Model` shell.

The numeric work all lives in the Rust core: this module marshals
arguments through the FFI, hands payloads off to ``_survival`` /
``_diagnose_plot`` helpers, and exposes Pythonic properties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence, cast, overload

import numpy as np
from numpy.typing import NDArray

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._partial_effect import PartialEffect
from ._sampling import PosteriorSamples
from ._schema import SchemaCheck
from ._summary import Summary
from ._predict_shape import shape_predict_response
from ._survival import (
    CompetingRisksCIF,
    CompetingRisksPrediction,
    SurvivalPrediction,
    TermBlock,
    competing_risks_cif,
    extract_row_ids,
    term_blocks_for_model,
)
from ._tables import (
    detect_table_kind,
    normalize_table,
    numpy_table_width,
    restore_output_table,
    table_columns,
)


AffineCoefficientFrame = Literal["full", "link_wiggle_joint"]


@dataclass(frozen=True, slots=True)
class AffineDesign:
    """Exact fitted affine predictor returned by :meth:`Model.design_matrix`.

    ``offset + matrix @ coefficients`` reproduces the fitted linear predictor.
    ``coefficient_frame`` names the coordinate system containing those exact
    coefficients; ``coefficient_slice`` is the represented half-open slice in
    that frame. The three covariance fields use that exact same frame. Each
    covariance definition remains separately optional: an unavailable
    smoothing-corrected covariance is never replaced by a conditional one.

    ``eta_gradient`` is ``d eta / d coefficients`` in that same frame, and it --
    not ``matrix`` -- is what pairs with the covariances::

        eta_variance = np.einsum("ij,jk,ik->i", a.eta_gradient, V, a.eta_gradient)

    The two are the SAME array whenever the fitted predictor is linear in its
    coefficients (``coefficient_frame == "full"``). They differ for a
    ``link_wiggle_joint`` fit: the warp basis is evaluated at an index that
    itself moves with the mean coefficients, so ``matrix`` reproduces the
    fitted ``eta`` exactly while ``eta_gradient`` carries the extra warp slope
    on the mean block. Using ``matrix`` for variance there would silently
    disagree with the standard errors :meth:`Model.predict` reports.
    """

    offset: NDArray[np.float64]
    matrix: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    coefficient_frame: AffineCoefficientFrame
    coefficient_start: int
    coefficient_stop: int
    covariance_conditional: NDArray[np.float64] | None
    covariance_smoothing_corrected: NDArray[np.float64] | None
    covariance_frequentist: NDArray[np.float64] | None
    eta_gradient: NDArray[np.float64]

    @property
    def coefficient_slice(self) -> slice:
        """Half-open slice represented inside :attr:`coefficient_frame`."""
        return slice(self.coefficient_start, self.coefficient_stop)


def _affine_design_from_payload(payload: Any) -> AffineDesign:
    """Shape the Rust-owned affine payload without duplicating model math."""
    return AffineDesign(
        offset=payload["offset"],
        matrix=payload["matrix"],
        coefficients=payload["coefficients"],
        coefficient_frame=cast(AffineCoefficientFrame, payload["coefficient_frame"]),
        coefficient_start=int(payload["coefficient_start"]),
        coefficient_stop=int(payload["coefficient_stop"]),
        covariance_conditional=payload["covariance_conditional"],
        covariance_smoothing_corrected=payload[
            "covariance_smoothing_corrected"
        ],
        covariance_frequentist=payload["covariance_frequentist"],
        eta_gradient=payload["eta_gradient"],
    )


class Model:
    """Fitted scalar GAM/GLM model shell.

    Instances are returned by :func:`gamfit.fit`, :func:`gamfit.fit_array`,
    and :func:`gamfit.loads` for scalar-response fits. The serialized Rust
    model payload is the source of truth; Python methods marshal table/array
    inputs into the Rust extension and shape the returned predictions,
    summaries, diagnostics, samples, and deployment extensions.

    Use :meth:`predict` for named table inputs, :meth:`predict_array` only for
    models fitted from positional arrays, :meth:`summary` for typed fit
    metadata, and :meth:`save` / :meth:`dumps` for persistence. ``pickle``,
    ``copy`` and ``joblib`` go through the same :meth:`dumps` bytes.
    """

    __slots__ = ("_model_bytes", "_prediction_model", "_training_table_kind")

    def __init__(
        self, *, _model_bytes: bytes, _training_table_kind: str | None = None
    ) -> None:
        self._model_bytes = _model_bytes
        try:
            self._prediction_model = rust_module().compile_model(_model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        # allow-list (a): a reloaded model reads its training table kind from
        # the compiled payload instead of re-parsing the saved bytes.
        self._training_table_kind = (
            self._prediction_model.training_table_kind
            if _training_table_kind is None
            else _training_table_kind
        )

    def __reduce__(self) -> tuple[Any, tuple[bytes]]:
        """Pickle (and ``copy.copy`` / ``copy.deepcopy``) through the saved-model
        bytes, rebuilt by :func:`gamfit.loads`: the archive :meth:`dumps` returns
        is the only serialized form, and the compiled prediction handle is
        rebuilt from it rather than pickled."""
        from ._api import loads  # local import avoids cycle

        return (loads, (self._model_bytes,))

    def predict(
        self,
        data: Any,
        *,
        interval: float | Literal["conformal"] | None = None,
        conformal_level: float = 0.9,
        calibration: Any | None = None,
        training_data: Any | None = None,
        covariance_mode: str | None = None,
        observation_interval: bool = False,
        return_type: str | None = None,
        id_column: str | None = None,
    ) -> Any:
        """Predict from new ``data``.

        Parameters
        ----------
        data : table-like
            Input rows in any format accepted by :func:`gamfit.fit`
            (``pandas.DataFrame``, ``pyarrow.Table``, ``polars.DataFrame``,
            ``dict`` of columns, ``list`` of record dicts, ...). Columns must
            cover every predictor referenced by the fitted formula. A 2-D
            NumPy array is positional: a model fitted from an array reads its
            columns as ``x0, x1, ...``; a model fitted from a named table binds
            them to its predictor columns in training-table order, and needs
            exactly that many columns.
        interval : float, "conformal", or None, default None
            Single uncertainty knob. ``None`` returns the point prediction(s)
            only. A float in ``(0, 1)`` (e.g. ``0.95``) requests the full
            uncertainty decomposition at that pointwise coverage; the output
            gains ``linear_predictor_standard_error`` (the posterior SD of η),
            ``posterior_mean_standard_error`` (the posterior SD of the
            response, from the same η integral as ``posterior_mean``),
            ``posterior_mean_lower``, and ``posterior_mean_upper`` columns
            alongside ``linear_predictor_plugin`` / ``mean_plugin`` /
            ``posterior_mean``. On survival models it
            also produces per-cell hazard / survival SEs. Issue #342
            collapsed the previous overlapping ``with_uncertainty`` boolean
            into this single flag (use ``interval=0.95`` for the SE-only
            case).

            Pass ``interval="conformal"`` for a distribution-free conformal
            band at ``conformal_level`` coverage in ``posterior_mean_lower`` /
            ``posterior_mean_upper`` — the same routes as ``gam predict
            --conformal``. Exactly one of ``training_data`` or ``calibration``
            is required. With ``training_data`` it is the full-conformal set
            built on the labeled rows plus the candidate test row: every
            labeled row is used for both fitting and calibration. A
            Gaussian-identity model gets the set of the fit that re-selects the
            smoothing strength by REML on the augmented rows (#942 Layer 3), so
            the finite-sample ``conformal_level`` coverage theorem holds; it
            costs one Cholesky per test point plus a cold REML refit at each
            finite endpoint. A Bernoulli-logit model with one selected strength
            re-selects it by augmented LAML once per label, using a canonical
            start and the augmented design's resolvability domain. A failed
            selection raises an error. Other Bernoulli-logit models (a subset of
            ``{0, 1}``), Poisson-log and negative-binomial-log (candidates
            enumerated up to a data-derived tail beyond which none can conform;
            NB theta frozen at its fitted value) and Gamma-log (Pearson score,
            so the set is a band in ``y / mu``) refit the augmented penalized
            likelihood per candidate at the frozen penalty. Discrete ties are
            broken by one independent uniform per prediction row. Numerical
            uncertainty is enclosed conservatively. Offsets are honoured. A model
            fitted with prior
            weights raises ``InvalidConfigurationError``: the candidate point
            has no weight, so use ``calibration=`` (split conformal) instead.
            The saved model carries only the ``p x p`` frozen penalty and its
            smoothing-parameter count, never per-row training data, so the
            labeled rows are passed again here. The per-row
            ``conformal_certificate`` output column is 0 (exact_frozen: nothing
            to re-select), 1 (honest_refit), or 2 (conservative_frozen: GLM
            numerical enclosure) where the guarantee holds; a
            negative code is a typed refusal where the row carries the
            frozen-penalty set with no finite-sample guarantee for the
            selection step (several smoothing parameters, a payload without the
            count, a degenerate criterion, or ``-7`` glm_frozen_penalty for a
            count/Gamma fit that selected a smoothing parameter or NB theta).
            Coverage is marginal under exchangeability of the supplied rows
            and a fixed symmetric design/penalty construction, not conditional
            on features. A training-only learned basis need not satisfy this.
            The set is a union of ``conformal_set_components`` intervals and
            the bounds report its outer envelope (NaN for an empty randomized
            set). ``conformal_lower_closed`` and ``conformal_upper_closed`` are
            1 when the corresponding finite endpoint is included and 0 otherwise;
            both are 0 for an empty set. With ``calibration`` it is the
            split-conformal band ``mu_hat(x) +/- q_hat * s(x)`` calibrated on
            that held-out fold, with finite-sample marginal coverage
            ``>= conformal_level`` regardless of model misspecification, for
            any standard GAM family.
        conformal_level : float, default 0.9
            Target marginal coverage in ``(0, 1)`` when ``interval="conformal"``
            (e.g. ``0.95`` for a 95% interval). Ignored when ``interval`` is a
            float or ``None``.
        calibration : table-like, optional
            Held-out *labeled* calibration fold (not used in fitting) for the
            split-conformal band; ``interval="conformal"`` only. It must contain
            the response column in addition to the predictors, and may be of
            any size independent of the training set.
        training_data : table-like, optional
            Labeled rows (predictors and response) for the exact full-conformal
            set; ``interval="conformal"`` only, and exclusive with
            ``calibration``. Normally the training table the model was fitted
            on (``gam predict --conformal --training-data``).
        covariance_mode : {"conditional", "smoothing"}, optional
            Posterior covariance source for the interval (CLI<->Python parity
            with ``gam predict --covariance-mode``). ``"conditional"`` uses the
            conditional posterior ``H^{-1}`` only; ``"smoothing"`` requires the
            first-order smoothing-corrected covariance
            ``H^{-1} + J Var(rho_hat) J^T`` and errors if it cannot be formed.
            ``None`` (the default) uses the covariance the fit *publishes* —
            the one ``summary()`` prices its standard errors from: the
            smoothing-corrected matrix whenever the fit carries it, otherwise
            the conditional one (a fit certified at an infinite-smoothing rail,
            for instance) — and the result names the resolved definition in
            ``covariance_source``. Read
            whenever ``interval`` is set, for every family — including the
            curved-inverse-link families (binomial / Bernoulli) whose default
            point is the posterior mean: the mode shapes the reported SE and the
            credible bounds, while the posterior-mean point itself always
            integrates the conditional posterior and is unaffected.
        observation_interval : bool, default False
            When ``True`` (and ``interval`` is set), the output also gains
            ``observation_lower`` / ``observation_upper`` columns — the
            response-scale *prediction* interval
            ``Var(y_new|x) = Var(mu_hat) + Var(Y|mu)`` — for families that
            support it (Gaussian, Poisson, Gamma, Negative-Binomial, Beta,
            Tweedie, and binomial/Bernoulli via the conditional ``p(1-p)``
            variance). The credible ``posterior_mean_lower`` /
            ``posterior_mean_upper`` are left
            untouched.
        return_type : {"dict", "pandas", "numpy", "polars", "pyarrow", "list"}, optional
            Force a specific output container. ``None`` (default) mirrors the
            shape of ``data`` (and the training table where unambiguous).
            ``"numpy"`` is a structured array with one named field per output
            column, read by the same names as a DataFrame
            (``pred["posterior_mean_lower"]``).
        id_column : str or None, default None
            Name of an identifier column in ``data`` to propagate as a row key
            in the output (so predictions can be joined back to the input).

        Returns
        -------
        ndarray | PredictionResult | DataFrame | SurvivalPrediction | CompetingRisksPrediction
            The shape depends on the model class and on whether ``interval``,
            ``id_column``, or ``return_type`` was set:

            * Standard GAM, no interval / id_column / return_type: a 1-D
              ``ndarray`` of point predictions on the response scale (the
              posterior-mean fitted value).
            * Standard GAM with ``interval`` / ``id_column`` / ``return_type``:
              a table (dict / DataFrame / ...) with columns
              ``linear_predictor_plugin`` (``X·beta_hat``), ``mean_plugin``
              (its inverse-link image), and ``posterior_mean`` (the default
              response-scale point prediction). When ``interval`` is set it
              adds ``linear_predictor_standard_error`` (``SE(η)``),
              ``posterior_mean_standard_error`` (``√Var[link^{-1}(η)]``) plus
              ``posterior_mean_lower`` / ``posterior_mean_upper`` (the
              inverse link of the η credible quantiles).
              When the requested table container is ``"dict"``, the return is
              a ``PredictionResult``: it supports normal mapping access
              (``pred["posterior_mean"]``) and column attributes
              (``pred.posterior_mean``, ``pred.posterior_mean_standard_error``,
              ``pred.posterior_mean_lower``, ``pred.posterior_mean_upper``).
            * Bernoulli marginal-slope: a 1-D ``ndarray`` of probabilities. Its
              table form carries ``mean`` and, for a fit with no score warp,
              link deviation or residual repair block, the analytic
              derivative of that posterior-mean probability in the score
              column as supplied: ``mean_score_derivative``
              (``d mean / dz``) and ``probit_score_derivative``
              (``d probit(mean) / dz``, NaN where ``mean`` rounds to 0 or 1).
              Both come from the posterior nodes that integrate ``mean``.
            * Transformation-normal: a 1-D ``ndarray`` of the response-scale
              conditional mean ``E[Y|x]`` (issue #1612), a covariate-only
              quantity that does not require the outcome column.
            * Survival models: :class:`SurvivalPrediction`.
            * Competing-risks models: :class:`CompetingRisksPrediction`.

        Notes
        -----
        The response-scale ``posterior_mean`` is the **posterior mean** point estimate,
        never the plug-in mode: for a curved inverse link it is the
        coefficient-uncertainty-integrated ``E[link^{-1}(X·beta)]`` (the value
        the ``gam predict`` CLI reports by default), not ``link^{-1}(X·beta_hat)``.
        The choice is a property of the model alone, so it is the same whether
        or not ``interval`` is requested. For effectively-linear models
        (identity-link Gaussian, …) the integral collapses to the plug-in, so
        the two coincide exactly.

        ``mean_plugin`` is always exactly
        ``link^{-1}(linear_predictor_plugin)``. For Gaussian / identity-link
        GLMs all three explicit estimands are numerically identical. For a
        curved inverse link, ``posterior_mean`` generally differs from
        ``mean_plugin`` by Jensen's term. The names intentionally expose that
        distinction instead of presenting two different estimands as a generic
        ``linear_predictor`` / ``mean`` pair (#2785).
        """
        required = rust_module().required_model_columns(self._prediction_model, False)
        if required is not None and id_column is not None:
            required = sorted(set(required) | {id_column})
        positional_headers = None
        if detect_table_kind(data) == "numpy":
            try:
                positional_headers = rust_module().positional_prediction_headers(
                    self._prediction_model, numpy_table_width(data)
                )
            except Exception as exc:
                raise map_exception(exc) from exc
        headers, rows, table_kind = normalize_table(
            data, required_columns=required, positional_headers=positional_headers
        )
        row_ids = extract_row_ids(headers, rows, id_column)
        # interval='conformal' runs the gam_predict::conformal_routes column
        # builders `gam predict --conformal` uses: the exact full-conformal set
        # without a calibration fold, the split-conformal band with one. The
        # returned payload has the model-based predict column schema, so
        # shape_predict_response is unchanged.
        if interval == "conformal":
            # allow-list (a): FFI input validation.
            if (training_data is None) == (calibration is None):
                raise ValueError(
                    'interval="conformal" requires exactly one of training_data= '
                    "(exact full conformal) or calibration= (split conformal)"
                )
            try:
                if training_data is not None:
                    train_headers, train_rows, _ = normalize_table(training_data)
                    payload = rust_module().predict_table_full_conformal(
                        self._prediction_model,
                        headers,
                        rows,
                        train_headers,
                        train_rows,
                        conformal_level,
                    )
                else:
                    cal_headers, cal_rows, _ = normalize_table(calibration)
                    opts_json = rust_module().build_model_predict_payload_json(
                        self._prediction_model,
                        headers,
                        rows,
                        conformal_level,
                        covariance_mode,
                        observation_interval,
                    )
                    payload = rust_module().predict_table_conformal(
                        self._prediction_model,
                        headers,
                        rows,
                        cal_headers,
                        cal_rows,
                        conformal_level,
                        opts_json,
                    )
            except Exception as exc:
                raise map_exception(exc) from exc
            return shape_predict_response(
                payload,
                table_kind=table_kind,
                training_table_kind=self._training_table_kind,
                interval=conformal_level,
                return_type=return_type,
                id_column=id_column,
                row_ids=row_ids,
                restore=restore_output_table,
            )
        if calibration is not None:
            raise ValueError('calibration= applies only to interval="conformal"')
        if training_data is not None:
            raise ValueError('training_data= applies only to interval="conformal"')
        try:
            payload = rust_module().predict_table(
                self._prediction_model,
                headers,
                rows,
                interval,
                covariance_mode,
                observation_interval,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return shape_predict_response(
            payload,
            table_kind=table_kind,
            training_table_kind=self._training_table_kind,
            interval=interval,
            return_type=return_type,
            id_column=id_column,
            row_ids=row_ids,
            restore=restore_output_table,
        )

    def transformation_score(
        self,
        data: Any,
        *,
        return_type: str | None = None,
        id_column: str | None = None,
    ) -> Any:
        """Evaluate ``Phi^-1(F_hat(y|x))`` on labelled rows.

        This method is defined for conditional transformation-normal models
        and outcome models containing a saved CTN. It requires the CTN's
        fitted covariates and the observed
        response column.  It is intentionally distinct from :meth:`predict`,
        whose CTM point estimate is the response-scale conditional mean
        ``E[Y|x]`` and therefore does not consume an observed response.

        The returned score is the generated regressor used by a downstream
        marginal-slope model.  By default this is a one-dimensional NumPy
        array; ``return_type=`` or ``id_column=`` requests a one-column table
        named ``score`` (plus the requested identifier).
        """
        required = rust_module().required_model_columns(self._prediction_model, True)
        if required is not None and id_column is not None:
            required = sorted(set(required) | {id_column})
        headers, rows, table_kind = normalize_table(data, required_columns=required)
        row_ids = extract_row_ids(headers, rows, id_column)
        try:
            scores = rust_module().transformation_score_table(
                self._prediction_model, headers, rows
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        if return_type is None and id_column is None:
            return scores
        columns: dict[str, list[Any]] = {"score": scores.tolist()}
        if id_column is not None:
            columns = {id_column: list(row_ids or []), **columns}
        return restore_output_table(
            columns,
            requested=return_type,
            input_kind=table_kind,
            training_kind=self._training_table_kind,
        )

    def latent_conditional_residual(
        self,
        data: Any,
        *,
        return_type: str | None = None,
        id_column: str | None = None,
    ) -> Any:
        """Evaluate the conditional latent residual ``(z - m(a)) / sqrt(v(a))``.

        This method is defined for marginal-slope models fitted with a
        conditional latent law (``latent_measure="conditional-location-scale"``).
        It applies the map the fit applied to its own score, so at the
        training rows it returns the fit's standardized score bit for bit, and
        on new rows it returns the residual a held-out adequacy check compares
        with the training residual law. The rows need the score column and the
        conditioning covariates; a survival model's time columns are not read.

        Returns ``None`` when the fit consumed no conditional latent law. By
        default the residual is a one-dimensional NumPy array;
        ``return_type=`` or ``id_column=`` requests a one-column table named
        ``residual`` (plus the requested identifier).
        """
        headers, rows, table_kind = normalize_table(data)
        row_ids = extract_row_ids(headers, rows, id_column)
        try:
            residual = rust_module().latent_conditional_residual_table(
                self._prediction_model, headers, rows
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        if residual is None or (return_type is None and id_column is None):
            return residual
        columns: dict[str, list[Any]] = {"residual": residual.tolist()}
        if id_column is not None:
            columns = {id_column: list(row_ids or []), **columns}
        return restore_output_table(
            columns,
            requested=return_type,
            input_kind=table_kind,
            training_kind=self._training_table_kind,
        )

    def predict_array(
        self,
        X: Any,
        *,
        interval: float | None = None,
        covariance_mode: str | None = None,
        observation_interval: bool = False,
    ) -> Any:
        """Predict directly from a numeric NumPy-compatible feature matrix.

        Only valid for models fitted via :func:`gamfit.fit_array` — positional
        column order is only well-defined when the model itself was fitted
        from a positional array, so the engine knows the predictor columns
        are the synthetic sequence ``x0, x1, ..., x{p-1}`` (issue #341). For
        models fitted from a named table (``gamfit.fit(df, formula)``), call
        :meth:`predict` with a ``dict`` / DataFrame instead so columns can
        be matched by name; silently mapping positional X to named features
        would misorder swapped columns and produce wrong predictions.

        ``interval`` is the single uncertainty knob (issue #342); see
        :meth:`predict` for its semantics. ``covariance_mode`` and
        ``observation_interval`` mirror :meth:`predict` (CLI<->Python parity
        with ``gam predict --covariance-mode``).
        """
        options: dict[str, Any] = {"interval": interval}
        if covariance_mode is not None:
            options["covariance_mode"] = covariance_mode
        if observation_interval:
            options["observation_interval"] = observation_interval
        try:
            rust = rust_module()
            result = rust.predict_array(
                self._prediction_model,
                rust.numeric_matrix_f64(X, "X"),
                json.dumps(options),
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        if interval is None:
            # Parity with :meth:`predict` (#1537): with no interval the result is
            # the 1-D response-scale prediction vector, not the engine's
            # full estimand-explicit column matrix. The FFI returns the lone
            # response-scale `posterior_mean` column as `(n, 1)`; drop the
            # trailing axis. A joint expectile fit's point is its `(n, K)`
            # level curves, returned as is.
            import numpy as np

            result = np.asarray(result)
            return result.reshape(-1) if result.shape[1] == 1 else result
        return result

    def summary(self) -> Summary:
        """Return the model summary (coefficients, family, deviance, REML score)."""
        try:
            payload = rust_module().summary_payload_from_model(self._prediction_model)
        except Exception as exc:
            raise map_exception(exc) from exc
        return Summary.from_dict(payload)

    def smoothing_parameters(self) -> dict[int, float]:
        """Return fitted smoothing/precision parameters by penalty index."""
        return dict(rust_module().smoothing_parameters_from_model(self._prediction_model))

    # -- fitted-result accessors -------------------------------------------------
    # Each reads one field of the Rust ``SummaryPayload`` (the same document
    # ``gam summary`` prints); nothing is recomputed here.

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Fitted coefficient vector ``beta_hat`` in design-column order."""
        return np.asarray(
            [record["estimate"] for record in self.summary().coefficients], dtype=float
        )

    @property
    def edf_total(self) -> float | None:
        """Total effective degrees of freedom
        ``tr(H^-1 X'WX) = p - sum_k tr(lambda_k H^-1 S_k)``; ``n - edf_total``
        is the residual degrees of freedom behind :attr:`scale`."""
        return self.summary().edf_total

    @property
    def smooth_edf(self) -> dict[str, float]:
        """Effective degrees of freedom of each smooth / random-effect term,
        keyed by term name (the ``edf`` column of
        :meth:`Summary.smooth_terms_frame`)."""
        return {
            str(record["name"]): float(record["edf"])
            for record in self.summary().smooth_terms
        }

    @property
    def scale(self) -> float | None:
        """Estimated dispersion ``phi_hat`` of the fitted family.

        Gaussian: ``sigma_hat^2 = RSS_w / (n - edf_total)`` (mgcv's
        ``gam.scale``); Gamma: ``1 / shape``; fixed-scale families (Poisson,
        binomial): ``1``. ``None`` exactly when the family's scale contract
        has no scalar dispersion: a custom family that declares none, or
        Royston-Parmar survival.
        """
        return self.summary().scale

    @property
    def log_likelihood(self) -> float | None:
        """Ordinary log-likelihood at the fitted coefficients and :attr:`scale`."""
        return self.summary().log_likelihood

    @property
    def deviance(self) -> float | None:
        """Model deviance at the fitted coefficients (prior weights included)."""
        return self.summary().deviance

    @property
    def n_obs(self) -> int | None:
        """Number of training rows the model was fitted on."""
        return self.summary().n_obs

    @property
    def convergence(self) -> dict[str, Any] | None:
        """The optimizer's convergence certificate; see :attr:`Summary.convergence`."""
        return self.summary().convergence

    @property
    def outer_iterations(self) -> int | None:
        """Outer (smoothing-parameter) iterations the convergence proof covers."""
        convergence = self.convergence
        return None if convergence is None else convergence["outer_iterations"]

    @property
    def inner_iterations(self) -> int | None:
        """Inner P-IRLS iterations of the final coefficient solve."""
        convergence = self.convergence
        return None if convergence is None else convergence["inner_iterations"]

    def residuals(
        self,
        data: Any,
        type: Literal["response", "working", "deviance", "pearson"] = "deviance",
    ) -> NDArray[np.float64]:
        """Per-row residuals of the fit on the labelled rows ``data``.

        The saved model carries no per-row training data, so the rows are
        passed back: the training table gives in-sample residuals, any other
        labelled table gives residuals at the fitted coefficients. The Rust
        core evaluates ``eta = X beta_hat + offset`` and the family's residual
        kernel, with prior weights as at fit time:

        - ``"response"``: ``y - mu``
        - ``"working"``: ``(y - mu) / (dmu/deta)``
        - ``"deviance"``: ``sign(y - mu) sqrt(d_i)``, so
          ``sum(r**2) == deviance`` on the training rows
        - ``"pearson"``: ``(y - mu) sqrt(w / V(mu))``

        ``gam residuals MODEL DATA --type TYPE`` returns the same values.
        """
        headers, rows, _ = normalize_table(data)
        try:
            return rust_module().residuals_table(
                self._prediction_model, headers, rows, type
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def check(self, data: Any) -> SchemaCheck:
        """Validate ``data`` against the model's training schema."""
        headers, rows, _ = normalize_table(data)
        try:
            payload = rust_module().check_payload_from_model(self._prediction_model, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        return SchemaCheck.from_dict(payload)

    def curvature(self, data: Any, *, level: float = 0.95) -> list[dict[str, Any]]:
        """Curvature-as-an-estimand report for every ``curv(...)`` smooth (#944).

        For each constant-curvature (``curv(...)``) smooth in the model this
        returns the fitted signed sectional curvature ``kappa_hat``, its
        profile-likelihood confidence interval ``(ci_lo, ci_hi)``, the geometry
        ``verdict`` from the CI sign (``"spherical"`` / ``"hyperbolic"`` /
        ``"flat"`` / ``"indistinguishable"``), and the interior :math:`\\kappa=0`
        likelihood-ratio flatness test (``flatness_lr_stat``,
        ``flatness_p_value`` — full :math:`\\chi^2_1`, since :math:`\\kappa=0`
        is an interior point of the :math:`S^d \\leftarrow \\mathbb{R}^d \\to H^d`
        family). This turns "we chose hyperbolic space" into
        ":math:`\\hat\\kappa = -1.8` (95% CI ...), flat rejected at p = ...".

        Each row also carries ``length_scale_hat`` — the geodesic kernel range
        :math:`\\hat\\ell` the criterion profiles to at :math:`\\hat\\kappa` — and
        ``length_scale_estimated``. The range is the smooth's second outer
        coordinate and every statistic in the row is a profile over it (#2747):
        :math:`\\hat\\kappa` is the argmin of
        :math:`V_p(\\kappa) = \\min_\\ell V(\\kappa, \\ell)`, the interval is a
        profile-likelihood interval, and the flatness LR compares two
        range-profiled values. Pinning the range with an explicit
        ``length_scale=`` sets ``length_scale_estimated`` to ``False`` and makes
        :math:`\\hat\\kappa` conditional on that choice — which is not
        recommended, because :math:`\\kappa` and :math:`\\ell` enter
        :math:`\\exp(-d_\\kappa/\\ell)` through one exponent and a
        :math:`\\kappa` fitted against a wrong range reports the range error.

        ``kappa_hat`` alone is also surfaced in :meth:`summary` with no refit;
        the CI and flatness test re-profile the criterion over :math:`\\kappa`,
        which is why they require the training ``data``.

        Returns an empty list when the model has no ``curv(...)`` smooth.
        """
        headers, rows, _ = normalize_table(data)
        try:
            raw = rust_module().curvature_inference_json(
                self._prediction_model, headers, rows, level
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        payload = json.loads(raw)
        return list(payload.get("curvature_terms", []))

    def smooth_significance(self, data: Any) -> list[dict[str, Any]]:
        """Per-term likelihood-ratio significance for every penalized smooth (#1063).

        :meth:`summary` reports a variance-component *score* statistic. The
        exact Lawley / Bartlett factor corrects the *likelihood-ratio*
        statistic, and a score statistic's second-order mean is not
        :math:`d + \\Delta\\varepsilon`, so dividing it by the LR factor would
        correct the wrong statistic. This method
        instead computes a genuine per-term LR statistic
        :math:`W = 2(\\ell_{\\text{full}} - \\ell_{\\text{null}})` by a
        constrained fit that fixes the smooth's coefficients at zero while
        holding every other smoothing parameter at the full fit's
        :math:`\\hat\\lambda`, then Bartlett-corrects *that*:
        :math:`W^* = W / c`, :math:`c = 1 + \\Delta\\varepsilon / d`.

        The reference :math:`W` is scored against is the statistic's own null
        law, not a distribution fitted to some of its moments. At fixed
        :math:`\\lambda` the penalized LR is exactly
        :math:`W = \\sum_j w_j \\chi^2_1` with
        :math:`w = \\mathrm{eig}(2F_{jj} - F_{jj}^2)` on the tested block of the
        influence matrix, so :math:`\\sum_j w_j` is Wood's ``edf1`` *and* the
        statistic's null mean — and the p-value is
        :math:`P(\\sum_j w_j \\chi^2_1 > W)` itself, obtained by Imhof inversion
        of the characteristic function. For an unpenalized block every
        :math:`w_j = 1` and this is exactly the textbook :math:`\\chi^2_q`.

        **When the scale is estimated the reference is not that law** (#2672).
        gam's profiled Gaussian log-likelihood is
        :math:`\\ell = -\\tfrac12[n\\ln 2\\pi + n\\ln(D/\\nu) - \\sum\\ln w_i +
        \\nu]`, so with no expansion anywhere
        :math:`W = n\\ln(1 + Q/V) + B`, where :math:`Q = (D_0 - D_f)/\\sigma^2`
        is the quantity ``reference_weights`` is the spectrum *of*,
        :math:`V = D_f/\\sigma^2` is a random variable of the same data, and
        :math:`B = n\\ln(\\nu_f/\\nu_0) + (\\nu_0 - \\nu_f)`. Scoring :math:`W`
        against :math:`Q`'s law is anti-conservative at :math:`O(1/\\nu)` —
        measured ``size@.05 = 0.0792`` against a nominal ``0.05`` at
        :math:`n \\in \\{30, 50\\}`. Because the map is monotone it inverts
        exactly: :math:`P(W > w) = P(Q - c(w)V > 0)` with
        :math:`c(w) = \\mathrm{expm1}((w - B)/n)`, a linear combination of
        independent chi-squares with a negative weight, evaluated at zero. It is
        the same reason mgcv's smooth-term p-values take an :math:`F` reference
        when the scale is estimated and a :math:`\\chi^2` when it is known.
        ``reference_residual_df`` (:math:`\\nu`) and
        ``reference_deterministic_offset`` (:math:`B`) are ``None`` on every
        family that carries its dispersion in the IRLS weight instead.

        ``reference_source`` says which lane produced it: ``"null_spectrum"``
        (the exact law above, with ``reference_weights`` carrying :math:`w`),
        ``"spectral_moment_match"`` (the fit could supply only the spectrum's
        first two moments, so the reference is the two-moment
        :math:`P(\\chi^2_\\nu > W/g)`, :math:`\\nu = (\\sum w)^2/\\sum w^2`,
        :math:`g = \\sum w^2/\\sum w` — accurate to a percent at
        :math:`\\alpha = 0.05` and up to 1.6x anti-conservative at
        :math:`10^{-4}`), or ``"unit_weight_fallback"``.

        It returns one row per tested smooth term, always with the same keys.
        The published p-value is exactly one of:

        * ``p_value`` — the tail of the Bartlett-corrected statistic, resolved
          to within ``p_value_bound``;
        * ``p_value_upper_bound`` — the published accuracy does not separate
          the tail from zero, so it is reported as ``p < p_value_upper_bound``
          (the top of the certified interval) rather than as a residue such
          as ``0.0``;
        * ``unavailable_reason`` — a stable label
          (``"empty_coefficient_block"``, ``"degenerate_reference"``,
          ``"full_refit_failed"``, ``"null_fit_not_converged"``,
          ``"null_fit_unsupported"``,
          ``"null_log_likelihood_not_finite"``, ``"tail_not_computable"``,
          ``"selection_refused"``)
          with ``unavailable_message`` saying what happened; every inference
          field of such a row is ``None``.

        For a term with an inference row it also carries
        ``statistic_lr`` (the raw :math:`W`), ``ref_df`` (the null mean
        :math:`d = \\sum_j w_j`, which is what the Bartlett factor is
        denominated in — *not* a chi-square degrees of freedom),
        ``reference_weights``/``reference_source``, ``reference_chi_square_df``
        :math:`\\nu` and ``reference_scale`` :math:`g` (the two-moment summary),
        ``reference_residual_df``/``reference_deterministic_offset`` (the
        estimated-scale channel above, ``None`` off the profiled Gaussian),
        ``bartlett_factor`` :math:`c` (the fixed-λ Lawley scale),
        ``statistic_corrected`` :math:`W^* = W/c`, ``p_value_uncorrected``,
        ``p_value_corrected`` (the raw evaluated tail behind ``p_value`` /
        ``p_value_upper_bound``), ``material`` (the
        n-too-small-here diagnostic — ``True`` when the correction moves the
        Bartlett factor or the p-value by more than 10%), and
        ``correction_provenance`` — ``"lawley_lr_fixed_lambda"`` when the
        family carries
        closed-form cumulant jets (gaussian / poisson / binomial / gamma) and the
        factor is computable at this ``n``, else
        ``"none"`` (the uncorrected reference stands, never weakened).

        A shape-constrained smooth (``shape=...``) gets no LR p-value. Its null
        :math:`f = 0` is the apex of the constraint cone and the fitted
        coefficients are a truncated posterior mean, so no :math:`\\chi^2`,
        spectral or chi-bar-square reference is calibrated. Such a term appears
        as a row with only ``name``, ``term_idx``, ``p_value_unavailable``
        (``"shape_constrained"``) and ``explanation``; every tested row carries
        no ``p_value_unavailable`` key.

        Needs the training ``data`` for the per-term null refits, exactly as
        :meth:`curvature` does. Rows are in term order. Returns an empty list
        when the model has no penalized smooth term.
        """
        headers, rows, _ = normalize_table(data)
        try:
            raw = rust_module().smooth_term_lr_inference_json(
                self._prediction_model, headers, rows
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        payload = json.loads(raw)
        terms = list(payload["smooth_terms"]) + list(payload["unavailable"])
        return sorted(terms, key=lambda row: row["term_idx"])

    def basis_check(self, data: Any) -> list[dict[str, Any]]:
        r"""Per-smooth basis-adequacy report: is each smooth's basis big enough (#2774)?

        A converged, ``certified`` fit says nothing about whether the basis it
        was given can represent the function it was asked to model. Both a
        smooth whose basis spans the truth and a smooth that cannot reach it
        reach a stationary REML point, certify, and report a per-term EDF that
        is some fraction of the term's column count. What separates them is
        whether the **residuals still carry structure in that smooth's own
        covariates**.

        For each smooth term this returns

        * ``basis_dim`` — the realized coefficient width :math:`k'`;
        * ``nullspace_dim`` — the dimension of the term's **joint** penalty null
          space, i.e. the directions no penalty on it touches at all, so
          ``basis_dim − nullspace_dim`` is the penalizable capacity. It is ``0``
          for every *double-penalized* smooth, which includes the whole radial
          family;
        * ``edf`` — the term's effective degrees of freedom, carried here beside
          the two above because the three only mean anything together. A
          ``d``-dimensional radial smooth carries an RKHS curvature penalty plus
          a complementary trend ridge on its ``d + 1``-column polynomial block;
          that block is only weakly penalized, so it carries most of the term's
          EDF and makes ``edf`` read near-saturated on a fit whose problem is the
          *span* of its basis rather than its rank. That reading is exactly what
          ``p_value`` replaces;
        * ``enrichment_dim`` / ``enrichment_rank`` — the width of the
          higher-resolution alternative the residuals were tested against, and
          how many of its directions survived projecting the fitted design out.
          The rank is the test's reference degrees of freedom and a direct
          measure of how much genuinely new resolution the alternative carried;
        * ``statistic`` / ``p_value`` — the penalized score (Rao) lack-of-fit
          test. Small ``p_value`` ⇒ there is signal in this smooth's covariates
          that its realized basis cannot represent;
        * ``provenance`` — ``"radial_enrichment"`` when a test ran, else the
          NAME of the evidence that was missing (``"no_continuous_covariates"``,
          ``"enrichment_budget_below_realized_width"``, ``"no_irls_row_state"``,
          ``"design_gram_unavailable"``, ``"null_fit_unavailable"``,
          ``"conditional_reference_unavailable"``, ...). ``p_value`` is present
          exactly when a test ran, so "adequate" and "not measured" are never
          confusable.

        Method
        ------
        The alternative is a Duchon kernel at data-driven centers over the
        term's standardized covariates, orthogonalized against the fitted design
        in the fit's own IRLS weight metric. The statistic is
        :math:`T = U^{\top} V^{-} U / \hat\varphi` with
        :math:`U = \tilde Z^{\top} s` and :math:`V = \tilde Z^{\top} W \tilde Z`,
        referred to :math:`\chi^2_r` (known dispersion) or, with the scale
        estimated on :math:`\nu` residual degrees of freedom, as the
        added-variable :math:`(T/r)(\nu - r)/(\nu - T)` to :math:`F(r, \nu - r)`.

        For a canonical binomial (logit) or Poisson (log) fit that reference is
        only first order, and at small ``n`` its error is not small (a
        conservative test is as miscalibrated as an anti-conservative one).
        There the score is instead evaluated at the unpenalized null MLE on the
        test's rows and referred to its law CONDITIONAL on the sufficient
        statistic :math:`X^{\top}(w \circ y)`, which removes the nuisance
        :math:`\beta` exactly: the score's conditional mean and covariance are
        corrected to :math:`O(1/n)` and its fourth cumulant matched by a scaled
        :math:`c\,\chi^2_{r/c}`. Where that expansion leaves its range of
        validity (high-leverage rows at an extreme fitted mean) the row reports
        ``"conditional_reference_unavailable"`` rather than a number.

        The projection is **orthogonal in the weight metric**, not the fit's
        penalized :math:`H^{-1}`. That is deliberate and it is what the test
        means: a penalized fit is biased, and its shrinkage bias lives entirely
        inside the span of the fitted design, so projecting it out makes the
        statistic blind to "λ is large" and sensitive only to structure the
        design *cannot represent at all*. Asking whether a direction the basis
        HAS is being over-shrunk is a smoothing-parameter question and this
        report deliberately declines to answer it.

        What it does not claim
        ----------------------
        :math:`\hat\lambda` is held at its fitted value and the alternative is
        fixed, so the test is conditional on both — exactly as the
        :meth:`summary` Wald column is conditional on :math:`\hat\lambda`. A
        rejection says there is signal outside the term's column span; it does
        not say how much of *your* estimand that signal moves. Refit with a
        larger basis for the flagged term and compare.

        Relationship to :attr:`Summary.basis_checks`
        -------------------------------------------
        :meth:`summary` reports what the FIT measured and persisted, with no
        data and no work. This method **recomputes** it, which requires refitting
        at the model's frozen spec first: the score is a function of converged
        IRLS row state (weights, working response, linear predictor) that a saved
        model does not carry. Use it for models saved before this check existed,
        or to run the check against rows other than the training ones. It is as
        expensive as a refit, exactly like :meth:`smooth_significance`.

        Returns an empty list when the model has no smooth terms.

        Examples
        --------
        >>> [(row["name"], row["p_value"]) for row in model.basis_check(train)]
        [('duchon(pc1, ..., centers=24)', 9.0e-16)]
        """
        headers, rows, _ = normalize_table(data)
        try:
            raw = rust_module().basis_adequacy_json(self._prediction_model, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        payload = json.loads(raw)
        return list(payload.get("basis_checks", []))

    def debiased_functional(
        self,
        data: Any,
        target: str,
        *,
        x0: dict[str, Any] | None = None,
        x1: dict[str, Any] | None = None,
        weights: list[float] | None = None,
        deriv_var: str | None = None,
    ) -> dict[str, float]:
        """Riesz-representer debiased / Neyman-orthogonal estimate of a smooth
        functional (#1055).

        Computes a second-order-accurate point estimate of a smooth functional
        ``θ = g(m)`` using the Riesz-representer one-step bias correction —
        the standard Neyman-orthogonal / doubly-robust estimator for penalized
        regression functionals. The orthogonal correction is always applied
        (it strictly improves coverage under regularization; no flag).

        Currently restricted to Gaussian/identity-link models (exact per-row
        score contributions). For other families supply raw arrays via the
        low-level ``gamfit._rust.debiased_functional(...)`` call.

        Parameters
        ----------
        data : table-like
            The **training** data used to fit this model, in any format
            accepted by :meth:`predict`. Required to reconstruct the per-row
            score contributions ``∂nll_i/∂β``.
        target : str
            The named functional estimand:

            * ``"point"`` — ``m(x0)``, the smooth evaluated at a query point.
              Requires ``x0``.
            * ``"contrast"`` — ``m(x0) − m(x1)`` (a treatment contrast).
              Requires ``x0`` and ``x1``.
            * ``"average_value"`` — ``mean_i w_i m(x_i)`` over training rows.
              Optional ``weights``.
            * ``"average_derivative"`` — ``mean_i w_i (∂m/∂x)(x_i)`` over
              training rows. Optional ``weights``.
        x0, x1 : dict or None
            Query-point column dicts for ``"point"`` and ``"contrast"``
            targets.  Keys must match the model's predictor column names.
        weights : list of float or None
            Per-row importance weights for ``"average_value"`` /
            ``"average_derivative"`` (length == number of training rows).
        deriv_var : str or None
            For ``"average_derivative"`` only: the covariate column to
            differentiate with respect to. Auto-selected when the model has a
            smooth over a single covariate; supply it explicitly when the model
            has smooths over more than one covariate.

        Returns
        -------
        dict
            ``theta_plugin`` (plug-in estimate without debiasing),
            ``theta_debiased`` (Neyman-orthogonal one-step estimate),
            ``se`` (influence-function standard error),
            ``penalty_bias`` (estimated regularization bias removed),
            ``ci_lower`` / ``ci_upper`` (95% normal-approximation CI).
        """
        headers, rows, _ = normalize_table(data)
        spec: dict[str, Any] = {"target": target}
        if x0 is not None:
            spec["x0"] = {k: v for k, v in x0.items()}
        if x1 is not None:
            spec["x1"] = {k: v for k, v in x1.items()}
        if weights is not None:
            spec["weights"] = list(weights)
        if deriv_var is not None:
            spec["deriv_var"] = deriv_var
        try:
            raw = rust_module().model_debiased_functional_json(
                self._prediction_model, headers, rows, json.dumps(spec)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return dict(json.loads(raw))

    def report(self, path: str | Path | None = None) -> str:
        """Generate a standalone HTML report of the fitted model."""
        try:
            html = rust_module().report_html(self._prediction_model)
        except Exception as exc:
            raise map_exception(exc) from exc
        # allow-list (a): FFI response marshaling for optional file output.
        if path is None:
            return str(html)
        Path(path).write_text(html, encoding="utf-8")
        return str(path)

    def sample(
        self,
        data: Any,
        *,
        samples: int | None = None,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Draw from the model's posterior with NUTS."""
        headers, rows, _ = normalize_table(data)
        try:
            ffi = rust_module()
            options_json = ffi.build_sample_payload_json(samples, seed)
            payload = ffi.sample_table(
                self._prediction_model,
                headers,
                rows,
                options_json,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return PosteriorSamples.from_ffi_payload(payload, model=self._prediction_model)

    def sample_replicates(
        self,
        data: Any,
        n_draws: int = 100,
        *,
        seed: int = 0,
    ) -> Any:
        """Materialize posterior-predictive replicate responses at ``data``.

        Each of the ``n_draws`` rows is a fresh synthetic response vector drawn
        from the fitted predictive distribution. The saved model's canonical
        generative capability supplies both its response-scale predictor and
        observation law; this includes exact spline-scan fits and fitted
        location/dispersion-scale or transformation-normal families without a
        second Python family allowlist.
        This is the *observation* replicate path (distinct from :meth:`sample`,
        which draws the *parameter* posterior) and is the engine for
        posterior-predictive checks, synthetic-data generation, and
        simulation-based calibration. The family, fitted dispersion, and any
        analytic row-weight column are read from the saved model — there is no
        family flag and no refit. Weighted models require that column in
        ``data`` because replacing missing weights by one would sample from a
        different observation law.

        Parameters
        ----------
        data : table-like
            New rows in any format accepted by :meth:`predict`. Must cover every
            predictor referenced by the fitted formula (the response column, if
            present, is ignored).
        n_draws : int, default 100
            Number of replicate response vectors to draw.
        seed : int, default 0
            Seed for the deterministic draw stream.

        Returns
        -------
        numpy.ndarray
            An ``(n_draws, n_rows)`` array of synthetic responses.

        Notes
        -----
        This convenience method allocates the complete result. Use
        :meth:`iter_replicates` when the requested draw matrix is large.
        """
        n_draws = int(n_draws)
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        headers, rows, _ = normalize_table(data)
        try:
            return rust_module().generative_replicates(
                self._prediction_model, headers, rows, n_draws, int(seed)
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def iter_replicates(
        self,
        data: Any,
        n_draws: int = 100,
        *,
        chunk_size: int,
        seed: int = 0,
    ) -> Iterator[NDArray[np.float64]]:
        """Stream posterior-predictive replicates in bounded draw chunks.

        This is the bounded-memory form of :meth:`sample_replicates`. Each
        yielded array has shape ``(min(chunk_size, remaining), n_rows)``. Draws
        are indexed globally, so changing ``chunk_size`` only changes batch
        boundaries: concatenating the chunks is bit-for-bit identical to
        ``sample_replicates(data, n_draws, seed=seed)``.

        The values always represent the saved model's observation law. For a
        single-cause or latent survival fit they are conditional event-in-window
        indicators (zero or one); for competing risks, zero means no event and
        positive integer labels identify the persisted cause. The sampler does
        not invent censoring or inspection records that the saved response law
        does not contain.

        Parameters
        ----------
        data : table-like
            New rows in any format accepted by :meth:`predict`.
        n_draws : int, default 100
            Total number of replicate response vectors to draw.
        chunk_size : int
            Maximum number of draw rows retained at once. It is required so
            the caller, not a hidden heuristic, owns the memory bound.
        seed : int, default 0
            Seed for the deterministic draw stream.

        Yields
        ------
        numpy.ndarray
            The next contiguous range of synthetic response vectors.
        """
        n_draws = int(n_draws)
        chunk_size = int(chunk_size)
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        headers, rows, _ = normalize_table(data)
        ffi = rust_module()

        def chunks() -> Iterator[NDArray[np.float64]]:
            for draw_start in range(0, n_draws, chunk_size):
                draw_count = min(chunk_size, n_draws - draw_start)
                try:
                    chunk = ffi.generative_replicate_chunk(
                        self._prediction_model,
                        headers,
                        rows,
                        draw_start,
                        draw_count,
                        int(seed),
                    )
                except Exception as exc:
                    raise map_exception(exc) from exc
                yield np.asarray(chunk, dtype=np.float64)

        return chunks()

    def design_matrix(self, data: Any) -> AffineDesign:
        """Return the exact fitted affine predictor design for ``data``.

        The result always has one typed shape.  For an ordinary standard GAM,
        ``offset`` is the model's row offset, ``matrix`` is the full saved-model
        design, and the coefficient frame is ``"full"``. For a link-wiggle
        fit, ``offset`` is the model row offset and ``matrix`` is the joint
        ``[X, B(warp_index)]`` design, with the warp basis evaluated at its
        exact fitted index (including the frozen #2141 shift); its frame is
        ``"link_wiggle_joint"``.

        In both cases, ``offset + matrix @ coefficients`` reproduces the fitted
        linear predictor. Available conditional, smoothing-corrected, and
        frequentist covariances are returned separately in the same frame, and
        ``eta_gradient @ covariance @ eta_gradient.T`` includes all cross-block
        terms. ``eta_gradient`` is the same array as ``matrix`` for the
        ``"full"`` frame and carries the extra warp slope for
        ``"link_wiggle_joint"``; see :class:`AffineDesign`. Scan-routed and
        coupled multi-surface models have no finite single-frame affine
        representation and are rejected explicitly.
        """
        headers, rows, _ = normalize_table(data)
        try:
            payload = rust_module().affine_design_table(self._prediction_model, headers, rows)
            return _affine_design_from_payload(payload)
        except Exception as exc:
            raise map_exception(exc) from exc

    def design_matrix_array(self, X: Any) -> AffineDesign:
        """Exact fitted affine predictor design for a numeric feature matrix."""
        try:
            rust = rust_module()
            return _affine_design_from_payload(
                rust.affine_design_array(
                    self._prediction_model,
                    rust.numeric_matrix_f64(X, "X"),
                )
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def difference_smooth(
        self,
        *,
        view: str,
        group: str | None = None,
        pairs: Sequence[tuple[Any, Any]] | None = None,
        n: int = 100,
        level: float | None = None,
        simultaneous: bool = False,
        n_sim: int | None = None,
        seed: int | None = None,
        marginalise_random: bool = True,
        group_means: bool = True,
        data: Any | None = None,
        return_type: str | None = None,
    ) -> Any:
        """Covariance-aware pairwise difference smooths (Rust-backed)."""
        template: dict[str, str] = {}
        # allow-list (a): FFI input marshaling for an optional template row.
        if data is not None:
            headers, rows, _ = normalize_table(data)
            # allow-list (a): FFI input marshaling for empty prediction tables.
            if rows:
                first = rows[0]
                # allow-list (a): FFI payload marshaling.
                template = dict(zip(headers, map(str, first), strict=True))
        try:
            # allow-list (a): FFI optional argument marshaling.
            group_arg = str(group) if group is not None else None
            # allow-list (a): FFI payload sequence marshaling.
            pairs_arg = (
                list(map(lambda pair: (str(pair[0]), str(pair[1])), pairs))
                if pairs is not None
                else None
            )
            # allow-list (a): FFI optional argument marshaling.
            seed_arg = int(seed) if seed is not None else None
            # allow-list (a): FFI optional argument marshaling.
            template_arg = None if not template else template
            request_json = rust_module().build_difference_smooth_request_json(
                str(view),
                group_arg,
                pairs_arg,
                int(n),
                float(level) if level is not None else None,
                bool(simultaneous),
                int(n_sim) if n_sim is not None else None,
                seed_arg,
                bool(marginalise_random),
                bool(group_means),
                template_arg,
            )
            rows_out = rust_module().difference_smooth_rows(
                self._prediction_model, request_json
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        # allow-list (a): FFI response marshaling for requested output type.
        if return_type == "list":
            return rows_out
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for DataFrame output; install pandas or set return_type='list'") from exc
        return pd.DataFrame(rows_out)

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to ``path``.

        The save is atomic: a failed save leaves the file at ``path`` as it
        was. On Unix it is durable before it returns.
        """
        rust_module().write_saved_model_file(path, self._model_bytes)

    def extend_with_group(
        self,
        new_group_spec: dict[str, Any],
        metadata: Any | None = None,
        prior: Any | None = None,
    ) -> "Model":
        """Return a no-refit model extended with deployment-time group levels."""
        # allow-list (a): FFI input validation.
        if not isinstance(new_group_spec, dict):
            raise TypeError("new_group_spec must be a dict")
        try:
            rust = rust_module()
            # allow-list (a): FFI optional argument marshaling.
            metadata_json = json.dumps(metadata) if metadata is not None else None
            # allow-list (a): FFI optional argument marshaling.
            prior_json = json.dumps(prior) if prior is not None else None
            payload_json = rust.build_extend_group_payload_json(
                json.dumps(new_group_spec),
                metadata_json,
                prior_json,
            )
            model_bytes = bytes(
                rust.extend_model_with_group(self._prediction_model, payload_json)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return Model(
            _model_bytes=model_bytes,
            _training_table_kind=self._training_table_kind,
        )

    def dumps(self) -> bytes:
        """Return the serialised model as raw bytes."""
        return self._model_bytes

    @property
    def formula(self) -> str:
        return self._prediction_model.formula

    @property
    def family_name(self) -> str:
        return self.summary().family_name

    @property
    def student_t_sigma(self) -> float | None:
        """LAML-estimated scale σ of a ``family="student-t"`` fit; ``None`` otherwise."""
        params = rust_module().student_t_parameters_from_model(self._prediction_model)
        return None if params is None else params[0]

    @property
    def student_t_nu(self) -> float | None:
        """LAML-estimated degrees of freedom ν of a ``family="student-t"`` fit; ``None`` otherwise."""
        params = rust_module().student_t_parameters_from_model(self._prediction_model)
        return None if params is None else params[1]

    @property
    def notes(self) -> list[str]:
        """Notes recorded while this model was fit, advisories first.

        An *advisory* says the fitted model differs from what was literally
        requested — e.g. ``"... basis reduced from k=10 to k=3 to match the
        covariate's 3 distinct value(s)"`` when a cubic-regression marginal is
        capped to the data support, or a basis-degradation note when a
        low-cardinality covariate cannot support the requested smooth.
        :func:`gamfit.fit` also emits each advisory as a
        :class:`gamfit.errors.GamInferenceWarning` at fit time.

        An *informational* note records a default the engine chose on the
        caller's behalf — e.g. the internal-knot count of a default B-spline
        smooth. These are not warnings; they are listed here and in
        :meth:`summary` so the choice stays inspectable (also after loading a
        saved model).

        Empty when the fit used exactly the requested configuration and chose
        no defaults worth recording.
        """
        advisories, informational = self._fit_notes()
        return [*advisories, *informational]

    def _fit_notes(self) -> tuple[list[str], list[str]]:
        compiled = self._prediction_model
        return list(compiled.inference_notes), list(compiled.informational_notes)

    @property
    def used_device(self) -> bool:
        return self._prediction_model.used_device

    @property
    def model_class(self) -> str:
        return self._model_class_from_payload()

    def _class_traits(self) -> dict[str, Any]:
        return rust_module().saved_model_class_traits(self._prediction_model)

    @property
    def is_survival(self) -> bool:
        return bool(self._class_traits()["is_survival"])

    @property
    def is_marginal_slope(self) -> bool:
        return bool(self._class_traits()["is_marginal_slope"])

    @property
    def is_transformation_normal(self) -> bool:
        return bool(self._class_traits()["is_transformation_normal"])

    @property
    def response_name(self) -> str | None:
        return rust_module().response_column_name(self.formula)

    @property
    def training_table_kind(self) -> str:
        return self._training_table_kind

    @property
    def group_metadata(self) -> dict[str, Any] | None:
        metadata: dict[str, Any] | None = rust_module().model_group_metadata(
            self._prediction_model
        )
        return metadata

    @property
    def deployment_extensions(self) -> tuple[dict[str, Any], ...]:
        return tuple(rust_module().model_deployment_extensions(self._prediction_model))

    @property
    def term_blocks(self) -> tuple[TermBlock, ...]:
        """Per-term coefficient column ranges in fitted coefficient order."""
        try:
            return term_blocks_for_model(self._prediction_model)
        except Exception as exc:
            raise map_exception(exc) from exc

    def _coefficient_state(self) -> dict[str, Any]:
        """Decode the Rust coefficient-state JSON payload."""
        try:
            state: dict[str, Any] = json.loads(
                rust_module().coefficient_state_json(self._prediction_model)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return state

    def partial_dependence(
        self,
        term: str,
        grid: Any | None = None,
        n_points: int = 100,
        level: float = 0.95,
    ) -> PartialEffect:
        """A term's partial effect, with pointwise intervals and a simultaneous band.

        For ``term`` this returns ``f_t(x) = X_t(x) β_t``, its standard error
        ``sqrt(diag(X_t V_t X_tᵀ))``, pointwise intervals and a simultaneous
        band at ``level``, all from the Rust core
        (``gam_predict::partial_effect::partial_effect``, the function the CLI's
        ``gam partial-effect`` also reads). The grid comes from the saved term
        specification: each numeric axis sweeps ``n_points`` values over its
        training range and each factor axis takes every level, as a product grid
        over the term's axes. A factor ``by=`` block holds the level its
        specification records. No other column enters ``X_t``, so the result
        never depends on a reference table.

        The curve is on the linear-predictor scale. For an additive predictor it
        equals Friedman's partial dependence up to a constant. Under a nonlinear
        link it is not an average on the response scale. For a numeric ``by=z``
        smooth the curve is the coefficient function ``f(x)``, with ``z`` held at
        one, and the term contributes ``z·f(x)`` to the predictor. mgcv's
        ``plot.gam`` draws the same curve for such a term.

        Parameters
        ----------
        term:
            Term name as it appears in :attr:`term_blocks` (e.g. ``"s(x1)"``,
            ``"te(x1, x2)"`` or a factor such as ``"group"``).
        grid:
            Optional explicit grid: 1-D for a single-axis term, or 2-D
            ``(n_points, d)`` with columns in the order of the returned ``axes``.
            A factor axis takes level codes, as ``axis_levels`` lists them.
        n_points:
            Values per numeric axis when ``grid`` is ``None``.
        level:
            Coverage level of the pointwise intervals and the simultaneous band.

        Returns
        -------
        PartialEffect
            See :class:`gamfit.results.PartialEffect`. ``surface()`` reshapes any series
            of a multi-axis term onto its product grid.
        """
        import numpy as np

        grid_matrix = None
        if grid is not None:
            grid_matrix = np.asarray(grid, dtype=float)
            if grid_matrix.ndim == 1:
                grid_matrix = grid_matrix.reshape(-1, 1)
            elif grid_matrix.ndim != 2:
                raise ValueError("partial_dependence: grid must be 1-D or 2-D")
            grid_matrix = np.ascontiguousarray(grid_matrix)
        try:
            raw = rust_module().model_partial_effect(
                self._prediction_model, term, grid_matrix, int(n_points), float(level)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return PartialEffect._from_rust(dict(raw))

    def plot_terms(
        self,
        terms: str | Sequence[str] | None = None,
        *,
        level: float = 0.95,
        n_points: int = 100,
        axes: Any | None = None,
    ) -> Any:
        """Draw each term's partial effect with matplotlib.

        A one-axis numeric term is a line with its pointwise interval and its
        simultaneous band; a factor term is one point per level with both
        intervals as error bars; a two-axis term such as ``te(x, z)`` is a
        filled contour of the surface with its standard-error contours. Every
        number comes from :meth:`partial_dependence`.

        ``terms`` defaults to every non-intercept term. ``axes`` is one
        matplotlib axes per term; by default a new figure holds them. Returns
        the list of axes drawn on.
        """
        from ._term_plot import plot_terms as _plot_terms

        return _plot_terms(self, terms, level=level, n_points=n_points, axes=axes)

    def variance_share(
        self,
        data: Any,
        term: str | None = None,
    ) -> dict[str, float] | float:
        """Term-wise variance decomposition ``cov(X_t β_t, X β) / var(X β)``.

        Computed by the Rust core (``model_variance_share``) on the rows of
        ``data``; the intercept is excluded. Cross-covariances between terms
        are split symmetrically (the Shapley allocation for a sum of terms),
        so the shares of all non-intercept terms sum to exactly 1 — unlike a
        naive ``var(f_t)/var(η)`` ratio, which drops every covariance term
        and need not sum to anything meaningful. A share can be negative (or
        exceed 1) when a term genuinely anticorrelates with the rest of the
        predictor. Returns ``{term: share}`` for every non-intercept term, or
        the scalar share when ``term`` is given.
        """
        headers, rows, _ = normalize_table(data)
        pairs = rust_module().model_variance_share(
            self._prediction_model, headers, rows, term
        )
        shares = {str(name): float(frac) for name, frac in pairs}
        if term is not None:
            if term not in shares:
                available = [b.name for b in self.term_blocks]
                raise ValueError(
                    f"variance_share: term {term!r} not found; available: {available}"
                )
            return shares[term]
        return shares

    def evidence_ratio_vs(self, other: "Model") -> float:
        """Akaike evidence ratio of this fit over ``other``.

        ``exp((other_aic - self_aic) / 2)`` on the smoothing-corrected AIC
        (``Summary.aic_corrected``) that ``gamfit.compare_models`` ranks on
        (Burnham & Anderson's relative likelihood). Returns ``> 1`` when this
        fit is better supported than ``other`` and ``< 1`` otherwise, agreeing
        with the winner ``gamfit.compare_models`` reports; ``inf`` / ``0.0``
        once the ratio leaves the float range (AIC_c gap past ~1419.6). Both
        fits must share the response family and the number of observations.

        This is **not** a Bayes factor: it integrates over no prior and must
        not be read against Jeffreys / Kass-Raftery thresholds.
        """
        # allow-list (a): FFI input validation.
        if not isinstance(other, Model):
            raise TypeError(
                f"evidence_ratio_vs expects a gamfit.Model, got {type(other).__name__}"
            )
        return rust_module().evidence_ratio(
            self._prediction_model, other._prediction_model
        )

    def _model_class_from_payload(self) -> str:
        return self._prediction_model.predict_class_name

    def _family_from_payload(self) -> str:
        return self._prediction_model.family

    def diagnose(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
    ) -> Diagnostics:
        """Score the fitted model on held-out ``data``."""
        from ._diagnose_plot import diagnose as _diagnose

        return _diagnose(self, data, y=y, interval=interval)

    def plot(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
        kind: str = "prediction",
        ax: Any | None = None,
    ) -> Any:
        """Plot the model's behaviour on ``data`` with matplotlib.

        ``kind="prediction"`` draws the fitted mean and its interval against the
        data's one feature column, for a single-feature model; for a model with
        several features use :meth:`plot_terms`, which draws each term's partial
        effect. ``"residuals"`` and ``"observed_vs_predicted"`` take any model.
        """
        from ._diagnose_plot import plot as _plot

        return _plot(self, data, y=y, interval=interval, kind=kind, ax=ax)

    def __repr__(self) -> str:
        summary = self.summary()
        parts = [
            f"formula={self.formula!r}",
            f"family_name={summary.family_name!r}",
            f"training_table_kind={self._training_table_kind!r}",
        ]
        # The objective's name is rendered in Rust (`SummaryEstimator`), so the
        # repr, `print(model)` and `gam summary` print the same words.
        if summary.convergence is not None:
            parts.append(f"estimator={summary.convergence['estimator']['text']!r}")
        return f"Model({', '.join(parts)})"

    def __str__(self) -> str:
        # Human-readable multi-line summary for ``print(model)``. The terse
        # developer one-liner stays on ``__repr__``. The rendering itself
        # lives in ``Summary.__str__`` so there is exactly one place that
        # knows how to format the summary fields. (issue #308)
        return str(self.summary())

    def _repr_html_(self) -> str:
        return self.report()


class MultinomialPrediction:
    """Multinomial class-probability prediction with delta-method uncertainty.

    Returned by :meth:`MultinomialModel.predict` with ``interval='confidence'``
    (#1101). Every array is ``(N, K)`` with columns aligned to :attr:`classes`
    (column ``j`` is class ``classes[j]``):

    * :attr:`mean` — fitted class probabilities (rows sum to 1);
    * :attr:`std_error` — delta-method per-class probability standard error
      ``SE(p_c)`` from the softmax Jacobian and the joint posterior covariance;
    * :attr:`mean_lower` / :attr:`mean_upper` — simplex-clamped band
      ``p_c ± z·SE(p_c)`` at the requested :attr:`level`.
    """

    __slots__ = ("classes", "mean", "std_error", "mean_lower", "mean_upper", "level")

    def __init__(
        self,
        *,
        classes: Sequence[Any],
        mean: NDArray[np.float64],
        std_error: NDArray[np.float64],
        mean_lower: NDArray[np.float64],
        mean_upper: NDArray[np.float64],
        level: float,
    ) -> None:
        self.classes = list(classes)
        self.mean = mean
        self.std_error = std_error
        self.mean_lower = mean_lower
        self.mean_upper = mean_upper
        self.level = float(level)

    def __repr__(self) -> str:
        n = getattr(self.mean, "shape", ["?"])[0]
        return (
            f"MultinomialPrediction(n={n}, classes={self.classes!r}, "
            f"level={self.level})"
        )


class MultinomialModel:
    """Fitted penalized multinomial-logit GAM.

    Returned by ``gamfit.fit(data, formula, family='multinomial')``. The
    underlying solver is the canonical
    ``gam::families::multinomial::fit_penalized_multinomial`` Newton solve
    against a reference-coded softmax likelihood with ``K − 1`` linear
    predictors; every (class, term) penalty carries its own smoothing
    parameter, selected jointly by REML/LAML. Class levels are the sorted label
    set of the categorical response column and the reference class is the last
    of them.

    Class names are preserved verbatim from the categorical response column,
    so :attr:`classes_` matches what ``predict`` columns line up with — no
    silent permutation.
    """

    __slots__ = ("_model_bytes", "_training_table_kind", "_metadata")

    def __init__(
        self,
        *,
        _model_bytes: bytes,
        _training_table_kind: str,
    ) -> None:
        self._model_bytes = _model_bytes
        self._training_table_kind = _training_table_kind
        # Cache the metadata dict on construction; it never changes for a
        # fitted model and downstream property accessors deserve a cheap
        # attribute read rather than an FFI round-trip per call.
        self._metadata = rust_module().multinomial_model_metadata_pyfunc(self._model_bytes)

    def __reduce__(self) -> tuple[Any, tuple[bytes]]:
        """Pickle (and ``copy.copy`` / ``copy.deepcopy``) through the saved-model
        bytes, rebuilt by :func:`gamfit.loads`, exactly as :meth:`Model.__reduce__`."""
        from ._api import loads  # local import avoids cycle

        return (loads, (self._model_bytes,))

    # ------------------------------------------------------------------ class metadata
    @property
    def classes_(self) -> list[str]:
        """Class labels in the order ``predict`` columns line up with.

        The last entry is the reference class. Matches the response level
        order recorded in the training dataset schema.
        """
        return list(self._metadata["class_levels"])

    @property
    def formula(self) -> str:
        return str(self._metadata["formula"])

    @property
    def family_name(self) -> str:
        return "multinomial"

    @property
    def training_table_kind(self) -> str:
        return self._training_table_kind

    @property
    def deviance(self) -> float:
        return float(self._metadata["deviance"])

    @property
    def n_iter_(self) -> int:
        return int(self._metadata["iterations"])

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        """Serialise the fitted multinomial model to ``path``.

        Mirrors :meth:`Model.save`, atomic and durable alike; the resulting
        file round-trips through :func:`gamfit.load`, which reconstructs a
        :class:`MultinomialModel`.
        """
        rust_module().write_saved_model_file(path, self._model_bytes)

    def dumps(self) -> bytes:
        """Return the serialised multinomial model as raw bytes."""
        return self._model_bytes

    # ------------------------------------------------------------------ predict
    @overload
    def predict(
        self, data: Any, *, interval: None = None, level: float = 0.95
    ) -> NDArray[np.float64]: ...

    @overload
    def predict(
        self, data: Any, *, interval: Literal["confidence"], level: float = 0.95
    ) -> MultinomialPrediction: ...

    def predict(
        self, data: Any, *, interval: Literal["confidence"] | None = None, level: float = 0.95
    ) -> NDArray[np.float64] | MultinomialPrediction:
        """Predict class probabilities for new rows.

        With ``interval=None`` (default) returns an ``(N, K)`` numpy array whose
        columns are aligned with :attr:`classes_` (column ``j`` is
        ``P(Y = self.classes_[j] | x)``); rows sum to 1.

        With ``interval='confidence'`` returns a
        :class:`MultinomialPrediction` carrying the same ``(N, K)`` ``mean``
        probabilities plus delta-method per-class probability standard errors
        (``std_error``) and simplex-clamped confidence bounds (``mean_lower`` /
        ``mean_upper``) at the requested ``level``. The bounds come from the
        softmax-Jacobian delta method ``p_c ± z·SE(p_c)`` against the joint
        Laplace posterior covariance ``H⁻¹`` (#1101). Available only for
        REML-fitted models (which carry the covariance); a model without stored
        covariance raises.
        """
        headers, rows, _ = normalize_table(data)
        if interval is None:
            try:
                probs = rust_module().predict_multinomial_formula_pyfunc(
                    self._model_bytes, headers, rows
                )
            except Exception as exc:
                raise map_exception(exc) from exc
            return probs
        if interval != "confidence":
            raise ValueError(
                f"MultinomialModel.predict: interval={interval!r} is not supported; "
                "use None or 'confidence'"
            )
        if not (0.0 < level < 1.0):
            raise ValueError(f"level must be in (0, 1), got {level}")
        # Pass the LEVEL, not a z-score. The pyfunction's signature is
        # `(model_bytes, headers, rows, level = 0.95)` and the Rust side does its
        # own `standard_normal_quantile(0.5 + 0.5 * level)`. Converting here sent
        # z = 1.96 in as a "level", so the quantile was taken at 0.5 + 0.98.
        try:
            out = rust_module().predict_multinomial_intervals_pyfunc(
                self._model_bytes, headers, rows, level
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        if out.get("prob_se") is None:
            raise ValueError(
                "MultinomialModel.predict(interval='confidence'): this model carries no "
                "posterior covariance (refit with the current REML path to enable intervals)"
            )
        return MultinomialPrediction(
            classes=self.classes_,
            mean=out["probs"],
            std_error=out["prob_se"],
            mean_lower=out["mean_lower"],
            mean_upper=out["mean_upper"],
            level=level,
        )

    def posterior_predict(
        self,
        data: Any,
        *,
        n_draws: int = 100,
        seed: int = 0,
    ) -> Any:
        """Draw posterior-predictive replicate class labels at ``data`` (#1101).

        Each of the ``n_draws`` rows is a fresh synthetic class-label vector
        drawn from the fitted predictive distribution — every row's label is
        sampled from ``Categorical(softmax(X·beta_hat))`` (the plug-in
        categorical observation noise around the fitted mean). This is the
        multinomial analogue of :meth:`Model.sample_replicates` and the engine
        for posterior-predictive checks and synthetic-data generation.

        Parameters
        ----------
        data : table-like
            New rows in any format accepted by :meth:`predict`. Must cover every
            predictor referenced by the fitted formula (the response column, if
            present, is ignored).
        n_draws : int, default 100
            Number of replicate label vectors to draw.
        seed : int, default 0
            Seed for the deterministic draw stream; the same
            ``(data, n_draws, seed)`` reproduce bit-identically.

        Returns
        -------
        numpy.ndarray
            An ``(n_draws, n_rows)`` object array of class labels (strings from
            :attr:`classes_`).
        """
        import numpy as np

        n_draws = int(n_draws)
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        headers, rows, _ = normalize_table(data)
        try:
            out = rust_module().posterior_predict_multinomial_pyfunc(
                self._model_bytes, headers, rows, n_draws, int(seed)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        idx = np.asarray(out["draws"])
        levels = list(out["class_levels"])
        labels = np.empty(idx.shape, dtype=object)
        for c, name in enumerate(levels):
            labels[idx == c] = name
        return labels

    def smooth_significance(self) -> list[dict[str, Any]]:
        """Variance-component score test of every smooth term (#1101, #3569).

        One row per ``(active class, smooth term)`` testing the term's effect
        on that class's log-odds against the reference class, plus, when there
        are three or more classes, one ``contrast == "joint"`` row per term
        testing that the term moves no class probability; that row does not
        depend on which class is the reference. Keys: ``contrast``
        (``"class"`` or ``"joint"``), ``class`` (``None`` on a joint row),
        ``term``, ``edf``, ``ref_df``, ``statistic``, ``p_value`` and
        ``p_value_unavailable``, the reason a row has no test (its numeric
        fields are then ``None``). Empty when the model has no smooth terms.
        """
        try:
            return list(
                rust_module().multinomial_smooth_significance_pyfunc(self._model_bytes)
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    # ------------------------------------------------------------------ summary
    def summary(self) -> str:
        """Human-readable summary covering convergence, classes, per-class λ and edf.

        Rendered by the Rust ``MultinomialSavedModel::summary_text``: the
        selected per-class REML λ, the per-class hat-matrix trace (effective
        degrees of freedom) when the inference block is available, the
        separation decision that fixed the published estimand, and the
        score-test smooth-significance table.
        """
        try:
            return str(rust_module().multinomial_summary_text_pyfunc(self._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc

    # ------------------------------------------------------------------ identity / repr
    def __repr__(self) -> str:
        return (
            f"MultinomialModel(formula={self.formula!r}, "
            f"classes={self.classes_!r})"
        )

    def __str__(self) -> str:
        return self.summary()


__all__ = [
    "AffineDesign",
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "MultinomialModel",
    "MultinomialPrediction",
    "SurvivalPrediction",
    "TermBlock",
    "competing_risks_cif",
]
