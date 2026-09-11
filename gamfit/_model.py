"""Public :class:`Model` shell.

The numeric work all lives in the Rust core: this module marshals
arguments through the FFI, hands payloads off to ``_survival`` /
``_diagnose_plot`` helpers, and exposes Pythonic properties.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._sampling import PosteriorSamples
from ._schema import SchemaCheck
from ._summary import Summary
from ._predict_shape import shape_predict_response
from ._survival import (
    CompetingRisksCIF,
    CompetingRisksPrediction,
    SurvivalPrediction,
    TermBlock,
    _MARGINAL_SLOPE_MODEL_CLASSES,
    _SURVIVAL_MODEL_CLASSES,
    _TRANSFORMATION_NORMAL_MODEL_CLASSES,
    competing_risks_cif,
    extract_row_ids,
    term_blocks_for_model,
)
from ._tables import (
    normalize_table,
    response_column_name,
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
    metadata, and :meth:`save` / :meth:`dumps` for persistence.
    """

    __slots__ = ("_model_bytes", "_prediction_model", "_training_table_kind")

    def __init__(self, *, _model_bytes: bytes, _training_table_kind: str) -> None:
        self._model_bytes = _model_bytes
        try:
            self._prediction_model = rust_module().compile_model(_model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        self._training_table_kind = _training_table_kind

    def predict(
        self,
        data: Any,
        *,
        interval: float | str | None = None,
        conformal_level: float = 0.9,
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
            cover every predictor referenced by the fitted formula.
        interval : float, "conformal", "full_conformal", or None, default None
            Single uncertainty knob. ``None`` returns the point prediction(s)
            only. A float in ``(0, 1)`` (e.g. ``0.95``) requests the full
            uncertainty decomposition at that pointwise coverage; the output
            gains ``posterior_mean_standard_error``,
            ``posterior_mean_lower``, and ``posterior_mean_upper`` columns
            alongside ``linear_predictor_plugin`` / ``mean_plugin`` /
            ``posterior_mean``. On survival models it
            also produces per-cell hazard / survival SEs. Issue #342
            collapsed the previous overlapping ``with_uncertainty`` boolean
            into this single flag (use ``interval=0.95`` for the SE-only
            case).

            Pass ``interval="conformal"`` to use distribution-free jackknife+
            prediction intervals (Barber et al. 2021) — no held-out
            calibration fold is required. The interval *targets*
            ``conformal_level`` (default ``0.9``) marginal coverage; the
            finite-sample guarantee the theorem certifies at this setting is
            the weaker ``2 * conformal_level - 1`` (coverage >= 1 - 2*alpha at
            alpha = 1 - level). This path requires a Gaussian-identity model
            fitted without prior weights, offsets, or a link wiggle; use
            :meth:`predict_conformal` for split-conformal intervals on other
            families.

            Pass ``interval="full_conformal"`` for the full-conformal set at
            the fitted (frozen) smoothing parameters (#942 Layer 1): every
            observation is used for both fitting and calibration, and the set
            is exact *given* the frozen penalty, computed from one Cholesky per
            test point with zero refits. Because the smoothing parameters were
            selected from all training responses, the distribution-free
            finite-sample ``conformal_level`` coverage theorem applies only
            where the per-row ``frozen_rho_certified`` output column is 1.0
            (the Layer-3 certificate that freezing the global smoothing
            parameter matches the honest ρ-re-selecting set, under a
            grid-checked Lipschitz assumption); rows with 0.0 carry no
            finite-sample guarantee. Same eligibility as ``"conformal"``;
            ``posterior_mean_lower`` / ``posterior_mean_upper`` report the outer envelope of the
            (possibly multi-interval) set.
        conformal_level : float, default 0.9
            Target marginal coverage in ``(0, 1)`` when ``interval="conformal"``
            or ``interval="full_conformal"``
            (e.g. ``0.95`` for a 95% interval). Ignored when ``interval`` is a
            float or ``None``.
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
              adds ``posterior_mean_standard_error`` plus
              ``posterior_mean_lower`` / ``posterior_mean_upper``.
              When the requested table container is ``"dict"``, the return is
              a ``PredictionResult``: it supports normal mapping access
              (``pred["posterior_mean"]``) and column attributes
              (``pred.posterior_mean``, ``pred.posterior_mean_standard_error``,
              ``pred.posterior_mean_lower``, ``pred.posterior_mean_upper``).
            * Bernoulli marginal-slope: a 1-D ``ndarray`` of probabilities.
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
        required = rust_module().required_model_columns(self._model_bytes, False)
        if id_column is not None:
            required = sorted(set(required) | {id_column})
        headers, rows, table_kind = normalize_table(data, required_columns=required)
        row_ids = extract_row_ids(headers, rows, id_column)
        # #1054: interval='conformal' routes to the exact Gaussian jackknife+
        # path (no held-out fold needed; targets conformal_level coverage with
        # the finite-sample floor 2*level-1 — see the Rust route for the
        # calibration decision, #1546). The returned JSON has the same column
        # schema as the model-based predict path so shape_predict_response is
        # unchanged.
        if interval == "conformal":
            try:
                raw = rust_module().predict_table_jackknife_plus(
                    self._model_bytes, headers, rows, conformal_level
                )
            except Exception as exc:
                raise map_exception(exc) from exc
            return shape_predict_response(
                raw,
                headers=headers,
                rows=rows,
                table_kind=table_kind,
                training_table_kind=self._training_table_kind,
                interval=conformal_level,
                return_type=return_type,
                id_column=id_column,
                row_ids=row_ids,
                restore=restore_output_table,
            )
        # #1098: interval='full_conformal' routes to the Gaussian
        # full-conformal set at frozen smoothing parameters (no held-out fold;
        # exact given Sλ; the finite-sample ≥conformal_level theorem holds per
        # row only where frozen_rho_certified=1 — see the docstring; #942
        # Layer 1). One Cholesky per test point, zero refits. The returned
        # JSON carries the same column schema plus that certificate column.
        if interval == "full_conformal":
            try:
                raw = rust_module().predict_table_full_conformal(
                    self._model_bytes, headers, rows, conformal_level
                )
            except Exception as exc:
                raise map_exception(exc) from exc
            return shape_predict_response(
                raw,
                headers=headers,
                rows=rows,
                table_kind=table_kind,
                training_table_kind=self._training_table_kind,
                interval=conformal_level,
                return_type=return_type,
                id_column=id_column,
                row_ids=row_ids,
                restore=restore_output_table,
            )
        try:
            raw = rust_module().predict_table(
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
            raw,
            headers=headers,
            rows=rows,
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

        This method is defined only for conditional transformation-normal
        models and requires both the fitted covariates and the observed
        response column.  It is intentionally distinct from :meth:`predict`,
        whose CTM point estimate is the response-scale conditional mean
        ``E[Y|x]`` and therefore does not consume an observed response.

        The returned score is the generated regressor used by a downstream
        marginal-slope model.  By default this is a one-dimensional NumPy
        array; ``return_type=`` or ``id_column=`` requests a one-column table
        named ``score`` (plus the requested identifier).
        """
        required = rust_module().required_model_columns(self._model_bytes, True)
        if id_column is not None:
            required = sorted(set(required) | {id_column})
        headers, rows, table_kind = normalize_table(data, required_columns=required)
        row_ids = extract_row_ids(headers, rows, id_column)
        try:
            scores = rust_module().transformation_score_table(
                self._model_bytes, headers, rows
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
                self._model_bytes,
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
            # trailing axis.
            import numpy as np

            return np.asarray(result).reshape(-1)
        return result

    def predict_conformal(
        self,
        data: Any,
        *,
        calibration: Any,
        conformal_level: float,
        covariance_mode: str | None = None,
        observation_interval: bool = False,
        return_type: str | None = None,
        id_column: str | None = None,
    ) -> Any:
        """Predict with distribution-free conformal prediction intervals.

        Runs the standard predictor on ``data``, then REPLACES the
        response-scale ``posterior_mean_lower`` / ``posterior_mean_upper`` columns with the
        split-conformal interval ``mu_hat(x) +/- q_hat * s(x)`` calibrated at
        ``conformal_level`` from the held-out ``calibration`` fold. The
        resulting interval carries finite-sample marginal coverage
        ``>= conformal_level`` regardless of model misspecification.

        Parameters
        ----------
        data : table-like
            Test inputs to predict, in any format accepted by
            :meth:`predict`. Must cover every predictor in the formula.
        calibration : table-like
            Held-out *labeled* calibration fold (not used in fitting). Must
            contain the response column in addition to the predictors; the
            conformal multiplier ``q_hat`` is computed from this fold's plain
            held-out residuals ``y_cal - mu_hat(x_cal)`` (normalized by the
            response-scale SE). The fold may be of any size, independent of the
            training set — no leave-one-out correction is applied because a
            held-out fold is already independent of the fitted model.
        conformal_level : float
            Target marginal coverage in ``(0, 1)`` (e.g. ``0.9``).
        covariance_mode : {"conditional", "smoothing"}, optional
            Covariance source for the per-point scale ``s(x)``; see
            :meth:`predict`.
        observation_interval : bool, default False
            Also emit ``observation_lower`` / ``observation_upper`` columns;
            see :meth:`predict`.
        return_type, id_column
            As in :meth:`predict`.

        Returns
        -------
        table
            A table with ``linear_predictor_plugin``, ``mean_plugin``,
            ``posterior_mean``, ``posterior_mean_standard_error``, and the
            conformal ``posterior_mean_lower`` / ``posterior_mean_upper``
            columns. Currently supported for standard GAM models only.
        """
        headers, rows, table_kind = normalize_table(data)
        cal_headers, cal_rows, _ = normalize_table(calibration)
        row_ids = extract_row_ids(headers, rows, id_column)
        opts_json = rust_module().build_model_predict_payload_json(
            self._model_bytes,
            headers,
            rows,
            conformal_level,
            covariance_mode,
            observation_interval,
        )
        try:
            raw = rust_module().predict_table_conformal(
                self._model_bytes,
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
            raw,
            headers=headers,
            rows=rows,
            table_kind=table_kind,
            training_table_kind=self._training_table_kind,
            interval=conformal_level,
            return_type=return_type,
            id_column=id_column,
            row_ids=row_ids,
            restore=restore_output_table,
        )

    def summary(self) -> Summary:
        """Return the model summary (coefficients, family, deviance, REML score)."""
        try:
            payload = rust_module().summary_payload_from_model(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        return Summary.from_dict(payload)

    def smoothing_parameters(self) -> dict[int, float]:
        """Return fitted smoothing/precision parameters by penalty index."""
        return dict(rust_module().smoothing_parameters_from_model(self._model_bytes))

    def check(self, data: Any) -> SchemaCheck:
        """Validate ``data`` against the model's training schema."""
        headers, rows, _ = normalize_table(data)
        try:
            payload = rust_module().check_payload_from_model(self._model_bytes, headers, rows)
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
                self._model_bytes, headers, rows, level
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        payload = json.loads(raw)
        return list(payload.get("curvature_terms", []))

    def smooth_significance(self, data: Any) -> list[dict[str, Any]]:
        """Per-term likelihood-ratio significance for every penalized smooth (#1063).

        :meth:`summary` reports Wood's rank-truncated *Wald* statistic
        :math:`T = \\hat\\beta'\\hat\\Sigma^- \\hat\\beta`. The exact Lawley /
        Bartlett factor corrects the *likelihood-ratio* statistic, and under
        penalization the Wald form is already a weighted :math:`\\chi^2` whose
        second-order mean is not :math:`d + \\Delta\\varepsilon`, so dividing
        :math:`T` by the LR factor would correct the wrong statistic. This method
        instead computes a genuine per-term LR statistic
        :math:`W = 2(\\ell_{\\text{full}} - \\ell_{\\text{null}})` by a
        constrained refit dropping the smooth, then Bartlett-corrects *that*:
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

        For each penalized (shape-unconstrained) smooth term it returns
        ``statistic_lr`` (the raw :math:`W`), ``ref_df`` (the null mean
        :math:`d = \\sum_j w_j`, which is what the Bartlett factor is
        denominated in — *not* a chi-square degrees of freedom),
        ``reference_weights``/``reference_source``, ``reference_chi_square_df``
        :math:`\\nu` and ``reference_scale`` :math:`g` (the two-moment summary),
        ``reference_residual_df``/``reference_deterministic_offset`` (the
        estimated-scale channel above, ``None`` off the profiled Gaussian),
        ``bartlett_factor``
        :math:`c`, ``statistic_corrected`` :math:`W^*`, ``p_value_uncorrected``,
        ``p_value_corrected`` (the magic-by-default value), ``material`` (the
        n-too-small-here diagnostic — ``True`` when the correction moves the
        Bartlett factor or the p-value by more than 10%), and
        ``correction_provenance`` — ``"lawley_lr"`` when the family carries
        closed-form cumulant jets (gaussian / poisson / binomial / gamma) and the
        null refit converged, else ``"none"`` (the uncorrected reference stands,
        never weakened).

        Needs the training ``data`` for the per-term null refits, exactly as
        :meth:`curvature` does. Returns an empty list when the model has no
        penalized smooth term.
        """
        headers, rows, _ = normalize_table(data)
        try:
            raw = rust_module().smooth_term_lr_inference_json(
                self._model_bytes, headers, rows
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        payload = json.loads(raw)
        return list(payload.get("smooth_terms", []))

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
          ``"design_gram_unavailable"``, ...). ``p_value`` is present exactly
          when a test ran, so "adequate" and "not measured" are never
          confusable.

        Method
        ------
        The alternative is a Duchon kernel at data-driven centers over the
        term's standardized covariates, orthogonalized against the fitted design
        in the fit's own IRLS weight metric. The statistic is
        :math:`T = U^{\top} V^{-} U / \hat\varphi` with
        :math:`U = \tilde Z^{\top} s` and :math:`V = \tilde Z^{\top} W \tilde Z`,
        referred to :math:`\chi^2_r` (known dispersion) or :math:`F(r, \nu)`
        (estimated).

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
            raw = rust_module().basis_adequacy_json(self._model_bytes, headers, rows)
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
                self._model_bytes, headers, rows, json.dumps(spec)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return dict(json.loads(raw))

    def report(self, path: str | Path | None = None) -> str:
        """Generate a standalone HTML report of the fitted model."""
        try:
            html = rust_module().report_html(self._model_bytes)
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
        warmup: int | None = None,
        chains: int | None = None,
        target_accept: float | None = None,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Draw from the model's posterior with NUTS."""
        headers, rows, _ = normalize_table(data)
        try:
            ffi = rust_module()
            options_json = ffi.build_sample_payload_json(
                samples, warmup, chains, target_accept, seed
            )
            payload = ffi.sample_table(
                self._model_bytes,
                headers,
                rows,
                options_json,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return PosteriorSamples.from_ffi_payload(payload, model_bytes=self._model_bytes)

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
                self._model_bytes, headers, rows, n_draws, int(seed)
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
                        self._model_bytes,
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
            payload = rust_module().affine_design_table(self._model_bytes, headers, rows)
            return _affine_design_from_payload(payload)
        except Exception as exc:
            raise map_exception(exc) from exc

    def design_matrix_array(self, X: Any) -> AffineDesign:
        """Exact fitted affine predictor design for a numeric feature matrix."""
        try:
            rust = rust_module()
            return _affine_design_from_payload(
                rust.affine_design_array(
                    self._model_bytes,
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
                self._model_bytes, request_json
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
        """Serialise the fitted model to ``path``."""
        Path(path).write_bytes(self._model_bytes)

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
                rust.extend_model_with_group(self._model_bytes, payload_json)
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
        return rust_module().required_saved_model_payload_string(
            self._model_bytes, "formula"
        )

    @property
    def family_name(self) -> str:
        return self.summary().family_name

    @property
    def notes(self) -> list[str]:
        """Inference advisories recorded while this model was fit.

        Each note is an mgcv-style advisory that the fitted model differs from
        what was literally requested — e.g. ``"... basis reduced from k=10 to
        k=3 to match the covariate's 3 distinct value(s)"`` when a cubic-
        regression marginal is capped to the data support, or a basis-
        degradation note when a low-cardinality covariate cannot support the
        requested smooth. :func:`gamfit.fit` also emits these as
        :class:`gamfit.GamInferenceWarning` at fit time; this property lets a
        caller inspect them after the fact (or after loading a saved model).
        Empty when the fit used exactly the requested configuration.
        """
        return list(rust_module().inference_notes_from_model(self._model_bytes))

    @property
    def used_device(self) -> bool:
        return rust_module().required_saved_model_payload_string(
            self._model_bytes, "used_device"
        ) == "true"

    @property
    def model_class(self) -> str:
        return self._model_class_from_payload()

    @property
    def is_survival(self) -> bool:
        return self.model_class in _SURVIVAL_MODEL_CLASSES

    @property
    def is_marginal_slope(self) -> bool:
        return self.model_class in _MARGINAL_SLOPE_MODEL_CLASSES

    @property
    def is_transformation_normal(self) -> bool:
        return self.model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES

    @property
    def response_name(self) -> str | None:
        return response_column_name(self.formula)

    @property
    def training_table_kind(self) -> str:
        return self._training_table_kind

    @property
    def group_metadata(self) -> dict[str, Any] | None:
        return rust_module().model_group_metadata(self._model_bytes)

    @property
    def deployment_extensions(self) -> tuple[dict[str, Any], ...]:
        return tuple(rust_module().model_deployment_extensions(self._model_bytes))

    @property
    def term_blocks(self) -> tuple[TermBlock, ...]:
        """Per-term coefficient column ranges in fitted coefficient order."""
        try:
            return term_blocks_for_model(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc

    def _coefficient_state(self) -> dict[str, Any]:
        """Decode the Rust coefficient-state JSON payload."""
        try:
            return json.loads(rust_module().coefficient_state_json(self._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc

    def partial_dependence(
        self,
        term: str,
        data: Any,
        grid: Any | None = None,
        n_points: int = 100,
    ) -> dict[str, Any]:
        """Per-term partial dependence with delta-method SE.

        Analogue of mgcv's ``plot.gam()`` per-term plot: for ``term`` this
        returns ``f_t(x) = X_t(x) β_t`` and the delta-method standard error
        ``sqrt(diag(X_t V_t X_tᵀ))``. The evaluation ``f_t`` and its SE are
        computed by the Rust core (``model_partial_dependence``); Python only
        constructs the evaluation grid table.

        Parameters
        ----------
        term:
            Term name as it appears in :attr:`term_blocks` (e.g. ``"s(x1)"``).
        data:
            Reference table; non-``term`` columns supply template values for
            the constructed grid rows.
        grid:
            Optional explicit grid (1-D for a single-axis smooth, or 2-D
            ``(n_points, d)`` for a multi-axis smooth). When ``None`` a
            1-D linspace over the term's training range is used.
        n_points:
            Grid resolution when ``grid`` is ``None``.

        Returns
        -------
        dict
            ``{"grid": array, "predicted": array, "standard_error": array,
            "covariance_source": str}``. The standard errors are priced off
            the covariance the fit publishes (the same one ``summary()``
            reports), and ``covariance_source`` names it: ``"smoothing-corrected"``
            whenever the fit carries that matrix, otherwise ``"conditional"``.
        """
        import numpy as np

        block = next((b for b in self.term_blocks if b.name == term), None)
        if block is None:
            available = [b.name for b in self.term_blocks]
            raise ValueError(
                f"partial_dependence: term {term!r} not found; available: {available}"
            )
        state = self._coefficient_state()
        schema_cols = list((state.get("schema") or {}).get("columns") or [])
        ranges = state.get("training_feature_ranges") or []
        names = [str(c.get("name")) for c in schema_cols]

        template: dict[str, Any] = {}
        # A partial-dependence template is user-facing table data, not an FFI
        # payload.  Reading it back from ``normalize_table`` consumed the
        # encoded representation: categorical cells retained the native
        # sentinel/type encoding and were then encoded a second time below,
        # producing a level the fitted schema had never seen (#2780).  Keep the
        # raw cells until the completed grid crosses the native boundary once.
        data_columns, _ = table_columns(data)
        if not data_columns or not next(iter(data_columns.values()), []):
            raise ValueError("table data cannot be empty")
        template.update({name: values[0] for name, values in data_columns.items()})
        for idx, col in enumerate(schema_cols):
            name = str(col.get("name"))
            if name in template:
                continue
            if col.get("kind") == "categorical":
                levels = col.get("levels") or ["0"]
                template[name] = str(levels[0])
            elif idx < len(ranges):
                lo, hi = map(float, ranges[idx])
                template[name] = str(0.5 * (lo + hi))
            else:
                template[name] = "0"

        # For a ``by=``-factor smooth block, the block name carries the factor
        # level as a ``:by=<var>[<level>]`` suffix (e.g. ``s(x, by=g):by=g[a]``).
        # Pin the grouping factor at the block's OWN level so that partial
        # dependence reflects a fixed model property rather than the arbitrary
        # value found in the first row of the passed frame (issue #2076).
        by_marker = ":by="
        by_pos = term.rfind(by_marker)
        if by_pos != -1 and term.endswith("]"):
            by_suffix = term[by_pos + len(by_marker) :]
            bracket = by_suffix.find("[")
            if bracket != -1:
                by_var = by_suffix[:bracket]
                by_level = by_suffix[bracket + 1 : -1]
                if by_var in names:
                    by_col = schema_cols[names.index(by_var)]
                    by_levels = [str(lvl) for lvl in (by_col.get("levels") or [])]
                    template[by_var] = next(
                        (lvl for lvl in by_levels if lvl == by_level), by_level
                    )

        term_args: tuple[str, ...] = ()
        if "(" in term and ")" in term:
            inside = term[term.index("(") + 1 : term.rindex(")")]
            term_args = tuple(
                a.strip() for a in inside.split(",") if a.strip() and a.strip() in names
            )

        if grid is None:
            if len(term_args) != 1:
                raise ValueError(
                    "partial_dependence: cannot infer a 1D sweep axis from term "
                    f"{term!r} (axes inferred: {term_args!r}); pass an explicit "
                    "`grid=` array. Multi-dimensional smooths always require an "
                    "explicit grid."
                )
            term_argument = term_args[0]
            col_idx = names.index(term_argument)
            lo, hi = (map(float, ranges[col_idx]) if col_idx < len(ranges) else (0.0, 1.0))
            lo, hi = float(lo), float(hi)
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
                lo, hi = 0.0, 1.0
            grid_arr = np.linspace(lo, hi, int(n_points))
            sweep_columns: tuple[str, ...] = (term_argument,)
            grid_matrix = grid_arr.reshape(-1, 1)
            grid_out: Any = grid_arr
        else:
            grid_matrix = np.asarray(grid, dtype=float)
            if grid_matrix.ndim == 1:
                if len(term_args) != 1:
                    raise ValueError(
                        "partial_dependence: a 1-D grid requires a single-axis "
                        f"term; {term!r} has axes {term_args!r}. Pass a 2-D grid."
                    )
                sweep_columns = (term_args[0],)
                grid_matrix = grid_matrix.reshape(-1, 1)
                grid_out = grid_matrix.reshape(-1)
            elif grid_matrix.ndim == 2:
                if len(term_args) != grid_matrix.shape[1]:
                    raise ValueError(
                        "partial_dependence: explicit grid shape "
                        f"{grid_matrix.shape} does not match term axes {term_args!r}"
                    )
                sweep_columns = term_args
                grid_out = grid_matrix
            else:
                raise ValueError("partial_dependence: grid must be 1-D or 2-D")

        # The FFI takes an encoded table (#2318 boundary hardening), so route
        # the constructed grid through the same normalize_table encode path
        # every other table-crossing call uses — a raw list of string rows is
        # rejected at the boundary.
        headers = list(template.keys())
        categorical_names = {
            str(col.get("name")) for col in schema_cols if col.get("kind") == "categorical"
        }
        columns: dict[str, list[Any]] = {h: [] for h in headers}
        for row_vals in grid_matrix:
            row = dict(template)
            for col_name, value in zip(sweep_columns, row_vals, strict=False):
                row[col_name] = str(float(value))
            for h in headers:
                columns[h].append(
                    row[h] if h in categorical_names else float(row[h])
                )
        enc_headers, enc_rows, _ = normalize_table(columns)
        predicted, se, covariance_source = rust_module().model_partial_dependence(
            self._model_bytes, term, enc_headers, enc_rows
        )
        return {
            "grid": grid_out,
            "predicted": np.asarray(predicted, dtype=float),
            "standard_error": np.asarray(se, dtype=float),
            "covariance_source": str(covariance_source),
        }

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
            self._model_bytes, headers, rows, term
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

    @property
    def evidence(self) -> float:
        """Model-selection cost for this fit, on the same rank scale used by
        ``gamfit.compare_models`` to pick its winner: the Occam-penalised
        conditional AIC (``-2*loglik + 2*edf``). Both the ordinary
        log-likelihood and effective degrees of freedom are required; raw REML /
        LAML is a different estimand and is never used as a fallback (#2079).
        It is a *cost*, so **lower is better** -- the model with the smaller
        ``evidence`` is the better-supported one, agreeing with the winner
        reported by ``gamfit.compare_models``. Use :meth:`evidence_ratio_vs` or
        ``gamfit.compare_models`` for a direct comparison. (The raw,
        un-penalised REML/LAML evidence headline remains available as
        ``Summary.reml_score`` / the ``score_table`` column.)
        """
        return float(rust_module().model_evidence(self._model_bytes))

    def evidence_ratio_vs(self, other: "Model") -> float:
        """Akaike evidence ratio of this fit over ``other``.

        ``exp(-(self.evidence - other.evidence) / 2)``: the relative likelihood
        of the two fits under the conditional-AIC criterion that
        ``gamfit.compare_models`` ranks on (Burnham & Anderson). Returns ``> 1``
        when this fit is better supported than ``other`` (i.e. has the lower
        :attr:`evidence` cost) and ``< 1`` otherwise, agreeing with the winner
        reported by ``gamfit.compare_models``.

        This is **not** a Bayes factor. A Bayes factor is a ratio of
        prior-integrated marginal likelihoods; this quantity integrates over no
        prior and must not be read against Jeffreys / Kass-Raftery thresholds.
        (The raw REML/LAML headline in the ``score_table`` of
        ``gamfit.compare_models`` is the Laplace-approximate marginal-likelihood
        diagnostic, kept on its own labelled scale.)
        """
        # allow-list (a): FFI input validation.
        if not isinstance(other, Model):
            raise TypeError(
                f"evidence_ratio_vs expects a gamfit.Model, got {type(other).__name__}"
            )
        log_ratio = rust_module().log_evidence_ratio(
            self._model_bytes, other._model_bytes
        )
        return math.exp(log_ratio)

    def bayes_factor_vs(self, other: "Model") -> float:
        """Deprecated spelling of :meth:`evidence_ratio_vs`.

        The value was never a Bayes factor -- it is the Akaike evidence ratio
        ``exp(-dAIC/2)`` -- so the name misdescribed it. Emits a
        ``DeprecationWarning`` and returns exactly what
        :meth:`evidence_ratio_vs` returns.
        """
        warnings.warn(
            "Model.bayes_factor_vs is the Akaike evidence ratio exp(-dAIC/2), not a "
            "Bayes factor; use Model.evidence_ratio_vs",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.evidence_ratio_vs(other)

    def _model_class_from_payload(self) -> str:
        return rust_module().saved_model_predict_class_name(self._model_bytes)

    def _family_from_payload(self) -> str:
        return rust_module().required_saved_model_payload_string(
            self._model_bytes, "family"
        )

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
        x: str | None = None,
        y: str | None = None,
        interval: float | None = 0.95,
        kind: str = "prediction",
        ax: Any | None = None,
    ) -> Any:
        """Plot the model's behaviour on ``data`` with matplotlib."""
        from ._diagnose_plot import plot as _plot

        return _plot(self, data, x=x, y=y, interval=interval, kind=kind, ax=ax)

    def __repr__(self) -> str:
        parts = [
            f"formula={self.formula!r}",
            f"family_name={self.family_name!r}",
            f"training_table_kind={self._training_table_kind!r}",
        ]
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

    def __init__(self, *, classes, mean, std_error, mean_lower, mean_upper, level):
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
    against a reference-coded softmax likelihood; the reference class is the
    last level recorded in the dataset schema (i.e. order of first appearance
    in the training table, which is stable across runs).

    Class names are preserved verbatim from the categorical response column,
    so :attr:`classes_` matches what ``predict`` columns line up with — no
    silent permutation.

    Slice A of issue #328: a single uniform smoothing parameter is shared
    across every penalty block and every active class. REML / LAML λ
    selection lands in the follow-up slice.
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

        Mirrors :meth:`Model.save`; the resulting file round-trips through
        :func:`gamfit.load`, which reconstructs a :class:`MultinomialModel`.
        """
        Path(path).write_bytes(self._model_bytes)

    def dumps(self) -> bytes:
        """Return the serialised multinomial model as raw bytes."""
        return self._model_bytes

    # ------------------------------------------------------------------ predict
    def predict(self, data: Any, *, interval: str | None = None, level: float = 0.95) -> Any:
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

    def std_error(self, data: Any) -> Any:
        """Delta-method per-class probability standard errors for new rows.

        Returns an ``(N, K)`` numpy array column-aligned with :attr:`classes_`.
        Equivalent to ``predict(data, interval='confidence').std_error``.
        """
        return self.predict(data, interval="confidence").std_error

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

    def smooth_significance(self) -> list[dict]:
        """Wood rank-truncated Wald smooth-term significance table (#1101).

        One row per ``(active class, smooth term)`` with keys ``class``,
        ``term``, ``edf``, ``ref_df``, ``statistic``, ``p_value`` — the same
        kernel the scalar :meth:`Model.summary` smooth-term p-values use. Empty
        when the model has no smooth terms or no stored covariance.
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

        REML-driven path: the Rust core selects per-active-class λ via the
        outer Laplace/REML loop, so this method reports both the selected
        λ_a and the per-class hat-matrix trace (effective degrees of
        freedom) when the inference block is available.
        """
        meta = self._metadata
        p = int(meta["p_per_class"])
        m = int(meta["n_active_classes"])
        levels = list(meta["class_levels"])
        ref = int(meta["reference_class_index"])
        lambdas = list(meta["lambdas"])
        lambdas_per_block = list(meta["lambdas_per_block"])
        # Per-penalty-component λ labels, parallel to a single class block's λ
        # slice (#1544). The Marra–Wood double penalty (and tensor/operator
        # smooths) emit more than one penalty component — hence more than one λ —
        # per smooth term, so these are NOT 1:1 with `term_labels`: a single
        # `s(x)` term yields a primary wiggliness λ and a null-space shrinkage λ,
        # each carrying its own label here. Pairing λ with these component labels
        # (rather than assuming one λ per term) is what keeps every λ in the
        # summary instead of silently truncating the null-space penalties.
        lambda_labels = list(meta["lambda_labels"])
        edf_per_class = meta.get("edf_per_class")

        lines = [
            f"MultinomialModel formula: {meta['formula']}",
            f"  classes: {levels}  (reference = {levels[ref]!r})",
            f"  active classes (K-1): {m}",
            f"  coefficients per class (P): {p}",
            f"  total coefficients: {p * m}",
            f"  iterations: {int(meta['iterations'])}",
            f"  deviance: {float(meta['deviance']):.6g}",
            f"  penalized -log L: {float(meta['penalized_neg_log_likelihood']):.6g}",
        ]
        # Which estimand this model publishes (#2612). The Jeffreys/Firth proper
        # prior is engaged automatically on separation evidence — `firth=` is
        # rejected on this family for exactly that reason — and armed, the
        # coefficients carry an O(1/n) pull toward the uniform simplex 1/K. A
        # reader comparing two fits, or scoring calibration, needs to know which
        # objective produced the numbers.
        #
        # Three states, not two. The FFI exports the key unconditionally
        # INCLUDING the `None` case precisely so that "the prior was disarmed"
        # and "this payload predates the field" stay distinguishable here; a
        # renderer that collapsed a missing key into "disarmed" would assert an
        # estimand the model never claimed.
        if "separation_evidence" not in meta:
            lines.append(
                "  separation: not recorded (model saved before the fit published "
                "its arming decision)"
            )
        elif meta["separation_evidence"] is None:
            lines.append(
                "  separation: none detected; Jeffreys/Firth prior disarmed "
                "(unbiased penalized-REML mode)"
            )
        else:
            lines.append(
                "  separation: Jeffreys/Firth proper prior ARMED (coefficients carry "
                f"the Firth bias correction) — {meta['separation_evidence']}"
            )
        # Per-class slope-norm + REML λ + hat-matrix trace rollup. Coefficients
        # are stored in row-major `(P, K-1)` order; column `a` is class
        # `levels[a]`.
        coefs = list(meta["coefficients_flat"])
        lambda_offset = 0
        for a in range(m):
            class_block = coefs[a::m]
            norm = math.sqrt(sum(c * c for c in class_block))
            row_bits = [f"‖β_a‖₂ = {norm:.4g}"]
            if lambdas_per_block[a] > 0:
                n_lam = lambdas_per_block[a]
                lam_chunk = lambdas[lambda_offset : lambda_offset + n_lam]
                lambda_offset += n_lam
                lam_strs = [
                    f"{label}: {float(value):.4g}"
                    for label, value in zip(lambda_labels, lam_chunk, strict=True)
                ]
                row_bits.append(f"λ = [{', '.join(lam_strs)}]")
            if edf_per_class is not None and a < len(edf_per_class):
                row_bits.append(f"edf = {float(edf_per_class[a]):.4g}")
            lines.append(
                f"    class {levels[a]!r} vs ref: " + ", ".join(row_bits)
            )
        # Wood rank-truncated Wald smooth-term significance table (#1101): the
        # same kernel the scalar `Model.summary` uses. Present only for
        # REML-fitted models carrying covariance + smooth terms.
        sig = self.smooth_significance()
        if sig:
            lines.append("  smooth terms (Wood rank-truncated Wald):")
            lines.append(
                "    class                 term            edf   ref.df    chi.sq   p-value"
            )
            for r in sig:
                lines.append(
                    "    {cls:<20} {term:<14} {edf:6.3g} {ref:7.3g} {stat:9.4g} {p:9.3g}".format(
                        cls=str(r["class"])[:20],
                        term=str(r["term"])[:14],
                        edf=float(r["edf"]),
                        ref=float(r["ref_df"]),
                        stat=float(r["statistic"]),
                        p=float(r["p_value"]),
                    )
                )
        return "\n".join(lines)

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
