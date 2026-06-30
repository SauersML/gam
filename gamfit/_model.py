"""Public :class:`Model` shell.

The numeric work all lives in the Rust core: this module marshals
arguments through the FFI, hands payloads off to ``_survival`` /
``_diagnose_plot`` helpers, and exposes Pythonic properties.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

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
from ._tables import normalize_table, response_column_name, restore_output_table


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

    __slots__ = ("_model_bytes", "_training_table_kind")

    def __init__(self, *, _model_bytes: bytes, _training_table_kind: str | None = None) -> None:
        self._model_bytes = _model_bytes
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
            gains ``std_error``, ``mean_lower``, and ``mean_upper`` columns
            alongside ``linear_predictor`` / ``mean``. On survival models it
            also produces per-cell hazard / survival SEs. Issue #342
            collapsed the previous overlapping ``with_uncertainty`` boolean
            into this single flag (use ``interval=0.95`` for the SE-only
            case).

            Pass ``interval="conformal"`` to use exact distribution-free
            jackknife+ prediction intervals (Barber et al. 2021) — no
            held-out calibration fold is required. The coverage level is
            controlled by ``conformal_level`` (default ``0.9``). This path
            requires a Gaussian-identity model fitted without prior weights,
            offsets, or a link wiggle; use :meth:`predict_conformal` for
            split-conformal intervals on other families.

            Pass ``interval="full_conformal"`` for the EXACT full-conformal
            set (#942 Layer 1): every observation is used for both fitting and
            calibration, the exact prediction set is computed from one Cholesky
            per test point with zero refits, and the output additionally gains a
            ``frozen_rho_certified`` column reporting the Layer-3 frozen-ρ
            self-diagnostic (whether freezing the global smoothing parameter is
            certified equal to the honest ρ-re-selecting set). Same eligibility
            as ``"conformal"``; ``mean_lower`` / ``mean_upper`` report the outer
            envelope of the (possibly multi-interval) exact set.
        conformal_level : float, default 0.9
            Target marginal coverage in ``(0, 1)`` when ``interval="conformal"``
            or ``interval="full_conformal"``
            (e.g. ``0.95`` for a 95% interval). Ignored when ``interval`` is a
            float or ``None``.
        covariance_mode : {"conditional", "smoothing", "required"}, optional
            Posterior covariance source for the interval (CLI<->Python parity
            with ``gam predict --covariance-mode``). ``"conditional"`` uses the
            conditional posterior ``H^{-1}`` only; ``"smoothing"`` (the
            default when ``None``) prefers the first-order smoothing-corrected
            covariance ``H^{-1} + J Var(rho_hat) J^T`` and falls back to
            conditional when it is unavailable; ``"required"`` demands the
            smoothing correction and errors if it cannot be formed. Read
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
            variance). The credible ``mean_lower`` / ``mean_upper`` are left
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
              ``linear_predictor`` (linear-predictor scale; equals ``mean``
              for identity-link models), ``mean`` (response scale; the point
              prediction), and — when ``interval`` is set — ``std_error``
              (response-scale standard error including both fixed-effect and
              smoothing uncertainty) plus ``mean_lower`` / ``mean_upper``
              (interval endpoints).
              When the requested table container is ``"dict"``, the return is
              a ``PredictionResult``: it supports normal mapping access
              (``pred["mean"]``) and column attributes (``pred.mean``,
              ``pred.std_error``, ``pred.mean_lower``, ``pred.mean_upper``).
            * Bernoulli marginal-slope: a 1-D ``ndarray`` of probabilities.
            * Transformation-normal: a 1-D ``ndarray`` of z-scores.
            * Survival models: :class:`SurvivalPrediction`.
            * Competing-risks models: :class:`CompetingRisksPrediction`.

        Notes
        -----
        The response-scale ``mean`` is the **posterior mean** point estimate,
        never the plug-in mode: for a curved inverse link it is the
        coefficient-uncertainty-integrated ``E[link^{-1}(X·beta)]`` (the value
        the ``gam predict`` CLI reports by default), not ``link^{-1}(X·beta_hat)``.
        The choice is a property of the model alone, so it is the same whether
        or not ``interval`` is requested. For effectively-linear models
        (identity-link Gaussian, …) the integral collapses to the plug-in, so
        the two coincide exactly.

        For Gaussian / identity-link GLMs, ``linear_predictor`` and ``mean``
        are numerically identical (linear-predictor scale == response scale).
        For non-identity links (logit, log, ...) ``linear_predictor`` is the
        plug-in linear predictor ``X·beta_hat`` while ``mean`` is the
        response-scale posterior mean, so the two carry distinct information and
        — under the posterior-mean integration above — ``mean`` is not simply
        ``link^{-1}(linear_predictor)``. Issues #310, #313, #342
        renamed the columns from the engine-internal ``eta`` /
        ``effective_se`` / ``effective_variance`` labels to the standard
        statistical names; ``effective_variance`` was dropped (it was always
        exactly ``std_error ** 2``, trivial to compute downstream).
        """
        headers, rows, table_kind = normalize_table(data)
        row_ids = extract_row_ids(headers, rows, id_column)
        # #1054: interval='conformal' routes to the exact Gaussian jackknife+
        # path (no held-out fold needed, finite-sample ≥conformal_level
        # marginal coverage). The returned JSON has the same column schema as
        # the model-based predict path so shape_predict_response is unchanged.
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
        # #1098: interval='full_conformal' routes to the EXACT Gaussian
        # full-conformal set (no held-out fold; finite-sample ≥conformal_level
        # marginal coverage; #942 Layer 1). One Cholesky per test point, zero
        # refits. The returned JSON carries the same column schema plus a
        # `frozen_rho_certified` column (the Layer-3 self-diagnostic).
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
        opts_json = rust_module().build_model_predict_payload_json(
            self._model_bytes,
            headers,
            rows,
            interval,
            covariance_mode,
            observation_interval,
        )
        try:
            raw = rust_module().predict_table(
                self._model_bytes, headers, rows, opts_json
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
            # `[linear_predictor, mean]` column matrix. The FFI returns the lone
            # response-scale `mean` column as `(n, 1)`; drop the trailing axis.
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
        response-scale ``mean_lower`` / ``mean_upper`` columns with the
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
        covariance_mode : {"conditional", "smoothing", "required"}, optional
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
            A table with ``linear_predictor``, ``mean``, ``std_error``, and
            the conformal ``mean_lower`` / ``mean_upper`` columns. Currently
            supported for standard GAM models only.
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

        For each penalized (shape-unconstrained) smooth term it returns
        ``statistic_lr`` (the raw :math:`W`), ``ref_df`` (the Wood truncation
        :math:`d`, the same reference the Wald row uses), ``bartlett_factor``
        :math:`c`, ``statistic_corrected`` :math:`W^*`, ``p_value_uncorrected``,
        ``p_value_corrected`` (the magic-by-default value), ``material`` (the
        n-too-small-here diagnostic — ``True`` when the correction moves the
        Bartlett factor or the p-value by more than 10%), and
        ``correction_provenance`` — ``"lawley_lr"`` when the family carries
        closed-form cumulant jets (gaussian / poisson / binomial / gamma) and the
        null refit converged, else ``"none"`` (the uncorrected
        :math:`\\chi^2_d` stands, never weakened).

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
            raw = ffi.sample_table(
                self._model_bytes,
                headers,
                rows,
                options_json,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return PosteriorSamples.from_ffi_json(raw, model_bytes=self._model_bytes)

    def sample_replicates(
        self,
        data: Any,
        n_draws: int = 100,
        *,
        seed: int = 0,
    ) -> Any:
        """Draw posterior-predictive replicate responses at ``data`` (#1057).

        Each of the ``n_draws`` rows is a fresh synthetic response vector drawn
        from the fitted predictive distribution — the family-aware observation
        noise (Gaussian / Poisson / Bernoulli / Gamma / Beta / Tweedie /
        Negative-Binomial) wrapped around the plug-in mean ``g^{-1}(X·beta_hat)``.
        This is the *observation* replicate path (distinct from :meth:`sample`,
        which draws the *parameter* posterior) and is the engine for
        posterior-predictive checks, synthetic-data generation, and
        simulation-based calibration. The family and fitted dispersion are read
        from the saved model — there is no family flag.

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

    def posterior_predictive_check(
        self,
        data: Any,
        *,
        n_draws: int = 200,
        seed: int = 0,
    ) -> dict[str, float]:
        """Posterior-predictive Bayesian p-values per discrepancy statistic (#1057).

        Draws ``n_draws`` replicate datasets at ``data`` (which must include the
        fitted response column), then for each summary statistic ``T`` reports
        the posterior-predictive p-value ``P(T(y_rep) >= T(y_obs))`` — the
        fraction of replicates whose statistic is at least as extreme as the
        observed one. Values near 0 or 1 flag a statistic the model fails to
        reproduce; values near 0.5 indicate calibration on that statistic.

        Parameters
        ----------
        data : table-like
            New rows including the response column (so ``T(y_obs)`` is defined).
        n_draws : int, default 200
            Number of replicate datasets used to estimate each p-value.
        seed : int, default 0
            Seed for the deterministic draw stream.

        Returns
        -------
        dict[str, float]
            Bayesian p-value per statistic (``mean``, ``sd``, ``min``, ``max``).
        """
        import numpy as np

        response = self.response_name
        if response is None:
            raise ValueError(
                "posterior_predictive_check requires a model with a named response "
                "column; this model's formula does not expose one"
            )
        headers, rows, _ = normalize_table(data)
        if response not in headers:
            raise ValueError(
                f"posterior_predictive_check requires the response column "
                f"'{response}' in data so the observed statistics are defined"
            )
        col = headers.index(response)
        y_obs = np.asarray([float(r[col]) for r in rows], dtype=float)
        reps = self.sample_replicates(data, n_draws, seed=seed)
        reps = np.asarray(reps, dtype=float)

        statistics = {
            "mean": np.mean,
            "sd": np.std,
            "min": np.min,
            "max": np.max,
        }
        out: dict[str, float] = {}
        for name, fn in statistics.items():
            t_obs = float(fn(y_obs))
            t_rep = np.asarray([float(fn(reps[d])) for d in range(reps.shape[0])])
            out[name] = float(np.mean(t_rep >= t_obs))
        return out

    def design_matrix(self, data: Any) -> Any:
        """Materialised design matrix for ``data`` against the saved model."""
        headers, rows, _ = normalize_table(data)
        return rust_module().design_matrix_table_dense(self._model_bytes, headers, rows)

    def design_matrix_array(self, X: Any) -> Any:
        """Materialised design matrix for a numeric feature matrix."""
        try:
            rust = rust_module()
            return rust.design_matrix_array(
                self._model_bytes,
                rust.numeric_matrix_f64(X, "X"),
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
        level: float = 0.95,
        simultaneous: bool = False,
        n_sim: int = 10_000,
        seed: int | None = 12345,
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
                float(level),
                bool(simultaneous),
                int(n_sim),
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
    def training_table_kind(self) -> str | None:
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
        """Per-term partial-dependence plot data with delta-method SE.

        Analogue of mgcv's ``plot.gam()`` per-term plot, and term-wise
        contribution computation. For the requested ``term`` this evaluates
        f_t(x) = X_t(x) beta_t and the matching delta-method SE
        sqrt(diag(X_t V_t X_t^T)) where V_t is the corresponding diagonal
        block of the joint coefficient covariance.

        Parameters
        ----------
        term:
            Term name as it appears in :attr:`term_blocks` (e.g.
            ``"s(x1)"`` for a 1D smooth).
        data:
            Reference table; non-``term`` columns supply template values
            for the constructed grid rows.
        grid:
            Optional explicit grid. For 1D smooths, a 1-D array of input
            values. For multi-D smooths, a 2-D array of shape
            ``(n_points, d)`` whose columns align with the smooth's input
            features in formula order.
        n_points:
            Number of grid points for 1D smooths when ``grid`` is ``None``.

        Returns
        -------
        dict
            ``{"grid": array, "predicted": array, "standard_error": array}``.
        """
        import numpy as np

        block = next(
            (b for b in self.term_blocks if b.name == term),
            None,
        )
        if block is None:
            available = [b.name for b in self.term_blocks]
            raise ValueError(
                f"partial_dependence: term {term!r} not found; available: {available}"
            )
        state = self._coefficient_state()
        beta_full = np.asarray(state.get("beta", []), dtype=float)
        cov_n = int(state.get("covariance_n", 0))
        cov_flat = np.asarray(state.get("covariance_flat", []), dtype=float)
        if cov_flat.size != cov_n * cov_n or beta_full.size != cov_n:
            raise ValueError("coefficient covariance payload has inconsistent dimensions")
        cov = cov_flat.reshape(cov_n, cov_n)
        cov = 0.5 * (cov + cov.T)
        cols = slice(block.start, block.end)
        beta_term = beta_full[cols]
        cov_term = cov[cols, cols]

        schema_cols = list((state.get("schema") or {}).get("columns") or [])
        ranges = state.get("training_feature_ranges") or []
        names = [str(c.get("name")) for c in schema_cols]

        # Build a template row from the supplied data.
        template: dict[str, Any] = {}
        headers, rows, _ = normalize_table(data)
        if rows:
            first = rows[0]
            template.update({h: first[i] for i, h in enumerate(headers)})
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

        # Heuristic axis inference: a smooth named e.g. ``s(x1)`` owns the
        # schema column ``x1``. Tensor / multi-d smooths expose multiple
        # arguments in the same parenthesized list.
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
            if col_idx < len(ranges):
                lo, hi = map(float, ranges[col_idx])
            else:
                lo, hi = 0.0, 1.0
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
                lo, hi = 0.0, 1.0
            grid_arr = np.linspace(float(lo), float(hi), int(n_points))
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
                        f"{grid_matrix.shape} does not match term axes "
                        f"{term_args!r}"
                    )
                sweep_columns = term_args
                grid_out = grid_matrix
            else:
                raise ValueError("partial_dependence: grid must be 1-D or 2-D")

        eval_rows = []
        for row_vals in grid_matrix:
            row = dict(template)
            for col_name, value in zip(sweep_columns, row_vals, strict=False):
                row[col_name] = str(float(value))
            eval_rows.append(row)

        design = np.asarray(self.design_matrix(eval_rows), dtype=float)
        x_term = design[:, cols]
        predicted = x_term @ beta_term
        var = np.einsum("ij,jk,ik->i", x_term, cov_term, x_term)
        se = np.sqrt(np.maximum(var, 0.0))
        return {
            "grid": grid_out,
            "predicted": predicted,
            "standard_error": se,
        }

    def variance_share(
        self,
        data: Any,
        term: str | None = None,
    ) -> dict[str, float] | float:
        """Term-wise variance decomposition on ``data``.

        Analogous to mgcv's ``summary(model)`` per-row ``edf`` and
        ``p-value`` columns, but reporting variance fractions instead:
        variance_share(t) = Var(X_t beta_t) / Var(X beta), with all
        variances computed empirically on the rows of ``data``.

        ``data`` is required because :class:`Model` does not persist
        training fitted values. The intercept is excluded from the
        per-term decomposition.

        Parameters
        ----------
        data:
            Table-like input used to evaluate the design matrix.
        term:
            If ``None``, returns ``{term_name: fraction}`` for every
            non-intercept term. If given, returns the scalar fraction.
        """
        import numpy as np

        beta_full = np.asarray(self._coefficient_state().get("beta", []), dtype=float)
        design = np.asarray(self.design_matrix(data), dtype=float)
        if design.shape[1] != beta_full.size:
            raise ValueError(
                "variance_share: design matrix and beta dimensions disagree: "
                f"design has {design.shape[1]} columns, beta has {beta_full.size}"
            )
        eta_total = design @ beta_full
        total_var = float(np.var(eta_total))

        def share_for(block: TermBlock) -> float:
            if not np.isfinite(total_var) or total_var <= 0.0:
                return 0.0
            cols = slice(block.start, block.end)
            contribution = design[:, cols] @ beta_full[cols]
            return float(np.var(contribution)) / total_var

        if term is not None:
            block = next((b for b in self.term_blocks if b.name == term), None)
            if block is None:
                available = [b.name for b in self.term_blocks]
                raise ValueError(
                    f"variance_share: term {term!r} not found; available: {available}"
                )
            return share_for(block)

        return {
            block.name: share_for(block)
            for block in self.term_blocks
            if block.kind != "intercept"
        }

    @property
    def evidence(self) -> float:
        """Minimised REML / LAML cost for this fit (penalised negative log
        marginal likelihood plus Laplace correction), on the same
        rank-normalized comparison scale used by ``gamfit.compare_models``.
        It is a *cost*, so **lower is better** -- the model with the smaller
        ``evidence`` is the better-supported one. Use :meth:`bayes_factor_vs`
        or ``gamfit.compare_models`` for a direct comparison.
        """
        return float(rust_module().model_evidence(self._model_bytes))

    def bayes_factor_vs(self, other: "Model") -> float:
        """Bayes factor of this fit against ``other``.

        Returns ``> 1`` when this fit is better supported than ``other``
        (i.e. has the lower :attr:`evidence` cost) and ``< 1`` otherwise,
        agreeing with the winner reported by ``gamfit.compare_models``.
        """
        # allow-list (a): FFI input validation.
        if not isinstance(other, Model):
            raise TypeError(
                f"bayes_factor_vs expects a gamfit.Model, got {type(other).__name__}"
            )
        log_diff = rust_module().bayes_factor_log_diff(
            self._model_bytes, other._model_bytes
        )
        return math.exp(log_diff)

    def _model_class_from_payload(self) -> str:
        return rust_module().required_saved_model_payload_string(
            self._model_bytes, "model_kind"
        )

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


def _multinomial_lambda_component_labels(
    lambda_labels: Sequence[str],
    term_labels: Sequence[str],
    n_lam: int,
) -> list[str]:
    """Return exactly ``n_lam`` labels for one class block's λ slice (#1544).

    Each active class block selects one λ per *penalty component*, and the
    Marra–Wood double penalty (plus tensor/operator smooths) emits more than one
    component per smooth term. The authoritative per-component labels live in
    ``lambda_labels`` (length ``n_lam``); when present they are used verbatim so
    each λ names both its term and its role (e.g. ``s(x)`` and
    ``s(x) [null space]``).

    The fallbacks exist only for models serialized before per-component labels
    were recorded. They must still yield *exactly* ``n_lam`` labels so the
    caller's ``zip(names, lam_chunk)`` never drops a λ — the silent truncation
    that the old ``zip(term_labels, lam_chunk)`` caused when there were fewer
    term labels than λ. If the component count is an exact multiple of the term
    count, components are grouped under their term with a 1-based suffix;
    otherwise positional ``λ{i}`` labels are used.
    """
    labels = list(lambda_labels)
    if len(labels) == n_lam:
        return labels
    terms = list(term_labels)
    if terms and n_lam % len(terms) == 0:
        comps = n_lam // len(terms)
        if comps == 1:
            return list(terms)
        return [f"{t} [{c + 1}]" for t in terms for c in range(comps)]
    return [f"λ{i}" for i in range(n_lam)]


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
        _training_table_kind: str | None = None,
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
    def converged(self) -> bool:
        return bool(self._metadata["converged"])

    @property
    def deviance(self) -> float:
        return float(self._metadata["deviance"])

    @property
    def n_iter_(self) -> int:
        return int(self._metadata["iterations"])

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
        # Two-sided normal quantile for the requested level.
        from statistics import NormalDist

        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        try:
            out = rust_module().predict_multinomial_intervals_pyfunc(
                self._model_bytes, headers, rows, z
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
        lambdas = list(meta.get("lambdas", []))
        lambdas_per_block = list(meta.get("lambdas_per_block", []))
        term_labels = list(meta.get("smooth_term_labels", []))
        # Per-penalty-component λ labels, parallel to a single class block's λ
        # slice (#1544). The Marra–Wood double penalty (and tensor/operator
        # smooths) emit more than one penalty component — hence more than one λ —
        # per smooth term, so these are NOT 1:1 with `term_labels`: a single
        # `s(x)` term yields a primary wiggliness λ and a null-space shrinkage λ,
        # each carrying its own label here. Pairing λ with these component labels
        # (rather than assuming one λ per term) is what keeps every λ in the
        # summary instead of silently truncating the null-space penalties.
        lambda_labels = list(meta.get("lambda_labels", []))
        edf_per_class = meta.get("edf_per_class")

        # Structural invariants that keep the per-class λ slicing below in
        # bounds: one λ-block per active class, and the blocks partition the flat
        # λ vector exactly. These guard genuine metadata corruption; the
        # label/λ cardinality is reconciled per-component in the renderer, never
        # assumed 1:1 with the term count (the #1544 root cause).
        if lambdas:
            if len(lambdas_per_block) != m:
                raise ValueError(
                    f"Multinomial lambda metadata mismatch: {len(lambdas_per_block)} blocks for {m} active classes"
                )
            if sum(lambdas_per_block) != len(lambdas):
                raise ValueError(
                    f"Multinomial lambda metadata mismatch: {len(lambdas)} lambdas but blocks ask for {sum(lambdas_per_block)}"
                )

        lines = [
            f"MultinomialModel formula: {meta['formula']}",
            f"  classes: {levels}  (reference = {levels[ref]!r})",
            f"  active classes (K-1): {m}",
            f"  coefficients per class (P): {p}",
            f"  total coefficients: {p * m}",
            f"  iterations: {int(meta['iterations'])}  converged: {bool(meta['converged'])}",
            f"  deviance: {float(meta['deviance']):.6g}",
            f"  penalized -log L: {float(meta['penalized_neg_log_likelihood']):.6g}",
        ]
        # Per-class slope-norm + REML λ + hat-matrix trace rollup. Coefficients
        # are stored in row-major `(P, K-1)` order; column `a` is class
        # `levels[a]`.
        coefs = list(meta["coefficients_flat"])
        lambda_offset = 0
        for a in range(m):
            class_block = coefs[a::m]
            norm = math.sqrt(sum(c * c for c in class_block))
            row_bits = [f"‖β_a‖₂ = {norm:.4g}"]
            if lambdas_per_block and lambdas_per_block[a] > 0:
                n_lam = lambdas_per_block[a]
                lam_chunk = lambdas[lambda_offset : lambda_offset + n_lam]
                lambda_offset += n_lam
                names = _multinomial_lambda_component_labels(
                    lambda_labels, term_labels, n_lam
                )
                lam_strs = [f"{t}: {float(v):.4g}" for t, v in zip(names, lam_chunk)]
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
            f"classes={self.classes_!r}, converged={self.converged})"
        )

    def __str__(self) -> str:
        return self.summary()


__all__ = [
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "MultinomialModel",
    "MultinomialPrediction",
    "SurvivalPrediction",
    "TermBlock",
    "competing_risks_cif",
]
