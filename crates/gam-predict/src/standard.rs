use super::*;

/// Standard (single-block) GAM predictor.
pub struct StandardPredictor {
    pub beta: Array1<f64>,
    pub family: gam::types::LikelihoodSpec,
    pub link_kind: Option<InverseLink>,
    pub covariance: Option<Array2<f64>>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl StandardPredictor {
    /// Build a `StandardPredictor` from a `UnifiedFitResult`, extracting beta
    /// from the first block and covariance from the unified result.
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        family: gam::types::LikelihoodSpec,
        link_kind: Option<InverseLink>,
        link_wiggle: Option<SavedLinkWiggleRuntime>,
    ) -> Result<Self, String> {
        let expected_linkwiggle = link_wiggle.is_some();
        if !expected_linkwiggle
            && (unified.n_blocks() != 1 || unified.block_by_role(BlockRole::LinkWiggle).is_some())
        {
            return Err(
                "StandardPredictor only supports single-block standard fits without link wiggles"
                    .to_string(),
            );
        }
        let beta = if expected_linkwiggle {
            unified
                .block_by_role(BlockRole::Mean)
                .map(|b| b.beta.clone())
                .ok_or_else(|| {
                    "standard link-wiggle unified fit is missing Mean coefficient block".to_string()
                })?
        } else {
            unified
                .blocks
                .first()
                .map(|b| b.beta.clone())
                .ok_or_else(|| {
                    "standard unified fit is missing its sole coefficient block".to_string()
                })?
        };
        let covariance = unified.covariance_conditional.clone();
        Ok(Self {
            beta,
            family,
            link_kind,
            covariance,
            link_wiggle,
        })
    }
}

impl StandardPredictor {
    /// Full-uncertainty η/mean point and standard errors for the link-wiggle
    /// path, computed from an arbitrary covariance `backend` (so the caller can
    /// select conditional vs. smoothing-corrected covariance). Mirrors the
    /// wiggle SE arm of [`predict_with_uncertainty`] but parameterised on the
    /// backend instead of the predictor's stored conditional covariance.
    fn wiggle_state_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let runtime = self.link_wiggle.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "standard link-wiggle uncertainty requires a link wiggle".to_string(),
            )
        })?;
        let eta_base = input.design.dot(&self.beta) + &input.offset;
        let eta = runtime.apply(&eta_base).map_err(EstimationError::from)?;
        let strategy = strategy_for_family(self.family.clone(), self.link_kind.as_ref());
        let (mean, dmu_deta) = inverse_link_mean_and_d1(&strategy, eta.view())?;
        let p_main = self.beta.len();
        let p_w = runtime.beta.len();
        let p_total = p_main + p_w;
        let eta_se = link_wiggle_eta_se_from_backend(
            backend,
            eta.len(),
            &input.design,
            &eta_base,
            runtime,
            LinkWiggleGradientLayout {
                p_main,
                p_total,
                wiggle_col_start: p_main,
            },
            "standard link-wiggle covariance dimension mismatch",
        )?;
        let mean_se = delta_method_mean_se_from_d1(&dmu_deta, &eta_se);
        Ok((eta, mean, eta_se, mean_se))
    }

    /// The wiggle-path posterior-mean state: η-scale SE through the link-wiggle
    /// chain rule, then the per-row coefficient-uncertainty-integrated mean.
    /// Only reached when a link wiggle is active; the wiggle-free path is the
    /// richer [`predict_gam_posterior_mean_from_backendwith_bc`] engine.
    fn wiggle_posterior_mean_state(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
    ) -> Result<LinearState, EstimationError> {
        let runtime = self.link_wiggle.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "standard wiggle posterior mean requires a link wiggle".to_string(),
            )
        })?;
        let plugin = self.predict_plugin_response(input)?;
        let eta_base = input.design.dot(&self.beta) + &input.offset;
        let strategy = strategy_for_family(self.family.clone(), self.link_kind.as_ref());
        let Some(backend) = posterior_mean_backend_or_warn(
            fit,
            self.covariance.as_ref(),
            self.beta.len() + runtime.beta.len(),
            "standard link-wiggle posterior mean",
        ) else {
            // No usable coefficient covariance at the full warp width. The
            // identifiable frozen-basis learnable-link fit (#1596) keeps its
            // covariance in a reduced coordinate that does not match the
            // widened standard-basis warp the predict layer reconstructs, so the
            // posterior-mean uncertainty integral is unavailable. Degrade
            // gracefully to the finite plug-in mean `g⁻¹(q̂)` (the reported
            // linear predictor) rather than failing the whole prediction;
            // `eta_se`/`mean_se` are then `None`.
            let mean = plugin
                .eta
                .iter()
                .map(|&e| strategy.inverse_link(e))
                .collect::<Result<Array1<f64>, _>>()?;
            return Ok(LinearState {
                eta: plugin.eta,
                mean,
                eta_se: None,
                mean_se: None,
                covariance_corrected_used: false,
            });
        };
        let p_main = self.beta.len();
        let p_w = runtime.beta.len();
        let p_total = p_main + p_w;
        let eta_se = link_wiggle_eta_se_from_backend(
            &backend,
            plugin.eta.len(),
            &input.design,
            &eta_base,
            runtime,
            LinkWiggleGradientLayout {
                p_main,
                p_total,
                wiggle_col_start: p_main,
            },
            "standard link-wiggle posterior mean covariance mismatch",
        )?;
        let quadctx = gam::quadrature::QuadratureContext::new();
        let mean = plugin
            .eta
            .iter()
            .zip(eta_se.iter())
            .map(|(&e, &se)| {
                // #1515: guard the wiggle posterior-mean path like the non-wiggle
                // one (predict_gam_posterior_mean_from_backendwith_bc). A degenerate
                // fit — near-singular Hessian, se in the thousands — overflows the
                // response integral E[g⁻¹(η)] to +inf, which serializes as a None
                // mean and crashes the Python shaper. Fall back to the finite plug-in
                // mean g⁻¹(η̂), consistent with the reported linear predictor.
                let pm = strategy.posterior_mean(&quadctx, e, se)?;
                if pm.is_finite() {
                    Ok(pm)
                } else {
                    strategy.inverse_link(e)
                }
            })
            .collect::<Result<Array1<f64>, _>>()?;
        Ok(LinearState {
            eta: plugin.eta,
            mean,
            eta_se: Some(eta_se),
            mean_se: None,
            // Posterior-mean integration always uses the conditional posterior.
            covariance_corrected_used: false,
        })
    }
}

/// Link-wiggle full-uncertainty / posterior-mean policy for the standard
/// predictor. Only the wiggle path routes through the generic drivers; the
/// wiggle-free path keeps the richer [`predict_gamwith_uncertainty`] /
/// [`predict_gam_posterior_mean_from_backendwith_bc`] engines (bias correction,
/// boundary/OOD inflation, smoothing-corrected backend selection), which are
/// the canonical standard engines, not duplicated boilerplate.
impl PredictionTransform for StandardPredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        let with_se = self.predict_with_uncertainty(input)?;
        Ok(LinearState {
            eta: with_se.eta,
            mean: with_se.mean,
            eta_se: with_se.eta_se,
            mean_se: with_se.mean_se,
            // Point state is built from the predictor's stored conditional
            // covariance.
            covariance_corrected_used: false,
        })
    }

    fn linear_state(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        pass: PredictPass,
        covariance_mode: InferenceCovarianceMode,
    ) -> Result<LinearState, EstimationError> {
        match pass {
            PredictPass::FullUncertainty => {
                // Build the SE backend from the requested covariance mode
                // (conditional vs. smoothing-corrected) rather than the
                // predictor's stored conditional covariance, so a caller asking
                // for corrected/full intervals gets the wider SEs and the
                // provenance flag is honest. Only the link-wiggle path reaches
                // here; the wiggle-free path is served by the dedicated
                // `predict_gamwith_uncertainty` engine, which already threads the
                // mode through `select_uncertainty_backend`.
                let runtime = self.link_wiggle.as_ref().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "standard link-wiggle uncertainty requires a link wiggle".to_string(),
                    )
                })?;
                let p_total = self.beta.len() + runtime.beta.len();
                let (backend, covariance_corrected_used) = fit.select_uncertainty_backend(
                    p_total,
                    covariance_mode,
                    "standard link-wiggle",
                )?;
                let (eta, mean, eta_se, mean_se) =
                    self.wiggle_state_from_backend(input, &backend)?;
                Ok(LinearState {
                    eta,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: Some(mean_se),
                    covariance_corrected_used,
                })
            }
            PredictPass::PosteriorMean => self.wiggle_posterior_mean_state(input, fit),
        }
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        let strategy = strategy_for_family(self.family.clone(), self.link_kind.as_ref());
        strategy.inverse_link_array(eta.view())
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // Wiggle full uncertainty reports a genuine η interval and a
            // delta-method response interval.
            PredictPass::FullUncertainty => ResponseInterval::SymmetricDelta,
            // Wiggle posterior-mean bounds transform the η endpoints through the
            // inverse link (the `enrich_posterior_mean_bounds` policy).
            PredictPass::PosteriorMean => ResponseInterval::TransformEta,
        }
    }

    fn bounds(&self) -> ResponseBounds {
        let spec = spec_from_family_link(self.family.clone(), self.link_kind.as_ref());
        ResponseBounds::for_family(&spec.response)
    }

    fn response_family(&self) -> ResponseFamily {
        self.family.response.clone()
    }
}

impl PredictableModel for StandardPredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta_base = input.design.dot(&self.beta) + &input.offset;
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(&eta_base).map_err(EstimationError::from)?
        } else {
            eta_base
        };
        let strategy = strategy_for_family(self.family.clone(), self.link_kind.as_ref());
        let mean = strategy.inverse_link_array(eta.view())?;
        Ok(PredictResult { eta, mean })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        // Compute eta once; if a covariance is available, jointly compute the
        // inverse-link mean and `dmu/deta` so the delta-method SE below can
        // reuse the d1 array instead of re-evaluating the (often nonlinear)
        // jet a second time.
        let eta_base = input.design.dot(&self.beta) + &input.offset;
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(&eta_base).map_err(EstimationError::from)?
        } else {
            eta_base
        };
        let strategy = strategy_for_family(self.family.clone(), self.link_kind.as_ref());
        // Cache d1 from the same jet that produces mean so we do not recompute it
        // in `delta_method_mean_se` below.
        let (mean, dmu_deta) = inverse_link_mean_and_d1(&strategy, eta.view())?;
        let result = PredictResult { eta, mean };
        let (eta_se, mean_se) = if let Some(ref cov) = self.covariance {
            let backend = PredictionCovarianceBackend::from_dense(cov.view());
            let se = if let Some(runtime) = self.link_wiggle.as_ref() {
                // `eta_base` is the pre-wiggle linear predictor; it differs
                // from `result.eta` only when a wiggle is active, so we
                // recompute it here to avoid the double matvec on the
                // common no-wiggle path.
                let eta_base = input.design.dot(&self.beta) + &input.offset;
                let p_main = self.beta.len();
                let p_w = runtime.beta.len();
                let p_total = p_main + p_w;
                link_wiggle_eta_se_from_backend(
                    &backend,
                    result.eta.len(),
                    &input.design,
                    &eta_base,
                    runtime,
                    LinkWiggleGradientLayout {
                        p_main,
                        p_total,
                        wiggle_col_start: p_main,
                    },
                    "standard link-wiggle covariance dimension mismatch",
                )?
            } else {
                eta_standard_errors_from_backend(&input.design, &backend)?
            };
            let mean_se = delta_method_mean_se_from_d1(&dmu_deta, &se);
            (Some(se), Some(mean_se))
        } else {
            (None, None)
        };
        Ok(PredictionWithSE {
            eta: result.eta,
            mean: result.mean,
            eta_se,
            mean_se,
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        // Wiggle-free standard fits use the richer dedicated engine (bias
        // correction, boundary/OOD inflation, smoothing-corrected backend);
        // the link-wiggle path shares the generic interval driver.
        if self.link_wiggle.is_none() {
            return predict_gamwith_uncertainty(
                input.design.clone(),
                self.beta.view(),
                input.offset.view(),
                spec_from_family_link(self.family.clone(), self.link_kind.as_ref()),
                fit,
                options,
            );
        }
        predict_full_uncertainty_generic(self, input, fit, options)
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // Wiggle-free standard fits use the dedicated posterior-mean engine
        // (bias correction via the fit-derived strategy); the link-wiggle path
        // shares the generic posterior-mean driver.
        if self.link_wiggle.is_none() {
            // POINT: the posterior mean `E[g⁻¹(η)]` integrates the *conditional*
            // posterior. This is the single reported point estimate regardless of
            // whether (or how) uncertainty is requested (issue #398), so it is
            // always built from the conditional covariance — never the
            // smoothing-widened one. `covariance_mode` only shapes the
            // uncertainty attached below.
            let backend = posterior_mean_backend_or_warn(
                fit,
                self.covariance.as_ref(),
                self.beta.len(),
                "standard posterior mean",
            )
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "posterior-mean prediction requires beta covariance or penalized Hessian"
                        .to_string(),
                )
            })?;
            let family = spec_from_family_link(self.family.clone(), self.link_kind.as_ref());
            let strategy = strategy_from_fit(&family, fit)?;
            // #1602: report the UNCORRECTED linear predictor η̂ = Xβ̂ here. The
            // exported coefficients (`summary().coefficients`) are the penalized
            // MLE / posterior mode β̂, and `docs/predictions.md` ("Raw design
            // matrix") promises `design_matrix(data) @ coef == linear_predictor`
            // for every family (and the `posterior.samples @ X.T` recipe). Adding
            // the O(1/n) frequentist bias-correction `b̂ = H⁻¹S(β̂−μ)` to η broke
            // that identity by exactly `X@b̂` for curved links (1.5–4% of the lp
            // range) while leaving identity-link Gaussian exact. It is also the
            // lone outlier among the sibling paths: the plug-in/full-uncertainty
            // arm sets `apply_bias_correction: false` (empirically worse against
            // truth, #398/#1536) and the link-wiggle posterior-mean path reports
            // the plug-in η. The Bayesian posterior mean `E[g⁻¹(η)]` should
            // integrate the conditional posterior of η, which is centered at the
            // mode Xβ̂ — not a frequentist-bias-shifted center — so dropping `b̂`
            // here is both contract-restoring and more principled. Pass `None`.
            let mut result = predict_gam_posterior_mean_from_backend(
                input.design.clone(),
                self.beta.view(),
                input.offset.view(),
                &backend,
                &strategy,
                "standard posterior mean",
            )?;
            if let Some(level) = options.confidence_level {
                // UNCERTAINTY: the reported SE, credible bounds and observation
                // band honour `covariance_mode`. We borrow the SE / TransformEta
                // mean bounds / mean-scale SE from the shared full-uncertainty
                // engine (which threads the mode through `select_uncertainty_
                // backend` and resolves any fitted adaptive-link state), but keep
                // the posterior-mean point above and re-centre the observation
                // band on it — so the point is unchanged while the bands widen to
                // include smoothing-parameter uncertainty exactly as for the
                // non-posterior-mean families. (Issues #811/#812: this dispatch
                // arm previously ignored both `observation_interval` and
                // `covariance_mode`.)
                let unc_options = PredictUncertaintyOptions {
                    confidence_level: level,
                    covariance_mode: options.covariance_mode,
                    mean_interval_method: MeanIntervalMethod::TransformEta,
                    // The observation band is recomputed below, centred on the
                    // posterior-mean point rather than the plug-in point.
                    includeobservation_interval: false,
                    // #1602: the reported point is the UNCORRECTED η̂ = Xβ̂ (the
                    // posterior-mean engine above is called without a bias-
                    // correction vector), matching the exported coefficients
                    // `summary().coefficients == β̂` so `design_matrix @ coef ==
                    // linear_predictor` holds for every link. Mirror that here so
                    // the borrowed SE / bounds are centred on the same η̂; we only
                    // consume the engine's SE / bounds, never its point.
                    apply_bias_correction: false,
                    ..PredictUncertaintyOptions::default()
                };
                let unc = predict_gamwith_uncertainty(
                    input.design.clone(),
                    self.beta.view(),
                    input.offset.view(),
                    self.family.clone(),
                    fit,
                    &unc_options,
                )?;
                // Adopt the covariance-mode η-scale SE, then re-derive the
                // TransformEta credible bounds from the posterior-mean point's
                // own η (the uncorrected η̂ = Xβ̂ per #1602) so the bounds stay
                // centred consistently with the reported point — only their width
                // changes with `covariance_mode`.
                result.eta_standard_error = unc.eta_standard_error;
                enrich_posterior_mean_bounds(
                    &mut result,
                    level,
                    self.family.clone(),
                    self.link_kind.as_ref(),
                )?;
                if options.include_observation_interval {
                    let z = standard_normal_quantile(0.5 + 0.5 * level)
                        .map_err(EstimationError::InvalidInput)?;
                    let z_row = Array1::from_elem(result.eta.len(), z);
                    let etavar = result.eta_standard_error.mapv(|s| s * s);
                    let (obs_lower, obs_upper) = family_observation_band(
                        &self.family.response,
                        &result.eta,
                        &etavar,
                        &result.mean,
                        &unc.mean_standard_error,
                        &z_row,
                        &z_row,
                        fit,
                        // Posterior-mean band: the analytic prior-weights path
                        // (#2077) is threaded through `predict_gamwith_uncertainty`
                        // for the effectively-linear Gaussian identity fits it
                        // targets; the curved posterior-mean families reaching here
                        // are unweighted-scalar for now (None ⇒ unchanged).
                        None,
                    );
                    result.observation_lower = obs_lower;
                    result.observation_upper = obs_upper;
                }
            }
            return Ok(result);
        }
        predict_posterior_mean_generic(self, input, fit, options)
    }

    fn n_blocks(&self) -> usize {
        if self.link_wiggle.is_some() { 2 } else { 1 }
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        if self.link_wiggle.is_some() {
            vec![BlockRole::Mean, BlockRole::LinkWiggle]
        } else {
            vec![BlockRole::Mean]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::types::StandardLink;
    use ndarray::{array, Array2};

    fn make_std(beta: Array1<f64>, family: LikelihoodSpec) -> StandardPredictor {
        StandardPredictor {
            beta,
            family,
            link_kind: None,
            covariance: None,
            link_wiggle: None,
        }
    }

    fn simple_input(design: Array2<f64>, offset: Array1<f64>) -> PredictInput {
        PredictInput {
            design: DesignMatrix::from(design),
            offset,
            design_noise: None,
            offset_noise: None,
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        }
    }

    /// Gaussian identity link: eta = X @ beta + offset, mean = eta (identity).
    #[test]
    fn gaussian_identity_eta_equals_x_beta() {
        let beta = array![1.0_f64, -0.5, 2.0];
        let pred = make_std(beta.clone(), LikelihoodSpec::gaussian_identity());
        let x = array![[1.0_f64, 0.3, -0.7], [1.0, -0.2, 1.1]];
        let offset = array![0.1_f64, -0.2];
        let input = simple_input(x.clone(), offset.clone());
        let result = pred.predict_plugin_response(&input).expect("plugin");
        let expected_eta = x.dot(&beta) + &offset;
        for i in 0..2 {
            assert!(
                (result.eta[i] - expected_eta[i]).abs() < 1e-12,
                "Gaussian identity eta[{i}]: got {:.6e}, expected {:.6e}",
                result.eta[i],
                expected_eta[i]
            );
            // Identity link: mean == eta
            assert!(
                (result.mean[i] - result.eta[i]).abs() < 1e-12,
                "Gaussian identity mean[{i}] must equal eta"
            );
        }
    }

    /// Poisson log link: mean = exp(X @ beta + offset).
    #[test]
    fn poisson_log_mean_is_exp_eta() {
        let beta = array![0.5_f64, -0.3];
        let pred = make_std(
            beta.clone(),
            LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            ),
        );
        let x = array![[1.0_f64, 0.8], [1.0, -0.5]];
        let offset = array![0.0_f64, 0.2];
        let input = simple_input(x.clone(), offset.clone());
        let result = pred.predict_plugin_response(&input).expect("plugin");
        let eta_exp = x.dot(&beta) + &offset;
        for i in 0..2 {
            assert!(
                (result.eta[i] - eta_exp[i]).abs() < 1e-12,
                "Poisson log eta[{i}]"
            );
            assert!(
                (result.mean[i] - eta_exp[i].exp()).abs() < 1e-12,
                "Poisson log mean[{i}]: got {:.6e}, expected {:.6e}",
                result.mean[i],
                eta_exp[i].exp()
            );
        }
    }

    /// Binomial logit link: mean = 1/(1 + exp(-eta)).
    #[test]
    fn binomial_logit_mean_is_sigmoid_eta() {
        let beta = array![0.0_f64]; // eta = 0 → mean = 0.5
        let pred = make_std(
            beta,
            LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
        );
        let input = simple_input(array![[1.0_f64]], array![0.0_f64]);
        let result = pred.predict_plugin_response(&input).expect("plugin");
        assert!(
            (result.mean[0] - 0.5).abs() < 1e-12,
            "logit(0) → mean = 0.5, got {}", result.mean[0]
        );
    }

    /// Covariance-backed point_state emits eta_se and mean_se;
    /// for identity link eta_se² = x @ Vb @ x.T (diagonal entries).
    #[test]
    fn point_state_with_covariance_emits_ses() {
        let beta = array![1.0_f64, 0.5];
        let cov = array![[0.04_f64, 0.0], [0.0, 0.01]];
        let mut pred = make_std(beta.clone(), LikelihoodSpec::gaussian_identity());
        pred.covariance = Some(cov.clone());
        let x = array![[1.0_f64, 2.0]]; // 1 row
        let offset = array![0.0_f64];
        let input = simple_input(x.clone(), offset);
        let state = pred.point_state(&input).expect("point_state");
        let eta_se = state.eta_se.expect("eta_se must be Some with covariance");
        // delta-method: eta_se² = x @ Vb @ x.T = 0.04 + 4 * 0.01 = 0.08
        let expected_var = 0.04_f64 + 4.0 * 0.01;
        let expected_se = expected_var.sqrt();
        assert!(
            (eta_se[0] - expected_se).abs() < 1e-12,
            "eta_se[0]: got {:.6e}, expected {:.6e}",
            eta_se[0],
            expected_se
        );
        // For Gaussian identity, dmu/deta = 1, so mean_se == eta_se
        let mean_se = state.mean_se.expect("mean_se must be Some with covariance");
        assert!(
            (mean_se[0] - eta_se[0]).abs() < 1e-12,
            "Gaussian identity: mean_se must equal eta_se"
        );
    }

    /// No covariance → point_state returns None for eta_se and mean_se.
    #[test]
    fn point_state_without_covariance_has_no_ses() {
        let pred = make_std(array![0.0_f64], LikelihoodSpec::gaussian_identity());
        let input = simple_input(array![[1.0_f64]], array![0.0_f64]);
        let state = pred.point_state(&input).expect("point_state");
        assert!(
            state.eta_se.is_none(),
            "no covariance → eta_se must be None"
        );
        assert!(
            state.mean_se.is_none(),
            "no covariance → mean_se must be None"
        );
    }
}
