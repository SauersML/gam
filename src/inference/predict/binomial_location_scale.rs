use super::*;

/// Binomial location-scale predictor: two blocks (threshold + log-sigma).
///
/// Predicts probabilities through the threshold-scale parameterisation:
///   eta_t = X_threshold @ beta_threshold + offset
///   eta_s = X_noise @ beta_noise + offset_noise
///   sigma = exp(eta_s)
///   q0    = -eta_t / sigma
///   prob  = inverse_link(q0)
///
/// Delta-method SEs propagate through the chain rule of q0 w.r.t. both
/// linear predictors.
pub struct BinomialLocationScalePredictor {
    pub beta_threshold: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub covariance: Option<Array2<f64>>,
    pub inverse_link: InverseLink,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl BinomialLocationScalePredictor {
    /// Compute q0 = -eta_t * exp(-eta_s) for each observation, where
    /// eta_t is the threshold linear predictor and sigma = exp(eta_s).
    ///
    /// Returns (q0_base, sigma, eta_t).
    fn compute_q0_and_sigma(
        &self,
        input: &PredictInput,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let eta_t = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Binomial location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let offset_noise = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(design_noise.nrows()), |o| o.clone());
        let eta_s = design_noise.dot(&self.beta_noise) + &offset_noise;
        // Floor sigma to prevent division by zero when eta_s underflows.
        let sigma = eta_s.mapv(|v| v.exp().max(f64::MIN_POSITIVE));
        let q0 = Array1::from_shape_fn(eta_t.len(), |i| {
            (-eta_t[i] / sigma[i]).clamp(
                -SURVIVAL_STANDARDIZED_ARG_CLAMP,
                SURVIVAL_STANDARDIZED_ARG_CLAMP,
            )
        });
        Ok((q0, sigma, eta_t))
    }

    /// Apply the saved wiggle (if present) and then the inverse link to q0.
    fn apply_link(&self, q0: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let (eta, prob, _) = self.apply_link_with_d1(q0)?;
        Ok((eta, prob))
    }

    /// Apply the saved wiggle (if present) and then the inverse link to q0,
    /// returning the inverse-link first derivative `dmu/deta` alongside `mu`.
    /// The jet (mu, d1) is computed once per row; reuse it instead of calling
    /// `inverse_link_jet_for_inverse_link` again on the same `eta[i]`.
    fn apply_link_with_d1(
        &self,
        q0: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(q0).map_err(EstimationError::from)?
        } else {
            q0.clone()
        };
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let n = eta.len();
        let pairs: Result<Vec<(f64, f64)>, EstimationError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                    &self.inverse_link,
                    eta[i],
                )?;
                Ok((jet.mu.clamp(0.0, 1.0), jet.d1))
            })
            .collect();
        let pairs = pairs?;
        let mut prob = Array1::<f64>::zeros(n);
        let mut d1 = Array1::<f64>::zeros(n);
        for (i, (mu, d1_i)) in pairs.into_iter().enumerate() {
            prob[i] = mu;
            d1[i] = d1_i;
        }
        Ok((eta, prob, d1))
    }
}

impl BinomialLocationScalePredictor {
    fn response_se_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
        q0_base: &Array1<f64>,
        sigma: &Array1<f64>,
        eta_t: &Array1<f64>,
        dmu_deta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let n = eta_t.len();
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_noise.len();
        let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
        let p_total = p_t + p_s + p_w;
        if backend.nrows() != p_total {
            return Err(EstimationError::InvalidInput(format!(
                "covariance dimension mismatch for binomial LS: expected parameter dimension {}, got {}",
                p_total,
                backend.nrows()
            )));
        }
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "binomial location-scale uncertainty requires noise design matrix".to_string(),
            )
        })?;
        linear_predictor_se_from_backend(backend, n, |rows| {
            let x_t = design_row_chunk(&input.design, rows.clone())?;
            let x_s = design_row_chunk(design_noise, rows.clone())?;
            let q0_chunk = q0_base.slice(ndarray::s![rows.clone()]).to_owned();
            let sigma_chunk = sigma.slice(ndarray::s![rows.clone()]);
            let eta_t_chunk = eta_t.slice(ndarray::s![rows.clone()]);
            let dmu_chunk = dmu_deta.slice(ndarray::s![rows.clone()]);
            let wiggle_design = if let Some(runtime) = self.link_wiggle.as_ref() {
                Some(runtime.design(&q0_chunk)?)
            } else {
                None
            };
            let dq_dq0 = if let Some(runtime) = self.link_wiggle.as_ref() {
                runtime.derivative_q0(&q0_chunk)?
            } else {
                Array1::ones(q0_chunk.len())
            };
            let rows_in_chunk = q0_chunk.len();
            let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
            for i in 0..rows_in_chunk {
                let dphi = dmu_chunk[i];
                let scale = dq_dq0[i];
                let dprob_deta_t = dphi * scale * (-1.0 / sigma_chunk[i]);
                let dprob_deta_s = dphi * scale * (eta_t_chunk[i] / sigma_chunk[i]);
                for j in 0..p_t {
                    grad[[i, j]] = dprob_deta_t * x_t[[i, j]];
                }
                for j in 0..p_s {
                    grad[[i, p_t + j]] = dprob_deta_s * x_s[[i, j]];
                }
                if let Some(wd) = wiggle_design.as_ref() {
                    for j in 0..p_w {
                        grad[[i, p_t + p_s + j]] = dphi * wd[[i, j]];
                    }
                }
            }
            Ok(vec![grad])
        })
    }

    /// Plug-in probability point + (covariance-derived) response-scale SE via
    /// the threshold/scale/wiggle chain rule. The η SE is reported equal to the
    /// response SE because the threshold-scale η interval is not meaningful on
    /// the probability scale and is collapsed onto the point predictor.
    fn plugin_state_from_covariance(
        &self,
        input: &PredictInput,
    ) -> Result<LinearState, EstimationError> {
        let with_se = self.predict_with_uncertainty_inner(input)?;
        Ok(LinearState {
            eta: with_se.eta,
            mean: with_se.mean,
            eta_se: with_se.mean_se.clone(),
            mean_se: with_se.mean_se,
            covariance_corrected_used: false,
        })
    }

    /// Delta-method response-scale SE for the binomial-LS probability from an
    /// arbitrary covariance `backend`, via the threshold/scale/wiggle chain
    /// rule. Shared by the conditional point path and the mode-selecting
    /// full-uncertainty path.
    fn mean_se_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
    ) -> Result<Array1<f64>, EstimationError> {
        let (q0_base, sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let (_, _, dmu_deta) = self.apply_link_with_d1(&q0_base)?;
        self.response_se_from_backend(input, backend, &q0_base, &sigma, &eta_t, &dmu_deta)
    }

    fn predict_with_uncertainty_inner(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let (q0_base, _, _) = self.compute_q0_and_sigma(input)?;
        let (eta, prob, _) = self.apply_link_with_d1(&q0_base)?;

        let mean_se = if let Some(ref cov) = self.covariance {
            let backend = PredictionCovarianceBackend::from_dense(cov.view());
            Some(self.mean_se_from_backend(input, &backend)?)
        } else {
            None
        };

        Ok(PredictionWithSE {
            eta,
            mean: prob,
            eta_se: None,
            mean_se,
        })
    }

    /// The coefficient-uncertainty-integrated posterior-mean probability + the
    /// response-scale SE, via the projected bivariate GHQ over the
    /// threshold/log-σ posterior. Used by the posterior-mean pass.
    fn posterior_mean_state(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
    ) -> Result<LinearState, EstimationError> {
        // Validation target for this projected 2D GHQ path:
        // compare against 100K Monte Carlo draws under strong threshold/scale
        // posterior correlation and require agreement within ~0.01; as
        // covariance -> 0, the integrated mean must converge to the plug-in
        // point prediction row-wise.
        let (q0_base, sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Binomial location-scale posterior mean requires noise design matrix".to_string(),
            )
        })?;
        let offset_noise = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(design_noise.nrows()), |o| o.clone());
        let eta_s = design_noise.dot(&self.beta_noise) + &offset_noise;
        let (eta, _, dmu_deta) = self.apply_link_with_d1(&q0_base)?;
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_noise.len();
        let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
        let p_total = p_t + p_s + p_w;
        let backend = require_posterior_mean_backend(
            fit,
            self.covariance.as_ref(),
            p_total,
            "binomial location-scale posterior mean",
        )?;

        let eta_se =
            self.response_se_from_backend(input, &backend, &q0_base, &sigma, &eta_t, &dmu_deta)?;

        let mean = if self.link_wiggle.is_none() {
            let (var_t, var_s, cov_ts) = project_two_block_linear_predictor_covariance(
                &input.design,
                design_noise,
                &backend,
                p_t,
                p_s,
                "binomial location-scale posterior mean",
            )?;
            let values: Result<Vec<_>, _> = (0..eta_t.len())
                .into_par_iter()
                .map(|i| {
                    PREDICT_QUADRATURE_CONTEXT.with(|quadctx| {
                        projected_bivariate_posterior_mean_result(
                            quadctx,
                            [eta_t[i], eta_s[i]],
                            [
                                [var_t[i].max(0.0), cov_ts[i]],
                                [cov_ts[i], var_s[i].max(0.0)],
                            ],
                            |eta_threshold, eta_log_sigma| {
                                let q0 = -eta_threshold * (-eta_log_sigma).exp();
                                let jet =
                                    crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                                        &self.inverse_link,
                                        q0,
                                    )?;
                                Ok(jet.mu.clamp(0.0, 1.0))
                            },
                        )
                    })
                })
                .collect();
            Array1::from_vec(values?)
        } else {
            let runtime = self.link_wiggle.as_ref().expect("checked above");
            let betaw = Array1::from_vec(runtime.beta.clone());
            let mut wiggle_basis_rhs = Array2::<f64>::zeros((p_total, p_w));
            for j in 0..p_w {
                wiggle_basis_rhs[[p_t + p_s + j, j]] = 1.0;
            }
            let covww = backend
                .apply_columns(&wiggle_basis_rhs)
                .map_err(EstimationError::InvalidInput)?
                .slice(ndarray::s![p_t + p_s..p_total, ..])
                .to_owned();
            let mut out = Array1::<f64>::zeros(eta.len());
            let chunk_rows = prediction_chunk_rows(p_total, 2, eta.len());
            let mut start = 0usize;
            while start < eta.len() {
                let end = (start + chunk_rows).min(eta.len());
                let rows = start..end;
                let rows_in_chunk = end - start;
                let x_t = design_row_chunk(&input.design, rows.clone())
                    .map_err(EstimationError::InvalidInput)?;
                let x_ls = design_row_chunk(design_noise, rows.clone())
                    .map_err(EstimationError::InvalidInput)?;
                let mut rhs = Array2::<f64>::zeros((p_total, rows_in_chunk * 2));
                rhs.slice_mut(ndarray::s![0..p_t, 0..rows_in_chunk])
                    .assign(&x_t.t());
                rhs.slice_mut(ndarray::s![
                    p_t..p_t + p_s,
                    rows_in_chunk..2 * rows_in_chunk
                ])
                .assign(&x_ls.t());
                let solved = backend
                    .apply_columns(&rhs)
                    .map_err(EstimationError::InvalidInput)?;
                let compute_chunk_row = |quadctx: &QuadratureContext, local_row: usize| {
                    let i = start + local_row;
                    let solved_t = solved.slice(ndarray::s![.., local_row]);
                    let solved_ls = solved.slice(ndarray::s![.., rows_in_chunk + local_row]);
                    let var_t = x_t
                        .row(local_row)
                        .dot(&solved_t.slice(ndarray::s![0..p_t]))
                        .max(0.0);
                    let var_ls = x_ls
                        .row(local_row)
                        .dot(&solved_ls.slice(ndarray::s![p_t..p_t + p_s]))
                        .max(0.0);
                    let cov_tls_t = x_t
                        .row(local_row)
                        .dot(&solved_ls.slice(ndarray::s![0..p_t]));
                    let cov_tls_ls = x_ls
                        .row(local_row)
                        .dot(&solved_t.slice(ndarray::s![p_t..p_t + p_s]));
                    let cov_tls = 0.5 * (cov_tls_t + cov_tls_ls);
                    let suv_t = solved_t.slice(ndarray::s![p_t + p_s..p_total]);
                    let suv_ls = solved_ls.slice(ndarray::s![p_t + p_s..p_total]);
                    let det = (var_t * var_ls - cov_tls * cov_tls).max(1e-12);
                    let inv_uu = [
                        [var_ls / det, -cov_tls / det],
                        [-cov_tls / det, var_t / det],
                    ];
                    let mut k0 = Array1::<f64>::zeros(p_w);
                    let mut k1 = Array1::<f64>::zeros(p_w);
                    for j in 0..p_w {
                        k0[j] = suv_t[j] * inv_uu[0][0] + suv_ls[j] * inv_uu[1][0];
                        k1[j] = suv_t[j] * inv_uu[0][1] + suv_ls[j] * inv_uu[1][1];
                    }
                    let mut covw_cond = covww.clone();
                    for r in 0..p_w {
                        for c in 0..p_w {
                            covw_cond[[r, c]] -= k0[r] * suv_t[c] + k1[r] * suv_ls[c];
                        }
                    }
                    crate::quadrature::normal_expectation_2d_adaptive_result(
                        quadctx,
                        [eta_t[i], eta_s[i]],
                        [[var_t, cov_tls], [cov_tls, var_ls]],
                        |t, ls| {
                            let q0 = -t * (-ls).exp();
                            let xw = runtime
                                .basis_row_scalar(q0)
                                .map_err(EstimationError::from)?;
                            let dt = t - eta_t[i];
                            let dls = ls - eta_s[i];
                            let meanw = q0 + xw.dot(&betaw) + dt * xw.dot(&k0) + dls * xw.dot(&k1);
                            let mut varw = 0.0;
                            for r in 0..p_w {
                                let xr = xw[r];
                                for c in 0..p_w {
                                    varw += xr * covw_cond[[r, c]] * xw[c];
                                }
                            }
                            let jet = crate::quadrature::integrated_inverse_link_jetwith_state(
                                quadctx,
                                self.inverse_link.link_function(),
                                meanw,
                                varw.max(0.0).sqrt(),
                                self.inverse_link.mixture_state(),
                                self.inverse_link.sas_state(),
                            )?;
                            Ok::<f64, EstimationError>(jet.mean.clamp(0.0, 1.0))
                        },
                    )
                };
                let chunk_values: Result<Vec<f64>, EstimationError> = (0..rows_in_chunk)
                    .into_par_iter()
                    .map(|local_row| {
                        PREDICT_QUADRATURE_CONTEXT
                            .with(|quadctx| compute_chunk_row(quadctx, local_row))
                    })
                    .collect();
                for (local_row, value) in chunk_values?.into_iter().enumerate() {
                    out[start + local_row] = value;
                }
                start = end;
            }
            out
        };
        // Binomial location-scale eta_se is response-scale (dprob/dθ chain
        // rule), so bounds are mean ± z·se clamped to [0, 1]. The threshold-scale
        // η interval is not meaningful, so it is collapsed onto the point
        // predictor and uncertainty flows through the delta-method response SE.
        // The response-scale `eta_se` is reported as both the η and mean SE so
        // the collapsed-delta policy carries it through unchanged.
        Ok(LinearState {
            eta,
            mean,
            eta_se: Some(eta_se),
            mean_se: None,
            covariance_corrected_used: false,
        })
    }
}

impl PredictionTransform for BinomialLocationScalePredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        self.plugin_state_from_covariance(input)
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
                // Build the response-scale SE from the requested covariance mode
                // and report which covariance was used. The full-uncertainty
                // path always has a fit (with at least a conditional backend);
                // when no covariance can be formed at all this surfaces a clean
                // error rather than the previous silent zero-SE collapse.
                let p_total = self.beta_threshold.len()
                    + self.beta_noise.len()
                    + self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
                let (backend, covariance_corrected_used) = fit.select_uncertainty_backend(
                    p_total,
                    covariance_mode,
                    "binomial location-scale",
                )?;
                let (q0_base, _, _) = self.compute_q0_and_sigma(input)?;
                let (eta, prob) = self.apply_link(&q0_base)?;
                let response_se = self.mean_se_from_backend(input, &backend)?;
                Ok(LinearState {
                    eta,
                    mean: prob,
                    eta_se: Some(response_se.clone()),
                    mean_se: Some(response_se),
                    covariance_corrected_used,
                })
            }
            PredictPass::PosteriorMean => {
                assert!(std::mem::size_of_val(&covariance_mode) > 0);
                self.posterior_mean_state(input, fit)
            }
        }
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        // Binomial LS forms its response interval by the delta method, not by
        // transforming η endpoints, so this maps the wiggle-applied η through
        // the inverse link for completeness only.
        let (_, prob) = self.apply_link(eta)?;
        Ok(prob)
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        assert!(std::mem::size_of_val(&pass) > 0);
        // Probability already evaluated post-transformation: delta-method
        // response interval with the threshold-scale η interval collapsed onto
        // the point predictor.
        ResponseInterval::CollapsedDelta
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::UNIT_PROBABILITY
    }

    fn response_family(&self) -> ResponseFamily {
        ResponseFamily::Binomial
    }
}

impl PredictableModel for BinomialLocationScalePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let (q0_base, _, _) = self.compute_q0_and_sigma(input)?;
        let (eta, prob) = self.apply_link(&q0_base)?;
        Ok(PredictResult { eta, mean: prob })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        self.predict_with_uncertainty_inner(input)
    }

    fn predict_noise_scale(
        &self,
        predict_input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        assert!(std::mem::size_of_val(predict_input) > 0);
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit_result: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        predict_full_uncertainty_generic(self, input, fit_result, options)
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        predict_posterior_mean_generic(self, input, fit, options)
    }

    fn n_blocks(&self) -> usize {
        if self.link_wiggle.is_some() { 3 } else { 2 }
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        if self.link_wiggle.is_some() {
            vec![BlockRole::Location, BlockRole::Scale, BlockRole::LinkWiggle]
        } else {
            vec![BlockRole::Location, BlockRole::Scale]
        }
    }
}

