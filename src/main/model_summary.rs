use super::*;

/// Effective degrees of freedom attributable to one term (one smooth, one
/// random effect), defined as the trace of the linear-smoother influence matrix
/// `F = H⁻¹X'WX` restricted to that term's coefficient block:
///
/// ```text
/// edf_term = Σ_{j ∈ coeff_range} F[j,j]
///          = |coeff_range| − Σ_{kk ∈ term} tr_kk,   tr_kk = λ_kk·tr(H⁻¹ S_kk).
/// ```
///
/// The two forms are algebraically identical (each penalty `S_kk` is local to
/// the term's coefficient range, so the rows it touches outside the range are
/// zero, and `Σ_{j∈range} diag(H⁻¹S) = Σ_{kk∈term} tr_kk`). Both are additive
/// across terms and sum exactly to `edf_total = p − Σ_all tr_kk`, so a term's
/// EDF can never exceed the model total.
///
/// The legacy decomposition summed the per-penalty-block EDFs
/// `Σ_kk(rank(S_kk) − tr_kk)`. That is a valid split only when penalty blocks act
/// on *disjoint* coefficient ranges (`Σ_kk rank(S_kk) = |coeff_range|`). For a
/// tensor product (`te`/`ti`) — and anisotropic/adaptive smooths — several
/// marginal penalties span the *same* shared coefficient block, so
/// `Σ_kk rank(S_kk) ≫ |coeff_range|` and the sum double-counts, reporting a
/// per-term EDF larger than `edf_total` and even than the design column count
/// (issue #1219).
///
/// Resolution order: the influence-matrix trace (the model's own definition,
/// available whenever the full `F` was materialised); else the basis-invariant
/// `|coeff_range| − Σ tr_kk` from the stored per-block traces (covers the
/// large-model path where `F` is not formed); else, only if neither is
/// available, the legacy block-sum as a last resort.
fn per_term_edf(
    fit: &UnifiedFitResult,
    coeff_range: std::ops::Range<usize>,
    penalty_cursor: usize,
    k: usize,
) -> f64 {
    let dim = coeff_range.len() as f64;
    // Primary: trace of the influence matrix over the term's coefficient block.
    if let Some(f) = fit.coefficient_influence()
        && coeff_range.end <= f.nrows()
        && coeff_range.end <= f.ncols()
    {
        let tr = coeff_range.clone().map(|j| f[[j, j]]).sum::<f64>();
        return tr.clamp(0.0, dim);
    }
    // Fallback: |coeff_range| − Σ tr_kk from the stored per-block traces. Equal
    // to the influence-matrix trace and basis-invariant, so it is exact even
    // when `F` was never materialised (large models).
    let traces = fit.penalty_block_trace();
    if k == 0 {
        // Unpenalised term: every coefficient carries one full degree of freedom.
        return dim;
    }
    if let Some(block) = traces.get(penalty_cursor..penalty_cursor + k) {
        let sum_trace = block.iter().sum::<f64>();
        return (dim - sum_trace).clamp(0.0, dim);
    }
    // Last resort: the legacy per-block EDF sum. Correct for disjoint penalties;
    // retained only for fits that recorded neither `F` nor per-block traces.
    fit.edf_by_block()
        .get(penalty_cursor..penalty_cursor + k)
        .map(|block| block.iter().sum::<f64>())
        .unwrap_or(0.0)
}

pub(crate) fn build_model_summary(
    design: &gam::smooth::TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    family: LikelihoodSpec,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> ModelSummary {
    const CONTINUOUS_ORDER_EPS: f64 = 1e-12;
    let se = fit
        .beta_standard_errors_corrected()
        .or(fit.beta_standard_errors());
    let cov_forwald = fit.beta_covariance_corrected().or(fit.beta_covariance());
    let scale_is_estimated = matches!(
        family.response,
        ResponseFamily::Gaussian | ResponseFamily::Gamma
    );
    let residual_df = (y.len() as f64 - fit.edf_total().unwrap_or(fit.beta.len() as f64)).max(1.0);
    let two_sided_parametric_p = |z: f64| -> Option<f64> {
        if !z.is_finite() {
            return None;
        }
        if scale_is_estimated {
            let dist = StudentsT::new(0.0, 1.0, residual_df).ok()?;
            Some((2.0 * (1.0 - dist.cdf(z.abs()))).clamp(0.0, 1.0))
        } else {
            Some((2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0))
        }
    };

    let nullmu = match family.response {
        ResponseFamily::Gaussian => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let ybar = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            Array1::from_elem(y.len(), ybar)
        }
        ResponseFamily::Binomial => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let p = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            Array1::from_elem(y.len(), p)
        }
        ResponseFamily::RoystonParmar => Array1::from_elem(y.len(), 0.0),
        ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let mean = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            let baseline = match family.response {
                ResponseFamily::Poisson => mean.max(0.0),
                ResponseFamily::Beta { .. } => {
                    mean.clamp(gam::pirls::BETA_MU_EPS, 1.0 - gam::pirls::BETA_MU_EPS)
                }
                _ => mean.max(1e-12),
            };
            Array1::from_elem(y.len(), baseline)
        }
    };
    let null_dev = {
        let null_likelihood = if family.is_royston_parmar() {
            gam::types::GlmLikelihoodSpec::canonical(gam::types::LikelihoodSpec::new(
                gam::types::ResponseFamily::Gaussian,
                gam::types::InverseLink::Standard(gam::types::StandardLink::Identity),
            ))
        } else {
            gam::types::GlmLikelihoodSpec::canonical(family.clone())
        };
        gam::pirls::calculate_deviance(y, &nullmu, &null_likelihood, weights)
    };
    let deviance_explained = if null_dev.is_finite() && null_dev > 0.0 {
        Some((1.0 - fit.deviance / null_dev).clamp(-9.0, 1.0))
    } else {
        None
    };

    let mut parametric_terms = Vec::<ParametricTermSummary>::new();
    let intercept_idx = design.intercept_range.start;
    let intercept_beta = fit.beta.get(intercept_idx).copied().unwrap_or(0.0);
    let intercept_se = se.and_then(|s| s.get(intercept_idx).copied());
    let interceptz = intercept_se.and_then(|s| (s > 0.0).then_some(intercept_beta / s));
    let intercept_p = interceptz.and_then(two_sided_parametric_p);
    parametric_terms.push(ParametricTermSummary {
        name: "Intercept".to_string(),
        estimate: intercept_beta,
        std_error: intercept_se,
        zvalue: interceptz,
        pvalue: intercept_p,
    });
    for (name, range) in &design.linear_ranges {
        let linear_meta = spec.linear_terms.iter().find(|term| term.name == *name);
        let geometry_label = match linear_meta {
            Some(LinearTermSpec {
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min,
                coefficient_max,
                ..
            }) => match (coefficient_min, coefficient_max) {
                (Some(lb), Some(ub)) => format!("{name} [coef in [{lb:.3}, {ub:.3}]]"),
                (Some(lb), None) => format!("{name} [coef >= {lb:.3}]"),
                (None, Some(ub)) => format!("{name} [coef <= {ub:.3}]"),
                (None, None) => name.clone(),
            },
            Some(LinearTermSpec {
                coefficient_geometry: LinearCoefficientGeometry::Bounded { min, max, prior },
                coefficient_min,
                coefficient_max,
                ..
            }) => {
                let prior_txt = match prior {
                    BoundedCoefficientPriorSpec::None => ", no-prior".to_string(),
                    BoundedCoefficientPriorSpec::Uniform => ", Uniform(log-Jacobian)".to_string(),
                    BoundedCoefficientPriorSpec::Beta { a, b } => {
                        format!(", Beta({a:.3},{b:.3})")
                    }
                };
                let constraint_txt = match (coefficient_min, coefficient_max) {
                    (Some(lb), Some(ub)) => format!(", coef in [{lb:.3}, {ub:.3}]"),
                    (Some(lb), None) => format!(", coef >= {lb:.3}"),
                    (None, Some(ub)) => format!(", coef <= {ub:.3}"),
                    (None, None) => String::new(),
                };
                format!("{name} [bounded {min:.3}..{max:.3}{prior_txt}{constraint_txt}]")
            }
            None => name.clone(),
        };
        for idx in range.start..range.end {
            let beta = fit.beta.get(idx).copied().unwrap_or(0.0);
            let se_i = se.and_then(|s| s.get(idx).copied());
            let z = se_i.and_then(|s| (s > 0.0).then_some(beta / s));
            let p = z.and_then(two_sided_parametric_p);
            let label = if range.end - range.start > 1 {
                format!("{geometry_label}[{}]", idx - range.start)
            } else {
                geometry_label.clone()
            };
            parametric_terms.push(ParametricTermSummary {
                name: label,
                estimate: beta,
                std_error: se_i,
                zvalue: z,
                pvalue: p,
            });
        }
    }

    let mut smooth_terms = Vec::<SmoothTermSummary>::new();
    let mut penalty_cursor = 0usize;
    for (name, range) in &design.random_effect_ranges {
        let edf = per_term_edf(fit, range.clone(), penalty_cursor, 1);
        penalty_cursor += 1;
        // Random-effect smooths are variance-component tests on the boundary;
        // a naive coefficient Wald χ² p-value is anti-conservative, so only EDF is reported.
        let chi_sq_opt: Option<f64> = None;
        let ref_df = edf.max(0.0);
        let pvalue: Option<f64> = None;
        smooth_terms.push(SmoothTermSummary {
            name: name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            pvalue,
            continuous_order: None,
            basis_note: None,
        });
    }
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        let term_penalty_start = penalty_cursor;
        let edf = per_term_edf(fit, term.coeff_range.clone(), penalty_cursor, k);
        penalty_cursor += k;
        let smooth_test = if term.shape == gam::smooth::ShapeConstraint::None {
            cov_forwald.and_then(|cov| {
                wood_smooth_test(SmoothTestInput {
                    beta: fit.beta.view(),
                    covariance: cov,
                    influence_matrix: fit.coefficient_influence(),
                    coeff_range: term.coeff_range.clone(),
                    edf,
                    nullspace_dim: term.nullspace_dims.iter().copied().sum::<usize>(),
                    residual_df,
                    scale: if scale_is_estimated {
                        SmoothTestScale::Estimated
                    } else {
                        SmoothTestScale::Known
                    },
                })
            })
        } else {
            None
        };
        let chi_sq_opt = smooth_test.as_ref().map(|test| test.statistic);
        let ref_df = smooth_test
            .as_ref()
            .map(|test| test.ref_df)
            .unwrap_or(edf.max(0.0));
        let pvalue = smooth_test.as_ref().map(|test| test.p_value);
        let continuous_order = if k == 3
            && term_penalty_start + 2 < fit.lambdas.len()
            && term_penalty_start + 2 < design.penaltyinfo.len()
        {
            // Unscaling identity for physical lambdas:
            //   S_tilde_k = S_k / c_k, and
            //   lambda_tilde_k * S_tilde_k = (lambda_tilde_k / c_k) * S_k.
            // Therefore physical lambda used by continuous-order diagnostics is
            //   lambda_k = lambda_tilde_k / c_k.
            let normalized_scale = |idx: usize| {
                let c = design.penaltyinfo[idx].penalty.normalization_scale;
                if c.is_finite() && c > 0.0 {
                    Some(c)
                } else {
                    None
                }
            };
            let lambda_tilde = [
                fit.lambdas[term_penalty_start],
                fit.lambdas[term_penalty_start + 1],
                fit.lambdas[term_penalty_start + 2],
            ];
            match (
                normalized_scale(term_penalty_start),
                normalized_scale(term_penalty_start + 1),
                normalized_scale(term_penalty_start + 2),
            ) {
                (Some(c0), Some(c1), Some(c2)) => Some(compute_continuous_smoothness_order(
                    lambda_tilde,
                    [c0, c1, c2],
                    CONTINUOUS_ORDER_EPS,
                )),
                _ => None,
            }
        } else {
            None
        };
        let basis_note = match &term.metadata {
            gam::basis::BasisMetadata::BSpline1D {
                auto_shrink_note, ..
            } => auto_shrink_note.clone(),
            _ => None,
        };
        smooth_terms.push(SmoothTermSummary {
            name: term.name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            pvalue,
            continuous_order,
            basis_note,
        });
    }

    ModelSummary {
        family: family.pretty_name().to_string(),
        deviance_explained,
        reml_score: Some(fit.reml_score),
        parametric_terms,
        smooth_terms,
    }
}

pub(crate) fn array2_to_nestedvec(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.axis_iter(Axis(0)).map(|row| row.to_vec()).collect()
}

pub(crate) fn covariance_from_model(
    model: &SavedModel,
    mode: CovarianceModeArg,
) -> Result<Array2<f64>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    let cov = match mode {
        CovarianceModeArg::Corrected => fit.beta_covariance_corrected().or(fit.beta_covariance()),
        CovarianceModeArg::Conditional => fit.beta_covariance(),
    };
    if let Some(cov) = cov {
        return Ok(cov.clone());
    }
    if let Some(hessian) = fit.penalized_hessian() {
        let backend = PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            hessian.clone(),
        ))
        .map_err(|e| format!("failed to factor saved penalized Hessian for prediction: {e}"))?;
        let dim = backend.nrows();
        let mut eye = Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            eye[[j, j]] = 1.0;
        }
        return backend.apply_columns(&eye).map_err(|e| {
            format!("failed to recover covariance from saved penalized Hessian: {e}")
        });
    }
    Err(
        "nonlinear posterior-mean prediction requires covariance or a saved penalized Hessian; refit"
            .to_string(),
    )
}

pub(crate) fn prediction_backend_from_model<'a>(
    model: &'a SavedModel,
    mode: CovarianceModeArg,
) -> Result<PredictionCovarianceBackend<'a>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    let covariance = match mode {
        CovarianceModeArg::Corrected => fit.beta_covariance_corrected().or(fit.beta_covariance()),
        CovarianceModeArg::Conditional => fit.beta_covariance(),
    };
    if let Some(covariance) = covariance {
        return Ok(PredictionCovarianceBackend::from_dense(covariance.view()));
    }
    if let Some(hessian) = fit.penalized_hessian() {
        // Surface the factorization error directly rather than swallowing it
        // and reporting the generic "model is missing either ..." message.
        // When the saved Hessian exists but cannot be factored (indefinite,
        // numerically degenerate, etc.) the user needs to see *why*, not a
        // confused "refit" instruction that doesn't match the real fault.
        return PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            hessian.clone(),
        ))
        .map_err(|e| format!("failed to factor saved penalized Hessian for prediction: {e}"));
    }
    Err(
        "nonlinear posterior-mean prediction requires either covariance or a saved penalized Hessian; refit"
            .to_string(),
    )
}

pub(crate) fn infer_covariance_mode(
    mode: CovarianceModeArg,
) -> gam::estimate::InferenceCovarianceMode {
    match mode {
        CovarianceModeArg::Conditional => gam::estimate::InferenceCovarianceMode::Conditional,
        CovarianceModeArg::Corrected => {
            gam::estimate::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred
        }
    }
}

pub(crate) fn response_interval_from_mean_sd(
    mean: ArrayView1<'_, f64>,
    response_sd: ArrayView1<'_, f64>,
    z: f64,
    lo: f64,
    hi: f64,
) -> (Array1<f64>, Array1<f64>) {
    let lower = Array1::from_iter(
        mean.iter()
            .zip(response_sd.iter())
            .map(|(&m, &s)| (m - z * s).clamp(lo, hi)),
    );
    let upper = Array1::from_iter(
        mean.iter()
            .zip(response_sd.iter())
            .map(|(&m, &s)| (m + z * s).clamp(lo, hi)),
    );
    (lower, upper)
}

pub(crate) fn invert_symmetric_matrix(a: &Array2<f64>) -> Result<Array2<f64>, CliError> {
    if a.nrows() != a.ncols() {
        return Err(CliError::Internal {
            reason: format!(
                "matrix must be square for inversion; got {}x{}",
                a.nrows(),
                a.ncols()
            ),
        });
    }
    let n = a.nrows();
    let h = gam::faer_ndarray::FaerArrayView::new(a);
    let mut rhs = FaerMat::zeros(n, n);
    for i in 0..n {
        rhs[(i, i)] = 1.0;
    }
    let factor = gam::faer_ndarray::factorize_symmetricwith_fallback(h.as_ref(), Side::Lower)
        .map_err(|err| CliError::Internal {
            reason: format!("failed to factorize matrix for inversion: {err}"),
        })?;
    factor.solve_in_place(rhs.as_mut());
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = rhs[(i, j)];
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(CliError::Internal {
            reason: "inversion produced non-finite entries".to_string(),
        });
    }
    Ok(out)
}

pub(crate) fn fit_result_from_external(ext: ExternalOptimResult) -> UnifiedFitResult {
    let log_lambdas = ext.lambdas.mapv(|v| v.max(1e-300).ln());
    let edf = ext
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);
    let geometry = ext
        .inference
        .as_ref()
        .map(|inf| gam::estimate::FitGeometry {
            penalized_hessian: inf.penalized_hessian.clone(),
            working_weights: inf.working_weights.clone(),
            working_response: inf.working_response.clone(),
        });
    // Boundary adapter: `inf.beta_covariance` is now the `PhiScaledCovariance`
    // newtype; unwrap to the raw `Array2<f64>` that the
    // `covariance_conditional` parts field still uses.
    let covariance_conditional = ext
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance.as_ref().map(|c| c.as_array().clone()));
    let covariance_corrected = ext
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance_corrected.clone());
    let penalized_objective = ext.reml_score;
    UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
        blocks: vec![gam::estimate::FittedBlock {
            beta: ext.beta.clone(),
            role: gam::estimate::BlockRole::Mean,
            edf,
            lambdas: ext.lambdas.clone(),
        }],
        log_lambdas,
        lambdas: ext.lambdas,
        likelihood_family: Some(ext.likelihood_family),
        likelihood_scale: ext.likelihood_scale,
        log_likelihood_normalization: ext.log_likelihood_normalization,
        log_likelihood: ext.log_likelihood,
        deviance: ext.deviance,
        reml_score: ext.reml_score,
        stable_penalty_term: ext.stable_penalty_term,
        penalized_objective,
        used_device: ext.used_device,
        outer_iterations: ext.iterations,
        outer_converged: ext.outer_converged,
        outer_gradient_norm: Some(ext.finalgrad_norm),
        standard_deviation: ext.standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference: ext.inference,
        fitted_link: ext.fitted_link,
        geometry,
        block_states: Vec::new(),
        pirls_status: ext.pirls_status,
        max_abs_eta: ext.max_abs_eta,
        constraint_kkt: ext.constraint_kkt,
        artifacts: ext.artifacts,
        inner_cycles: 0,
    })
    .expect("external optimizer returned invalid fit metrics")
}
