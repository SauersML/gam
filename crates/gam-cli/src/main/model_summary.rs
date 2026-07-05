use super::*;

// The per-term effective-degrees-of-freedom decomposition lives on
// `UnifiedFitResult::per_term_edf` (in the library crate) so that BOTH this
// in-process CLI/report summary and the persisted-model summary the Python API
// reads (`crates/gam-pyffi` → `summary_smooth_terms`) resolve it identically.
// A previous copy here meant the #1219 influence-trace fix shipped only on the
// in-process path while the persisted path kept double-counting shared tensor
// coefficients (#1277).

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
            // Use the fit's *estimated* scale metadata, not the family default.
            // For Gamma the unit deviance is multiplied by the estimated shape
            // (calculate_deviance, deviance.rs), and `fit.deviance` already
            // carries that estimated shape (PIRLS bakes it in via
            // `with_gamma_shape`). `canonical(family)` would reset the shape to
            // the default 1.0, so the null deviance would be scaled differently
            // from the full deviance and the ratio `1 − D_full/D_null` would be
            // contaminated by the shape factor. Threading the fitted scale here
            // keeps both deviances on the same scale so it cancels exactly (the
            // same applies to any other scale-carrying family, e.g. Beta φ).
            gam::types::GlmLikelihoodSpec {
                spec: family.clone(),
                scale: fit.likelihood_scale,
            }
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
    // The fit's GLOBAL penalty layout (and thus `penalty_block_trace`) opens with a
    // single shared `LinearTermRidge` block IFF any linear term has
    // `double_penalty=true` (`design_construction.rs`). Random-effect and smooth
    // penalty blocks follow it. Seeding `penalty_cursor` at 0 ignored that leading
    // block, sliding every per-term trace window off by one whenever a penalized
    // linear term was present and masking the bug only on small dense fits (where
    // `per_term_edf` reads the influence matrix instead, #1372). Start the cursor
    // PAST any leading `LinearTermRidge` block by counting it in the recorded
    // global ordering rather than re-deriving it.
    let mut penalty_cursor = design
        .penaltyinfo
        .iter()
        .filter(|info| {
            matches!(
                &info.penalty.source,
                gam::basis::PenaltySource::Other(s) if s == "LinearTermRidge"
            )
        })
        .count();
    for (re_idx, (name, range)) in design.random_effect_ranges.iter().enumerate() {
        // Only PENALIZED random-effect blocks contribute a penalty block to the
        // flat `lambdas`/`penalty_block_trace`/`edf_by_block` layout: design
        // assembly skips the unpenalised ones (`design_construction.rs` RE-penalty
        // loop `continue`s on `!penalized`). A factor `by=` smooth injects exactly
        // such an UNPENALISED treatment-coded factor main-effect block; walking
        // `penalty_cursor` by one for it (the #1368 defect) slid the cursor one
        // block past every smooth term that followed, so the LAST by-level smooth's
        // `penalty_cursor..+k` window ran off the end of the per-block traces and
        // `per_term_edf` returned 0 (and, with EDF 0, the Wood test was skipped,
        // leaving ref_df/chi_sq/p_value at 0/NaN). Advance the cursor by the number
        // of penalty blocks the term actually owns: 1 if penalized, 0 if not.
        let penalized = spec
            .random_effect_terms
            .get(re_idx)
            .map(|t| t.penalized)
            .unwrap_or(true);
        // The design's RE-penalty loop skips a block when EITHER it is
        // unpenalised OR its coefficient range is empty (`design_construction.rs`
        // `range.is_empty() || !penalized` → `continue`). A penalised RE term
        // with zero kept groups (every level filtered) is exactly such an
        // empty-range penalised block: it owns NO entry in the flat
        // `lambdas`/`penalty_block_trace`/`edf_by_block` layout. Counting it here
        // (advancing the cursor by 1) would slide `penalty_cursor` one block past
        // every RE/smooth term that followed — the #1368 desync, just triggered by
        // an empty range rather than `!penalized`. Mirror BOTH design conditions.
        let k_pen = usize::from(penalized && !range.is_empty());
        let edf = fit.per_term_edf(range.clone(), penalty_cursor, k_pen);
        penalty_cursor += k_pen;
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
    // `SmoothTerm::coeff_range` is block-local (0-based within the smooth block);
    // the global coefficient layout is [intercept | linear | random | smooth], so
    // every term's block must be shifted by `smooth_start` before indexing the
    // global `fit.beta` / covariance / influence matrix. Omitting this offset
    // (the #1360 defect) slid each smooth's window one-per-preceding-column off,
    // folding the intercept and a neighbouring term's coefficients into the test.
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        let term_penalty_start = penalty_cursor;
        let global_range =
            (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let edf = fit.per_term_edf(global_range.clone(), penalty_cursor, k);
        penalty_cursor += k;
        let smooth_test = if term.shape == gam::smooth::ShapeConstraint::None {
            cov_forwald.and_then(|cov| {
                wood_smooth_test(SmoothTestInput {
                    beta: fit.beta.view(),
                    covariance: cov,
                    influence_matrix: fit.coefficient_influence(),
                    // Wood (2013) design-whitening Gram `X'WX = H − S(λ)` in the
                    // original coefficient basis; without it the rank-r cut keeps
                    // the wrong eigen-subspace and a dominant wiggly smooth reads
                    // as non-significant (#2142). `None` degrades to the raw
                    // covariance for a persisted model with no inference block.
                    whitening_gram: fit.weighted_gram(),
                    coeff_range: global_range.clone(),
                    edf,
                    nullspace_dim: term.wald_unpenalized_dim(),
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

pub(crate) fn infer_covariance_mode(mode: CovarianceModeArg) -> InferenceCovarianceMode {
    match mode {
        CovarianceModeArg::Conditional => InferenceCovarianceMode::Conditional,
        CovarianceModeArg::Corrected => InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
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

#[cfg(test)]
mod per_term_edf_tests {
    use super::*;
    use csv::StringRecord;
    // `FitConfig`/`FitResult` are already in scope via `super::*` (re-exported in
    // `main.rs`); only the formula-fit entry points need an explicit import.
    use gam::{encode_recordswith_inferred_schema, fit_from_formula};

    /// Regression for issue #1219: the per-term effective degrees of freedom of a
    /// tensor-product smooth `te(x, z)` must never exceed the model total EDF (nor
    /// the design column count), and the per-term EDFs must sum to the total.
    ///
    /// A `te()`/`ti()` term carries one penalty per marginal (here two) acting on a
    /// *single shared* coefficient block. The legacy decomposition summed the
    /// per-penalty-block EDFs `Σ_kk(rank(S_kk) − tr_kk)`, which counts the shared
    /// coefficients once per marginal and reports a per-term EDF larger than
    /// `edf_total` and even than `ncols(X)`. The fix defines the per-term EDF as the
    /// trace of the influence matrix `F = H⁻¹X'WX` over the term's coefficient
    /// block, `Σ_{j∈range} F[j,j]`, which is additive across terms and sums to
    /// `edf_total`. This test drives a real Gaussian `te(x, z)` fit through the
    /// public formula path and pins those invariants on the assembled summary; it
    /// fails on the old per-block-sum code and passes on the influence-trace fix.
    #[test]
    fn tensor_product_per_term_edf_does_not_exceed_total() {
        // Small synthetic surface y = sin(x*z) + noise on a deterministic grid.
        // A 18×18 grid (n = 324) is ample for a unit test and keeps it fast.
        let g = 18usize;
        let n = g * g;
        let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
        let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
        // Deterministic LCG noise — no external rng dependency, reproducible.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next_noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map the high bits to a centered uniform in roughly [-0.05, 0.05].
            let u = ((state >> 33) as f64) / ((1u64 << 31) as f64); // [0,1)
            0.1 * (u - 0.5)
        };
        for i in 0..g {
            let x = i as f64 / (g as f64 - 1.0); // [0,1]
            for j in 0..g {
                let z = j as f64 / (g as f64 - 1.0); // [0,1]
                let y = (3.0 * x * z).sin() + next_noise();
                rows.push(StringRecord::from(vec![
                    x.to_string(),
                    z.to_string(),
                    y.to_string(),
                ]));
            }
        }
        let data = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");

        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let fitted = fit_from_formula("y ~ te(x, z, k=[6,6])", &data, &config)
            .expect("te(x, z) gaussian fit should succeed");
        let FitResult::Standard(std_fit) = fitted else {
            panic!("expected a Standard fit result for a Gaussian te(x, z) model");
        };

        // Build the model summary exactly as the CLI/report path does.
        let y_col = data
            .headers
            .iter()
            .position(|h| h == "y")
            .expect("response column 'y' present");
        let y = data.values.column(y_col).to_owned();
        let weights = Array1::<f64>::ones(y.len());
        let summary = build_model_summary(
            &std_fit.design,
            &std_fit.resolvedspec,
            &std_fit.fit,
            LikelihoodSpec::gaussian_identity(),
            y.view(),
            weights.view(),
        );

        let edf_total = std_fit
            .fit
            .edf_total()
            .expect("a converged fit exposes the model total EDF");
        let ncols = std_fit.design.design.ncols() as f64;
        let tol = 1e-6;

        // The te() term must appear and carry a finite, non-negative EDF.
        assert!(
            !summary.smooth_terms.is_empty(),
            "te(x, z) must produce at least one smooth-term summary row"
        );

        let mut per_term_sum = 0.0;
        for term in &summary.smooth_terms {
            assert!(
                term.edf.is_finite() && term.edf >= -tol,
                "per-term EDF for {} must be finite and non-negative, got {}",
                term.name,
                term.edf
            );
            // The core #1219 invariant: a single term can never claim more EDF
            // than the whole model (the old per-block sum double-counted the
            // shared tensor coefficients and violated this).
            assert!(
                term.edf <= edf_total + tol,
                "per-term EDF for {} ({}) must not exceed model total EDF ({})",
                term.name,
                term.edf,
                edf_total
            );
            per_term_sum += term.edf;
        }

        // edf_total itself is bounded by the design column count (rank of X).
        assert!(
            edf_total <= ncols + tol,
            "model total EDF ({edf_total}) must not exceed design column count ({ncols})"
        );

        // mgcv trace-decomposition identity: the per-term EDFs (smooth terms, plus
        // the unpenalised intercept = 1 parametric dof) sum to the model total.
        // The summary's smooth rows cover every penalized block, so their sum plus
        // the parametric (intercept + any linear) dof recovers edf_total.
        let parametric_dof = summary.parametric_terms.len() as f64;
        let reconstructed = per_term_sum + parametric_dof;
        assert!(
            (reconstructed - edf_total).abs() <= 1e-4 * edf_total.max(1.0),
            "Σ per-term EDF (smooth {per_term_sum} + parametric {parametric_dof} = {reconstructed}) \
             must match model total EDF ({edf_total}) within tolerance"
        );
    }
}
