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
) -> Result<ModelSummary, String> {
    if y.len() != fit.training_sample_size() {
        return Err(format!(
            "summary response has {} rows but the fitted model records {} training rows",
            y.len(),
            fit.training_sample_size()
        ));
    }
    // Definition-consistent SE pair (#2296): corrected-preferred, with the
    // exact definition recorded on the summary so the displayed SEs are never
    // an unlabeled mix of covariance definitions.
    let display_uncertainty = fit.display_coefficient_uncertainty();
    let se = display_uncertainty
        .as_ref()
        .map(|view| view.standard_errors);
    // Wood (2013) design-whitening metric for the Wald smooth test (#2142):
    // the exact weighted Gram `X'WX` when the inference block is present, else
    // the unweighted `X'X` from the summary design (here the real training
    // design, so this is exact — mgcv whitens with the unweighted prediction
    // Gram anyway). `None` → the test falls back to the raw covariance.
    // Materialize `X'X` only when the exact `X'WX` is unavailable — `to_dense()`
    // on a large training design would otherwise be a needless O(n·p) copy.
    let design_gram = if fit.weighted_gram().is_none() {
        let x = design.design.to_dense();
        (x.ncols() == fit.beta.len()).then(|| x.t().dot(&x))
    } else {
        None
    };
    let whitening_gram_full: Option<&Array2<f64>> = fit.weighted_gram().or(design_gram.as_ref());
    // One metadata-owned definition serves this live-data summary and the
    // persisted-model summary the Python API reads, exactly as
    // `wald_residual_degrees_of_freedom` below does for the denominator. Each
    // surface used to spell the predicate out itself, and they had drifted:
    // this one read the scale metadata, the Python one read the family NAME, so
    // a Gamma fit with a pinned shape got two different reference distributions
    // for one fitted model (#2470).
    let scale_is_estimated = fit.likelihood_scale.wald_scale_is_estimated();
    // One fit-owned definition serves this live-data summary and the loaded
    // model summary. The response view is retained for null deviance only; its
    // length is checked above, never used as a second Wald authority.
    let residual_df = fit.wald_residual_degrees_of_freedom();
    let two_sided_parametric_p = |z: f64| -> Option<f64> {
        if !z.is_finite() {
            return None;
        }
        // Both tails come from the function that computes them, never from
        // `1 - CDF` (#2562). A CDF saturates to exactly 1.0 once its upper tail
        // drops below half an ulp of one, so the subtraction reports p = 0 for
        // every |z| above 8.3 and is 7% high already at |z| = 8 -- and the
        // `.clamp(0.0, 1.0)` that used to wrap it made the zero look legal.
        let p_value = if scale_is_estimated {
            // For t > 0, P(T > t) = I_{nu/(nu+t^2)}(nu/2, 1/2) / 2, so the
            // two-sided p-value is that regularized incomplete beta outright.
            student_t_two_sided_probability(z, residual_df?)
        } else {
            // 2 * (1 - Phi(|z|)) is erfc(|z|/sqrt2), one call and no subtraction.
            normal_two_sided_probability(z)
        };
        p_value.is_finite().then_some(p_value)
    };

    let null_likelihood = gam::types::GlmLikelihoodSpec {
        spec: family.clone(),
        scale: fit.likelihood_scale,
    };
    let null_dev = gam::pirls::calculate_null_deviance(y, &null_likelihood, weights)
        .map_err(|error| format!("null-model deviance evaluation failed: {error}"))?;
    let deviance_explained = if null_dev.is_finite() && null_dev > 0.0 {
        let value = 1.0 - fit.deviance / null_dev;
        value.is_finite().then_some(value)
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

    // The walk over the fit's flat penalty layout — the `LinearTermRidge`
    // prologue, the random-effect blocks that own no entry, the block-local →
    // global coefficient shift, the per-term influence trace, and the Wood test
    // itself — is ONE accounting shared with the persisted-model summary the
    // Python API reads (#2470). It was written out here and again in
    // `gam-pyffi`, which is why #1219, #1277, #1360, #1368 and #1372 each had to
    // be landed twice. This surface's only distinctive input is the whitening
    // Gram, because it holds the real training design.
    let smooth_terms = smooth_term_summary_rows(design, spec, fit, whitening_gram_full);

    Ok(ModelSummary {
        family: family.pretty_name().to_string(),
        deviance_explained,
        reml_score: fit.reml_score(),
        parametric_terms,
        smooth_terms,
        coefficient_se_source: display_uncertainty.map(|view| view.definition),
    })
}

/// Rebuild `Vb` from a saved fit that persisted only the penalized Hessian.
///
/// The two CLI entry points below both need this fallback, and both used to
/// hand-roll it as `from_factorized_hessian_scaled(H, 1.0)` — which is `φ = 1` and *no*
/// constrained correction. That is not `Vb` for either of the two reasons the
/// library's own fallback (`gam-predict`'s `conditional_prediction_backend`)
/// handles:
///
/// * the module invariant is `Vb = φ·H⁻¹`, and `φ` is the profiled residual
///   variance `σ̂²` for the scale-free profiled Gaussian (`1.0` for every family
///   whose IRLS weight already carries the dispersion, #679); and
/// * a fit that accepted inequality constraints has a **truncated** Laplace
///   posterior, whose covariance is `Σ − GΔGᵀ`, not the ambient `Σ`. Where a
///   constraint is active the ambient covariance is simply the wrong object:
///   it describes spread along directions the feasible set does not have
///   (#2385).
///
/// The two are one fix rather than two, because the correction's `lift` and
/// `removed_normal_variance` already live on the φ-scaled covariance metric
/// (see `PredictionCovarianceBackend::Factorized`). Subtracting a φ-scaled
/// `GΔGᵀ` from an unscaled `H⁻¹` would be dimensionally inconsistent, so the
/// correction cannot be routed here without also honoring `φ`.
///
/// Consequence of routing both through the library's constructor: a fit with no
/// engine-level family (custom / GAMLSS) has no scalar coefficient-covariance
/// scale and is now refused here, exactly as the library path already refuses
/// it, instead of silently returning an unscaled `H⁻¹` labelled `Vb`.
fn factorized_covariance_fallback(fit: &UnifiedFitResult) -> Option<Result<PredictionCovarianceBackend<'_>, String>> {
    if let Err(error) = fit.require_posterior_mean("coefficient covariance summary") {
        return Some(Err(error.to_string()));
    }
    let hessian = fit.penalized_hessian()?;
    let scale = match fit.coefficient_covariance_scale() {
        Ok(scale) => scale,
        Err(error) => {
            return Some(Err(format!(
                "saved model persisted only a penalized Hessian, so the reported covariance must be \
                 reconstructed as Vb = phi*H^-1, but this fit has no scalar coefficient-covariance \
                 scale: {error}"
            )));
        }
    };
    let constrained_correction = match fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
    {
        Some(posterior) => match posterior.correction() {
            Ok(correction) => correction,
            Err(reason) => return Some(Err(reason)),
        },
        None => None,
    };
    Some(
        PredictionCovarianceBackend::from_factorized_hessian_scaled_with_correction(
            SymmetricMatrix::Dense(hessian.clone()),
            scale,
            constrained_correction,
        )
        .map_err(|e| format!("failed to factor saved penalized Hessian for prediction: {e}")),
    )
}

/// The covariance definition a `gam predict` invocation uses: the explicit
/// `--covariance-mode` when given, else the definition the saved fit
/// publishes — the same resolution the Python bindings apply to
/// `covariance_mode=None`, so the two front ends default identically.
pub(crate) fn resolved_covariance_mode(
    args: &PredictArgs,
    model: &SavedModel,
) -> InferenceCovarianceMode {
    args.covariance_mode.unwrap_or_else(|| {
        model
            .fit_result
            .as_ref()
            .map_or(InferenceCovarianceMode::Conditional, |fit| {
                fit.published_covariance_mode()
            })
    })
}

pub(crate) fn covariance_from_model(
    model: &SavedModel,
    mode: InferenceCovarianceMode,
) -> Result<Array2<f64>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    fit.require_posterior_mean("saved-model covariance summary")
        .map_err(|error| error.to_string())?;
    if mode == InferenceCovarianceMode::SmoothingCorrected {
        if let Some(cov) = fit.beta_covariance_corrected() {
            return Ok(cov.clone());
        }
        // With NO smoothing coordinates the correction J·V_rho·Jᵀ is the unique
        // zero-dimensional zero matrix, so Vp = Vb EXACTLY. This is an identity
        // of the definition, not a fallback to a weaker uncertainty object, and
        // the library predict path already applies it (`gam-predict`'s
        // `select_uncertainty_backend`, the `fit.lambdas.is_empty()` branch).
        // The CLI never did, so every SAVED fit with an empty lambda vector —
        // a fully parametric survival fit is the common case — refused the
        // DEFAULT `gam predict` invocation (`--mode posterior-mean
        // --covariance-mode corrected`) with "refit before requesting", an
        // instruction no refit could satisfy because there is no correction to
        // compute. A fit that DOES carry smoothing coordinates keeps the hard
        // refusal: there the correction is a real, absent term.
        if !fit.lambdas.is_empty() {
            return Err(
                "saved model does not contain smoothing-corrected covariance; refit before requesting --covariance-mode corrected"
                    .to_string(),
            );
        }
    }
    if let Some(cov) = fit.beta_covariance() {
        return Ok(cov.clone());
    }
    if let Some(backend) = factorized_covariance_fallback(fit) {
        let backend = backend?;
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
    mode: InferenceCovarianceMode,
) -> Result<PredictionCovarianceBackend<'a>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    if mode == InferenceCovarianceMode::SmoothingCorrected {
        if let Some(covariance) = fit.beta_covariance_corrected() {
            return Ok(PredictionCovarianceBackend::from_dense(covariance.view()));
        }
        // Same zero-smoothing-coordinate identity as `covariance_from_model`
        // above: Vp = Vb when there is no rho to integrate over. Falling
        // through to the conditional sources is the CORRECTED answer here, not
        // a substitution of a narrower band.
        if !fit.lambdas.is_empty() {
            return Err(
                "saved model does not contain smoothing-corrected covariance; refit before requesting --covariance-mode corrected"
                    .to_string(),
            );
        }
    }
    if let Some(covariance) = fit.beta_covariance() {
        return Ok(PredictionCovarianceBackend::from_dense(covariance.view()));
    }
    if let Some(backend) = factorized_covariance_fallback(fit) {
        // Surface the factorization error directly rather than swallowing it
        // and reporting the generic "model is missing either ..." message.
        // When the saved Hessian exists but cannot be factored (indefinite,
        // numerically degenerate, etc.) the user needs to see *why*, not a
        // confused "refit" instruction that doesn't match the real fault.
        return backend;
    }
    Err(
        "nonlinear posterior-mean prediction requires either covariance or a saved penalized Hessian; refit"
            .to_string(),
    )
}


/// Render the covariance-provenance suffix for `gam predict` from
/// RESULT-OWNED sources (#2296): what the evaluator actually consumed for the
/// point estimate and for the attached uncertainty. A request is never
/// evidence — callers must pass the sources reported by the prediction
/// result (or the mode they themselves resolved against the saved matrices,
/// where the CLI owns the selection and absence is a hard error).
///
/// Curved-link point predictions integrate the conditional posterior by
/// definition while the band may be smoothing-corrected; when the two
/// definitions differ the note names both, because one tag cannot represent
/// two sources.
pub(crate) fn covariance_provenance_note(
    point: Option<InferenceCovarianceMode>,
    uncertainty: Option<InferenceCovarianceMode>,
) -> String {
    match (point, uncertainty) {
        (None, None) => String::new(),
        (Some(source), None) | (None, Some(source)) => {
            format!(" [covariance={}]", source.as_str())
        }
        (Some(point_source), Some(uncertainty_source)) => {
            if point_source == uncertainty_source {
                format!(" [covariance={}]", point_source.as_str())
            } else {
                format!(
                    " [point-covariance={} uncertainty-covariance={}]",
                    point_source.as_str(),
                    uncertainty_source.as_str()
                )
            }
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
        )
        .expect("model summary must resolve its exact null likelihood");

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
