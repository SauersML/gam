use super::*;

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

/// The coefficient-covariance backend a saved-model prediction applies under
/// `mode`, selected by the library's own
/// [`UncertaintyCovarianceSource::select_uncertainty_backend`] — the selection
/// every library and Python predict path already uses — so `gam predict` reads
/// one definition instead of a hand-rolled copy of it: the dense `Vb`/`Vp`,
/// the factorized `Vp = Vb + B·Bᵀ` (#3283), `Vp = Vb` exactly when the fit has
/// no smoothing coordinates, and otherwise `Vb = φ·H⁻¹` from the saved
/// penalized Hessian, lifted through the coefficient gauge (#1561) and
/// truncated by an active constraint set (#2385). A fit that withheld its
/// covariance (#2718, #2985) or declined its posterior moments is refused with
/// its own reason, in both modes.
pub(crate) fn prediction_backend_from_model<'a>(
    model: &'a SavedModel,
    mode: InferenceCovarianceMode,
) -> Result<PredictionCovarianceBackend<'a>, String> {
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())?;
    fit.select_uncertainty_backend(fit.beta.len(), mode, "saved-model prediction")
        .map(|(backend, _)| backend)
        .map_err(|error| {
            // Refitting with the same binary publishes the same absent
            // correction, so the refusal names the modes that exist (#2677).
            let correction_absent = mode == InferenceCovarianceMode::SmoothingCorrected
                && fit.has_smoothing_coordinate()
                && fit.beta_covariance_corrected().is_none()
                && fit.smoothing_correction_factorized().is_none();
            if correction_absent {
                format!(
                    "{error}; request --covariance-mode conditional, or omit --covariance-mode \
                     to use the covariance the fit publishes"
                )
            } else {
                error.to_string()
            }
        })
}

/// The dense coefficient covariance [`prediction_backend_from_model`] applies,
/// for the consumers that take the matrix itself.
pub(crate) fn covariance_from_model(
    model: &SavedModel,
    mode: InferenceCovarianceMode,
) -> Result<Array2<f64>, String> {
    let backend = prediction_backend_from_model(model, mode)?;
    backend
        .apply_columns(&Array2::<f64>::eye(backend.nrows()))
        .map_err(|e| {
            format!("failed to recover the coefficient covariance from its saved source: {e}")
        })
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

/// The line a prediction prints when its posterior-mean point integrates a
/// covariance the fit withheld (gam#2985), so the note reaches the reader beside
/// the output it qualifies.
pub(crate) fn point_covariance_provenance_line(
    provenance: Option<&gam_predict::PointCovarianceProvenance>,
) -> Option<String> {
    provenance.map(|provenance| format!("note: {}", provenance.explain()))
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
    use gam::estimate::parametric_term_summary_rows;
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

        // Build the summary rows exactly as the CLI/report path does.
        let smooth_terms = smooth_term_summary_rows(
            &std_fit.design,
            &std_fit.fit,
            SummaryBlockOffset::default(),
        );
        let parametric_terms = parametric_term_summary_rows(
            &std_fit.design,
            &std_fit.resolvedspec,
            &std_fit.fit,
            SummaryBlockOffset::default(),
        );

        let edf_total = std_fit
            .fit
            .edf_total()
            .expect("a converged fit exposes the model total EDF");
        let ncols = std_fit.design.design.ncols() as f64;
        let tol = 1e-6;

        // The te() term must appear and carry a finite, non-negative EDF.
        assert!(
            !smooth_terms.is_empty(),
            "te(x, z) must produce at least one smooth-term summary row"
        );

        let mut per_term_sum = 0.0;
        for term in &smooth_terms {
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
        let parametric_dof = parametric_terms.len() as f64;
        let reconstructed = per_term_sum + parametric_dof;
        assert!(
            (reconstructed - edf_total).abs() <= 1e-4 * edf_total.max(1.0),
            "Σ per-term EDF (smooth {per_term_sum} + parametric {parametric_dof} = {reconstructed}) \
             must match model total EDF ({edf_total}) within tolerance"
        );
    }
}

#[cfg(test)]
mod point_covariance_provenance_tests {
    use super::*;

    /// gam#2985: `gam predict` prints this line after the predictions it
    /// qualifies whenever the resolved columns carry a provenance (the columns'
    /// half is pinned in gam-predict), and prints nothing extra otherwise.
    #[test]
    fn a_withheld_fit_prints_its_point_provenance_2985() {
        let declined = gam::estimate::CovarianceDeclined::
            BmsGeneratedRegressorResidualRepairChannelUnavailable {
                unavailable_channel: "the pin's missing channel".to_string(),
            };
        let provenance =
            gam_predict::PointCovarianceProvenance::ConditionalOnFittedLatentLaw { declined };
        let line = point_covariance_provenance_line(Some(&provenance))
            .expect("a withheld fit's prediction prints a note");
        assert!(
            line.starts_with("note: posterior mean conditional on the fitted latent law")
                && line.contains("the pin's missing channel"),
            "the note names what the point is conditional on and why: {line}"
        );
        assert_eq!(point_covariance_provenance_line(None), None);
    }
}
