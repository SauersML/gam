//! What a saved model reports about itself: the summary payload and the report
//! card, with one owner for every front door (#2899 F3).
//!
//! `gamfit`'s `Model.summary()` serializes [`saved_model_summary`] and its
//! `Model.report()` renders [`saved_model_report_input`]. `gam report` starts
//! from the same report input and adds only what the data it is given can show.
//! Before this module the summary payload was assembled inside the Python
//! binding, and each front door built its own report card. The two named the
//! model class and the family differently, and the Python report left out the
//! criterion certificate, the smoothing forensics, the anisotropic scales, the
//! measure-jet spectrum and the residual-cascade route.

use crate::inference::model::{FittedFamily, FittedModel, GroupMetadata, SavedDeploymentExtension};
use crate::survival::predict::fit_result_from_saved_model_for_prediction;
use gam_report::{
    AnisotropicScalesRow, AsymptoteRailRow, BasisCheckRow, CoefficientRow, CriterionCertificateRow,
    CriterionStationarityRow, EdfBlockRow, MeasureJetSpectrumRow, ReportInput,
    SmoothingForensicsRow,
};
use gam_solve::estimate::UnifiedFitResult;
use gam_terms::smooth::TermCollectionSpec;
use ndarray::Array2;
use serde::Serialize;

/// The model-class label a saved model publishes in its prediction, sampling
/// and summary payloads and on its report card.
pub fn prediction_model_class_label(model: &FittedModel) -> String {
    let payload = model.payload();
    match &payload.family_state {
        FittedFamily::Survival {
            survival_likelihood,
            ..
        } => match survival_likelihood
            .as_deref()
            .or(payload.survival_likelihood.as_deref())
        {
            _ if payload
                .survival_cause_count
                .is_some_and(|cause_count| cause_count > 1) =>
            {
                "competing risks survival".to_string()
            }
            Some("marginal-slope") => "survival marginal-slope".to_string(),
            Some("location-scale") => "survival location-scale".to_string(),
            None
            | Some("latent")
            | Some("latent-binary")
            | Some("transformation")
            | Some("weibull")
            | Some("royston-parmar")
            | Some(_) => model.predict_model_class().name().to_string(),
        },
        FittedFamily::LatentSurvival { .. } => "latent survival".to_string(),
        FittedFamily::LatentBinary { .. } => "latent binary".to_string(),
        FittedFamily::Standard { .. }
        | FittedFamily::LocationScale { .. }
        | FittedFamily::MarginalSlope { .. }
        | FittedFamily::TransformationNormal { .. } => {
            model.predict_model_class().name().to_string()
        }
    }
}

/// Synthesize a small representative data matrix from saved per-axis training
/// ranges, used only to rebuild the design *structure* (per-term coefficient
/// ranges, nullspace dimensions, penalty counts) for the summary smooth-term
/// table. The basis layout these fields describe is fixed by the frozen
/// `resolved_termspec`, not by the data values, so axis-spanning midpoints
/// reproduce the training-time block layout deterministically while remaining
/// inside the training bounding box (no extrapolation artefacts).
///
/// Columns named in `factor_levels` are categorical factors (main effects,
/// factor-aware interactions, `by=` factors, or factor-smooth groups): their
/// values must be bit-identical to a frozen level or the design rebuild fails
/// (#1370/#2787). Those columns are filled by cycling through the exact frozen
/// level bit patterns instead of axis midpoints, so every categorical design
/// block is represented. The row count grows to cover the widest factor so no
/// level is dropped.
fn representative_data_from_ranges(
    ranges: &[(f64, f64)],
    factor_levels: &std::collections::BTreeMap<usize, Vec<u64>>,
) -> Array2<f64> {
    const REP_ROWS_MIN: usize = 16;
    // Enough rows that the widest factor has every level represented.
    let max_levels = factor_levels.values().map(|lv| lv.len()).max().unwrap_or(0);
    let rep_rows = REP_ROWS_MIN.max(max_levels).max(1);
    let n_cols = ranges.len();
    let mut data = Array2::<f64>::zeros((rep_rows, n_cols));
    for (col, &(lo, hi)) in ranges.iter().enumerate() {
        if let Some(lv) = factor_levels.get(&col) {
            // Categorical column: cycle through the frozen level bit patterns so
            // every level appears and each value is a valid, bit-exact level.
            if !lv.is_empty() {
                for row in 0..rep_rows {
                    data[[row, col]] = f64::from_bits(lv[row % lv.len()]);
                }
                continue;
            }
        }
        let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi >= lo {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        for row in 0..rep_rows {
            let frac = if rep_rows > 1 {
                row as f64 / (rep_rows - 1) as f64
            } else {
                0.5
            };
            data[[row, col]] = lo + frac * (hi - lo);
        }
    }
    data
}

/// Dense, space-filling reconstruction of the training inputs for the Wald
/// design-whitening Gram (#2142). Denser than `representative_data_from_ranges`
/// so a high-basis univariate smooth gets a full-rank Gram, and with each
/// continuous column swept in an independent coprime order so multivariate
/// (tensor) margins are not collinear on the diagonal — the shared-ramp
/// representative grid samples only the diagonal line and would make every
/// tensor Gram rank-deficient. `rows` is forced to a power of two so every odd
/// per-column stride is coprime to it and therefore traverses the full
/// evenly-spaced grid. Categorical columns keep cycling their frozen levels.
fn whitening_data_from_ranges(
    ranges: &[(f64, f64)],
    factor_levels: &std::collections::BTreeMap<usize, Vec<u64>>,
    rows: usize,
) -> Array2<f64> {
    let rows = rows.max(2);
    let n_cols = ranges.len();
    let mut data = Array2::<f64>::zeros((rows, n_cols));
    for (col, &(lo, hi)) in ranges.iter().enumerate() {
        if let Some(lv) = factor_levels.get(&col) {
            if !lv.is_empty() {
                for row in 0..rows {
                    data[[row, col]] = f64::from_bits(lv[row % lv.len()]);
                }
                continue;
            }
        }
        let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi >= lo {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        // Odd stride is coprime to the power-of-two `rows`, so `(row*stride) %
        // rows` is a full-period permutation of the evenly-spaced grid — a
        // different one per column, breaking the diagonal collinearity.
        let stride = 2 * col + 1;
        for row in 0..rows {
            let idx = row.wrapping_mul(stride) % rows;
            let frac = idx as f64 / (rows - 1) as f64;
            data[[row, col]] = lo + frac * (hi - lo);
        }
    }
    data
}

/// Reconstruct the design-whitening Gram `X'X` for the summary Wald smooth test
/// from the frozen basis (#2142). The persisted summary path drops the fit's
/// inference block, so the exact weighted Gram `X'WX` is gone; mgcv itself
/// whitens the Wood (2013) statistic with the *unweighted* prediction-matrix
/// Gram, so `X'X` at representative inputs is the intended object (and it
/// reduces to `X'WX` for the Gaussian identity case). Returns the full `p×p`
/// Gram in the trained coefficient layout, or `None` when the rebuilt design's
/// column count does not match the trained coefficient count — a stale/mismatched
/// spec, in which case the test falls back to the un-whitened raw covariance.
fn summary_whitening_gram(
    spec: &gam_terms::smooth::TermCollectionSpec,
    ranges: &[(f64, f64)],
    factor_levels: &std::collections::BTreeMap<usize, Vec<u64>>,
    expected_ncols: usize,
) -> Option<Array2<f64>> {
    if expected_ncols == 0 {
        return None;
    }
    let rows = (4 * expected_ncols).max(64).next_power_of_two();
    let data = whitening_data_from_ranges(ranges, factor_levels, rows);
    let design = gam_terms::smooth::build_term_collection_design(data.view(), spec).ok()?;
    let x = design.design.to_dense();
    if x.ncols() != expected_ncols {
        return None;
    }
    // Lower triangle of `X'X` is the true Gram; the whitening eigendecomposition
    // reads only that side, so no explicit symmetrization is needed.
    Some(x.t().dot(&x))
}

#[cfg(test)]
mod whitening_gram_tests {
    //! Direct tests of the #2142 design-whitening-Gram reconstruction grid used
    //! when a summary is built from an inference-stripped (compact) model. The
    //! whitening math itself is covered by `gam-terms` `smooth_test` tests; here
    //! we only verify the reconstruction *inputs* are non-degenerate.
    use super::whitening_data_from_ranges;
    use std::collections::BTreeMap;

    #[test]
    fn dense_grid_spans_range_and_breaks_diagonal_collinearity() {
        let ranges = [(0.0_f64, 1.0_f64), (-2.0, 4.0)];
        let levels = BTreeMap::new();
        let data = whitening_data_from_ranges(&ranges, &levels, 64);
        assert_eq!(data.nrows(), 64);
        assert_eq!(data.ncols(), 2);
        // Each continuous column sweeps its full [lo, hi] range.
        for (c, &(lo, hi)) in ranges.iter().enumerate() {
            let col = data.column(c);
            let cmin = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let cmax = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!((cmin - lo).abs() < 1e-9, "col {c} min {cmin} != {lo}");
            assert!((cmax - hi).abs() < 1e-9, "col {c} max {cmax} != {hi}");
        }
        // The shared-ramp representative grid puts both columns on the same
        // diagonal (Pearson r == 1), collapsing every tensor Gram. The
        // independent coprime sweeps must break that: |r| strictly below 1.
        let c0 = data.column(0);
        let c1 = data.column(1);
        let m0 = c0.mean().unwrap();
        let m1 = c1.mean().unwrap();
        let (mut cov, mut v0, mut v1) = (0.0, 0.0, 0.0);
        for i in 0..64 {
            let (a, b) = (c0[i] - m0, c1[i] - m1);
            cov += a * b;
            v0 += a * a;
            v1 += b * b;
        }
        let r = cov / (v0.sqrt() * v1.sqrt());
        assert!(
            r.abs() < 0.9,
            "columns must not be collinear (diagonal grid), got r={r}"
        );
    }

    #[test]
    fn categorical_columns_cycle_every_frozen_level() {
        let ranges = [(0.0_f64, 1.0_f64), (0.0, 0.0)];
        let mut levels = BTreeMap::new();
        let lv = vec![1.0_f64.to_bits(), 2.0_f64.to_bits(), 3.0_f64.to_bits()];
        levels.insert(1usize, lv.clone());
        let data = whitening_data_from_ranges(&ranges, &levels, 64);
        let allowed: Vec<f64> = lv.iter().map(|&b| f64::from_bits(b)).collect();
        for i in 0..64 {
            let v = data[[i, 1]];
            assert!(
                allowed.iter().any(|&a| a == v),
                "row {i} value {v} is not a frozen level"
            );
        }
        for &a in &allowed {
            assert!(
                (0..64).any(|i| data[[i, 1]] == a),
                "frozen level {a} never appears"
            );
        }
    }
}

/// Build the mgcv-style per-smooth significance table for the FFI summary.
///
/// This is marshalling, not a second summary: the table comes from
/// `gam_solve::estimate::smooth_term_summary_rows`, the one walk of the
/// fit's penalty layout that the in-process CLI summary also uses (#2470). Its
/// contract is unchanged — random-effect smooths report `edf` only (their
/// boundary variance-component test is not a Wald χ²), and penalized smooth
/// terms get the Wood (2013) rank-truncated Wald statistic and p-value.
///
/// The "Mirrors `main.rs::build_model_summary`'s smooth-term loop" this
/// sentence used to open with was accurate and was the problem: a comment
/// asserting parity between two implementations is the marker for a wiring that
/// should have had one.
///
/// `Ok(rows)` is the table (empty exactly when the model has no smooth or
/// random-effect terms). `Err(reason)` is every other way the table can be
/// absent — a model saved without `resolved_termspec` or training feature
/// ranges, a frozen spec that fails validation, or a frozen-basis design
/// replay that fails — and the caller publishes that reason as
/// `smooth_terms_unavailable` next to the empty table. `summary()` still
/// succeeds; what it no longer does is report an absence without its reason.
/// The replay failure in particular used to be swallowed to an empty vector,
/// which is how a categorical main effect erased every co-fitted `s(x)` row
/// (#2787): the table looked exactly like "no smooth terms".
fn summary_smooth_terms(
    model: &FittedModel,
    fit: &gam_solve::estimate::UnifiedFitResult,
) -> Result<Vec<SummarySmoothTermRow>, String> {
    let payload = model.payload();
    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return Err("model was saved without `resolved_termspec`; refit to recover the \
                    per-smooth table"
            .to_string());
    };
    spec.validate_frozen("resolved_termspec")
        .map_err(|err| format!("saved `resolved_termspec` failed validation: {err}"))?;
    let Some(ranges) = payload.training_feature_ranges.as_ref() else {
        return Err("model was saved without training feature ranges; refit to recover the \
                    per-smooth table"
            .to_string());
    };
    let Some(headers) = payload.training_headers.as_ref() else {
        return Err("model was saved without training headers; refit to recover the \
                    per-smooth table"
            .to_string());
    };
    if ranges.len() != headers.len() {
        return Err(format!(
            "saved training feature ranges ({}) and headers ({}) disagree on the column count",
            ranges.len(),
            headers.len()
        ));
    }
    // Every categorical column — including a fixed main effect represented by
    // a frozen random-effect block — must carry a valid saved level or the
    // design rebuild fails. That failure was swallowed to an EMPTY table for
    // the whole model, erasing co-fitted `s(x)` rows too (#1370/#2787).
    // Synthesize representative data from the term collection's authoritative
    // factor vocabulary so every categorical carrier replays coherently.
    let factor_levels = spec.frozen_factor_levels_by_col();
    let data = representative_data_from_ranges(ranges, &factor_levels);
    let design = gam_terms::smooth::build_term_collection_design(data.view(), spec)
        .map_err(|err| format!("frozen-basis design replay failed: {err}"))?;

    // Wood (2013) design-whitening metric for the Wald smooth test (#2142).
    // Prefer the fit's exact weighted Gram `X'WX` when the inference block
    // survived; on the persisted summary path (inference dropped) reconstruct
    // the unweighted `X'X` from the frozen basis. `None` → un-whitened fallback.
    let reconstructed_gram = if fit.weighted_gram().is_none() {
        summary_whitening_gram(spec, ranges, &factor_levels, design.design.ncols())
    } else {
        None
    };
    let whitening_gram_full: Option<&Array2<f64>> =
        fit.weighted_gram().or(reconstructed_gram.as_ref());
    // The walk over the fit's flat penalty layout — the `LinearTermRidge`
    // prologue, the random-effect blocks that own no entry, the block-local →
    // global coefficient shift, the per-term influence trace, and the Wood test
    // with its reference distribution — is ONE accounting, shared with the
    // in-process CLI summary (#2470). Both surfaces spelled it out, so #1219,
    // #1277, #1360, #1368 and #1372 each had to be landed twice; the comment
    // this replaces recorded its own copy of #1368 as "fixed on the in-process
    // path but never propagated here". What genuinely differs on this persisted
    // path is the EVIDENCE — a frozen-basis replay instead of the training
    // design, so the whitening Gram is reconstructed rather than exact — and
    // that is the only thing handed over. Both reference-distribution inputs
    // (`wald_residual_degrees_of_freedom`, `wald_scale_is_estimated`) are read
    // off the fit inside that walk, which is where `fd998d957` put them.
    let rows =
        gam_solve::estimate::smooth_term_summary_rows(&design, spec, fit, whitening_gram_full);
    Ok(rows
        .into_iter()
        .map(|row| SummarySmoothTermRow {
            name: row.name,
            edf: row.edf,
            ref_df: row.ref_df,
            chi_sq: row.chi_sq,
            p_value: row.pvalue,
        })
        .collect())
}

/// Read the fitted κ̂ off every `curv(...)` constant-curvature smooth in the
/// resolved (fitted) spec (#944). κ̂ is the outer optimiser's argmin of the
/// profiled criterion over κ; it lives in the basis spec the fit wrote back, so
/// surfacing it is a pure read — no refit, no original data needed. The verdict
/// here is the point-estimate sign tag only; the level-α geometry decision
/// (and the κ = 0 flatness p-value) is the profile-CI from
/// `curvature_inference_json`, which re-profiles `V_p(κ)` against the data.
fn summary_curvature_estimands(model: &FittedModel) -> Vec<SummaryCurvatureRow> {
    use gam_terms::smooth::SmoothBasisSpec;
    let payload = model.payload();
    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return Vec::new();
    };
    let mut out = Vec::<SummaryCurvatureRow>::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let SmoothBasisSpec::ConstantCurvature { spec: cc, .. } = &term.basis else {
            continue;
        };
        if !cc.kappa.is_finite() {
            continue;
        }
        // Sign-of-κ̂ point tag. The flatness band is a fixed, small absolute
        // window on the curvature scale — a screening label only; the
        // statistically-honest "flat vs curved" call is the κ = 0 LR test.
        let geometry = if cc.kappa > 1e-6 {
            "spherical"
        } else if cc.kappa < -1e-6 {
            "hyperbolic"
        } else {
            "flat"
        };
        out.push(SummaryCurvatureRow {
            name: term.name.clone(),
            term_idx,
            kappa_hat: cc.kappa,
            geometry,
        });
    }
    out
}

/// Canonical fitted quantities reconstructed from a spline-scan model's saved
/// `SplineScanFit` (#1046).
///
/// A scan-routed model (the single-1-D-smooth, Gaussian-identity shape that
/// `spline_scan_fast_path` diverts to the exact O(n) state-space smoother)
/// carries **no dense `fit_result`** — by design the smoother keeps only the
/// per-knot posterior, never a dense design/Gram. So the FFI summary surface
/// reconstructs exactly what the smoother retained: the selected smoothing
/// parameter, the effective d.o.f. (tr S), the diffuse REML score, the ordinary
/// Gaussian log-likelihood, the profiled scale, and the recovered deviance.
/// This mirrors the numbers the CLI fit log already prints from the same state.
///
/// The dense per-coefficient β / covariance is deliberately NOT materialized:
/// the smoother's natural parameters are the per-knot function values, of which
/// there are ~`n` (the scan exists precisely to avoid an O(n) design and an
/// O(n²) covariance). The summary therefore reports the model the way
/// `summary.gam` does — a parametric block (empty here; the smoother absorbs the
/// polynomial null space into the smooth) plus a smooth-terms table keyed on EDF
/// — rather than dumping ~`n` basis coefficients.
pub struct ScanIntrospection {
    pub feature_column: String,
    /// Original row count before tied abscissae were pooled into knots.
    pub training_sample_size: usize,
    /// Effective degrees of freedom (tr S), strictly between the polynomial
    /// null-space dimension `order` and `n`.
    pub edf: f64,
    /// Selected smoothing parameter `λ` (always positive).
    pub lambda: f64,
    /// Diffuse REML expressed as a COST (lower is better): the negative
    /// restricted log marginal likelihood, on the same sign convention as the
    /// dense `reml_score`. It is exact up to the λ-free additive constant the
    /// concentrated criterion drops, which is what makes it an exact criterion
    /// for the smoother's own λ selection. That constant depends on (n, order,
    /// knots), so it does NOT cancel when comparing different fits — treat this
    /// as a within-fit quantity and prefer held-out predictive metrics for
    /// cross-model comparison.
    pub reml_cost: f64,
    /// Fully normalized weighted Gaussian log-likelihood at the fitted mean and
    /// profiled scale, on the conditional-AIC ranking scale.
    pub log_likelihood: f64,
    /// Gaussian deviance — the weighted residual sum of squares.
    pub deviance: f64,
    /// Number of pooled knots (the smoother's natural coefficient count).
    pub n_knots: usize,
}

/// Reconstruct the canonical fitted quantities for a spline-scan model, or
/// `Ok(None)` for a dense model that should follow the standard `fit_result`
/// path (#1046).
pub fn scan_introspection(model: &FittedModel) -> Result<Option<ScanIntrospection>, String> {
    let Some((feature_column, fit)) = model.saved_spline_scan().map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    Ok(Some(ScanIntrospection {
        feature_column: feature_column.to_string(),
        training_sample_size: fit.training_sample_size(),
        edf: fit.edf(),
        lambda: fit.lambda(),
        reml_cost: -fit.restricted_loglik,
        log_likelihood: fit.log_likelihood,
        deviance: fit.deviance(),
        n_knots: fit.knots.len(),
    }))
}

/// Display label for the single smooth a scan model carries, e.g. `s(x)`.
pub fn scan_smooth_label(scan: &ScanIntrospection) -> String {
    format!("s({})", scan.feature_column)
}

/// Build the canonical FFI summary payload for a scan-routed model (#1046):
/// scalar fitted quantities plus a one-row smooth table keyed on EDF. The
/// parametric coefficient block is empty (the smoother absorbs the polynomial
/// null space) and no dense covariance is emitted — keeping `summary()` O(1) in
/// `n` regardless of how many knots the smoother spans.
fn scan_summary_payload(model: &FittedModel, scan: &ScanIntrospection) -> SummaryPayload {
    let smooth_terms = vec![SummarySmoothTermRow {
        name: scan_smooth_label(scan),
        edf: scan.edf,
        ref_df: scan.edf,
        // The rank-truncated Wald smooth test needs the joint coefficient
        // covariance, which the O(n) smoother does not retain; report EDF only,
        // as `summary.gam` does for terms whose Wald test is unavailable.
        chi_sq: None,
        p_value: None,
    }];
    SummaryPayload {
        formula: model.payload().formula.clone(),
        family_name: model.display_family_name(),
        model_class: prediction_model_class_label(model),
        group_metadata: model.payload().group_metadata.clone(),
        deployment_extensions: model.payload().deployment_extensions.clone(),
        // Read from the payload rather than hardcoded empty (#2774). The O(n)
        // scan route returns before the fit-time adequacy seam, so today this
        // is empty — but "empty because the fit recorded none" and "empty
        // because this constructor forgot" are different states, and only the
        // first one keeps saying the truth if the scan route ever records a
        // row.
        basis_checks: summary_basis_checks(model),
        deviance: scan.deviance,
        log_likelihood: Some(scan.log_likelihood),
        n_obs: Some(scan.training_sample_size),
        // The scan does not compute the penalized-Hessian null-space logdet the TK
        // normalizer needs, so it has no comparable criterion: the raw cost is
        // published as `raw_reml_score` only. `evidence()` ranks on conditional
        // AIC and stays well-defined.
        reml_score: None,
        raw_reml_score: Some(scan.reml_cost),
        reml_score_unavailable: Some(
            gam_solve::estimate::NO_COMPARABLE_CRITERION_WITHOUT_NULL_SPACE,
        ),
        null_space_logdet: None,
        null_dim: None,
        iterations: 0,
        edf_total: Some(scan.edf),
        edf_rank_bound: Vec::new(),
        lambdas: vec![scan.lambda],
        coefficients: Vec::new(),
        smooth_terms,
        smooth_terms_unavailable: None,
        covariance_kind: None,
        covariance_n: None,
        covariance_flat: None,
        coefficient_se_source: None,
        // Scan-routed (O(n) 1D spline) models carry no `curv(...)` curvature
        // smooths, so there are no curvature estimands to report.
        curvature_estimands: Vec::new(),
        // The O(n) smoother solves no inner P-IRLS and no outer stationarity
        // equation, so there is no termination to certify. Reporting a
        // fabricated "certified" block here would be the exact confusion
        // #2411 exists to remove.
        convergence: None,
    }
}

/// Project the fit's sealed convergence evidence onto the summary surface
/// (#2411).
///
/// Strictly a read: every value comes from the `FitConvergenceEvidence` the
/// fitted result already owns — the same object the mint gate consulted and
/// the same one `Deserialize` revalidates on load. Recomputing any of it here
/// would create a second source of truth that could drift from the verdict
/// that allowed the fit to exist.
fn summary_convergence(fit: &gam_solve::estimate::UnifiedFitResult) -> SummaryConvergence {
    use gam_solve::rho_optimizer::OuterStationarityCertificate;

    let evidence = fit.convergence_evidence();
    let certificate = evidence.outer_certificate();
    let outer = certificate.map(|certificate| SummaryOuterCertificate {
        // The residual means a different thing in each arm, so the kind
        // travels with the numbers rather than being inferred from them.
        kind: match &certificate.stationarity {
            OuterStationarityCertificate::AnalyticGradient { .. } => "analytic_gradient",
            OuterStationarityCertificate::FixedPoint { .. } => "fixed_point",
            OuterStationarityCertificate::AsymptoteRail { .. } => "asymptote_rail",
        }
        .to_string(),
        gradient_norm: certificate.stationarity.raw_norm(),
        projected_gradient_norm: certificate.stationarity.projected_norm(),
        stationarity_bound: certificate.stationarity.bound(),
        hessian_psd: certificate.hessian_psd(),
        lambdas_railed: certificate.lambdas_railed.clone(),
    });
    SummaryConvergence {
        // No outer coordinate was optimized in the `Fixed` arm, so there is no
        // outer equation that could fail: the converged inner mode is the
        // complete proof, and the fit's existence already attests to it.
        certified: certificate.is_none_or(|certificate| certificate.certifies()),
        inner_status: evidence.inner_status().label().to_string(),
        outer_iterations: evidence.outer_iterations(),
        outer,
    }
}

/// The #2774 basis-adequacy rows a fitted model carries, mapped onto the FFI
/// summary shape. A pure read of persisted fit-time evidence — no refit, no
/// recomputation, and no fabrication when the fit recorded none.
fn summary_basis_checks(model: &FittedModel) -> Vec<SummaryBasisCheckRow> {
    model
        .payload()
        .basis_adequacy
        .iter()
        .map(|row| SummaryBasisCheckRow {
            name: row.name.clone(),
            term_idx: row.term_idx,
            basis_dim: row.basis_dim,
            nullspace_dim: row.nullspace_dim,
            edf: row.edf,
            enrichment_dim: row.enrichment_dim,
            enrichment_rank: row.enrichment_rank,
            statistic: row.statistic,
            p_value: row.p_value,
            provenance: row.provenance.label(),
        })
        .collect()
}

/// The summary payload of a saved model: its scalar fitted quantities, the
/// coefficient table and covariance, the per-smooth significance table, the
/// curvature and basis-adequacy rows the fit recorded, and the fit's own
/// convergence certificate. A spline-scan model reports what the O(n) smoother
/// retained.
pub fn saved_model_summary(model: &FittedModel) -> Result<SummaryPayload, String> {
    if let Some(scan) = scan_introspection(model)? {
        return Ok(scan_summary_payload(model, &scan));
    }
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let (smooth_terms, smooth_terms_unavailable) = match summary_smooth_terms(model, &fit) {
        Ok(rows) => (rows, None),
        Err(reason) => (Vec::new(), Some(reason)),
    };
    // Definition-consistent coefficient uncertainty (#2296): the SE column,
    // the exported covariance matrix, and their labels all come from ONE
    // covariance definition. Independently selected `corrected.or(conditional)`
    // fields could pair corrected SEs with a conditional matrix (or vice
    // versa) on fits that persist only one half of a definition.
    let display_uncertainty = fit.display_coefficient_uncertainty();
    let standard_errors = display_uncertainty
        .as_ref()
        .map(|view| &view.standard_errors);
    let covariance = display_uncertainty.as_ref().and_then(|view| {
        view.covariance
            .map(|cov| (view.definition.as_str().to_string(), cov))
    });
    let coefficients = fit
        .beta
        .iter()
        .enumerate()
        .map(|(index, estimate)| SummaryCoefficientRow {
            index,
            estimate: *estimate,
            std_error: standard_errors.and_then(|values| values.get(index).copied()),
        })
        .collect();
    // A fit with no criterion normalizes to no criterion: the Tierney-Kadane
    // term is a correction TO a score, not a score, so applying it to a stand-in
    // would manufacture exactly the comparable number this fit cannot have.
    let raw_reml_score = fit.reml_score();
    let reml_score = fit
        .comparable_reml_score()
        .map_err(|err| format!("failed to compute comparable REML score: {err}"))?;
    Ok(SummaryPayload {
        formula: model.payload().formula.clone(),
        family_name: model.display_family_name(),
        model_class: prediction_model_class_label(model),
        group_metadata: model.payload().group_metadata.clone(),
        deployment_extensions: model.payload().deployment_extensions.clone(),
        deviance: fit.deviance,
        // Declined at the same boundary and for the same reason as the
        // criterion: with `φ̂ = 0` there is no normalized density, and the
        // stored `0.0` is the `UserProvided` tag saying so. Emitting it as a
        // number lets `compare_models` rank an exact fit on `−2·0 + 2·edf`.
        log_likelihood: fit.reported_log_likelihood(),
        n_obs: Some(fit.training_sample_size()),
        reml_score,
        raw_reml_score,
        reml_score_unavailable: match (raw_reml_score, reml_score) {
            (None, _) => Some(gam_solve::estimate::NO_CRITERION_AT_EXACT_FIT),
            (Some(_), None) => Some(gam_solve::estimate::NO_COMPARABLE_CRITERION_WITHOUT_NULL_SPACE),
            (Some(_), Some(_)) => None,
        },
        null_space_logdet: fit.artifacts.null_space_logdet,
        null_dim: fit.artifacts.null_space_dim.map(|dim| dim as f64),
        iterations: fit.outer_iterations,
        edf_total: fit.edf_total(),
        edf_rank_bound: fit.edf_rank_bound().to_vec(),
        lambdas: fit.lambdas.to_vec(),
        coefficients,
        smooth_terms,
        smooth_terms_unavailable,
        curvature_estimands: summary_curvature_estimands(model),
        basis_checks: summary_basis_checks(model),
        covariance_kind: covariance.as_ref().map(|(kind, _)| kind.clone()),
        covariance_n: covariance.as_ref().map(|(_, cov)| cov.nrows()),
        covariance_flat: covariance.map(|(_, cov)| cov.iter().copied().collect()),
        coefficient_se_source: display_uncertainty.map(|view| view.definition.as_str().to_string()),
        convergence: Some(summary_convergence(&fit)),
    })
}

#[derive(Serialize)]
pub struct SummaryCoefficientRow {
    pub index: usize,
    pub estimate: f64,
    pub std_error: Option<f64>,
}

/// Per-smooth significance row for the FFI summary — the canonical mgcv
/// `summary.gam` smooth-term table (`edf`, reference d.f., test statistic, and
/// p-value). Random-effect smooths report only `edf` (their boundary
/// variance-component test is not a Wald χ²); penalized smooth terms carry the
/// Wood (2013) rank-truncated Wald `chi_sq` / `p_value`. The shape mirrors the
/// CLI's `SmoothTermSummary`.
///
/// This `p_value` is the *first-order* Wald reference. The summary table is
/// built from a saved model without the training rows, so it cannot run the
/// per-term constrained refits the second-order test needs. The
/// **second-order-accurate, Bartlett-corrected likelihood-ratio** p-value is
/// computed on demand by `smooth_term_lr_inference_json` (Python
/// `Model.smooth_significance(data)`), which auto-applies the exact Lawley
/// factor whenever the family carries closed-form cumulant jets (#939/#1063).
#[derive(Serialize)]
pub struct SummarySmoothTermRow {
    pub name: String,
    pub edf: f64,
    pub ref_df: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chi_sq: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_value: Option<f64>,
}

/// The fitted curvature estimate for one `curv(...)` constant-curvature smooth
/// (#944). κ̂ is read directly off the resolved (fitted) basis spec — the
/// outer optimiser's argmin of the profiled criterion over κ — so it is an
/// estimate the fit ALREADY produced and is surfaced with zero refit. The
/// profile-CI and the interior κ = 0 flatness LR test require re-profiling
/// `V_p(κ)` against the original data and are produced on demand by the
/// `curvature_inference_json` entry (which the data-carrying caller invokes).
#[derive(Serialize)]
pub struct SummaryCurvatureRow {
    /// `curv(...)` term name.
    pub name: String,
    /// Smooth-term index of the constant-curvature term.
    pub term_idx: usize,
    /// Fitted signed sectional curvature κ̂.
    pub kappa_hat: f64,
    /// Sign-of-κ̂ geometry tag: `"spherical"` (κ̂>0), `"flat"` (κ̂≈0), or
    /// `"hyperbolic"` (κ̂<0). A point estimate only — the level-α verdict comes
    /// from the profile-CI endpoints via `curvature_inference_json`.
    pub geometry: &'static str,
}


/// One smooth term's #2774 basis-adequacy row for the FFI summary.
///
/// Measured AT FIT TIME and persisted with the model, because the residual
/// lack-of-fit score needs the converged IRLS row state (weights, working
/// response, linear predictor) and a saved model carries none of it. A caller
/// holding the training rows can recompute the whole table — including on a
/// model saved before this existed — via `basis_adequacy_json` (Python
/// `Model.basis_check(data)`), which refits at the frozen spec first.
///
/// `p_value` is present exactly when `provenance == "radial_enrichment"`; every
/// other provenance value NAMES the evidence that was missing, so "adequate"
/// and "not measured" are never confusable.
#[derive(Serialize)]
pub struct SummaryBasisCheckRow {
    pub name: String,
    pub term_idx: usize,
    /// Realized coefficient width `k'` of the term.
    pub basis_dim: usize,
    /// Dimension of the term's JOINT penalty null space — the directions no
    /// penalty on it touches — so `basis_dim − nullspace_dim` is the penalizable
    /// capacity. `0` for every double-penalized smooth, the radial family
    /// included: a Duchon term carries an RKHS curvature Gram AND a
    /// complementary trend ridge on its polynomial block, so nothing in it is
    /// completely unshrunk. That block is only WEAKLY penalized, though, so it
    /// carries most of the term's EDF and makes an EDF-vs-`basis_dim` reading
    /// look saturated on a fit whose problem is its basis's SPAN (#2774).
    pub nullspace_dim: usize,
    /// The term's effective degrees of freedom, carried BESIDE `basis_dim` and
    /// `nullspace_dim` because the three only mean anything together. The
    /// penalized occupancy is `edf − nullspace_dim` out of a capacity of
    /// `basis_dim − nullspace_dim`; the natural summary-table reading of `edf`
    /// against `basis_dim` counts the always-full null space as evidence of
    /// saturation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edf: Option<f64>,
    /// Width of the higher-resolution alternative the residuals were tested
    /// against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enrichment_dim: Option<usize>,
    /// Estimable alternative directions after the fitted design was projected
    /// out — the test's reference d.f., and a direct measure of how much NEW
    /// resolution the alternative carried.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enrichment_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistic: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_value: Option<f64>,
    pub provenance: &'static str,
}

#[derive(Serialize)]
pub struct SummaryPayload {
    pub formula: String,
    pub family_name: String,
    pub model_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_metadata: Option<GroupMetadata>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deployment_extensions: Vec<SavedDeploymentExtension>,
    pub deviance: f64,
    /// Reported log-likelihood at the converged mode. Carried so
    /// `compare_models` can form the Occam-penalised conditional AIC it ranks on
    /// (issue #1362). `None` means the fit has no normalized likelihood at the
    /// exact zero-dispersion boundary; it never means "not recorded".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_likelihood: Option<f64>,
    /// Number of original rows the fit was trained on. Carried so `compare_models`
    /// can REFUSE to rank fits made on different-sized (hence different) data:
    /// `−2·loglik` / REML evidence grow with `n`, so a score gap between two fits
    /// with different `n` is not a Bayes factor (#1384 sibling — the same
    /// fail-loud contract as the family guard). Model summaries source it only
    /// from their required persisted training-sample-size authority; `None` is
    /// reserved for non-model summary kinds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_obs: Option<usize>,
    /// Cross-model comparable criterion: `raw_reml_score` plus the rank-aware
    /// Tierney-Kadane normalizer over the penalty null space.
    ///
    /// `null` in two cases, each named by `reml_score_unavailable`, and no
    /// surface substitutes a number for it:
    /// - the fit has **no** criterion at all (`raw_reml_score` is `null` too),
    ///   which is a different statement from "not recorded": an
    ///   exactly-interpolating Gaussian fit has `φ̂ = 0`, so its restricted
    ///   likelihood is unbounded and every score derived from it is undefined
    ///   (#2595);
    /// - the fit has a raw criterion but no penalty null-space metadata, so the
    ///   normalizer cannot be formed and no score comparable across fits exists
    ///   (#2627).
    pub reml_score: Option<f64>,
    /// The outer optimizer's own criterion value, un-normalized. `null` exactly
    /// when the fit has no criterion at all.
    pub raw_reml_score: Option<f64>,
    /// Why `reml_score` is `null`. Present iff `reml_score` is `null`, so a
    /// reader never sees an absence without its reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reml_score_unavailable: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub null_space_logdet: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub null_dim: Option<f64>,
    pub iterations: usize,
    pub edf_total: Option<f64>,
    /// Each penalty block's rank-bound status beside the EDF fields (#2901). An
    /// `Uncertified` or `NotAssessed` block's trace and EDF are published unclamped,
    /// and so is `edf_total` when any block is not certified. Empty when the fit
    /// recorded none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub edf_rank_bound: Vec<gam_solve::estimate::EdfRankBound>,
    pub lambdas: Vec<f64>,
    pub coefficients: Vec<SummaryCoefficientRow>,
    /// Per-smooth significance table (mgcv-style). Empty when the model has no
    /// smooth/random-effect terms — and ONLY then without a reason: every other
    /// absence names itself in `smooth_terms_unavailable`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub smooth_terms: Vec<SummarySmoothTermRow>,
    /// Why `smooth_terms` could not be built: the model was saved without
    /// `resolved_termspec` / training feature ranges, the frozen spec failed
    /// validation, or the frozen-basis design replay failed. Present iff the
    /// table was omitted for one of those reasons, so an absence is never
    /// reported without its reason — the same contract as
    /// `reml_score_unavailable`. A replay failure used to be swallowed into an
    /// empty table indistinguishable from "no smooth terms", which is how a
    /// categorical main effect erased every co-fitted `s(x)` row (#2787).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smooth_terms_unavailable: Option<String>,
    /// Fitted curvature estimates for any `curv(...)` constant-curvature smooths
    /// (#944). Empty when the model has no constant-curvature term. κ̂ is the
    /// estimate the fit already produced; the CI and flatness p-value are
    /// produced on demand by `curvature_inference_json`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub curvature_estimands: Vec<SummaryCurvatureRow>,
    /// Per-smooth basis-adequacy evidence (#2774): does the fit's residual still
    /// carry structure in each smooth's covariates that its realized basis
    /// cannot represent? Empty for a model saved before the check existed, or
    /// fitted through a route that retains no IRLS row state. The convergence
    /// certificate below covers the OPTIMIZER only — it makes no statement about
    /// the adequacy of the basis it converged on, and this table is where that
    /// separate question is answered.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub basis_checks: Vec<SummaryBasisCheckRow>,
    pub covariance_kind: Option<String>,
    pub covariance_n: Option<usize>,
    pub covariance_flat: Option<Vec<f64>>,
    /// Exact covariance definition behind the coefficient `std_error` column
    /// (#2296): `"conditional"` or `"smoothing-corrected"`, recorded from the
    /// definition-consistent pair the summary actually consumed. `None` when
    /// the fit carries no coefficient standard errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coefficient_se_source: Option<String>,
    /// The convergence certificate the fit itself carries (#2411). `None` only
    /// for routes that solve no optimizer whose termination could be certified
    /// (the O(n) spline scan).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub convergence: Option<SummaryConvergence>,
}

/// How the optimization that produced this fit terminated.
///
/// Every field is read from the sealed `FitConvergenceEvidence` the fitted
/// result owns; nothing here is recomputed. That matters: the evidence object
/// is the same one the mint gate consulted and the same one `Deserialize`
/// revalidates on load, so this block cannot drift into disagreeing with the
/// verdict that allowed the fit to exist.
#[derive(Serialize)]
pub struct SummaryConvergence {
    /// The verdict the mint gate used. A returned fit always carries a
    /// certified evidence object, so this is `true` on any model that exists —
    /// it is reported so consumers can assert on it rather than infer it.
    pub certified: bool,
    /// Terminal status of the certified inner P-IRLS solve.
    pub inner_status: String,
    /// Outer iterations covered by the proof.
    pub outer_iterations: usize,
    /// `None` when no smoothing coordinate was optimized: there is no outer
    /// stationarity equation to solve, which is a different statement from a
    /// projected gradient that happened to be zero.
    pub outer: Option<SummaryOuterCertificate>,
}

/// The outer (smoothing-parameter) stationarity certificate.
///
/// `projected_gradient_norm` and `stationarity_bound` are in the same gauge, so
/// a consumer can check `projected_gradient_norm <= stationarity_bound`
/// directly, or impose a tolerance of their own without parsing a log.
#[derive(Serialize)]
pub struct SummaryOuterCertificate {
    /// Which stationarity equation was certified: `"analytic_gradient"`,
    /// `"fixed_point"`, or `"asymptote_rail"`. The residual means different
    /// things across the three, so the kind travels with the numbers.
    pub kind: String,
    /// Unprojected residual norm (gradient norm, fixed-point residual, or the
    /// interior projected gradient for an asymptote rail).
    pub gradient_norm: f64,
    /// Residual after projecting onto the feasible directions at the optimum.
    pub projected_gradient_norm: f64,
    /// Bound the projected residual had to clear.
    pub stationarity_bound: f64,
    /// Whether the final outer Hessian was positive semidefinite. `None` when
    /// the solver tracked no final Hessian.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_psd: Option<bool>,
    /// Smoothing coordinates pinned at a box bound at the optimum.
    pub lambdas_railed: Vec<usize>,
}


/// The report card of a saved model: every quantity the model itself records,
/// and no data-dependent diagnostic. `model_path` is the label the card prints
/// for the model. `gam report` adds the diagnostics of the data it is given.
pub fn saved_model_report_input(
    model: &FittedModel,
    model_path: String,
) -> Result<ReportInput, String> {
    if let Some((feature_column, scan)) = model
        .saved_spline_scan()
        .map_err(|err| err.to_string())?
    {
        return Ok(spline_scan_report_input(
            model,
            model_path,
            feature_column,
            &scan,
        ));
    }
    if let Some((feature_columns, fit)) = model
        .saved_residual_cascade()
        .map_err(|err| err.to_string())?
    {
        return Ok(residual_cascade_report_input(
            model,
            model_path,
            feature_columns,
            &fit,
        ));
    }
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    // Total EDF: shown on the summary card and used as the residual degrees
    // of freedom in the dispersion estimates behind the report residuals.
    let edf_total = model
        .unified()
        .and_then(|unified| unified.edf_total())
        .unwrap_or_else(|| fit.edf_total().unwrap_or(0.0));
    // Definition-consistent SE column (#2296): corrected-preferred, but never
    // an unlabeled mix of covariance definitions.
    let display_uncertainty = fit.display_coefficient_uncertainty();
    let standard_errors = display_uncertainty
        .as_ref()
        .map(|view| &view.standard_errors);
    let coefficients = fit
        .beta
        .iter()
        .copied()
        .enumerate()
        .map(|(index, estimate)| CoefficientRow {
            index,
            estimate,
            std_error: standard_errors.and_then(|values| values.get(index).copied()),
        })
        .collect();
    let edf_blocks: Vec<EdfBlockRow> = match model.unified() {
        Some(unified) => unified
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| EdfBlockRow {
                index,
                edf: block.edf,
                role: Some(block.role.name().to_string()),
            })
            .collect(),
        None => fit
            .edf_by_block()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, edf)| EdfBlockRow {
                index,
                edf,
                role: None,
            })
            .collect(),
    };
    let mut notes = Vec::new();
    if let Some(unified) = model.unified() {
        if unified.blocks.len() > 1 {
            let role_labels: Vec<&str> =
                unified.blocks.iter().map(|block| block.role.name()).collect();
            notes.push(format!("Block roles: {}", role_labels.join(", ")));
        }
        notes.push(format!(
            "Outer iterations: {} (status: {})",
            unified.outer_iterations,
            unified.convergence_evidence().inner_status().label()
        ));
        notes.push(format!(
            "Log-likelihood: {:.4}, penalized objective: {}",
            unified.log_likelihood,
            gam_report::criterion_display(unified.penalized_objective())
        ));
    }
    let criterion_certificate = fit.artifacts.criterion_certificate.as_ref().map(|cert| {
        use gam_solve::model_types::OuterStationarityCertificate;
        let stationarity = match &cert.stationarity {
            OuterStationarityCertificate::AnalyticGradient {
                grad_norm,
                projected_grad_norm,
                bound,
                rung,
            } => CriterionStationarityRow::AnalyticGradient {
                grad_norm: *grad_norm,
                projected_grad_norm: *projected_grad_norm,
                bound: *bound,
                rung: (rung.label.clone(), rung.derived_standard),
            },
            OuterStationarityCertificate::FixedPoint {
                residual_inf_norm,
                projected_residual_inf_norm,
                bound,
                rung,
                covered_coordinates,
            } => CriterionStationarityRow::FixedPoint {
                residual_inf_norm: *residual_inf_norm,
                projected_residual_inf_norm: *projected_residual_inf_norm,
                bound: *bound,
                rung: (rung.label.clone(), rung.derived_standard),
                covered_coordinates: *covered_coordinates,
            },
            OuterStationarityCertificate::AsymptoteRail {
                interior_projected_grad_norm,
                bound,
                rung,
                rails,
            } => CriterionStationarityRow::AsymptoteRail {
                interior_projected_grad_norm: *interior_projected_grad_norm,
                bound: *bound,
                rung: (rung.label.clone(), rung.derived_standard),
                rails: rails
                    .iter()
                    .map(|rail| AsymptoteRailRow {
                        index: rail.index,
                        upper: rail.side
                            == gam_solve::rho_optimizer::asymptote_certificate::AsymptoteSide::Upper,
                        tail_constant: rail.tail_constant,
                        value_gap: rail.value_gap,
                        estimand_travel_bound: rail.estimand_travel_bound,
                    })
                    .collect(),
            },
        };
        CriterionCertificateRow {
            stationarity,
            hessian_psd: cert.hessian_psd(),
            lambdas_railed: cert.lambdas_railed.clone(),
            railed_facts: cert
                .railed_facts
                .iter()
                .map(|fact| (fact.index, fact.theta, fact.lower, fact.upper, fact.margin))
                .collect(),
            stationary: cert.is_stationary(),
            clean: cert.is_clean(),
        }
    });
    let smoothing_forensics = smoothing_forensics_rows(&fit, &edf_blocks);
    let termspec = model.payload().resolved_termspec.as_ref();
    Ok(ReportInput {
        model_path,
        family_name: model.display_family_name(),
        model_class: prediction_model_class_label(model),
        formula: model.payload().formula.clone(),
        n_obs: Some(fit.training_sample_size()),
        deviance: fit.deviance,
        reml_score: fit
            .comparable_reml_score()
            .map_err(|err| format!("failed to compute comparable REML score: {err}"))?,
        raw_reml_score: fit.reml_score(),
        iterations: fit.outer_iterations,
        convergence_status: fit
            .convergence_evidence()
            .inner_status()
            .label()
            .to_string(),
        converged: true,
        outer_gradient_norm: fit.outer_gradient_norm,
        criterion_certificate,
        smoothing_forensics,
        edf_total,
        r_squared: None,
        coefficients,
        edf_blocks,
        continuous_order: Vec::new(),
        anisotropic_scales: anisotropic_scales_rows(termspec),
        measure_jet_spectra: measure_jet_spectrum_rows_from_spec(termspec),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        basis_checks: report_basis_checks(model),
        notes,
    })
}

/// Map a fitted model's persisted #2774 basis-adequacy rows onto the renderer's
/// plain row type. A pure read: the report shows what the FIT measured, and
/// shows nothing when it measured nothing.
fn report_basis_checks(model: &FittedModel) -> Vec<BasisCheckRow> {
    model
        .payload()
        .basis_adequacy
        .iter()
        .map(|row| BasisCheckRow {
            name: row.name.clone(),
            basis_dim: row.basis_dim,
            nullspace_dim: row.nullspace_dim,
            edf: row.edf,
            enrichment_rank: row.enrichment_rank,
            statistic: row.statistic,
            p_value: row.p_value,
            provenance: row.provenance.label().to_string(),
        })
        .collect()
}

fn smoothing_forensics_rows(
    fit: &UnifiedFitResult,
    edf_blocks: &[EdfBlockRow],
) -> Vec<SmoothingForensicsRow> {
    let sigma2 = (fit.standard_deviation * fit.standard_deviation).max(0.0);
    let assembly_edfs = fit.edf_by_block();
    fit.blocks
        .iter()
        .enumerate()
        .map(|(block_idx, block)| {
            let lambda_path = block.lambdas.iter().copied().collect::<Vec<_>>();
            let edf_criterion = edf_blocks
                .iter()
                .find(|row| row.index == block_idx)
                .map(|row| row.edf)
                .or(Some(block.edf));
            let edf_assembly = assembly_edfs.get(block_idx).copied().or(Some(block.edf));
            let role = block.role.name();
            SmoothingForensicsRow {
                term: format!("block {block_idx} ({role})"),
                lambda_path,
                sigma2_path: vec![sigma2],
                edf_criterion,
                edf_assembly,
                double_penalty_range: None,
                double_penalty_null_space: fit.artifacts.null_space_dim.and_then(|dim| {
                    if dim > 0 && block_idx == 0 {
                        Some(dim as f64)
                    } else {
                        None
                    }
                }),
                seed_screening: Vec::new(),
            }
        })
        .collect()
}

/// The report card of a spline-scan model (#1046), from the scalar quantities
/// the saved `SplineScanFit` retains and the single smooth's EDF block. The
/// smoother keeps the per-knot posterior, not a dense design or Gram, so there
/// is no coefficient table.
fn spline_scan_report_input(
    model: &FittedModel,
    model_path: String,
    feature_column: &str,
    scan: &gam_solve::spline_scan::SplineScanFit,
) -> ReportInput {
    ReportInput {
        model_path,
        family_name: model.display_family_name(),
        model_class: prediction_model_class_label(model),
        formula: model.payload().formula.clone(),
        n_obs: Some(scan.training_sample_size()),
        deviance: scan.deviance(),
        // The scan has no penalty null-space metadata, so it has no comparable
        // criterion; its raw criterion is shown as raw (#2627).
        reml_score: None,
        raw_reml_score: Some(-scan.restricted_loglik),
        iterations: 0,
        convergence_status: "exact (state-space spline scan)".to_string(),
        converged: true,
        outer_gradient_norm: None,
        criterion_certificate: None,
        smoothing_forensics: Vec::new(),
        edf_total: scan.edf(),
        r_squared: None,
        coefficients: Vec::new(),
        edf_blocks: vec![EdfBlockRow {
            index: 0,
            edf: scan.edf(),
            role: Some("smooth".to_string()),
        }],
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        basis_checks: report_basis_checks(model),
        notes: vec![format!(
            "Exact O(n) state-space spline scan for s({feature_column}): \
             λ={:.4e}, EDF={:.3}, knots={}. The smoother retains the per-knot \
             posterior, not a dense design/Gram, so no coefficient table is shown.",
            scan.lambda(),
            scan.edf(),
            scan.knots.len(),
        )],
    }
}

/// The report card of a residual-cascade model (#1032), from the scalar
/// quantities the saved multilevel Wendland posterior retains. The cascade keeps
/// no dense design or Gram, so there is no coefficient table.
fn residual_cascade_report_input(
    model: &FittedModel,
    model_path: String,
    feature_columns: &[String],
    fit: &gam_solve::residual_cascade::ResidualCascadeFit,
) -> ReportInput {
    use gam_solve::residual_cascade::LogdetMethod;
    let lambda = fit.lambda();
    let mut notes = vec![format!(
        "Exact O(n log n) multiresolution residual cascade for s({features}): \
         λ={lambda:.4e}, σ²={sigma2:.4e}, levels={levels}, centers={centers}, \
         coeffs={coeffs}. CG backward error={cg_resid:.2e} over {cg_iters} \
         iterations. The cascade retains the multilevel Wendland posterior, not \
         a dense design/Gram, so no coefficient table is shown.",
        features = feature_columns.join(", "),
        sigma2 = fit.sigma2,
        levels = fit.num_levels(),
        centers = fit.num_centers(),
        coeffs = fit.num_coeffs(),
        cg_resid = fit.certificate.solve_rel_residual,
        cg_iters = fit.certificate.solve_iters,
    )];
    // Which machinery selected λ is a property of the fit a reader cannot infer
    // from the numbers above: past the dense sizing cap the profiled residual and
    // its λ derivatives come from a Golub–Meurant quadrature whose own convergence
    // admitted it, or — when none did — from a solve at every λ. Those are
    // different criteria and the report says which one this fit was selected on.
    notes.push(
        match fit.certificate.logdet_method {
            LogdetMethod::DenseExact => {
                "The log-determinant came from exact dense linear algebra — a dense \
                 Cholesky at the fitted lambda, or the lambda-independent Schur \
                 eigendecomposition the certified selection is built from: exact, \
                 with no stochastic estimate anywhere in the criterion."
            }
            LogdetMethod::SparseExact => {
                "The log-determinant came from an exact sparse direct Cholesky of the \
                 normal equations at the fitted lambda: exact, with no stochastic \
                 estimate anywhere in the criterion."
            }
            LogdetMethod::Slq => {
                "The log-determinant came from a diagonal control variate plus \
                 stochastic Lanczos quadrature on fixed deterministic probes; the \
                 coefficient solve's backward error above is the accompanying \
                 solve certificate."
            }
        }
        .to_string(),
    );
    if let Some(refinement) = fit.refinement.as_ref() {
        notes.push(format!(
            "Refinement stopped where one more level stopped earning its own Occam \
             factor: the complete candidate level buys {:.2e} of penalized objective \
             against a break-even of {:.2e}, for {:+.2e} of restricted log-likelihood.",
            refinement.gain, refinement.tolerance, refinement.evidence,
        ));
    }
    ReportInput {
        model_path,
        family_name: model.display_family_name(),
        model_class: prediction_model_class_label(model),
        formula: model.payload().formula.clone(),
        n_obs: Some(fit.training_sample_size()),
        // Gaussian-identity deviance ≡ the penalized residual quadratic
        // `y'Wy − ĉ'X'Wy` the fit profiles σ² from.
        deviance: fit.rss_pen,
        // The cascade has no penalty null-space metadata, so it has no comparable
        // criterion; its raw criterion is shown as raw (#2627).
        reml_score: None,
        raw_reml_score: Some(-fit.restricted_loglik),
        iterations: 0,
        convergence_status: "exact (multiresolution residual cascade)".to_string(),
        converged: true,
        outer_gradient_norm: None,
        criterion_certificate: None,
        smoothing_forensics: Vec::new(),
        edf_total: 0.0,
        r_squared: None,
        coefficients: Vec::new(),
        edf_blocks: Vec::new(),
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        basis_checks: report_basis_checks(model),
        notes,
    }
}

/// Anisotropic spatial-geometry report rows from an optional resolved spec.
fn anisotropic_scales_rows(spec: Option<&TermCollectionSpec>) -> Vec<AnisotropicScalesRow> {
    use gam_terms::smooth::{get_spatial_aniso_log_scales, get_spatial_length_scale};
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        let axes = eta
            .iter()
            .enumerate()
            .map(|(a, &eta_a)| {
                let (length_a, kappa_a) = if let Some(ls) = ls {
                    (Some(ls * (-eta_a).exp()), Some((1.0 / ls) * eta_a.exp()))
                } else {
                    (None, None)
                };
                (a, eta_a, length_a, kappa_a)
            })
            .collect();
        rows.push(AnisotropicScalesRow {
            term_name: term.name.clone(),
            global_length_scale: ls,
            axes,
        });
    }
    rows
}

/// Measure-jet spectrum report rows from a saved (frozen) spec alone: the
/// realized band and the spec's order, with no per-scale λ̂. Those need the
/// rebuilt design's penalty layout, which only a report given data has.
fn measure_jet_spectrum_rows_from_spec(
    spec: Option<&TermCollectionSpec>,
) -> Vec<MeasureJetSpectrumRow> {
    use gam_terms::smooth::SmoothBasisSpec;
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for term in &spec.smooth_terms {
        let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &term.basis else {
            continue;
        };
        let Some(frozen) = mj.frozen_quadrature.as_ref() else {
            continue;
        };
        let (Some(&eps_min), Some(&eps_max)) = (frozen.eps_band.first(), frozen.eps_band.last())
        else {
            continue;
        };
        rows.push(MeasureJetSpectrumRow {
            term_name: term.name.clone(),
            eps_min,
            eps_max,
            n_scales: frozen.eps_band.len(),
            length_scale: mj.length_scale,
            spec_order_s: mj.order_s,
            per_scale: Vec::new(),
            implied_order: None,
        });
    }
    rows
}
