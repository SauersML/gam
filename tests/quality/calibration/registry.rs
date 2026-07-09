//! The UQ-surface registry and completeness lint (issue #1891).
//!
//! This is the *contract* half of the standing calibration harness. The engine
//! (SBC rank uniformity + coverage sweep + Wilson verdict) and the registry
//! TYPES live in `gam_test_support::calibration`; here we (a) enumerate every
//! uncertainty surface the library exposes as a [`CalibrationTarget`], and (b)
//! run the completeness lint that walks the public result payloads field by
//! field and asserts each uncertainty-bearing field maps to a registered target.
//!
//! The lint has two independent teeth:
//!   * **Membership** — every field classified `AuditedBy(t)` must name a
//!     registered target (`assert_registry_covers_fields`).
//!   * **Exhaustiveness** — the classification walk destructures the payload
//!     struct with NO `..` rest pattern, so adding a new field to
//!     `PredictUncertaintyResult` / `CoefficientUncertaintyResult` fails to
//!     compile until it is classified here. That is what stops the next
//!     recycled-SE (#1875) from shipping unaudited.
//!
//! The registry deliberately lists surfaces that are gated by *other* files too
//! (the six standing `tests/sbc_*.rs` gates, the conformal route quality test):
//! the registry is the single index of what is audited and by which gate, not a
//! second copy of those gates.

use gam_predict::{
    CoefficientUncertaintyResult, InferenceCovarianceMode, MeanIntervalMethod,
    PredictUncertaintyResult,
};
use gam_test_support::calibration::{
    AuditMode, CalibrationTarget, FieldAudit, SurfaceKind, assert_registry_covers_fields,
    assert_registry_well_formed,
};
use ndarray::Array1;

/// The definitive registry of the library's uncertainty surfaces.
///
/// Every row is a surface a user can read a coverage/credibility/size claim off
/// of. `audited_by` names the gate test that exercises it end to end; `guards`
/// lists the cluster issues whose regression that gate would catch.
pub fn uq_surface_registry() -> Vec<CalibrationTarget> {
    vec![
        // ---- Credible bands (the #1870/#1871 core) ------------------------
        CalibrationTarget {
            name: "eta_credible_band",
            kind: SurfaceKind::CredibleBand { smoothing_corrected: false },
            mode: AuditMode::CoverageSweep,
            guards: &[1870, 1871],
            // Identity-link Gaussian: the η band IS the mean band, so the
            // gaussian smooth gate audits it directly; the GLM gates audit its
            // monotone transform (TransformEta).
            audited_by: "sbc_gaussian_smooth_band_coverage",
        },
        CalibrationTarget {
            name: "mean_credible_band_conditional",
            kind: SurfaceKind::CredibleBand { smoothing_corrected: false },
            mode: AuditMode::CoverageSweep,
            guards: &[1870, 1871, 1878],
            audited_by: "sbc_glm_binomial_band_coverage + sbc_glm_poisson_band_coverage",
        },
        CalibrationTarget {
            name: "mean_credible_band_smoothing_corrected",
            kind: SurfaceKind::CredibleBand { smoothing_corrected: true },
            mode: AuditMode::CoverageSweep,
            guards: &[1871],
            audited_by: "sbc_gaussian_smooth_band_coverage",
        },
        // ---- Predictive (observation) intervals per family ----------------
        CalibrationTarget {
            name: "predictive_interval_gaussian",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 1878],
            audited_by: "sbc_gaussian_predictive_interval_coverage",
        },
        CalibrationTarget {
            name: "predictive_interval_location_scale",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1561],
            audited_by: "sbc_location_scale_predictive_coverage",
        },
        CalibrationTarget {
            name: "survival_probability_band",
            kind: SurfaceKind::CredibleBand { smoothing_corrected: true },
            mode: AuditMode::CoverageSweep,
            guards: &[1869, 1870, 1871],
            audited_by: "sbc_survival_probability_band_coverage",
        },
        // ---- Coefficient Wald/delta intervals -----------------------------
        CalibrationTarget {
            name: "coefficient_wald_interval",
            kind: SurfaceKind::WaldDeltaInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1878],
            audited_by: "calibration::coefficient_wald::coefficient_wald_interval_covers",
        },
        // ---- ALO / LOO predictive standard error (#1869) ------------------
        CalibrationTarget {
            name: "alo_predictive_standard_error",
            kind: SurfaceKind::AloStandardError,
            mode: AuditMode::CoverageSweep,
            guards: &[1869],
            audited_by: "calibration::alo_se::alo_predictive_se_covers",
        },
        // ---- Conformal predictive intervals -------------------------------
        CalibrationTarget {
            name: "conformal_interval",
            kind: SurfaceKind::ConformalInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[942, 1054, 1098],
            audited_by: "full_conformal_predict_route_quality",
        },
        // ---- Frequentist test p-values (size curves) ----------------------
        CalibrationTarget {
            name: "likelihood_ratio_test_pvalue",
            kind: SurfaceKind::TestPValue,
            mode: AuditMode::TestSizeCurve,
            guards: &[1872],
            audited_by: "calibration::test_size::lr_test_size_under_null",
        },
        CalibrationTarget {
            name: "smooth_bartlett_lawley_test_pvalue",
            kind: SurfaceKind::TestPValue,
            mode: AuditMode::TestSizeCurve,
            guards: &[1873],
            audited_by: "calibration::test_size::bartlett_lawley_test_size_under_null",
        },
        // ---- Posterior surfaces (SBC rank uniformity) ---------------------
        CalibrationTarget {
            name: "rho_posterior_certificate",
            kind: SurfaceKind::PosteriorSample,
            mode: AuditMode::SbcRankUniformity,
            guards: &[1810],
            audited_by: "calibration::rho_posterior::rho_posterior_ranks_uniform",
        },
    ]
}

/// Exhaustive classification of every field of [`PredictUncertaintyResult`].
///
/// The destructure names EVERY field with no `..` rest pattern: a new payload
/// field will fail to compile here until it is classified, which is the
/// completeness lint's exhaustiveness tooth.
fn predict_payload_field_audits(payload: &PredictUncertaintyResult) -> Vec<FieldAudit> {
    let PredictUncertaintyResult {
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_mode_requested,
        covariance_corrected_used,
    } = payload;
    // Bind-and-drop so the exhaustive pattern is not flagged unused; the values
    // themselves are irrelevant — only the field set matters to the lint.
    let _ = (
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_mode_requested,
        covariance_corrected_used,
    );
    vec![
        FieldAudit::point("eta"),
        FieldAudit::point("mean"),
        FieldAudit::audited("eta_standard_error", "eta_credible_band"),
        FieldAudit::audited("mean_standard_error", "mean_credible_band_conditional"),
        FieldAudit::audited("eta_lower", "eta_credible_band"),
        FieldAudit::audited("eta_upper", "eta_credible_band"),
        FieldAudit::audited("mean_lower", "mean_credible_band_conditional"),
        FieldAudit::audited("mean_upper", "mean_credible_band_conditional"),
        FieldAudit::audited("observation_lower", "predictive_interval_gaussian"),
        FieldAudit::audited("observation_upper", "predictive_interval_gaussian"),
        // Provenance metadata, not a coverage-bearing value.
        FieldAudit::point("covariance_mode_requested"),
        FieldAudit::point("covariance_corrected_used"),
    ]
}

/// Exhaustive classification of every field of [`CoefficientUncertaintyResult`].
fn coefficient_payload_field_audits(payload: &CoefficientUncertaintyResult) -> Vec<FieldAudit> {
    let CoefficientUncertaintyResult {
        estimate,
        standard_error,
        lower,
        upper,
        corrected,
        covariance_mode_requested,
    } = payload;
    let _ = (estimate, standard_error, lower, upper, corrected, covariance_mode_requested);
    vec![
        FieldAudit::point("estimate"),
        FieldAudit::audited("standard_error", "coefficient_wald_interval"),
        FieldAudit::audited("lower", "coefficient_wald_interval"),
        FieldAudit::audited("upper", "coefficient_wald_interval"),
        FieldAudit::point("corrected"),
        FieldAudit::point("covariance_mode_requested"),
    ]
}

/// A minimal well-formed `PredictUncertaintyResult` used only as the target of
/// the exhaustive destructure — the field SET, not the values, is under audit.
fn payload_probe() -> PredictUncertaintyResult {
    let one = Array1::<f64>::zeros(1);
    PredictUncertaintyResult {
        eta: one.clone(),
        mean: one.clone(),
        eta_standard_error: one.clone(),
        mean_standard_error: one.clone(),
        eta_lower: one.clone(),
        eta_upper: one.clone(),
        mean_lower: one.clone(),
        mean_upper: one.clone(),
        observation_lower: None,
        observation_upper: None,
        covariance_mode_requested: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
        covariance_corrected_used: false,
    }
}

/// A minimal well-formed `CoefficientUncertaintyResult` probe.
fn coefficient_probe() -> CoefficientUncertaintyResult {
    let one = Array1::<f64>::zeros(1);
    CoefficientUncertaintyResult {
        estimate: one.clone(),
        standard_error: one.clone(),
        lower: one.clone(),
        upper: one.clone(),
        corrected: false,
        covariance_mode_requested: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
    }
}

#[test]
fn registry_is_internally_well_formed() {
    assert_registry_well_formed(&uq_surface_registry());
}

#[test]
fn predict_payload_uncertainty_fields_are_all_registered() {
    let registry = uq_surface_registry();
    let audits = predict_payload_field_audits(&payload_probe());
    assert_registry_covers_fields(&audits, &registry);
}

#[test]
fn coefficient_payload_uncertainty_fields_are_all_registered() {
    let registry = uq_surface_registry();
    let audits = coefficient_payload_field_audits(&coefficient_probe());
    assert_registry_covers_fields(&audits, &registry);
}

/// The covariance/interval MODES a caller can select are each backed by a
/// registered band target — the "credible bands (conditional and smoothing-
/// corrected)" completeness requirement in #1891. The `match` is exhaustive (no
/// wildcard), so a new mode variant must be classified here.
#[test]
fn covariance_and_interval_modes_map_to_registered_bands() {
    let registry = uq_surface_registry();
    let names: std::collections::BTreeSet<&str> = registry.iter().map(|t| t.name).collect();

    for mode in [
        InferenceCovarianceMode::Conditional,
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired,
    ] {
        let target = match mode {
            InferenceCovarianceMode::Conditional => "mean_credible_band_conditional",
            InferenceCovarianceMode::ConditionalPlusSmoothingPreferred
            | InferenceCovarianceMode::ConditionalPlusSmoothingRequired => {
                "mean_credible_band_smoothing_corrected"
            }
        };
        assert!(
            names.contains(target),
            "covariance mode {mode:?} maps to unregistered band `{target}`"
        );
    }

    for method in [MeanIntervalMethod::TransformEta, MeanIntervalMethod::Delta] {
        // Both mean-scale interval constructions feed the same registered mean
        // band; the exhaustive match forces a new method to be classified.
        let target = match method {
            MeanIntervalMethod::TransformEta | MeanIntervalMethod::Delta => {
                "mean_credible_band_conditional"
            }
        };
        assert!(names.contains(target), "interval method {method:?} unmapped");
    }
}
