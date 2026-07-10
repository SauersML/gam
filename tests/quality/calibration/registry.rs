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
//!   * **Exhaustiveness** — the classification walk destructures each payload
//!     struct with NO `..` rest pattern (`PredictUncertaintyResult`,
//!     `PredictPosteriorMeanResult`, `CoefficientUncertaintyResult`) and matches
//!     the covariance/interval mode enums with no wildcard, so adding a new field
//!     or mode fails to compile until it is classified here. That is what stops
//!     the next recycled-SE (#1875) from shipping unaudited.
//!
//! The registry deliberately lists surfaces that are gated by *other* files too
//! (the six standing `tests/sbc_*.rs` gates, the conformal route quality test):
//! the registry is the single index of what is audited and by which gate, not a
//! second copy of those gates.

use gam::families::multinomial::MultinomialPredictionIntervals;
use gam::families::survival::predict::SurvivalPredictResult;
use gam_predict::{
    CoefficientUncertaintyResult, InferenceCovarianceMode, MeanIntervalMethod,
    PredictPosteriorMeanResult, PredictUncertaintyResult,
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
        // ---- Credible bands, split by scale × covariance mode -------------
        // The band is the #1870/#1871 core. The covariance mode is a real axis:
        // the conditional band uses H⁻¹; the smoothing-corrected band adds
        // J·Var(ρ̂)·Jᵀ. #1870 (mean-band collapse to 0.157) and #1871 (0.731 vs
        // INLA) are the SAME field (mean_lower/upper) under the two modes, which
        // is exactly why both are registered and both gated by the new
        // covariance-mode sweep.
        CalibrationTarget {
            name: "eta_credible_band_conditional",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: false,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1870],
            audited_by: "sbc_gaussian_mean_band_covariance_modes (Conditional arm)",
        },
        CalibrationTarget {
            name: "eta_credible_band_smoothing_corrected",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: true,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1871],
            audited_by: "sbc_gaussian_mean_band_covariance_modes (Preferred arm) \
                         + sbc_gaussian_smooth_band_coverage",
        },
        CalibrationTarget {
            name: "mean_credible_band_conditional",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: false,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1870, 1878],
            audited_by: "sbc_gaussian_mean_band_covariance_modes (Conditional arm) \
                         + sbc_glm_binomial_band_coverage + sbc_glm_poisson_band_coverage",
        },
        CalibrationTarget {
            name: "mean_credible_band_smoothing_corrected",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: true,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1871],
            audited_by: "sbc_gaussian_smooth_band_coverage \
                         + sbc_gaussian_mean_band_covariance_modes (Preferred arm)",
        },
        // ---- Predictive (observation) intervals, per response family -----
        // `family_observation_band` (gam-predict lib.rs:2041) emits a predictive
        // interval Var(μ̂)+Var(Y|μ) for each family below. Gaussian is gated
        // (in-process + location-scale + conformal); the non-Gaussian families
        // are gated by the parameterized per-family predictive-interval sweep
        // `sbc_family_predictive_interval_coverage`, which draws a genuinely NEW
        // family response and checks the skew-correct observation band covers it
        // (#817/#1193/#1194 skew + #1875/#1878 composition). `royston_parmar`
        // returns (None,None) (lib.rs:2261): it exposes NO predictive interval
        // and is deliberately NOT registered (the family enumeration test pins
        // that as a known gap, not drift).
        CalibrationTarget {
            name: "predictive_interval_gaussian",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 1878],
            audited_by: "sbc_gaussian_predictive_interval_coverage \
                         + sbc_location_scale_predictive_coverage",
        },
        CalibrationTarget {
            name: "predictive_interval_poisson",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 817],
            audited_by: "sbc_family_predictive_interval_coverage \
                         (poisson_predictive_interval_covers_new_observation_at_nominal)",
        },
        CalibrationTarget {
            name: "predictive_interval_negative_binomial",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 1193],
            audited_by: "sbc_family_predictive_interval_coverage \
                         (negative_binomial_predictive_interval_covers_new_observation_at_nominal)",
        },
        CalibrationTarget {
            name: "predictive_interval_tweedie",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 817],
            audited_by: "sbc_family_predictive_interval_coverage \
                         (tweedie_predictive_interval_covers_new_observation_at_nominal)",
        },
        CalibrationTarget {
            name: "predictive_interval_gamma",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 817],
            audited_by: "sbc_family_predictive_interval_coverage \
                         (gamma_predictive_interval_covers_new_observation_at_nominal)",
        },
        CalibrationTarget {
            name: "predictive_interval_beta",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 1194],
            audited_by: "sbc_family_predictive_interval_coverage \
                         (beta_predictive_interval_covers_new_observation_at_nominal)",
        },
        CalibrationTarget {
            name: "predictive_interval_binomial",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875],
            audited_by: "sbc_family_predictive_interval_coverage \
                         (binomial_predictive_interval_covers_new_observation_at_nominal)",
        },
        // Heteroscedastic location-scale predictive interval (#1561).
        CalibrationTarget {
            name: "predictive_interval_location_scale",
            kind: SurfaceKind::PredictiveInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1561],
            audited_by: "sbc_location_scale_predictive_coverage",
        },
        // Survival S(t|x) band (CLI-level mean_lower/mean_upper columns).
        CalibrationTarget {
            name: "survival_probability_band",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: true,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1869, 1870, 1871],
            audited_by: "sbc_survival_probability_band_coverage",
        },
        // `SurvivalPredictResult::survival_se`/`eta_se` (#1891 follow-up): a
        // SEPARATE code path from `survival_probability_band` above — the
        // spherical-radial posterior-quadrature pipeline
        // (`predict_survival_posterior_mean`) that `predict_survival` runs when
        // `with_uncertainty = true`, not the CLI's band columns. Found
        // unregistered by a completeness sweep of every public payload struct.
        CalibrationTarget {
            name: "survival_posterior_mean_se",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: false,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1891],
            audited_by: "sbc_survival_prediction_se_coverage \
                         (survival_posterior_mean_se_covers_true_survival_probability_at_nominal \
                         + survival_location_scale_delta_method_se_covers_true_survival_probability_at_nominal)",
        },
        // ---- Coefficient Wald/delta intervals -----------------------------
        CalibrationTarget {
            name: "coefficient_wald_interval",
            kind: SurfaceKind::WaldDeltaInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[1875, 1878],
            audited_by: "quality::misc::quality_vs_statsmodels_negbin_coefficient_se",
        },
        // ---- ALO / LOO predictive standard errors (#1869) -----------------
        // AloDiagnostics exposes two SEs (alo.rs:88/91): the Bayesian H⁻¹ SE and
        // the frequentist sandwich SE. Both are validated against exact n-refit
        // brute-force LOO (ground truth) — the strongest possible audit.
        CalibrationTarget {
            name: "alo_se_bayes",
            kind: SurfaceKind::AloStandardError,
            mode: AuditMode::CoverageSweep,
            guards: &[1869],
            audited_by: "quality::families::quality_vs_loo_psis_gaussian_smooth \
                         + quality_vs_brute_force_loo_binomial_logit \
                         + quality_vs_brute_force_loo_poisson_log",
        },
        CalibrationTarget {
            name: "alo_se_sandwich",
            kind: SurfaceKind::AloStandardError,
            mode: AuditMode::CoverageSweep,
            guards: &[1869],
            audited_by: "quality::families::quality_vs_scipy_sandwich_glm_gaussian",
        },
        // ---- Conformal predictive intervals -------------------------------
        CalibrationTarget {
            name: "conformal_interval",
            kind: SurfaceKind::ConformalInterval,
            mode: AuditMode::CoverageSweep,
            guards: &[942, 1054, 1098],
            audited_by: "full_conformal_predict_route_quality",
        },
        // ---- Frequentist test p-values (type-I size curves) ---------------
        // Skovgaard r*: first-order, corrected, corrected-empirical p-values
        // (skovgaard.rs:132/134/143). #1872 (post-selection LR anti-conservative)
        // is the corrected p-value's size under selection.
        CalibrationTarget {
            name: "skovgaard_lr_pvalue",
            kind: SurfaceKind::TestPValue,
            mode: AuditMode::TestSizeCurve,
            guards: &[1872, 939],
            audited_by: "sbc_skovgaard_rstar_size_curve \
                         (skovgaard_rstar_corrected_pvalue_is_not_oversized_under_the_null) \
                         + bug_hunt_smooth_significance_ref_df_floor_and_null_fpr_test",
        },
        // Wood smooth Wald test + Bartlett/Lawley LR correction (#1873).
        CalibrationTarget {
            name: "wood_smooth_test_pvalue",
            kind: SurfaceKind::TestPValue,
            mode: AuditMode::TestSizeCurve,
            guards: &[1873],
            audited_by: "bug_hunt_smooth_significance_ref_df_floor_and_null_fpr_test",
        },
        // ---- Posterior surfaces ------------------------------------------
        // The ρ-posterior (smoothing-hyperparameter) certificate is a posterior
        // surface; SBC rank uniformity is its ideal audit. Today it is gated by
        // its consumed form — the ρ-quadrature MIXTURE must move a truth-known
        // smooth band's coverage toward nominal (the #938 tier-1 gate) — which
        // is why the mode is CoverageSweep, not SBC. A bespoke ρ|data SBC
        // rank-uniformity gate is the noted stronger form (see report).
        CalibrationTarget {
            name: "rho_posterior_certificate",
            kind: SurfaceKind::PosteriorSample,
            mode: AuditMode::CoverageSweep,
            guards: &[1810, 938],
            audited_by: "perf_scale::sae::rho_posterior_tier1_sae_coverage",
        },
        // ---- Multinomial mean-probability prediction interval -------------
        // A completeness sweep of the library's public payload structs (#1891
        // follow-up) found `MultinomialPredictionIntervals` — the separate
        // `gam-models::multinomial` driver's own mean/SE/interval payload — was
        // not one of the three structs this file's completeness lint walked, so
        // nothing forced it onto the registry. Registered here and given its own
        // real end-to-end coverage sweep.
        CalibrationTarget {
            name: "multinomial_mean_prediction_interval",
            kind: SurfaceKind::CredibleBand {
                smoothing_corrected: false,
            },
            mode: AuditMode::CoverageSweep,
            guards: &[1891],
            audited_by: "sbc_multinomial_prediction_interval_coverage \
                         (multinomial_mean_prediction_interval_covers_true_probability_at_nominal)",
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
    // Consume the bound references so the exhaustive pattern is not flagged
    // unused; the values themselves are irrelevant, only the field set matters.
    std::hint::black_box((
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
    ));
    vec![
        FieldAudit::point("eta"),
        FieldAudit::point("mean"),
        FieldAudit::audited("eta_standard_error", "eta_credible_band_conditional"),
        FieldAudit::audited("mean_standard_error", "mean_credible_band_conditional"),
        FieldAudit::audited("eta_lower", "eta_credible_band_conditional"),
        FieldAudit::audited("eta_upper", "eta_credible_band_conditional"),
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
    std::hint::black_box((
        estimate,
        standard_error,
        lower,
        upper,
        corrected,
        covariance_mode_requested,
    ));
    vec![
        FieldAudit::point("estimate"),
        FieldAudit::audited("standard_error", "coefficient_wald_interval"),
        FieldAudit::audited("lower", "coefficient_wald_interval"),
        FieldAudit::audited("upper", "coefficient_wald_interval"),
        FieldAudit::point("corrected"),
        FieldAudit::point("covariance_mode_requested"),
    ]
}

/// Exhaustive classification of every field of [`PredictPosteriorMeanResult`] —
/// the posterior-mean predict path surfaced by the FFI/CLI predict tables
/// (`std_error` / `mean_lower` / `mean_upper` columns, #1536). This is the very
/// surface a recycled/mis-scaled response SE (#1875) would ship on, so walking
/// it here is the completeness lint's highest-value tooth.
fn posterior_mean_payload_field_audits(payload: &PredictPosteriorMeanResult) -> Vec<FieldAudit> {
    let PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean,
        mean_standard_error,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
    } = payload;
    std::hint::black_box((
        eta,
        eta_standard_error,
        mean,
        mean_standard_error,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
    ));
    vec![
        FieldAudit::point("eta"),
        FieldAudit::audited("eta_standard_error", "eta_credible_band_conditional"),
        FieldAudit::point("mean"),
        FieldAudit::audited("mean_standard_error", "mean_credible_band_conditional"),
        FieldAudit::audited("mean_lower", "mean_credible_band_conditional"),
        FieldAudit::audited("mean_upper", "mean_credible_band_conditional"),
        FieldAudit::audited("observation_lower", "predictive_interval_gaussian"),
        FieldAudit::audited("observation_upper", "predictive_interval_gaussian"),
    ]
}

/// Exhaustive classification of every field of [`MultinomialPredictionIntervals`].
fn multinomial_payload_field_audits(
    payload: &MultinomialPredictionIntervals,
) -> Vec<FieldAudit> {
    let MultinomialPredictionIntervals {
        mean,
        standard_error,
        mean_lower,
        mean_upper,
        level,
    } = payload;
    std::hint::black_box((mean, standard_error, mean_lower, mean_upper, level));
    vec![
        FieldAudit::point("mean"),
        FieldAudit::audited("standard_error", "multinomial_mean_prediction_interval"),
        FieldAudit::audited("mean_lower", "multinomial_mean_prediction_interval"),
        FieldAudit::audited("mean_upper", "multinomial_mean_prediction_interval"),
        FieldAudit::point("level"),
    ]
}

/// Exhaustive classification of every field of [`SurvivalPredictResult`].
fn survival_payload_field_audits(payload: &SurvivalPredictResult) -> Vec<FieldAudit> {
    let SurvivalPredictResult {
        times,
        hazard,
        survival,
        cumulative_hazard,
        linear_predictor,
        likelihood_mode,
        survival_se,
        eta_se,
    } = payload;
    std::hint::black_box((
        times,
        hazard,
        survival,
        cumulative_hazard,
        linear_predictor,
        likelihood_mode,
        survival_se,
        eta_se,
    ));
    vec![
        // Point surfaces: the plug-in/posterior-mean curves themselves carry
        // no coverage claim on their own (only their SEs do).
        FieldAudit::point("times"),
        FieldAudit::point("hazard"),
        FieldAudit::point("survival"),
        FieldAudit::point("cumulative_hazard"),
        FieldAudit::point("linear_predictor"),
        FieldAudit::point("likelihood_mode"),
        FieldAudit::audited("survival_se", "survival_posterior_mean_se"),
        FieldAudit::audited("eta_se", "survival_posterior_mean_se"),
    ]
}

/// A minimal well-formed `SurvivalPredictResult` probe.
fn survival_probe() -> SurvivalPredictResult {
    use gam::families::survival::construction::SurvivalLikelihoodMode;
    let one1 = Array1::<f64>::zeros(1);
    let one2 = ndarray::Array2::<f64>::zeros((1, 1));
    SurvivalPredictResult {
        times: vec![1.0],
        hazard: one2.clone(),
        survival: one2.clone(),
        cumulative_hazard: one2.clone(),
        linear_predictor: one1.clone(),
        likelihood_mode: SurvivalLikelihoodMode::Weibull,
        survival_se: Some(one2),
        eta_se: Some(one1),
    }
}

/// A minimal well-formed `MultinomialPredictionIntervals` probe.
fn multinomial_probe() -> MultinomialPredictionIntervals {
    let one = ndarray::Array2::<f64>::zeros((1, 1));
    MultinomialPredictionIntervals {
        mean: one.clone(),
        standard_error: one.clone(),
        mean_lower: one.clone(),
        mean_upper: one,
        level: 0.95,
    }
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

/// A minimal well-formed `PredictPosteriorMeanResult` probe.
fn posterior_mean_probe() -> PredictPosteriorMeanResult {
    let one = Array1::<f64>::zeros(1);
    PredictPosteriorMeanResult {
        eta: one.clone(),
        eta_standard_error: one.clone(),
        mean: one.clone(),
        mean_standard_error: None,
        mean_lower: None,
        mean_upper: None,
        observation_lower: None,
        observation_upper: None,
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
fn posterior_mean_payload_uncertainty_fields_are_all_registered() {
    let registry = uq_surface_registry();
    let audits = posterior_mean_payload_field_audits(&posterior_mean_probe());
    assert_registry_covers_fields(&audits, &registry);
}

#[test]
fn coefficient_payload_uncertainty_fields_are_all_registered() {
    let registry = uq_surface_registry();
    let audits = coefficient_payload_field_audits(&coefficient_probe());
    assert_registry_covers_fields(&audits, &registry);
}

#[test]
fn multinomial_payload_uncertainty_fields_are_all_registered() {
    let registry = uq_surface_registry();
    let audits = multinomial_payload_field_audits(&multinomial_probe());
    assert_registry_covers_fields(&audits, &registry);
}

#[test]
fn survival_predict_payload_uncertainty_fields_are_all_registered() {
    let registry = uq_surface_registry();
    let audits = survival_payload_field_audits(&survival_probe());
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
        assert!(
            names.contains(target),
            "interval method {method:?} unmapped"
        );
    }
}

/// Every response family whose `family_observation_band` (gam-predict lib.rs:2041)
/// emits a predictive interval must have a registered `predictive_interval_<fam>`
/// target; every family that returns `(None, None)` (RoystonParmar, lib.rs:2261)
/// must NOT — so a family silently gaining OR losing a predictive interval trips
/// this gate. This is the per-family completeness half the single family-agnostic
/// `observation_lower/upper` field cannot express on its own.
#[test]
fn every_family_predictive_interval_is_registered_or_a_known_gap() {
    let registry = uq_surface_registry();
    let names: std::collections::BTreeSet<&str> = registry.iter().map(|t| t.name).collect();
    // (family, exposes a closed-form predictive/observation interval).
    let families = [
        ("gaussian", true),
        ("poisson", true),
        ("negative_binomial", true),
        ("tweedie", true),
        ("gamma", true),
        ("beta", true),
        ("binomial", true),
        // Royston–Parmar returns (None, None): mean band only, no predictive
        // interval. A KNOWN gap, pinned so it can't drift unnoticed.
        ("royston_parmar", false),
    ];
    let mut drift = Vec::new();
    for (fam, has_interval) in families {
        let target = format!("predictive_interval_{fam}");
        let registered = names.contains(target.as_str());
        if has_interval && !registered {
            drift.push(format!(
                "family `{fam}` emits a predictive interval but `{target}` is unregistered"
            ));
        }
        if !has_interval && registered {
            drift.push(format!(
                "family `{fam}` emits NO predictive interval yet `{target}` is registered"
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "family predictive-interval registry drift:\n{}",
        drift.join("\n")
    );
}
