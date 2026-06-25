//! End-to-end quality: OBJECTIVE calibration of gam's *response-scale* mean
//! confidence intervals under a Gaussian identity link.
//!
//! What this asserts (the quality claim). A 95% confidence interval is only
//! "good" if it actually covers the truth ~95% of the time. We therefore build
//! data from a KNOWN mean function μ(pc1, pc2) = f(pc1) + b·pc2 with known
//! Gaussian noise, draw many independent noise replicates, fit gam on each,
//! form the response-scale 95% mean interval [mean_lower, mean_upper] at every
//! design point, and measure the EMPIRICAL COVERAGE — the fraction of
//! (replicate, point) pairs whose interval brackets the true μ. The primary
//! pass/fail is that gam's empirical coverage lands inside the nominal band
//! 0.95 ± 0.06. That is an objective property of gam's own intervals against
//! ground truth; it does not depend on any other tool's output.
//!
//! Why the identity link is the clean case. For Gaussian with the identity link
//! the inverse link is g⁻¹(η) = η, so dμ/dη ≡ 1 and the response-scale SE is
//! algebraically identical to the η-scale SE — no transformation happens. We
//! still verify that mathematical identity directly (response SE == eta SE to
//! numerical precision); that is a correctness check against an EXACT analytic
//! quantity (Jacobian = 1), not a "matches a peer tool" check, so it stays.
//!
//! mgcv as a BASELINE TO MATCH-OR-BEAT (not as the truth). mgcv is the mature
//! penalized-GAM reference, so we additionally fit the SAME data with
//! `mgcv::gam` + `predict.gam(type="response", se.fit=TRUE)`, build its 95%
//! Gaussian mean intervals the identical way, and measure ITS empirical
//! coverage against the same ground truth. We assert gam's coverage is at least
//! as close to nominal as mgcv's, minus a small slack — i.e. gam calibrates as
//! well as or better than mgcv. We also print the SE relative-L2 for context,
//! but matching mgcv's SE is NOT a pass criterion: a noisy reference fit could
//! be miscalibrated, and reproducing it would prove nothing.
//!
//! Matching conventions. mgcv's default `predict.gam` SEs use the posterior
//! covariance `Vp` conditional on the estimated smoothing parameters
//! (`unconditional = FALSE`), while `unconditional = TRUE` adds the Wood--Pya--
//! Säfken smoothing-parameter correction. gam is checked both ways: explicit
//! `Conditional` intervals stay conditional, and the default
//! `ConditionalPlusSmoothingPreferred` path must widen from `Vb` to the
//! rho-marginalized `Vp` whenever the fitted smooth exposes that correction.

use csv::StringRecord;
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const PI: f64 = std::f64::consts::PI;
const N: usize = 300;
const REPLICATES: usize = 30;
const SIGMA: f64 = 0.30;
const NOMINAL: f64 = 0.95;
/// Linear coefficient on pc2 in the ground-truth mean.
const B_PC2: f64 = 0.80;

/// Known ground-truth mean: a smooth wiggle in pc1 plus a linear pc2 effect.
/// This is exactly the "smooth term + linear term" mixed design the capability
/// targets, but now with a function we know exactly so coverage is measurable.
fn mu_true(pc1: f64, pc2: f64) -> f64 {
    (2.0 * PI * pc1).sin() + 0.5 * pc1 + B_PC2 * pc2
}

/// Fixed design points (the predictor grid is shared across all replicates so
/// that only the noise changes — coverage is measured over the same μ values).
fn design_points(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let mut pc1: Vec<f64> = (0..N).map(|_| u.sample(&mut rng)).collect();
    pc1.sort_by(|a, b| a.partial_cmp(b).expect("finite pc1"));
    let pc2: Vec<f64> = (0..N).map(|_| u.sample(&mut rng) * 2.0 - 1.0).collect();
    (pc1, pc2)
}

/// Count how many of the `n` intervals bracket the true mean.
fn covered(lower: &[f64], upper: &[f64], truth: &[f64]) -> usize {
    lower
        .iter()
        .zip(upper)
        .zip(truth)
        .filter(|((lo, hi), t)| **lo <= **t && **t <= **hi)
        .count()
}

/// Objective calibration test: gam's 95% response-scale mean CIs must cover the
/// known truth at ~95%, and do so at least as well as mgcv.
#[test]
fn response_scale_ci_is_calibrated_and_matches_or_beats_mgcv() {
    init_parallelism();

    let (pc1, pc2) = design_points(20_260_530);
    let truth: Vec<f64> = pc1.iter().zip(&pc2).map(|(&a, &b)| mu_true(a, b)).collect();

    let gaussian_identity = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );

    // Accumulators for empirical coverage across all (replicate, point) pairs.
    let total = N * REPLICATES;
    let mut gam_covered = 0usize;
    let mut mgcv_covered = 0usize;
    // SE agreement is printed for context only (NOT a pass criterion).
    let mut worst_rel_se = 0.0f64;
    let mut rho_marginalized_covered = 0usize;
    let mut rho_marginalized_strictly_wider = 0usize;
    let mut rho_marginalized_narrower = 0usize;
    let mut mgcv_unconditional_narrower = 0usize;
    // Self-consistency of the identity-link Jacobian, worst over replicates.
    let mut worst_self_consistency = 0.0f64;

    for rep in 0..REPLICATES {
        // ---- draw a fresh noise replicate around the SAME truth ----------
        let mut rng = StdRng::seed_from_u64(100 + rep as u64);
        let noise = Normal::new(0.0, SIGMA).expect("normal");
        let y: Vec<f64> = truth.iter().map(|&m| m + noise.sample(&mut rng)).collect();

        // ---- build the dataset and fit gam: y ~ s(pc1) + pc2 -------------
        let headers = ["pc1", "pc2", "y"].into_iter().map(String::from).collect();
        let rows: Vec<StringRecord> = (0..N)
            .map(|i| {
                StringRecord::from(vec![
                    pc1[i].to_string(),
                    pc2[i].to_string(),
                    y[i].to_string(),
                ])
            })
            .collect();
        let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode replicate");

        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ s(pc1) + pc2", &ds, &cfg).expect("gam fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit");
        };

        let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
            .expect("rebuild design at training points");
        let dense = design.design.to_dense();
        assert_eq!(dense.nrows(), N, "design row count must equal data rows");
        assert_eq!(
            dense.ncols(),
            fit.fit.beta.len(),
            "design columns must match beta length"
        );

        let offset = Array1::<f64>::zeros(N);
        let pred = predict_gamwith_uncertainty(
            dense.clone(),
            fit.fit.beta.view(),
            offset.view(),
            gaussian_identity.clone(),
            &fit.fit,
            &PredictUncertaintyOptions {
                confidence_level: NOMINAL,
                // Conditional Vb path, used as the baseline for the explicit
                // rho-marginalized default below.
                covariance_mode: InferenceCovarianceMode::Conditional,
                mean_interval_method: MeanIntervalMethod::Delta,
                includeobservation_interval: false,
                apply_bias_correction: false,
                edgeworth_one_sided: false,
                boundary_correction: false,
                ood_inflation: false,
                multi_point_joint: false,
                ..PredictUncertaintyOptions::default()
            },
        )
        .expect("gam response-scale uncertainty prediction");

        let mean_lower = pred.mean_lower.to_vec();
        let mean_upper = pred.mean_upper.to_vec();
        let gam_mean_se = pred.mean_standard_error.to_vec();
        let gam_eta_se = pred.eta_standard_error.to_vec();

        // EXACT identity: identity-link Jacobian = 1 => response SE == eta SE.
        let self_consistency = gam_mean_se
            .iter()
            .zip(&gam_eta_se)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        worst_self_consistency = worst_self_consistency.max(self_consistency);

        gam_covered += covered(&mean_lower, &mean_upper, &truth);

        let pred_rho_marginalized = predict_gamwith_uncertainty(
            dense,
            fit.fit.beta.view(),
            offset.view(),
            gaussian_identity.clone(),
            &fit.fit,
            &PredictUncertaintyOptions {
                confidence_level: NOMINAL,
                covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
                mean_interval_method: MeanIntervalMethod::Delta,
                includeobservation_interval: false,
                apply_bias_correction: false,
                edgeworth_one_sided: false,
                boundary_correction: false,
                ood_inflation: false,
                multi_point_joint: false,
                ..PredictUncertaintyOptions::default()
            },
        )
        .expect("gam rho-marginalized uncertainty prediction");
        assert!(
            pred_rho_marginalized.covariance_corrected_used,
            "smooth fit should expose and use the smoothing-parameter-corrected covariance"
        );
        let rho_mean_lower = pred_rho_marginalized.mean_lower.to_vec();
        let rho_mean_upper = pred_rho_marginalized.mean_upper.to_vec();
        let rho_mean_se = pred_rho_marginalized.mean_standard_error.to_vec();
        rho_marginalized_covered += covered(&rho_mean_lower, &rho_mean_upper, &truth);
        for (&rho_se, &cond_se) in rho_mean_se.iter().zip(&gam_mean_se) {
            if rho_se + 1.0e-10 < cond_se {
                rho_marginalized_narrower += 1;
            }
            if rho_se > cond_se + 1.0e-10 {
                rho_marginalized_strictly_wider += 1;
            }
        }

        // ---- fit the SAME data with mgcv; build its 95% mean CIs ----------
        let r = run_r(
            &[
                Column::new("pc1", &pc1),
                Column::new("pc2", &pc2),
                Column::new("y", &y),
            ],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(pc1) + pc2, data = df, family = gaussian(), method = "REML")
            p <- predict(m, newdata = df, se.fit = TRUE, type = "response")
            pu <- predict(m, newdata = df, se.fit = TRUE, type = "response", unconditional = TRUE)
            z <- qnorm(0.975)
            emit("fit", as.numeric(p$fit))
            emit("se", as.numeric(p$se.fit))
            emit("lower", as.numeric(p$fit - z * p$se.fit))
            emit("upper", as.numeric(p$fit + z * p$se.fit))
            emit("se_unconditional", as.numeric(pu$se.fit))
            emit("lower_unconditional", as.numeric(pu$fit - z * pu$se.fit))
            emit("upper_unconditional", as.numeric(pu$fit + z * pu$se.fit))
            "#,
        );
        let mgcv_se = r.vector("se");
        let mgcv_lower = r.vector("lower");
        let mgcv_upper = r.vector("upper");
        let mgcv_unconditional_se = r.vector("se_unconditional");
        let mgcv_unconditional_lower = r.vector("lower_unconditional");
        let mgcv_unconditional_upper = r.vector("upper_unconditional");
        assert_eq!(mgcv_se.len(), N, "mgcv se.fit length mismatch");
        assert_eq!(
            mgcv_unconditional_se.len(),
            N,
            "mgcv unconditional se.fit length mismatch"
        );

        mgcv_covered += covered(mgcv_lower, mgcv_upper, &truth);
        for (&uncond, &cond) in mgcv_unconditional_se.iter().zip(mgcv_se) {
            if uncond + 1.0e-10 < cond {
                mgcv_unconditional_narrower += 1;
            }
        }
        let mgcv_unconditional_covered =
            covered(mgcv_unconditional_lower, mgcv_unconditional_upper, &truth);
        assert!(
            mgcv_unconditional_covered <= N,
            "mgcv unconditional coverage count must stay within the replicate size"
        );
        worst_rel_se = worst_rel_se.max(relative_l2(&gam_mean_se, mgcv_se));
    }

    let gam_coverage = gam_covered as f64 / total as f64;
    let mgcv_coverage = mgcv_covered as f64 / total as f64;
    let rho_marginalized_coverage = rho_marginalized_covered as f64 / total as f64;
    let gam_err = (gam_coverage - NOMINAL).abs();
    let mgcv_err = (mgcv_coverage - NOMINAL).abs();
    let rho_marginalized_err = (rho_marginalized_coverage - NOMINAL).abs();
    eprintln!(
        "identity-link 95% mean-CI coverage over {REPLICATES} reps x {N} pts: \
         gam_conditional={gam_coverage:.4} gam_rho_marginalized={rho_marginalized_coverage:.4} \
         mgcv={mgcv_coverage:.4} (nominal={NOMINAL}); \
         gam_err={gam_err:.4} rho_marginalized_err={rho_marginalized_err:.4} \
         mgcv_err={mgcv_err:.4}; \
         rho_marginalized_strictly_wider={rho_marginalized_strictly_wider} \
         rho_marginalized_narrower={rho_marginalized_narrower} \
         mgcv_unconditional_narrower={mgcv_unconditional_narrower}; \
         worst SE rel_l2 vs mgcv={worst_rel_se:.4}; \
         worst response-vs-eta SE |Δ|={worst_self_consistency:.3e}"
    );

    // (1) EXACT analytic identity: under the identity link the delta-method
    //     response SE equals the eta SE to numerical precision (Jacobian = 1).
    assert!(
        worst_self_consistency < 1e-10,
        "identity-link delta-method response SE must equal eta SE exactly, \
         got worst max|Δ|={worst_self_consistency:.3e}"
    );

    // (2) PRIMARY objective quality claim: gam's own 95% intervals cover the
    //     known truth at ~95%. This is a property of gam against ground truth,
    //     independent of any reference tool.
    assert!(
        (gam_coverage - NOMINAL).abs() <= 0.06,
        "gam 95% response-scale mean CI is miscalibrated: empirical coverage \
         {gam_coverage:.4} is outside {NOMINAL} ± 0.06"
    );

    // (3) BASELINE match-or-beat: gam calibrates at least as well as mgcv
    //     (allowing a small Monte-Carlo slack), so gam is no worse than the
    //     mature reference on the metric that actually matters for a CI.
    assert!(
        gam_err <= mgcv_err + 0.03,
        "gam CI calibration is worse than mgcv's: gam coverage error \
         {gam_err:.4} exceeds mgcv coverage error {mgcv_err:.4} + 0.03"
    );

    // (4) The rho-marginalized prediction path must add, never subtract, the
    //     Kass--Steffey / Wood--Pya--Säfken variance contribution. A genuinely
    //     smooth fit should see a non-zero widening on at least some rows.
    assert_eq!(
        rho_marginalized_narrower, 0,
        "rho-marginalized gam SEs must be >= conditional SEs rowwise"
    );
    assert!(
        rho_marginalized_strictly_wider > 0,
        "rho-marginalized gam SEs should strictly widen for at least one smooth row"
    );

    // (5) The widened default bands should move calibration toward nominal,
    //     unless the conditional bands were already at nominal within
    //     Monte-Carlo slack. This is the objective quality acceptance for the
    //     rho-marginalized predict path.
    assert!(
        gam_err <= 0.02 || rho_marginalized_err <= gam_err + 0.01,
        "rho-marginalized gam intervals should be closer to nominal than \
         conditional bands (or conditional should already be nominal): \
         rho_err={rho_marginalized_err:.4}, conditional_err={gam_err:.4}"
    );
}
