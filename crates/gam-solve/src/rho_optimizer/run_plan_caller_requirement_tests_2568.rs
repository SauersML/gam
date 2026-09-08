//! #2568 caller-side outer stationarity requirement.
//!
//! Split out of `run_plan_tests.rs` when that file reached 10,063 lines and
//! tripped the 10,000-line ban gate (#780), which aborts the ROOT crate's
//! build and therefore every root-crate target. Declared with `#[path]` from
//! `run_plan.rs` exactly as its sibling is, so `use super::*` resolves
//! identically; no test changed subject, assertion or name.

use super::*;
use ndarray::array;

#[test]
fn caller_requirement_caps_curvature_and_reproducibility_certificates_2627() {
    for curvature in [false, true] {
        for required in [None, Some(1.0e-7)] {
            let theta = 2.0e-6;
            let config = OuterConfig {
                tolerance: 1.0e-12,
                required_projected_gradient_norm: required,
                ..OuterConfig::default()
            };
            let mut objective = OuterProblem::new(1)
                .with_gradient(Derivative::Analytic)
                .with_hessian(if curvature {
                    DeclaredHessianForm::Dense
                } else {
                    DeclaredHessianForm::Unavailable
                })
                .build_objective(
                    (),
                    |_: &mut (), rho: &Array1<f64>| Ok(0.5 * rho[0] * rho[0]),
                    move |_: &mut (), rho: &Array1<f64>| {
                        Ok(OuterEval {
                            cost: 0.5 * rho[0] * rho[0],
                            gradient: rho.clone(),
                            hessian: if curvature {
                                HessianValue::Dense(array![[1.0]])
                            } else {
                                HessianValue::Unavailable
                            },
                            inner_beta_hint: None,
                        })
                    },
                    None::<fn(&mut ())>,
                    None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
                )
                .with_criterion_resolution(|_: &mut ()| Some(1.0e-7));
            let mut result = OuterResult::new(
                array![theta],
                0.5 * theta * theta,
                1,
                true,
                OuterPlan {
                    solver: Solver::Bfgs,
                    hessian_source: HessianSource::BfgsApprox,
                },
            );
            // Simulate a previous noisy measurement only for the reproducibility
            // case; the curvature case has an exactly reproducible gradient.
            result.final_gradient = Some(array![if curvature { theta } else { 0.0 }]);
            let certificate = certify_outer_optimality(
                &mut objective,
                &config,
                "caller limit after all widening",
                &mut result,
            );
            if required.is_some() {
                let refusal = certificate.expect_err("caller limit must refuse this point");
                assert!(refusal.to_string().contains("rung=caller-requirement"));
            } else {
                let certificate = certificate.expect("uncapped measured rung must certify");
                assert!(certificate.certifies());
                assert_eq!(
                    certificate.stationarity.rung().label,
                    if curvature {
                        "curvature-resolvability"
                    } else {
                        "gradient-reproducibility"
                    },
                );
            }
        }
    }
}

// ─── #2568 caller-side outer stationarity requirement ──────────────

#[test]
fn an_unsatisfiable_requirement_is_rejected_at_the_builder_2568() {
    // `0.0` and `NaN` are not stricter standards, they are unsatisfiable ones.
    // Honouring them would grind the outer loop to `max_iter` and then refuse
    // every fit -- a performance cliff wearing an accuracy requirement's clothes.
    for bad in [0.0, -1.0e-3, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let cfg = OuterProblem::new(1)
            .with_required_projected_gradient_norm(Some(bad))
            .config();
        assert_eq!(
            cfg.required_projected_gradient_norm, None,
            "requirement {bad} must be rejected, not clamped"
        );
    }
    let cfg = OuterProblem::new(1)
        .with_required_projected_gradient_norm(Some(1.0e-7))
        .config();
    assert_eq!(cfg.required_projected_gradient_norm, Some(1.0e-7));
}

#[test]
fn the_caller_requirement_rung_names_itself_2568() {
    // A refusal that reported `solver-band` while the caller's number decided it
    // would be unauditable in exactly the way #2465 is about.
    assert_eq!(
        StationarityBoundSource::CallerRequirement.label(),
        "caller-requirement"
    );
    assert!(
        !StationarityBoundSource::CallerRequirement.is_derived_standard(),
        "the caller's requirement is not the engine's derived resolvability \
         standard and must not claim to be"
    );
}

// ─── #2688 the band and the rung that produced it travel together ──────────

/// Each band decider names itself (#2688), and since #2812 there are two: the
/// engine's declared band and the caller's requirement. The point-anchored
/// `tol · (1 + |V|)` widening that used to be the third is gone — it was a
/// statement about the criterion's magnitude, not the gradient's resolution.
#[test]
fn each_band_decider_names_itself_2688() {
    let engine_only = OuterConfig {
        tolerance: 1e-5,
        objective_scale: None,
        required_projected_gradient_norm: None,
        ..Default::default()
    };
    let bare = outer_stationarity_band_and_rung(&engine_only);
    assert_eq!(bare.source, StationarityBoundSource::SolverBand);
    assert_eq!(bare.bound, 1e-5);
    let capped = outer_stationarity_band_and_rung(&OuterConfig {
        required_projected_gradient_norm: Some(1.0e-6),
        ..engine_only.clone()
    });
    assert_eq!(
        capped.source,
        StationarityBoundSource::CallerRequirement,
        "a bound the caller decided must not be reported as the engine's"
    );
    assert_eq!(capped.bound, 1.0e-6);
    // ... and the engine's own bound is still reported beside it, which is what
    // lets a reader tell "the engine refused this" from "the engine would have
    // certified this and the caller would not".
    assert_eq!(capped.engine_bound, bare.bound);
    assert_eq!(capped.engine_source, bare.source);
    assert_eq!(
        [bare.source.label(), capped.source.label()],
        ["solver-band", "caller-requirement"]
    );
}

/// A requirement that neither tightens nor is reached must not claim the rung.
#[test]
fn a_requirement_that_decided_nothing_does_not_claim_the_rung_2688() {
    let config = OuterConfig {
        tolerance: 1e-5,
        objective_scale: None,
        required_projected_gradient_norm: Some(1.0e9),
        ..Default::default()
    };
    let band = outer_stationarity_band_and_rung(&config);
    assert_eq!(
        band.source,
        StationarityBoundSource::SolverBand,
        "a requirement looser than the engine's band is not a request for \
         anything and must not be credited with the bound"
    );
    // A requirement exactly equal to the band decided nothing either.
    let tied = outer_stationarity_band_and_rung(&OuterConfig {
        required_projected_gradient_norm: Some(band.bound),
        ..config
    });
    assert_eq!(tied.bound, band.bound);
    assert_eq!(tied.source, StationarityBoundSource::SolverBand);
}

/// The solver band is the engine band capped at the requirement and nothing
/// else, and the certificate's first-order band is that same number (#2812):
/// no anchor on the judged point survives.
#[test]
fn the_certificate_band_is_the_solver_band_capped_at_the_requirement_2812() {
    for scale in [None, Some(1.0), Some(1.0e6)] {
        for required in [None, Some(1.0e-9), Some(1.0e-3), Some(1.0), Some(1.0e9)] {
            let config = OuterConfig {
                tolerance: 1e-5,
                objective_scale: scale,
                required_projected_gradient_norm: required,
                ..Default::default()
            };
            let engine_band = config
                .objective_scale
                .map(|s| config.tolerance.max(s * f64::EPSILON.sqrt()))
                .unwrap_or(config.tolerance);
            let expected = required.map_or(engine_band, |r| engine_band.min(r));
            assert_eq!(
                outer_gradient_tolerance(&config).abs,
                expected,
                "solver band moved: scale={scale:?} required={required:?}"
            );
            let got = outer_stationarity_band_and_rung(&config);
            assert_eq!(
                got.bound.to_bits(),
                expected.to_bits(),
                "certificate band moved: scale={scale:?} required={required:?}"
            );
            assert!(
                got.engine_bound >= got.bound,
                "the engine's own bound can only be loosened by a cap"
            );
        }
    }
}
