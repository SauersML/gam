//! #2568 caller-side outer stationarity requirement.
//!
//! Split out of `run_plan_tests.rs` when that file reached 10,063 lines and
//! tripped the 10,000-line ban gate (#780), which aborts the ROOT crate's
//! build and therefore every root-crate target. Declared with `#[path]` from
//! `run_plan.rs` exactly as its sibling is, so `use super::*` resolves
//! identically; no test changed subject, assertion or name.

use super::*;

// ─── #2568 caller-side outer stationarity requirement ──────────────

#[test]
fn a_caller_requirement_tightens_the_solver_band_2568() {
    // The load-bearing half. If the requirement did not reach the SOLVER's
    // band, the outer loop would still stop at the sealed bound and the
    // requirement could only ever be a post-hoc rejection.
    let engine = outer_gradient_tolerance(&wide_band_config_2568(None)).abs;
    assert!(
        engine > 1.0e-3,
        "fixture must reproduce a saturated engine band, got {engine:.6e}"
    );
    let tightened = outer_gradient_tolerance(&wide_band_config_2568(Some(1.0e-3))).abs;
    assert_eq!(
        tightened, 1.0e-3,
        "the caller's requirement must become the band the optimizer is told to \
         reach (engine band was {engine:.6e})"
    );
}

#[test]
fn a_caller_requirement_survives_the_score_relative_widening_2568() {
    // The widening is what produced `bound = 1.000e0`, so a requirement applied
    // before it would be defeated by precisely the case it exists for.
    let cost = 1.0e6;
    let engine = outer_stationarity_band_at(&wide_band_config_2568(None), cost);
    assert!(
        engine > 1.0e-3,
        "fixture must reproduce a widened certificate band, got {engine:.6e}"
    );
    let capped = outer_stationarity_band_at(&wide_band_config_2568(Some(1.0e-3)), cost);
    assert_eq!(
        capped, 1.0e-3,
        "the certificate band must be capped at the caller's requirement, not \
         widened past it (engine band was {engine:.6e})"
    );
}

#[test]
fn a_looser_caller_requirement_never_weakens_the_engine_band_2568() {
    // A caller asking for less accuracy than the engine already guarantees is
    // not asking for anything. If this inverted, the knob would be a way to
    // launder a fit past a standard the engine derived from the criterion.
    let cfg_tight_engine = OuterConfig {
        tolerance: 1.0e-9,
        ..Default::default()
    };
    let engine = outer_gradient_tolerance(&cfg_tight_engine).abs;
    let with_loose = outer_gradient_tolerance(&OuterConfig {
        required_projected_gradient_norm: Some(1.0),
        ..cfg_tight_engine.clone()
    })
    .abs;
    assert_eq!(
        with_loose, engine,
        "a requirement of 1.0 must leave a {engine:.6e} engine band in force"
    );
    let cost = 1.0e3;
    let engine_band = outer_stationarity_band_at(&cfg_tight_engine, cost);
    let loose_band = outer_stationarity_band_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(1.0e9),
            ..cfg_tight_engine
        },
        cost,
    );
    assert_eq!(
        loose_band, engine_band,
        "nor may it widen the certificate band"
    );
}

#[test]
fn an_absent_requirement_reproduces_the_engine_exactly_2568() {
    // The compatibility claim, stated as a test rather than asserted in a doc
    // comment: `None` must be byte-for-byte today's behaviour on both bands.
    for scale in [None, Some(1.0), Some(1.0e6)] {
        for cost in [0.0, 1.0, -3.7e2, 1.0e6, f64::INFINITY] {
            let cfg = OuterConfig {
                tolerance: 1e-5,
                objective_scale: scale,
                required_projected_gradient_norm: None,
                ..Default::default()
            };
            let bare = OuterConfig {
                tolerance: 1e-5,
                objective_scale: scale,
                ..Default::default()
            };
            assert_eq!(
                outer_gradient_tolerance(&cfg).abs,
                outer_gradient_tolerance(&bare).abs,
                "solver band moved with scale={scale:?}"
            );
            assert_eq!(
                outer_stationarity_band_at(&cfg, cost),
                outer_stationarity_band_at(&bare, cost),
                "certificate band moved with scale={scale:?} cost={cost}"
            );
        }
    }
}

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

/// The defect, stated as the thing that must not happen again.
///
/// Before #2688 the certificate's band was decided three ways and labelled one
/// way. The three ways are here as three configurations that differ in ONE
/// input each, and the assertion is that the rung MOVES with them. A rung that
/// is the same value across conditions which visibly move the number is not a
/// measurement of anything -- it is a default wearing a measurement's name, and
/// the old `let mut bound_source = StationarityBoundSource::SolverBand;` at the
/// call site is exactly that.
#[test]
fn each_of_the_three_band_deciders_names_itself_2688() {
    // (1) The engine's declared band. No scale, so the arithmetic floor is
    // `tolerance`; a criterion value small enough that the point-anchored
    // widening `tol*(1+|cost|)` cannot exceed it.
    let engine_only = OuterConfig {
        tolerance: 1e-5,
        objective_scale: None,
        required_projected_gradient_norm: None,
        ..Default::default()
    };
    let bare = outer_stationarity_band_and_rung_at(&engine_only, 0.0);
    assert_eq!(bare.source, StationarityBoundSource::SolverBand);
    assert_eq!(bare.bound, 1e-5);

    // (2) The SAME config, moving only the judged point. The number moves by
    // six orders; before #2688 the label did not move at all.
    let widened = outer_stationarity_band_and_rung_at(&engine_only, -1.0e6);
    assert_eq!(
        widened.source,
        StationarityBoundSource::CertificateScoreRelative,
        "a band set by the point-anchored widening must not report the engine's \
         declared band, which is anchored on the DECLARED problem and cannot \
         move with the criterion value (bound {:.6e} vs {:.6e})",
        widened.bound,
        bare.bound
    );
    assert!(
        widened.bound > bare.bound * 1.0e5,
        "fixture must actually widen: {:.6e} vs {:.6e}",
        widened.bound,
        bare.bound
    );

    // (3) The SAME config and the SAME point, adding only the caller's cap.
    let capped = outer_stationarity_band_and_rung_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(1.0e-3),
            ..engine_only.clone()
        },
        -1.0e6,
    );
    assert_eq!(
        capped.source,
        StationarityBoundSource::CallerRequirement,
        "a bound the caller decided must not be reported as the engine's"
    );
    assert_eq!(capped.bound, 1.0e-3);
    // ... and the engine's own bound is still reported beside it, which is what
    // lets a reader tell "the engine refused this" from "the engine would have
    // certified this and the caller would not".
    assert_eq!(capped.engine_bound, widened.bound);
    assert_eq!(capped.engine_source, widened.source);

    // Three conditions, three rungs. That is the property; the labels are named
    // here so a rename cannot silently collapse two of them onto one string.
    let labels = [
        bare.source.label(),
        widened.source.label(),
        capped.source.label(),
    ];
    assert_eq!(
        labels,
        [
            "solver-band",
            "certificate-score-relative",
            "caller-requirement"
        ]
    );
    assert!(
        !StationarityBoundSource::CertificateScoreRelative.is_derived_standard(),
        "the score-relative widening is a gradient-magnitude substitute, not the \
         derived resolvability standard"
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
    let band = outer_stationarity_band_and_rung_at(&config, -1.0e6);
    assert_eq!(
        band.source,
        StationarityBoundSource::CertificateScoreRelative,
        "a requirement looser than the engine's band is not a request for \
         anything and must not be credited with the bound"
    );
    // A requirement exactly equal to the band decided nothing either.
    let tied = outer_stationarity_band_and_rung_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(band.bound),
            ..config
        },
        -1.0e6,
    );
    assert_eq!(tied.bound, band.bound);
    assert_eq!(tied.source, StationarityBoundSource::CertificateScoreRelative);
}

/// The refactor must not move a single band. `outer_gradient_tolerance` is the
/// engine band capped at the requirement and nothing else, and the certificate
/// band is `min(max(engine, score-relative), required)` -- algebraically the
/// same number the pre-#2688 `min(max(min(engine, required), score), required)`
/// produced, which is a claim worth a sweep rather than a comment.
#[test]
fn the_rung_refactor_moved_no_band_2688() {
    for scale in [None, Some(1.0), Some(1.0e6)] {
        for required in [None, Some(1.0e-9), Some(1.0e-3), Some(1.0), Some(1.0e9)] {
            let config = OuterConfig {
                tolerance: 1e-5,
                objective_scale: scale,
                required_projected_gradient_norm: required,
                ..Default::default()
            };
            let engine_band = {
                let mut abs = config
                    .objective_scale
                    .map(|s| config.tolerance.max(s * f64::EPSILON.sqrt()))
                    .unwrap_or(config.tolerance);
                if let Some(s) = config.objective_scale {
                    abs = abs.max(config.tolerance * (1.0 + s));
                }
                abs
            };
            assert_eq!(
                outer_gradient_tolerance(&config).abs,
                required.map_or(engine_band, |r| engine_band.min(r)),
                "solver band moved: scale={scale:?} required={required:?}"
            );
            for cost in [0.0, 1.0, -3.7e2, 1.0e6, f64::INFINITY, f64::NAN] {
                let widened = if cost.is_finite() {
                    engine_band.max(config.tolerance * (1.0 + cost.abs()))
                } else {
                    engine_band
                };
                let expected = required.map_or(widened, |r| widened.min(r));
                let got = outer_stationarity_band_and_rung_at(&config, cost);
                assert_eq!(
                    got.bound.to_bits(),
                    expected.to_bits(),
                    "certificate band moved: scale={scale:?} required={required:?} cost={cost}"
                );
                assert!(
                    got.engine_bound >= got.bound,
                    "the engine's own bound can only be loosened by a cap"
                );
            }
        }
    }
}
