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

/// A declared size, so the certificate has a formation count to charge
/// rounding at.
const SIZE_2688: crate::rho_optimizer::OuterProblemSize = crate::rho_optimizer::OuterProblemSize {
    n_obs: Some(1_000),
    p_coefficients: Some(10),
    information_count: None,
};

/// One coordinate's published gradient parts at rank `rank`, penalty energy
/// channel `fixed_beta`, and a `logdet_h` that makes the total `total`.
fn one_coordinate_evidence_2688(
    rank: usize,
    fixed_beta: f64,
    total: f64,
) -> crate::estimate::outer_eval_capture::CertificateEvidence {
    let logdet_s = -0.5 * rank as f64;
    let logdet_h = total - fixed_beta - logdet_s;
    crate::estimate::outer_eval_capture::CertificateEvidence {
        parts: vec![crate::estimate::outer_eval_capture::RhoGradientParts {
            index: 0,
            lambda: 1.0,
            block_quadratic: 2.0 * fixed_beta,
            rank,
            dim: rank,
            fixed_beta,
            logdet_h,
            frozen_logdet_h: logdet_h,
            mode_response_logdet_h: 0.0,
            logdet_s,
            total,
        }],
        ..Default::default()
    }
}

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
    // (1) The engine's declared band: no parts published, so `tolerance`.
    let engine_only = OuterConfig {
        tolerance: 1e-5,
        required_projected_gradient_norm: None,
        problem_size: SIZE_2688,
        ..Default::default()
    };
    let gradient = ndarray::array![3.0e-5];
    let bare = outer_certificate_band_at(
        &engine_only,
        &gradient,
        &crate::estimate::outer_eval_capture::CertificateEvidence::default(),
    );
    assert_eq!(bare.source, StationarityBoundSource::SolverBand);
    assert_eq!(bare.bound, 1e-5);

    // (2) The SAME config and gradient, moving only the published parts: a
    // rank-8 penalty sets the coordinate's own scale, and the band follows it.
    let evidence = one_coordinate_evidence_2688(8, 1.5, 3.0e-5);
    let coordinate = outer_certificate_band_at(&engine_only, &gradient, &evidence);
    assert_eq!(
        coordinate.source,
        StationarityBoundSource::CoordinateBand,
        "a band set by the coordinate's own parts must not report the engine's \
         declared band (bound {:.6e} vs {:.6e})",
        coordinate.bound,
        bare.bound
    );
    assert!(
        coordinate.bound > bare.bound,
        "fixture must actually widen: {:.6e} vs {:.6e}",
        coordinate.bound,
        bare.bound
    );

    // (3) The SAME config and parts, adding only the caller's cap.
    let capped = outer_certificate_band_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(1.0e-5),
            ..engine_only.clone()
        },
        &gradient,
        &evidence,
    );
    assert_eq!(
        capped.source,
        StationarityBoundSource::CallerRequirement,
        "a bound the caller decided must not be reported as the engine's"
    );
    assert_eq!(capped.bound, 1.0e-5);
    // ... and the engine's own bound is still reported beside it, which is what
    // lets a reader tell "the engine refused this" from "the engine would have
    // certified this and the caller would not".
    assert_eq!(capped.engine_bound, coordinate.bound);
    assert_eq!(capped.engine_source, coordinate.source);

    // Three conditions, three rungs. That is the property; the labels are named
    // here so a rename cannot silently collapse two of them onto one string.
    let labels = [
        bare.source.label(),
        coordinate.source.label(),
        capped.source.label(),
    ];
    assert_eq!(
        labels,
        ["solver-band", "coordinate-band", "caller-requirement"]
    );
    assert!(
        StationarityBoundSource::CoordinateBand.is_derived_standard(),
        "the per-coordinate Theorem 9 band is derived from the gradient's own \
         scale and rounding"
    );
    assert!(!StationarityBoundSource::ArithmeticLimited.is_derived_standard());
}

/// A requirement that neither tightens nor is reached must not claim the rung.
#[test]
fn a_requirement_that_decided_nothing_does_not_claim_the_rung_2688() {
    let config = OuterConfig {
        tolerance: 1e-5,
        required_projected_gradient_norm: Some(1.0e9),
        problem_size: SIZE_2688,
        ..Default::default()
    };
    let gradient = ndarray::array![3.0e-5];
    let evidence = one_coordinate_evidence_2688(8, 1.5, 3.0e-5);
    let band = outer_certificate_band_at(&config, &gradient, &evidence);
    assert_eq!(
        band.source,
        StationarityBoundSource::CoordinateBand,
        "a requirement looser than the engine's band is not a request for \
         anything and must not be credited with the bound"
    );
    // A requirement exactly equal to the band decided nothing either.
    let tied = outer_certificate_band_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(band.bound),
            ..config
        },
        &gradient,
        &evidence,
    );
    assert_eq!(tied.bound, band.bound);
    assert_eq!(tied.source, StationarityBoundSource::CoordinateBand);
}

/// `outer_gradient_tolerance` is the declared band capped at the requirement
/// and nothing else, and so is the certificate band where no parts are
/// published: a sweep rather than a comment.
#[test]
fn the_declared_band_is_the_tolerance_capped_at_the_requirement_2688() {
    for required in [None, Some(1.0e-9), Some(1.0e-3), Some(1.0), Some(1.0e9)] {
        let config = OuterConfig {
            tolerance: 1e-5,
            required_projected_gradient_norm: required,
            ..Default::default()
        };
        let expected = required.map_or(config.tolerance, |r| config.tolerance.min(r));
        assert_eq!(
            outer_gradient_tolerance(&config).abs,
            expected,
            "solver band moved: required={required:?}"
        );
        for gradient in [0.0, 1.0e-7, 3.7e2] {
            let got = outer_certificate_band_at(
                &config,
                &ndarray::array![gradient],
                &crate::estimate::outer_eval_capture::CertificateEvidence::default(),
            );
            assert_eq!(
                got.bound.to_bits(),
                expected.to_bits(),
                "certificate band moved: required={required:?} gradient={gradient}"
            );
            assert!(
                got.engine_bound >= got.bound,
                "the engine's own bound can only be loosened by a cap"
            );
        }
    }
}

// ─── #2568 caller-requirement band gates, restored (#2818) ──────────────────

/// The load-bearing half of #2568. If the caller's requirement did not reach
/// the SOLVER's band, the outer loop would still stop at the sealed bound and
/// the requirement could only ever be a post-hoc rejection.
#[test]
fn a_caller_requirement_tightens_the_solver_band_2568() {
    let band_config = |required: Option<f64>| OuterConfig {
        tolerance: 1e-2,
        required_projected_gradient_norm: required,
        ..Default::default()
    };

    let engine = outer_gradient_tolerance(&band_config(None)).abs;
    assert!(
        engine > 1.0e-3,
        "fixture must start from an engine band looser than the requirement, got {engine:.6e}"
    );
    let tightened = outer_gradient_tolerance(&band_config(Some(1.0e-3))).abs;
    assert_eq!(
        tightened, 1.0e-3,
        "the caller's requirement must become the band the optimizer is told to \
         reach (engine band was {engine:.6e})"
    );
}

/// The per-coordinate band widens past the declared tolerance with the
/// coordinate's own scale, so a requirement applied before it would be
/// defeated by precisely the case it exists for.
#[test]
fn a_caller_requirement_survives_the_coordinate_band_2568() {
    let config = |required: Option<f64>| {
        OuterConfig {
            tolerance: 1e-5,
            required_projected_gradient_norm: required,
            problem_size: SIZE_2688,
            ..Default::default()
        }
    };
    let gradient = ndarray::array![3.0e-5];
    let evidence = one_coordinate_evidence_2688(40, 7.0, 3.0e-5);
    let engine = outer_certificate_band_at(&config(None), &gradient, &evidence).bound;
    assert!(
        engine > 1.0e-4,
        "fixture must reproduce a widened certificate band, got {engine:.6e}"
    );
    let capped = outer_certificate_band_at(&config(Some(1.0e-5)), &gradient, &evidence).bound;
    assert_eq!(
        capped, 1.0e-5,
        "the certificate band must be capped at the caller's requirement, not \
         widened past it (engine band was {engine:.6e})"
    );
}

/// A caller asking for LESS accuracy than the engine already guarantees is not
/// asking for anything. If this inverted, the knob would be a way to launder a
/// fit past a standard the engine derived from the criterion.
#[test]
fn a_looser_caller_requirement_never_weakens_the_engine_band_2568() {
    let cfg_tight_engine = OuterConfig {
        tolerance: 1.0e-9,
        problem_size: SIZE_2688,
        ..Default::default()
    };
    let engine = outer_gradient_tolerance(&cfg_tight_engine).abs;
    // Non-vacuity: a requirement can only be shown not to LOOSEN a band that is
    // tighter than the requirement to begin with.
    assert!(
        engine < 1.0,
        "the engine band {engine:.6e} must be tighter than the 1.0 requirement below, or \
         'never weakens' is asserted where there was nothing to weaken"
    );
    let with_loose = outer_gradient_tolerance(&OuterConfig {
        required_projected_gradient_norm: Some(1.0),
        ..cfg_tight_engine.clone()
    })
    .abs;
    assert_eq!(
        with_loose, engine,
        "a requirement of 1.0 must leave a {engine:.6e} engine band in force"
    );
    let gradient = ndarray::array![3.0e-9];
    let evidence = one_coordinate_evidence_2688(8, 1.5, 3.0e-9);
    let engine_band = outer_certificate_band_at(&cfg_tight_engine, &gradient, &evidence).bound;
    assert!(
        engine_band < 1.0e9,
        "the engine certificate band {engine_band:.6e} must be tighter than the 1e9 \
         requirement below for the same reason"
    );
    let loose_band = outer_certificate_band_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(1.0e9),
            ..cfg_tight_engine
        },
        &gradient,
        &evidence,
    )
    .bound;
    assert_eq!(
        loose_band, engine_band,
        "nor may it widen the certificate band"
    );
}
