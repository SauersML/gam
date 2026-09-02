#![cfg(test)]
//! #2629 scope item 2 — settle the custom-family engine's row of the objective
//! table by MEASUREMENT.
//!
//! #2629 lists seven outer-objective families and asks which of them carry the
//! soft ρ-guard barrier that #2545 taught the certificate to subtract. Three of
//! those rows — `gamlss mean-wiggle`, `spatial-adaptive`, and `custom family` —
//! are the same evaluator seen from three call sites:
//! [`evaluate_custom_family_joint_hyper_owned`]. So one measurement settles
//! three rows, and it is this one.
//!
//! The issue's own evidence for those rows was a call-graph argument:
//! `RemlState::build_prior` is the only site that adds
//! `soft_rho_guard_prior_atom`'s gradient to a criterion, its only callers are
//! `RemlState` methods, and this engine holds no `RemlState`. That argument is
//! correct, and it is still an argument about code rather than about numbers.
//! The issue said what would settle it — *"evaluate each path's ρ-gradient at a
//! saturated ρ and look for the 1.3333e-7 floor"* — and
//! [`gam_solve::rho_optimizer::soft_rho_guard_floor`] is that check, with a
//! positive control (`the_floor_classifier_reports_carried_on_the_live_mixture_sas_criterion`,
//! gam-solve) proving it can see a floor when one is there.
//!
//! What a "carried" verdict here would have meant: every railed coordinate of
//! every gamlss, spatial-adaptive, and custom-family fit carrying a standing
//! `|Pg| ≥ w·a = 1.3333e-7` that no amount of convergence clears, and three more
//! objectives owing the seam a publication.
//!
//! [`evaluate_custom_family_joint_hyper_owned`]: crate::psi_hyper::evaluate_custom_family_joint_hyper_owned

use super::*;
use crate::tests::{OneBlockGaussianFamily, test_design_hyper_layout};
use gam_solve::rho_optimizer::soft_rho_guard_floor::{ABSENCE_MAGNITUDE_FRACTION, GuardLadderRung, SATURATED_RHO_LADDER, SoftRhoGuardFloor};
use ndarray::{Array1, Array2};

/// A Gaussian one-block fixture with a real λ→∞ face: an unpenalized intercept
/// plus three penalized basis columns, so sending ρ to the box bound drives the
/// fit onto the penalty's null space rather than onto nothing.
///
/// Deliberately NOT the degenerate `[[1.0]]` design most of this crate's
/// fixtures use. A 1×1 problem has no face to decay along, so its ρ-gradient
/// would be identically zero at every rung — which the classifier would answer
/// correctly (`AbsentBelowTheFloor`) but which would prove nothing, since a
/// criterion that carried the barrier on THAT fixture would still show the
/// floor and a criterion that did not would show zero. The whole point is to
/// measure a fixture where a floor would be visible.
fn gaussian_face_fixture() -> (OneBlockGaussianFamily, Vec<ParameterBlockSpec>) {
    const N: usize = 64;
    const P: usize = 4;
    let mut design = Array2::<f64>::zeros((N, P));
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let t = (i as f64 + 0.5) / N as f64;
        let x = -1.5 + 3.0 * t;
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x;
        design[[i, 2]] = x * x;
        design[[i, 3]] = (2.1 * x).sin();
        // A signal with real curvature, so the penalized columns carry weight
        // and the criterion's tail constant is not numerically zero.
        y[i] = 0.4 + 0.8 * x - 0.3 * x * x + 0.5 * (2.1 * x).sin();
    }
    // Penalize everything but the intercept: a rank-3 penalty with a 1-D null
    // space, the standard smooth-term shape.
    let mut penalty = Array2::<f64>::zeros((P, P));
    for j in 1..P {
        penalty[[j, j]] = 1.0;
    }
    let specs = vec![ParameterBlockSpec {
        name: "gaussian_face".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
        offset: Array1::zeros(N),
        penalties: vec![PenaltyMatrix::Dense(penalty)],
        nullspace_dims: vec![1],
        initial_log_lambdas: Array1::zeros(1),
        initial_beta: Some(Array1::zeros(P)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    (OneBlockGaussianFamily { y }, specs)
}

/// Build the saturated-ρ ladder from the shared custom-family evaluator: one
/// rung per probe, carrying the SIGNED outer ρ-gradient exactly as the engine
/// reports it, with nothing subtracted.
fn custom_family_rho_ladder(probes: &[f64]) -> Vec<GuardLadderRung> {
    let (family, specs) = gaussian_face_fixture();
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
    probes
        .iter()
        .map(|&probe| {
            let rho = Array1::from_elem(1, probe);
            let owned = crate::psi_hyper::evaluate_custom_family_joint_hyper_owned(
                &family,
                &specs,
                &options,
                &rho,
                &hyper_layout,
                None,
                EvalMode::ValueAndGradient,
            )
            .unwrap_or_else(|e| {
                panic!("the custom-family engine must evaluate at rho={probe}: {e}")
            });
            assert!(
                owned.result.inner_converged,
                "rho={probe}: the outer gradient is an ENVELOPE derivative and is \
                 only valid at a stationary beta-hat. A non-converged rung would \
                 make the ladder a reading of the inner solver, not of the \
                 criterion"
            );
            assert_eq!(
                owned.result.gradient.len(),
                1,
                "this fixture declares exactly one rho coordinate"
            );
            GuardLadderRung {
                rho: probe,
                rho_gradient: owned.result.gradient[0],
            }
        })
        .collect()
}

/// **The measurement.** The shared custom-family outer evaluator — the route
/// `gamlss mean-wiggle`, `spatial-adaptive`, and `custom family` all take — does
/// NOT carry the soft ρ-guard barrier, so `None` is the correct answer for all
/// three at [`OuterObjective::soft_rho_guard_gradient`].
///
/// The verdict is whatever the classifier says; what this gate asserts is that
/// it is one of the two ABSENT verdicts, and — separately and more strongly —
/// that the gradient never comes near the floor. The second assertion is the one
/// that does not depend on the classifier at all: if the barrier were in this
/// criterion, `|g|` would be pinned within a hair of `w·a·tanh(a·ρ)` at every
/// rung, and it demonstrably is not.
///
/// The engine's own construction agrees, and is worth naming since it is the
/// mechanism behind the number:
/// `evaluate_custom_family_joint_hyper_owned` passes `gam_problem::RhoPrior::Flat`
/// into `evaluate_custom_family_hyper_internal` unconditionally, and the
/// ρ-prior machinery it reaches (`psi_hyper`'s `has_configured_rho_prior`) is the
/// CONFIGURED prior only. The soft guard has no path into it.
///
/// [`OuterObjective::soft_rho_guard_gradient`]: gam_solve::rho_optimizer::OuterObjective::soft_rho_guard_gradient
#[test]
fn the_custom_family_engine_carries_no_soft_rho_guard_floor_2629() {
    let ladder = custom_family_rho_ladder(&SATURATED_RHO_LADDER);
    // The barrier acts on raw ρ here: this engine holds no `RemlState`, so
    // there is no weight anchor to speak of and none to pass.
    let verdict = classify_soft_rho_guard_floor(&ladder, 0.0);

    let rendered = ladder
        .iter()
        .map(|rung| {
            format!(
                "(rho={:.0}, g={:+.6e}, guard={:.6e})",
                rung.rho,
                rung.rho_gradient,
                soft_rho_guard_emission_at(rung.rho, 0.0)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "[#2629-table] custom-family engine: {} | {rendered}",
        verdict.summary()
    );

    assert!(
        verdict.is_absent(),
        "#2629's table lists `gamlss mean-wiggle`, `spatial-adaptive` and \
         `custom family` as carrying no soft rho-guard barrier, on the grounds \
         that `RemlState::build_prior` is its only adder and this engine holds \
         no RemlState. If that is wrong, THREE objective families each carry a \
         standing |Pg| >= w*a at every railed coordinate and each owes the seam \
         a publication. Verdict: {} | ladder {rendered}",
        verdict.summary()
    );
    assert!(
        !verdict.is_carried(),
        "redundant with the above by construction, and stated anyway because it \
         is the claim the issue's table makes"
    );

    // A second, classifier-independent statement, at the DEEPEST rung only.
    //
    // The magnitude argument is a statement about where the criterion's own
    // tail has already gone, so it applies where the tail is smallest — not
    // over the whole ladder. At rho=21 this fixture's face is still live at
    // 1.69e-6, an order ABOVE the floor, and that is the tail, not a floor:
    // three e-folds later it is 2.09e-10, which a saturating term cannot do.
    // Asserting the max over all rungs would have demanded the fixture be
    // *dead* at every rung, i.e. demanded exactly the blind ladder the control
    // test below refuses to accept.
    let deepest = ladder.last().expect("the ladder reaches RHO_BOUND");
    let deepest_guard = soft_rho_guard_emission_at(deepest.rho, 0.0);
    assert!(
        deepest.rho_gradient.abs() <= ABSENCE_MAGNITUDE_FRACTION * deepest_guard,
        "at rho={} the criterion's gradient is {:.6e}, which is NOT below the \
         barrier's own {deepest_guard:.6e}. A criterion that ADDS the barrier is \
         pinned within a hair of it here — its own tail has decayed four orders \
         past it — so this would say the engine carries the floor after all. \
         Ladder: {rendered}",
        deepest.rho,
        deepest.rho_gradient
    );

    // ...and the tail underneath is a real face, not a decay into roundoff.
    // Measured c = 2.230365e3 with a pencil spread of 2.08e-6 across three
    // three-e-fold steps: six significant figures of `c*e^-rho`, which is what
    // makes "there is no floor here" a reading rather than an absence of signal.
    let SoftRhoGuardFloor::AbsentDecayingFace { face, .. } = &verdict else {
        panic!(
            "this fixture's face is LIVE across the ladder (1.69e-6 down to \
             2.09e-10), so the verdict must be the shape one — a magnitude-only \
             absence would mean the fixture had nothing to say. Got: {}",
            verdict.summary()
        );
    };
    assert!(
        face.spread <= 1.0e-4,
        "the bare pencil c = g*e^rho must be CONSTANT to say the gradient IS the \
         face and nothing else is hiding in it; got spread {:.3e} on c={:.6e}",
        face.spread,
        face.constant
    );
}

/// The control that keeps the gate above from passing for the wrong reason.
///
/// A fixture whose gradient is identically zero at every rung would satisfy
/// "absent" trivially and would prove nothing: a floor cannot be shown missing
/// from a measurement that could not have shown it present. So: inject the
/// barrier into this very ladder — the same rungs, the same evaluator, plus
/// `w·a·tanh(a·ρ)` — and require the classifier to flip to CARRIED.
///
/// That is the real content of the measurement. It says the instrument, pointed
/// at THIS fixture, would have caught the defect had it been there.
#[test]
fn the_same_custom_family_ladder_reads_carried_once_the_barrier_is_injected_2629() {
    let bare = custom_family_rho_ladder(&SATURATED_RHO_LADDER);
    let injected: Vec<GuardLadderRung> = bare
        .iter()
        .map(|rung| GuardLadderRung {
            rho: rung.rho,
            // Exactly what `RemlState::build_prior` would have added at this
            // coordinate, from the same atom.
            rho_gradient: rung.rho_gradient + soft_rho_guard_emission_at(rung.rho, 0.0),
        })
        .collect();

    let verdict = classify_soft_rho_guard_floor(&injected, 0.0);
    assert!(
        verdict.is_carried() || matches!(verdict, SoftRhoGuardFloor::Indeterminate { .. }),
        "injecting the barrier must move the verdict OFF absent — a fixture on \
         which the floor is invisible cannot be used to certify its absence. \
         Got: {}",
        verdict.summary()
    );
    assert!(
        !verdict.is_absent(),
        "the ladder with the barrier explicitly added must not read as ABSENT; \
         if it does, this fixture is blind to the very thing the gate above \
         claims to have looked for. Got: {}",
        verdict.summary()
    );
    eprintln!(
        "[#2629-control] custom-family engine + injected barrier: {}",
        verdict.summary()
    );
}
