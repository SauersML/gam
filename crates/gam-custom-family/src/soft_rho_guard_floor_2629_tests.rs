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

