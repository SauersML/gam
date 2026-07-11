//! #1007 planted-bifurcation oracle: when the data admits TWO equally good
//! manifold completions, the certified curvature-homotopy walk must NOT
//! silently pick a branch and report a certified unique continuation.
//!
//! Fixture: a single circle atom (K = 1, d = 1, one harmonic) fit to the
//! exactly symmetric union of two planted circles living in mutually
//! orthogonal ambient planes — same radius, same angle ladder, same row
//! count. A one-atom dictionary must commit to ONE of the planes, and by
//! construction both choices have identical objective value: the η = 0
//! Eckart-Young anchor has a tied boundary singular pair (σ_r = σ_{r+1}),
//! so the global rank-2 relaxation is non-unique and no certified unique
//! branch to η = 1 exists.
//!
//! Contract under test (#1007 build-plan item 4): the walk must produce a
//! DETECTED event — either a recorded `CurvatureBifurcation` (pivot
//! collapse / tied-anchor detection) or a refusal to certify
//! (`arrived = false`, deferring to the documented seed cascade). Arriving
//! with `bifurcation: None` on this fixture is precisely the "silent branch
//! choice" failure mode the certificate exists to prevent.

use gam::terms::sae::manifold::CurvatureWalkReport;
use gam::terms::latent::LatentManifold;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldOuterObjective, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2, s};
use std::sync::Arc;

const N: usize = 400;
const P: usize = 24;
const M: usize = 3; // const + 1 harmonic -> circle
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

/// Two p×2 orthonormal frames with mutually orthogonal column spans
/// (deterministic Gram-Schmidt of four smooth ambient vectors).
fn planted_orthogonal_frames() -> (Array2<f64>, Array2<f64>) {
    let mut raw = Array2::<f64>::zeros((P, 4));
    for j in 0..4 {
        for i in 0..P {
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((P, 4));
    for j in 0..4 {
        let mut v = raw.column(j).to_owned();
        for prev in 0..j {
            let qp = q.column(prev);
            let dot: f64 = qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for i in 0..P {
                v[i] -= dot * qp[i];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..P {
            q[[i, j]] = v[i] / nrm;
        }
    }
    let u_a = q.slice(s![.., 0..2]).to_owned();
    let u_b = q.slice(s![.., 2..4]).to_owned();
    (u_a, u_b)
}

/// Exactly symmetric two-completion response: even rows trace the circle in
/// plane A, odd rows trace the SAME angle ladder in plane B. Equal radius,
/// equal counts, no noise — the per-plane second-moment matrices are
/// identical, so the rank-2 Eckart-Young problem has a tied boundary
/// singular pair and two equally good completions.
fn planted_symmetric_response() -> Array2<f64> {
    let (u_a, u_b) = planted_orthogonal_frames();
    let mut z = Array2::<f64>::zeros((N, P));
    let half = N / 2;
    for i in 0..N {
        let pair_idx = (i / 2) as f64;
        let t = pair_idx / half as f64; // same ladder for both planes
        let (c, s_) = (
            (2.0 * std::f64::consts::PI * t).cos(),
            (2.0 * std::f64::consts::PI * t).sin(),
        );
        let frame = if i % 2 == 0 { &u_a } else { &u_b };
        for col in 0..P {
            z[[i, col]] = frame[[col, 0]] * c + frame[[col, 1]] * s_;
        }
    }
    z
}

/// Cold one-atom term with a symmetry-neutral coordinate ladder and a zero
/// decoder — the walk's η = 0 anchor stage owns the linear-span solve, so
/// nothing in the seed breaks the planted tie.
fn build_cold_single_atom_term() -> SaeManifoldTerm {
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let coords = Array2::from_shape_fn((N, 1), |(i, _)| ((i as f64) * 0.061_803).rem_euclid(1.0));
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new(
        "circle_0".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((M, P)),
        Array2::<f64>::eye(M),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

#[test]
fn planted_two_completion_symmetry_is_a_detected_event_not_a_silent_branch_choice() {
    let z = planted_symmetric_response();
    let term = build_cold_single_atom_term();
    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); 1]);
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );

    let walk = objective.run_curvature_homotopy_entry();
    match walk {
        Err(err) => {
            // A hard anchor-construction failure is loud and hands control to
            // the seed cascade — an acceptable (non-silent) outcome.
            println!("planted-bifurcation walk hard-errored (non-silent outcome): {err}");
        }
        Ok(arrived) => {
            let report: CurvatureWalkReport = objective
                .curvature_walk_report()
                .expect("a walk that ran must record a report")
                .clone();
            println!("=== SAE planted-bifurcation oracle (#1007) ===");
            println!(
                "arrived={} eta_steps={} step_halvings={} reseeds={} collapse_events={} \
                 anchor_residual_norm_sq={:.6e} bifurcation={:?}",
                report.arrived,
                report.eta_steps,
                report.step_halvings,
                report.reseeds,
                report.collapse_events,
                report.anchor_residual_norm_sq,
                report.bifurcation,
            );
            assert!(
                !(arrived && report.bifurcation.is_none()),
                "the walk certified a unique continuation (arrived=true, no bifurcation) on a \
                 fixture with TWO exactly equally good completions — this is the silent branch \
                 choice #1007's certificate exists to prevent: a tied Eckart-Young boundary \
                 pair (σ_r = σ_r+1) means no unique global anchor branch exists"
            );
        }
    }
}
