//! #2231 Inc-B CONTRACT pins: `log_lambda_block` as outer ρ coordinates.
//!
//! These tests pin the unification contract BEFORE the outer wiring lands
//! (TDD; red is the honest signal that the coordinates exist but the criterion
//! does not yet price them — see the Inc-B audit on #2231). Green requires:
//!
//! 1. every eval lane rescales the stacked target's block columns from the
//!    pristine copy at ρ-materialization (`√λ_ℓ`, drift-free), and
//! 2. the criterion carries the profiled block form
//!    `(n·p̃/2)·log(RSS/(n·p̃)) − Σ_ℓ (n·p_ℓ/2)·log λ_ℓ`
//!    whose stationary point is the landed M1 closed form
//!    `λ_ℓ = (R_x/p_x)/(R_ℓ/p_ℓ)` (behavior.rs:703).
//!
//! The scan below is a TEST oracle over a 1-D grid of candidate λ values —
//! it verifies the criterion's shape; production selection stays REML through
//! the outer engine (no grid search in production).

use super::*;
use gam_solve::rho_optimizer::OuterObjective;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn noise_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        // splitmix-style deterministic uniform in [-1, 1] — same convention as
        // the sibling crosscoder tests (no external RNG dependency).
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A K=1 always-on softmax circle term at augmented width `p_tot`, mirroring
/// the sibling `tests_crosscoder_multiblock` builder (private there).
fn build_k1_circle(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> SaeManifoldTerm {
    let n = coords.nrows();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "cc2231",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), 40.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

/// Planted two-layer stacked target: anchor (clean) + one block (noisier),
/// both harmonic images of the SAME circle coordinate. Returns
/// `(stacked_target, coords, p_x, p_1, closed_form_log_lambda_1)` where the
/// closed form is computed from the PLANTED noise variances — the population
/// value of `λ_1 = (R_x/p_x)/(R_1/p_1)` the fitted residuals estimate.
fn planted_two_layer() -> (Array2<f64>, Array2<f64>, usize, usize, f64) {
    let n = 96usize;
    let (p_x, p_1) = (4usize, 4usize);
    let (sigma_x, sigma_1) = (0.03_f64, 0.12_f64);
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let mut z = Array2::<f64>::zeros((n, p_x + p_1));
    let mut nx = noise_stream(0x2231_0001);
    let mut n1 = noise_stream(0x2231_0002);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + sigma_x * nx();
        z[[i, 1]] = theta.sin() + sigma_x * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma_x * nx();
        z[[i, 3]] = 0.3 * (2.0 * theta).sin() + sigma_x * nx();
        z[[i, p_x]] = theta.cos() + sigma_1 * n1();
        z[[i, p_x + 1]] = 0.8 * theta.sin() + sigma_1 * n1();
        z[[i, p_x + 2]] = 0.5 * (2.0 * theta).sin() + sigma_1 * n1();
        z[[i, p_x + 3]] = 0.3 * theta.cos() + sigma_1 * n1();
    }
    // Population closed form from the planted per-column noise variances
    // (uniform noise on [-1,1] scaled by σ has variance σ²/3; the /3 cancels
    // in the ratio): λ_1 = σ_x² / σ_1².
    let closed_form_log_lambda = (sigma_x * sigma_x / (sigma_1 * sigma_1)).ln();
    (z, coords, p_x, p_1, closed_form_log_lambda)
}

/// #2231 Inc-B pin 1 — the outer criterion must PRICE the block-relevance
/// coordinate: two evaluations differing only in `log λ_1` must return
/// materially different costs, and the planted closed-form value must beat a
/// grossly mis-weighted one. Until the Inc-B wiring lands the objective
/// ignores `log_lambda_block` entirely (both costs identical) and this test
/// is the honest RED pin for the missing unification core.
#[test]
fn outer_criterion_prices_block_relevance_2231() {
    let (z, coords, _p_x, _p_1, closed_form) = planted_two_layer();
    let p_tot = z.ncols();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let term = build_k1_circle(&evaluator, &coords, p_tot);
    let rho_template = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .with_log_lambda_block(vec![0.0]);
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        rho_template.clone(),
        8,
        0.04,
        1.0e-6,
        1.0e-6,
    );

    let flat_at = |log_lambda: f64| -> Array1<f64> {
        let mut flat = rho_template.to_flat();
        let last = flat.len() - 1;
        flat[last] = log_lambda;
        flat
    };
    let cost_at_closed_form = objective
        .eval_cost(&flat_at(closed_form))
        .expect("criterion at the closed-form λ must evaluate");
    let cost_far_off = objective
        .eval_cost(&flat_at(closed_form + 6.0))
        .expect("criterion at e^6-misweighted λ must evaluate");
    let scale = 1.0 + cost_at_closed_form.abs().max(cost_far_off.abs());
    assert!(
        (cost_at_closed_form - cost_far_off).abs() > 1.0e-6 * scale,
        "the outer criterion does not price log_lambda_block at all \
         (cost {cost_at_closed_form:.9e} at the closed form == {cost_far_off:.9e} at e^6 off): \
         the #2231 Inc-B wiring (block rescale + profiled criterion + Jacobian term) is missing"
    );
    assert!(
        cost_at_closed_form < cost_far_off,
        "the closed-form λ_1 (log λ = {closed_form:.4}) must be cheaper than an e^6 \
         mis-weighting; got {cost_at_closed_form:.6e} vs {cost_far_off:.6e}"
    );
}

/// #2231 Inc-B pin 2 — the criterion's 1-D shape in `log λ_1` must be
/// minimized near the planted closed form `λ_1 = σ_x²/σ_1²` (a coarse scan
/// oracle: the minimum over the scanned grid sits within one grid step of the
/// closed form). Red together with pin 1 until the wiring lands.
#[test]
fn block_relevance_stationary_point_matches_m1_closed_form_2231() {
    let (z, coords, _p_x, _p_1, closed_form) = planted_two_layer();
    let p_tot = z.ncols();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let term = build_k1_circle(&evaluator, &coords, p_tot);
    let rho_template = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .with_log_lambda_block(vec![0.0]);
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        rho_template.clone(),
        8,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    let step = 1.0_f64;
    let offsets: Vec<f64> = (-3..=3).map(|k| k as f64 * step).collect();
    let mut best = (f64::INFINITY, f64::NAN);
    for &off in &offsets {
        let mut flat = rho_template.to_flat();
        let last = flat.len() - 1;
        flat[last] = closed_form + off;
        let cost = objective
            .eval_cost(&flat)
            .expect("scan point must evaluate");
        if cost < best.0 {
            best = (cost, off);
        }
    }
    assert!(
        best.1.abs() <= step,
        "criterion minimum over the log λ_1 scan sits {} steps from the M1 closed form \
         (population log λ_1 = {closed_form:.4}); the profiled block criterion's stationary \
         point must reproduce the landed λ_ℓ = (R_x/p_x)/(R_ℓ/p_ℓ) fixed point",
        best.1 / step
    );
}
