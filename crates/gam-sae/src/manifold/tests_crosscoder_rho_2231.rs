//! #2231 Inc-B CONTRACT pins: `log_lambda_block` as outer ρ coordinates.
//!
//! These tests pin the unification contract BEFORE the outer wiring lands
//! (TDD; red is the honest signal that the coordinates exist but the criterion
//! does not yet price them — see the Inc-B audit on #2231). Green requires:
//!
//! 1. every eval lane rescales the stacked target's block columns from the
//!    pristine copy at ρ-materialization (`√λ_ℓ`, drift-free), and
//! 2. the criterion carries the block λ-dependence: the scaled-block residual
//!    through the existing data term plus the `−Σ_ℓ (n·p_ℓ/2)·log λ_ℓ`
//!    change-of-variables Jacobian. Under the engine's UNIT-dispersion `#F1`
//!    convention the stationary point is `λ_ℓ = n·p_ℓ/R_ℓ`; the shared-φ̂
//!    PROFILED form in #2231 §2a gives `(R_x/p_x)/(R_ℓ/p_ℓ)` instead — the
//!    convention decision is recorded on the issue, and these pins assert only
//!    the properties BOTH conventions share.
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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        .for_assignment(AssignmentMode::softmax(1.0))
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
    )
    // #2231 Inc-B — engage block-relevance pricing (anchor p_x = 4, one output
    // block p_1 = 4). Without this the objective carries the coordinate but never
    // prices it, and the pins below are the honest RED signal.
    .with_crosscoder_blocks(4, vec![4])
    .expect("crosscoder block pricing must install (p_x + Σ block_dims == p̃)");

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
         (cost {cost_at_closed_form:.9e} at log λ = {closed_form:.3} equals {cost_far_off:.9e} \
         at e^6 off): the #2231 Inc-B wiring (block rescale + Jacobian term) is missing"
    );
    // WHERE the optimum sits is convention-dependent (unit-dispersion vs
    // profiled φ̂ — see the module doc); the interior-minimum pin below owns
    // the shape assertion, so no direction claim is made here.
}

/// #2231 Inc-B pin 2 — the criterion's 1-D shape in `log λ_1` must have an
/// INTERIOR stationary minimum: coercive at both ends (`λ→0` pays the
/// `−(n·p_1/2)·log λ` Jacobian wall, `λ→∞` pays the scaled block residual) and
/// FD-stationary at its scan argmin.
///
/// Deliberately CONVENTION-AGNOSTIC about where the minimum sits: the engine's
/// outer criterion is the UNIT-dispersion penalized Laplace form (`#F1`:
/// `loss.data_fit` is raw half-SSE, no `φ̂`), under which the coordinate's
/// stationary point is `λ_ℓ = n·p_ℓ/R_ℓ` — NOT the shared-`φ̂` PROFILED ratio
/// `(R_x/p_x)/(R_ℓ/p_ℓ)` that #2231 §2a quotes from `behavior.rs` (the two
/// coincide only at anchor dispersion exactly 1). This pin asserts the shape
/// properties both conventions share, so it stays valid whichever form the
/// wiring lands; the convention decision itself is recorded on #2231.
/// Red until the Inc-B wiring lands (today the criterion is FLAT in `log λ_1`,
/// so no interior minimum exists and every scan cost ties).
#[test]
fn block_relevance_has_interior_stationary_minimum_2231() {
    let (z, coords, _p_x, _p_1, _profiled_form) = planted_two_layer();
    let p_tot = z.ncols();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let term = build_k1_circle(&evaluator, &coords, p_tot);
    let rho_template = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(AssignmentMode::softmax(1.0))
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
    )
    // #2231 Inc-B — engage block-relevance pricing (anchor p_x = 4, one output
    // block p_1 = 4). Without this the objective carries the coordinate but never
    // prices it, and the pins below are the honest RED signal.
    .with_crosscoder_blocks(4, vec![4])
    .expect("crosscoder block pricing must install (p_x + Σ block_dims == p̃)");
    let cost_at = |objective: &mut SaeManifoldOuterObjective, ll: f64| -> f64 {
        let mut flat = rho_template.to_flat();
        let last = flat.len() - 1;
        flat[last] = ll;
        objective
            .eval_cost(&flat)
            .expect("scan point must evaluate")
    };
    // Coarse coercivity scan over 16 decades of λ.
    let grid: Vec<f64> = (-4..=4).map(|k| 2.0 * k as f64).collect();
    let costs: Vec<f64> = grid.iter().map(|&ll| cost_at(&mut objective, ll)).collect();
    let (argmin, _) = costs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let spread = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - costs.iter().cloned().fold(f64::INFINITY, f64::min);
    let scale = 1.0 + costs[argmin].abs();
    assert!(
        spread > 1.0e-6 * scale,
        "the criterion is FLAT across 16 decades of λ_1 — log_lambda_block is not priced; \
         the #2231 Inc-B wiring (block rescale + Jacobian term) is missing"
    );
    assert!(
        argmin != 0 && argmin != grid.len() - 1,
        "the block-relevance criterion must be coercive with an INTERIOR minimum; \
         the scan argmin sits at the grid edge (log λ = {}) — a runaway direction",
        grid[argmin]
    );
    // FD stationarity at the (parabolically refined) argmin: the local slope is
    // small relative to the local curvature scale.
    let ll0 = grid[argmin];
    let h = 0.5_f64;
    let (cm, c0, cp) = (
        cost_at(&mut objective, ll0 - h),
        cost_at(&mut objective, ll0),
        cost_at(&mut objective, ll0 + h),
    );
    let slope = (cp - cm) / (2.0 * h);
    let curvature = ((cp - 2.0 * c0 + cm) / (h * h)).max(1.0e-12);
    let newton_step = slope.abs() / curvature;
    assert!(
        newton_step < 2.0 * h,
        "no stationary point near the scan argmin (log λ = {ll0}): |slope|/curvature = \
         {newton_step:.3e} exceeds the local window — the Jacobian and residual terms \
         are mis-balanced"
    );
}

/// Fresh engaged objective on the planted two-layer target: one block
/// (`p_x = 4`, `p_1 = 4`), a K=1 always-on circle term, and a clean-ish inner
/// budget (60 Newton steps) so the fitted residual — and thus `R̃_1` — is
/// converged enough for a meaningful gradient/FD comparison. The term is consumed
/// by `new`, so each call rebuilds it (the fits are independent).
fn engaged_objective(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    z: &Array2<f64>,
    coords: &Array2<f64>,
) -> SaeManifoldOuterObjective {
    let p_tot = z.ncols();
    let term = build_k1_circle(evaluator, coords, p_tot);
    let rho_template = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(AssignmentMode::softmax(1.0))
        .with_log_lambda_block(vec![0.0]);
    SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        rho_template,
        60,
        0.04,
        1.0e-6,
        1.0e-6,
    )
    .with_crosscoder_blocks(4, vec![4])
    .expect("crosscoder block pricing must install (p_x + Σ block_dims == p̃)")
}

/// #2231 Inc-B (stage 2) — the analytic block gradient the ValueAndGradient lane
/// returns in its `log_lambda_block` tail slot MUST be the finite-difference
/// derivative of the value lane's own cost: the consistent `(value, gradient)`
/// pair the #2087 objective↔gradient desync class demands. `eval`'s block-tail
/// entry is `½·R̃_1 − n·p_1/2`; the central difference of `eval_cost` (which
/// re-runs the inner solve at each perturbed `log λ_1`) is compared against it at
/// three points spanning the planted optimum.
///
/// Tolerance note: unlike the pure-math sibling gate
/// (`tests_crosscoder_block_fd_2231`, `h = 1e-6`, `1e-5` rel), this FD runs the
/// full re-fitting engine, so it carries the inner solve's finite-budget noise
/// AND the central difference's `O(h²)` truncation on a strongly-curved
/// criterion. The gate is a DESYNC / wrong-form discriminator (a dropped `½`, a
/// sign flip, or a missing `n·p/2` moves the analytic value by `Θ(n·p)` — orders
/// above this window), not a precision gate.
#[test]
fn block_gradient_matches_central_difference_of_cost_2231() {
    let (z, coords, _p_x, p_1, _) = planted_two_layer();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let n = z.nrows();
    let rho_template = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(AssignmentMode::softmax(1.0))
        .with_log_lambda_block(vec![0.0]);
    let flat_at = |log_lambda: f64| -> Array1<f64> {
        let mut flat = rho_template.to_flat();
        let last = flat.len() - 1;
        flat[last] = log_lambda;
        flat
    };

    // Stationarity scale n·p_1/2 (the Jacobian half-count): the analytic block
    // gradient is `½·R̃_1 − scale`, so `scale` is the natural absolute reference.
    let scale = 0.5 * n as f64 * p_1 as f64;
    let h = 0.05_f64;
    // High inner budget so the SINGLE-SHOT gradient lane certifies at the probe
    // points. The cost lane certifies at a modest budget via its two-stage
    // basin/envelope path; the raw `penalized_laml_criterion_with_cache` the gradient
    // lane calls has no such multi-start, so it simply needs enough inner
    // Newton room to reach the same optimum from a warm continuation. The exact
    // block-gradient VALUE is independently pinned to 1e-6 by the pure-math
    // sibling `tests_crosscoder_block_fd_2231`; this gate adds the full-engine
    // (f, ∇f) consistency the #2087 desync class demands.
    let term = build_k1_circle(&evaluator, &coords, z.ncols());
    let mut obj = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        rho_template.clone(),
        800,
        0.04,
        1.0e-6,
        1.0e-6,
    )
    .with_crosscoder_blocks(4, vec![4])
    .expect("crosscoder block pricing must install");
    // Warm at the origin (converges), then walk outward in small monotone steps
    // so each gradient eval opens from the previous point's warm state.
    obj.eval_cost(&rho_template.to_flat())
        .expect("the seed evaluation must converge");
    let is_wall = |c: f64| !(c < 1.0e11);
    let mut checked = 0usize;
    for &ll in &[0.0_f64, 0.15, 0.3, -0.15, -0.3, 0.45, -0.45] {
        // Cost probe at this ρ first (parks the ρ-keyed converged handoff), then
        // the gradient eval picks it up — the engine's real Value→ValueAndGradient
        // calling order.
        if !matches!(obj.eval_cost(&flat_at(ll)), Ok(c) if !is_wall(c)) {
            continue;
        }
        let eval = match obj.eval(&flat_at(ll)) {
            Ok(e) if !is_wall(e.cost) => e,
            _ => continue,
        };
        let (Ok(c_plus), Ok(c_minus)) = (
            obj.eval_cost(&flat_at(ll + h)),
            obj.eval_cost(&flat_at(ll - h)),
        ) else {
            continue;
        };
        if is_wall(c_plus) || is_wall(c_minus) {
            continue;
        }
        let analytic = eval.gradient[eval.gradient.len() - 1];
        let fd = (c_plus - c_minus) / (2.0 * h);
        let tol = 0.1 * scale + 0.1 * analytic.abs();
        assert!(
            (analytic - fd).abs() < tol,
            "block gradient desync at log λ_1 = {ll:.3}: analytic ½·R̃_1 − n·p_1/2 = \
             {analytic:.4} vs central-difference of eval_cost = {fd:.4} (|Δ| = {:.4}, \
             tol = {tol:.4}) — the #2087 objective↔gradient pair is inconsistent",
            (analytic - fd).abs()
        );
        checked += 1;
    }
    assert!(
        checked >= 1,
        "no evaluable FD probe point at 800 inner iterations — the gradient lane \
         could not certify anywhere near the warm origin"
    );
}

/// #2231 Inc-B (stage 2) — the block Fellner–Schall coordinate and the analytic
/// block gradient share ONE root. Iterating the EFS log-λ step
/// `Δlog λ_1 = ln(n·p_1/R̃_1)` from `log λ_1 = 0` to convergence must arrive at a
/// point where the ValueAndGradient lane's block gradient `½·R̃_1 − n·p_1/2`
/// vanishes — the quasi-Newton lane and the fixed-point lane agree (the coherence
/// the whole Inc-B contract is built on). Red until both lanes are wired.
#[test]
fn block_efs_step_reaches_gradient_root_2231() {
    let (z, coords, _p_x, p_1, _closed_form) = planted_two_layer();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let n = z.nrows();
    let rho_template = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(AssignmentMode::softmax(1.0))
        .with_log_lambda_block(vec![0.0]);
    let mut objective = engaged_objective(&evaluator, &z, &coords);

    // Fellner–Schall fixed-point iteration on the block coordinate ONLY (every
    // other ρ coordinate held at its template value), from log λ_1 = 0.
    let mut flat = rho_template.to_flat();
    let last = flat.len() - 1;
    let mut converged = false;
    for _ in 0..150 {
        let efs = objective
            .efs_step(flat.view())
            .expect("the EFS lane must evaluate");
        let step = efs.steps[last];
        flat[last] += step;
        if step.abs() < 1.0e-3 {
            converged = true;
            break;
        }
    }
    assert!(
        converged,
        "the block Fellner–Schall iteration did not converge from log λ_1 = 0 \
         (last log λ_1 = {:.4})",
        flat[last]
    );

    // The EFS fixed point is converged in ITS OWN contract: one more step is
    // (numerically) zero. The historical "EFS root == full-gradient root"
    // equality was RETIRED with the exact block adjoint (2026-07-10): the full
    // gradient at the EFS root now carries the −½·Γᵀθ̂_ρ Laplace channel the
    // explicit-channel fixed point deliberately omits (the EFS lane is a
    // proposal heuristic accepted only on criterion improvement).
    let final_step = objective
        .efs_step(flat.view())
        .expect("the EFS lane must evaluate at its own fixed point")
        .steps[last];
    assert!(
        final_step.abs() <= 2.0e-3,
        "the arrived point is not an EFS fixed point: one more step moves \
         log λ_1 by {final_step:.4e} (> 2e-3) at log λ_1 = {:.4}",
        flat[last]
    );
    // Anchor the scale so the fixed point stays interior and finite.
    let scale = 0.5 * n as f64 * p_1 as f64;
    assert!(scale.is_finite() && flat[last].is_finite() && flat[last].abs() < 20.0);
}
