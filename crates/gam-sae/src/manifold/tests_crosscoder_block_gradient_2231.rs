//! #2231 — the crosscoder block weight `log λ_ℓ` entry of the outer ρ-gradient is
//! the derivative of the criterion the objective reports.
//!
//! Block `ℓ` enters the criterion only through its target columns `Z̃_ℓ = √λ_ℓ·Y_ℓ`
//! and the change-of-variables Jacobian `−(n·p_ℓ/2)·log λ_ℓ`. The check is
//! factor-wise at one converged state, as in #2935:
//! - the fixed-state partial (everything but the implicit correction) against a
//!   difference of the frozen cost over `log λ_ℓ`;
//! - the implicit right-hand side against a difference of the inner gradient;
//! - the implicit correction `½Γᵀθ̂` through the frozen cost along `θ̂ = −A⁺g_λ`.
#![cfg(test)]
use super::tests_kappa_outer_gradient_2935::{
    arrow_gap, arrow_dot, arrow_max, arrow_norm, arrow_scaled_difference, converged_anchor,
    curvature_fixture, displaced, inner_gradient, richardson,
};
use super::*;
use ndarray::{Array2, s};

const ANCHOR_WIDTH: usize = 1;

fn scaled_target(target: &Array2<f64>, log_lambda: f64) -> Array2<f64> {
    let mut out = target.clone();
    out.slice_mut(s![.., ANCHOR_WIDTH..])
        .mapv_inplace(|value| value * (0.5 * log_lambda).exp());
    out
}

#[test]
fn crosscoder_block_gradient_factors_are_derivatives_of_the_criterion_2231() {
    let (term, target, rho) = curvature_fixture();
    let (state, anchor, _) = converged_anchor(term, &target, rho);
    let n = target.nrows() as f64;
    let block_width = target.ncols() - ANCHOR_WIDTH;
    let mut at = anchor.clone();
    at.log_lambda_block = vec![0.0];
    let mut objective =
        SaeManifoldOuterObjective::new(state, target.clone(), None, at, 0, 0.4, 1.0e-6, 1.0e-6)
            .with_crosscoder_blocks(ANCHOR_WIDTH, vec![block_width])
            .expect("one anchor column and one output block");
    let at = objective.baseline_rho.clone();
    let block = at.block_flat_range().start;
    let evaluation = objective
        .evaluate_outer_criterion_route(&at, true, false)
        .expect("the dense route prices the state");
    let complete = objective
        .analytic_gradient_for_outer_evaluation(&at, &evaluation)
        .expect("the dense route differentiates the state")[block];
    let state = objective.term.clone();

    let mut priced = state.clone();
    let (_cost, loss, cache, geometry) = priced
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &at,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
            true,
        )
        .expect("the criterion prices the converged state");
    let geometry = geometry.expect("the dense criterion hands out the block it priced");
    let components = priced
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &at,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("dense gradient components at the converged state");
    let implicit = components.third_order_correction[block];
    let partial = complete - implicit;
    let residual = inner_gradient(&state, target.view(), &at);
    let g_block = priced
        .crosscoder_block_ift_rhs(&cache, target.view(), ANCHOR_WIDTH..target.ncols())
        .expect("block implicit right-hand side");
    let a_pinv_g = priced
        .solve_exact_stationarity(&at, target.view(), &cache, &g_block)
        .expect("A⁺ g_λ");
    let theta_hat = SaeArrowVector {
        t: a_pinv_g.t.mapv(|value| -value),
        beta: a_pinv_g.beta.mapv(|value| -value),
    };

    let price = |at_state: &SaeManifoldTerm, log_lambda: f64| -> f64 {
        let mut arm = at_state.clone();
        let mut rho_at = at.clone();
        rho_at.log_lambda_block = vec![log_lambda];
        arm.penalized_quasi_laplace_criterion_with_cache(
            scaled_target(&target, log_lambda).view(),
            &rho_at,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("the criterion prices the fixed state")
        .0 - 0.5 * n * block_width as f64 * log_lambda
    };
    let cost_over_lambda =
        |step: f64| -> f64 { (price(&state, step) - price(&state, -step)) / (2.0 * step) };
    let gradient_over_lambda = |step: f64| -> SaeArrowVector {
        arrow_scaled_difference(
            &inner_gradient(&state, scaled_target(&target, step).view(), &at),
            &inner_gradient(&state, scaled_target(&target, -step).view(), &at),
            step,
        )
    };
    let direction_scale = arrow_norm(&theta_hat).max(1.0);
    let cost_along_response = |eps: f64| -> f64 {
        (price(&displaced(&state, &theta_hat, eps), 0.0)
            - price(&displaced(&state, &theta_hat, -eps), 0.0))
            / (2.0 * eps)
    };

    // The fixture's joint block turns indefinite by `|log λ| = 4e-3`, and the log-determinant
    // bends sharply on the way there, so the cost is differenced well inside that radius.
    let cost_step = 3.125e-5_f64;
    let (partial_fd, partial_spread) =
        richardson(cost_over_lambda(cost_step), cost_over_lambda(0.5 * cost_step));
    let step = 1.0e-3_f64;
    let g_coarse = gradient_over_lambda(step);
    let g_fine = gradient_over_lambda(0.5 * step);
    let response_step = 1.0e-4 / direction_scale;
    let (directional_fd, directional_spread) = richardson(
        cost_along_response(response_step),
        cost_along_response(0.5 * response_step),
    );
    let implicit_fd = directional_fd - arrow_dot(&residual, &theta_hat);
    println!(
        "[#2231 block] complete={complete:.12e} partial={partial:.12e} partial_fd={partial_fd:.12e} \
         (spread {partial_spread:.3e}) implicit={implicit:.12e} implicit_fd={implicit_fd:.12e} \
         (spread {directional_spread:.3e}) |g_λ|∞={:.6e} max|g_λ − fd|={:.3e} (spread {:.3e}) \
         |g|={:.3e}",
        arrow_max(&g_block),
        arrow_gap(&g_block, &g_fine),
        arrow_gap(&g_coarse, &g_fine),
        arrow_norm(&residual)
    );

    let partial_tolerance = 10.0 * partial_spread + 1.0e-6 * partial_fd.abs().max(1.0);
    assert!(
        (partial - partial_fd).abs() <= partial_tolerance,
        "fixed-state block partial {partial} is not the frozen cost's log λ derivative \
         {partial_fd} (|Δ| = {}, tolerance {partial_tolerance})",
        (partial - partial_fd).abs()
    );
    let g_tolerance =
        10.0 * arrow_gap(&g_coarse, &g_fine) + 1.0e-6 * arrow_max(&g_fine).max(1.0);
    assert!(
        arrow_gap(&g_block, &g_fine) <= g_tolerance,
        "g_λ is not the inner gradient's log λ derivative (max |Δ| = {}, tolerance {g_tolerance})",
        arrow_gap(&g_block, &g_fine)
    );
    let implicit_tolerance = 10.0 * directional_spread + 1.0e-6 * directional_fd.abs().max(1.0);
    assert!(
        (implicit - implicit_fd).abs() <= implicit_tolerance,
        "the implicit correction {implicit} is not ½Γᵀθ̂ = {implicit_fd} measured along θ̂ \
         (|Δ| = {}, tolerance {implicit_tolerance})",
        (implicit - implicit_fd).abs()
    );
}

/// #3270 — the streaming exact-A route prices an output-scale coordinate's
/// `½tr(A⁺ ∂A/∂log λ)` from its probe bundle, row by row, and so returns the dense route's
/// entry at one state. Before, it refused the coordinate for want of that trace.
fn assert_streaming_output_scale_entry_matches_dense(label: &str, global_dispersion: bool) {
    let (term, target, rho) = curvature_fixture();
    let (state, anchor, _) = converged_anchor(term, &target, rho);
    let block_width = target.ncols() - ANCHOR_WIDTH;
    let mut at = anchor.clone();
    at.log_lambda_block = vec![0.0];
    // A zero inner budget prices both routes at the converged state.
    let objective =
        SaeManifoldOuterObjective::new(state, target.clone(), None, at, 0, 0.4, 1.0e-6, 1.0e-6);
    let mut objective = if global_dispersion {
        objective
            .with_global_dispersion(0)
            .expect("one output-scale coordinate over every column")
    } else {
        objective
            .with_crosscoder_blocks(ANCHOR_WIDTH, vec![block_width])
            .expect("one anchor column and one output block")
    };
    let at = objective.baseline_rho.clone();
    let coord = at.block_flat_range().start;
    let dense_evaluation = objective
        .evaluate_outer_criterion_route(&at, true, false)
        .expect("the dense route prices the state");
    let dense = objective
        .analytic_gradient_for_outer_evaluation(&at, &dense_evaluation)
        .expect("the dense route differentiates the state")[coord];
    let streaming_evaluation = objective
        .evaluate_outer_criterion_route(&at, false, false)
        .expect("the streaming route prices the state");
    let streaming = objective
        .analytic_gradient_for_outer_evaluation(&at, &streaming_evaluation)
        .expect("the streaming route differentiates the output-scale coordinate")[coord];
    println!(
        "[#3270 {label}] dense cost={:.12e} streaming cost={:.12e} dense dV/dlog λ={dense:.12e} \
         streaming dV/dlog λ={streaming:.12e}",
        dense_evaluation.cost, streaming_evaluation.cost
    );
    assert!(
        (dense_evaluation.cost - streaming_evaluation.cost).abs()
            <= 1.0e-8 * dense_evaluation.cost.abs().max(1.0),
        "{label}: both routes must price one state ({} vs {})",
        dense_evaluation.cost,
        streaming_evaluation.cost
    );
    assert!(
        dense.abs() > 1.0e-3,
        "{label}: the entry must be material for route parity to mean anything ({dense})"
    );
    let tolerance = 1.0e-6 * dense.abs().max(1.0);
    assert!(
        (streaming - dense).abs() <= tolerance,
        "{label}: the streaming entry {streaming} is not the dense entry {dense} at one state \
         (|Δ| = {}, tolerance {tolerance})",
        (streaming - dense).abs()
    );
}

#[test]
fn streaming_route_crosscoder_block_gradient_matches_the_dense_route_3270() {
    assert_streaming_output_scale_entry_matches_dense("crosscoder block", false);
}

#[test]
fn streaming_route_global_dispersion_gradient_matches_the_dense_route_3270() {
    assert_streaming_output_scale_entry_matches_dense("global dispersion", true);
}
