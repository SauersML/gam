#![cfg(test)]
//! #4077 — finite-difference receipts for the half-line mass's DERIVATIVE channels.
//!
//! `pinned_half_line_channels` hands every exact-`A` route three legs: the θ-adjoint
//! (σ-leg through the `PinnedSlotWeight` adjoint + gradient-leg rows
//! `2κ₁·sign(g)·[A_raw]_u`), the ρ curvature trace, and — when crosscoder pricing
//! spans are installed — the #2231 span probe. The maintainer's suite FDs the ARD
//! ρ-partial end to end (`pinned_half_line_mass_enters_the_outer_gradient_4077`),
//! which contracts the θ-adjoint along ONE response direction. What it never
//! measures, and these tests do:
//!
//! 1. the θ-adjoint AS A VECTOR: `Σ_u c_u` displaced along an arbitrary inner
//!    direction `d` (zero on the pinned slots, where production responses are
//!    zero) must differentiate to `⟨gamma, d⟩` in FULL `log|A|` units — the
//!    module's own units convention, verified rather than trusted;
//! 2. the per-slot Jacobian claim `d g_u = ⟨row_curvature, d_t⟩ +
//!    ⟨border_curvature, d_β⟩` at GENUINELY pinned slots (the operands test
//!    checks the rows against dense `A` at an interior slot named as pinned —
//!    equality with `A`, not the derivative of `g`);
//! 3. the #2231 crosscoder span leg: the probe scales the block's target columns
//!    by `1.5` and reads `σ(1.5) − σ(1)` as `∂σ/∂log λ`. Since `c = √λ` and
//!    `A_raw` is affine in the target, `[σ(1+δ/2) − σ(1−δ/2)]/(2δ)` must
//!    reproduce it to rounding — an independent measurement with different
//!    probe points. The block's implicit right-hand side entries at the pinned
//!    slots are checked the same way against `∂g_u/∂log λ`.

use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

const LO: f64 = -1.0;
const HI: f64 = 1.0;

fn interval_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let p = 2usize;
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch basis"));
    // Generating latent: three rows sit beyond the interval's upper end, so the
    // fitted coordinates of those rows are pushed onto `hi` with `g < 0` there.
    let truth: [f64; 8] = [-0.7, -0.3, 0.1, 0.4, 1.6, 1.8, 2.0, 0.8];
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth[row].clamp(-0.9, 0.9));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("coords evaluate");
    let width = phi.ncols();
    let decoder = Array2::<f64>::from_shape_fn((width, p), |(b, o)| {
        [[0.1, -0.2], [1.0, 0.6], [0.2, -0.3]][b][o]
    });
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let s = truth[row];
        for o in 0..p {
            target[[row, o]] = decoder[[0, o]]
                + decoder[[1, o]] * s
                + decoder[[2, o]] * s * s
                + 0.03 * (1.7 * row as f64 + 0.9 * o as f64).sin();
        }
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "interval".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(width),
    )
    .expect("atom shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Interval { lo: LO, hi: HI }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 1.0, vec![array![-3.0]]);
    (term, target, rho)
}

/// The fixture converged at its seed `ρ`, the state every #4077 half-line check
/// reads (mirrors `converged_interval_state` in `tests_interval_active_bound_3438`).
fn converged_interval_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (mut term, target, rho) = interval_fixture();
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .expect("the criterion prices the interval fixture");
    (term, target, rho)
}

/// The fitted state's `B` system and its evidence factor.
fn fitted_cache(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let system = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("the inner system assembles at the fitted state");
    let options = term.evidence_factor_options();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("the evidence factor holds at the fitted state");
    cache
}

/// The term moved by `scale·d` over the inner coordinates `(t, β)` — one
/// coordinate per row for `t`, the basis-major decoder for `β` — with the basis
/// refreshed at the displaced coordinates, so every operand the half-line mass
/// reads moves with the state.
fn displaced(term: &SaeManifoldTerm, d: &SaeArrowVector, scale: f64) -> SaeManifoldTerm {
    let mut moved = term.clone();
    let coords = moved.assignment.coords[0].as_matrix();
    let coordinate_step =
        Array2::from_shape_vec(coords.dim(), d.t.to_vec()).expect("one coordinate per row");
    moved.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
        (&coords + &(coordinate_step * scale)).view(),
        LatentIdMode::None,
        LatentManifold::Interval { lo: LO, hi: HI },
    );
    let decoder = moved.atoms[0].decoder_coefficients().clone();
    let decoder_step =
        Array2::from_shape_vec(decoder.dim(), d.beta.to_vec()).expect("basis-major decoder");
    moved.atoms[0]
        .set_decoder_coefficients(&decoder + &(decoder_step * scale))
        .expect("the decoder keeps its shape");
    moved
        .refresh_basis_from_current_coords()
        .expect("the basis refreshes at the displaced coordinates");
    moved
}

/// The pinned slots' `Σ_u c_u` in full `log|A|` units, at the term's state.
fn half_line_mass(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    cache: &ArrowFactorCache,
) -> f64 {
    term.pinned_half_line_slots(rho, target, cache)
        .expect("half-line operands")
        .iter()
        .map(|slot| slot.log_det_correction())
        .sum()
}

/// #4077 — the θ-adjoint channel is the inner-coordinate derivative of the
/// half-line mass, in FULL `log|A|` units, with the gradient leg the raw
/// curvature rows and the σ-leg the weight-adjoint.
#[test]
fn half_line_theta_adjoint_is_the_inner_coordinate_derivative_4077() {
    let (mut term, target, rho) = converged_interval_state();
    let cache = fitted_cache(&mut term, &target, &rho);
    let slots = term
        .pinned_half_line_slots(&rho, target.view(), &cache)
        .expect("half-line operands");
    assert!(!slots.is_empty(), "no slot is pinned at the fitted state");
    let Some((_, gamma)) = term
        .pinned_half_line_channels(&rho, target.view(), &cache)
        .expect("the channels price at the fitted state")
    else {
        panic!("the fitted state pins slots, so the channels must be Some");
    };

    // A deterministic direction over `(t, β)`, ZERO on the pinned slots: the
    // production IFT response is projected to zero there, so the adjoint's
    // components on those coordinates are never contracted against motion.
    let pinned_rows: Vec<usize> = term
        .last_pinned_bound_slots
        .iter()
        .map(|&(row, _)| row)
        .collect();
    let mut d = SaeArrowVector {
        t: Array1::from_shape_fn(term.n_obs(), |row| {
            (pinned_rows.contains(&row))
                .then(|| 0.0)
                .unwrap_or(0.7 - 0.11 * row as f64)
        }),
        beta: Array1::from_shape_fn(cache.k, |j| 0.31 - 0.07 * j as f64),
    };
    // Normalise so `‖d‖_∞ = 1`: the step sizes below are in state units.
    let direction_scale =
        d.t.iter()
            .chain(d.beta.iter())
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    d.t.mapv_inplace(|value| value / direction_scale);
    d.beta.mapv_inplace(|value| value / direction_scale);

    let mass =
        |scale: f64| half_line_mass(&displaced(&term, &d, scale), &rho, target.view(), &cache);
    let operand = |scale: f64| {
        displaced(&term, &d, scale)
            .pinned_half_line_slots(&rho, target.view(), &cache)
            .expect("half-line operands at the displaced state")
    };
    let richardson = |coarse: f64, fine: f64| ((4.0 * fine - coarse) / 3.0, (coarse - fine).abs());
    let fd = |eps: f64, probe: &dyn Fn(f64) -> f64| (probe(eps) - probe(-eps)) / (2.0 * eps);
    let step = 1.0e-4_f64;
    let (mass_fd, mass_spread) = richardson(fd(step, &mass), fd(0.5 * step, &mass));
    let gamma_dot_d = gamma.t.dot(&d.t) + gamma.beta.dot(&d.beta);

    // 1. The vector claim: the FULL-units derivative of `Σ_u c_u` along `d`.
    println!(
        "[#4077 θ-adjoint] ⟨gamma, d⟩ = {gamma_dot_d:.12e}, FD of Σc_u = {mass_fd:.12e} \
         (spread {mass_spread:.3e})"
    );
    let mass_tolerance = 10.0 * mass_spread + 1.0e-6 * mass_fd.abs().max(1.0);
    assert!(
        (gamma_dot_d - mass_fd).abs() <= mass_tolerance,
        "the θ-adjoint {gamma_dot_d} is not the full-units inner derivative of the \
         half-line mass {mass_fd} (|Δ| = {:.3e}, tolerance {mass_tolerance:.3e})",
        (gamma_dot_d - mass_fd).abs()
    );

    // 2. The per-slot gradient-leg Jacobian at the GENUINELY pinned slots, and
    //    the σ-leg by subtraction (leg 3).
    let plus = operand(step);
    let minus = operand(-step);
    let mut gradient_leg = 0.0_f64;
    let mut sigma_leg_fd = 0.0_f64;
    for (position, slot) in slots.iter().enumerate() {
        let dg_fd = (plus[position].gradient - minus[position].gradient) / (2.0 * step);
        let d_sigma_fd = (plus[position].curvature - minus[position].curvature) / (2.0 * step);
        let border_dot: f64 = slot
            .border_curvature
            .iter()
            .zip(d.beta.iter())
            .map(|(&(_, value), &step_value)| value * step_value)
            .sum();
        // `row_curvature` is ROW-LOCAL (`q` entries); the joint offset is the
        // row's block start, exactly where the channels scatter it
        // (`theta.t[base + other]`).
        let base = cache.row_offsets[slot.row];
        let q = slot.row_curvature.len();
        let dg_analytic = slot.row_curvature.dot(&d.t.slice(s![base..base + q])) + border_dot;
        println!(
            "[#4077 θ-adjoint] slot {}: dg FD {dg_fd:.12e} vs rows {dg_analytic:.12e}; \
             dσ FD {d_sigma_fd:.12e} (κ₂+κ₁² = {:.12e})",
            slot.index,
            slot.curvature_coefficient()
        );
        let gradient_tolerance = 1.0e-6 * dg_fd.abs().max(1.0e-3);
        assert!(
            (dg_fd - dg_analytic).abs() <= gradient_tolerance,
            "slot {}: the raw curvature rows are not the Jacobian of g_u: FD {dg_fd:.12e} \
             vs analytic {dg_analytic:.12e}",
            slot.index
        );
        gradient_leg += slot.gradient_coefficient() * dg_fd;
        sigma_leg_fd += slot.curvature_coefficient() * d_sigma_fd;
    }

    // 3. The σ-leg by subtraction: what the weight-adjoint must have contributed.
    let sigma_leg_analytic = gamma_dot_d - gradient_leg;
    println!(
        "[#4077 θ-adjoint] σ-leg by subtraction {sigma_leg_analytic:.12e} vs FD \
         {sigma_leg_fd:.12e}; gradient leg {gradient_leg:.12e}"
    );
    let sigma_tolerance = 10.0 * mass_spread + 1.0e-6 * sigma_leg_fd.abs().max(1.0e-3);
    assert!(
        (sigma_leg_analytic - sigma_leg_fd).abs() <= sigma_tolerance,
        "the weight-adjoint σ-leg {sigma_leg_analytic} is not Σ(κ₂+κ₁²)·dσ = {sigma_leg_fd} \
         (|Δ| = {:.3e})",
        (sigma_leg_analytic - sigma_leg_fd).abs()
    );
}

/// #2231/#4077 — with crosscoder pricing spans installed, the span leg's `1.5`
/// probe reads `∂σ_u/∂log λ` exactly (affine target, `c = √λ`), and the block's
/// implicit right-hand side carries `∂g_u/∂log λ` on the pinned slots.
#[test]
fn crosscoder_span_leg_prices_exact_sigma_and_gradient_derivatives_4077() {
    let (mut term, target, mut rho) = converged_interval_state();
    let p = target.ncols();
    let cache = fitted_cache(&mut term, &target, &rho);
    let base = term
        .pinned_half_line_slots(&rho, target.view(), &cache)
        .expect("half-line operands");
    assert!(!base.is_empty(), "no slot is pinned at the fitted state");

    // One crosscoder block over every output column, weight `λ = 1`
    // (`log λ = 0`), exactly as the outer objective installs pricing spans.
    rho.log_lambda_block = vec![0.0];
    term.crosscoder_pricing_spans = Some((0, vec![p]));
    let block = rho.block_flat_range().start;
    let Some((trace, _)) = term
        .pinned_half_line_channels(&rho, target.view(), &cache)
        .expect("the channels price with spans installed")
    else {
        panic!("the fitted state pins slots, so the channels must be Some");
    };

    // The block's target columns scaled by `factor`, everything else fixed.
    let scaled = |factor: f64| {
        let mut shifted = target.clone();
        shifted
            .slice_mut(s![.., 0..p])
            .mapv_inplace(|value| factor * value);
        term.pinned_half_line_slots(&rho, shifted.view(), &cache)
            .expect("half-line operands at the scaled target")
    };
    // `c = 1 ± δ/2` moves `log λ` by `±δ` (c = √λ), so the central difference
    // over `2δ` IS `∂/∂log λ` — with the affineness making it exact to rounding.
    let delta = 1.0e-6_f64;
    let plus = scaled(1.0 + delta / 2.0);
    let minus = scaled(1.0 - delta / 2.0);
    let moved = scaled(1.5);

    let mut probe_gap = 0.0_f64;
    let mut sigma_leg = 0.0_f64;
    let mut gradient_leg_fd = 0.0_f64;
    for (position, slot) in base.iter().enumerate() {
        let d_sigma_fd = (plus[position].curvature - minus[position].curvature) / (2.0 * delta);
        let d_sigma_probe = moved[position].curvature - slot.curvature;
        println!(
            "[#4077 span] slot {}: ∂σ/∂log λ probe {d_sigma_probe:.12e} vs FD {d_sigma_fd:.12e}",
            slot.index
        );
        let sigma_scale = d_sigma_fd.abs().max(1.0e-6);
        assert!(
            (d_sigma_probe - d_sigma_fd).abs() <= 1.0e-7 * sigma_scale,
            "slot {}: the 1.5 probe {d_sigma_probe} is not ∂σ/∂log λ = {d_sigma_fd}",
            slot.index
        );
        probe_gap = probe_gap.max((d_sigma_probe - d_sigma_fd).abs());
        sigma_leg += 0.5 * slot.curvature_coefficient() * d_sigma_fd;

        let d_gradient_fd = (plus[position].gradient - minus[position].gradient) / (2.0 * delta);
        gradient_leg_fd += 0.5 * slot.gradient_coefficient() * d_gradient_fd;
    }

    // The trace's block entry is the criterion-unit σ-leg (and nothing else:
    // no prior operator owns a block coordinate).
    println!(
        "[#4077 span] trace[block] = {:.12e}, σ-leg FD {sigma_leg:.12e} (max probe gap \
         {probe_gap:.3e}); gradient leg FD {gradient_leg_fd:.12e}",
        trace[block]
    );
    let trace_tolerance = 1.0e-7 * sigma_leg.abs().max(1.0e-6);
    assert!(
        (trace[block] - sigma_leg).abs() <= trace_tolerance,
        "trace[block] {} is not the span σ-leg {} (|Δ| = {:.3e})",
        trace[block],
        sigma_leg,
        (trace[block] - sigma_leg).abs()
    );

    // The block's implicit right-hand side carries `∂g_u/∂log λ` on the pinned
    // slots — the entries the outer gradient's half-line leg multiplies.
    let rhs = term
        .crosscoder_block_ift_rhs(&cache, target.view(), 0..p)
        .expect("the block implicit right-hand side");
    for (position, slot) in base.iter().enumerate() {
        let d_gradient_fd = (plus[position].gradient - minus[position].gradient) / (2.0 * delta);
        let rhs_entry = rhs.t[slot.index];
        println!(
            "[#4077 span] slot {}: rhs ∂g/∂log λ {rhs_entry:.12e} vs FD {d_gradient_fd:.12e}",
            slot.index
        );
        let rhs_tolerance = 1.0e-6 * d_gradient_fd.abs().max(1.0e-3);
        assert!(
            (rhs_entry - d_gradient_fd).abs() <= rhs_tolerance,
            "slot {}: the block rhs entry {rhs_entry} is not ∂g_u/∂log λ = {d_gradient_fd}",
            slot.index
        );
    }
}
