//! #2627 — the Steihaug-CG stop reads the attainable accuracy of its recursive
//! residual, never an absolute threshold.
//!
//! `residual_gap` bounds `‖b − A x̂_k − r̂_k‖` by `D_k`, built from the operator's
//! declared rounding and norms the loop holds, so the bound moves with the system.
//!
//! * A binary rescale of `A` and `b` makes every operation of the loop exact under
//!   scaling, so the stop reason, the iteration count, both reported ratios and the
//!   scaled solution must reproduce to the bit. The absolute `1e-14` backstop this
//!   replaced fails it: at `b·2^-20` the requested threshold falls below `1e-14` and the
//!   backstop stops early, and at `b·2^-50` the right-hand side itself is below `1e-14`
//!   and the backstop returned the zero step as converged.
//! * A relative tolerance of one unit roundoff is unattainable, because
//!   `D_1 ≥ u·(|α_0|·‖Âp_0‖ + ‖r̂_1‖) ≥ u·‖b‖`. The loop stops at the floor, and the
//!   explicitly evaluated residual respects `2·D_k` plus its own evaluation band. The
//!   same system at `1e-4` meets its tolerance, so each arm fires for its own reason.
//! * A tolerance the zero step already meets reports the zero step's true relative
//!   residual, `1.0`, where the backstop's initial exit reported `0.0`.
//! * The matrix-free reduced border declares a norm bound its operator respects, and a
//!   declaration without the row coupling undercuts it; the same operator's PCG meets a
//!   reachable tolerance and stops at the floor on an unattainable one.

#![cfg(test)]

use super::*;
use crate::arrow_schur::reduced_solve::{
    IdentityPreconditioner, ReducedSchurOperator, dense_matvec, euclidean_norm,
    run_pcg_with_preconditioner, steihaug_dense_system, symmetrize_upper_from_lower,
};
use crate::arrow_schur::residual_gap::MatvecRoundingBound;
use crate::arrow_schur::solve_options::CpuBatchedBlockSolver;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::roundoff::{UNIT_ROUNDOFF, symmetric_spectrum_rounding_band};

/// `A = H·diag(λ)·H` with `H = I − 2vvᵀ/‖v‖²` a Householder reflector and `λ` geometric
/// from `1` to `1/condition`. The reflector spreads every eigenvector over all
/// coordinates, so each matvec row accumulates across the whole spectrum, and the
/// geometric grading makes the residual fall over many iterations rather than a few.
fn graded_system(n: usize, condition: f64, rhs_scale: f64) -> (Array2<f64>, Array1<f64>) {
    let v = Array1::from_shape_fn(n, |i| ((i as f64 + 1.0) * 0.7).sin() + 0.1);
    let vv = v.dot(&v);
    let reflector = Array2::from_shape_fn((n, n), |(i, j)| {
        let identity = if i == j { 1.0 } else { 0.0 };
        identity - 2.0 * v[i] * v[j] / vv
    });
    let spectrum =
        Array1::from_shape_fn(n, |i| condition.powf(-(i as f64) / ((n - 1) as f64)));
    let weighted = Array2::from_shape_fn((n, n), |(i, j)| reflector[[i, j]] * spectrum[j]);
    let mut a = weighted.dot(&reflector);
    symmetrize_upper_from_lower(&mut a);
    let b = Array1::from_shape_fn(n, |i| rhs_scale * ((i as f64 + 1.0) * 0.37).cos());
    (a, b)
}

/// An unbounded Steihaug solve with the identity preconditioner. `max_iterations` is a
/// harness bound, set far above what the pins need, and each pin asserts it did not bind.
fn steihaug(
    a: &Array2<f64>,
    b: &Array1<f64>,
    relative_tolerance: f64,
    max_iterations: usize,
) -> (Array1<f64>, ArrowPcgDiagnostics) {
    steihaug_dense_system(
        a,
        b,
        &IdentityPreconditioner,
        &ArrowPcgOptions {
            max_iterations,
            relative_tolerance,
        },
        &ArrowTrustRegionOptions::default(),
    )
    .expect("an SPD system never refuses an unbounded Steihaug solve")
}

/// `‖b − A x̂‖` evaluated in floating point, and the band that evaluation carries: the
/// matvec's declared band `ν_A·‖x̂‖` plus one rounded subtraction `u·(‖b‖ + ‖fl(A x̂)‖)`.
fn explicit_residual(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x: &Array1<f64>,
    bound: &MatvecRoundingBound,
) -> (f64, f64) {
    let mut ax = Array1::<f64>::zeros(b.len());
    dense_matvec(a, x, &mut ax);
    let residual = b - &ax;
    let band = bound.apply_band * euclidean_norm(x.view())
        + UNIT_ROUNDOFF * (euclidean_norm(b.view()) + euclidean_norm(ax.view()));
    (euclidean_norm(residual.view()), band)
}

#[test]
fn a_binary_rescale_reproduces_the_stop_to_the_bit_2627() {
    let n = 32;
    let (a, b) = graded_system(n, 1.0e3, 1.0e-6);
    let relative_tolerance = 1.0e-4;
    let max_iterations = 64 * n;
    let (x, diag) = steihaug(&a, &b, relative_tolerance, max_iterations);
    assert_eq!(
        diag.stopping_reason,
        PcgStopReason::RelativeToleranceMet,
        "the base system must meet its tolerance (iterations {}, relative residual {:e}, gap {:e})",
        diag.iterations,
        diag.final_relative_residual,
        diag.relative_residual_gap_bound
    );
    assert!(diag.iterations < max_iterations, "the harness bound must not bind");

    let unit = 2.0_f64.powi(20);
    let b_norm = euclidean_norm(b.view());
    // Premises of the mutant arms: at `b·2^-20` the requested threshold is below the
    // retired `1e-14` backstop, and at `b·2^-50` so is the right-hand side itself.
    assert!(relative_tolerance * b_norm / unit < 1.0e-14);
    assert!(b_norm * 2.0_f64.powi(-50) < 1.0e-14);
    let scales = [
        (unit, 1.0 / unit),
        (1.0 / unit, unit),
        (unit, unit),
        (1.0 / unit, 1.0 / unit),
        (1.0, 2.0_f64.powi(-50)),
    ];
    for (s, t) in scales {
        let (xs, ds) = steihaug(
            &a.mapv(|value| value * s),
            &b.mapv(|value| value * t),
            relative_tolerance,
            max_iterations,
        );
        assert_eq!(
            ds.stopping_reason, diag.stopping_reason,
            "rescale A·{s:e}, b·{t:e} changed the stop reason"
        );
        assert_eq!(
            ds.iterations, diag.iterations,
            "rescale A·{s:e}, b·{t:e} changed the iteration count"
        );
        assert_eq!(
            ds.final_relative_residual.to_bits(),
            diag.final_relative_residual.to_bits(),
            "rescale A·{s:e}, b·{t:e} changed the relative residual: {:e} vs {:e}",
            ds.final_relative_residual,
            diag.final_relative_residual
        );
        assert_eq!(
            ds.relative_residual_gap_bound.to_bits(),
            diag.relative_residual_gap_bound.to_bits(),
            "rescale A·{s:e}, b·{t:e} changed the relative gap bound: {:e} vs {:e}",
            ds.relative_residual_gap_bound,
            diag.relative_residual_gap_bound
        );
        for (index, (scaled, base)) in xs.iter().zip(x.iter()).enumerate() {
            assert_eq!(
                scaled.to_bits(),
                (base * (t / s)).to_bits(),
                "rescale A·{s:e}, b·{t:e} changed solution coordinate {index}: {scaled:e} vs {:e}",
                base * (t / s)
            );
        }
    }
}

#[test]
fn an_unattainable_relative_tolerance_stops_at_the_floor_2627() {
    let n = 48;
    let (a, b) = graded_system(n, 1.0e6, 1.0);
    let max_iterations = 64 * n;
    let b_norm = euclidean_norm(b.view());
    let bound = MatvecRoundingBound::dense(a.view());

    let (x_met, diag_met) = steihaug(&a, &b, 1.0e-4, max_iterations);
    assert_eq!(
        diag_met.stopping_reason,
        PcgStopReason::RelativeToleranceMet,
        "a reachable tolerance must be met (iterations {}, relative residual {:e}, gap {:e})",
        diag_met.iterations,
        diag_met.final_relative_residual,
        diag_met.relative_residual_gap_bound
    );
    let (true_met, band_met) = explicit_residual(&a, &b, &x_met, &bound);
    assert!(
        true_met <= 1.0e-4 * b_norm + band_met,
        "the met tolerance must hold for the explicit residual: {true_met:e} vs {:e} + {band_met:e}",
        1.0e-4 * b_norm
    );

    let (x_floor, diag_floor) = steihaug(&a, &b, UNIT_ROUNDOFF, max_iterations);
    assert_eq!(
        diag_floor.stopping_reason,
        PcgStopReason::AttainableFloorReached,
        "one unit roundoff is unattainable (iterations {}, relative residual {:e}, gap {:e})",
        diag_floor.iterations,
        diag_floor.final_relative_residual,
        diag_floor.relative_residual_gap_bound
    );
    assert!(
        diag_floor.iterations < max_iterations,
        "the floor, not the harness bound, must stop the loop"
    );
    assert!(
        diag_floor.relative_residual_gap_bound > UNIT_ROUNDOFF,
        "the floor must exceed the requested tolerance: {:e}",
        diag_floor.relative_residual_gap_bound
    );
    let (true_floor, band_floor) = explicit_residual(&a, &b, &x_floor, &bound);
    assert!(
        true_floor <= 2.0 * diag_floor.relative_residual_gap_bound * b_norm + band_floor,
        "the explicit residual must respect twice the gap bound: {true_floor:e} vs 2·{:e} + {band_floor:e}",
        diag_floor.relative_residual_gap_bound * b_norm
    );
}

#[test]
fn a_tolerance_the_zero_step_meets_reports_its_true_residual_2627() {
    let n = 16;
    let (a, b) = graded_system(n, 1.0e2, 1.0);
    let (x, diag) = steihaug(&a, &b, 1.0, 64 * n);
    assert_eq!(diag.stopping_reason, PcgStopReason::RelativeToleranceMet);
    assert_eq!(diag.iterations, 0);
    assert_eq!(
        diag.final_relative_residual, 1.0,
        "the zero step leaves the whole right-hand side as its residual"
    );
    assert_eq!(diag.relative_residual_gap_bound, 0.0);
    assert!(x.iter().all(|&value| value == 0.0));
}

/// `n` scalar latent rows and a border of `k`: `H_tt^(i) = 0.5 + 0.1·i`, a dense cross block
/// of scale `coupling`, and `H_ββ = border_diagonal·I`. The factors are the scalar square
/// roots of the row blocks.
fn scalar_row_border(
    n: usize,
    k: usize,
    coupling: f64,
    border_diagonal: f64,
) -> (ArrowSchurSystem, ArrowFactorSlab) {
    let mut sys = ArrowSchurSystem::new(n, 1, k);
    for (row_index, row) in sys.rows.iter_mut().enumerate() {
        row.htt[[0, 0]] = 0.5 + 0.1 * row_index as f64;
        for c in 0..k {
            row.htbeta[[0, c]] = coupling * (((row_index * k + c) as f64) * 0.73).sin();
        }
    }
    for a in 0..k {
        sys.hbb[[a, a]] = border_diagonal;
    }
    let factors = ArrowFactorSlab::from_blocks(
        sys.rows
            .iter()
            .map(|row| Array2::from_elem((1, 1), row.htt[[0, 0]].sqrt()))
            .collect(),
    );
    (sys, factors)
}

#[test]
fn the_matrix_free_border_declares_a_covering_norm_and_stops_at_its_floor_2627() {
    let backend = CpuBatchedBlockSolver;
    let k = 5;

    // A coupling-dominated border: `Σ H_βt A⁻¹ H_tβ` swamps `H_ββ = 0.1·I`.
    let (coupled, coupled_factors) = scalar_row_border(6, k, 2.0, 0.1);
    let operator = ReducedSchurOperator::new(&coupled, &coupled_factors, 0.0, &backend, None);
    let mut dense = Array2::<f64>::zeros((k, k));
    let mut unit = Array1::<f64>::zeros(k);
    let mut column = Array1::<f64>::zeros(k);
    for j in 0..k {
        unit.fill(0.0);
        unit[j] = 1.0;
        operator.apply_into(&unit, &mut column);
        dense.column_mut(j).assign(&column);
    }
    symmetrize_upper_from_lower(&mut dense);
    let eigenvalues = dense.eigh(Side::Lower).expect("reduced border EVD").0;
    let reference = eigenvalues.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let band = symmetric_spectrum_rounding_band(eigenvalues.as_slice().expect("contiguous"));
    let declared = MatvecRoundingBound::reduced_border(&coupled, &coupled_factors, 0.0)
        .expect("the dense cross block declares its bounds");
    assert!(
        declared.norm_upper >= reference - band,
        "the declared {:e} must cover ‖S‖₂ = {reference:e} (band {band:e})",
        declared.norm_upper
    );
    assert!(
        0.1 < reference - band,
        "mutant: the border alone, N_β + ρ_β = 0.1, must undercut ‖S‖₂ = {reference:e} on a \
         coupling-dominated border"
    );
    assert!(
        declared.apply_band > 0.0 && declared.apply_band < declared.norm_upper,
        "the apply band {:e} must be a positive fraction of the norm bound {:e}",
        declared.apply_band,
        declared.norm_upper
    );

    // A positive definite border, solved through the matrix-free route.
    let (definite, definite_factors) = scalar_row_border(6, k, 0.5, 500.0);
    let rhs = Array1::from_shape_fn(k, |a| ((a as f64) + 1.0).cos());
    let max_iterations = 64 * k;
    let solve = |relative_tolerance: f64| {
        run_pcg_with_preconditioner(
            &definite,
            &definite_factors,
            0.0,
            &rhs,
            |r| r.clone(),
            &ArrowPcgOptions {
                max_iterations,
                relative_tolerance,
            },
            &ArrowTrustRegionOptions {
                max_iterations,
                steihaug_relative_tolerance: relative_tolerance,
                ..ArrowTrustRegionOptions::default()
            },
            &backend,
            None,
            None,
        )
        .expect("a positive definite border never refuses an unbounded solve")
    };
    let (_met_step, met) = solve(1.0e-4);
    assert_eq!(
        met.stopping_reason,
        PcgStopReason::RelativeToleranceMet,
        "a reachable tolerance must be met (iterations {}, relative residual {:e}, gap {:e})",
        met.iterations,
        met.final_relative_residual,
        met.relative_residual_gap_bound
    );
    let (_floor_step, floor) = solve(UNIT_ROUNDOFF);
    assert_eq!(
        floor.stopping_reason,
        PcgStopReason::AttainableFloorReached,
        "one unit roundoff is unattainable (iterations {}, relative residual {:e}, gap {:e})",
        floor.iterations,
        floor.final_relative_residual,
        floor.relative_residual_gap_bound
    );
    assert!(
        floor.iterations < max_iterations && floor.relative_residual_gap_bound > UNIT_ROUNDOFF,
        "the floor, not the harness bound, must stop the loop: iterations {}, gap {:e}",
        floor.iterations,
        floor.relative_residual_gap_bound
    );
}
