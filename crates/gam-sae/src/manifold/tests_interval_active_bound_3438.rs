#![cfg(test)]
//! #3438 — an Interval atom whose fitted coordinates pile up at the upper bound.
//!
//! At an active bound (`t ≥ hi` with `g < 0`) `B`'s Riemannian conversion projects
//! the slot out: zero gradient, zero `H_tt` row and column, zero `H_tβ` row. The
//! slot is a flat direction of `B`, and the retraction holds the coordinate on the
//! bound for every nearby `(ρ, β)`, so its mode response is zero. `ΔC` must enter
//! `A = B + ΔC` with the same slot projected out. Otherwise `A` carries the slot's
//! raw curvature on its diagonal and a `ΔC_uβ` coupling, the IFT moves a coordinate
//! the retraction holds fixed, and `½log|A|` prices a direction the mode cannot take.

use super::construction::ExactHessianDeltaRow;
use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_terms::latent::LatentManifold;
use ndarray::{Array2, array};
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

/// Largest `|ΔC|` entry on the pinned slots: the `tt` row and column and the
/// `tβ` row.
fn pinned_delta_c_magnitude(rows: &[ExactHessianDeltaRow], pinned: &[(usize, usize)]) -> f64 {
    let mut magnitude = 0.0_f64;
    for &(row, local) in pinned {
        let block = &rows[row];
        for value in block
            .tt
            .row(local)
            .iter()
            .chain(block.tt.column(local).iter())
            .chain(block.tbeta.row(local).iter())
        {
            magnitude = magnitude.max(value.abs());
        }
    }
    magnitude
}

/// Largest off-diagonal `|A|` entry on the pinned rows of the dense `A`, and the
/// pinned diagonal entries.
fn pinned_a_coupling(
    a: &Array2<f64>,
    cache: &ArrowFactorCache,
    pinned: &[(usize, usize)],
) -> (f64, Vec<f64>) {
    let mut coupling = 0.0_f64;
    let mut diagonal = Vec::with_capacity(pinned.len());
    for &(row, local) in pinned {
        let index = cache.row_offsets[row] + local;
        for column in 0..a.ncols() {
            if column == index {
                continue;
            }
            coupling = coupling
                .max(a[[index, column]].abs())
                .max(a[[column, index]].abs());
        }
        diagonal.push(a[[index, index]]);
    }
    (coupling, diagonal)
}

fn pinned_step_components(
    step: &SaeArrowVector,
    cache: &ArrowFactorCache,
    pinned: &[(usize, usize)],
) -> Vec<f64> {
    pinned
        .iter()
        .map(|&(row, local)| step.t[cache.row_offsets[row] + local])
        .collect()
}

fn arrow_max_abs(vector: &SaeArrowVector) -> f64 {
    vector
        .t
        .iter()
        .chain(vector.beta.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

#[test]
fn interval_active_bound_slot_leaves_the_exact_information_3438() {
    let (mut term, target, rho) = interval_fixture();
    let (cost, ..) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            60,
            0.4,
            1.0e-8,
            1.0e-8,
        )
        .expect("the criterion prices the interval fixture");
    let coords = term.assignment.coords[0].as_matrix().column(0).to_vec();
    println!("[#3438] cost={cost:.12e} coords={coords:?}");
    assert!(
        cost.is_finite(),
        "the criterion is finite at the bound: {cost}"
    );

    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the inner system assembles at the fitted state");
    let pinned = term.last_pinned_bound_slots.clone();
    println!("[#3438] pinned (row, slot) = {pinned:?}");
    // Premise: the fixture drives coordinates onto the bound with the descent
    // direction leaving the interval, and `B` holds each such slot flat.
    assert!(
        !pinned.is_empty(),
        "no interval coordinate sits at an active bound; coords {coords:?}"
    );
    for &(row, local) in &pinned {
        let vars = term
            .row_vars_for_row_dim(row, system.rows[row].gt.len())
            .expect("row variables");
        let SaeLocalRowVar::Coord { atom, axis } = vars[local] else {
            panic!(
                "pinned slot (row {row}, slot {local}) is not a coordinate: {:?}",
                vars[local]
            );
        };
        let t = term.assignment.coords[atom].as_matrix()[[row, axis]];
        assert!(
            t <= LO || t >= HI,
            "pinned coordinate (row {row}) = {t} lies inside the interval"
        );
        let block = &system.rows[row];
        assert_eq!(block.gt[local], 0.0, "B's gradient keeps the pinned slot");
        assert!(
            block
                .htt
                .row(local)
                .iter()
                .chain(block.htt.column(local).iter())
                .all(|v| *v == 0.0),
            "B's H_tt keeps the pinned slot (row {row}): {:?}",
            block.htt
        );
    }

    let options = term.evidence_factor_options();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("the evidence factor holds at the fitted state");

    // The unprojected ΔC: what `A = B + ΔC` carried before #3438. The pinned list
    // is the only thing the unprojected state lacks.
    let mut unprojected = term.clone();
    unprojected.last_pinned_bound_slots.clear();
    let raw_rows = unprojected
        .assemble_exact_hessian_minus_b_rows(&rho, target.view(), &cache.row_dims, cache.k)
        .expect("unprojected ΔC rows");
    let raw_magnitude = pinned_delta_c_magnitude(&raw_rows, &pinned);
    let rows = term
        .assemble_exact_hessian_minus_b_rows(&rho, target.view(), &cache.row_dims, cache.k)
        .expect("projected ΔC rows");
    let projected_magnitude = pinned_delta_c_magnitude(&rows, &pinned);
    println!(
        "[#3438] max |ΔC| on pinned slots: unprojected {raw_magnitude:.6e}, projected \
         {projected_magnitude:.6e}"
    );
    // Premise: the likelihood and prior curvature of the pinned slot is live, so
    // the fixture measures the projection and not an accidentally flat slot.
    assert!(
        raw_magnitude > 0.0,
        "the unprojected ΔC carries no curvature on the pinned slots"
    );
    // The projection writes an exact zero; nothing is added after it.
    assert_eq!(
        projected_magnitude, 0.0,
        "ΔC keeps curvature on a slot B projected out"
    );

    let (raw_a, _) = unprojected
        .materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache)
        .expect("unprojected dense A");
    let (a, _) = term
        .materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache)
        .expect("dense A");
    let (raw_coupling, raw_diagonal) = pinned_a_coupling(&raw_a, &cache, &pinned);
    let (coupling, diagonal) = pinned_a_coupling(&a, &cache, &pinned);
    let a_scale = a.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    println!(
        "[#3438] dense A pinned rows: unprojected coupling {raw_coupling:.6e} diagonal \
         {raw_diagonal:?}; projected coupling {coupling:.6e} diagonal {diagonal:?}; max|A| \
         {a_scale:.6e}"
    );
    // With `ΔC` projected, the pinned row of `A` is the evidence row deflation
    // alone: the flat slot carried at the metric's unit stiffness and coupled to
    // nothing, so the pencil `(A, Φ)` prices it at `log 1 = 0`. What remains is
    // the rounding of recovering `B_raw` from the conditioned factor,
    // `ε·max|A|` per entry.
    let rounding = f64::EPSILON * a_scale.max(1.0);
    assert!(
        coupling <= rounding,
        "A couples the pinned slot to the rest of θ: {coupling:.3e} (rounding {rounding:.3e})"
    );
    for value in &diagonal {
        assert!(
            (value - 1.0).abs() <= rounding,
            "A prices curvature on the pinned slot beyond the metric's unit stiffness: \
             {value:.6e} (rounding {rounding:.3e})"
        );
    }

    let log_det = term
        .exact_observed_information_log_dets(&rho, target.view(), &cache)
        .expect("½log|A| prices the fitted state");
    let raw_log_det = unprojected.exact_observed_information_log_dets(&rho, target.view(), &cache);
    println!("[#3438] log|A|: unprojected {raw_log_det:?}, projected {log_det:.12e}");
    assert!(log_det.is_finite(), "log|A| = {log_det}");

    // IFT: the mode response to the ARD log-precision, `θ̂ = −A⁺ g_ρ`. The
    // retraction holds a pinned coordinate on its bound, so its response is zero.
    let flat = rho.ard_flat_index(0, 0);
    let g_rho = term
        .outer_rho_gradient_ift_rhs(&rho, flat, &cache)
        .expect("ARD implicit right-hand side");
    let step = term
        .solve_exact_stationarity(&rho, target.view(), &cache, &g_rho)
        .expect("A⁺ g_ρ");
    let components = pinned_step_components(&step, &cache, &pinned);
    let step_scale = arrow_max_abs(&step);
    let raw_components = unprojected
        .solve_exact_stationarity(&rho, target.view(), &cache, &g_rho)
        .map(|raw| pinned_step_components(&raw, &cache, &pinned));
    println!(
        "[#3438] A⁺g_ρ on pinned slots: unprojected {raw_components:?}, projected \
         {components:?}; max|A⁺g_ρ| {step_scale:.6e}"
    );
    // The pinned slot is an exact null of the pencil `(A, Φ)` with `Φ` holding it at
    // unit stiffness and uncoupled, so it lies in the band and every retained mode is
    // `Φ`-orthogonal to it. The pencil resolves modes to its floor `√ε`, which bounds
    // a retained mode's leakage onto the slot relative to the step.
    let leakage = f64::EPSILON.sqrt() * step_scale.max(f64::MIN_POSITIVE);
    for value in &components {
        assert!(
            value.abs() <= leakage,
            "the IFT moves a coordinate the bound holds fixed: {value:.6e} (bound {leakage:.3e})"
        );
    }
}
