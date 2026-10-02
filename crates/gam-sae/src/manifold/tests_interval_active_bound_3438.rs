#![cfg(test)]
//! #3438 — an Interval atom whose fitted coordinates pile up at the upper bound.
//!
//! At an active bound (`t ≥ hi` with `g < 0`) `B`'s Riemannian conversion projects
//! the slot out: zero gradient, zero `H_tβ` row, and an `H_tt` row and column that
//! are the complementary projector's unit direction (#4077, as a sphere's normal).
//! The slot carries no curvature of `B`, and the retraction holds the coordinate on the
//! bound for every nearby `(ρ, β)`, so its mode response is zero. `ΔC` must enter
//! `A = B + ΔC` with the same slot projected out. Otherwise `A` carries the slot's
//! raw curvature on its diagonal and a `ΔC_uβ` coupling, the IFT moves a coordinate
//! the retraction holds fixed, and `½log|A|` prices a direction the mode cannot take.
//!
//! #4077 — the same slot must leave the DERIVATIVE of `log|A|`. With `A`'s pinned row
//! and column the metric's constant unit direction (asserted below to `ε·max|A|`), that
//! row moves with no outer coordinate at all, so every map of `∂A/∂ρ` and every θ-adjoint
//! contraction has to read a zero there. Before this the per-coordinate ρ maps still
//! carried the raw ARD diagonal at the pinned slot, and with `A⁺` reading `1` on that
//! same slot the direct ρ trace `½⟨A⁺, ∂A/∂ρ⟩` priced a leg the value prices at
//! `log 1 = 0`.

use super::construction::ExactHessianDeltaRow;
use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2, array};
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
        let q = block.htt.nrows();
        assert!(
            (0..q).all(|other| {
                let unit = if other == local { 1.0 } else { 0.0 };
                block.htt[[local, other]] == unit && block.htt[[other, local]] == unit
            }),
            "B's H_tt row of the pinned slot (row {row}) is not the unit direction: {:?}",
            block.htt
        );
        assert!(
            block.htbeta.row(local).iter().all(|v| *v == 0.0),
            "B's H_tβ keeps the pinned slot (row {row})"
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
    let raw_g_rho = term
        .outer_rho_gradient_ift_rhs(&rho, flat, &cache)
        .expect("ARD implicit right-hand side");
    // The outer gradient differentiates the PROJECTED stationarity, whose pinned entry
    // is identically zero near the fitted state.
    let mut g_rho = raw_g_rho.clone();
    term.project_pinned_ift_rhs(&cache, &mut g_rho);
    assert!(
        pinned_step_components(&raw_g_rho, &cache, &pinned).iter().any(|v| *v != 0.0),
        "the raw ARD right-hand side carries nothing on the pinned slots, so the projection \
         is unmeasured here"
    );
    let step = term
        .solve_exact_stationarity(&rho, target.view(), &cache, &g_rho)
        .expect("A⁺ g_ρ");
    let components = pinned_step_components(&step, &cache, &pinned);
    let step_scale = arrow_max_abs(&step);
    let raw_components = unprojected
        .solve_exact_stationarity(&rho, target.view(), &cache, &raw_g_rho)
        .map(|raw| pinned_step_components(&raw, &cache, &pinned));
    println!(
        "[#3438] A⁺g_ρ on pinned slots: unprojected {raw_components:?}, projected \
         {components:?}; max|A⁺g_ρ| {step_scale:.6e}"
    );
    // `A` carries the pinned slot as the uncoupled unit direction and the projected
    // right-hand side is zero there, so the exact response on the slot is zero. The
    // pencil resolves modes to its floor `√ε`, which bounds a retained mode's leakage
    // onto the slot relative to the step.
    let leakage = f64::EPSILON.sqrt() * step_scale.max(f64::MIN_POSITIVE);
    for value in &components {
        assert!(
            value.abs() <= leakage,
            "the IFT moves a coordinate the bound holds fixed: {value:.6e} (bound {leakage:.3e})"
        );
    }
}

/// #4077 — the per-outer-coordinate maps of `∂A/∂ρ` read an exact zero on every slot
/// the assembly pinned at an active bound, and the unprojected assembly does not.
///
/// The existing test above pins the value side: `A`'s pinned row and column are the
/// metric's unit direction to `ε·max|A|`, so `A[u,·] ≡ e_uᵀ` at every `(ρ, θ)`. A
/// constant row has no derivative. The maps are assembled on the RAW slots, though,
/// where the ARD prior still writes `w_row·α` on the pinned coordinate's diagonal, and
/// the pencil's `A⁺` reads `1/1 = 1` there because the slot is retained at the metric's
/// unit stiffness. So the direct trace `½⟨A⁺, ∂A/∂ρ⟩` used to charge `½·w_row·α` per
/// pinned slot to a direction `½log|A|` prices at `log 1 = 0`.
///
/// The unprojected clone is the positive control: the same map, with only the pinned
/// list cleared, must still carry that diagonal, or this test would pass on a fixture
/// whose prior happens to be silent at the bound.
#[test]
fn the_rho_maps_of_the_exact_information_drop_the_pinned_bound_slots_4077() {
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

    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the inner system assembles at the fitted state");
    let pinned = term.last_pinned_bound_slots.clone();
    println!("[#4077] pinned (row, slot) = {pinned:?}");
    assert!(
        !pinned.is_empty(),
        "no interval coordinate sits at an active bound"
    );

    let options = term.evidence_factor_options();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("the evidence factor holds at the fitted state");

    let mut unprojected = term.clone();
    unprojected.last_pinned_bound_slots.clear();
    let raw_maps = unprojected
        .exact_stationarity_penalty_derivatives_by_flat(&rho, &cache)
        .expect("unprojected ∂A/∂ρ maps");
    let maps = term
        .exact_stationarity_penalty_derivatives_by_flat(&rho, &cache)
        .expect("∂A/∂ρ maps");

    let indices: Vec<usize> = pinned
        .iter()
        .map(|&(row, local)| cache.row_offsets[row] + local)
        .collect();

    let mut raw_pinned_magnitude = 0.0_f64;
    let mut projected_pinned_magnitude = 0.0_f64;
    for (flat, da) in &maps {
        let raw = raw_maps
            .get(flat)
            .expect("the projection removes no outer coordinate from the map");
        for &index in &indices {
            for other in 0..da.nrows() {
                raw_pinned_magnitude = raw_pinned_magnitude
                    .max(raw[[index, other]].abs())
                    .max(raw[[other, index]].abs());
                projected_pinned_magnitude = projected_pinned_magnitude
                    .max(da[[index, other]].abs())
                    .max(da[[other, index]].abs());
            }
        }
    }
    println!(
        "[#4077] max |∂A/∂ρ| on the pinned rows and columns over {} outer coordinates: \
         unprojected {raw_pinned_magnitude:.6e}, projected {projected_pinned_magnitude:.6e}",
        maps.len()
    );
    // Premise: the prior's curvature at the pinned coordinate is live, so this fixture
    // measures the projection and not an accidentally silent map.
    assert!(
        raw_pinned_magnitude > 0.0,
        "the unprojected ∂A/∂ρ maps carry nothing on the pinned slots, so the projection \
         is unmeasured here"
    );
    // The projection writes an exact zero; nothing is added after it.
    assert_eq!(
        projected_pinned_magnitude, 0.0,
        "a ∂A/∂ρ map keeps a leg on a slot A carries at the metric's constant unit stiffness"
    );
}

/// The fixture converged at its seed `ρ`, the state every #4077 half-line check reads.
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

/// The inner stationarity residual `(g_t, g_β)` the assembly writes at the term's state.
fn assembled_gradient(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> SaeArrowVector {
    let mut state = term.clone();
    state
        .refresh_basis_from_current_coords()
        .expect("the basis refreshes at the current coordinates");
    let system = state
        .assemble_arrow_schur(target, rho, None)
        .expect("the inner system assembles at the state");
    SaeArrowVector {
        t: Array1::from_iter(system.rows.iter().flat_map(|row| row.gt.iter().copied())),
        beta: system.gb.clone(),
    }
}

/// The single interval atom moved by `scale·step`: one coordinate per row, the decoder
/// basis-major.
fn displaced_interval(term: &SaeManifoldTerm, step: &SaeArrowVector, scale: f64) -> SaeManifoldTerm {
    let mut moved = term.clone();
    let coords = moved.assignment.coords[0].as_matrix();
    let coordinate_step =
        Array2::from_shape_vec(coords.dim(), step.t.to_vec()).expect("one coordinate per row");
    moved.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
        (&coords + &(coordinate_step * scale)).view(),
        LatentIdMode::None,
        LatentManifold::Interval { lo: LO, hi: HI },
    );
    let decoder = moved.atoms[0].decoder_coefficients().clone();
    let decoder_step =
        Array2::from_shape_vec(decoder.dim(), step.beta.to_vec()).expect("basis-major decoder");
    moved.atoms[0]
        .set_decoder_coefficients(&decoder + &(decoder_step * scale))
        .expect("the decoder keeps its shape");
    moved
}

/// The fitted state's `B` system and its evidence factor.
fn fitted_cache(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (ArrowSchurSystem, ArrowFactorCache) {
    let system = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("the inner system assembles at the fitted state");
    let options = term.evidence_factor_options();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("the evidence factor holds at the fitted state");
    (system, cache)
}

/// #4077 — the half-line operands are the unprojected slot's own: at an INTERIOR coordinate,
/// where nothing is projected, the raw gradient is the assembled `g_t` entry and the raw
/// curvature row is the dense exact `A`'s row, entry for entry. Naming the interior slot as
/// pinned exercises the same reader the pinned slots take.
#[test]
fn half_line_operands_are_the_unprojected_gradient_and_exact_curvature_row_4077() {
    let (mut term, target, rho) = converged_interval_state();
    let (system, cache) = fitted_cache(&mut term, &target, &rho);
    let pinned_rows: Vec<usize> =
        term.last_pinned_bound_slots.iter().map(|&(row, _)| row).collect();
    let interior = (0..term.n_obs())
        .find(|row| !pinned_rows.contains(row))
        .expect("the fixture keeps interior rows");
    let (a, _) = term
        .materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache)
        .expect("dense A");
    let mut probe = term.clone();
    probe.last_pinned_bound_slots = vec![(interior, 0)];
    let slots = probe
        .pinned_half_line_slots(&rho, target.view(), &cache)
        .expect("the interior slot's half-line operands");
    assert_eq!(slots.len(), 1);
    let slot = &slots[0];
    let index = cache.row_offsets[interior];
    let scale = a.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
    println!(
        "[#4077] interior row {interior}: g_raw {:.12e} vs assembled {:.12e}; A_raw[u,u] {:.12e} \
         vs dense {:.12e}",
        slot.gradient,
        system.rows[interior].gt[0],
        slot.curvature,
        a[[index, index]]
    );
    let gradient_gap = (slot.gradient - system.rows[interior].gt[0]).abs();
    let gradient_scale = system.rows[interior].gt[0].abs().max(1.0);
    assert!(
        gradient_gap <= 1.0e-12 * gradient_scale,
        "the raw slot gradient is not the assembled one: gap {gradient_gap:.3e}"
    );
    for (other, value) in slot.row_curvature.iter().enumerate() {
        let dense = a[[index, cache.row_offsets[interior] + other]];
        assert!(
            (value - dense).abs() <= 1.0e-10 * scale,
            "row block entry {other}: {value:.12e} vs dense A {dense:.12e}"
        );
    }
    let total_t = cache.delta_t_len();
    assert_eq!(slot.border_curvature.len(), cache.k);
    for &(column, value) in &slot.border_curvature {
        let dense = a[[index, total_t + column]];
        assert!(
            (value - dense).abs() <= 1.0e-10 * scale,
            "border entry {column}: {value:.12e} vs dense A {dense:.12e}"
        );
    }
}

/// #4077 — every pinned slot adds its half-line mass `−2·log J(|g|, σ) + log 2π` to the
/// log-determinant, and as the multiplier vanishes that mass is half the Gaussian one:
/// `log σ + 2·log 2`.
#[test]
fn pinned_slots_carry_their_half_line_mass_4077() {
    let (mut term, target, rho) = converged_interval_state();
    let (_, cache) = fitted_cache(&mut term, &target, &rho);
    let slots = term
        .pinned_half_line_slots(&rho, target.view(), &cache)
        .expect("half-line operands");
    assert!(!slots.is_empty(), "no slot is pinned at the fitted state");
    let mut total = 0.0_f64;
    for slot in &slots {
        println!(
            "[#4077] pinned row {} (joint index {}): g {:.6e} σ {:.6e} c {:.6e}",
            slot.row,
            slot.index,
            slot.gradient,
            slot.curvature,
            slot.log_det_correction()
        );
        // The outward gradient: `g < 0` on `hi`, `g > 0` on `lo`.
        let t = term.assignment.coords[0].as_matrix()[[slot.row, 0]];
        let outward = if t >= HI { slot.gradient < 0.0 } else { t <= LO && slot.gradient > 0.0 };
        assert!(outward, "pinned row {} at t = {t} with g = {}", slot.row, slot.gradient);
        assert!(slot.log_det_correction().is_finite());
        total += slot.log_det_correction();
        // The vanishing-multiplier limit, read off the same jet.
        let limit = gam_math::gaussian_reciprocal::half_line_gaussian_log_jet(0.0, slot.curvature)
            .expect("a positive curvature leaves a half-line mass at μ = 0");
        let limit_correction = -2.0 * limit[0] + std::f64::consts::TAU.ln();
        let expected = slot.curvature.ln() + 2.0 * std::f64::consts::LN_2;
        assert!(
            (limit_correction - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "μ → 0 prices {limit_correction:.15e}, half the Gaussian mass is {expected:.15e}"
        );
        // A positive multiplier leaves strictly less mass than the half Gaussian.
        assert!(slot.log_det_correction() > limit_correction);
    }
    let correction = term
        .pinned_half_line_log_det_correction(&rho, target.view(), &cache)
        .expect("the correction prices");
    assert_eq!(correction, total);
}

/// #4077 — the ARD log-precision entry of the outer gradient differentiates the criterion the
/// value reports, half-line mass included, with the pinned set held fixed.
///
/// Factor-wise at the converged state, as `tests_kappa_outer_gradient_2935` does: the
/// fixed-state partial against a central difference of the frozen cost over `ρ`, and the
/// implicit correction against the frozen cost along the mode response `θ̂ = −A⁺g_ρ`, whose
/// slope is `gᵀθ̂ + ½Γᵀθ̂` with `g` the inner residual. The response holds each pinned
/// coordinate on its bound, as the retraction does.
#[test]
fn pinned_half_line_mass_enters_the_outer_gradient_4077() {
    let (mut state, target, rho) = converged_interval_state();
    let flat = rho.ard_flat_index(0, 0);
    let (cost, loss, cache, geometry) = state
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-8,
            1.0e-8,
            true,
        )
        .expect("the criterion prices the converged state");
    let geometry = geometry.expect("the dense criterion hands out the block it priced");
    let pinned = state.last_pinned_bound_slots.clone();
    assert!(!pinned.is_empty(), "no slot is pinned at the fitted state");
    let pinned_indices: Vec<usize> = pinned
        .iter()
        .map(|&(row, local)| cache.row_offsets[row] + local)
        .collect();
    let residual = assembled_gradient(&state, target.view(), &rho);
    let components = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("dense gradient components at the converged state");
    let partial =
        components.explicit[flat] + components.logdet_trace[flat] + components.occam[flat];
    let implicit = components.third_order_correction[flat];
    let g_rho = state
        .outer_rho_gradient_ift_rhs(&rho, flat, &cache)
        .expect("ARD implicit right-hand side");
    let a_pinv_g = state
        .solve_exact_stationarity(&rho, target.view(), &cache, &g_rho)
        .expect("A⁺ g_ρ");
    let mut theta_hat = SaeArrowVector {
        t: a_pinv_g.t.mapv(|value| -value),
        beta: a_pinv_g.beta.mapv(|value| -value),
    };
    for &index in &pinned_indices {
        theta_hat.t[index] = 0.0;
    }

    let price = |at_state: &SaeManifoldTerm, at: &SaeManifoldRho| -> (f64, Vec<(usize, usize)>) {
        let mut arm = at_state.clone();
        let value = arm
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                at,
                None,
                0,
                0.4,
                1.0e-8,
                1.0e-8,
            )
            .expect("the criterion prices the fixed state")
            .0;
        (value, arm.last_pinned_bound_slots.clone())
    };
    let at_rho = |shift: f64| -> SaeManifoldRho {
        let mut at = rho.clone();
        at.log_ard[0][0] += shift;
        at
    };
    let cost_over_rho = |step: f64| -> f64 {
        let (plus, plus_pinned) = price(&state, &at_rho(step));
        let (minus, minus_pinned) = price(&state, &at_rho(-step));
        assert_eq!(plus_pinned, pinned, "the pinned set moved with ρ");
        assert_eq!(minus_pinned, pinned, "the pinned set moved with ρ");
        (plus - minus) / (2.0 * step)
    };
    let direction_scale = theta_hat
        .t
        .iter()
        .chain(theta_hat.beta.iter())
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    let cost_along_response = |eps: f64| -> f64 {
        let (plus, plus_pinned) = price(&displaced_interval(&state, &theta_hat, eps), &rho);
        let (minus, minus_pinned) = price(&displaced_interval(&state, &theta_hat, -eps), &rho);
        assert_eq!(plus_pinned, pinned, "the pinned set moved along the response");
        assert_eq!(minus_pinned, pinned, "the pinned set moved along the response");
        (plus - minus) / (2.0 * eps)
    };
    let richardson =
        |coarse: f64, fine: f64| ((4.0 * fine - coarse) / 3.0, (coarse - fine).abs());
    let rho_step = 1.0e-3_f64;
    let (partial_fd, partial_spread) =
        richardson(cost_over_rho(rho_step), cost_over_rho(0.5 * rho_step));
    let response_step = 1.0e-4 / direction_scale;
    let (directional_fd, directional_spread) = richardson(
        cost_along_response(response_step),
        cost_along_response(0.5 * response_step),
    );
    let residual_along_response =
        residual.t.dot(&theta_hat.t) + residual.beta.dot(&theta_hat.beta);
    let implicit_fd = directional_fd - residual_along_response;
    println!(
        "[#4077] cost={cost:.12e} partial={partial:.12e} partial_fd={partial_fd:.12e} (spread \
         {partial_spread:.3e}) implicit={implicit:.12e} implicit_fd={implicit_fd:.12e} (spread \
         {directional_spread:.3e})"
    );
    let partial_tolerance = 10.0 * partial_spread + 1.0e-6 * partial_fd.abs().max(1.0);
    assert!(
        (partial - partial_fd).abs() <= partial_tolerance,
        "fixed-state ARD partials {partial} are not the frozen cost's derivative {partial_fd} \
         (|Δ| = {:.3e}, tolerance {partial_tolerance:.3e})",
        (partial - partial_fd).abs()
    );
    let implicit_tolerance = 10.0 * directional_spread + 1.0e-6 * directional_fd.abs().max(1.0);
    assert!(
        (implicit - implicit_fd).abs() <= implicit_tolerance,
        "the implicit correction {implicit} is not ½Γᵀθ̂ = {implicit_fd} measured along θ̂ \
         (|Δ| = {:.3e}, tolerance {implicit_tolerance:.3e})",
        (implicit - implicit_fd).abs()
    );
}
