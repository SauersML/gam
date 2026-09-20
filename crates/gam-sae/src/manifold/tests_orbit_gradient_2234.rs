//! #2234 — the dense outer ρ-gradient of an orbit-priced criterion is the derivative of that criterion.
//!
//! A periodic atom with ARD enabled carries a closure-certified circle orbit, so its dense evaluation
//! prices `log|A_s| − log det N − 2·log I + log 2π` and its channels differentiate that value through
//! the orbit legs (`compact_orbit_differential`). The arbiter is factor-wise at one converged state, as
//! #2935's: the fixed-state partials against a central difference of the frozen criterion over the ρ
//! coordinate, and the implicit correction against the frozen criterion along `θ̂ = −A⁺g_ρ`. The
//! positive control removes the orbit legs and must miss the ARD partial.

use super::*;
use ndarray::{Array1, Array2};

const ROWS: usize = 42;
const OUTPUT: usize = 8;

fn planted_circle() -> Array2<f64> {
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let b0: Vec<f64> = (0..OUTPUT).map(|_| 2.0 * unit() - 1.0).collect();
    let b1: Vec<f64> = (0..OUTPUT).map(|_| 2.0 * unit() - 1.0).collect();
    let mut z = Array2::<f64>::zeros((ROWS, OUTPUT));
    for i in 0..ROWS {
        let theta = std::f64::consts::TAU * unit();
        for j in 0..OUTPUT {
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + 0.01 * (2.0 * unit() - 1.0);
        }
    }
    z
}

pub(super) fn periodic_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    planted_fixture("periodic")
}

/// One atom of `basis` with native ARD on the planted circle.
fn planted_fixture(basis: &str) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let z = planted_circle();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec![basis.to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        seed_refine_routing: minimal.refine_routing,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    (seed.base_term, z, seed.initial_rho)
}

pub(super) fn arrow_norm(vector: &SaeArrowVector) -> f64 {
    (vector.t.dot(&vector.t) + vector.beta.dot(&vector.beta)).sqrt()
}

pub(super) fn arrow_dot(x: &SaeArrowVector, y: &SaeArrowVector) -> f64 {
    x.t.dot(&y.t) + x.beta.dot(&y.beta)
}

pub(super) fn richardson(coarse: f64, fine: f64) -> (f64, f64) {
    ((4.0 * fine - coarse) / 3.0, (fine - coarse).abs())
}

pub(super) fn inner_gradient(term: &SaeManifoldTerm, target: ArrayView2<'_, f64>, rho: &SaeManifoldRho) -> SaeArrowVector {
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

/// The term moved by `scale·step`: row-major coordinates on the atom's own manifold and the
/// basis-major decoder (no decoder frame at this output width).
pub(super) fn displaced(term: &SaeManifoldTerm, step: &SaeArrowVector, scale: f64) -> SaeManifoldTerm {
    let mut moved = term.clone();
    let coords = moved.assignment.coords[0].as_matrix().to_owned();
    let manifold = moved.assignment.coords[0].manifold().clone();
    let coordinate_step =
        Array2::from_shape_vec(coords.dim(), step.t.to_vec()).expect("row-major coordinate layout");
    moved.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
        (&coords + &(coordinate_step * scale)).view(),
        LatentIdMode::None,
        manifold,
    );
    let decoder = moved.atoms[0].decoder_coefficients().clone();
    let decoder_step =
        Array2::from_shape_vec(decoder.dim(), step.beta.to_vec()).expect("basis-major decoder layout");
    moved.atoms[0]
        .set_decoder_coefficients(&decoder + &(decoder_step * scale))
        .expect("the decoder keeps its shape");
    moved
}

#[test]
fn orbit_priced_dense_gradient_factors_are_derivatives_of_the_criterion_2234() {
    let (term, target, rho) = periodic_fixture();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.05, 1.0e-6, 1.0e-6);
    let anchor = objective.baseline_rho.clone();
    objective
        .evaluate_outer_criterion_route(&anchor, true, false)
        .expect("the dense route converges the anchor");
    let mut state = objective.term.clone();
    assert!(state.atoms[0].decoder_frame.is_none(), "the fixture's border is the full decoder");
    let (cost, loss, cache, geometry) = state
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &anchor,
            None,
            0,
            0.05,
            1.0e-6,
            1.0e-6,
            true,
        )
        .expect("the criterion prices the converged state");
    let mut geometry = geometry.expect("the dense criterion hands out the block it priced");
    // Premise: the evaluation integrates the atom's orbit.
    assert!(
        !geometry.orbit_generators.is_empty() && geometry.block.orbit.is_some(),
        "the periodic ARD atom must be priced through its circle orbit"
    );
    let components = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &anchor,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("dense gradient components at the converged state");
    let residual = inner_gradient(&state, target.view(), &anchor);
    let flat = anchor.flat_coordinates();

    let price = |at_state: &SaeManifoldTerm, at: &SaeManifoldRho| -> f64 {
        let mut arm = at_state.clone();
        arm.penalized_quasi_laplace_criterion_with_cache(target.view(), at, None, 0, 0.05, 1.0e-6, 1.0e-6)
            .expect("the criterion prices the fixed state")
            .0
    };
    let moved_rho = |index: usize, step: f64| -> SaeManifoldRho {
        let mut coordinates = flat.clone();
        coordinates[index] += step;
        anchor.from_flat(coordinates.view()).expect("a nearby ρ")
    };

    let coordinates = [
        ("ard", anchor.ard_flat_index(0, 0)),
        ("smooth", anchor.smooth_flat_index(0)),
    ];
    let mut ard_partial = f64::NAN;
    let mut ard_partial_fd = f64::NAN;
    let mut ard_tolerance = f64::NAN;
    for (label, index) in coordinates {
        let partial = components.explicit[index] + components.logdet_trace[index] + components.occam[index];
        let implicit = components.third_order_correction[index];
        let cost_over_rho = |step: f64| -> f64 {
            (price(&state, &moved_rho(index, step)) - price(&state, &moved_rho(index, -step))) / (2.0 * step)
        };
        let step = 1.0e-3_f64;
        let (partial_fd, partial_spread) = richardson(cost_over_rho(step), cost_over_rho(0.5 * step));
        let g_rho = state
            .outer_rho_gradient_ift_rhs(&anchor, index, &cache)
            .expect("implicit right-hand side");
        let a_pinv_g = state
            .solve_exact_stationarity(&anchor, target.view(), &cache, &g_rho)
            .expect("A⁺ g_ρ");
        let theta_hat = SaeArrowVector {
            t: a_pinv_g.t.mapv(|value| -value),
            beta: a_pinv_g.beta.mapv(|value| -value),
        };
        let response_step = 1.0e-4 / arrow_norm(&theta_hat).max(1.0);
        let cost_along_response = |eps: f64| -> f64 {
            (price(&displaced(&state, &theta_hat, eps), &anchor) - price(&displaced(&state, &theta_hat, -eps), &anchor))
                / (2.0 * eps)
        };
        let (directional_fd, directional_spread) =
            richardson(cost_along_response(response_step), cost_along_response(0.5 * response_step));
        let implicit_fd = directional_fd - arrow_dot(&residual, &theta_hat);
        let partial_tolerance = 10.0 * partial_spread + 1.0e-6 * partial_fd.abs().max(1.0);
        let implicit_tolerance = 10.0 * directional_spread + 1.0e-6 * directional_fd.abs().max(1.0);
        eprintln!(
            "[#2234 orbit gradient] {label}: cost={cost:.12e} partial={partial:.12e} partial_fd={partial_fd:.12e} \
             (spread {partial_spread:.3e}) implicit={implicit:.12e} implicit_fd={implicit_fd:.12e} \
             (spread {directional_spread:.3e})"
        );
        assert!(
            (partial - partial_fd).abs() <= partial_tolerance,
            "{label}: fixed-state partials {partial} are not the frozen criterion's derivative {partial_fd} \
             (|Δ| = {}, tolerance {partial_tolerance})",
            (partial - partial_fd).abs()
        );
        assert!(
            (implicit - implicit_fd).abs() <= implicit_tolerance,
            "{label}: the implicit correction {implicit} is not ½Γᵀθ̂ = {implicit_fd} (|Δ| = {}, tolerance {implicit_tolerance})",
            (implicit - implicit_fd).abs()
        );
        if label == "ard" {
            ard_partial = partial;
            ard_partial_fd = partial_fd;
            ard_tolerance = partial_tolerance;
        }
    }
    eprintln!("[#2234 orbit gradient] ard partial {ard_partial:.12e} against {ard_partial_fd:.12e}");

    // Positive control: without the orbit legs the channels contract the stiffened block's own weights.
    geometry.orbit_generators.clear();
    let control = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &anchor,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("control gradient components");
    let index = anchor.ard_flat_index(0, 0);
    let control_partial = control.explicit[index] + control.logdet_trace[index] + control.occam[index];
    eprintln!("[#2234 orbit gradient] control ard partial {control_partial:.12e}");
    assert!(
        (control_partial - ard_partial_fd).abs() > 10.0 * ard_tolerance,
        "the control without orbit legs already matches the frozen criterion ({control_partial} vs \
         {ard_partial_fd}, tolerance {ard_tolerance}), so the pin cannot see the orbit legs"
    );
}

/// Both routes price the orbit state: the dense route through its stiffened block, and the same state
/// routed to the arrow evaluation by its admission input through the arrow orbit lane (#2234 step 1a),
/// to the resolution of the two inner solves each objective converges. An atom with no compact orbit
/// takes the arrow route's ordinary evidence.
#[test]
fn streaming_route_prices_the_orbit_criterion_as_the_dense_route_does_2234() {
    let (term, target, rho) = periodic_fixture();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.05, 1.0e-6, 1.0e-6);
    let at = objective.baseline_rho.clone();
    let dense = objective
        .evaluate_outer_criterion_route(&at, true, false)
        .expect("the dense route prices the orbit state");
    assert!(dense.cost.is_finite(), "the dense orbit criterion must be finite, got {}", dense.cost);
    let (_, _, _, geometry) = objective
        .term
        .clone()
        .penalized_quasi_laplace_criterion_with_geometry(target.view(), &at, None, 0, 0.05, 1.0e-6, 1.0e-6, true)
        .expect("the criterion prices the converged state");
    assert!(
        geometry.is_some_and(|geometry| !geometry.orbit_generators.is_empty()),
        "the dense route must price the atom through its circle orbit"
    );
    let streaming = objective.evaluate_outer_criterion_route(&at, false, false);
    eprintln!(
        "[#2234 orbit routes] dense cost={:.12e} streaming={:?}",
        dense.cost,
        streaming.as_ref().map(|evaluation| evaluation.cost)
    );
    let streaming = streaming.expect("the arrow orbit lane prices the orbit state");
    let resolution = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * dense.cost.abs().max(1.0);
    assert!(
        (streaming.cost - dense.cost).abs() <= resolution,
        "the arrow orbit lane's cost {} misses the dense route's {} (|Δ| {:e}, resolution {resolution:e})",
        streaming.cost,
        dense.cost,
        (streaming.cost - dense.cost).abs()
    );

    let (term, target, rho) = planted_fixture("linear");
    let mut control =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.05, 1.0e-6, 1.0e-6);
    let at = control.baseline_rho.clone();
    let routed = control.evaluate_outer_criterion_route(&at, false, false);
    eprintln!(
        "[#2234 orbit routes] control linear atom on the arrow route: {:?}",
        routed.as_ref().map(|evaluation| evaluation.cost)
    );
    assert!(
        !matches!(routed, Err(SaeCriterionError::OrbitCriterionUnavailableOnArrowRoute { .. })),
        "a linear atom carries no compact orbit, so the arrow route must not refuse it for one"
    );
}

const TOPK_ROWS: usize = 48;
const TOPK_OUTPUT: usize = 6;

/// Two planted circles in orthogonal output planes, the rows alternating between them.
fn planted_two_circles() -> Array2<f64> {
    let mut state = 0x1357_9bdf_2468_ace0u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut z = Array2::<f64>::zeros((TOPK_ROWS, TOPK_OUTPUT));
    for i in 0..TOPK_ROWS {
        let theta = std::f64::consts::TAU * unit();
        let plane = 2 * (i % 2);
        z[[i, plane]] = 2.0 * theta.cos();
        z[[i, plane + 1]] = 2.0 * theta.sin();
        for j in 0..TOPK_OUTPUT {
            z[[i, j]] += 0.01 * (2.0 * unit() - 1.0);
        }
    }
    z
}

/// Two periodic atoms with native ARD under hard TopK(1): each row holds one atom's coordinate,
/// so the cache's layout is compact.
pub(super) fn topk_two_circle_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let z = planted_two_circles();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["periodic".to_string(); 2],
        atom_dim: vec![1, 1],
        assignment_kind: SaeFitAssignmentKind::TopK,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: Some(1),
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::TopK,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: Some(1),
        threshold: 0.0,
        seed_refine_routing: minimal.refine_routing,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    (seed.base_term, z, seed.initial_rho)
}

/// The term moved by `scale·step` in the compact hard-TopK layout: each row's selected coordinates
/// at their compact slots, and every atom's basis-major decoder block of the border.
pub(super) fn displaced_compact(
    term: &SaeManifoldTerm,
    row_offsets: &[usize],
    step: &SaeArrowVector,
    scale: f64,
) -> SaeManifoldTerm {
    let mut moved = term.clone();
    let layout = moved
        .last_row_layout
        .clone()
        .expect("a hard-TopK state carries its row layout");
    for atom in 0..moved.k_atoms() {
        let mut coords = moved.assignment.coords[atom].as_matrix().to_owned();
        for row in 0..layout.active_atoms.len() {
            if let Some(position) = layout.active_atoms[row].iter().position(|&active| active == atom) {
                coords[[row, 0]] += scale * step.t[row_offsets[row] + layout.coord_starts[row][position]];
            }
        }
        let manifold = moved.assignment.coords[atom].manifold().clone();
        moved.assignment.coords[atom] =
            LatentCoordValues::from_matrix_with_manifold(coords.view(), LatentIdMode::None, manifold);
    }
    let offsets = moved.factored_border_offsets();
    for atom in 0..moved.k_atoms() {
        let decoder = moved.atoms[atom].decoder_coefficients().clone();
        let block = step
            .beta
            .slice(ndarray::s![offsets[atom]..offsets[atom] + decoder.len()])
            .to_vec();
        let decoder_step = Array2::from_shape_vec(decoder.dim(), block).expect("basis-major decoder layout");
        moved.atoms[atom]
            .set_decoder_coefficients(&decoder + &(decoder_step * scale))
            .expect("the decoder keeps its shape");
    }
    moved
}

/// #2234 item 2 — two circle orbits in a compact hard-TopK layout are each integrated exactly, and
/// the criterion's dense ρ-gradient through both orbits' legs is its derivative. Before, the
/// compact layout priced every atom by Laplace (`CompactRowLayout`), and two certified orbits
/// did as well (`MultipleCompactOrbits`). The positive control drops the orbit legs.
#[test]
fn two_compact_topk_orbits_are_integrated_and_differentiated_2234() {
    let (term, target, rho) = topk_two_circle_fixture();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.05, 1.0e-6, 1.0e-6);
    let anchor = objective.baseline_rho.clone();
    objective
        .evaluate_outer_criterion_route(&anchor, true, false)
        .expect("the dense route converges the anchor");
    let mut state = objective.term.clone();
    let (cost, loss, cache, geometry) = state
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &anchor,
            None,
            0,
            0.05,
            1.0e-6,
            1.0e-6,
            true,
        )
        .expect("the criterion prices the converged state");
    let mut geometry = geometry.expect("the dense criterion hands out the block it priced");
    let layout = state
        .last_row_layout
        .clone()
        .expect("hard TopK assembles a compact row layout");
    let selecting: Vec<usize> = (0..2)
        .map(|atom| layout.active_atoms.iter().filter(|active| active.contains(&atom)).count())
        .collect();
    let orbit_atoms: Vec<usize> = geometry.orbit_generators.iter().map(|generator| generator.atom).collect();
    let orbit_rows: Vec<usize> = geometry
        .orbit_generators
        .iter()
        .map(|generator| generator.prior_rows.len())
        .collect();
    eprintln!(
        "[#2234 compact orbits] cost={cost:.12e} delta_t_len={} dense n·q={} border={} selecting rows={selecting:?} \
         orbit atoms={orbit_atoms:?} orbit prior rows={orbit_rows:?}",
        cache.delta_t_len(),
        TOPK_ROWS * state.assignment.row_block_dim(),
        cache.k,
    );
    // Premise: a compact layout, both atoms' orbits certified and separated, each over the rows
    // that select its atom.
    assert!(
        cache.delta_t_len() < TOPK_ROWS * state.assignment.row_block_dim(),
        "hard TopK(1) must drop the unselected atoms' coordinates"
    );
    assert_eq!(orbit_atoms, vec![0, 1], "both periodic atoms must be integrated through their orbits");
    assert_eq!(orbit_rows, selecting, "each orbit's prior covers exactly the rows selecting its atom");
    assert!(geometry.block.orbit.is_some(), "the block must carry the eliminated orbits");

    let components = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &anchor,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("dense gradient components at the converged state");
    let residual = inner_gradient(&state, target.view(), &anchor);
    let flat = anchor.flat_coordinates();
    let row_offsets = cache.row_offsets.to_vec();
    let price = |at_state: &SaeManifoldTerm, at: &SaeManifoldRho| -> f64 {
        let mut arm = at_state.clone();
        arm.penalized_quasi_laplace_criterion_with_cache(target.view(), at, None, 0, 0.05, 1.0e-6, 1.0e-6)
            .expect("the criterion prices the fixed state")
            .0
    };
    let moved_rho = |index: usize, step: f64| -> SaeManifoldRho {
        let mut coordinates = flat.clone();
        coordinates[index] += step;
        anchor.from_flat(coordinates.view()).expect("a nearby ρ")
    };
    let coordinates = [
        ("ard0", anchor.ard_flat_index(0, 0)),
        ("ard1", anchor.ard_flat_index(1, 0)),
        ("smooth0", anchor.smooth_flat_index(0)),
    ];
    let mut ard_checks = Vec::new();
    for (label, index) in coordinates {
        let partial = components.explicit[index] + components.logdet_trace[index] + components.occam[index];
        let implicit = components.third_order_correction[index];
        let cost_over_rho = |step: f64| -> f64 {
            (price(&state, &moved_rho(index, step)) - price(&state, &moved_rho(index, -step))) / (2.0 * step)
        };
        let step = 1.0e-3_f64;
        let (partial_fd, partial_spread) = richardson(cost_over_rho(step), cost_over_rho(0.5 * step));
        let g_rho = state
            .outer_rho_gradient_ift_rhs(&anchor, index, &cache)
            .expect("implicit right-hand side");
        let a_pinv_g = state
            .solve_exact_stationarity(&anchor, target.view(), &cache, &g_rho)
            .expect("A⁺ g_ρ");
        let theta_hat = SaeArrowVector {
            t: a_pinv_g.t.mapv(|value| -value),
            beta: a_pinv_g.beta.mapv(|value| -value),
        };
        let response_step = 1.0e-4 / arrow_norm(&theta_hat).max(1.0);
        let cost_along_response = |eps: f64| -> f64 {
            (price(&displaced_compact(&state, &row_offsets, &theta_hat, eps), &anchor)
                - price(&displaced_compact(&state, &row_offsets, &theta_hat, -eps), &anchor))
                / (2.0 * eps)
        };
        let (directional_fd, directional_spread) =
            richardson(cost_along_response(response_step), cost_along_response(0.5 * response_step));
        let implicit_fd = directional_fd - arrow_dot(&residual, &theta_hat);
        let partial_tolerance = 10.0 * partial_spread + 1.0e-6 * partial_fd.abs().max(1.0);
        let implicit_tolerance = 10.0 * directional_spread + 1.0e-6 * directional_fd.abs().max(1.0);
        eprintln!(
            "[#2234 compact orbits] {label}: partial={partial:.12e} partial_fd={partial_fd:.12e} \
             (spread {partial_spread:.3e}) implicit={implicit:.12e} implicit_fd={implicit_fd:.12e} \
             (spread {directional_spread:.3e})"
        );
        assert!(
            (partial - partial_fd).abs() <= partial_tolerance,
            "{label}: fixed-state partials {partial} are not the frozen criterion's derivative {partial_fd} \
             (|Δ| = {}, tolerance {partial_tolerance})",
            (partial - partial_fd).abs()
        );
        assert!(
            (implicit - implicit_fd).abs() <= implicit_tolerance,
            "{label}: the implicit correction {implicit} is not ½Γᵀθ̂ = {implicit_fd} (|Δ| = {}, tolerance {implicit_tolerance})",
            (implicit - implicit_fd).abs()
        );
        if label.starts_with("ard") {
            ard_checks.push((index, partial_fd, partial_tolerance));
        }
    }

    // Positive control: without the orbit legs the channels contract the stiffened block's own
    // weights, and at least one ARD partial misses the frozen criterion.
    geometry.orbit_generators.clear();
    let control = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &anchor,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("control gradient components");
    let misses: Vec<f64> = ard_checks
        .iter()
        .map(|&(index, partial_fd, tolerance)| {
            let control_partial = control.explicit[index] + control.logdet_trace[index] + control.occam[index];
            (control_partial - partial_fd).abs() / tolerance
        })
        .collect();
    eprintln!("[#2234 compact orbits] control |Δ|/tolerance per ARD axis {misses:?}");
    assert!(
        misses.iter().any(|&miss| miss > 10.0),
        "the control without orbit legs already matches the frozen criterion on every ARD axis ({misses:?}), \
         so the pin cannot see the orbit legs"
    );
}
