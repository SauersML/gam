//! #2935 — the κ entry of the outer ρ-gradient is the derivative of the criterion
//! each route reports.
//!
//! A constant-curvature atom's criterion moves with its raw sectional curvature κ
//! only through the penalty Gram `S(κ)`: the penalty energy `½λ<B, S B>`, the
//! observed information `½log|A|`, the rank charge's basis EDF `tr(G(G+λS)⁻¹)`, the
//! smoothing-prior normalizer `½r·log|S|_+`, and the fitted state that re-converges
//! against all of them.
//!
//! The arbiter is factor-wise at one converged state, not a central difference of
//! the re-converged cost. On this fixture the implicit response amplifies the κ
//! right-hand side about 360-fold and `‖Γ‖` is about 6e3, so a cost difference over
//! re-converged endpoints would need roots accurate to about 1e-13, while the
//! production inner solve stops near 1e-4. Each factor is instead checked against
//! its own central difference at the converged state:
//! - the fixed-state partials (explicit with the rank charge's direct part, log-det
//!   trace, Occam) against a difference of the frozen cost over κ;
//! - the implicit right-hand side `g_κ` against a difference of the inner gradient
//!   over κ;
//! - `θ̂_κ = −A⁺g_κ` through `A·θ̂_κ = −g_κ`, where `A·θ̂_κ` is a difference of the
//!   inner gradient along `θ̂_κ`;
//! - the implicit correction `½Γᵀθ̂_κ` through the frozen cost along `θ̂_κ`, whose
//!   slope is `gᵀθ̂_κ + ½Γᵀθ̂_κ` with `g` the inner residual measured at that state.
#![cfg(test)]
use super::*;
use ndarray::{Array1, Array2, ArrayView2};

const KAPPA: f64 = 0.3;

/// One constant-curvature tangent-chart atom observed with a small deterministic
/// residual, so the exact observed information carries residual curvature.
///
/// The signal dominates both seed priors, so the inner fit keeps the latent chart
/// excited. At a decoder amplitude of 0.4 with `log λ = −1`, the smoothing prior
/// shrinks the linear decoder columns to about a fifth of their data value. Every
/// coordinate then collapses onto the origin within 40 inner iterations, the
/// data-supported reduction keeps only the constant column, and κ no longer moves
/// the criterion.
fn curvature_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 24usize;
    let p = 3usize;
    let coords = Array2::from_shape_fn((n, 2), |(row, axis)| {
        let angle = std::f64::consts::TAU * 0.618_033_988_75 * row as f64;
        let radius = 0.15 + 0.45 * (row as f64 + 0.5) / n as f64;
        if axis == 0 {
            radius * angle.cos()
        } else {
            radius * angle.sin()
        }
    });
    let plan = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Poincare,
        2,
        SaeBasisResolution::Polynomial {
            degree: SAE_EUCLIDEAN_PATCH_MAX_DEGREE,
        },
        SaeReferenceMetricPlan::ConstantCurvatureChart {
            kappa: KAPPA,
            reference_coords: coords.clone(),
        },
    )
    .expect("constant-curvature plan");
    let bundle = plan
        .evaluate_bundle(coords.view())
        .expect("reference rows lie inside the chart");
    let m = bundle.basis_values.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
        4.0 * ((1 + basis_col + 3 * out_col) as f64).sin()
    });
    let mut target = bundle.basis_values.dot(&decoder);
    for ((row, col), value) in target.indexed_iter_mut() {
        *value += 0.02 * ((7 * row + 3 * col) as f64).sin();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "curvature",
        SaeAtomBasisKind::Poincare,
        2,
        bundle.basis_values,
        bundle.basis_jacobian,
        decoder,
        bundle.reference_penalty,
    )
    .expect("atom from the plan's own bundle")
    .with_basis_second_jet(bundle.evaluator)
    .with_geometry_plan(plan)
    .expect("installed Gram is the plan's Gram");
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("one coordinate block for one atom");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("single-atom term");
    let rho = SaeManifoldRho::new(-1.0, -5.0, vec![Array1::from_elem(2, -1.0)]);
    (term, target, rho)
}

fn arrow_norm(vector: &SaeArrowVector) -> f64 {
    (vector.t.dot(&vector.t) + vector.beta.dot(&vector.beta)).sqrt()
}

fn arrow_dot(x: &SaeArrowVector, y: &SaeArrowVector) -> f64 {
    x.t.dot(&y.t) + x.beta.dot(&y.beta)
}

fn arrow_max(vector: &SaeArrowVector) -> f64 {
    vector
        .t
        .iter()
        .chain(vector.beta.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn arrow_gap(x: &SaeArrowVector, y: &SaeArrowVector) -> f64 {
    assert_eq!(x.t.len(), y.t.len(), "coordinate blocks share one layout");
    assert_eq!(x.beta.len(), y.beta.len(), "decoder blocks share one layout");
    x.t.iter()
        .zip(y.t.iter())
        .chain(x.beta.iter().zip(y.beta.iter()))
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
}

fn arrow_scaled_difference(plus: &SaeArrowVector, minus: &SaeArrowVector, step: f64) -> SaeArrowVector {
    SaeArrowVector {
        t: (&plus.t - &minus.t) / (2.0 * step),
        beta: (&plus.beta - &minus.beta) / (2.0 * step),
    }
}

/// `(4·fine − coarse)/3` and `|coarse − fine|`.
fn richardson(coarse: f64, fine: f64) -> (f64, f64) {
    ((4.0 * fine - coarse) / 3.0, (coarse - fine).abs())
}

/// The inner stationarity residual `(g_t, g_β)` at the term's coordinates and decoder.
fn inner_gradient(
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

/// The term moved by `scale·step` in the arrow layout: row-major coordinates and the
/// basis-major decoder of the single atom.
fn displaced(term: &SaeManifoldTerm, step: &SaeArrowVector, scale: f64) -> SaeManifoldTerm {
    let mut moved = term.clone();
    let coords = moved.assignment.coords[0].as_matrix();
    let coordinate_step =
        Array2::from_shape_vec(coords.dim(), step.t.to_vec()).expect("row-major coordinate layout");
    moved.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
        (&coords + &(coordinate_step * scale)).view(),
        LatentIdMode::None,
        LatentManifold::Euclidean,
    );
    let decoder = moved.atoms[0].decoder_coefficients().clone();
    let decoder_step =
        Array2::from_shape_vec(decoder.dim(), step.beta.to_vec()).expect("basis-major decoder layout");
    moved.atoms[0]
        .set_decoder_coefficients(&decoder + &(decoder_step * scale))
        .expect("the decoder keeps its shape");
    moved
}

/// The dense route's converged anchor, checked to keep every penalized column.
fn converged_anchor(
    term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: SaeManifoldRho,
) -> (SaeManifoldTerm, SaeManifoldRho, usize) {
    let mut objective =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    let anchor = objective.baseline_rho.clone();
    let flat = anchor
        .kappa_flat_index(0)
        .expect("the curvature atom owns an outer coordinate");
    assert_eq!(
        anchor.kappa[0].to_bits(),
        KAPPA.to_bits(),
        "the objective seeds κ from the atom's geometry plan"
    );
    objective
        .evaluate_outer_criterion_route(&anchor, true, false)
        .expect("the dense route converges the anchor");
    // Premise: the fit kept every penalized column. A collapsed chart leaves the atom
    // on its penalty's null space, where κ has nothing to move.
    let live_width = objective.term.atoms[0].basis_size();
    let full_width = objective.term.atoms[0].full_basis_size();
    println!("[#2935] live basis width {live_width} of {full_width}");
    assert_eq!(
        live_width, full_width,
        "the inner fit reduced the curvature atom to {live_width} of {full_width} basis \
         columns, so κ cannot move the criterion"
    );
    (objective.term.clone(), anchor, flat)
}

#[test]
fn dense_kappa_gradient_factors_are_derivatives_of_the_criterion_2935() {
    let (term, target, rho) = curvature_fixture();
    let (mut state, anchor, flat) = converged_anchor(term, &target, rho);
    let (cost, loss, cache, geometry) = state
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &anchor,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
            true,
        )
        .expect("the criterion prices the converged state");
    let geometry = geometry.expect("the dense criterion hands out the block it priced");
    let residual = inner_gradient(&state, target.view(), &anchor);
    let lambda_smooth = anchor.lambda_smooth_vec().expect("smoothing strengths");
    let solver = state
        .outer_gradient_arrow_solver(&cache, &lambda_smooth)
        .expect("dense outer gradient solver");
    let components = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &anchor,
            &loss,
            &cache,
            &solver,
            None,
            None,
            Some(&geometry),
        )
        .expect("dense κ gradient components at the converged state");
    let partial = components.explicit[flat] + components.logdet_trace[flat] + components.occam[flat];
    let implicit = components.third_order_correction[flat];
    let complete = components.gradient()[flat];
    let g_kappa = state
        .outer_rho_gradient_ift_rhs(&anchor, flat, &cache)
        .expect("κ implicit right-hand side");
    let a_pinv_g = state
        .solve_exact_stationarity(&anchor, target.view(), &cache, &g_kappa)
        .expect("A⁺ g_κ");
    let theta_hat = SaeArrowVector {
        t: a_pinv_g.t.mapv(|value| -value),
        beta: a_pinv_g.beta.mapv(|value| -value),
    };
    let negative_g_kappa = SaeArrowVector {
        t: g_kappa.t.mapv(|value| -value),
        beta: g_kappa.beta.mapv(|value| -value),
    };

    let price = |at_state: &SaeManifoldTerm, at: &SaeManifoldRho| -> f64 {
        let mut arm = at_state.clone();
        arm.penalized_quasi_laplace_criterion_with_cache(target.view(), at, None, 0, 0.4, 1.0e-6, 1.0e-6)
            .expect("the criterion prices the fixed state")
            .0
    };
    let at_kappa = |kappa: f64| -> (SaeManifoldTerm, SaeManifoldRho) {
        let mut arm = state.clone();
        let prepared = arm.atoms[0]
            .prepare_constant_curvature(kappa)
            .expect("the trial curvature lies inside the chart's domain");
        arm.atoms[0].commit_prepared_constant_curvature(prepared);
        let mut at = anchor.clone();
        at.kappa[0] = kappa;
        (arm, at)
    };
    let cost_over_kappa = |step: f64| -> f64 {
        let (plus, plus_rho) = at_kappa(KAPPA + step);
        let (minus, minus_rho) = at_kappa(KAPPA - step);
        (price(&plus, &plus_rho) - price(&minus, &minus_rho)) / (2.0 * step)
    };
    let gradient_over_kappa = |step: f64| -> SaeArrowVector {
        let (plus, plus_rho) = at_kappa(KAPPA + step);
        let (minus, minus_rho) = at_kappa(KAPPA - step);
        arrow_scaled_difference(
            &inner_gradient(&plus, target.view(), &plus_rho),
            &inner_gradient(&minus, target.view(), &minus_rho),
            step,
        )
    };
    let direction_scale = arrow_norm(&theta_hat).max(1.0);
    let gradient_along_response = |eps: f64| -> SaeArrowVector {
        arrow_scaled_difference(
            &inner_gradient(&displaced(&state, &theta_hat, eps), target.view(), &anchor),
            &inner_gradient(&displaced(&state, &theta_hat, -eps), target.view(), &anchor),
            eps,
        )
    };
    let cost_along_response = |eps: f64| -> f64 {
        (price(&displaced(&state, &theta_hat, eps), &anchor)
            - price(&displaced(&state, &theta_hat, -eps), &anchor))
            / (2.0 * eps)
    };

    let kappa_step = 1.0e-3_f64;
    let (partial_fd, partial_spread) =
        richardson(cost_over_kappa(kappa_step), cost_over_kappa(0.5 * kappa_step));
    let g_kappa_coarse = gradient_over_kappa(kappa_step);
    let g_kappa_fine = gradient_over_kappa(0.5 * kappa_step);
    let response_step = 1.0e-4 / direction_scale;
    let a_theta_coarse = gradient_along_response(response_step);
    let a_theta_fine = gradient_along_response(0.5 * response_step);
    let (directional_fd, directional_spread) = richardson(
        cost_along_response(response_step),
        cost_along_response(0.5 * response_step),
    );
    // The frozen cost's slope along θ̂_κ is gᵀθ̂_κ + ½Γᵀθ̂_κ: the penalized loss
    // contributes its inner residual g, measured at this same state.
    let residual_along_response = arrow_dot(&residual, &theta_hat);
    let implicit_fd = directional_fd - residual_along_response;

    println!(
        "[#2935 dense] cost={cost:.12e} |g|={:.6e} |θ̂_κ|={:.6e} gᵀθ̂_κ={residual_along_response:.12e} \
         partial={partial:.12e} partial_fd={partial_fd:.12e} (spread {partial_spread:.3e}) \
         implicit={implicit:.12e} directional_fd={directional_fd:.12e} (spread {directional_spread:.3e}) \
         implicit_fd={implicit_fd:.12e} complete={complete:.12e}",
        arrow_norm(&residual),
        arrow_norm(&theta_hat)
    );
    println!(
        "[#2935 dense] |g_κ|∞={:.6e} max|g_κ − fd|={:.3e} (spread {:.3e}); \
         max|A·θ̂_κ + g_κ| by difference={:.3e} (spread {:.3e})",
        arrow_max(&g_kappa),
        arrow_gap(&g_kappa, &g_kappa_fine),
        arrow_gap(&g_kappa_coarse, &g_kappa_fine),
        arrow_gap(&negative_g_kappa, &a_theta_fine),
        arrow_gap(&a_theta_coarse, &a_theta_fine)
    );

    // Every premise below is measured by difference, independent of the channel it
    // licenses, so a missing channel fails its own factor check rather than a premise.
    let partial_tolerance = 10.0 * partial_spread + 1.0e-6 * partial_fd.abs().max(1.0);
    assert!(
        partial_fd.abs() > 1.0e3 * partial_tolerance,
        "the frozen cost must move materially with κ against the partial tolerance \
         ({partial_fd} vs {partial_tolerance:.3e})"
    );
    assert!(
        (partial - partial_fd).abs() <= partial_tolerance,
        "fixed-state κ partials {partial} are not the frozen cost's κ derivative {partial_fd} \
         (|Δ| = {}, tolerance {partial_tolerance})",
        (partial - partial_fd).abs()
    );
    let g_kappa_tolerance = 10.0 * arrow_gap(&g_kappa_coarse, &g_kappa_fine)
        + 1.0e-6 * arrow_max(&g_kappa_fine).max(1.0);
    assert!(
        arrow_max(&g_kappa_fine) > 1.0e3 * g_kappa_tolerance,
        "the inner gradient must move materially with κ against its tolerance \
         ({:.3e} vs {g_kappa_tolerance:.3e})",
        arrow_max(&g_kappa_fine)
    );
    assert!(
        arrow_gap(&g_kappa, &g_kappa_fine) <= g_kappa_tolerance,
        "g_κ is not the inner gradient's κ derivative (max |Δ| = {}, tolerance {g_kappa_tolerance})",
        arrow_gap(&g_kappa, &g_kappa_fine)
    );
    let response_tolerance = 10.0 * arrow_gap(&a_theta_coarse, &a_theta_fine)
        + 1.0e-6 * arrow_max(&g_kappa_fine).max(1.0);
    assert!(
        arrow_gap(&negative_g_kappa, &a_theta_fine) <= response_tolerance,
        "θ̂_κ = −A⁺g_κ does not solve the linearized stationarity: max |A·θ̂_κ + g_κ| by \
         difference = {}, tolerance {response_tolerance}",
        arrow_gap(&negative_g_kappa, &a_theta_fine)
    );
    let implicit_tolerance = 10.0 * directional_spread + 1.0e-6 * directional_fd.abs().max(1.0);
    assert!(
        implicit_fd.abs() > 1.0e3 * implicit_tolerance,
        "the frozen cost must move materially along θ̂_κ against the implicit tolerance \
         ({implicit_fd} vs {implicit_tolerance:.3e})"
    );
    assert!(
        (implicit - implicit_fd).abs() <= implicit_tolerance,
        "the implicit correction {implicit} is not ½Γᵀθ̂_κ = {implicit_fd} measured along θ̂_κ \
         (|Δ| = {}, tolerance {implicit_tolerance})",
        (implicit - implicit_fd).abs()
    );
    assert!(
        (complete - (partial_fd + implicit_fd)).abs() <= partial_tolerance + implicit_tolerance,
        "the complete κ entry {complete} is not the verified partials {partial_fd} plus the \
         verified implicit response {implicit_fd}"
    );
}

#[test]
fn streaming_route_kappa_gradient_matches_the_dense_route_at_one_state_2935() {
    let (term, target, rho) = curvature_fixture();
    let (state, anchor, _) = converged_anchor(term, &target, rho);
    // A zero inner budget prices both routes at the converged state.
    let mut objective =
        SaeManifoldOuterObjective::new(state, target.clone(), None, anchor, 0, 0.4, 1.0e-6, 1.0e-6);
    let at = objective.baseline_rho.clone();
    let flat = at
        .kappa_flat_index(0)
        .expect("the curvature atom owns an outer coordinate");
    let dense_evaluation = objective
        .evaluate_outer_criterion_route(&at, true, false)
        .expect("the dense route prices the state");
    let dense = objective
        .analytic_gradient_for_outer_evaluation(&at, &dense_evaluation)
        .expect("the dense route differentiates the state")[flat];
    let streaming_evaluation = objective
        .evaluate_outer_criterion_route(&at, false, false)
        .expect("the streaming route prices the state");
    let streaming = objective
        .analytic_gradient_for_outer_evaluation(&at, &streaming_evaluation)
        .expect("the streaming route differentiates the state")[flat];
    println!(
        "[#2935 routes] dense cost={:.12e} streaming cost={:.12e} dense dV/dκ={dense:.12e} \
         streaming dV/dκ={streaming:.12e}",
        dense_evaluation.cost, streaming_evaluation.cost
    );
    assert!(
        (dense_evaluation.cost - streaming_evaluation.cost).abs()
            <= 1.0e-8 * dense_evaluation.cost.abs().max(1.0),
        "both routes must price one state ({} vs {})",
        dense_evaluation.cost,
        streaming_evaluation.cost
    );
    assert!(
        dense.abs() > 1.0e-3,
        "the κ entry must be material for route parity to mean anything ({dense})"
    );
    let tolerance = 1.0e-6 * dense.abs().max(1.0);
    assert!(
        (streaming - dense).abs() <= tolerance,
        "the streaming κ entry {streaming} is not the dense κ entry {dense} at one state \
         (|Δ| = {}, tolerance {tolerance})",
        (streaming - dense).abs()
    );
}

/// A data-supported reduction may keep only directions in the penalty's null
/// space. A trial curvature must then install the reduced congruence `QᵀS(κ)Q`,
/// which is zero there, instead of refusing the atom. A reduction that keeps a
/// penalized direction installs the matching block of the full Gram.
#[test]
fn curvature_trial_installs_the_reduced_congruence_of_the_full_gram_2935() {
    let (term, _target, _rho) = curvature_fixture();
    let trial = 0.4_f64;
    let full = term.atoms[0]
        .geometry_plan()
        .expect("the fixture atom carries its geometry plan")
        .at_constant_curvature(trial)
        .and_then(|plan| plan.build_reference_penalty())
        .expect("full Gram at the trial curvature");
    let scale = full.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        scale > 1.0,
        "the full Gram must be materially nonzero (max |S| = {scale})"
    );
    let m = full.nrows();
    // Column 0 of the monomial tangent basis is the constant, the Dirichlet null
    // space. Retaining [1, t₁] keeps one penalized direction (the control arm);
    // retaining [1] keeps none.
    for retained in [2usize, 1] {
        let mut q = Array2::<f64>::zeros((m, retained));
        for col in 0..retained {
            q[[col, col]] = 1.0;
        }
        let mut atom = term.atoms[0].clone();
        atom.reduce_basis_to_subspace(&q)
            .expect("the reduction retains orthonormal columns of the basis");
        let prepared = atom
            .prepare_constant_curvature(trial)
            .unwrap_or_else(|err| panic!("retained {retained}: trial curvature refused: {err}"));
        atom.commit_prepared_constant_curvature(prepared);
        let expected = q.t().dot(&full).dot(&q);
        let installed = atom.smooth_penalty();
        let gap = installed
            .iter()
            .zip(expected.iter())
            .map(|(value, reference)| (value - reference).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "[#2935 reduced congruence] retained={retained} installed={installed:?} \
             expected={expected:?} gap={gap:.3e}"
        );
        assert_eq!(installed.dim(), (retained, retained));
        assert!(
            gap <= 1.0e-10 * scale,
            "retained {retained}: installed reduced Gram departs from QᵀS(κ)Q by {gap}"
        );
        assert!(
            atom.smooth_penalty_kappa_derivative()
                .expect("dS/dκ sits at the reduced basis width")
                .is_some_and(|derivative| derivative.dim() == (retained, retained)),
            "retained {retained}: the reduced dS/dκ must be installed at the reduced width"
        );
    }
}

/// An affine gauge re-expresses a curvature atom's decoder as `T·B`. Its `∂S/∂κ` must
/// move by the same congruence as `S`, so the κ energy channel `½λ<B, ∂S/∂κ B>` stays
/// the derivative of the same function energy. The reference is measured in the old
/// chart from the untouched geometry plan.
#[test]
fn affine_gauge_transports_the_curvature_derivative_with_the_gram_2935() {
    let (mut term, _target, rho) = curvature_fixture();
    let rho = rho
        .for_assignment(&term.assignment)
        .with_curvature(vec![(0, KAPPA)]);
    let lambda = rho.lambda_smooth_vec().expect("smoothing strengths");
    let flat = rho
        .kappa_flat_index(0)
        .expect("the curvature atom owns an outer coordinate");
    let old_decoder = term.atoms[0].decoder_coefficients().clone();
    let plan = term.atoms[0]
        .geometry_plan()
        .expect("the fixture atom carries its geometry plan")
        .clone();
    let energy = |decoder: &Array2<f64>, gram: &Array2<f64>| -> f64 {
        0.5 * lambda[0] * (decoder * &gram.dot(decoder)).sum()
    };
    let old_chart_energy = |kappa: f64| -> f64 {
        let gram = plan
            .at_constant_curvature(kappa)
            .and_then(|at| at.build_reference_penalty())
            .expect("plan Gram at the trial curvature");
        energy(&old_decoder, &gram)
    };
    let energy_before = energy(&old_decoder, term.atoms[0].smooth_penalty());
    term.canonicalize_atom_affine_gauge(0, None)
        .expect("the affine gauge runs on the atom");
    let new_decoder = term.atoms[0].decoder_coefficients().clone();
    let decoder_scale = old_decoder
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let decoder_moved = new_decoder
        .iter()
        .zip(old_decoder.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    let energy_after = energy(&new_decoder, term.atoms[0].smooth_penalty());
    let channel = term
        .decoder_smoothness_kappa_energy_derivatives(&rho, &lambda)
        .expect("κ energy channel after the gauge")
        .into_iter()
        .find(|(index, _)| *index == flat)
        .map(|(_, value)| value)
        .expect("the channel lands on the κ coordinate");
    let step = 1.0e-3_f64;
    let (fd, spread) = richardson(
        (old_chart_energy(KAPPA + step) - old_chart_energy(KAPPA - step)) / (2.0 * step),
        (old_chart_energy(KAPPA + 0.5 * step) - old_chart_energy(KAPPA - 0.5 * step)) / step,
    );
    println!(
        "[#2935 transport] decoder moved {decoder_moved:.3e} of {decoder_scale:.3e}; energy \
         {energy_before:.12e} -> {energy_after:.12e}; κ energy channel {channel:.12e} vs \
         old-chart difference {fd:.12e} (spread {spread:.3e})"
    );
    assert!(
        decoder_moved > 1.0e-3 * decoder_scale,
        "the gauge must actually re-express the decoder for the check to mean anything"
    );
    assert!(
        (energy_after - energy_before).abs() <= 1.0e-8 * energy_before.abs().max(1.0),
        "the transported Gram prices the same function energy ({energy_before} vs {energy_after})"
    );
    let tolerance = 10.0 * spread + 1.0e-6 * fd.abs().max(1.0);
    assert!(
        fd.abs() > 1.0e3 * tolerance,
        "the energy must move materially with κ ({fd} vs {tolerance:.3e})"
    );
    assert!(
        (channel - fd).abs() <= tolerance,
        "the κ energy channel {channel} after the gauge is not the energy's κ derivative {fd} \
         (|Δ| = {}, tolerance {tolerance})",
        (channel - fd).abs()
    );
}

/// A restore brings back `∂S/∂κ` and the geometry plan with `S`. Restoring a
/// full-width snapshot over a reduced atom must leave the κ channels at the restored
/// width and value.
#[test]
fn snapshot_restore_carries_the_curvature_derivative_2935() {
    let (mut term, _target, rho) = curvature_fixture();
    let rho = rho
        .for_assignment(&term.assignment)
        .with_curvature(vec![(0, KAPPA)]);
    let lambda = rho.lambda_smooth_vec().expect("smoothing strengths");
    let full_width = term.atoms[0].basis_size();
    let before = term
        .decoder_smoothness_kappa_energy_derivatives(&rho, &lambda)
        .expect("κ energy channel at the full width");
    let snapshot = term.snapshot_mutable_state();
    let mut q = Array2::<f64>::zeros((full_width, 2));
    q[[0, 0]] = 1.0;
    q[[1, 1]] = 1.0;
    term.atoms[0]
        .reduce_basis_to_subspace(&q)
        .expect("the reduction retains orthonormal columns of the basis");
    assert_eq!(term.atoms[0].basis_size(), 2, "the reduction narrowed the atom");
    term.restore_mutable_state(&snapshot)
        .expect("the full-width snapshot restores");
    let after = term
        .decoder_smoothness_kappa_energy_derivatives(&rho, &lambda)
        .unwrap_or_else(|err| panic!("the κ energy channel refused the restored atom: {err}"));
    println!(
        "[#2935 restore] width {full_width} -> 2 -> {}; κ energy channel {before:?} -> {after:?}",
        term.atoms[0].basis_size()
    );
    assert_eq!(
        term.atoms[0].basis_size(),
        full_width,
        "the restore brought back the full width"
    );
    assert_eq!(before.len(), 1, "one curvature coordinate");
    assert!(
        before[0].1.abs() > 1.0e-3,
        "the κ energy channel must be material ({})",
        before[0].1
    );
    assert!(
        after.len() == 1 && (after[0].1 - before[0].1).abs() <= 1.0e-12 * before[0].1.abs().max(1.0),
        "the restored κ energy channel {after:?} is not the snapshot state's {before:?}"
    );
}

/// A dictionary that carries both crosscoder block weights and a curvature
/// coordinate lays the block tail out BEFORE the curvature tail. At `λ_block = 1`
/// the block pricing leaves the target and the fitted state as they are, so the
/// κ entry of the dense gradient must be the one the plain dictionary reports.
/// Locating the block tail as an offset from the end of ρ handed the κ slot the
/// block's implicit right-hand side and the block slot the κ one.
#[test]
fn crosscoder_block_weights_leave_the_curvature_gradient_entry_in_place_2935() {
    let (term, target, rho) = curvature_fixture();
    let (state, anchor, _) = converged_anchor(term, &target, rho);
    let dense_gradient = |blocks: bool| -> (SaeManifoldRho, Array1<f64>) {
        let mut at = anchor.clone();
        if blocks {
            at.log_lambda_block = vec![0.0];
        }
        let objective =
            SaeManifoldOuterObjective::new(state.clone(), target.clone(), None, at, 0, 0.4, 1.0e-6, 1.0e-6);
        let mut objective = if blocks {
            objective
                .with_crosscoder_blocks(1, vec![2])
                .expect("one anchor column and one two-column output block")
        } else {
            objective
        };
        let at = objective.baseline_rho.clone();
        let evaluation = objective
            .evaluate_outer_criterion_route(&at, true, false)
            .expect("the dense route prices the state");
        let gradient = objective
            .analytic_gradient_for_outer_evaluation(&at, &evaluation)
            .expect("the dense route differentiates the state");
        (at, gradient)
    };
    let (plain_rho, plain) = dense_gradient(false);
    let (block_rho, priced) = dense_gradient(true);
    let plain_flat = plain_rho.kappa_flat_index(0).expect("curvature coordinate");
    let block_flat = block_rho.kappa_flat_index(0).expect("curvature coordinate");
    let block_range = block_rho.block_flat_range();
    println!(
        "[#2935 block tail] plain κ entry {:.12e}; priced κ entry {:.12e}; block range \
         {block_range:?}, κ at {block_flat}; priced gradient {priced:?}",
        plain[plain_flat], priced[block_flat]
    );
    assert_eq!(block_range, plain_flat..plain_flat + 1, "the block tail precedes κ");
    assert_eq!(block_flat, plain_flat + 1, "κ is the last coordinate");
    assert_eq!(priced.len(), plain.len() + 1);
    assert!(
        plain[plain_flat].abs() > 1.0e-3,
        "the κ entry must be material for the comparison to mean anything ({})",
        plain[plain_flat]
    );
    let tolerance = 1.0e-8 * plain[plain_flat].abs().max(1.0);
    assert!(
        (priced[block_flat] - plain[plain_flat]).abs() <= tolerance,
        "installing block pricing moved the κ entry from {} to {}",
        plain[plain_flat],
        priced[block_flat]
    );
    for coord in 0..plain_flat {
        assert!(
            (priced[coord] - plain[coord]).abs() <= 1.0e-8 * plain[coord].abs().max(1.0),
            "installing block pricing moved entry {coord} from {} to {}",
            plain[coord],
            priced[coord]
        );
    }
    assert!(priced[block_range.start].is_finite(), "the block entry is finite");
}
