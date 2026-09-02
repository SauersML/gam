//! #2762 — the span the likelihood is flat along must be MINIMIZED over, not
//! assumed flat for the posterior.
//!
//! `quotient_residual_norm_sq` used to project the chart-gauge orbit out of the
//! KKT residual alongside the decoder nulls, on the premise that the penalized
//! objective is flat along all of them. The orbit is an exact first-order
//! symmetry of the RECONSTRUCTION and not of the penalized objective — the ARD
//! prior on `t` and the smoothness prior on `β` are written on the chart
//! coordinates — so the premise was false wherever those priors are active, and
//! the failure was self-concealing: the solver can only descend the transverse
//! block, so the retained fraction tends to zero at any fixed point whether or
//! not it is stationary. #2720 removed the orbit from that projection; this
//! module owns the block that DESCENDS it instead.
//!
//! The measurement that forced [`SaeManifoldTerm::descend_gauge_orbit`] is on
//! that function's own doc. These are its properties, gated from three angles
//! that do not share a failure mode:
//!
//! 1. the basis the descent minimizes over CONTAINS the basis the projection
//!    removes, strictly (an identity and a containment, both checked as such);
//! 2. a call that commits nothing leaves the state bit-for-bit (so an inert
//!    call can never be the thing that moved a fit);
//! 3. on a state whose orbit carries live slope, the descent cashes a decrease
//!    the AMBIENT line search provably cannot — which is the whole defect,
//!    reproduced in miniature and without any reference to the solver that
//!    stalls on it.

use super::tests_outer_quasi_laplace_probe_budget_2080::two_circle_wide_target;
use super::*;
use crate::manifold::fit_drivers::JointFitTermination;
use ndarray::Array1;

/// The `#2080` two-circle fixture at a size that fits a unit test, with the
/// atoms deliberately left at their PCA seed — the state the inner solve starts
/// from, where the chart phase is whatever the seed happened to produce and the
/// priors have not yet been minimized over it.
fn seeded_two_circle_term(
    n: usize,
    p: usize,
    k: usize,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let z = two_circle_wide_target(n, p, 0.03);
    let (term, dispersion) =
        super::tests_outer_quasi_laplace_probe_budget_2080::two_circle_periodic_term(
            z.view(),
            k,
            2,
        );
    // The dispersion scaling is not optional decoration: without it the priors
    // are written at unit scale against a target whose seeded decoder is orders
    // larger, and the fixture's penalized objective lands at `2.35e7` with
    // `‖g‖ = 2.85e5` — a state in which every gauge direction is a near-null
    // eigenvector and nothing about the orbit is representative. This is the
    // same construction `#2080`'s own fixture uses.
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![ndarray::array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    (term, z, rho)
}

/// The joint `[g_t; g_β]` residual in the dense layout the gauge basis is
/// written in.
fn dense_gradient(term: &mut SaeManifoldTerm, z: &Array2<f64>, rho: &SaeManifoldRho) -> Array1<f64> {
    let system = term
        .assemble_arrow_schur(z.view(), rho, None)
        .expect("the seeded two-circle fixture assembles");
    let n = term.n_obs();
    let q = term.assignment.row_block_dim();
    let dense_len = n * q;
    let mut gradient = Array1::<f64>::zeros(dense_len + system.gb.len());
    for (row_index, row) in system.rows.iter().enumerate() {
        let base = system.row_offsets[row_index];
        for (axis, &value) in row.gt.iter().enumerate() {
            gradient[base + axis] = value;
        }
    }
    for (index, &value) in system.gb.iter().enumerate() {
        gradient[dense_len + index] = value;
    }
    gradient
}

/// ANGLE 1 — the projection removes EXACTLY the posterior-null span, and the
/// descent block strictly CONTAINS it.
///
/// `posterior_null_quotient_basis` was factored out of
/// `quotient_residual_norm_sq`, which had interleaved its Gram--Schmidt with the
/// projection. This pins the halves of that refactor as an IDENTITY rather than
/// as a code reading: the returned vectors are orthonormal, and the shipped
/// quotient norm equals `‖r‖² − Σᵢ (r·vᵢ)²` on a residual that has support on
/// every block.
///
/// #2720 split the one span into two, and the second clause here is that split
/// stated as a containment: every direction the gate REMOVES is a direction the
/// descent MINIMIZES over, but not conversely — the chart orbit is descended and
/// not removed, because the penalized objective is not flat along it. A refactor
/// that re-merges them fails this test on whichever side it collapses.
#[test]
fn quotient_span_is_orthonormal_reproduces_the_projection_and_sits_inside_the_descent_block_2762() {
    let k = 2usize;
    let (mut term, z, rho) = seeded_two_circle_term(36, 12, k);
    let lambda_smooth = rho
        .lambda_smooth_vec()
        .expect("the fixture rho carries one smoothing block per atom");
    // Assemble once so the term holds the row layout the basis is written
    // against, exactly as every production caller does.
    let gradient = dense_gradient(&mut term, &z, &rho);
    let basis = term
        .posterior_null_quotient_basis(&lambda_smooth)
        .expect("the seeded fixture has a well-defined posterior-null span");
    let descent_block = term
        .likelihood_flat_block_basis(&lambda_smooth)
        .expect("the seeded fixture has a well-defined descent block");
    assert!(
        !descent_block.is_empty(),
        "K={k} periodic atoms must contribute at least one chart-gauge direction to the \
         descent block"
    );
    // Containment, checked as a projection identity rather than by comparing
    // vectors: each quotient direction must be reproduced by its own components
    // in the descent block's orthonormal frame.
    for (index, vector) in basis.iter().enumerate() {
        let mut residual = vector.clone();
        for other in &descent_block {
            let coeff = residual.dot(other);
            for i in 0..residual.len() {
                residual[i] -= coeff * other[i];
            }
        }
        let leak = residual.dot(&residual).sqrt();
        assert!(
            leak <= 1.0e-9,
            "quotient direction {index} leaves the descent block by {leak:.6e}; every direction \
             removed from a convergence measure must be one the descent block can act on, or \
             there is a direction the gate ignores and no mover reduces"
        );
    }
    assert!(
        descent_block.len() > basis.len(),
        "the descent block ({}) must be STRICTLY wider than the quotient span ({}) on a \
         periodic fixture: the chart orbit is descended and deliberately not removed (#2720)",
        descent_block.len(),
        basis.len(),
    );

    for (i, left) in basis.iter().enumerate() {
        assert_eq!(
            left.len(),
            gradient.len(),
            "gauge direction {i} must live in the same joint layout as the residual"
        );
        for (j, right) in basis.iter().enumerate() {
            let inner = left.dot(right);
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (inner - expected).abs() < 1.0e-10,
                "gauge basis must be orthonormal: ⟨v{i}, v{j}⟩ = {inner}"
            );
        }
    }

    // A residual with support on every block, built without randomness so the
    // failure is reproducible.
    let residual: Array1<f64> = (0..gradient.len())
        .map(|index| ((index % 7) as f64 - 3.0) * 0.25 + 0.5)
        .collect::<Vec<_>>()
        .into();
    let shipped = term
        .quotient_residual_norm_sq(residual.clone(), &lambda_smooth)
        .expect("the quotient of a finite residual is finite");
    let mut from_basis = residual.dot(&residual);
    for vector in &basis {
        let coeff = residual.dot(vector);
        from_basis -= coeff * coeff;
    }
    assert!(
        (shipped - from_basis).abs() <= 1.0e-9 * (1.0 + from_basis.abs()),
        "the shipped quotient must equal ‖r‖² − Σ(r·vᵢ)² over the published basis: \
         {shipped} vs {from_basis}"
    );
}

/// ANGLE 2 — an inert call is bit-for-bit inert.
///
/// The descent runs on the inner solve's no-strict-decrease path, where the
/// evidence lane's idempotence certificate requires that a pass which reports no
/// move genuinely recurred its entry state. A restore that is merely "close"
/// would make every such certificate a lie. Driving `max_rounds = 0` exercises
/// the same entry/exit contract with the commit path removed, so the assertion
/// is about the plumbing rather than about whether this particular fixture
/// happens to have slope.
#[test]
fn gauge_orbit_descent_commits_nothing_and_moves_nothing_at_zero_rounds_2762() {
    let k = 2usize;
    let (mut term, z, rho) = seeded_two_circle_term(36, 12, k);
    let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
    let before = term.snapshot_mutable_state();
    let objective_before = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("the seeded fixture has a finite penalized objective");

    let outcome = term
        .descend_gauge_orbit(z.view(), &rho, None, &lambda_smooth, 0)
        .expect("a zero-round descent cannot fail");
    assert_eq!(outcome.rounds, 0);
    assert_eq!(outcome.evaluations, 0);
    assert!(!outcome.moved());

    let after = term.snapshot_mutable_state();
    assert_eq!(
        before.logits, after.logits,
        "an inert gauge descent must leave the assignment logits bit-identical"
    );
    for (atom_index, (left, right)) in before.atoms.iter().zip(after.atoms.iter()).enumerate() {
        assert_eq!(
            left.decoder_coefficients, right.decoder_coefficients,
            "an inert gauge descent must leave atom {atom_index}'s decoder bit-identical"
        );
    }
    let objective_after = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("the objective is still finite");
    assert_eq!(
        objective_before, objective_after,
        "an inert gauge descent must leave the objective bit-identical"
    );
}

/// #2762 PROBE — the removed span's whole first-order state at the seed and
/// after a planted displacement inside it, plus the layout the step application
/// actually uses. Printed, not asserted: this exists so the planted-witness test
/// beside it is premised on measurements rather than on what the layout comment
/// says.
#[test]
fn zz2762_removed_span_slope_and_layout_census() {
    let k = 2usize;
    let (mut term, z, rho) = seeded_two_circle_term(48, 16, k);
    let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
    let n = term.n_obs();
    let q = term.assignment.row_block_dim();
    let dense_len = n * q;

    let system = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("assembles");
    let compact_total: usize = system.row_dims.iter().sum();
    let border_dim = term.factored_border_dim();
    eprintln!(
        "[zz2762] n={n} q={q} dense_len={dense_len} compact_total={compact_total} \
         border_dim={border_dim} gb_len={} frames_active={}",
        system.gb.len(),
        term.last_frames_active,
    );
    drop(system);

    let seed_objective = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite");
    let gradient = dense_gradient(&mut term, &z, &rho);
    let basis = term
        .likelihood_flat_block_basis(&lambda_smooth)
        .expect("descent block");
    eprintln!(
        "[zz2762] seed objective={seed_objective:.9e} ‖g‖={:.6e} span_dim={}",
        gradient.dot(&gradient).sqrt(),
        basis.len(),
    );
    for (index, vector) in basis.iter().enumerate() {
        let coord_mass = vector
            .slice(ndarray::s![..dense_len])
            .iter()
            .map(|v| v * v)
            .sum::<f64>();
        eprintln!(
            "[zz2762]   v{index}: gᵀv={:.6e}  coord_mass={coord_mass:.6e}",
            gradient.dot(vector),
        );
    }

    // Is the plant's first-order effect consistent with its objective effect?
    // A displacement `α v` on a direction of curvature `c` must raise `f` by
    // `½cα²` AND leave slope `cα` behind. If the second does not follow the
    // first, the plant is not moving the state the gradient is assembled from.
    let plant = basis[0].clone();
    let restore = term.snapshot_mutable_state();
    for exponent in 0..5 {
        let plant_alpha = 10.0_f64.powi(exponent);
        term.apply_newton_step(
            plant.slice(ndarray::s![..dense_len]),
            plant.slice(ndarray::s![dense_len..]),
            plant_alpha,
        )
        .expect("plant applies");
        let value = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("finite");
        let after = dense_gradient(&mut term, &z, &rho);
        let recomputed = term
            .likelihood_flat_block_basis(&lambda_smooth)
            .expect("descent block at the planted state");
        let mut recomputed_max = 0.0_f64;
        let mut recomputed_at = usize::MAX;
        let mut best_overlap = 0.0_f64;
        for (index, vector) in recomputed.iter().enumerate() {
            let slope = after.dot(vector).abs();
            if slope > recomputed_max {
                recomputed_max = slope;
                recomputed_at = index;
            }
            best_overlap = best_overlap.max(plant.dot(vector).abs());
        }
        eprintln!(
            "[zz2762] plant α={plant_alpha:.1e}: rise={:.6e}  gᵀplant={:.6e}  ‖g‖={:.6e}  \
             recomputed_dim={}  max|gᵀv|={recomputed_max:.6e} at v{recomputed_at}  \
             max|plant·v|={best_overlap:.6e}",
            value - seed_objective,
            after.dot(&plant),
            after.dot(&after).sqrt(),
            recomputed.len(),
        );
        term.restore_mutable_state(&restore).expect("restores");
    }

    term.apply_newton_step(
        plant.slice(ndarray::s![..dense_len]),
        plant.slice(ndarray::s![dense_len..]),
        1.0,
    )
    .expect("plant applies");
    let planted_objective = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite");
    let planted_gradient = dense_gradient(&mut term, &z, &rho);
    let planted_basis = term
        .likelihood_flat_block_basis(&lambda_smooth)
        .expect("descent block");
    eprintln!(
        "[zz2762] planted objective={planted_objective:.9e} (rise {:.6e}) ‖g‖={:.6e} span_dim={}",
        planted_objective - seed_objective,
        planted_gradient.dot(&planted_gradient).sqrt(),
        planted_basis.len(),
    );
    for (index, vector) in planted_basis.iter().enumerate() {
        eprintln!(
            "[zz2762]   v{index}: gᵀv={:.6e}  overlap_with_plant={:.6e}",
            planted_gradient.dot(vector),
            plant.dot(vector),
        );
    }

    // The sweep the descent runs, printed rather than summarized.
    let mut direction = Array1::<f64>::zeros(planted_gradient.len());
    for vector in &planted_basis {
        let coeff = planted_gradient.dot(vector);
        for index in 0..direction.len() {
            direction[index] -= coeff * vector[index];
        }
    }
    let slope = direction.dot(&direction).sqrt();
    for value in direction.iter_mut() {
        *value /= slope;
    }
    let snapshot = term.snapshot_mutable_state();
    eprintln!("[zz2762] projected slope ‖Π_V g‖={slope:.6e}; sweep along −Π_V g:");
    let mut alpha = term.inner_iterate_scale();
    let floor = 1.0e-8 * (1.0 + planted_objective.abs());
    while alpha >= floor / slope {
        let applied = term
            .apply_newton_step(
                direction.slice(ndarray::s![..dense_len]),
                direction.slice(ndarray::s![dense_len..]),
                alpha,
            )
            .is_ok();
        let value = if applied {
            term.penalized_objective_total(z.view(), &rho, None, 1.0)
                .unwrap_or(f64::INFINITY)
        } else {
            f64::INFINITY
        };
        eprintln!(
            "[zz2762]   α={alpha:.6e}  f={value:.9e}  drop={:.6e}",
            planted_objective - value
        );
        term.restore_mutable_state(&snapshot).expect("restores");
        alpha *= 0.5;
    }
    eprintln!("[zz2762] material floor={floor:.6e}");
}

/// The largest `|gᵀvᵢ|` over the removed span at the term's current state.
fn removed_span_max_slope(
    term: &mut SaeManifoldTerm,
    z: &Array2<f64>,
    rho: &SaeManifoldRho,
    lambda_smooth: &[f64],
) -> f64 {
    let gradient = dense_gradient(term, z, rho);
    let basis = term
        .likelihood_flat_block_basis(lambda_smooth)
        .expect("the descent block is well-defined");
    let mut max_slope = 0.0_f64;
    for vector in &basis {
        if vector.len() == gradient.len() {
            max_slope = max_slope.max(gradient.dot(vector).abs());
        }
    }
    max_slope
}

/// ANGLE 3 — the contract every caller relies on, on a state the descent has
/// something to say about.
///
/// Two clauses, and they are the ones a wrong refactor breaks:
///
/// * MONOTONE — the objective after is never above the objective before. The
///   descent runs on the inner solve's no-strict-decrease path, where an
///   increase would silently un-do the fit the Armijo and proximal gates just
///   protected.
/// * HONEST — the `objective_decrease` it REPORTS is the decrease it MADE. The
///   caller turns that number into `state_moved`, which feeds the evidence
///   lane's idempotence certificate, so a report that overstates its move is a
///   certificate that lies.
///
/// Both hold whether or not this fixture's span happens to carry a committable
/// decrease, which is deliberate: the census beside this shows that whether a
/// given seed does is a property of the seed.
#[test]
fn gauge_orbit_descent_is_monotone_and_reports_the_decrease_it_made_2762() {
    let k = 2usize;
    let (mut term, z, rho) = seeded_two_circle_term(48, 16, k);
    let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
    let before = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the seed");

    let outcome = term
        .descend_gauge_orbit(z.view(), &rho, None, &lambda_smooth, 8)
        .expect("the descent runs on the seeded fixture");
    let after = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective after the descent");

    // The bar is the objective's OWN stated resolution, not exact equality, and
    // for a measured reason: `descend_gauge_orbit` assembles once per round, and
    // an assembly re-freezes the decoder-repulsion / amplitude / separation
    // gates that `penalized_objective_total` then reads. That is the
    // "per-assembly gate-freeze drift" the evidence lane's material floor exists
    // to reject, and it is pre-existing — the shipped Armijo line search
    // apply/evaluate/restore cycle carries it too. Measured here at one ulp
    // (`2.3483247973234504e7 → 2.3483247973236360e7`, relative 8e-17). What must
    // never happen is a RESOLVABLE rise, which is what this asserts.
    let resolution = 1.0e-8 * (1.0 + before.abs());
    assert!(
        after <= before + resolution,
        "the gauge descent must never raise the penalized objective by a resolvable amount: \
         {before} → {after} (resolution {resolution})"
    );
    assert!(
        (before - after - outcome.objective_decrease).abs()
            <= resolution + 1.0e-6 * outcome.objective_decrease.abs(),
        "the reported decrease must be the measured one: reported {}, measured {}",
        outcome.objective_decrease,
        before - after,
    );
    if outcome.moved() {
        assert!(
            before - after > resolution,
            "a descent that reports a move must have made a RESOLVABLE one: \
             {before} → {after} over {} round(s)",
            outcome.rounds,
        );
    }
}

/// ANGLE 4 — the refine loop's terminal mover and terminal restore must have
/// ONE state authority.
///
/// `terminal_exact_newton_polish` may save a smaller-decrement state and then
/// leave a different excursion live.  The historical third #2762 call descended
/// that excursion; the refusal immediately restored the saved state and threw
/// the committed descent away.  Plant a material likelihood-flat displacement
/// in the saved state while leaving the live excursion at the seed.  The
/// terminal wrapper must restore and descend the planted state, report that
/// state's measured decrease, and consume its now-stale certificate.
#[test]
fn terminal_gauge_descent_moves_the_state_that_best_seen_would_restore_2762() {
    let k = 2usize;
    let (mut term, z, rho) = seeded_two_circle_term(48, 16, k);
    let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
    let n = term.n_obs();
    let q = term.assignment.row_block_dim();
    let dense_len = n * q;

    let seed_objective = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the live excursion");
    let seed_state = term.snapshot_mutable_state();
    let gradient = dense_gradient(&mut term, &z, &rho);
    let basis = term
        .likelihood_flat_block_basis(&lambda_smooth)
        .expect("the seeded fixture has a likelihood-flat block");

    // Plant in the ASCENT direction of the same projected gradient the mover
    // descends.  The search endpoints are the production endpoints: the
    // iterate's own scale and the first-order material-resolution boundary.
    let mut ascent = Array1::<f64>::zeros(gradient.len());
    for vector in &basis {
        let coefficient = gradient.dot(vector);
        for index in 0..ascent.len() {
            ascent[index] += coefficient * vector[index];
        }
    }
    let slope = ascent.dot(&ascent).sqrt();
    assert!(
        slope.is_finite() && slope > 0.0,
        "fixture needs live flat-block slope"
    );
    for value in ascent.iter_mut() {
        *value /= slope;
    }
    let material_floor = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * (1.0 + seed_objective.abs());
    let near_alpha = material_floor / slope;
    let mut alpha = term.inner_iterate_scale();
    let mut planted = None;
    while alpha >= near_alpha {
        let applied = term
            .apply_newton_step(
                ascent.slice(ndarray::s![..dense_len]),
                ascent.slice(ndarray::s![dense_len..]),
                alpha,
            )
            .is_ok();
        if applied
            && let Ok(objective) = term.penalized_objective_total(z.view(), &rho, None, 1.0)
            && objective.is_finite()
            && planted.as_ref().is_none_or(|(best, _)| objective > *best)
        {
            planted = Some((objective, term.snapshot_mutable_state()));
        }
        term.restore_mutable_state(&seed_state)
            .expect("the live excursion restores");
        alpha *= 0.5;
    }
    let (saved_objective, saved_state) = planted.expect("the production sweep has a finite plant");
    assert!(
        saved_objective - seed_objective > material_floor,
        "fixture must admit a material ascent inside the production sweep: \
         seed={seed_objective:.9e}, saved={saved_objective:.9e}, floor={material_floor:.6e}"
    );

    // `self` is the live excursion; `best_seen` names a different terminal
    // state.  This is the exact state split at the objective-stall refusal.
    let mut best_seen = Some((0.0, slope, saved_state));
    let outcome = term
        .descend_gauge_orbit_at_terminal_candidate(
            z.view(),
            &rho,
            None,
            &lambda_smooth,
            &mut best_seen,
            8,
        )
        .expect("terminal candidate descent runs");
    let after = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective after terminal descent");
    let resolution = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * (1.0 + saved_objective.abs());

    assert!(
        outcome.moved(),
        "the planted best-seen state has material flat-block descent"
    );
    assert!(
        best_seen.is_none(),
        "a committed move must consume the certificate for the pre-move state"
    );
    assert!(
        saved_objective - after > resolution,
        "the mover must descend the saved terminal candidate, not the live excursion: \
         saved={saved_objective:.9e}, live={seed_objective:.9e}, after={after:.9e}"
    );
    assert!(
        (saved_objective - after - outcome.objective_decrease).abs()
            <= resolution + 1.0e-6 * outcome.objective_decrease.abs(),
        "the reported decrease must be measured from best_seen: reported {}, measured {}",
        outcome.objective_decrease,
        saved_objective - after,
    );
}

/// ANGLE 5 — THE SOLVER POST-CONDITION, which is what #2762 is actually about.
///
/// The defect was never "the descent is missing" in the abstract; it was that
/// the inner fit could conclude `NoStrictDecrease` or `Heuristic` — the exits
/// that tell the refine loop this state is a fixed point of the inner map —
/// while a material decrease was still sitting in the span the convergence
/// measure removes. Measured at the `zz2015` refusal: `1.090e-1` of objective,
/// 559 times the `1e-8` relative resolution that same loop calls "no meaningful
/// change".
///
/// So the invariant is a POST-condition of the inner fit and is checked as one.
/// It is CONDITIONAL on the exit actually claiming a fixed point, and that is
/// not hedging — an exit on budget claims nothing, and asserting the invariant
/// there would be asserting that a truncated optimization is optimal.
///
/// MEASURED SCOPE, so nobody reads more into a green than is there: this
/// fixture's evidence-lane fit is still strictly descending at 4096 iterations
/// (`IterationGrantExhausted`, 19.1 s), so on THIS fixture the fixed-point
/// clause is a forward guard rather than a live one, and what it establishes
/// today is the unconditional half — the fit descends, and the block's verdict
/// at the exit is monotone and honestly reported. The live evidence for the
/// conditional half is `zz2015_tiny_inner_crawl_terminates`, where the pre-fix
/// engine took the `objective stalled for 3 consecutive refine rounds` exit with
/// that `1.090e-1` unspent.
#[test]
fn the_inner_fit_never_exits_with_material_decrease_left_in_the_removed_span_2762() {
    let k = 2usize;
    let (mut term, z, mut rho) = seeded_two_circle_term(48, 16, k);
    let lambda_smooth = rho.lambda_smooth_vec().expect("one block per atom");
    let entry = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the seed");
    let outcome_fit = term
        .run_joint_fit_arrow_schur_with_termination_policy(
            z.view(),
            &mut rho,
            None,
            256,
            0.05,
            1.0e-6,
            1.0e-6,
            // The EVIDENCE lane, which is the one #2762 is about: its
            // `NoStrictDecrease` exit becomes the refine loop's
            // `criterion_fixed_point`, so a fixed-point claim that is false
            // about the removed span turns into a refusal.
            false,
        )
        .expect("the seeded two-circle fixture fits");

    let settled = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the inner fit's exit");
    assert!(
        settled < entry,
        "the fit must strictly descend for its exit state to be worth checking: \
         {entry} → {settled}"
    );

    let residual_slope = removed_span_max_slope(&mut term, &z, &rho, &lambda_smooth);
    let outcome = term
        .descend_gauge_orbit(z.view(), &rho, None, &lambda_smooth, 8)
        .expect("the descent runs at the inner fit's exit");
    let after = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective after the exit-state descent");
    let resolution = 1.0e-8 * (1.0 + settled.abs());
    assert!(
        after <= settled + resolution,
        "the exit-state descent must not raise the objective by a resolvable amount: \
         {settled} → {after}"
    );
    // The decrease telescopes against the entry value the descent measured
    // under ITS gate: `settled` above was evaluated under the gate the inner
    // fit's last assembly left, and the two differ at the gate-shift scale
    // (`penalized_objective_is_a_pure_function_of_the_mutable_state_2762`).
    let entry = outcome
        .entry_objective
        .expect("a descent that moved evaluated its entry objective");
    assert!(
        (entry - settled).abs() <= resolution + 1.0e-5 * (1.0 + settled.abs()),
        "the descent's entry value must be the caller's value up to the gate shift: \
         entry {entry:.12e}, settled {settled:.12e}"
    );
    assert!(
        (entry - after - outcome.objective_decrease).abs()
            <= resolution + 1.0e-6 * outcome.objective_decrease.abs(),
        "the exit-state descent must report the decrease it made: reported {}, measured {}",
        outcome.objective_decrease,
        entry - after,
    );

    if matches!(
        outcome_fit.termination,
        JointFitTermination::Heuristic
            | JointFitTermination::NoStrictDecrease
            | JointFitTermination::Frozen
    ) {
        assert!(
            !outcome.moved(),
            "the inner fit exited {:?} leaving {:.6e} of penalized objective in the span its \
             own convergence measure removes ({} round(s), span dim {}, \
             maxᵢ|gᵀvᵢ|={residual_slope:.6e}, objective {settled:.9e}) — that is the #2762 \
             defect, not a tolerance question",
            outcome_fit.termination,
            outcome.objective_decrease,
            outcome.rounds,
            outcome.dimension,
        );
    }
}

/// gam#2762 — the decrease `descend_gauge_orbit` reports is a SUM of per-round
/// `base − committed` re-evaluations, and the fixtures above grade it against
/// one before/after pair. That only telescopes if evaluating the objective is a
/// pure function of the mutable state: the same state must give the same value
/// whether it is reached fresh, evaluated twice in a row, or restored from a
/// snapshot after a speculative trial (the descent's own round structure). At
/// `07513ef38` the exit-state fixture missed by `4.5e-7` on a decrease of
/// `4.46e-2`; this pins which of those three paths, if any, moves the value.
#[test]
fn penalized_objective_is_a_pure_function_of_the_mutable_state_2762() {
    let (mut term, z, rho) = seeded_two_circle_term(48, 16, 2);
    let first = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the seed");
    let second = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective at the seed, second evaluation");
    assert_eq!(
        first.to_bits(),
        second.to_bits(),
        "two consecutive evaluations of one state differ: {first:.17e} vs {second:.17e}"
    );
    let snapshot = term.snapshot_mutable_state();
    term.restore_mutable_state(&snapshot)
        .expect("restoring the state just snapshotted");
    let restored = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective after a snapshot/restore round trip");
    assert_eq!(
        first.to_bits(),
        restored.to_bits(),
        "a snapshot/restore round trip moves the objective of an unchanged state: \
         {first:.17e} vs {restored:.17e} (difference {:.3e})",
        restored - first
    );
    // The descent assembles the arrow-Schur system before it evaluates its
    // per-round base objective, and assembly refreshes the barrier/repulsion
    // gates the objective reads (`refresh_*_gate` in penalties.rs). So the
    // value is a function of (state, frozen gate), not of the state alone:
    // measured `2.34832479732345045e7 -> 2.34832479732363597e7` (498 ulps)
    // across one assembly at the seed. What the descent's round accounting
    // relies on is the weaker, true invariant pinned here: refreshing the
    // gates at an unchanged state is idempotent, so two assemblies in a row
    // give one value.
    let system = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("the arrow-Schur system assembles at the seed");
    assert_eq!(system.rows.len(), 48, "one arrow row per observation");
    drop(system);
    let after_assembly = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective after assembling the arrow-Schur system");
    println!(
        "2762-gate-shift: seed objective {first:.17e} -> {after_assembly:.17e} across one assembly \
         (difference {:.3e})",
        after_assembly - first
    );
    let system = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("the arrow-Schur system assembles again at the seed");
    assert_eq!(system.rows.len(), 48, "one arrow row per observation");
    drop(system);
    let after_second_assembly = term
        .penalized_objective_total(z.view(), &rho, None, 1.0)
        .expect("finite objective after a second assembly");
    assert_eq!(
        after_assembly.to_bits(),
        after_second_assembly.to_bits(),
        "refreshing the gates at an unchanged state must be idempotent: \
         {after_assembly:.17e} vs {after_second_assembly:.17e}"
    );
}
