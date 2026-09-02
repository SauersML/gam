//! Streaming/matrix-free evidence route — outer-gradient lane parity and the
//! large-K/wide-border completion contract (W11).
//!
//! Two properties are pinned here that the pre-existing #1026 streaming-cache
//! test (`tests_streaming_efs_cache_1026`) did NOT cover:
//!
//!  1. **Outer-gradient parity.** The #1026 test proved the cache returned by
//!     `penalized_quasi_laplace_criterion_streaming_exact_with_cache` is a drop-in for the EFS
//!     consumers (`ard_inverse_traces` / `reconstruction_dispersion`). But the
//!     ANALYTIC OUTER ρ-GRADIENT lane (`outer_gradient_arrow_solver` →
//!     `analytic_outer_rho_gradient_components`) also reads the returned cache,
//!     and it is that lane the seed startup-validation and the small-BFGS regime
//!     consume. This test forces the streaming route at a size where the dense
//!     path also fits and asserts the outer gradient assembled off the streaming
//!     cache is bit-identical to the one assembled off the dense cache — i.e. the
//!     streaming cache is a faithful drop-in for the gradient lane, not just the
//!     EFS traces.
//!
//!  2. **Large-K/wide-border completion.** A whitened (`WhitenedStructured` row
//!     metric) fit at K=32, p=128, n=500 — the composition regime whose predicted
//!     dense evidence cache (`N·q·border_dim`, q=K(1+d), border_dim=Σ_k M_k·p)
//!     exceeds the in-core budget — must ROUTE to the streaming criterion and
//!     COMPLETE with a finite penalized quasi-Laplace value rather than hard-erroring. We pin both
//!     halves deterministically: (a) the memory planner refuses the dense direct
//!     plan at this shape but admits the matrix-free plan, so the auto-router
//!     selects streaming; and (b) the streaming value path itself returns a finite
//!     criterion on the whitened term.

use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use approx::assert_abs_diff_eq;
use gam_solve::rho_optimizer::{FixedPointCoordinateCertificate, OuterObjective};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

use super::tests::{
    PlantedCircleAssignmentMode, TestPeriodicEvaluator, periodic_basis, planted_circle_embedded,
    planted_circle_seed_term, small_two_atom_periodic_term,
};
use std::sync::Arc;

// ---- Large-K / wide-border whitened completion ------------------------------

/// A K-atom periodic term over `(n, p)` with a softmax assignment (non-ordered Beta--Bernoulli, so the
/// streaming reduced-Schur log-det has a matrix-free route). Each atom carries the
/// `TestPeriodicEvaluator` — REQUIRED by the streaming path, which re-evaluates
/// Φ(t) per chunk via `materialize_chunk` — and a distinct nonzero decoder so the
/// reconstruction (and hence the residual the row metric whitens) is genuinely
/// nonzero. Mirrors the `small_two_atom_periodic_term` fixture the parity test
/// above uses, generalized to K atoms and a `p`-channel decoder.
fn build_softmax_term(n: usize, p: usize, k: usize) -> SaeManifoldTerm {
    let coord_cols: Vec<Array2<f64>> = (0..k)
        .map(|i| {
            Array2::<f64>::from_shape_fn((n, 1), |(r, _)| {
                (0.03 + 0.11 * i as f64 + 0.017 * (i + 1) as f64 * r as f64).rem_euclid(1.0)
            })
        })
        .collect();
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|i| {
            let (phi, jet) = periodic_basis(&coord_cols[i]);
            let f = (i as f64) + 1.0;
            // Periodic basis width is 3 ([1, sin, cos]); decoder is (3, p).
            let decoder = Array2::<f64>::from_shape_fn((3, p), |(m, c)| {
                0.1 * f * ((m + 1) as f64) - 0.05 * (c as f64) + 0.02 * f
            });
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("atom{i}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .expect("the fixture's basis, decoder and Gram blocks agree in dimension")
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        })
        .collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let logits =
        Array2::<f64>::from_shape_fn((n, k), |(r, c)| 0.3 * (c as f64) - 0.1 * (r as f64) + 0.2);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_cols,
        manifolds,
        AssignmentMode::softmax(0.8),
    )
    .expect("the fixture's logits, coordinate blocks and manifolds agree in length");
    SaeManifoldTerm::new(atoms, assignment)
        .expect("the fixture's atoms and assignment describe the same latent blocks")
}

/// At K=32, p=128 the width-2 euclidean border is `border_dim = Σ_k M_k·p =
/// 64·128 = 8192`, so the dense direct evidence peak (`N·q·border_dim`,
/// q=K(1+d)=64) is ≈2.6 GB and exceeds a representative 2 GiB in-core budget,
/// while the matrix-free plan's peak (chunk window + sparse row-cross + border
/// vector workspace) stays in the tens of MB. The planner must therefore REFUSE
/// the dense direct plan (routing the criterion to streaming) while ADMITTING the
/// matrix-free plan — the exact regime the streaming route was built for.
#[test]
fn wide_border_routes_to_streaming_with_complete_analytic_gradient_certificate() {
    let (n, p, k, d_max) = (500usize, 128usize, 32usize, 1usize);
    let total_basis = 2 * k; // width-2 euclidean basis per atom.
    let border_dim = total_basis * p;
    let budget = 2 * 1024 * 1024 * 1024usize; // 2 GiB representative in-core budget.
    let host_available = 8 * 1024 * 1024 * 1024usize;
    let chunk_window = SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE;
    let plan = sae_streaming_plan_from_budget(
        n,
        total_basis,
        k,
        d_max,
        border_dim,
        budget,
        chunk_window,
        host_available,
    );
    assert!(
        !plan.direct_admitted,
        "the dense direct evidence peak ({} bytes) must exceed the 2 GiB budget so the \
         criterion routes to streaming",
        plan.estimated_direct_peak_bytes
    );
    assert!(
        plan.matrix_free_admitted,
        "the matrix-free plan ({} bytes) must be admitted so the fit has a route",
        plan.estimated_matrix_free_peak_bytes
    );
    assert!(
        plan.streaming,
        "a non-direct-admitted plan must select streaming"
    );
    assert_eq!(
        sae_outer_gradient_capability(),
        Derivative::Analytic,
        "matrix-free SAE must advertise the complete rational-value/single-adjoint gradient"
    );
    let dense_plan = sae_streaming_plan_from_budget(
        n,
        total_basis,
        k,
        d_max,
        border_dim,
        usize::MAX,
        chunk_window,
        usize::MAX,
    );
    assert!(dense_plan.direct_admitted);
    assert_eq!(
        sae_outer_gradient_capability(),
        Derivative::Analytic,
        "dense SAE retains its exact joint-Hessian IFT gradient"
    );
    let (_representative_term, _, representative_rho) = small_two_atom_periodic_term();
    assert_eq!(
        assignment_strength_gradient_coordinate(&representative_rho),
        representative_rho.sparse_flat_index(),
        "every active assignment strength must enter Hybrid-EFS's \
         exact-gradient block; the outer-plan crossover decides whether that block \
         is consumed, not whether the coordinate has an analytic root"
    );
    // The admission gate must accept the plan (no 'working set exceeds budget'
    // hard error) precisely because the matrix-free lane is admitted.
    plan.admitted_or_error(n, border_dim, k)
        .expect("matrix-free-admitted plan must not hard-error at the admission gate");
}

/// Production-objective routing pin for #2080(A). Force the small, exactly
/// checkable planted-circle objective through the same streaming artifact used
/// when the memory planner rejects direct evidence, then compare its returned
/// `(value, gradient)` with the ordinary dense production evaluation. At this
/// tiny border the derived-rank surrogate captures the whole reduced space, so
/// the comparison is an exact-route parity check rather than a stochastic error
/// budget. Calling the objective helper (not the component assembler directly)
/// prevents the production branch from regressing to a zero gradient while the
/// lower-level parity test remains green.
#[test]
fn production_objective_forced_streaming_value_gradient_matches_dense() {
    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    let mut dense = SaeManifoldOuterObjective::new(
        term.clone(),
        target.clone(),
        None,
        seed_rho.clone(),
        40,
        1.0,
        1.0e-6,
        1.0e-6,
    );
    let mut streaming =
        SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6);

    // Construction binds the outer-coordinate layout to the assignment family.
    // In particular K=1 Softmax has no entropy-strength coordinate, so the
    // unbound constructor seed has three coordinates while each objective owns
    // the correct two-coordinate layout.  Drive each route from that owned
    // authority; retaining the pre-construction seed here would test a phantom
    // parameter that the production objective correctly refuses.
    let rho_flat = dense.baseline_rho.to_flat();
    let rho = streaming
        .baseline_rho
        .from_flat(rho_flat.view())
        .expect("dense and streaming objectives must own the same typed rho layout");
    assert_eq!(
        rho_flat.len(),
        2,
        "K=1 Softmax has no assignment-strength coordinate"
    );

    let dense_eval =
        OuterObjective::eval(&mut dense, &rho_flat).expect("dense production value+gradient");
    let streaming_artifact = streaming
        .evaluate_outer_criterion_route(&rho, false, false)
        .expect("forced streaming production artifact");
    let streaming_gradient = streaming
        .analytic_gradient_for_outer_evaluation(&rho, &streaming_artifact)
        .expect("forced streaming production gradient");
    let streaming_eval = OuterEval {
        cost: streaming_artifact.cost,
        gradient: streaming_gradient,
        hessian: HessianValue::Unavailable,
        inner_beta_hint: Some(streaming.term.flatten_beta()),
    };

    assert!(dense_eval.cost.is_finite() && streaming_eval.cost.is_finite());
    assert_eq!(dense_eval.gradient.len(), streaming_eval.gradient.len());
    let dense_norm_sq = dense_eval.gradient.dot(&dense_eval.gradient);
    assert!(
        dense_norm_sq.is_finite() && dense_norm_sq > 1.0e-12,
        "route parity must exercise a nonzero analytic gradient; norm^2={dense_norm_sq}"
    );
    assert_abs_diff_eq!(streaming_eval.cost, dense_eval.cost, epsilon = 1.0e-7);
    for (coordinate, (&streamed, &direct)) in streaming_eval
        .gradient
        .iter()
        .zip(dense_eval.gradient.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(streamed, direct, epsilon = 1.0e-6);
        assert!(
            streamed.is_finite(),
            "streaming gradient coordinate {coordinate} is non-finite"
        );
    }
}

/// #2515 blocker 3 — WHICH assembly the stale-pair guard is comparing.
///
/// `production_objective_forced_streaming_value_gradient_matches_dense` dies on
/// `matrix_free_arrow_operator_apply refuses a stale matrix-free system/cache
/// pair`, with the MANIFOLD fingerprint equal and the ROW-HESSIAN fingerprint
/// different. The pair is `converged_cache` — factored during
/// `converge_inner_for_undamped_logdet` from an `assemble_arrow_schur` on the term
/// itself — against the system `assemble_full_matrix_free_evidence_system`
/// re-assembles afterwards through `materialize_chunk`. Two candidate causes, and
/// they are separable at a FIXED state with no solve involved:
///
///  (a) the collapse-prevention gates. `converge_inner_for_undamped_logdet`
///      freezes them, converges, then RESTORES the flag, so the later assembly
///      re-refreshes all three from the moved state
///      (`assemble_arrow_schur_scaled` is gated on `streaming_gates_frozen`).
///  (b) the two assemblers are not the same code path. One goes through
///      `materialize_chunk`, which re-materialises the row window and copies a
///      SUBSET of the term's state; the other assembles from the term directly.
///
/// (a) is real — measured below, the gate state alone moves the row fingerprint.
/// But it is not sufficient: holding the freeze across the whole criterion
/// evaluation leaves `production_objective_..._matches_dense` failing with the
/// identical refusal class. So this pins BOTH comparisons, at one frozen state, so
/// the next reader does not have to re-derive which one is load-bearing.
#[test]
fn evidence_assembly_row_fingerprint_sources_2515() {
    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    let rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);

    // Freeze once, exactly as converge_inner_for_undamped_logdet does on entry.
    term.refresh_decoder_repulsion_gate();
    term.refresh_barrier_coactivation_gate();
    term.refresh_amplitude_barrier_gate();
    term.streaming_gates_frozen = true;

    // (b) TWO ASSEMBLERS, ONE STATE, GATES HELD FROZEN THROUGHOUT. Nothing about
    // the gates can differ here, so any fingerprint gap is the assembler itself.
    let direct = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("direct arrow-Schur assembly");
    let (chunked, _chunk_term) = term
        .assemble_full_matrix_free_evidence_system(target.view(), &rho, None, None)
        .expect("matrix-free evidence assembly");
    let direct_fp = direct.current_row_hessian_fingerprint();
    let chunked_fp = chunked.current_row_hessian_fingerprint();
    println!(
        "[#2515 B3-SOURCE] gates frozen throughout: direct_assembly_row_fp={direct_fp} \
         chunked_evidence_row_fp={chunked_fp} equal={}",
        direct_fp == chunked_fp
    );

    // (a) THE GATE STATE ALONE, at one state, same assembler both times.
    let (frozen, _) = term
        .assemble_full_matrix_free_evidence_system(target.view(), &rho, None, None)
        .expect("frozen-gate evidence system");
    term.streaming_gates_frozen = false;
    let (refreshed, _) = term
        .assemble_full_matrix_free_evidence_system(target.view(), &rho, None, None)
        .expect("refreshed-gate evidence system");
    println!(
        "[#2515 B3-SOURCE] same assembler, gate state only: frozen_row_fp={} \
         refreshed_row_fp={} equal={}",
        frozen.row_hessian_fingerprint,
        refreshed.row_hessian_fingerprint,
        frozen.row_hessian_fingerprint == refreshed.row_hessian_fingerprint
    );

    // The MANIFOLD fingerprint is what makes the production refusal message
    // diagnostic rather than ambiguous: it stays equal under every variation here,
    // so a row-fingerprint mismatch can never be misread as the atoms changing.
    // This is the invariant the guard's own message relies on, and it is stable
    // across whatever repair lands.
    assert_eq!(
        direct.manifold_mode_fingerprint, chunked.manifold_mode_fingerprint,
        "#2515: the two assemblers must agree on the MANIFOLD fingerprint — the \
         stale-pair guard reports it alongside the row fingerprint precisely so a \
         row mismatch can be read as an operator difference and not as a changed \
         dictionary"
    );
    assert_eq!(
        frozen.manifold_mode_fingerprint, refreshed.manifold_mode_fingerprint,
        "#2515: the collapse-prevention gate state must not move the MANIFOLD \
         fingerprint; it is a property of the atoms, not of the penalty gates"
    );
    // Non-vacuity: a fingerprint of 0 is the constructor sentinel, and comparing
    // two sentinels would satisfy the assertions above while measuring nothing.
    assert_ne!(direct_fp, 0, "#2515: the direct assembly must publish a real row fingerprint");
    assert_ne!(chunked_fp, 0, "#2515: the chunked assembly must publish a real row fingerprint");
}

/// #2515 — the row-Hessian fingerprint must be a function of the OPERATOR:
/// invariant under rebuild, sensitive to every field that defines it.
///
/// `60feddc2e` replaced an `Arc` POINTER ADDRESS proxy, which made the fingerprint
/// an identity of the ALLOCATION — every rebuild of an unchanged operator produced
/// a different value, so `validate_matrix_free_arrow_pair` refused every
/// system/cache pair on the matrix-free path (which is every SAE fit).
///
/// The two failure modes are NOT symmetric, and that asymmetry is why this test
/// has a negative arm at all. The address identity failed LOUDLY AND ALWAYS. A
/// content identity that omits a field fails SILENTLY: two genuinely different
/// operators hash equal and the guard ACCEPTS a stale pair. A fingerprint function
/// that simply returned a constant would satisfy the invariance arm perfectly, so
/// invariance alone is not evidence of anything.
///
/// POSITIVE: two independent assemblies of one state agree.
/// NEGATIVE: perturbing the state disagrees — once per field of
/// `SaeKroneckerRows`, so the completeness claim is checked field by field rather
/// than asserted. (`SaeKroneckerRows::content_fingerprint` also destructures
/// exhaustively, so a NEW field breaks the build; these arms cover the fields that
/// exist.)
#[test]
fn row_hessian_fingerprint_is_a_function_of_the_operator_2515() {
    use super::kronecker::SaeKroneckerRows;

    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    let rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    term.refresh_decoder_repulsion_gate();
    term.refresh_barrier_coactivation_gate();
    term.refresh_amplitude_barrier_gate();
    term.streaming_gates_frozen = true;

    // POSITIVE — two independent assemblies of ONE state, through the two
    // different assemblers, must agree.
    let direct = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("direct arrow-Schur assembly");
    let (chunked, _chunk_term) = term
        .assemble_full_matrix_free_evidence_system(target.view(), &rho, None, None)
        .expect("matrix-free evidence assembly");
    assert!(
        direct.htbeta_matvec.is_some() && chunked.htbeta_matvec.is_some(),
        "#2515: both assemblies must install the matrix-free cross-block operator, \
         or this test does not exercise the path the fingerprint defect lived on"
    );
    assert_eq!(
        direct.current_row_hessian_fingerprint(),
        chunked.current_row_hessian_fingerprint(),
        "#2515: two assemblies of the same state must produce the same row-Hessian \
         fingerprint. A guard that cannot be satisfied by an unchanged operator is \
         not strict, it is inoperative — that was the defect, and the pointer \
         address is what made it unsatisfiable."
    );

    // NEGATIVE — per field of `SaeKroneckerRows`. Without these the invariance
    // above is satisfied by any constant.
    let p = 4usize;
    let a_phi: Arc<[Vec<(usize, f64)>]> =
        Arc::from(vec![vec![(0usize, 1.0_f64), (3, -0.5)], vec![(1, 0.25)]].into_boxed_slice());
    let local_jac: Arc<[Vec<f64>]> =
        Arc::from(vec![vec![0.5_f64; p], vec![-0.25_f64; p]].into_boxed_slice());
    let base = SaeKroneckerRows::new(p, Arc::clone(&a_phi), Arc::clone(&local_jac));
    let base_fp = base.content_fingerprint();

    let mut a_phi_moved = a_phi.to_vec();
    a_phi_moved[0][0].1 += 1.0e-9;
    let changed_a_phi = SaeKroneckerRows::new(
        p,
        Arc::from(a_phi_moved.into_boxed_slice()),
        Arc::clone(&local_jac),
    );
    assert_ne!(
        base_fp,
        changed_a_phi.content_fingerprint(),
        "#2515: a change to the sparse support weights must move the operator \
         fingerprint (perturbation 1e-9 on one weight)"
    );

    let mut jac_moved = local_jac.to_vec();
    jac_moved[1][0] += 1.0e-9;
    let changed_jac = SaeKroneckerRows::new(
        p,
        Arc::clone(&a_phi),
        Arc::from(jac_moved.into_boxed_slice()),
    );
    assert_ne!(
        base_fp,
        changed_jac.content_fingerprint(),
        "#2515: a change to the local Jacobian must move the operator fingerprint"
    );

    let wider_jac: Arc<[Vec<f64>]> =
        Arc::from(vec![vec![0.5_f64; p + 1], vec![-0.25_f64; p + 1]].into_boxed_slice());
    let changed_p = SaeKroneckerRows::new(p + 1, Arc::clone(&a_phi), wider_jac);
    assert_ne!(
        base_fp,
        changed_p.content_fingerprint(),
        "#2515: a change to the decoder output dimension must move the operator \
         fingerprint"
    );

    // The metric field: presence is what this fingerprint carries (its CONTENT
    // also enters `htt`, which the row fingerprint hashes directly), so presence
    // is what must move the value.
    let identity_metric = gam_problem::RowMetric::euclidean(p, p)
        .expect("a p-dimensional Euclidean row metric is constructible");
    let changed_metric =
        SaeKroneckerRows::new(p, a_phi, local_jac).with_output_metric(Some(identity_metric));
    assert_ne!(
        base_fp,
        changed_metric.content_fingerprint(),
        "#2515: installing an output metric must move the operator fingerprint"
    );
}

/// Hybrid-EFS must replace the former held-zero non-ordered Beta--Bernoulli assignment coordinate
/// with the exact penalized quasi-Laplace derivative and expose that same root-equivalent update to
/// the final fixed-point proof hook. This dense fixture exercises the exact dense
/// sibling cheaply; the complete-gradient parity test below pins the matrix-free
/// sibling to identical math.
#[test]
fn fixed_point_certificate_covers_non_ordered_beta_bernoulli_exact_gradient() {
    let make_objective = || {
        let (term, target, rho) = small_two_atom_periodic_term();
        let rho_flat = rho.to_flat();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("non-ordered Beta--Bernoulli EFS startup evaluation");
    let gradient = iteration
        .psi_gradient
        .as_ref()
        .expect("assignment strength must be the Hybrid-EFS gradient block")[0];
    assert_eq!(
        iteration.psi_indices.as_deref(),
        Some(&[0][..]),
        "the Hybrid-EFS gradient must map back to log_lambda_sparse"
    );
    assert!(gradient.is_finite(), "assignment gradient must be finite");
    assert_abs_diff_eq!(
        iteration.steps[0],
        -gradient / gradient.abs().max(1.0),
        epsilon = 1.0e-12
    );

    let (mut proof_objective, proof_rho) = make_objective();
    let proof = proof_objective
        .eval_fixed_point_certificate(&proof_rho)
        .expect("fixed-point proof hook must evaluate");
    let (mut exact_objective, exact_rho) = make_objective();
    let exact = exact_objective
        .eval(&exact_rho)
        .expect("authoritative analytic gradient");
    assert_eq!(proof.coordinates.len(), proof_rho.len());
    match &proof.coordinates[0] {
        FixedPointCoordinateCertificate::Covered { update, scale } => {
            assert_abs_diff_eq!(*update, -exact.gradient[0], epsilon = 1.0e-12);
            assert_eq!(*scale, 1.0);
        }
        FixedPointCoordinateCertificate::Uncovered { reason } => panic!(
            "the exact assignment-strength derivative must certify this coordinate: {reason}"
        ),
    }
}

/// Learnable ordered Beta--Bernoulli concentration uses the same complete criterion
/// derivative as every other assignment-strength coordinate. This guards
/// against reintroducing the removed occupancy-only alpha fixed point, whose
/// stationarity equation omitted the inner response and log-determinant terms.
#[test]
fn fixed_point_certificate_covers_ordered_beta_bernoulli_complete_gradient() {
    let make_objective = || {
        let (mut term, target, mut rho) = small_two_atom_periodic_term();
        term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.8, 1.0, true);
        rho.log_lambda_sparse = 0.7_f64.ln();
        let rho_flat = rho.to_flat();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("ordered Beta--Bernoulli EFS startup evaluation");
    // #2330: `psi_gradient: None` has TWO producers, and they mean opposite things.
    // The assignment-strength block at `outer_objective.rs:3219` leaves it `None`
    // when the coordinate is structurally absent, which is what this assertion is
    // about. But `infeasible_evaluation` (`outer_objective.rs:3079`) ALSO returns
    // `psi_gradient: None`, together with `cost = INFINITY`, when the evaluation was
    // refused outright.
    //
    // The refusal reason is NOT lost where it is produced: `infeasible_evaluation`
    // embeds it in every coordinate certificate as `fixed-point evidence
    // unavailable: {reason}`, and `efs_step_with_certificate` returns those
    // certificates alongside the eval. It is lost one caller later, at
    // `outer_objective.rs:3033`:
    //
    //     self.efs_step_with_certificate(rho_flat)
    //         .map(|(evaluation, _)| evaluation)
    //
    // The `_` is the certificate vector. `eval_efs` is built on `efs_step`, so this
    // test can only ever see the reasonless `EfsEval`. Recovering the reason here
    // does not need a new field on `EfsEval` (26 construction sites) — it needs a
    // caller that keeps the certificates.
    //
    // Until then, separate the two producers on the observable that distinguishes
    // them, so an infeasible evaluation is not reported as a missing gradient block.
    assert!(
        iteration.cost.is_finite(),
        "the evaluation was refused before any gradient block was reached \
         (cost={}); this is an infeasibility whose reason string was dropped by \
         `infeasible_evaluation`, NOT a missing learnable-concentration gradient",
        iteration.cost,
    );
    let gradient = iteration
        .psi_gradient
        .as_ref()
        .expect("learnable concentration must use the complete gradient block")[0];
    assert!(gradient.is_finite());
    assert_eq!(iteration.psi_indices.as_deref(), Some(&[0][..]));
    assert_abs_diff_eq!(
        iteration.steps[0],
        -gradient / gradient.abs().max(1.0),
        epsilon = 1.0e-12
    );

    let (mut proof_objective, proof_rho) = make_objective();
    let proof = proof_objective
        .eval_fixed_point_certificate(&proof_rho)
        .expect("ordered Beta--Bernoulli fixed-point proof hook must evaluate");
    let (mut exact_objective, exact_rho) = make_objective();
    let exact = exact_objective
        .eval(&exact_rho)
        .expect("authoritative analytic gradient");
    match &proof.coordinates[0] {
        FixedPointCoordinateCertificate::Covered { update, scale } => {
            assert_abs_diff_eq!(*update, -exact.gradient[0], epsilon = 1.0e-12);
            assert_eq!(*scale, 1.0);
        }
        FixedPointCoordinateCertificate::Uncovered { reason } => panic!(
            "the complete ordered Beta--Bernoulli concentration derivative must certify this coordinate: {reason}"
        ),
    }
}

/// The non-ordered Beta--Bernoulli assignment-strength `0.5 tr(H^-1 dH/dlog_lambda_sparse)` channel
/// must be reconstructible from the same reduced-Schur inverse-probe bundle as
/// the smoothness, ARD, and theta-adjoint channels. Full-basis probes with exact
/// dense `S^-1` make the bundle identity exact, so this isolates the new matrix-
/// free contraction from stochastic-CG error.
#[test]
fn assignment_strength_trace_from_probes_matches_dense_softmax() {
    let (n, p, k) = (24usize, 2usize, 2usize);
    let term = build_softmax_term(n, p, k);
    let rho = SaeManifoldRho::new(
        0.7_f64.ln(),
        0.8_f64.ln(),
        vec![Array1::from_elem(1, 1.2_f64.ln()); k],
    );
    // Keep the fixture on the same positive-rank Laplace branch that the
    // production criterion admits.  The old unrelated synthetic target made
    // both decoders fall below the hard MP edge, so the canonical complete
    // gradient correctly refused the rank-zero branch before this test could
    // reach its dense-vs-probe identity.  A deterministic residual around
    // this term's own nonzero reconstruction exercises the identical trace and
    // IFT seams without relying on a value-invalid atom.
    let fitted = term
        .try_fitted_for_rho(&rho)
        .expect("softmax positive-rank fixture reconstruction");
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        fitted[[row, col]] + 1.0e-3 * ((row + 2 * col) as f64 * 0.17).sin()
    });
    let (system, _chunk_term) = term
        .assemble_full_matrix_free_evidence_system(target.view(), &rho, None, None)
        .expect("softmax matrix-free evidence system");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("direct factorization");
    assert!(
        cache.deflated_row_directions.iter().all(Vec::is_empty),
        "the probe identity is defined on the plain undeflated fixture"
    );

    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("dense assignment-strength trace");

    let border_dim = cache.k;
    let sqrt_dim = (border_dim as f64).sqrt();
    let probes = (0..border_dim)
        .map(|column| {
            let mut probe = Array1::<f64>::zeros(border_dim);
            probe[column] = sqrt_dim;
            probe
        })
        .collect::<Vec<_>>();
    let inverse_probes = probes
        .iter()
        .map(|probe| {
            cache
                .schur_inverse_apply(probe.view())
                .expect("exact reduced-Schur inverse probe")
        })
        .collect::<Vec<_>>();
    let matrix_free = term
        .assignment_log_strength_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &inverse_probes,
            EvidenceOperator::Majorizer,
        )
        .expect("matrix-free assignment-strength trace");

    assert!(
        dense.abs() > 1.0e-12,
        "fixture must excite a nonzero assignment-strength trace"
    );
    assert_abs_diff_eq!(matrix_free, dense, epsilon = 1.0e-9);
}

