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
//!     `analytic_outer_rho_gradient_components_with_bundle`) also reads the returned cache,
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
use crate::manifold::tests_dense_solver_oracles::DeflatedArrowSolver;
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
/// while the matrix-free plan's peak (chunk window + per-row `q_row × p`
/// Kronecker Jacobian + per-row `H_tt` and factor blocks + border vector
/// workspace, q_row = K−1+K·d = 63) stays in the tens of MB (#4262). The
/// planner must therefore REFUSE the dense direct plan (routing the criterion
/// to streaming) while ADMITTING the matrix-free plan — the exact regime the
/// streaming route was built for.
#[test]
fn wide_border_routes_to_streaming_with_complete_analytic_gradient_certificate() {
    let (n, p, k, d_max) = (500usize, 128usize, 32usize, 1usize);
    let total_basis = 2 * k; // width-2 euclidean basis per atom.
    let border_dim = total_basis * p;
    // Softmax rows: `K − 1` gate logits plus `K · d` chart coordinates;
    // the unframed matrix-free cross block is `q_row × p` per row (#4262).
    let row_dim = (k - 1) + k * d_max;
    let budget = 2 * 1024 * 1024 * 1024usize; // 2 GiB representative in-core budget.
    let host_available = 8 * 1024 * 1024 * 1024usize;
    let chunk_window = SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE;
    let plan = sae_streaming_plan_from_budget(
        n,
        total_basis,
        k,
        d_max,
        row_dim,
        p,
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
        row_dim,
        p,
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
    // Each route converges its own inner solve; the log names which acceptance each took.
    gam_runtime::test_support::install_diagnostic_logger();
    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    assert_forced_streaming_value_gradient_matches_dense(term, target);
}

/// #3439 — the periodic phase circle volume on the streaming lane. A prior period of 2 on
/// the planted circle's 1-periodic harmonic basis leaves the atom outside every exact orbit
/// (`CompactOrbitLaplaceReason::PeriodMismatch`), so both routes price it by Laplace and the
/// dense value carries `periodic_phase_marginal`'s `−2·log M(Sᴮ, P) − log Sᵀ + log 2π`. The
/// streaming value used to omit that correction and its gradient, which moved the two routes'
/// costs apart by half of it. The same parity as the period-1 pin must hold with it priced.
#[test]
fn production_objective_forced_streaming_matches_dense_with_circle_phase_marginal_3439() {
    gam_runtime::test_support::install_diagnostic_logger();
    let target = planted_circle_embedded(32, 4, 0.02);
    let base = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        base.assignment.logits.clone(),
        vec![base.assignment.coords[0].as_matrix()],
        vec![LatentManifold::Circle { period: 2.0 }],
        base.assignment.mode,
    )
    .expect("the seed's logits and coordinates describe one circle block");
    let mut term = SaeManifoldTerm::new(base.atoms.clone(), assignment)
        .expect("the seed atom and the re-periodized assignment describe the same block");
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    assert_forced_streaming_value_gradient_matches_dense(term, target);
}

/// Prices `term` at its objective's baseline rho once on the dense production route and once
/// on the forced streaming route, and asserts the two `(value, gradient)` pairs agree.
fn assert_forced_streaming_value_gradient_matches_dense(
    term: SaeManifoldTerm,
    target: Array2<f64>,
) {
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
    let rho_flat = dense.baseline_rho.flat_coordinates();
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
    // Two objectives price the two routes, and each converges its own inner solve
    // at this rho first, so their costs can agree only to the resolution of those
    // solves: the inner objective stall band. A fixed absolute epsilon has no scale;
    // census job 532879 at 4bf15f660 measured a 1.75e-7 gap (1.6e-9 relative)
    // against 1e-7 on a cost of 111.25.
    let cost_resolution =
        SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * dense_eval.cost.abs().max(1.0);
    assert_abs_diff_eq!(streaming_eval.cost, dense_eval.cost, epsilon = cost_resolution);
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

/// #979 — the production gradient lane where the memory planner admits matrix-free evidence
/// and refuses direct evidence. That lane used to return the streaming criterion paired with a
/// zero gradient, while `capability()` declares an analytic gradient, so a gradient reader saw a
/// stationary point.
///
/// The host reading is the term's own budget seam. Below the host reserve floor the in-core
/// budget is zero and the planner refuses both routes. Above it, a shape whose direct peak and
/// exact-stationarity resident both sit under `SAE_DIRECT_ALWAYS_ADMIT_BYTES` is admitted
/// directly, so the n=32 planted circle has no reading that takes this lane. At n=256 the
/// exact-stationarity resident exceeds that bound, and the reading one byte short of the reserve
/// floor plus that resident refuses exact stationarity while the matrix-free peak still fits.
/// The premise asserts both production predicates on the resulting plan.
///
/// Asked for value and gradient, the lane must return the analytic gradient of the streaming
/// artifact, priced by a second objective at the same reading and rho.
#[test]
fn production_gradient_lane_returns_the_streaming_gradient_where_direct_logdet_is_not_admitted_979() {
    gam_runtime::test_support::install_diagnostic_logger();
    let target = planted_circle_embedded(256, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    term.gpu_policy = gam_gpu::GpuPolicy::Off;
    let default_plan = term
        .streaming_plan()
        .expect("streaming plan at the default host reading");
    term.host_available_bytes = super::streaming_plan::SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES
        .saturating_add(default_plan.estimated_exact_stationarity_bytes)
        .saturating_sub(1);
    let starved_plan = term
        .streaming_plan()
        .expect("streaming plan at the starved host reading");
    assert!(
        starved_plan.matrix_free_admitted && !starved_plan.direct_logdet_admitted(),
        "premise: the planner admits matrix-free evidence and refuses direct evidence; \
         plan={starved_plan:?}"
    );
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    let mut production = SaeManifoldOuterObjective::new(
        term.clone(),
        target.clone(),
        None,
        seed_rho.clone(),
        40,
        1.0,
        1.0e-6,
        1.0e-6,
    );
    let mut reference =
        SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6);
    let rho_flat = production.baseline_rho.flat_coordinates();
    let rho = reference
        .baseline_rho
        .from_flat(rho_flat.view())
        .expect("both objectives own the same typed rho layout");
    let production_eval = OuterObjective::eval(&mut production, &rho_flat)
        .expect("production value+gradient on the streaming route");
    let artifact = reference
        .evaluate_outer_criterion_route(&rho, false, false)
        .expect("streaming artifact at the same rho");
    let reference_gradient = reference
        .analytic_gradient_for_outer_evaluation(&rho, &artifact)
        .expect("streaming analytic gradient at the same rho");
    let reference_norm_sq = reference_gradient.dot(&reference_gradient);
    assert!(
        reference_norm_sq.is_finite() && reference_norm_sq > 1.0e-12,
        "the streaming analytic gradient must be nonzero here; norm^2={reference_norm_sq}"
    );
    assert!(production_eval.cost.is_finite());
    assert_eq!(production_eval.gradient.len(), reference_gradient.len());
    for (coordinate, (&returned, &expected)) in production_eval
        .gradient
        .iter()
        .zip(reference_gradient.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(returned, expected, epsilon = 1.0e-6);
        assert!(
            returned.is_finite(),
            "production gradient coordinate {coordinate} is non-finite"
        );
    }
}

/// A K=1 softmax term on a non-periodic `EuclideanPatch` atom (latent dimension 1, the
/// given monomial degree) decoding a planted parabolic arc embedded in `R^p`, seeded at
/// the planted coordinate with its least-squares decoder. The chart has no orbit generator, so the
/// streaming route prices it through the rational lane and its derivative bundle.
fn planted_arc_seed_term(
    n: usize,
    p: usize,
    degree: usize,
    sigma: f64,
) -> (SaeManifoldTerm, Array2<f64>) {
    use super::tests::deterministic_circle_noise;
    use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        -1.0 + 2.0 * (row as f64) / (n as f64 - 1.0)
    });
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        let t = coords[[row, 0]];
        deterministic_circle_noise(col, 0) * t
            + deterministic_circle_noise(col, 1) * t * t
            + sigma * deterministic_circle_noise(row, col)
    });
    let evaluator = Arc::new(
        crate::basis::EuclideanPatchEvaluator::new(1, degree)
            .expect("a positive latent dimension is a valid Euclidean patch"),
    );
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the planted coordinates lie in the Euclidean patch domain");
    let decoder = fast_ata(&phi)
        .cholesky(Side::Lower)
        .expect("the planted arc's patch Gram is positive definite")
        .solve_mat(&fast_atb(&phi, &target));
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "arc",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("phi, jet, decoder and gram were built with matching shapes")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("one logit column, coordinate block and manifold");
    (
        SaeManifoldTerm::new(vec![atom], assignment).expect("one atom over the same rows"),
        target,
    )
}

/// #2933 F29 — the premise [`SaeManifoldOuterObjective::logdet_gradient_probe_samples`]
/// splits the surrogate gradient on: the streaming gradient is affine in its derivative
/// bundle's second moment under the channels' `1/|V|` normalization, so the gradient on
/// the whole bundle `V` is the mean of the gradients taken on each vector of `V` alone.
/// A channel normalized by anything else, or nonlinear in a vector, breaks the identity
/// and with it the per-probe rows the recertification's standard error comes from.
///
/// The host reading is starved the way the #979 lane test starves it, so the evaluation
/// takes the streaming route; the chart is a non-periodic arc, so that route carries a
/// derivative bundle rather than the circle orbit lane's exact elimination.
#[test]
fn streaming_gradient_is_affine_in_the_derivative_bundle_second_moment_2933() {
    gam_runtime::test_support::install_diagnostic_logger();
    let (mut objective, _) = starved_planted_arc_objective();
    let rho_flat = objective.baseline_rho.flat_coordinates();
    let rho = objective
        .baseline_rho
        .from_flat(rho_flat.view())
        .expect("typed rho layout");
    let evaluation = objective
        .evaluate_outer_criterion_route(&rho, false, false)
        .expect("streaming evaluation");
    let (full, rows) = objective
        .streaming_gradient_per_bundle_vector(&rho, &evaluation)
        .expect("gradients on the bundle and on each of its vectors")
        .expect("premise: the starved host reading evaluates on the streaming route");
    assert!(
        rows.len() >= 2,
        "premise: the bundle carries several vectors to average; it carries {}",
        rows.len()
    );
    // The two sides sum the same terms in different orders: each gradient reduces
    // over the `n` data rows, and the mean adds `r` per-vector gradients. The
    // first-order bound on reordering a sum of `n + r` terms is `(n + r)·ε` times
    // the largest magnitude summed.
    let count = rows.len() as f64;
    let n_rows = 256.0;
    for coordinate in 0..full.len() {
        let mean = rows.iter().map(|row| row[coordinate]).sum::<f64>() / count;
        let magnitude = rows
            .iter()
            .map(|row| row[coordinate].abs())
            .fold(full[coordinate].abs(), f64::max);
        let tolerance = (n_rows + count) * f64::EPSILON * magnitude;
        let gap = (full[coordinate] - mean).abs();
        assert!(
            gap <= tolerance,
            "coordinate {coordinate}: gradient on the bundle {:.17e} differs from the mean \
             of its {} per-vector gradients {mean:.17e} by {gap:.3e} > {tolerance:.3e}",
            full[coordinate],
            rows.len()
        );
    }
}

/// The planted arc of [`planted_arc_seed_term`] with the host reading one byte short of
/// the reserve floor plus its exact-stationarity resident, so the planner admits the
/// matrix-free route and refuses the direct one; also returns that default reading.
fn starved_planted_arc_objective() -> (SaeManifoldOuterObjective, usize) {
    let (mut term, target) = planted_arc_seed_term(256, 4, 2, 0.02);
    term.gpu_policy = gam_gpu::GpuPolicy::Off;
    let default_host = term.host_available_bytes;
    let default_plan = term
        .streaming_plan()
        .expect("streaming plan at the default host reading");
    term.host_available_bytes = super::streaming_plan::SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES
        .saturating_add(default_plan.estimated_exact_stationarity_bytes)
        .saturating_sub(1);
    let starved_plan = term
        .streaming_plan()
        .expect("streaming plan at the starved host reading");
    assert!(
        starved_plan.matrix_free_admitted && !starved_plan.direct_logdet_admitted(),
        "premise: the planner admits matrix-free evidence and refuses direct evidence; \
         plan={starved_plan:?}"
    );
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    let objective =
        SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6);
    (objective, default_host)
}

/// The planted arc on a degree-7 patch (`M = 8`, border `k = M·p = 32`), whose dense
/// exact-A pencil lane's eight `k × k` blocks outgrow the matrix-free resident, with the
/// host reading at the reserve floor plus exactly that resident. The planner then admits
/// matrix-free evidence, refuses the direct route, and the pencil lane does not fit the
/// budget, so log|S| is priced on the rational surrogate plan. The border is widened
/// through `M`, not `p`: past `p = M` the decoder frame caps it at `M·r`. The rows are
/// few enough that the matrix-free resident, linear in `n`, stays under the pencil, and
/// many enough that the exact stationarity route is not admitted as a tiny allocation.
fn rational_lane_planted_arc_objective() -> SaeManifoldOuterObjective {
    let (mut term, target) = planted_arc_seed_term(200, 4, 7, 0.02);
    term.gpu_policy = gam_gpu::GpuPolicy::Off;
    let default_plan = term
        .streaming_plan()
        .expect("streaming plan at the default host reading");
    term.host_available_bytes = super::streaming_plan::SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES
        .saturating_add(default_plan.estimated_matrix_free_peak_bytes);
    let plan = term
        .streaming_plan()
        .expect("streaming plan at the matrix-free reading");
    let pencil_bytes = gam_solve::arrow_schur::dense_lane_exact_a_pencil_peak_bytes(term.beta_dim())
        .expect("the pencil lane's byte count fits usize");
    assert!(
        plan.matrix_free_admitted
            && !plan.direct_logdet_admitted()
            && pencil_bytes > plan.in_core_budget_bytes,
        "premise: matrix-free evidence is admitted, the direct route refused, and the pencil \
         lane's {pencil_bytes} bytes exceed the budget; plan={plan:?}"
    );
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6)
}

/// The route is read from the live state at every probe, so a fit can cross from the
/// streaming lane onto the direct one (an atom rank-reduces and its smaller resident
/// fits the host). The basin envelope the direct route engages must be sized by that
/// same state. Sized once at construction, where the direct route was refused, its
/// capacity was 0 and the first envelope probe refused its own seed.
///
/// The crossing is driven here by restoring the host reading after construction; the
/// shape-driven crossing on the planted circle reaches it too, but that fit then stops
/// on the circle orbit's missing arrow-route criterion (#2234).
#[test]
fn basin_envelope_is_sized_by_the_state_that_admits_it_2933() {
    gam_runtime::test_support::install_diagnostic_logger();
    let (mut objective, default_host) = starved_planted_arc_objective();
    objective.term.host_available_bytes = default_host;
    assert!(
        objective
            .term
            .streaming_plan()
            .expect("streaming plan at the restored host reading")
            .direct_logdet_admitted(),
        "premise: the restored host reading admits the direct route"
    );
    let seed = objective.baseline_rho.flat_coordinates();
    let evaluation =
        OuterObjective::eval(&mut objective, &seed).expect("the envelope admits its own seed");
    assert!(
        evaluation.cost.is_finite(),
        "the direct route prices the planted arc's seed; cost={}",
        evaluation.cost
    );
    let telemetry = objective.probe_telemetry();
    assert!(
        telemetry.basin_envelope_evals > 0,
        "premise: the direct route prices through the envelope; telemetry={telemetry:?}"
    );
    assert!(
        telemetry.basin_max_members >= 1
            && telemetry.basin_member_capacity >= telemetry.basin_max_members,
        "the envelope retains its seed within the capacity of the state it priced; \
         telemetry={telemetry:?}"
    );
}

/// #2933 F29 — a certificate the streaming lane's rational log|S| surrogate issued is
/// judged on probes the search never saw before it is stamped. At the planted arc's
/// seed the criterion's gradient is far from zero, so a certificate claiming a band of
/// 0 must be contradicted by the unseen probes, returned as `SurrogateDisagrees` at the
/// installed point with the doubled plan installed. A band the gradient clears stands,
/// and stamps the criterion re-scored on the (again doubled) validation plan.
#[test]
fn surrogate_certificate_is_rescored_on_unseen_probes_before_stamping_2933() {
    use gam_solve::model_types::{
        CertifiedRung, CurvatureEvidence, OuterCriterionCertificate,
        OuterStationarityCertificate,
    };
    gam_runtime::test_support::install_diagnostic_logger();
    let mut objective = rational_lane_planted_arc_objective();
    let seed = objective.baseline_rho.flat_coordinates();
    let evaluation = OuterObjective::eval(&mut objective, &seed).expect("streaming evaluation");
    assert!(evaluation.cost.is_finite(), "cost={}", evaluation.cost);
    let search_probes = sae_surrogate_lane_config().num_probes;
    assert_eq!(
        objective.surrogate_probe_count(),
        Some(search_probes),
        "premise: the matrix-free reading prices log|S| on the search's rational plan"
    );
    let certificate = |bound: f64| OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::AnalyticGradient {
            grad_norm: 0.0,
            projected_grad_norm: 0.0,
            bound,
            rung: CertifiedRung {
                label: "#2933-F29-test".to_string(),
                derived_standard: false,
            },
        },
        curvature: CurvatureEvidence::NotAvailable,
        lambdas_railed: Vec::new(),
        railed_facts: Vec::new(),
        curvature_floor: None,
        newton_polish: None,
        criterion_error: None,
    };
    match objective.validate_surrogate_certificate(&certificate(0.0)) {
        Err(SaeOuterCertificationError::SurrogateDisagrees { rho, detail }) => {
            assert_eq!(
                rho.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                seed.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "the disagreement is reported at the installed point"
            );
            assert!(detail.contains("unseen probes"), "detail: {detail}");
        }
        other => panic!("a band of 0 at a non-stationary seed must disagree; got {other:?}"),
    }
    assert_eq!(
        objective.surrogate_probe_count(),
        Some(2 * search_probes),
        "the disagreement leaves the doubled plan installed for the resumed search"
    );
    let stamped = objective
        .validate_surrogate_certificate(&certificate(f64::INFINITY))
        .expect("a band the gradient clears stands")
        .expect("the streaming lane re-scores the criterion it stamps");
    assert!(stamped.is_finite(), "stamped criterion {stamped}");
    assert_eq!(
        objective.surrogate_probe_count(),
        Some(4 * search_probes),
        "the stamped criterion is re-scored on the plan drawn past every probe it judged"
    );
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
        let rho_flat = rho.flat_coordinates();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    // A refused startup evaluation names its reason only in `efs_step`'s warn line.
    gam_runtime::test_support::install_diagnostic_logger();
    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("non-ordered Beta--Bernoulli EFS startup evaluation");
    assert!(
        iteration.cost.is_finite(),
        "the startup evaluation was refused before any gradient block was reached \
         (cost={}); the refusal reason is the `SAE EFS evaluation refused` warn line above, \
         NOT a missing assignment-strength gradient",
        iteration.cost,
    );
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
        let rho_flat = rho.flat_coordinates();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    // A refused startup evaluation names its reason only in `efs_step`'s warn line.
    gam_runtime::test_support::install_diagnostic_logger();
    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("ordered Beta--Bernoulli EFS startup evaluation");
    // #2330: `psi_gradient: None` has two producers that mean opposite things. The
    // assignment-strength block leaves it `None` when the coordinate is structurally
    // absent, and `infeasible_evaluation` returns it with `cost = INFINITY` when the
    // evaluation was refused. Separate them on the cost, so a refusal (named by
    // `efs_step`'s `SAE EFS evaluation refused` warn line) is not reported as a
    // missing gradient block.
    assert!(
        iteration.cost.is_finite(),
        "the evaluation was refused before any gradient block was reached \
         (cost={}); the refusal reason is the `SAE EFS evaluation refused` warn line above, \
         NOT a missing learnable-concentration gradient",
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

