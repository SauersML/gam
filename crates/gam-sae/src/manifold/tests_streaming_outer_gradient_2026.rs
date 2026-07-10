//! Streaming/matrix-free evidence route — outer-gradient lane parity and the
//! large-K/wide-border completion contract (W11).
//!
//! Two properties are pinned here that the pre-existing #1026 streaming-cache
//! test (`tests_streaming_efs_cache_1026`) did NOT cover:
//!
//!  1. **Outer-gradient parity.** The #1026 test proved the cache returned by
//!     `reml_criterion_streaming_exact_with_cache` is a drop-in for the EFS
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
//!     COMPLETE with a finite REML value rather than hard-erroring. We pin both
//!     halves deterministically: (a) the memory planner refuses the dense direct
//!     plan at this shape but admits the matrix-free plan, so the auto-router
//!     selects streaming; and (b) the streaming value path itself returns a finite
//!     criterion on the whitened term.

use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use approx::assert_abs_diff_eq;
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::rho_optimizer::{FixedPointCoordinateCertificate, OuterObjective};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

use super::tests::{TestPeriodicEvaluator, periodic_basis, small_two_atom_periodic_term};
use std::sync::Arc;

/// The analytic outer ρ-gradient assembled off the STREAMING cache
/// (`reml_criterion_streaming_exact_with_cache`) must be bit-identical to the one
/// assembled off the DENSE cache (`reml_criterion_with_cache`). Both entries
/// converge the inner (t, β) state through the SAME
/// `converge_inner_for_undamped_logdet` driver with the SAME undamped Direct
/// options, so the returned factor caches — and therefore the selected-inverse
/// reads the outer-gradient solver takes (logdet trace, third-order envelope
/// correction) — must agree. A regression that let the streaming cache diverge
/// from the dense one on the gradient lane (stale inner state, mismatched Schur
/// factor, wrong deflation) would surface here on a small dictionary where BOTH
/// caches are formable, rather than only in a multi-GB large-K fit where the dense
/// cache cannot be built at all. This is the gradient-lane analogue of the #1026
/// EFS-trace drop-in contract.
#[test]
fn streaming_cache_outer_gradient_matches_dense_cache() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let mut dense = term0.clone();
    let mut streaming = term0;

    let (dense_cost, dense_loss, dense_cache) = dense
        .reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .expect("dense cache criterion");
    let (stream_cost, stream_loss, stream_cache) = streaming
        .reml_criterion_streaming_exact_with_cache(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
        )
        .expect("streaming cache criterion");

    // Precondition: the two entries agree on the scalar criterion (the #1026
    // contract) — so any gradient difference below is a gradient-lane defect, not
    // an inner-state divergence.
    assert_abs_diff_eq!(stream_cost, dense_cost, epsilon = 1.0e-8);

    // Assemble the analytic outer ρ-gradient off EACH cache through the identical
    // production path the seed-validation / small-BFGS lane uses.
    let smooth = rho.lambda_smooth_vec();
    let dense_solver = dense
        .outer_gradient_arrow_solver(&dense_cache, &smooth)
        .expect("dense outer-gradient solver");
    let dense_grad = dense
        .analytic_outer_rho_gradient_components(
            target.view(),
            &rho,
            &dense_loss,
            &dense_cache,
            &dense_solver,
        )
        .expect("dense outer-gradient components")
        .gradient();

    let stream_solver = streaming
        .outer_gradient_arrow_solver(&stream_cache, &smooth)
        .expect("streaming outer-gradient solver");
    let stream_grad = streaming
        .analytic_outer_rho_gradient_components(
            target.view(),
            &rho,
            &stream_loss,
            &stream_cache,
            &stream_solver,
        )
        .expect("streaming outer-gradient components")
        .gradient();

    assert_eq!(
        dense_grad.len(),
        stream_grad.len(),
        "streaming outer gradient has a different ρ dimension than the dense one"
    );
    for (i, (d, s)) in dense_grad.iter().zip(stream_grad.iter()).enumerate() {
        assert!(
            d.is_finite() && s.is_finite(),
            "outer-gradient component {i} must be finite (dense={d}, streaming={s})"
        );
        assert_abs_diff_eq!(d, s, epsilon = 1.0e-8);
    }
    // The gradient must be non-trivial (a zero vector would make the parity
    // assertion vacuous).
    let g2: f64 = dense_grad.iter().map(|v| v * v).sum();
    assert!(
        g2 > 0.0 && g2.is_finite(),
        "the dense outer gradient must be non-trivial to make the parity check meaningful; ‖g‖²={g2}"
    );
    assert_abs_diff_eq!(stream_loss.total(), dense_loss.total(), epsilon = 1.0e-8);
}

// ---- Large-K / wide-border whitened completion ------------------------------

/// Deterministic standard-normal draws (Box–Muller over an LCG) so the whitening
/// factor fitted below is reproducible bit-for-bit.
fn lcg_uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_uniform(s).max(1e-12);
    let u2 = lcg_uniform(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A K-atom periodic term over `(n, p)` with a softmax assignment (non-IBP, so the
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
                (0.03 + 0.11 * i as f64 + 0.017 * r as f64).rem_euclid(1.0)
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
            SaeManifoldAtom::new(
                format!("atom{i}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
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
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// A `WhitenedStructured` per-row precision fitted over `(n, p)` correlated,
/// heteroscedastic residuals (mirrors the #2021 fixture).
fn fit_structured_metric(n: usize, p: usize) -> gam_problem::RowMetric {
    let lam = [1.0_f64, -0.7, 0.4, 0.9, -0.5];
    let dscale = [0.10_f64, 0.55, 0.95, 0.30, 0.70];
    let mut seed = 0x2026_00D5_1234_ABCDu64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    let mut activity = Array1::<f64>::zeros(n);
    for row in 0..n {
        let common = lcg_normal(&mut seed);
        activity[row] = 0.25 + (row as f64) / (n as f64);
        let amp = activity[row].sqrt();
        for i in 0..p {
            residuals[[row, i]] = amp * lam[i % lam.len()] * common
                + dscale[i % dscale.len()] * lcg_normal(&mut seed);
        }
    }
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 2,
    })
    .expect("StructuredResidualModel::fit");
    model.row_metric(n).expect("row_metric")
}

/// At K=32, p=128 the width-2 euclidean border is `border_dim = Σ_k M_k·p =
/// 64·128 = 8192`, so the dense direct evidence peak (`N·q·border_dim`,
/// q=K(1+d)=64) is ≈2.6 GB and exceeds a representative 2 GiB in-core budget,
/// while the matrix-free plan's peak (chunk window + sparse row-cross + border
/// vector workspace) stays in the tens of MB. The planner must therefore REFUSE
/// the dense direct plan (routing the criterion to streaming) while ADMITTING the
/// matrix-free plan — the exact regime the streaming route was built for.
#[test]
fn wide_border_routes_to_streaming_without_fake_gradient_certificate() {
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
        sae_outer_gradient_capability(plan),
        Derivative::Unavailable,
        "matrix-free SAE must not advertise the startup-only zero vector as an analytic gradient"
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
        sae_outer_gradient_capability(dense_plan),
        Derivative::Analytic,
        "dense SAE retains its exact joint-Hessian IFT gradient"
    );
    // The admission gate must accept the plan (no 'working set exceeds budget'
    // hard error) precisely because the matrix-free lane is admitted.
    plan.admitted_or_error(n, border_dim, k)
        .expect("matrix-free-admitted plan must not hard-error at the admission gate");
}

/// The matrix-free planner's unavailable-gradient declaration is only the first
/// half of the certification contract.  Exercise the objective-owned FINAL
/// fixed-point proof hook itself on a non-IBP assignment and pin the formerly
/// dangerous coordinate: `eval_efs` holds `log_lambda_sparse` at a zero iteration
/// step because softmax entropy has no Fellner--Schall equation, but the proof
/// hook must report that zero as `Uncovered`, never as a solved residual.
///
/// This is deliberately a small dense fixture so the regression is cheap.  The
/// hook and its coordinate semantics are identical in the wide matrix-free route;
/// only that route consumes the hook for final certification because its analytic
/// gradient capability is unavailable.
#[test]
fn fixed_point_certificate_hook_refuses_non_ibp_held_zero() {
    let make_objective = || {
        let (term, target, rho) = small_two_atom_periodic_term();
        let rho_flat = rho.to_flat();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    // Pin the iteration-side precondition: this coordinate really is the held
    // zero that previously masqueraded as convergence.
    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("non-IBP EFS startup evaluation");
    assert_eq!(
        iteration.steps[0], 0.0,
        "softmax log_lambda_sparse has no EFS equation and must remain held"
    );

    // A fresh objective calls the final-proof hook, not the iteration surface.
    // Its result must preserve the distinction between held and proved zero.
    let (mut proof_objective, proof_rho) = make_objective();
    let proof = proof_objective
        .eval_fixed_point_certificate(&proof_rho)
        .expect("fixed-point proof hook must evaluate");
    assert_eq!(proof.coordinates.len(), proof_rho.len());
    match &proof.coordinates[0] {
        FixedPointCoordinateCertificate::Uncovered { reason } => assert!(
            reason.contains("no root-equivalent fixed-point equation"),
            "non-IBP held coordinate must explain the missing equation; got: {reason}"
        ),
        FixedPointCoordinateCertificate::Covered { update, scale } => panic!(
            "a held non-IBP zero is not a stationarity proof (update={update}, scale={scale})"
        ),
    }
}

/// End-to-end: the whitened streaming REML criterion (`reml_criterion_streaming_
/// exact`) must COMPLETE with a finite value rather than surfacing the
/// `cost-only streaming route is required` hard-error class. The streaming lane is
/// size-INVARIANT — it runs the identical `converge_inner_for_undamped_logdet` +
/// chunked `streaming_exact_arrow_log_det` code regardless of K/p — so, exactly as
/// the sibling #1026 streaming-cache test pins its equivalence at small K
/// ("infeasible to exercise [at massive K] in a unit test"), we exercise the full
/// streaming path here at a small, fast, memory-bounded whitened multi-atom fit.
/// The production K=32/p=128 shape is covered upstream by
/// `wide_border_routes_to_streaming_without_fake_gradient_certificate`, which pins that the memory
/// planner refuses the dense direct plan and admits the matrix-free plan at that
/// shape — the two together establish that a wide-border large-K whitened fit
/// routes to, and runs through, the streaming lane without hard-erroring.
#[test]
fn whitened_streaming_criterion_completes() {
    let (n, p, k) = (128usize, 16usize, 8usize);
    let mut term = build_softmax_term(n, p, k);
    let metric = fit_structured_metric(n, p);
    assert!(
        metric.whitens_likelihood(),
        "the fitted structured-residual metric must whiten the likelihood"
    );
    term.set_row_metric(metric).unwrap();

    let target = Array2::<f64>::from_shape_fn((n, p), |(r, c)| {
        0.4 - 0.15 * (r as f64 / n as f64)
            + 0.25 * (c as f64 / p as f64)
            + 0.05 * (((r + c) % 7) as f64)
    });
    let rho = SaeManifoldRho::new(
        -1.0_f64,
        0.7_f64.ln(),
        vec![Array1::<f64>::from_elem(1, 0.0); k],
    );

    let (cost, loss) = term
        .reml_criterion_streaming_exact(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .expect("whitened streaming criterion must complete, not hard-error");
    assert!(
        cost.is_finite(),
        "streaming REML criterion must be finite; got {cost}"
    );
    assert!(
        loss.total().is_finite() && loss.data_fit.is_finite(),
        "whitened loss components must be finite (data_fit={}, total={})",
        loss.data_fit,
        loss.total()
    );
}
