//! Smoke test: bernoulli marginal-slope must complete quickly on a small,
//! well-conditioned problem.
//!
//! This is a guard against regressions in the inner-Newton / outer-κ
//! interplay that turn small problems into slow problems. The large-scale
//! reproducers (`tests/large_scale_margslope_repro.rs`,
//! `tests/inference/optimization/margslope_inner_pirls_scaling.rs`) sweep n up
//! to 100k+; they are NOT skipped — `#[ignore]` is a hard build abort here
//! (`build.rs` "#[ignore] test" rule, enforcing SPEC.md's ban on the XFAIL
//! pattern) — so this file is not a stand-in for a disabled sibling. It is the
//! cheap always-on guard: a single n=2000 fit asserting both convergence and a
//! wall-clock budget that is generous for a healthy solver but tight enough to
//! catch a slow-loop regression like the CTN exact-fn rejection cycle that
//! recently cost ≥14h of CI.

use gam::ResourcePolicy;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineKnotSpec, OneDimensionalBoundary,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

const SEED: u64 = 0x5CA1_AB1E_5C0F_E5A1;

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn run_one(flex: bool, label: &str, budget_s: f64) {
    // A slow-loop regression is diagnosed from the per-cycle instrumentation
    // the flex path already emits — the intercept seed short-circuit counters
    // and the cell-moment LRU hit rate — and the `log` facade drops every one
    // of those records until a backend is installed. Without this call the
    // budget assertion below can only report THAT the fit was slow, never
    // which of the two loops it was.
    gam::test_support::install_diagnostic_logger();
    gam::init_parallelism();
    let (data, spec) = build_problem(2000, flex);
    let options = BlockwiseFitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let policy = ResourcePolicy::default_library();
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options,
        kappa_options,
        policy,
    });

    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();

    let out = match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("{label}: wrong FitResult variant"),
        Err(e) => panic!("{label}: fit failed at n=2000 flex={flex}: {e}"),
    };

    eprintln!(
        "[{label}] n=2000 flex={flex} elapsed_s={elapsed:.3} outer_iters={} inner_cycles={} converged=certified",
        out.fit.outer_iterations, out.fit.inner_cycles
    );

    // Fit existence is the sealed convergence proof (SPEC 20).
    assert!(
        elapsed < budget_s,
        "{label}: small good-condition fit took {elapsed:.2}s, expected <{budget_s:.0}s — slow-loop regression?"
    );
}

#[test]
fn margslope_rigid_small_good_condition_completes_quickly() {
    // Rigid probit: closed-form vectorized inner solve. n=2000 should
    // complete in well under a second of compute on any reasonable
    // hardware; allow 30s to absorb CI-runner variance.
    run_one(false, "MS-RIGID-SMOKE", 30.0);
}

#[test]
fn margslope_flex_small_good_condition_completes_quickly() {
    // Flex probit: cubic score_warp + link_dev deviation blocks exercise
    // the per-row sextic-kernel cell evaluator at every inner-PIRLS
    // iteration. At n=2000 this is the large-scale production code path on
    // small data — must still finish well within 60s on a healthy solver.
    run_one(true, "MS-FLEX-SMOKE", 60.0);
}
