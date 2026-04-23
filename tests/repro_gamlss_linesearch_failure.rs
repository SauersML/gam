use ndarray::Array1;

use gam::solver::outer_strategy::{Derivative, EfsEval, HessianResult, OuterEval, OuterProblem};

/// Verify that `run_outer` propagates the error when the objective only
/// produces finite values at the origin and errors everywhere else — the
/// pathological case that triggers a line-search failure inside BFGS.
#[test]
fn repro_outer_smoothing_linesearch_failure_via_run_outer() {
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_tolerance(1e-5)
        .with_max_iter(60)
        .with_seed_config(gam::solver::seeding::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            screen_max_inner_iterations: 1,
            ..Default::default()
        })
        .with_rho_bound(30.0)
        .with_initial_rho(Array1::zeros(3));

    let mut obj =
        problem.build_objective(
            0i32,
            |ctx: &mut i32, x: &Array1<f64>| {
                *ctx += 1;
                let r2 = x.dot(x);
                if r2 <= 1e-24 {
                    Ok(833.403058988699)
                } else {
                    Ok(f64::INFINITY)
                }
            },
            |ctx: &mut i32, x: &Array1<f64>| {
                *ctx += 1;
                let r2 = x.dot(x);
                if r2 <= 1e-24 {
                    Ok(OuterEval {
                        cost: 833.403058988699,
                        gradient: ndarray::array![1.1751972450892738, 0.0, 0.0],
                        hessian: HessianResult::Unavailable,
                    })
                } else {
                    Ok(OuterEval {
                        cost: f64::INFINITY,
                        gradient: Array1::zeros(3),
                        hessian: HessianResult::Unavailable,
                    })
                }
            },
            None::<fn(&mut i32)>,
            None::<
                fn(
                    &mut i32,
                    &Array1<f64>,
                ) -> Result<EfsEval, gam::solver::estimate::EstimationError>,
            >,
        );

    let result = problem.run(&mut obj, "repro-linesearch");

    // The objective only has a finite value at the origin with a nonzero
    // gradient — any step away returns INFINITY. The solver must either
    // converge trivially or propagate the failure.
    match result {
        Err(_) => { /* expected: solver could not improve */ }
        Ok(r) => {
            // If it "converges" it should be at the origin with the initial cost.
            assert!(
                r.rho.iter().all(|&v: &f64| v.abs() <= 1e-6),
                "expected convergence at origin, got rho={:?}",
                r.rho
            );
        }
    }
}
