use ndarray::Array1;

use gam::solver::outer_strategy::{
    ClosureObjective, Derivative, EfsEval, HessianResult, OuterCapability, OuterConfig, OuterEval,
};

/// Verify that `run_outer` propagates the error when the objective only
/// produces finite values at the origin and errors everywhere else — the
/// pathological case that triggers a line-search failure inside BFGS.
#[test]
fn repro_outer_smoothing_linesearch_failure_via_run_outer() {
    let mut dummy = ();
    let mut obj = ClosureObjective {
        state: &mut dummy,
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 3,
            all_penalty_like: true,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
        },
        cost_fn: |ctx: &mut &mut (), x: &Array1<f64>| {
            let r2 = x.dot(x);
            if r2 <= 1e-24 {
                Ok(833.403058988699)
            } else {
                Ok(f64::INFINITY)
            }
        },
        eval_fn: |ctx: &mut &mut (), x: &Array1<f64>| {
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
        reset_fn: None::<fn(&mut &mut ())>,
        efs_fn: None::<
            fn(
                &mut &mut (),
                &Array1<f64>,
            ) -> Result<EfsEval, gam::solver::estimate::EstimationError>,
        >,
    };

    let config = OuterConfig {
        tolerance: 1e-5,
        max_iter: 60,
        fd_step: 1e-5,
        bounds: None,
        seed_config: gam::solver::seeding::SeedConfig {
            max_seeds: 1,
            screening_budget: 1,
            ..Default::default()
        },
        rho_bound: 30.0,
        heuristic_lambdas: None,
        initial_rho: Some(Array1::zeros(3)),
        fallback_policy: Default::default(),
    };

    let result = gam::solver::outer_strategy::run_outer(&mut obj, &config, "repro-linesearch");

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
