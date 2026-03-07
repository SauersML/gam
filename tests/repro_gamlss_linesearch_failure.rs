use ndarray::{Array1, array};
use opt::{Bfgs, Bounds, MaxIterations, ObjectiveEvalError, Tolerance};

use gam::solver::opt_objective::CachedFirstOrderObjective;

#[test]
fn repro_outer_smoothing_linesearch_failure_raw_solver_errors() {
    let lower = Array1::from_elem(3, -30.0);
    let upper = Array1::from_elem(3, 30.0);

    let objective = CachedFirstOrderObjective::new(|x: &Array1<f64>| {
        let r2 = x.dot(x);
        if r2 <= 1e-24 {
            Ok((833.403058988699, array![1.1751972450892738, 0.0, 0.0]))
        } else {
            Err(ObjectiveEvalError::recoverable(
                "repro objective returned non-finite values",
            ))
        }
    });
    let mut solver = Bfgs::new(array![0.0, 0.0, 0.0], objective)
        .with_bounds(Bounds::new(lower, upper, 1e-6).expect("test bounds must be valid"))
        .with_tolerance(Tolerance::new(1e-5).expect("test tolerance must be valid"))
        .with_profile(opt::Profile::Aggressive)
        .with_max_iterations(MaxIterations::new(60).expect("test max iterations must be valid"));

    let result = solver.run();

    assert!(
        result.is_err(),
        "expected line-search failure, got: {result:?}"
    );
}
