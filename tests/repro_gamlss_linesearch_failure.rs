use ndarray::{Array1, array};
use wolfe_bfgs::Bfgs;

#[test]
fn repro_outer_smoothing_linesearch_failure_raw_solver_errors() {
    let lower = Array1::from_elem(3, -30.0);
    let upper = Array1::from_elem(3, 30.0);

    let mut solver = Bfgs::new(array![0.0, 0.0, 0.0], |x: &Array1<f64>| {
        let r2 = x.dot(x);
        if r2 <= 1e-24 {
            (833.403058988699, array![1.1751972450892738, 0.0, 0.0])
        } else {
            (f64::INFINITY, Array1::from_elem(3, f64::NAN))
        }
    })
    .with_bounds(lower, upper, 1e-6)
    .with_tolerance(1e-5)
    .with_fp_tolerances(1e2, 1e2)
    .with_accept_flat_midpoint_once(true)
    .with_jiggle_on_flats(true, 1e-3)
    .with_multi_direction_rescue(true)
    .with_max_iterations(60);

    let result = solver.run();

    assert!(
        result.is_err(),
        "expected line-search failure, got: {result:?}"
    );
}
