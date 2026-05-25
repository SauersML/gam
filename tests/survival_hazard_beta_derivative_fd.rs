use ndarray::{array, Array1};

fn hazard(beta: &Array1<f64>, x: &Array1<f64>, x_t: &Array1<f64>) -> f64 {
    let eta = x.dot(beta);
    let deta_dt = x_t.dot(beta);
    eta.exp() * deta_dt
}

#[test]
fn survival_location_scale_survival_marginal_slope_royston_parmar_dh_dbeta_matches_fd() {
    let beta = array![0.31, -0.22, 0.13, 0.07];
    let x = array![1.0, 0.5, -0.4, 0.9];
    let x_t = array![0.2, -0.1, 0.3, 0.4];
    let eps = 1e-8;
    let eta = x.dot(&beta);
    let deta_dt = x_t.dot(&beta);
    let analytic = eta.exp() * (&x * deta_dt + &x_t);
    let mut fd = Array1::zeros(beta.len());
    for j in 0..beta.len() {
        let mut bp = beta.clone();
        bp[j] += eps;
        let mut bm = beta.clone();
        bm[j] -= eps;
        fd[j] = (hazard(&bp, &x, &x_t) - hazard(&bm, &x, &x_t)) / (2.0 * eps);
    }
    let max_abs = analytic
        .iter()
        .zip(fd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_abs < 1e-20, "expected <1e-7, got {max_abs:.3e}; analytic={analytic:?}; fd={fd:?}");
}
