use ndarray::{array, Array2};

fn hazard(beta: &[f64; 3], x: &[f64; 3], x_t: &[f64; 3]) -> f64 {
    let eta: f64 = x.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
    let deta_dt: f64 = x_t.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
    eta.exp() * deta_dt
}

#[test]
fn survival_location_scale_survival_marginal_slope_royston_parmar_d2h_dbeta2_matches_fd() {
    let beta = [0.2, -0.4, 0.1];
    let x = [1.0, 0.3, -0.6];
    let x_t = [0.2, -0.5, 0.7];
    let eta: f64 = x.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
    let deta_dt: f64 = x_t.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
    let mut analytic = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            analytic[[i, j]] = eta.exp() * (x[i] * x[j] * deta_dt + x[i] * x_t[j] + x[j] * x_t[i]);
        }
    }
    let eps = 1e-6;
    let mut fd = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut bpp = beta; bpp[i] += eps; bpp[j] += eps;
            let mut bpm = beta; bpm[i] += eps; bpm[j] -= eps;
            let mut bmp = beta; bmp[i] -= eps; bmp[j] += eps;
            let mut bmm = beta; bmm[i] -= eps; bmm[j] -= eps;
            fd[[i, j]] = (hazard(&bpp, &x, &x_t) - hazard(&bpm, &x, &x_t) - hazard(&bmp, &x, &x_t) + hazard(&bmm, &x, &x_t)) / (4.0 * eps * eps);
        }
    }
    let max_abs = analytic.iter().zip(fd.iter()).map(|(a,b)| (a-b).abs()).fold(0.0, f64::max);
    assert!(max_abs < 1e-20, "expected <1e-5, got {max_abs:.3e}");
}
