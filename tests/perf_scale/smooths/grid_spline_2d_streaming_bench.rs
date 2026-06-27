//! #1031 acceptance bench certifying the O(n) streaming pass at biobank scale
//! (n = 1e7) with a modest K so the dense REML side stays cheap in
//! debug-profile CI.

use std::time::Instant;

use gam::terms::grid_spline_2d::GridSpline2dDesign;

const N: usize = 10_000_000;
const CHECKS_PER_AXIS: usize = 50;
const NOISE_AMP: f64 = 0.3;

fn truth(x1: f64, x2: f64) -> f64 {
    (3.0 * x1).sin() * (2.0 * x2).cos() + 0.4 * x1 * x2
}

#[test]
fn n_10_000_000_streaming_acceptance_bench() {
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut w = Vec::with_capacity(N);
    let a1 = 0.754_877_666_246_692_7;
    let a2 = 0.569_840_290_998_053_2;
    let golden = 0.618_033_988_749_894_9;

    for i in 0..N {
        let u1 = ((i + 1) as f64 * a1).fract();
        let u2 = ((i + 1) as f64 * a2).fract();
        let noise = ((i as f64 * golden).fract() - 0.5) * 2.0 * NOISE_AMP;
        x1.push(u1);
        x2.push(u2);
        y.push(truth(u1, u2) + noise);
        w.push(1.0);
    }

    let build_start = Instant::now();
    let design = GridSpline2dDesign::build(&x1, &x2, &y, &w, 12, [1.0, 1.0]).expect("design");
    let build_elapsed = build_start.elapsed();
    println!(
        "grid_spline_2d n=1e7 streaming build seconds: {:.3}",
        build_elapsed.as_secs_f64()
    );

    let fit_start = Instant::now();
    let fit = design.fit_reml().expect("REML-selected fit");
    let fit_elapsed = fit_start.elapsed();
    println!(
        "grid_spline_2d n=1e7 fit_reml seconds: {:.3}",
        fit_elapsed.as_secs_f64()
    );

    let mut mse = 0.0;
    for i in 0..CHECKS_PER_AXIS {
        let px1 = (i as f64 + 1.0) / (CHECKS_PER_AXIS as f64 + 1.0);
        for j in 0..CHECKS_PER_AXIS {
            let px2 = (j as f64 + 1.0) / (CHECKS_PER_AXIS as f64 + 1.0);
            let (mean, var) = fit.predict(0, px1, px2).expect("predict check point");
            assert!(var > 0.0, "posterior variance must be positive");
            let err = mean - truth(px1, px2);
            mse += err * err;
        }
    }
    mse /= (CHECKS_PER_AXIS * CHECKS_PER_AXIS) as f64;

    let true_var = NOISE_AMP * NOISE_AMP / 3.0;
    assert!(
        mse < 0.5 * true_var,
        "REML fit must recover truth below half the noise variance: mse={mse}, true_var={true_var}"
    );
    assert!(
        fit.sigma2[0] > 0.4 * true_var && fit.sigma2[0] < 2.5 * true_var,
        "profiled sigma2 {} far from true noise variance {true_var}",
        fit.sigma2[0]
    );
}
