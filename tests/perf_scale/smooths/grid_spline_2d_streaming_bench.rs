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

