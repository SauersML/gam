//! Bug hunt: `sphere_frechet_mean` returns a NON-minimizing point for points
//! spread symmetrically around a great circle of S².
//!
//! The intrinsic Fréchet / Karcher mean must minimize the sum of squared
//! geodesic distances `F(μ) = Σ wᵢ · d_g(μ, pᵢ)²`. For the three vertices of an
//! equilateral triangle on the equator of S², the obvious symmetric candidate
//! is a pole, with `F(pole) = 3·(π/2)² ≈ 7.402`. The routine instead returns one
//! of the *data points* on the equator, whose objective is `0 + 2·(2π/3)² ≈
//! 8.773` — strictly worse. A function that advertises the Fréchet mean cannot
//! return a point when a demonstrably lower-objective point exists.
//!
//! Root cause (`src/geometry/sphere.rs`): the Karcher iteration is seeded only
//! from `sphere_mean_candidates`, which offers (a) the normalized *extrinsic*
//! mean `Σ wᵢ pᵢ / ‖·‖` and (b) the *dominant* eigenvector (± of it) of the
//! second-moment matrix `M = Σ wᵢ pᵢ pᵢᵀ`. For points balanced around a great
//! circle the extrinsic mean is the zero vector (dropped, `ex_norm == 0` at
//! lines ~569-572), and the dominant eigenvector of `M = diag(1.5, 1.5, 0)`
//! lies *in the equatorial plane* — so every seed is on the equator and the
//! descent converges to an equatorial stationary point. The actual Fréchet mean
//! sits on the axis orthogonal to the data spread (the pole), i.e. the
//! *smallest*-eigenvalue eigenvector of `M`, which is never offered as a seed.
//!
//! This test builds the equilateral-triangle-on-equator configuration, takes the
//! Fréchet mean, and asserts its objective does not exceed that of the pole (it
//! should be ≤, since the pole is a feasible competitor).
//!
//! FIXED: `sphere_mean_candidates` (crates/gam-geometry/src/manifolds/sphere.rs)
//! now seeds the Karcher descent from the FULL orthonormal eigenbasis of
//! `M = Σ wᵢ pᵢ pᵢᵀ` (both signs) via `sphere_eigenbasis`, which Gram–Schmidt
//! completes the null space — so the orthogonal axis (the pole) is now an offered
//! seed and the caller keeps the lowest-objective converged result. The returned
//! objective is therefore ≤ the pole's. This file is now a regression guard.

use gam::geometry::sphere::sphere_frechet_mean;
use ndarray::{ArrayView1, ArrayView2, array};

/// Sum of squared geodesic distances on the unit sphere (uniform weights).
fn karcher_objective(points: ArrayView2<'_, f64>, mu: ArrayView1<'_, f64>) -> f64 {
    let mu_norm = (mu.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let mut acc = 0.0;
    for row in points.rows() {
        let dot: f64 = row.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>() / mu_norm;
        let d = dot.clamp(-1.0, 1.0).acos();
        acc += d * d;
    }
    acc
}

#[test]
fn sphere_frechet_mean_is_a_genuine_minimizer_on_an_equatorial_triangle() {
    // Equilateral triangle on the equator: angles 0, 2π/3, 4π/3.
    let s = (3.0_f64).sqrt() / 2.0; // sin(2π/3) = √3/2
    let points = array![[1.0, 0.0, 0.0], [-0.5, s, 0.0], [-0.5, -s, 0.0],];

    let mu = sphere_frechet_mean(points.view(), None, 1.0e-12, 256)
        .expect("spherical Fréchet mean should be identifiable for a triangle on the equator");
    let mu = ndarray::Array1::from(mu);

    // The pole is a feasible competitor; its objective is 3·(π/2)² ≈ 7.402.
    let pole = array![0.0, 0.0, 1.0];
    let f_returned = karcher_objective(points.view(), mu.view());
    let f_pole = karcher_objective(points.view(), pole.view());

    assert!(
        f_returned <= f_pole + 1e-6,
        "sphere_frechet_mean returned a non-minimizing point: objective {f_returned:.6} \
         exceeds the pole's objective {f_pole:.6} (returned mean = {mu:?}). The Fréchet mean \
         must minimize the sum of squared geodesic distances, and a strictly better point \
         (the pole) exists."
    );
}
