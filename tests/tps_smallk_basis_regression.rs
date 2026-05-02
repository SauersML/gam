//! TPS basis-quality regression tests for the small-n / high-k regime.
//!
//! These tests encode the hypothesis that rust's k-knot maximin TPS basis
//! is model-space limited at small n: even given full freedom (unpenalized
//! OLS on the basis), the basis cannot approximate genuinely smooth target
//! functions to better than the threshold the test asserts.
//!
//! If the hypothesis is true, the tests fail RED at the time of writing
//! (basis can't fit smooth functions well at small n). If the basis is
//! later improved (spectral mode selection, hybrid enrichment, etc.) the
//! tests pass GREEN. They then act as permanent regression guards: any
//! future change that degrades small-n smooth-function coverage trips them.
//!
//! Coverage strategy (independent of optimizer / REML / fit_gam):
//!   - Test A: build the basis directly, do unpenalized OLS via Cholesky,
//!     measure held-out test R² on a smooth bivariate target. This isolates
//!     the basis from REML and the outer optimizer.
//!   - Test B: same setup, but ridge-stabilize OLS with a tiny α relative
//!     to the design Gram. This isolates "is it numerical conditioning"
//!     from "is it model-space" — if Test B passes but Test A fails, the
//!     issue is conditioning. If both fail, it's model-space.
//!   - Test C: 1D analog with k=3 knots and n=40, mirroring the seed-100
//!     fuzz scenario (additive_interaction signal style at extreme small n).

use faer::Side;
use gam::basis::{create_thin_plate_spline_basis, create_thin_plate_spline_basis_with_knot_count};
use gam::faer_ndarray::FaerCholesky;
use ndarray::{Array1, Array2};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

// ─── Data generators ──────────────────────────────────────────────────────────

/// Smooth bivariate target on [0,1]^2:  f(x,y) = sin(2π x) + 0.5 (y − 0.5)².
/// This function is in the smooth class TPS is supposed to handle well; an
/// 18-knot basis on n=120 points should be more than enough to span it.
fn smooth_2d(x: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter((0..x.nrows()).map(|i| {
        let a = x[[i, 0]];
        let b = x[[i, 1]];
        (2.0 * std::f64::consts::PI * a).sin() + 0.5 * (b - 0.5).powi(2)
    }))
}

/// Smooth univariate target on [0,1]:  f(x) = sin(2π x) + 0.3 x.
fn smooth_1d(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|a| (2.0 * std::f64::consts::PI * a).sin() + 0.3 * a)
}

fn uniform_2d(rng: &mut StdRng, n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x[[i, 0]] = rng.random_range(0.0..1.0);
        x[[i, 1]] = rng.random_range(0.0..1.0);
    }
    x
}

fn uniform_1d(rng: &mut StdRng, n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|_| rng.random_range(0.0..1.0)))
}

// ─── Linear-algebra helpers ──────────────────────────────────────────────────

/// Solve β = (X'X + ε·I)^{-1} X'y where ε is `relative_ridge` × max diagonal of
/// X'X. With `relative_ridge` ≈ 0 this is OLS modulo conditioning; with a
/// nonzero value, this is the smallest-possible-shrinkage ridge regression.
fn solve_ridge(x: &Array2<f64>, y: &Array1<f64>, relative_ridge: f64) -> Array1<f64> {
    let p = x.ncols();
    let xtx = x.t().dot(x);
    let mut g = xtx.clone();
    let max_diag = xtx.diag().iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let eps = (relative_ridge * max_diag).max(1e-12 * max_diag);
    for i in 0..p {
        g[[i, i]] += eps;
    }
    let xty = x.t().dot(y);
    let chol = g
        .cholesky(Side::Lower)
        .expect("Cholesky factorization of X'X + ε·I");
    chol.solvevec(&xty)
}

/// Coefficient of determination on a held-out set.
fn r_squared(y: &Array1<f64>, yhat: &Array1<f64>) -> f64 {
    let ybar = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - ybar).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(yhat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    if ss_tot <= f64::EPSILON {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// **TEST A — basis-coverage on smooth 2D function, near-OLS.**
///
/// Setup: n_train=120, n_test=300, x ~ Uniform([0,1]²), y = smooth_2d(x) + N(0,0.05²).
/// Build TPS basis with k=18 maximin knots from the training data. Compute
/// β via near-OLS (tiny relative ridge = 1e-10) on the train basis. Predict
/// on the test set using the SAME knots.
///
/// Assertion: held-out R² ≥ 0.85.
///
/// What this catches:
///   - If rust's k=18 maximin TPS basis cannot represent the smooth target
///     even given full freedom, this fails RED.
///   - The result is independent of REML, the outer optimizer, and lambda
///     selection — it's a pure statement about the basis function space.
///
/// Threshold rationale: the target is C^∞ and easily representable by 18
/// well-placed basis functions in 2D. mgcv's `bs='tp', k=18` reaches R²
/// > 0.95 on this kind of problem; rust's basis should clear 0.85 with
/// margin if it spans low-frequency space adequately.
#[test]
fn tps_k18_basis_must_span_smooth_bivariate_function() {
    let mut rng = StdRng::seed_from_u64(0xBA51_C046_E2A6E);
    let xtr = uniform_2d(&mut rng, 120);
    let xte = uniform_2d(&mut rng, 300);
    let ytr_clean = smooth_2d(&xtr);
    let yte_clean = smooth_2d(&xte);
    let noise = Normal::new(0.0, 0.05).unwrap();
    let ytr = Array1::from_iter(ytr_clean.iter().map(|v| v + noise.sample(&mut rng)));

    let (basis_train, knots) = create_thin_plate_spline_basis_with_knot_count(xtr.view(), 18)
        .expect("TPS basis on training data");
    let basis_test = create_thin_plate_spline_basis(xte.view(), knots.view())
        .expect("TPS basis on test data with shared knots");

    let beta = solve_ridge(&basis_train.basis, &ytr, 1e-10);
    let yhat_test = basis_test.basis.dot(&beta);
    let r2 = r_squared(&yte_clean, &yhat_test);

    assert!(
        r2 >= 0.85,
        "TPS k=18 basis fails to span smooth 2D target: held-out R² = {:.4} < 0.85. \
         Basis is model-space limited at small-n / high-k.",
        r2
    );
}

/// **TEST B — conditioning vs model-space discrimination.**
///
/// Same setup as TEST A, but with `relative_ridge = 1e-4`. If the basis is
/// numerically ill-conditioned but mathematically rich enough, a moderate
/// ridge stabilizes the solve and recovers good fit. If the basis itself
/// can't span the target, even a stabilized solve fits poorly.
///
/// Decision matrix:
///   - TEST A pass ∧ TEST B pass  → no problem (current TPS is fine).
///   - TEST A fail ∧ TEST B pass  → conditioning issue, fixable with ridge.
///   - TEST A fail ∧ TEST B fail  → model-space issue, basis must change.
///   - TEST A pass ∧ TEST B fail  → impossible by construction (B is more forgiving).
///
/// Threshold: R² ≥ 0.85 (same as Test A — ridge shouldn't degrade fit on
/// a basis that already works).
#[test]
fn tps_k18_basis_must_span_smooth_bivariate_function_ridge_stabilized() {
    let mut rng = StdRng::seed_from_u64(0xBA51_C046_E2A6E);
    let xtr = uniform_2d(&mut rng, 120);
    let xte = uniform_2d(&mut rng, 300);
    let ytr_clean = smooth_2d(&xtr);
    let yte_clean = smooth_2d(&xte);
    let noise = Normal::new(0.0, 0.05).unwrap();
    let ytr = Array1::from_iter(ytr_clean.iter().map(|v| v + noise.sample(&mut rng)));

    let (basis_train, knots) = create_thin_plate_spline_basis_with_knot_count(xtr.view(), 18)
        .expect("TPS basis on training data");
    let basis_test = create_thin_plate_spline_basis(xte.view(), knots.view())
        .expect("TPS basis on test data with shared knots");

    let beta = solve_ridge(&basis_train.basis, &ytr, 1e-4);
    let yhat_test = basis_test.basis.dot(&beta);
    let r2 = r_squared(&yte_clean, &yhat_test);

    assert!(
        r2 >= 0.85,
        "TPS k=18 basis fails on smooth 2D target even with α=1e-4 ridge stabilization: \
         held-out R² = {:.4} < 0.85. The failure is model-space, not conditioning.",
        r2
    );
}

/// **TEST C — 1D analog at extreme small n (seed-100 regime).**
///
/// Setup: n_train=40, n_test=200, x ~ Uniform([0,1]), y = smooth_1d(x) + N(0,0.05²).
/// k=3 maximin knots — the same per-smooth knot count rust uses for the
/// seed-100 fuzz scenario.
///
/// Assertion: held-out R² ≥ 0.70.
///
/// Threshold rationale: with only 3 knots in 1D, the basis dimension is
/// 3 + 2 polynomial = 5. That's tight for sin(2πx), which has effective
/// dimension ~4 in a low-order polynomial basis. R² ≥ 0.70 is a mild
/// requirement for a well-spanning small-rank basis on this target.
#[test]
fn tps_k3_basis_must_span_smooth_univariate_function() {
    let mut rng = StdRng::seed_from_u64(0x5EED_100_BA51C04);
    let xtr_vec = uniform_1d(&mut rng, 40);
    let xte_vec = uniform_1d(&mut rng, 200);
    let xtr = xtr_vec.view().insert_axis(ndarray::Axis(1)).to_owned();
    let xte = xte_vec.view().insert_axis(ndarray::Axis(1)).to_owned();
    let ytr_clean = smooth_1d(&xtr_vec);
    let yte_clean = smooth_1d(&xte_vec);
    let noise = Normal::new(0.0, 0.05).unwrap();
    let ytr = Array1::from_iter(ytr_clean.iter().map(|v| v + noise.sample(&mut rng)));

    let (basis_train, knots) = create_thin_plate_spline_basis_with_knot_count(xtr.view(), 3)
        .expect("TPS basis on training data");
    let basis_test = create_thin_plate_spline_basis(xte.view(), knots.view())
        .expect("TPS basis on test data with shared knots");

    let beta = solve_ridge(&basis_train.basis, &ytr, 1e-10);
    let yhat_test = basis_test.basis.dot(&beta);
    let r2 = r_squared(&yte_clean, &yhat_test);

    assert!(
        r2 >= 0.70,
        "TPS k=3 basis fails on smooth 1D target at n=40: held-out R² = {:.4} < 0.70. \
         Mirrors the seed-100 fuzz failure regime — basis cannot span smooth \
         functions even with the freedom of unpenalized fit.",
        r2
    );
}

/// **TEST D — REML must find an interior optimum, not the upper boundary.**
///
/// This test will be added in a follow-up once the rust API for fitting a
/// TPS smooth via fit_gam is reconciled with the current PenaltySpec /
/// FitOptions surface. The intended assertion: when fitting a smooth 2D
/// target with TPS k=18 on n=120, the converged log-λ is strictly below
/// the upper rho bound (e.g. < 10) — i.e. REML finds an interior optimum
/// rather than driving smoothing to maximum. Failure indicates the REML
/// surface is monotone-toward-over-smoothing on this basis.
///
/// Placeholder kept here to document the intended coverage.
#[test]
#[ignore = "REML interior-optimum coverage pending fit_gam wiring"]
fn tps_reml_must_find_interior_optimum_on_smooth_bivariate() {}
