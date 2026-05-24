//! TPS basis-quality regression tests for the small-n / high-k regime.
//!
//! These tests encode the hypothesis that rust's k-center equal-mass TPS basis
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
//!     from "is it model-space" - if Test B passes but Test A fails, the
//!     issue is conditioning. If both fail, it's model-space.
//!   - Test C: 1D analog with k=3 knots and n=40, mirroring the seed-100
//!     fuzz scenario (additive_interaction signal style at extreme small n).

use faer::Side;
use gam::basis::{
    BasisMetadata, CenterStrategy, SpatialIdentifiability, ThinPlateBasisSpec,
    build_thin_plate_basis,
};
use gam::estimate::{AdaptiveRegularizationOptions, FitOptions};
use gam::faer_ndarray::FaerCholesky;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design,
};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, s};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Normal};

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

fn skewed_2d(rng: &mut StdRng, n: usize) -> Array2<f64> {
    let exp = Exp::new(1.0).expect("valid exponential rate");
    let mut x = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x[[i, 0]] = exp.sample(rng);
        x[[i, 1]] = exp.sample(rng);
    }
    for col in 0..2 {
        let mean = x.column(col).iter().sum::<f64>() / n as f64;
        for i in 0..n {
            x[[i, col]] -= mean;
        }
    }
    x
}

fn standardize(mut y: Array1<f64>) -> Array1<f64> {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    for v in y.iter_mut() {
        *v -= mean;
    }
    let var = y.iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
    let sd = var.sqrt();
    if sd > 1e-12 {
        for v in y.iter_mut() {
            *v /= sd;
        }
    }
    y
}

fn zscore_train(mut train: Array2<f64>) -> Array2<f64> {
    for col in 0..train.ncols() {
        let mean = train.column(col).iter().sum::<f64>() / train.nrows() as f64;
        let var = train
            .column(col)
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / train.nrows().max(1) as f64;
        let sd = var.sqrt().max(1e-12);
        for i in 0..train.nrows() {
            train[[i, col]] = (train[[i, col]] - mean) / sd;
        }
    }
    train
}

fn sawtooth_signal(x: &Array1<f64>) -> Array1<f64> {
    let freq = 3.5;
    let two_pi = 2.0 * std::f64::consts::PI;
    x.mapv(|v| {
        let z = freq * v / two_pi;
        2.0 * (z - (0.5 + z).floor())
    })
}

fn polynomial_signal(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 0.35 * v.powi(4) - 0.8 * v.powi(3) + 0.25 * v.powi(2) + 0.5 * v)
}

fn equal_mass_tps_train_test_designs(
    x_train: &Array2<f64>,
    x_test: &Array2<f64>,
    centers: usize,
) -> (Array2<f64>, Array2<f64>) {
    let train_spec = ThinPlateBasisSpec {
        center_strategy: CenterStrategy::EqualMass {
            num_centers: centers,
        },
        periodic: None,
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::None,
        radial_reparam: None,
    };
    let train = build_thin_plate_basis(x_train.view(), &train_spec)
        .expect("training TPS basis with equal-mass centers");
    let (fit_centers, radial_reparam) = match &train.metadata {
        BasisMetadata::ThinPlate {
            centers,
            radial_reparam,
            ..
        } => (centers.clone(), radial_reparam.clone()),
        _ => panic!("expected ThinPlate metadata"),
    };
    let test_spec = ThinPlateBasisSpec {
        center_strategy: CenterStrategy::UserProvided(fit_centers),
        periodic: None,
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::None,
        radial_reparam,
    };
    let test = build_thin_plate_basis(x_test.view(), &test_spec)
        .expect("test TPS basis with frozen centers");
    (train.design.to_dense(), test.design.to_dense())
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

fn standard_fit_options(max_iter: usize) -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter,
        tol: 1e-7,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: Some(AdaptiveRegularizationOptions {
            enabled: false,
            ..Default::default()
        }),
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

fn marginal_tps_spec(num_centers: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: (0..2)
            .map(|feature| SmoothTermSpec {
                name: format!("x{feature}_tps"),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![feature],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::EqualMass { num_centers },
                        periodic: None,
                        length_scale: 1.0,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::default(),
                        radial_reparam: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            })
            .collect(),
    }
}

fn seed118_style_training_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(0x118_BA51_C046E);
    let x_all_raw = skewed_2d(&mut rng, 150);
    let y0_all = standardize(sawtooth_signal(&x_all_raw.column(0).to_owned()));
    let y1_all = standardize(polynomial_signal(&x_all_raw.column(1).to_owned()));
    let y_all_clean = y0_all + &(0.75 * y1_all);

    let xtr_raw = x_all_raw.slice(s![0..120, ..]).to_owned();
    let xtr = zscore_train(xtr_raw);
    let ytr_clean = y_all_clean.slice(s![0..120]).to_owned();
    let noise = Normal::new(0.0, 0.02).unwrap();
    let ytr = Array1::from_iter(ytr_clean.iter().map(|v| v + noise.sample(&mut rng)));
    (xtr, ytr, ytr_clean)
}

fn seed118_basis_oracle_r2() -> (Array2<f64>, Array1<f64>, Array1<f64>, f64) {
    let (xtr, ytr, ytr_clean) = seed118_style_training_data();
    let design = build_term_collection_design(xtr.view(), &marginal_tps_spec(18))
        .expect("formula-path marginal TPS design");
    let x_train = design.design.to_dense();
    let beta = solve_ridge(&x_train, &ytr, 1e-10);
    let yhat_train = x_train.dot(&beta);
    let r2 = r_squared(&ytr_clean, &yhat_train);
    (xtr, ytr, ytr_clean, r2)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// **TEST A — basis-coverage on smooth 2D function, near-OLS.**
///
/// Setup: n_train=120, n_test=300, x ~ Uniform([0,1]²), y = smooth_2d(x) + N(0,0.05²).
/// Build TPS basis with k=18 equal-mass centers from the training data. Compute
/// β via near-OLS (tiny relative ridge = 1e-10) on the train basis. Predict
/// on the test set using the SAME knots.
///
/// Assertion: held-out R² ≥ 0.85.
///
/// What this catches:
///   - If rust's k=18 equal-mass TPS basis cannot represent the smooth target
///     even given full freedom, this fails RED.
///   - The result is independent of REML, the outer optimizer, and lambda
///     selection — it's a pure statement about the basis function space.
///
/// Threshold rationale: the target is C^∞ and easily representable by 18
/// well-placed basis functions in 2D. mgcv's `bs='tp', k=18` reaches R²
/// > 0.95 on this kind of problem; rust's basis should clear 0.85 with
/// > margin if it spans low-frequency space adequately.
#[test]
fn tps_k18_basis_must_span_smooth_bivariate_function() {
    let mut rng = StdRng::seed_from_u64(0xBA51_C046_E2A6E);
    let xtr = uniform_2d(&mut rng, 120);
    let xte = uniform_2d(&mut rng, 300);
    let ytr_clean = smooth_2d(&xtr);
    let yte_clean = smooth_2d(&xte);
    let noise = Normal::new(0.0, 0.05).unwrap();
    let ytr = Array1::from_iter(ytr_clean.iter().map(|v| v + noise.sample(&mut rng)));

    let (basis_train, basis_test) = equal_mass_tps_train_test_designs(&xtr, &xte, 18);

    let beta = solve_ridge(&basis_train, &ytr, 1e-10);
    let yhat_test = basis_test.dot(&beta);
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

    let (basis_train, basis_test) = equal_mass_tps_train_test_designs(&xtr, &xte, 18);

    let beta = solve_ridge(&basis_train, &ytr, 1e-4);
    let yhat_test = basis_test.dot(&beta);
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
/// k=3 equal-mass centers - the same per-smooth center count rust uses for the
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

    let (basis_train, basis_test) = equal_mass_tps_train_test_designs(&xtr, &xte, 3);

    let beta = solve_ridge(&basis_train, &ytr, 1e-10);
    let yhat_test = basis_test.dot(&beta);
    let r2 = r_squared(&yte_clean, &yhat_test);

    assert!(
        r2 >= 0.70,
        "TPS k=3 basis fails on smooth 1D target at n=40: held-out R² = {:.4} < 0.70. \
         Mirrors the seed-100 fuzz failure regime — basis cannot span smooth \
         functions even with the freedom of unpenalized fit.",
        r2
    );
}

/// **TEST D — seed-118 marginal geometry, without REML.**
///
/// The seed-118 fuzz failure is not a single 2D TPS; it is two independent
/// one-dimensional TPS smooths with k=18 on skewed covariates:
///
///   y ~ s(x0, type=tps, centers=18) + s(x1, type=tps, centers=18)
///
/// This test builds exactly that term-collection design and fits it by near-OLS
/// on the training rows of a deterministic additive sawtooth-plus-polynomial
/// target. We assert a moderate training-space R² because this regression
/// isolates basis span. A skewed tail holdout is an extrapolation diagnostic,
/// not a basis-capacity diagnostic, and would confound the question this test
/// is meant to answer.
#[test]
fn tps_two_marginal_k18_blocks_must_span_seed118_style_additive_signal() {
    let (_, _, _, r2) = seed118_basis_oracle_r2();

    assert!(
        r2 >= 0.70,
        "Two marginal TPS k=18 blocks fail to span seed-118-style additive signal: \
         training-space R² = {:.4} < 0.70. This localizes the failure to marginal \
         basis capacity rather than REML.",
        r2
    );
}

/// **TEST E — full REML fit must preserve a seed-118-style additive signal.**
///
/// The basis-only test above gives the achievable envelope for the formula-path
/// TPS design. This test adds the real Gaussian REML/PIRLS smoothing-selection
/// layer and requires it to stay close to that envelope. If this fails while the
/// basis-only test passes, the mechanism is smoothing-parameter selection rather
/// than center placement or basis capacity.
#[test]
fn tps_reml_fit_must_not_oversmooth_seed118_style_additive_signal() {
    let (xtr, ytr, ytr_clean, oracle_r2) = seed118_basis_oracle_r2();
    let weights = Array1::ones(ytr.len());
    let offset = Array1::zeros(ytr.len());

    let fitted = gam::smooth::fit_term_collection_forspec(
        xtr.view(),
        ytr.view(),
        weights.view(),
        offset.view(),
        &marginal_tps_spec(18),
        LikelihoodFamily::GaussianIdentity,
        &standard_fit_options(80),
    )
    .expect("seed-118-style marginal TPS REML fit should succeed");

    let yhat = fitted.design.design.dot(&fitted.fit.beta);
    let r2 = r_squared(&ytr_clean, &yhat);
    assert!(
        r2 >= oracle_r2 - 0.10,
        "REML over-smoothed a seed-118-style additive signal relative to the basis envelope: \
         fit R² = {:.4}, basis-oracle R² = {:.4}, gap = {:.4}, log_lambdas={:?}, \
         reml_score={:.6e}, outer_gradient_norm={:?}.",
        r2,
        oracle_r2,
        oracle_r2 - r2,
        fitted.fit.log_lambdas.to_vec(),
        fitted.fit.reml_score,
        fitted.fit.outer_gradient_norm
    );
    assert!(
        r2 >= 0.65,
        "REML fit on seed-118-style additive signal is too weak in absolute terms: R² = {:.4}",
        r2
    );
}

// Placeholder for "REML must find an interior optimum, not the upper boundary"
// was removed: the empty `#[test]` body asserted nothing and was caught by the
// build-time ban on assertion-less tests. Reintroduce with a real fit + bound
// check once the TPS fit API is reconciled.
