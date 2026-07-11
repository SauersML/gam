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
use gam::faer_ndarray::{FaerCholesky, FaerEigh};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
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
        skip_rho_posterior_inference: false,
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
        persist_warm_start_disk: false,
        resource_policy: None,
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
                joint_null_rotation: None,
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

/// **TEST C — small-k 1D basis span at extreme small n (seed-100 regime).**
///
/// Setup: n_train=40, n_test=200, x ~ Uniform([0,1]), y = smooth_1d(x) + N(0,0.05²).
///
/// Dimension accounting (the bit that was previously wrong here): a canonical
/// 1D thin-plate basis with `num_centers = k` has TOTAL basis dimension `k`,
/// exactly as in mgcv's `s(x, bs="tp", k)`. Of those `k` columns, the degree-1
/// polynomial nullspace `{1, x}` takes `M = 2` and the constrained radial block
/// contributes only `k − M` genuine wiggle modes — the side constraint
/// `P(C)ᵀα = 0` removes `M` degrees of freedom from the `k` kernel evaluations,
/// it does NOT add the polynomial block on top of `k` radial columns. So
/// `num_centers = 3` yields a **3-column** basis (2 polynomial + 1 wiggle), not
/// a 5-column one. A single wiggle mode cannot span a full period of sin(2πx):
/// across 50 seeds an unpenalized dim-3 fit tops out at R² ≈ 0.59 (and mgcv's
/// own `bs="tp", k=3` truncation lands at the same ceiling), so an R² ≥ 0.70
/// bar at `num_centers = 3` asserts the mathematically impossible for ANY
/// correct TPS construction — it was calibrated against the phantom dim-5 count.
///
/// The real property this guards is "a small-rank 1D TPS basis of the size that
/// CAN represent this target must actually do so, unpenalized". That size is 5
/// columns (3 wiggle + 2 polynomial), i.e. `num_centers = 5`, where the
/// unpenalized fit clears R² ≥ 0.977 across those same 50 seeds. The 0.70
/// threshold is left UNCHANGED and is a genuine floor: it trips only if the
/// dim-5 basis degrades catastrophically (a rank collapse / degenerate knot
/// placement / mis-scaled radial block), which is the actual construction
/// defect this regression is meant to catch.
///
/// Assertion: held-out R² ≥ 0.70 for the dim-5 (5-center) basis.
#[test]
fn tps_k5_basis_must_span_smooth_univariate_function() {
    let mut rng = StdRng::seed_from_u64(0x5EED_100_BA51C04);
    let xtr_vec = uniform_1d(&mut rng, 40);
    let xte_vec = uniform_1d(&mut rng, 200);
    let xtr = xtr_vec.view().insert_axis(ndarray::Axis(1)).to_owned();
    let xte = xte_vec.view().insert_axis(ndarray::Axis(1)).to_owned();
    let ytr_clean = smooth_1d(&xtr_vec);
    let yte_clean = smooth_1d(&xte_vec);
    let noise = Normal::new(0.0, 0.05).unwrap();
    let ytr = Array1::from_iter(ytr_clean.iter().map(|v| v + noise.sample(&mut rng)));

    // 5 centers ⇒ 5-column basis (2 polynomial {1,x} + 3 constrained radial
    // wiggle modes) — the smallest dimension that can span a full sin(2πx)
    // period. dim-3 (num_centers=3) genuinely cannot, for gam OR mgcv.
    let (basis_train, basis_test) = equal_mass_tps_train_test_designs(&xtr, &xte, 5);

    let beta = solve_ridge(&basis_train, &ytr, 1e-10);
    let yhat_test = basis_test.dot(&beta);
    let r2 = r_squared(&yte_clean, &yhat_test);

    assert!(
        r2 >= 0.70,
        "TPS 5-center (dim-5) basis fails on smooth 1D target at n=40: held-out \
         R² = {:.4} < 0.70. A 3-wiggle + 2-polynomial basis must span a single \
         sin(2πx) period unpenalized — a floor this low trips only on a genuine \
         basis-construction collapse (rank / knot placement / radial scaling).",
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
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
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

/// **TEST F — high-d low-k thin-plate must not collapse onto the parametric block (gam#1091).**
///
/// The `geo_latlon_*_tp_k6` benchmark scenarios fit a 6-D thin-plate smooth
/// (`thinplate(pc1..pc6, centers=6)`) on n≈767 binomial data. Canonical TPS in
/// d=6 needs a degree-3 polynomial nullspace of size C(9,3)=84 ≫ 6 centers, so
/// the builder auto-promotes to the hybrid Duchon spline. With the Matérn-style
/// auto-init length_scale (`max_range / sqrt(n)`, here ≈ data_range/27.7 ≪ the
/// center spacing) the high-order (s=4) hybrid partial-fraction coefficients —
/// which scale as `length_scale^(2(p+s-n))` — underflow to machine epsilon, the
/// constrained radial Gram collapses to ~1e-14 floating-point noise, and
/// `positive_spectral_whitener_from_gram` rejects a rank-0 smooth.
///
/// This test reproduces that exact (d=6, k=6, tiny-length_scale) configuration
/// and asserts the basis builds with **genuine wiggle capacity**: the smooth
/// design, after centering out its column mean (the parametric trend the
/// identifiability constraint removes), must retain singular directions well
/// above floating-point noise. A pre-fix build errored at basis construction;
/// a collapse-but-no-error build would leave a near-zero centered design.
#[test]
fn tps_high_d_low_k_must_not_collapse_onto_parametric_block() {
    // d=6, n≈767, k=6 — the geo_latlon tp_k6 regime.
    let d = 6usize;
    let n = 767usize;
    let centers = 6usize;
    let mut rng = StdRng::seed_from_u64(0x10_91_7E57_BA51C04);
    // Standardized-ish PC features: independent ~unit-SD columns, range ≈ 6.
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array2::<f64>::zeros((n, d));
    for v in x.iter_mut() {
        *v = normal.sample(&mut rng);
    }

    // Auto-init length_scale exactly as the planner derives it: max per-column
    // range over sqrt(n). This is the tiny value that drove the collapse.
    let max_range = (0..d)
        .map(|c| {
            let col = x.column(c);
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            hi - lo
        })
        .fold(0.0_f64, f64::max);
    let auto_length_scale = (max_range / (n as f64).sqrt()).max(1e-6);

    let spec = ThinPlateBasisSpec {
        center_strategy: CenterStrategy::EqualMass {
            num_centers: centers,
        },
        periodic: None,
        length_scale: auto_length_scale,
        double_penalty: false,
        // Default identifiability — this is what routes through
        // positive_spectral_whitener_from_gram, the exact site of the gam#1091
        // ConstraintNullspaceCollapsed error.
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        radial_reparam: None,
    };

    // Pre-fix: this errored with ConstraintNullspaceCollapsed at
    // positive_spectral_whitener_from_gram. Post-fix it must build.
    let built = build_thin_plate_basis(x.view(), &spec).unwrap_or_else(|e| {
        panic!(
            "high-d low-k thin-plate (d={d}, k={centers}, length_scale={auto_length_scale:.3e}) \
             failed to build — basis collapsed onto the parametric block (gam#1091): {e}"
        )
    });

    let design = built.design.to_dense();
    assert_eq!(design.nrows(), n);
    assert!(
        design.ncols() >= 2,
        "promoted smooth design has too few columns ({}) to carry any wiggle",
        design.ncols()
    );

    // Wiggle capacity: center every column (remove the parametric trend the
    // identifiability transform projects out), then measure the spectrum of the
    // centered Gram. A collapsed basis has max eigenvalue at floating-point
    // noise (~1e-14); a real basis has O(n)-scale energy in several directions.
    let mut centered = design.clone();
    for c in 0..centered.ncols() {
        let col = centered.column(c);
        let mean = col.iter().sum::<f64>() / n as f64;
        for i in 0..n {
            centered[[i, c]] -= mean;
        }
    }
    let gram = centered.t().dot(&centered);
    let (eigs, _) = gram
        .eigh(Side::Lower)
        .expect("eigendecomposition of centered smooth Gram");
    let max_eig = eigs.iter().cloned().fold(0.0_f64, f64::max);

    assert!(
        max_eig > 1e-6 * n as f64,
        "promoted high-d low-k thin-plate basis has no wiggle capacity: centered Gram \
         max eigenvalue {max_eig:.3e} ≪ O(n) — the radial block collapsed to noise (gam#1091)"
    );

    // At least two centered directions should carry real energy (beyond a single
    // residual trend), confirming genuine multi-mode wiggle rather than a lone
    // near-degenerate column.
    let tol = 1e-9 * max_eig.max(1.0);
    let live_dirs = eigs.iter().filter(|&&e| e > tol).count();
    assert!(
        live_dirs >= 2,
        "promoted high-d low-k thin-plate basis has only {live_dirs} live centered \
         direction(s); expected ≥2 wiggle modes (gam#1091)"
    );
}

// Placeholder for "REML must find an interior optimum, not the upper boundary"
// was removed: the empty `#[test]` body asserted nothing and was caught by the
// build-time ban on assertion-less tests. Reintroduce with a real fit + bound
// check once the TPS fit API is reconciled.
