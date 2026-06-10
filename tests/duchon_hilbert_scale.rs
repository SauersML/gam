//! TARGET behavior of the redesigned non-periodic Euclidean Duchon penalty:
//! the "Hilbert scale of FUNCTION penalties, all on by default" contract.
//!
//! These tests pin down the EVENTUAL behavior of the redesign now being built.
//! They will FAIL until the core lands — that is honest red, not a bug in the
//! tests. The contract under test:
//!
//! A default `duchon(x, k=...)` smooth (no `magnitude=`) emits a Hilbert scale
//! of FUNCTION penalties, ALL ON BY DEFAULT, with REML deselecting the unused
//! ones:
//!
//!   * `PenaltySource::Primary`               — the exact RKHS curvature Gram in
//!                                               centers space (the native
//!                                               reproducing-norm `(m+s)`-order
//!                                               seminorm).
//!   * `PenaltySource::OperatorTension`       — first-order energy `Σ‖∇f‖²`,
//!                                               collocated on an O(k)
//!                                               farthest-point sample of the
//!                                               data.
//!   * `PenaltySource::OperatorMass`          — amplitude `Σ(f−f̄)²` (centered),
//!                                               on the SAME O(k) sample.
//!   * `PenaltySource::DoublePenaltyNullspace`— global-slope trend ridge on the
//!                                               affine block.
//!
//! Because every block is on by default, the headline contract is that REML can
//! and does deselect the unhelpful blocks so they never hurt truth recovery:
//! the all-on default must MATCH-OR-BEAT mgcv `bs="ds"`, must recover a smooth
//! truth, and must collapse toward the null when `x` is irrelevant. The
//! Hilbert-scale penalties live in centers space, so their matrices are
//! `k×k`-ish and do NOT grow with `n`.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

// ── fixtures ────────────────────────────────────────────────────────────────

/// `n` rows in `[-1, 1]^d`, deterministic from `seed`.
fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

/// The DEFAULT non-periodic Euclidean Duchon spec — the magic cubic smoother a
/// user gets from `duchon(x, k=...)` with no extra dials. The redesign makes
/// this default emit the full Hilbert scale of function penalties; the spec
/// fields here are exactly the defaults the formula path resolves to, so the
/// `DuchonOperatorPenaltySpec::default()` carried in the spec is what selects
/// the all-on behavior at build time.
fn default_duchon_spec(k: usize, d: usize) -> DuchonBasisSpec {
    DuchonBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        // Cubic default `s = (d−1)/2`: 0 in 1-D (→ r³), 0.5 in 2-D. (`power=0.5`
        // in 1-D is invalid — CPD requires `2s < d`.)
        power: (d as f64 - 1.0) / 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

/// The set of ACTIVE penalty sources emitted by a built Duchon basis.
fn active_sources(spec: &DuchonBasisSpec, data: &Array2<f64>) -> Vec<PenaltySource> {
    let built = build_duchon_basis(data.view(), spec).expect("build_duchon_basis succeeded");
    built
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone())
        .collect()
}

fn has_source(sources: &[PenaltySource], target: &PenaltySource) -> bool {
    sources.iter().any(|s| s == target)
}

// ── (1) all four sources on by default ──────────────────────────────────────

/// HEADLINE CONTRACT. A default `duchon(x, k=20)` build (no `magnitude=`) emits
/// the ENTIRE Hilbert scale of function penalties — Primary, OperatorTension,
/// OperatorMass, and DoublePenaltyNullspace — all ACTIVE. This is the "all on by
/// default" promise: REML later deselects whatever does not help, but at build
/// time every block must be present so REML actually has them to choose from.
#[test]
fn all_on_by_default() {
    let data = synthetic_data(220, 1, 7);
    let sources = active_sources(&default_duchon_spec(20, 1), &data);

    for required in [
        PenaltySource::Primary,
        PenaltySource::OperatorTension,
        PenaltySource::OperatorMass,
        PenaltySource::DoublePenaltyNullspace,
    ] {
        assert!(
            has_source(&sources, &required),
            "default duchon(x, k=20) must emit {required:?} as an ACTIVE penalty \
             (all-on-by-default Hilbert scale); active sources were {sources:?}"
        );
    }
}

// ── (2) all-on default ≥ mgcv (REML deselects the unhelpful blocks) ──────────

/// DESELECTION RECOVERS TRUTH. Fit a Gaussian default `duchon(x, k=20)` (all
/// four penalties on) on a smooth sine. REML must deselect the blocks that do
/// not help, so the all-on default both recovers the truth (RMSE clearly below
/// the trivial predictor) AND matches-or-beats mgcv `s(x, bs="ds", k=20,
/// m=c(2,0))` within a 10% accuracy margin. If the always-on tension/mass/ridge
/// blocks ever degraded the fit, this would fail — that is exactly the
/// regression the all-on design must avoid.
#[test]
fn deselection_recovers_truth() {
    init_parallelism();

    // Smooth low-frequency truth: f(x) = sin(2π·x), one period over [0,1], which
    // a k=20 cubic Duchon basis resolves comfortably so REML can drive the error
    // to the noise floor if (and only if) it deselects the unhelpful penalties.
    let n = 200usize;
    let mut rng = StdRng::seed_from_u64(123);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let two_pi = 2.0 * std::f64::consts::PI;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic dataset");
    let x_idx = ds.column_map()["x"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x, k=20)", &ds, &cfg).expect("gam duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian Duchon smooth");
    };

    // Dense interior grid (avoid extrapolation edges).
    let m = 201usize;
    let x_test: Vec<f64> = (0..m)
        .map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0))
        .collect();
    let y_truth: Vec<f64> = x_test.iter().map(|&t| (two_pi * t).sin()).collect();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        gam_fitted.iter().all(|v| v.is_finite()),
        "all-on default Duchon produced non-finite fitted values"
    );

    // Same data fit by mgcv bs="ds", m=c(2,0): the mature Duchon baseline.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));
    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(y ~ s(x, bs = "ds", k = 20, m = c(2, 0)), data = train, method = "REML")
        emit("fitted", as.numeric(predict(m, newdata = grid)))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), m, "mgcv prediction length mismatch");

    let gam_truth_rmse = rmse(&gam_fitted, &y_truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &y_truth);
    eprintln!(
        "duchon-deselection-recovers-truth: n={n} grid={m} sigma=0.05 \
         gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4}"
    );

    // (a) Genuine recovery: a constant predictor scores RMS(sin)≈0.707; a real
    // reconstruction of a single-period sine sits far below. 0.20 is a
    // non-degeneracy floor a deselecting all-on default must clear.
    assert!(
        gam_truth_rmse < 0.20,
        "all-on default Duchon failed to recover sin(2πx): RMSE-vs-truth={gam_truth_rmse:.4} \
         (REML must deselect the unhelpful blocks; trivial predictor ≈ 0.707)"
    );

    // (b) Match-or-beat mgcv on truth recovery within a 10% accuracy margin. The
    // all-on default must be at least as accurate as the mature Duchon — REML
    // deselecting the unused tension/mass/ridge blocks is what makes this hold.
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "all-on default Duchon recovers the truth worse than mgcv: \
         gam RMSE-vs-truth={gam_truth_rmse:.4} > 1.10 * mgcv RMSE-vs-truth={mgcv_truth_rmse:.4} \
         (the always-on penalties must be deselected by REML, not left to hurt)"
    );
}

// ── (3) the term collapses toward the null when x is irrelevant ──────────────

/// NULL RECOVERY. Generate `y` independent of `x` (pure Gaussian noise, no
/// x-effect). The fitted smooth must collapse toward the null: its fitted values
/// are nearly FLAT, i.e. `max|f̂ − mean(f̂)|` is small relative to the noise sd.
/// With every penalty on by default, REML must drive the smooth's amplitude to
/// (near) zero rather than chasing noise — the Hilbert scale collapses the term
/// when `x` carries no signal.
#[test]
fn null_recovery() {
    init_parallelism();

    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(98_765);
    let sigma = 1.0;
    let noise = Normal::new(0.0, sigma).expect("normal");
    // x is a real covariate but y does NOT depend on it.
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise.sample(&mut rng)).collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode null dataset");
    let x_idx = ds.column_map()["x"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x, k=20)", &ds, &cfg).expect("gam duchon null fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the null Duchon smooth");
    };

    // Evaluate the fitted smooth on a dense grid.
    let m = 201usize;
    let x_test: Vec<f64> = (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at null test grid");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        fitted.iter().all(|v| v.is_finite()),
        "null-recovery Duchon produced non-finite fitted values"
    );

    let mean_fit = fitted.iter().sum::<f64>() / m as f64;
    let max_dev = fitted
        .iter()
        .map(|v| (v - mean_fit).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "duchon-null-recovery: n={n} sigma={sigma} max|f_hat - mean(f_hat)|={max_dev:.4} \
         (target: collapses flat, ≪ sigma)"
    );

    // When x is irrelevant the term must collapse: the fitted smooth's swing
    // about its mean is a small fraction of the noise sd. 0.25·sigma is a
    // principled "nearly flat" bar — a smoother that chased noise would wiggle
    // on the order of sigma.
    assert!(
        max_dev < 0.25 * sigma,
        "null-recovery: smooth did not collapse toward the null when x is irrelevant — \
         max|f_hat - mean|={max_dev:.4} is not ≪ sigma={sigma:.4} (REML must shrink the \
         all-on Hilbert scale to ~zero amplitude here)"
    );
}

// ── (4) Hilbert-scale penalty matrices are k×k-ish, independent of n ─────────

/// PENALTY COST IS n-INDEPENDENT. The Hilbert-scale penalties live in centers
/// space, so every emitted penalty matrix is `k×k`-ish (centers + a small affine
/// polynomial block) and its dimension does NOT grow with `n`. Build the basis
/// at n=2_000 and n=40_000 with the same `k`; assert every emitted penalty's
/// `nrows()` is bounded by ~`k + poly`, never by `n`. This is what makes the
/// collocated tension/mass penalties affordable at large scale.
#[test]
fn penalty_cost_is_n_independent() {
    let k = 30usize;
    // Generous upper bound on the coefficient-space dimension: k centers plus a
    // small affine polynomial block (constant + linear in d ≤ a few dims). 4·k
    // can only OVER-count, so a penalty whose size scaled with n (2_000 / 40_000)
    // would blow straight past it.
    let coeff_dim_upper = 4 * k + 8;

    for &n in &[2_000usize, 40_000usize] {
        let data = synthetic_data(n, 1, 0xDEAD ^ n as u64);
        let built = build_duchon_basis(data.view(), &default_duchon_spec(k, 1))
            .expect("build_duchon_basis");

        assert!(
            !built.penalties.is_empty(),
            "n={n}: default Duchon emitted no penalties at all"
        );

        for (idx, pen) in built.penalties.iter().enumerate() {
            let source = built
                .penaltyinfo
                .iter()
                .filter(|info| info.active)
                .nth(idx)
                .map(|info| info.source.clone());
            assert_eq!(
                pen.nrows(),
                pen.ncols(),
                "n={n}: penalty #{idx} ({source:?}) is not square: {}×{}",
                pen.nrows(),
                pen.ncols()
            );
            assert!(
                pen.nrows() <= coeff_dim_upper,
                "n={n}: penalty #{idx} ({source:?}) has dimension {} > {coeff_dim_upper} \
                 (≈k+poly bound); a Hilbert-scale penalty must live in centers space and \
                 NOT grow with n",
                pen.nrows()
            );
            assert!(
                pen.nrows() < n,
                "n={n}: penalty #{idx} ({source:?}) dimension {} is on the order of n — \
                 the penalty cost must be n-independent",
                pen.nrows()
            );
        }
    }
}
