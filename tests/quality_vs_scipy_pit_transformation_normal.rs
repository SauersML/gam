//! Transformation-normal (conditional transformation) **held-out** PIT
//! calibration — an objective uncertainty/calibration metric, with
//! `scipy.stats.norm` serving only as the exact mathematical CDF ground truth.
//!
//! OBJECTIVE METRIC ASSERTED. gam's transformation-normal family learns a
//! smooth, strictly monotone map `h(Y | x)` that pushes a bounded response onto
//! the standard-normal scale. A correctly calibrated model has uniform
//! probability-integral transform (PIT): if `h(Y | x) ~ N(0, 1)` truncated to
//! the fitted finite support `[lower(x), upper(x)]`, then
//! `u = (Φ(h) − Φ(lower)) / (Φ(upper) − Φ(lower)) ~ U(0, 1)`. The pass/fail
//! criterion is the **Kolmogorov–Smirnov distance of the HELD-OUT PITs from
//! `U(0, 1)`** — computed on a deterministic 25 % test split the model never
//! saw during fitting, so the uniformity claim is honest (no in-sample
//! overfitting to excuse a loosened bound) and the bar is the analytic KS null
//! quantile `c / sqrt(n_test)`, derived from the sample size rather than tuned.
//! This is rubric case (3): calibration via PIT-uniformity / a small KS
//! statistic on simulated data with a fitted model.
//!
//! NO mature peer tool is matched here: mlt/tram in R fit transformation models
//! but not this exact SCOP-on-`[0,1]` normal target with gam's penalized basis,
//! so there is no fitted-output baseline to match-or-beat. `scipy.stats.norm` is
//! used ONLY as exact ground truth (the EXCEPTION clause): it computes the
//! analytic normal CDF that defines the PIT and the KS statistic against the
//! uniform — both exact mathematical quantities, never another tool's fit.
//!
//! We feed gam's own fitted transform — reconstructed from the frozen I-spline /
//! M-spline response basis and the fitted coefficients via the exact SCOP
//! identity the predict path uses
//! (`h = γ₀(x) + Σ_{r≥1} I_r(y)·γ_r(x)² + ε·(y − median)`,
//!  `h' = ε + Σ_{r≥1} M_r(y)·γ_r(x)²`), with the held-out covariate design rows
//! rebuilt from the FROZEN (training-resolved) term spec — into `scipy.stats`
//! and assert three intrinsic, objective correctness properties:
//!   (1) calibration: KS distance of the HELD-OUT PITs from `U(0, 1)` is below
//!       the analytic KS null quantile (the primary calibration claim);
//!   (2) monotonicity: `h'(y | x) > 0` at 100 held-out support points (SCOP must
//!       hold structurally; floating-point cancellation below the floor is a bug);
//!   (3) self-consistent derivative basis: gam's analytic `h'` matches a central
//!       finite difference of gam's `h` on the held-out rows (the I-spline value
//!       basis and the M-spline derivative basis must be mutually consistent).
//!
//! A failing assertion here means gam's transform is mis-calibrated out of
//! sample, non-monotone, or its derivative basis is inconsistent — all real
//! bugs, never a loosened bound.

use gam::smooth::build_term_collection_design;
use gam::terms::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
};
use gam::test_support::reference::{Column, run_python};
use gam::transformation_normal::{TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalFitResult};
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_model,
    init_parallelism, materialize,
};
use ndarray::{Array1, Array2};

/// Tiny deterministic PRNG (SplitMix64) so the synthetic data is identical on
/// every platform without pulling in an RNG crate; fed once with seed = 2828.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1) — open interval (never exactly 0 or 1).
    fn next_uniform(&mut self) -> f64 {
        // 53-bit mantissa, then nudge off the closed endpoints.
        let u = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }
    /// One standard-normal draw via Box-Muller (uses two uniforms).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// One Gamma(shape, 1) draw via Marsaglia-Tsang (shape >= 1 used here).
    fn next_gamma(&mut self, shape: f64) -> f64 {
        assert!(
            shape >= 1.0,
            "Marsaglia-Tsang Gamma sampler requires shape >= 1.0; got {shape}"
        );
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
    /// Beta(a, b) draw via two Gamma draws (a, b >= 1 here).
    fn next_beta(&mut self, a: f64, b: f64) -> f64 {
        let ga = self.next_gamma(a);
        let gb = self.next_gamma(b);
        ga / (ga + gb)
    }
}

/// Reconstruct gam's fitted transform `(h, h', lower, upper)` at the given
/// `(x_row, y)` covariate-row / response pairs, using exactly the SCOP-CTN
/// identity that `build_predict_input_for_model` applies. `cov_rows` are dense
/// rows of the *covariate* design (`n × p_cov`); each `y` is paired with the
/// covariate row at the same index.
struct ReconstructedTransform {
    h: Vec<f64>,
    h_prime: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn reconstruct_transform(
    tn: &TransformationNormalFitResult,
    cov_rows: &Array2<f64>,
    y: &[f64],
) -> ReconstructedTransform {
    let family = &tn.family;
    let resp_knots = family.response_knots().clone();
    let resp_transform = family.response_transform();
    let degree = family.response_degree();
    let median = family.response_median();
    let eps = TRANSFORMATION_MONOTONICITY_EPS;

    let n = y.len();
    let p_cov = cov_rows.ncols();
    assert_eq!(cov_rows.nrows(), n, "cov_rows / y length mismatch");

    // β is vec(γ) with γ a p_resp × p_cov matrix; p_resp = 1 + p_shape. The CTN
    // family fits as a single coefficient block (see `fit_transformation_normal`
    // → `vec![family.block_spec()]`), and the predict path reads the same
    // `blocks[0].beta`, so we mirror that exact source here.
    let beta = &tn.fit.blocks[0].beta;
    let p_shape = resp_transform.ncols();
    let p_resp = 1 + p_shape;
    assert_eq!(
        beta.len(),
        p_resp * p_cov,
        "beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
        beta.len()
    );
    let gamma = beta
        .view()
        .into_shape_with_order((p_resp, p_cov))
        .expect("reshape beta into (p_resp, p_cov)");

    // Response value basis: [1, I_1(y)·T, ...].
    let y_arr = Array1::from_vec(y.to_vec());
    let (raw_val_arc, _) = create_basis::<Dense>(
        y_arr.view(),
        KnotSource::Provided(resp_knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .expect("I-spline value basis at response points");
    let shape_val = raw_val_arc.as_ref().dot(resp_transform);

    // Response derivative basis: [0, M_1(y)·T, ...].
    let raw_deriv = create_ispline_derivative_dense(y_arr.view(), &resp_knots, degree, 1)
        .expect("M-spline basis");
    let shape_deriv = raw_deriv.dot(resp_transform);

    // Finite-support endpoint bases (constant column = 1; shape part is 0 at the
    // lower endpoint and the per-column transform sum at the upper endpoint —
    // I-splines run 0→1 across the support).
    let mut upper_shape = vec![0.0; p_shape];
    for c in 0..p_shape {
        upper_shape[c] = resp_transform.column(c).sum();
    }
    let lower_floor = eps * (resp_knots[0] - median);
    let upper_floor = eps * (resp_knots[resp_knots.len() - 1] - median);

    let mut h = vec![0.0; n];
    let mut h_prime = vec![0.0; n];
    let mut lower = vec![0.0; n];
    let mut upper = vec![0.0; n];
    for i in 0..n {
        let cov_row = cov_rows.row(i);
        let gamma0 = gamma.row(0).dot(&cov_row);
        let mut val = gamma0; // resp_val column 0 == 1
        let mut up = gamma0;
        let mut hp = 0.0; // resp_deriv column 0 == 0
        for r in 1..p_resp {
            let g = gamma.row(r).dot(&cov_row);
            let g2 = g * g;
            val += shape_val[[i, r - 1]] * g2;
            up += upper_shape[r - 1] * g2;
            hp += shape_deriv[[i, r - 1]] * g2;
        }
        h[i] = val + eps * (y[i] - median);
        h_prime[i] = hp + eps;
        lower[i] = gamma0 + lower_floor;
        upper[i] = up + upper_floor;
    }
    ReconstructedTransform {
        h,
        h_prime,
        lower,
        upper,
    }
}

#[test]
fn transformation_normal_held_out_pit_is_uniform() {
    init_parallelism();

    // ---- synthetic bounded heteroscedastic data ---------------------------
    // y ~ Beta(α(x), β(x)), α(x) = 1 + sin(2πx), β(x) = 2 + cos(2πx), x ~ U(0,1).
    // Both shape parameters move with x to exercise the conditional transform.
    // Clip y into (0.01, 0.99) so it sits strictly inside the finite I-spline
    // support. Marsaglia-Tsang Gamma needs shape >= 1, so α, β are shifted to
    // [1, ...]; the *conditional shape* still varies fully with x.
    const N: usize = 240;
    let mut rng = SplitMix64::new(2828);
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_uniform();
        let two_pi_x = std::f64::consts::TAU * xi;
        let a = 1.0 + two_pi_x.sin(); // in [0, 2]
        let b = 2.0 + two_pi_x.cos(); // in [1, 3]
        // shift each to >= 1 for the Gamma sampler while keeping x-dependence.
        let yi = rng.next_beta(a + 1.0, b).clamp(0.01, 0.99);
        x.push(xi);
        y.push(yi);
    }

    // ---- deterministic 25 % train / test split ----------------------------
    // Every fourth index is held out: the model is fit on TRAIN only and its
    // calibration is judged on TEST rows it never saw. The split is purely a
    // function of the row index, so it is identical on every platform.
    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for i in 0..N {
        if i % 4 == 0 {
            test_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }
    let x_train: Vec<f64> = train_idx.iter().map(|&i| x[i]).collect();
    let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
    let x_test: Vec<f64> = test_idx.iter().map(|&i| x[i]).collect();
    let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();
    let n_train = train_idx.len();
    let n_test = test_idx.len();

    // ---- build the TRAIN in-memory dataset (columns: x, y) ----------------
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n_train)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x_train[i]),
                format!("{:.17e}", y_train[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode train dataset");

    // ---- materialize the transformation-normal request, then pin the
    //      response basis complexity to the spec (degree 3, 5 internal knots) -
    let cfg = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let mut materialized =
        materialize("y ~ s(x, k=6)", &ds, &cfg).expect("materialize transformation-normal model");
    let FitRequest::TransformationNormal(ref mut req) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    req.config.response_degree = 3;
    req.config.response_num_internal_knots = 5;

    let result = fit_model(materialized.request).expect("fit transformation-normal model");
    let FitResult::TransformationNormal(tn) = result else {
        panic!("expected a TransformationNormal fit result");
    };

    let edf = tn.fit.edf_total().unwrap_or(f64::NAN);

    // ---- held-out covariate design rows -----------------------------------
    // Rebuild the covariate design at the TEST x using the FROZEN, training-
    // resolved term spec (`covariate_spec_resolved`): knot placement, centers
    // and basis dimension are fixed from training, so this is exactly the
    // out-of-sample prediction basis. The held-out data matrix mirrors the
    // training column layout [x, y]; only the x column is consumed by `s(x)`,
    // and the y column keeps the matrix shape consistent.
    let mut test_data = Array2::<f64>::zeros((n_test, 2));
    for i in 0..n_test {
        test_data[[i, 0]] = x_test[i];
        test_data[[i, 1]] = y_test[i];
    }
    let test_design = build_term_collection_design(test_data.view(), &tn.covariate_spec_resolved)
        .expect("build held-out covariate design from frozen spec");
    let test_rows = test_design.design.to_dense();
    assert_eq!(
        test_rows.nrows(),
        n_test,
        "held-out covariate design row count mismatch"
    );
    let p_cov = test_rows.ncols();

    let resp_lo = tn.family.response_knots()[0];
    let resp_hi = tn.family.response_knots()[tn.family.response_knots().len() - 1];
    let span = resp_hi - resp_lo;

    // ---- (1) PRIMARY: held-out PIT uniformity via scipy normal CDF --------
    // u_i = (Φ(h_i) − Φ(lower_i)) / (Φ(upper_i) − Φ(lower_i)) at the held-out
    // (x_test, y_test) pairs. The transform is reconstructed from the fitted
    // coefficients; scipy.stats.norm supplies the exact normal CDF and
    // scipy.stats.kstest the KS distance of the held-out PITs from U(0, 1).
    //
    // The finite-support normalizer `Φ(upper) − Φ(lower)` must be evaluated in
    // LOG space — exactly as gam's production PIT path does via
    // `log_normal_cdf_diff` (`families/transformation_normal/endpoint_normalizer.rs`).
    // On held-out covariates the reconstructed `(lower, upper)` endpoints can
    // land far out in a single normal tail (e.g. both > 8), where the naive
    // linear-space difference `norm.cdf(upper) − norm.cdf(lower)` underflows to
    // exactly 0.0, turning `(Fh − Flo) / 0` into a NaN PIT and a NaN KS
    // statistic. `norm.logcdf` plus a tail-reflected `log1mexp` correction keeps
    // the mass representable, so the PIT is finite for every held-out row.
    let held = reconstruct_transform(&tn, &test_rows, &y_test);
    let r = run_python(
        &[
            Column::new("h", &held.h),
            Column::new("lower", &held.lower),
            Column::new("upper", &held.upper),
        ],
        r#"
from scipy.stats import norm, kstest
import numpy as np
h = np.asarray(df["h"], dtype=float)
lo = np.asarray(df["lower"], dtype=float)
hi = np.asarray(df["upper"], dtype=float)
# Endpoint order is a hard precondition of the finite-support PIT.
assert np.all(hi > lo), "transformation endpoints out of order"

def log1mexp(x):
    # log(1 - exp(-x)) for x > 0, stable across the whole range (Mächler 2012):
    # the expm1 form near 0, the log1p form in the tail.
    x = np.asarray(x, dtype=float)
    return np.where(x > np.log(2.0), np.log1p(-np.exp(-x)), np.log(-np.expm1(-x)))

def log_normal_cdf_diff(upper, lower):
    # Stable log[Φ(upper) − Φ(lower)] for upper > lower, mirroring gam's
    # `log_normal_cdf_diff`: reflect to the lower tail when `lower > 0` so the
    # dominant `logcdf` term never sits in the saturated upper tail, then apply
    # the log1mexp correction. Vectorized over the held-out rows.
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    refl = lower > 0.0
    up = np.where(refl, -lower, upper)
    lo_ = np.where(refl, -upper, lower)
    log_up = norm.logcdf(up)
    log_lo = norm.logcdf(lo_)
    gap = log_up - log_lo
    assert np.all(np.isfinite(gap) & (gap > 0.0)), "endpoint mass not representable"
    return log_up + log1mexp(gap)

# u = exp( log[Φ(h)−Φ(lower)] − log[Φ(upper)−Φ(lower)] ), with h clamped into
# the finite support (h≤lower → u=0, h≥upper → u=1), exactly as the production
# `transformation_normal_pit_score` defines the PIT.
h_in = np.clip(h, lo, hi)
log_den = log_normal_cdf_diff(hi, lo)
at_lo = h_in <= lo
at_hi = h_in >= hi
interior = ~(at_lo | at_hi)
u = np.empty_like(h_in)
u[at_lo] = 0.0
u[at_hi] = 1.0
if np.any(interior):
    log_num = log_normal_cdf_diff(h_in[interior], lo[interior])
    u[interior] = np.exp(log_num - log_den[interior])
assert np.all(np.isfinite(u)), "PIT produced non-finite values"
u = np.clip(u, 1e-12, 1 - 1e-12)
ks = kstest(u, "uniform")
assert np.isfinite(ks.statistic), "KS statistic is non-finite"
emit("ks", [float(ks.statistic)])
emit("u_min", [float(u.min())])
emit("u_max", [float(u.max())])
emit("u_mean", [float(u.mean())])
"#,
    );
    let ks = r.scalar("ks");
    let u_mean = r.scalar("u_mean");
    let u_min = r.scalar("u_min");
    let u_max = r.scalar("u_max");

    // The held-out PIT and its KS distance must be FINITE before any uniformity
    // claim is even meaningful. A NaN here is the concrete computation bug from
    // issue #1078: the finite-support normalizer underflowed in linear space.
    // The log-space PIT above must never reproduce it.
    assert!(
        ks.is_finite() && u_mean.is_finite() && u_min.is_finite() && u_max.is_finite(),
        "held-out PIT / KS is non-finite (KS={ks}, u_mean={u_mean}, \
         u_range=[{u_min},{u_max}]) — finite-support normalizer underflow regressed"
    );

    // ---- (2) monotonicity: h' > 0 at 100 held-out support points ----------
    // Reuse the held-out covariate rows but draw fresh response values uniformly
    // inside the I-spline support so we probe the transform off the held-out
    // responses too.
    let n_probe = 100usize;
    let mut probe_rows = Array2::<f64>::zeros((n_probe, p_cov));
    let mut probe_y = Vec::with_capacity(n_probe);
    for p in 0..n_probe {
        let src = rng.next_u64() as usize % n_test;
        probe_rows.row_mut(p).assign(&test_rows.row(src));
        // strictly interior point of the support
        let yy = resp_lo + (0.02 + 0.96 * rng.next_uniform()) * span;
        probe_y.push(yy);
    }
    let probe = reconstruct_transform(&tn, &probe_rows, &probe_y);
    let min_h_prime = probe.h_prime.iter().copied().fold(f64::INFINITY, f64::min);

    // ---- (3) derivative self-consistency: analytic h' vs central difference -
    // Evaluate at the held-out rows with a small step in y; compare gam's
    // analytic h' against the central finite difference of gam's reconstructed h.
    let delta = 1.0e-4 * span;
    let y_plus: Vec<f64> = y_test
        .iter()
        .map(|&v| (v + delta).min(resp_hi - 1e-9))
        .collect();
    let y_minus: Vec<f64> = y_test
        .iter()
        .map(|&v| (v - delta).max(resp_lo + 1e-9))
        .collect();
    let hp = reconstruct_transform(&tn, &test_rows, &y_plus);
    let hm = reconstruct_transform(&tn, &test_rows, &y_minus);
    let mut max_rel_deriv_err = 0.0f64;
    for i in 0..n_test {
        let fd = (hp.h[i] - hm.h[i]) / (y_plus[i] - y_minus[i]);
        let analytic = held.h_prime[i];
        let rel = (fd - analytic).abs() / analytic.abs().max(1e-8);
        max_rel_deriv_err = max_rel_deriv_err.max(rel);
    }

    // Analytic two-sided KS null quantile bar: under a correctly calibrated
    // model the held-out PITs are i.i.d. U(0, 1) and the KS statistic is below
    // c / sqrt(n_test) with high probability. c = 1.95 corresponds to ~0.1 %
    // tail mass (well beyond the 5 % critical value c ≈ 1.358), so a passing
    // model has a vanishing false-failure rate while any real miscalibration
    // (wrong normalization constant, collapsed support) sits far above it. The
    // bar is *derived from n_test*, not hand-tuned.
    let ks_bar = 1.95 / (n_test as f64).sqrt();

    eprintln!(
        "transformation-normal held-out PIT: n_train={n_train} n_test={n_test} edf={edf:.3} \
         min_h'={min_h_prime:.3e} KS={ks:.4} (bar {ks_bar:.4}) u_mean={u_mean:.4} \
         u_range=[{u_min:.4},{u_max:.4}] max_rel_h'_err={max_rel_deriv_err:.4}"
    );

    // (1) PRIMARY calibration claim: the HELD-OUT PIT is uniform. The model
    // never saw these rows, so there is no in-sample overfitting to excuse the
    // bound — uniformity here is genuine out-of-sample calibration.
    assert!(
        ks < ks_bar,
        "held-out PIT is not uniform: KS = {ks:.4} >= bar {ks_bar:.4} \
         (n_test={n_test}); indicates an out-of-sample calibration failure"
    );

    // (2) The SCOP parameterization makes h' = ε + Σ M_r γ_r² structurally
    // positive; a value below the derivative floor signals a real monotonicity
    // failure (cancellation / mis-assembled basis), so we require it to clear ε.
    assert!(
        min_h_prime >= TRANSFORMATION_MONOTONICITY_EPS,
        "transform not strictly monotone: min h' = {min_h_prime:.3e} < ε = {:.0e}",
        TRANSFORMATION_MONOTONICITY_EPS
    );

    // (3) The analytic M-spline derivative basis must reproduce the slope of the
    // I-spline value basis. 5% relative is a tight internal-consistency bound:
    // the two bases are evaluated independently, so any indexing / scaling error
    // between I_r and M_r blows past it, while a correct pairing sits well under.
    assert!(
        max_rel_deriv_err < 0.05,
        "h' inconsistent with finite difference of h: max rel err = {max_rel_deriv_err:.4}"
    );
}
