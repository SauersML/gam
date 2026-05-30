//! Transformation-normal (conditional transformation) PIT calibration vs.
//! `scipy.stats` — the numerical-CDF ground-truth comparator.
//!
//! gam's transformation-normal family learns a smooth, strictly monotone map
//! `h(Y | x)` that pushes a bounded / awkward response onto the standard normal
//! scale. By *construction* a correctly implemented model is auto-calibrated:
//! if `h(Y | x) ~ N(0, 1)` then the probability-integral transform (PIT) of the
//! training responses is uniform. This is a **distinctive-axis** capability —
//! no mature tool ships the same finite-support I-spline normalization plus the
//! inverse-Hessian machinery gam uses (mlt/tram in R fit transformation models
//! but not this exact SCOP-on-`[0,1]` normal target with gam's penalized basis),
//! so the fragmentation is itself the finding. The honest reference role for
//! `scipy.stats` here is the one place it can serve as exact ground truth: the
//! truncated-normal CDF that turns gam's fitted `(h, lower, upper)` triple into
//! a PIT value, and the Kolmogorov-Smirnov statistic against `U(0, 1)`.
//!
//! We therefore feed gam's own fitted transform — reconstructed from the frozen
//! I-spline / M-spline response basis and the fitted coefficients via the exact
//! SCOP identity the predict path uses
//! (`h = γ₀(x) + Σ_{r≥1} I_r(y)·γ_r(x)² + ε·(y − median)`,
//!  `h' = ε + Σ_{r≥1} M_r(y)·γ_r(x)²`) — into `scipy.stats.norm` and assert three
//! intrinsic correctness properties that together certify the implementation:
//!   (1) monotonicity: `h'(y | x) > 0` at 100 random support points (SCOP must
//!       hold structurally; floating-point cancellation below the floor is a bug);
//!   (2) uniform PIT: KS distance of the training PITs from `U(0, 1)` is small;
//!   (3) self-consistent derivative basis: gam's analytic `h'` matches a central
//!       finite difference of gam's `h` (the I-spline value basis and the
//!       M-spline derivative basis must be mutually consistent).
//!
//! A failing assertion here means gam's transform is non-monotone, mis-calibrated,
//! or its derivative basis is inconsistent — all real bugs, never a loosened bound.

use gam::smooth::TermCollectionDesign;
use gam::terms::basis::{BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense};
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
    let raw_deriv =
        create_ispline_derivative_dense(y_arr.view(), &resp_knots, degree, 1).expect("M-spline basis");
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
fn transformation_normal_pit_is_uniform_on_bounded_support() {
    init_parallelism();

    // ---- synthetic bounded heteroscedastic data ---------------------------
    // y ~ Beta(α(x), β(x)), α(x) = 1 + sin(2πx), β(x) = 2 + cos(2πx), x ~ U(0,1).
    // Both shape parameters move with x to exercise the conditional transform.
    // Clip y into (0.01, 0.99) so it sits strictly inside the finite I-spline
    // support. Marsaglia-Tsang Gamma needs shape >= 1, so α, β are shifted to
    // [1, ...]; the *conditional shape* still varies fully with x.
    const N: usize = 180;
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

    // ---- build an in-memory dataset --------------------------------------
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode dataset");

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

    // Dense covariate design at the training rows (n × p_cov).
    let cov_design: &TermCollectionDesign = &tn.covariate_design;
    let cov_rows = cov_design.design.to_dense();
    let n = y.len();
    assert_eq!(cov_rows.nrows(), n, "covariate design row count mismatch");

    // ---- (1) monotonicity: h' > 0 at 100 random (x_row, y) support points --
    // Reuse the fitted covariate rows but draw fresh response values uniformly
    // inside the I-spline support so we probe the transform off the training
    // responses too.
    let resp_lo = tn.family.response_knots()[0];
    let resp_hi = tn.family.response_knots()[tn.family.response_knots().len() - 1];
    let span = resp_hi - resp_lo;
    let n_probe = 100usize;
    let mut probe_rows = Array2::<f64>::zeros((n_probe, cov_rows.ncols()));
    let mut probe_y = Vec::with_capacity(n_probe);
    for p in 0..n_probe {
        let src = rng.next_u64() as usize % n;
        probe_rows.row_mut(p).assign(&cov_rows.row(src));
        // strictly interior point of the support
        let yy = resp_lo + (0.02 + 0.96 * rng.next_uniform()) * span;
        probe_y.push(yy);
    }
    let probe = reconstruct_transform(&tn, &probe_rows, &probe_y);
    let min_h_prime = probe.h_prime.iter().copied().fold(f64::INFINITY, f64::min);

    // ---- (3) derivative self-consistency: analytic h' vs central difference -
    // Evaluate at training rows with a small step in y; compare gam's analytic
    // h' against the central finite difference of gam's reconstructed h.
    let base = reconstruct_transform(&tn, &cov_rows, &y);
    let delta = 1.0e-4 * span;
    let y_plus: Vec<f64> = y.iter().map(|&v| (v + delta).min(resp_hi - 1e-9)).collect();
    let y_minus: Vec<f64> = y.iter().map(|&v| (v - delta).max(resp_lo + 1e-9)).collect();
    let hp = reconstruct_transform(&tn, &cov_rows, &y_plus);
    let hm = reconstruct_transform(&tn, &cov_rows, &y_minus);
    let mut max_rel_deriv_err = 0.0f64;
    for i in 0..n {
        let fd = (hp.h[i] - hm.h[i]) / (y_plus[i] - y_minus[i]);
        let analytic = base.h_prime[i];
        let rel = (fd - analytic).abs() / analytic.abs().max(1e-8);
        max_rel_deriv_err = max_rel_deriv_err.max(rel);
    }

    // ---- (2) uniform PIT via scipy.stats truncated-normal CDF -------------
    // u_i = (Φ(h_i) − Φ(lower_i)) / (Φ(upper_i) − Φ(lower_i)). scipy.stats.norm
    // is the exact numerical CDF reference; scipy.stats.kstest gives the KS
    // distance of the PITs from U(0,1).
    let r = run_python(
        &[
            Column::new("h", &base.h),
            Column::new("lower", &base.lower),
            Column::new("upper", &base.upper),
        ],
        r#"
from scipy.stats import norm, kstest
import numpy as np
h = np.asarray(df["h"], dtype=float)
lo = np.asarray(df["lower"], dtype=float)
hi = np.asarray(df["upper"], dtype=float)
# Endpoint order is a hard precondition of the finite-support PIT.
assert np.all(hi > lo), "transformation endpoints out of order"
Fh = norm.cdf(h)
Flo = norm.cdf(lo)
Fhi = norm.cdf(hi)
denom = Fhi - Flo
u = np.clip((Fh - Flo) / denom, 1e-12, 1 - 1e-12)
ks = kstest(u, "uniform")
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

    eprintln!(
        "transformation-normal PIT: n={n} edf={edf:.3} min_h'={min_h_prime:.3e} \
         KS={ks:.4} u_mean={u_mean:.4} u_range=[{u_min:.4},{u_max:.4}] \
         max_rel_h'_err={max_rel_deriv_err:.4}"
    );

    // (1) The SCOP parameterization makes h' = ε + Σ M_r γ_r² structurally
    // positive; a value below the derivative floor signals a real monotonicity
    // failure (cancellation / mis-assembled basis), so we require it to clear ε.
    assert!(
        min_h_prime >= TRANSFORMATION_MONOTONICITY_EPS,
        "transform not strictly monotone: min h' = {min_h_prime:.3e} < ε = {:.0e}",
        TRANSFORMATION_MONOTONICITY_EPS
    );

    // (2) A correct finite-support normal transform makes the in-sample PIT
    // uniform. This is the *training* PIT (the model saw these points), so it
    // overfits slightly and we loosen the usual hold-out bound to 0.12 — still
    // far below what any genuine miscalibration (e.g. a wrong normalization
    // constant or a collapsed support) would produce; for context the
    // asymptotic 5% KS critical value at n=180 is ≈ 0.101.
    assert!(
        ks < 0.12,
        "training PIT is not uniform: KS = {ks:.4} (>= 0.12 indicates a calibration pathology)"
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
