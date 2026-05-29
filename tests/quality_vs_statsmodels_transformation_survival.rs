//! End-to-end quality: gam's transformation AFT survival model must reproduce
//! the predicted survival curve `S(t | x)` of a mature parametric reference,
//! across a dense covariate-by-time grid — not merely match a single fitted
//! coefficient.
//!
//! Reference. There is no single mature "transformation-AFT" Python API, but a
//! correct parametric Weibull AFT fit is *uniquely* determined by the censored
//! likelihood, and `scipy` (optimizer + `scipy.stats`/`np` for the analytic
//! Weibull survival function) is the exact ground truth for that likelihood.
//! We fit the Weibull AFT log-likelihood
//!
//!     log T = mu + b1*x1 + b2*x2 + sigma*W,   W ~ standard Gumbel(min),
//!     <=>  S(t|x) = exp( -(t / scale_i)^shape ),
//!          scale_i = exp(mu + b1*x1 + b2*x2),  shape = 1/sigma,
//!
//! to convergence with `scipy.optimize.minimize` on the *identical* censored
//! data handed to gam, then evaluate `S(t|x)` on the shared grid. This is the
//! end-user-facing prediction (beta -> design -> transformed quantile ->
//! survival CDF), so it is the right quantity to benchmark.
//!
//! gam parameterizes the same model on the log-cumulative-hazard / Royston-
//! Parmar net scale with a parametric Weibull baseline:
//!
//!     S(t|x) = exp( -(t/lambda)^p * exp(gamma . x) ),
//!
//! where `lambda` (scale) and `p` (shape) are recovered into `baseline_cfg` and
//! `gamma` is the covariate-block log-hazard-ratio vector (the tail of `beta`).
//! The Weibull is simultaneously PH and AFT, so the two parameterizations
//! describe the *same* `S(t|x)` surface: `mu = log lambda`, `b_j = -gamma_j/p`.
//! We compare the survival surfaces directly, so the parameterization map
//! cancels and any real divergence in gam's prediction pathway is exposed.
//!
//! Bound. Both engines fit the same parametric likelihood on the same data, so
//! the predicted `S(t|x)` surface is *deterministic* up to optimizer tolerance;
//! the only slack is small-sample (n=250) convergence noise between two
//! independent optimizers. relative_l2 <= 0.015 and Pearson >= 0.998 over the
//! 10x10 covariate grid x 10 time points (1000 points) are tight, principled
//! bounds that still leave room for optimizer noise but fail on any genuine
//! divergence in the design / baseline / covariate-effect pathway. We never
//! weaken them and never edit gam to pass.

use gam::families::survival_construction::SurvivalBaselineTarget;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::{Array1, Array2};

/// Deterministic, dependency-free PRNG (SplitMix64) so the synthetic data is
/// reproducible bit-for-bit and fed IDENTICALLY to gam and to scipy.
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
    /// Uniform in (0, 1).
    fn next_uniform(&mut self) -> f64 {
        // 53-bit mantissa, shifted off 0 so log() / inverse-CDF stay finite.
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller (one of the pair).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn gam_transformation_survival_prediction_grid_matches_scipy() {
    init_parallelism();

    // ---- synthesize the fixed-seed n=250 Weibull-AFT dataset --------------
    // (x1, x2) ~ BivarNormal(0, [[1, 0.3], [0.3, 1]]) via Cholesky of the
    // correlation matrix: x1 = z1, x2 = 0.3*z1 + sqrt(1-0.3^2)*z2.
    // t ~ Weibull(shape = exp(0.5), scale = exp(-(0.3*x1 + 0.2*x2))) via the
    // inverse-CDF: t = scale * (-log U)^(1/shape).
    // event ~ Bernoulli(0.65). Censored rows (event=0) carry their drawn t as a
    // right-censoring time — the standard generative convention.
    let n = 250usize;
    let shape_true = 0.5_f64.exp();
    let rho = 0.3_f64;
    let mut rng = SplitMix64::new(0x5EED_5EED_1234_ABCD);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for _ in 0..n {
        let z1 = rng.next_normal();
        let z2 = rng.next_normal();
        let a = z1;
        let b = rho * z1 + (1.0 - rho * rho).sqrt() * z2;
        let scale_i = (-(0.3 * a + 0.2 * b)).exp();
        let u = rng.next_uniform();
        let ti = scale_i * (-u.ln()).powf(1.0 / shape_true);
        let event = if rng.next_uniform() < 0.65 { 1.0 } else { 0.0 };
        x1.push(a);
        x2.push(b);
        t.push(ti);
        d.push(event);
    }
    assert_eq!(t.len(), n);
    assert!(
        t.iter().all(|&v| v.is_finite() && v > 0.0),
        "all survival times must be positive and finite"
    );
    let n_events: f64 = d.iter().sum();
    assert!(
        n_events > 100.0 && n_events < (n as f64),
        "expected a healthy mix of events/censoring, got {n_events} events of {n}"
    );

    // ---- fit with gam: parametric Weibull transformation AFT --------------
    // `survival_likelihood = "weibull"` is gam's parametric Weibull baseline
    // (the transformation / Royston-Parmar net model with a 2-column linear
    // time basis [1, log t] seeded by scale/shape). The formula carries the
    // explicit `survmodel(spec="transformation", distribution="weibull")` term
    // to declare intent; the library path resolves the likelihood mode from
    // FitConfig. Two linear covariates x1, x2 are appended after the time
    // basis, so beta = [time0, time1(=shape), gamma_x1, gamma_x2].
    let headers = vec![
        "t".to_string(),
        "d".to_string(),
        "x1".to_string(),
        "x2".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                t[i].to_string(),
                d[i].to_string(),
                x1[i].to_string(),
                x2[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode survival data");

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(t, d) ~ x1 + x2 + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &data,
        &cfg,
    )
    .expect("gam Weibull transformation AFT fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit result for survival_likelihood=weibull");
    };
    assert_eq!(
        fit.baseline_cfg.target,
        SurvivalBaselineTarget::Weibull,
        "gam must report a fitted Weibull baseline"
    );
    let lambda = fit.baseline_cfg.scale.expect("fitted Weibull scale lambda");
    let p = fit.baseline_cfg.shape.expect("fitted Weibull shape p");
    assert!(
        lambda.is_finite() && lambda > 0.0 && p.is_finite() && p > 0.0,
        "fitted Weibull (scale={lambda}, shape={p}) must be positive and finite"
    );

    let beta = &fit.fit.beta;
    // beta = [time-basis cols (2 for the linear Weibull basis), covariate cols].
    // The covariate block is the tail; build the covariate design at the grid
    // to extract gamma . x robustly (no assumption about x1/x2 column order).
    let n_cov = beta.len() - 2;
    assert_eq!(
        n_cov, 2,
        "expected 2 covariate coefficients (x1, x2) after the 2-col Weibull time basis; beta.len()={}",
        beta.len()
    );
    let gamma: Array1<f64> = beta.slice(ndarray::s![2..]).to_owned();

    // ---- shared prediction grid: (x1, x2) in [-2, 2]^2 (10x10), t in [0.5, 20] (10) ----
    let grid_n = 10usize;
    let x1_pts: Vec<f64> = (0..grid_n)
        .map(|k| -2.0 + 4.0 * (k as f64) / ((grid_n - 1) as f64))
        .collect();
    let x2_pts: Vec<f64> = x1_pts.clone();
    let t_lo = 0.5_f64;
    let t_hi = 20.0_f64;
    let t_pts: Vec<f64> = (0..grid_n)
        .map(|k| t_lo + (t_hi - t_lo) * (k as f64) / ((grid_n - 1) as f64))
        .collect();

    // Covariate grid points (10x10 = 100 (x1, x2) pairs).
    let mut cov_pairs: Vec<(f64, f64)> = Vec::with_capacity(grid_n * grid_n);
    for &a in &x1_pts {
        for &bb in &x2_pts {
            cov_pairs.push((a, bb));
        }
    }

    // Build the covariate design at the grid covariate points, using the frozen
    // term-collection spec from the fit (this is gam's real prediction path).
    let col = data.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let mut cov_grid = Array2::<f64>::zeros((cov_pairs.len(), data.headers.len()));
    for (i, &(a, bb)) in cov_pairs.iter().enumerate() {
        cov_grid[[i, x1_idx]] = a;
        cov_grid[[i, x2_idx]] = bb;
    }
    let cov_design = build_term_collection_design(cov_grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at grid points");
    assert_eq!(
        cov_design.design.ncols(),
        gamma.len(),
        "covariate design width must match the covariate-coefficient block"
    );
    // eta_x = gamma . x for each covariate grid point.
    let eta_x: Vec<f64> = cov_design.design.apply(&gamma).to_vec();
    assert_eq!(eta_x.len(), cov_pairs.len());

    // gam predicted survival surface, row-major over (cov pair, time):
    // S(t|x) = exp( -(t/lambda)^p * exp(gamma . x) ).
    let mut gam_surv: Vec<f64> = Vec::with_capacity(cov_pairs.len() * t_pts.len());
    for &e in &eta_x {
        let cov_mult = e.exp();
        for &tt in &t_pts {
            let cum_haz = (tt / lambda).powf(p) * cov_mult;
            gam_surv.push((-cum_haz).exp());
        }
    }

    // ---- fit the SAME data with scipy (exact parametric ground truth) -----
    // Hand scipy the identical (t, d, x1, x2) columns plus the prediction grid;
    // it maximizes the Weibull-AFT censored log-likelihood and evaluates the
    // analytic survival function on the matching grid (same row-major order).
    let grid_x1: Vec<f64> = cov_pairs.iter().map(|&(a, _)| a).collect();
    let grid_x2: Vec<f64> = cov_pairs.iter().map(|&(_, b)| b).collect();
    let t_pts_csv = t_pts
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let grid_x1_csv = grid_x1
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let grid_x2_csv = grid_x2
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");

    let r = run_python(
        &[
            Column::new("t", &t),
            Column::new("d", &d),
            Column::new("x1", &x1),
            Column::new("x2", &x2),
        ],
        &format!(
            r#"
import numpy as np
from scipy.optimize import minimize

tt = np.asarray(df["t"], dtype=float)
dd = np.asarray(df["d"], dtype=float)
xx1 = np.asarray(df["x1"], dtype=float)
xx2 = np.asarray(df["x2"], dtype=float)
logt = np.log(tt)

# Weibull AFT: log T = mu + b1*x1 + b2*x2 + sigma*W, W ~ standard Gumbel(min).
# shape = 1/sigma, scale_i = exp(mu + b1*x1 + b2*x2).
# Censored (Weibull) log-likelihood in terms of z = (logt - linpred)/sigma:
#   event:    -log(sigma) + z - exp(z)
#   censored:            - exp(z)
def negloglik(par):
    mu, b1, b2, logsig = par
    sig = np.exp(logsig)
    linpred = mu + b1 * xx1 + b2 * xx2
    z = (logt - linpred) / sig
    ll = dd * (-logsig + z - np.exp(z)) + (1.0 - dd) * (-np.exp(z))
    return -np.sum(ll)

# Sensible starts: mu ~ mean log-time, unit scale.
x0 = np.array([float(np.mean(logt)), 0.0, 0.0, 0.0])
res = minimize(negloglik, x0, method="Nelder-Mead",
               options=dict(xatol=1e-10, fatol=1e-12, maxiter=200000, maxfev=200000))
res = minimize(negloglik, res.x, method="BFGS",
               options=dict(gtol=1e-10, maxiter=100000))
mu, b1, b2, logsig = res.x
sig = np.exp(logsig)
shape = 1.0 / sig

gx1 = np.array([{grid_x1_csv}])
gx2 = np.array([{grid_x2_csv}])
gt = np.array([{t_pts_csv}])

# Analytic Weibull survival on the grid, row-major over (cov pair, time).
surv = []
for a, bcoord in zip(gx1, gx2):
    scale_i = np.exp(mu + b1 * a + b2 * bcoord)
    for s in gt:
        surv.append(np.exp(-(s / scale_i) ** shape))
emit("surv", surv)
emit("shape", [shape])
emit("converged", [1.0 if res.success else 0.0])
"#
        ),
    );

    let scipy_surv = r.vector("surv");
    let scipy_shape = r.scalar("shape");
    assert_eq!(
        scipy_surv.len(),
        gam_surv.len(),
        "scipy survival grid length mismatch: gam={} scipy={}",
        gam_surv.len(),
        scipy_surv.len()
    );

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_surv, scipy_surv);
    let corr = pearson(&gam_surv, scipy_surv);

    eprintln!(
        "transformation Weibull S(t|x) grid: n={n} events={n_events} \
         gam(lambda={lambda:.4}, shape_p={p:.4}, gamma={:?}) scipy_shape={scipy_shape:.4} \
         grid={}x{} (cov 10x10, t 10) rel_l2={rel:.5} pearson={corr:.6}",
        gamma.to_vec(),
        cov_pairs.len(),
        t_pts.len()
    );

    // Same likelihood, same data => the predicted S(t|x) surface is
    // deterministic up to optimizer tolerance. These tight bounds (rel_l2 from
    // a parametric reference, Pearson over 1000 grid points) fail on any real
    // divergence in gam's beta -> design -> transformed-quantile -> survival
    // pathway while tolerating small-sample optimizer noise.
    assert!(
        corr >= 0.998,
        "predicted survival surfaces diverge from scipy: pearson={corr:.6}"
    );
    assert!(
        rel <= 0.015,
        "predicted survival surfaces diverge from scipy: rel_l2={rel:.5}"
    );
}
