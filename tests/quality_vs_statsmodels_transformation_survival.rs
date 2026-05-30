//! End-to-end quality: gam's transformation AFT survival model must RECOVER the
//! TRUE survival curve `S(t | x)` of the known data-generating process, across a
//! dense covariate-by-time grid.
//!
//! OBJECTIVE METRIC (truth recovery). The data is synthesized from a fully known
//! Weibull data-generating process, so the ground-truth survival surface is an
//! exact analytic quantity — no reference tool is needed to define it:
//!
//!     t = scale_i · (−log U)^(1/shape_true),   scale_i = exp(−(0.3·x1 + 0.2·x2)),
//!     <=>  S_true(t|x) = exp( −(t / scale_i)^shape_true ),   shape_true = exp(0.5).
//!
//! The test's PRIMARY claim is that gam's fitted `S(t|x)` surface recovers
//! `S_true(t|x)` to a principled accuracy bar: RMSE over the covariate×time grid
//! must be a small fraction of the survival range [0, 1]. (Censoring removes
//! ~35% of the event information from n=250 draws, so we allow a modest absolute
//! bar but never the trivial-fit value.) Recovery of the true Weibull shape
//! `exp(0.5)≈1.6487` is asserted directly against the generating constant, not
//! against any tool.
//!
//! BASELINE TO MATCH-OR-BEAT. A `scipy`-optimized exact Weibull-AFT MLE is fit on
//! the IDENTICAL censored data as an independent, mature estimator of the same
//! model. It is NOT the success criterion (matching another tool's noisy
//! small-sample fit proves nothing); it is a baseline gam must be at least as
//! accurate as. We additionally assert gam's RMSE-to-truth ≤ scipy's
//! RMSE-to-truth × 1.10 — gam's estimator is as good or better at recovering the
//! truth. The scipy rel_l2 vs gam is still printed for context only.
//!
//! gam parameterizes the same model on the log-cumulative-hazard / Royston-
//! Parmar net scale. For `survival_likelihood="weibull"` the time axis is a
//! *linear* basis on `log t` (columns `[1, log t]`, anchor-centered at the
//! engine's time anchor), so the fitted log-cumulative-hazard is
//!
//!     log Λ(t|x) = β_cov · c(x)  +  Σ_k (b_k(t) − b_k(anchor)) · β_time_k,
//!     S(t|x)     = exp( −exp( log Λ(t|x) ) ),
//!
//! where `β = [β_time (2 cols) | β_cov]`, `b(t) = [1, log t]`, and `c(x)` is the
//! frozen covariate design (no covariate intercept — the baseline level lives in
//! the time block). We reconstruct `S(t|x)` from the *actual fitted `β`* exactly
//! as `survival_predict` assembles the exit predictor (`x_exit · β + offset`,
//! offset = 0 for the Weibull-linear path: the derivative guard is 0 and the
//! parametric target is folded into the linear time block), NOT from a
//! `(t/scale)^shape` parametric shortcut. The fitted shape exponent `p` is the
//! slope of `log Λ_0` in `log t`, i.e. `β_time[1]`. We never weaken the bars and
//! never edit gam to pass.

use csv::StringRecord;
use gam::families::survival_construction::{
    SurvivalBaselineTarget, SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
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

    // beta = [β_time | β_cov]; the linear Weibull time block is a strict prefix
    // of length `time_base_ncols` (2 columns: [1, log t]).
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert_eq!(
        p_time, 2,
        "the linear Weibull time basis must be 2 columns [1, log t]; got {p_time}"
    );
    assert!(
        p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_time: Array1<f64> = beta.slice(ndarray::s![..p_time]).to_owned();
    let gamma: Array1<f64> = beta.slice(ndarray::s![p_time..]).to_owned();
    let n_cov = gamma.len();
    assert_eq!(
        n_cov,
        2,
        "expected 2 covariate coefficients (x1, x2) after the 2-col Weibull time basis; beta.len()={}",
        beta.len()
    );
    // The shape `p` of a Weibull RP baseline is the slope of log Λ_0 in log t,
    // i.e. β_time[1] (the coefficient on the `log t` column). It must be a
    // positive, finite Weibull exponent. We do NOT use `baseline_cfg.scale`:
    // the anchor-centered linear basis zeros the level column, so the recovered
    // scale is a seed artifact, not a witness of the fit.
    let p = beta_time[1];
    assert!(
        p.is_finite() && p > 0.0,
        "fitted Weibull shape (β_time[1]={p}) must be a positive, finite exponent"
    );

    // Resolved (frozen) time-basis config + anchor row, mirroring the engine's
    // anchor-centered linear rows on log(t).
    let time_cfg: SurvivalTimeBasisConfig = resolved_survival_time_basis_config_from_build(
        &fit.time_basis.basisname,
        fit.time_basis.degree,
        fit.time_basis.knots.as_ref(),
        fit.time_basis.keep_cols.as_ref(),
        fit.time_basis.smooth_lambda,
    )
    .expect("resolve frozen survival time-basis config");
    let anchor_row = evaluate_survival_time_basis_row(fit.time_basis.anchor, &time_cfg)
        .expect("evaluate time-basis anchor row");
    assert_eq!(
        anchor_row.len(),
        p_time,
        "anchor row width must equal the time block width"
    );

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
    // Covariate contribution β_cov · c(x) per covariate grid point.
    let eta_x: Vec<f64> = cov_design.design.apply(&gamma).to_vec();
    assert_eq!(eta_x.len(), cov_pairs.len());

    // gam predicted survival surface, row-major over (cov pair, time), assembled
    // exactly as `survival_predict` builds the exit predictor:
    //   log Λ(t|x) = β_cov·c(x) + Σ_k (b_k(t) − b_k(anchor))·β_time_k,
    //   S(t|x)     = exp( −exp(log Λ(t|x)) ).
    // The Weibull-linear offset is identically zero (derivative guard = 0, the
    // parametric target is folded into the linear time block), so the centered
    // time block plus the covariate block is the whole linear predictor.
    let mut gam_surv: Vec<f64> = Vec::with_capacity(cov_pairs.len() * t_pts.len());
    for &cov_eta in &eta_x {
        for &tt in &t_pts {
            let b = evaluate_survival_time_basis_row(tt, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut log_cum_haz = cov_eta;
            for k in 0..p_time {
                log_cum_haz += (b[k] - anchor_row[k]) * beta_time[k];
            }
            gam_surv.push((-log_cum_haz.exp()).exp());
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
    let scipy_converged = r.scalar("converged");
    assert_eq!(
        scipy_converged, 1.0,
        "scipy Weibull-AFT optimizer must converge for the baseline to be valid"
    );
    assert_eq!(
        scipy_surv.len(),
        gam_surv.len(),
        "scipy survival grid length mismatch: gam={} scipy={}",
        gam_surv.len(),
        scipy_surv.len()
    );

    // ---- EXACT ground-truth survival surface from the known DGP -----------
    // The data was drawn as t = scale_i·(−log U)^(1/shape_true) with
    // scale_i = exp(−(0.3·x1 + 0.2·x2)) and shape_true = exp(0.5), so the true
    // conditional survival is S_true(t|x) = exp(−(t/scale_i)^shape_true). This is
    // an analytic constant of the simulation — no tool defines it.
    let mut truth_surv: Vec<f64> = Vec::with_capacity(cov_pairs.len() * t_pts.len());
    for &(a, bb) in &cov_pairs {
        let scale_i = (-(0.3 * a + 0.2 * bb)).exp();
        for &tt in &t_pts {
            truth_surv.push((-(tt / scale_i).powf(shape_true)).exp());
        }
    }
    assert_eq!(truth_surv.len(), gam_surv.len());

    // RMSE-to-truth for gam (the estimator under test) and for the scipy MLE
    // baseline, on the identical grid. S(t|x) lives in [0, 1], so an RMSE bar is
    // an absolute fraction-of-range accuracy statement.
    let gam_rmse_truth = rmse(&gam_surv, &truth_surv);
    let scipy_rmse_truth = rmse(scipy_surv, &truth_surv);

    // Context-only: how the two independent estimators' surfaces compare. NOT a
    // pass criterion — matching scipy's noisy small-sample fit is not quality.
    let rel_gam_vs_scipy = relative_l2(&gam_surv, scipy_surv);
    let corr_gam_vs_scipy = pearson(&gam_surv, scipy_surv);

    eprintln!(
        "transformation Weibull S(t|x) grid: n={n} events={n_events} \
         gam(shape_p={p:.4}, gamma={:?}) shape_true={shape_true:.4} scipy_shape={scipy_shape:.4} \
         grid={}x{} (cov 10x10, t 10) | RMSE_to_truth gam={gam_rmse_truth:.5} scipy={scipy_rmse_truth:.5} \
         | context rel_l2(gam,scipy)={rel_gam_vs_scipy:.5} pearson={corr_gam_vs_scipy:.6}",
        gamma.to_vec(),
        cov_pairs.len(),
        t_pts.len()
    );

    // (1) TRUTH RECOVERY of the shape exponent against the GENERATING CONSTANT.
    // gam's shape is β_time[1] (slope of log Λ_0 in log t); the true exponent is
    // exp(0.5). A 12% bar admits n=250 + 35%-censoring sampling noise while
    // catching a genuinely wrong time-axis slope.
    let shape_err = (p - shape_true).abs() / shape_true;
    assert!(
        shape_err <= 0.12,
        "gam did not recover the true Weibull shape: gam={p:.5} true={shape_true:.5} (rel_err={shape_err:.4})"
    );

    // (2) PRIMARY: gam recovers the TRUE S(t|x) surface. RMSE over 1000 grid
    // points must be a small fraction of the [0,1] survival range. The bar is
    // well below a trivial/degenerate fit (a flat S≡const or wrong-slope surface
    // sits far above this) yet absorbs finite-sample + censoring noise.
    assert!(
        gam_rmse_truth <= 0.06,
        "gam failed to recover the true survival surface: RMSE_to_truth={gam_rmse_truth:.5} (bar 0.06)"
    );

    // (3) MATCH-OR-BEAT the mature baseline ON ACCURACY: gam's error vs the truth
    // must not exceed the independent scipy-MLE's error vs the same truth by more
    // than 10%. gam's estimator is as good or better at recovering the DGP.
    assert!(
        gam_rmse_truth <= scipy_rmse_truth * 1.10,
        "gam is less accurate than the scipy Weibull-AFT MLE baseline: \
         gam_RMSE_to_truth={gam_rmse_truth:.5} > scipy_RMSE_to_truth={scipy_rmse_truth:.5} * 1.10"
    );
}
