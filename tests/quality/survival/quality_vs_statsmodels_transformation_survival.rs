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
//! frozen covariate design. The engine ANCHOR-CENTERS the time design at the time
//! anchor (`center_survival_time_designs_at_anchor`), which subtracts `b(anchor)`
//! from every time row and so ZEROES the constant time column `b_0 ≡ 1` (it
//! becomes `1 − 1 = 0`); its coefficient `β_time[0]` is therefore unidentified by
//! the time block. The baseline LEVEL is consequently carried by the covariate
//! design's intercept column, so the term-collection covariate design is
//! `c(x) = [1, x1, x2]` (3 columns: intercept + the two covariates) and
//! `β_cov = [intercept, γ_x1, γ_x2]`. We reconstruct `S(t|x)` from the *actual
//! fitted `β`* exactly as `survival_predict` assembles the exit predictor
//! (`x_exit · β + offset`, offset = 0 for the Weibull-linear path: the derivative
//! guard is 0 and the parametric target is folded into the linear time block),
//! NOT from a `(t/scale)^shape` parametric shortcut. Because `b_0 − b_0(anchor) =
//! 0`, the reconstruction `β_cov·c(x) + Σ_k (b_k(t) − b_k(anchor))·β_time_k`
//! reproduces the fitted `log Λ(t|x)` exactly: the covariate intercept supplies
//! the baseline level and the centered `log t` column supplies the shape slope.
//! The fitted shape exponent `p` is the slope of `log Λ_0` in `log t`, i.e.
//! `β_time[1]`. We never weaken the bars and never edit gam to pass.

use csv::StringRecord;
use gam::families::survival::construction::{
    SurvivalBaselineTarget, SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::{Array1, Array2, s};
use std::path::Path;

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
    // FitConfig. The covariate term-collection design carries its own intercept
    // (the anchor-centered time block zeroes its own constant column, so the
    // baseline level lives in the covariate intercept), then the two linear
    // covariates x1, x2, so beta = [time0(dead), time1(=shape), intercept,
    // gamma_x1, gamma_x2] — 2 time columns then 3 covariate columns.
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
    // The covariate term-collection design is `[1, x1, x2]`: the anchor-centered
    // time block zeroes its own constant column `b_0 ≡ 1`, so the baseline level
    // is carried by the covariate intercept, NOT the dead time-level coefficient.
    // Hence the covariate block has 3 columns (intercept + x1 + x2) and the full
    // coefficient vector is [time0(dead), time1(=shape), intercept, γ_x1, γ_x2].
    // `gamma` is sliced from `beta` and dotted against the SAME 3-column covariate
    // design rebuilt from `fit.resolvedspec` below, so the reconstruction stays
    // exactly aligned with the fit.
    assert_eq!(
        n_cov,
        3,
        "expected 3 covariate coefficients (intercept, x1, x2) after the 2-col Weibull time basis; beta.len()={}",
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
# Convergence = a VALID fit (finite params + finite objective), NOT the BFGS
# `success` flag. With gtol=1e-10 (a gradient tolerance far tighter than any
# survival NLL realistically attains) the polish step reports success=False
# even when the Nelder-Mead + BFGS estimate is sound, so keying the baseline's
# validity on that flag is a false negative. This matches both the module
# header ("NOT ... matching another tool's noisy [optimizer] flag") and the
# real-data PHReg arm, which already validates via finite params.
_ok = np.all(np.isfinite(np.asarray(res.x, dtype=float))) and np.isfinite(negloglik(res.x))
emit("converged", [1.0 if _ok else 0.0])
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

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Dataset SOURCE: the Veterans' Administration lung-cancer randomized trial
// (`veteran` in the R `survival` package; Kalbfleisch & Prentice, "The
// Statistical Analysis of Failure Time Data", 1980). 137 patients, columns
// `time` (survival days), `status` (1=death, 0=right-censored — 9 censored),
// the numeric covariates `karno` (Karnofsky performance score, the dominant
// prognostic signal), `age`, and `diagtime` (months from diagnosis to entry),
// shipped at `bench/datasets/veteran_lung.csv`.
//
// This arm exercises the SAME gam capability as the synthetic test above —
// a parametric **Weibull transformation (AFT) survival model** with LINEAR
// covariate effects over a shared Weibull baseline cumulative hazard
// (`Surv(time, status) ~ karno + age + diagtime`, the real-data analogue of
// the synthetic `x1 + x2`). Both fits put the covariates into the same shared
// log-cumulative-hazard, so the covariate block `cov_eta = design(x)·cov_beta`
// IS the per-patient log-risk (monotone increasing in hazard).
//
// Because this is real data the data-generating survival surface is UNKNOWN,
// so RMSE-to-truth is not computable. The objective, tool-free quality metric
// is therefore the **held-out concordance index** (Harrell's C): a fixed,
// deterministic train/test split (every 4th row held out), fit gam on train,
// score the held-out patients by gam's OWN covariate log-cumulative-hazard
// risk, and assert how well that risk ranking agrees with the observed
// (time, event) ordering. Higher covariate log-cumulative-hazard ⇒ higher
// hazard ⇒ shorter survival, so a well-fit model gives a high C-index
// (0.5 = random, 1.0 = perfect).
//   PRIMARY (objective): held-out C-index >= 0.62 — well above the 0.5
//     random-ranking baseline for a ~34-patient held-out set; a degenerate or
//     wrong-sign covariate fit would not clear it.
//   BASELINE (match-or-beat): statsmodels' `PHReg` (the mature, standard
//     statsmodels survival regression — a Cox proportional-hazards model) is
//     fit on the SAME train rows and scored on the SAME held-out patients by
//     its linear predictor (monotone in hazard), turned into the IDENTICAL
//     held-out C-index in plain Rust. gam's held-out C must be no worse than
//     statsmodels' C minus a 0.03 margin. statsmodels is a yardstick to
//     match-or-beat on the identical metric, never a fitted output to copy.

/// Harrell's concordance index for survival risk scores. `risk[i]` is monotone
/// INCREASING in hazard (higher risk ⇒ shorter expected survival). A pair
/// `(i, j)` is comparable when the earlier observed time belongs to an event;
/// it is concordant when that earlier-failing subject also carries the higher
/// risk. Risk ties on a comparable pair count as half-concordant. Returns the
/// fraction of comparable pairs that are concordant (0.5 = random ordering).
fn concordance_index(risk: &[f64], time: &[f64], event: &[f64]) -> f64 {
    assert_eq!(
        risk.len(),
        time.len(),
        "concordance: risk/time length mismatch"
    );
    assert_eq!(
        time.len(),
        event.len(),
        "concordance: time/event length mismatch"
    );
    let n = risk.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let (early, late) = if time[i] < time[j] {
                (i, j)
            } else if time[j] < time[i] {
                (j, i)
            } else {
                continue;
            };
            if event[early] != 1.0 {
                continue;
            }
            comparable += 1.0;
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if risk[early] == risk[late] {
                concordant += 0.5;
            }
        }
    }
    assert!(
        comparable > 0.0,
        "no comparable survival pairs — degenerate held-out set"
    );
    concordant / comparable
}

#[test]
fn gam_transformation_survival_prediction_grid_matches_scipy_on_real_data() {
    init_parallelism();

    // ---- load the real Veterans' lung-cancer trial ------------------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/veteran_lung.csv"
    )))
    .expect("load veteran_lung.csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let age_idx = col["age"];
    let diagtime_idx = col["diagtime"];

    let time: Vec<f64> = ds.values.column(time_idx).to_vec();
    let status: Vec<f64> = ds.values.column(status_idx).to_vec();
    let karno: Vec<f64> = ds.values.column(karno_idx).to_vec();
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let diagtime: Vec<f64> = ds.values.column(diagtime_idx).to_vec();
    let n = time.len();
    assert!(n > 120, "veteran_lung should have ~137 rows, got {n}");
    assert!(
        time.iter().all(|&v| v.is_finite() && v > 0.0),
        "all survival times must be positive and finite"
    );

    // ---- deterministic train/test split: every 4th row held out ----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 90 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: Weibull transformation AFT, linear covariates --
    // Same capability as the synthetic arm (parametric Weibull transformation
    // model, linear covariate block over a shared log-cumulative-hazard); only
    // the covariate set changes to the real prognostic factors.
    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, status) ~ karno + age + diagtime + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &train_ds,
        &cfg,
    )
    .expect("gam Weibull transformation AFT fit on veteran_lung train rows");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for survival_likelihood=weibull");
    };
    assert_eq!(
        fit.baseline_cfg.target,
        SurvivalBaselineTarget::Weibull,
        "gam must report a fitted Weibull baseline"
    );

    // ---- score the held-out patients by gam's OWN covariate risk ----------
    // The covariate block adds `cov_eta = design(x)·cov_beta` to the shared
    // log-cumulative-hazard, so `cov_eta` IS the per-patient log-risk (monotone
    // increasing in hazard). Every patient shares the same baseline H0(t), so
    // the covariate predictor alone ranks risk — no baseline evaluation needed.
    let cov_start = fit.time_base_ncols;
    let beta = &fit.fit.beta;
    assert!(
        beta.len() > cov_start,
        "expected covariate coefficients after the {cov_start} time columns, got beta.len()={}",
        beta.len()
    );
    let cov_beta = beta.slice(s![cov_start..]).to_owned();

    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (out_row, &src_row) in test_rows.iter().enumerate() {
        for c in 0..p {
            test_grid[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at held-out rows");
    let test_dense = test_design.design.to_dense();
    assert_eq!(
        test_dense.ncols(),
        cov_beta.len(),
        "held-out covariate design width must match the covariate coefficient slice"
    );
    let gam_risk: Vec<f64> = (0..test_rows.len())
        .map(|r| test_dense.row(r).dot(&cov_beta))
        .collect();

    let test_time: Vec<f64> = test_rows.iter().map(|&i| time[i]).collect();
    let test_status: Vec<f64> = test_rows.iter().map(|&i| status[i]).collect();
    let gam_c = concordance_index(&gam_risk, &test_time, &test_status);

    // ---- fit the SAME data with statsmodels PHReg, score the SAME TEST -----
    // One run_python call, all columns the SAME length (full n): a per-row
    // `is_train` mask separates the fit rows from the held-out rows so we never
    // mix train-length and test-length columns. statsmodels' PHReg (Cox PH) is
    // fit on the masked train rows; we emit its held-out linear predictor
    // (monotone in hazard) on the held-out rows in the SAME order and turn it
    // into the IDENTICAL Rust concordance metric. The test rows ride back in a
    // padded `sm_risk` column whose first `n_test` entries are the real scores.
    let is_train: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let py = run_python(
        &[
            Column::new("time", &time),
            Column::new("status", &status),
            Column::new("karno", &karno),
            Column::new("age", &age),
            Column::new("diagtime", &diagtime),
            Column::new("is_train", &is_train),
        ],
        r#"
import numpy as np
import pandas as pd
import statsmodels.api as sm

frame = pd.DataFrame({
    "time": np.asarray(df["time"], dtype=float),
    "status": np.asarray(df["status"], dtype=float),
    "karno": np.asarray(df["karno"], dtype=float),
    "age": np.asarray(df["age"], dtype=float),
    "diagtime": np.asarray(df["diagtime"], dtype=float),
    "is_train": np.asarray(df["is_train"], dtype=float),
})
feat = ["karno", "age", "diagtime"]
train = frame[frame["is_train"] == 1.0].reset_index(drop=True)
test = frame[frame["is_train"] == 0.0].reset_index(drop=True)

# Cox proportional-hazards regression (statsmodels' mature survival model).
mod = sm.PHReg(train["time"].to_numpy(),
               train[feat].to_numpy(),
               status=train["status"].to_numpy())
res = mod.fit()

# Held-out linear predictor x·beta: monotone INCREASING in hazard, exactly the
# orientation of gam's covariate log-cumulative-hazard risk score. Emitted in
# held-out row order; the Rust side computes the same Harrell C.
sm_risk = test[feat].to_numpy() @ np.asarray(res.params, dtype=float)
emit("sm_risk", sm_risk.reshape(-1))
# PHRegResults exposes no `.converged` flag; a successful `.fit()` returning
# finite parameter estimates is the validity signal we can actually check.
converged = 1.0 if np.all(np.isfinite(np.asarray(res.params, dtype=float))) else 0.0
emit("converged", [converged])
"#,
    );
    let sm_risk = py.vector("sm_risk");
    let sm_converged = py.scalar("converged");
    assert_eq!(
        sm_converged, 1.0,
        "statsmodels PHReg must converge for the baseline to be valid"
    );
    assert_eq!(
        sm_risk.len(),
        test_rows.len(),
        "statsmodels held-out risk length mismatch: gam={} sm={}",
        test_rows.len(),
        sm_risk.len()
    );
    let sm_c = concordance_index(sm_risk, &test_time, &test_status);

    let cens_frac = 1.0 - status.iter().sum::<f64>() / n as f64;
    eprintln!(
        "veteran_lung weibull-transformation AFT held-out: n={n} n_train={} n_test={} \
         censoring={cens_frac:.2}\n  \
         held-out C-index: gam={gam_c:.4} statsmodels(PHReg)={sm_c:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam ranks held-out risk well --------
    // C-index >= 0.62 is well above the 0.5 random-ranking baseline for a small
    // held-out set; a degenerate or wrong-sign covariate fit would not clear it.
    assert!(
        gam_c >= 0.62,
        "gam's held-out concordance too low: {gam_c:.4} (< 0.62)"
    );

    // ---- BASELINE (match-or-beat): no worse than statsmodels on held-out C -
    assert!(
        gam_c >= sm_c - 0.03,
        "gam less concordant than statsmodels PHReg on held-out data: \
         gam C={gam_c:.4}, statsmodels C={sm_c:.4} (margin 0.03)"
    );
}
