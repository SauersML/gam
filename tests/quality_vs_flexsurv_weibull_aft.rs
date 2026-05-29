//! End-to-end quality: gam's parametric Weibull survival baseline (the
//! transformation-mode / Royston-Parmar net model with a *parametric Weibull*
//! baseline) must reproduce the mature R reference for parametric AFT survival:
//! `survival::survreg(dist = "weibull")` and `flexsurv::flexsurvreg(dist =
//! "weibull")`.
//!
//! Why these references. `survreg` and `flexsurvreg` are the gold-standard
//! parametric survival fitters in R. The Weibull distribution is the unique
//! distribution that is simultaneously proportional-hazards (PH) and
//! accelerated-failure-time (AFT), so a *single* fitted Weibull is comparable
//! across the three engines despite their different internal parameterizations.
//! gam parameterizes the model on the log-cumulative-hazard scale (Royston-
//! Parmar net):
//!
//!     log H(t|x) = beta0 + beta1*log t + gamma*x,
//!     beta1 = shape p,  scale lambda = exp(-beta0 / beta1),  gamma = log-HR,
//!     S(t|x) = exp(-(t/lambda)^p * exp(gamma*x)).
//!
//! survreg reports the *AFT* parameterization log T = mu + alpha*x + sigma*W
//! (W = standard extreme-value). The closed-form Weibull PH <-> AFT map is
//! exact:
//!
//!     intercept mu = log(lambda),         (location)
//!     log(sigma)   = -log(p),             (log-scale; survreg `Log(scale)`)
//!     alpha        = -gamma / p.          (AFT covariate coefficient)
//!
//! Because the AFT coefficients and the scale parameter are *parametric*
//! targets — invariant to any smoothing — agreement should be tight. We assert
//!   1. max |gam - survreg| over (mu, log sigma, alpha) <= 0.05, and
//!   2. Pearson correlation of S(t|x) on a time grid (both arms, from
//!      flexsurvreg) >= 0.995.
//! These bounds are loose enough to absorb the small-sample (n=24) optimizer
//! differences between three independent fitters yet tight enough that any real
//! divergence in gam's parametric Weibull baseline or covariate effect fails
//! the test. We never weaken them and never edit gam to pass.

use gam::families::survival_construction::SurvivalBaselineTarget;
use gam::test_support::reference::{Column, max_abs_diff, pearson, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use std::fs;
use std::path::Path;

const BONE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/bone.csv");

#[test]
fn gam_weibull_aft_matches_flexsurv_and_survreg_on_bone() {
    init_parallelism();

    // ---- load the real bone-marrow-transplant survival dataset ------------
    // bone.csv columns: t (time), d (event 0/1), trt (factor: "allo"/"auto").
    // We encode trt as a numeric 0/1 dummy (allo = 0 reference, auto = 1) and
    // feed the IDENTICAL numeric columns to gam and to R, so the single
    // covariate coefficient is directly comparable (no contrast-coding
    // ambiguity between engines).
    let raw = fs::read_to_string(Path::new(BONE_CSV)).expect("read bone.csv");
    let mut t: Vec<f64> = Vec::new();
    let mut d: Vec<f64> = Vec::new();
    let mut trt: Vec<f64> = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 {
            // header
            assert!(line.starts_with("\"t\""), "unexpected bone.csv header: {line}");
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 3, "bone.csv row {i} should have 3 fields: {line}");
        t.push(fields[0].trim().parse::<f64>().expect("parse t"));
        d.push(fields[1].trim().parse::<f64>().expect("parse d"));
        let arm = fields[2].trim().trim_matches('"');
        let code = match arm {
            "allo" => 0.0,
            "auto" => 1.0,
            other => panic!("unexpected trt level {other:?} in bone.csv"),
        };
        trt.push(code);
    }
    let n = t.len();
    assert_eq!(n, 23, "bone.csv should have 23 data rows, got {n}");
    assert!(
        trt.iter().any(|&v| v == 0.0) && trt.iter().any(|&v| v == 1.0),
        "both treatment arms must be present"
    );

    // ---- fit with gam: parametric Weibull survival ------------------------
    // `survival_likelihood = "weibull"` is gam's pure parametric Weibull
    // baseline: a 2-column linear time basis [1, log t] seeded by scale/shape,
    // with the single linear covariate appended. The fitted Weibull
    // (scale, shape) is recovered into `baseline_cfg`; the covariate
    // coefficient is the last beta entry.
    let headers = vec!["t".to_string(), "d".to_string(), "trt".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                t[i].to_string(),
                d[i].to_string(),
                trt[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode bone data");

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, d) ~ trt", &data, &cfg).expect("gam Weibull AFT fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit result for survival_likelihood=weibull");
    };

    assert_eq!(
        fit.baseline_cfg.target,
        SurvivalBaselineTarget::Weibull,
        "gam must report a fitted Weibull baseline"
    );
    let scale = fit.baseline_cfg.scale.expect("fitted Weibull scale lambda");
    let shape = fit.baseline_cfg.shape.expect("fitted Weibull shape p");
    assert!(
        scale.is_finite() && scale > 0.0 && shape.is_finite() && shape > 0.0,
        "fitted Weibull (scale={scale}, shape={shape}) must be positive and finite"
    );

    // beta layout: [time0, time1(=shape), covariate]. With a single numeric
    // covariate and no covariate intercept (the baseline absorbs it), the trt
    // PH log-hazard-ratio gamma is the last coefficient.
    let beta = &fit.fit.beta;
    assert_eq!(
        beta.len(),
        3,
        "Weibull beta = [time0, time1, trt] expected length 3, got {}",
        beta.len()
    );
    let gamma = beta[beta.len() - 1];

    // Map gam's PH-scale Weibull to the AFT parameterization survreg reports.
    let gam_aft_intercept = scale.ln(); // mu = log(lambda)
    let gam_aft_log_scale = -(shape.ln()); // log(sigma) = -log(p)
    let gam_aft_trt = -gamma / shape; // alpha = -gamma / p
    let gam_aft = [gam_aft_intercept, gam_aft_log_scale, gam_aft_trt];

    // gam survival curve on a shared grid, analytic from the fitted Weibull +
    // PH covariate effect: S(t|x) = exp(-(t/lambda)^p * exp(gamma*x)).
    let t_max = t.iter().cloned().fold(f64::MIN, f64::max);
    let grid: Vec<f64> = (1..=40).map(|k| t_max * (k as f64) / 40.0).collect();
    let mut gam_surv: Vec<f64> = Vec::with_capacity(grid.len() * 2);
    for &x in &[0.0_f64, 1.0_f64] {
        for &tt in &grid {
            let h = (tt / scale).powf(shape) * (gamma * x).exp();
            gam_surv.push((-h).exp());
        }
    }

    // ---- fit the SAME data with survreg (AFT) and flexsurvreg ------------
    let grid_csv = grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let r = run_r(
        &[
            Column::new("t", &t),
            Column::new("d", &d),
            Column::new("trt", &trt),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(survival))
            suppressPackageStartupMessages(library(flexsurv))
            grid <- c({grid_csv})

            # survreg AFT parameterization: log T = mu + alpha*trt + sigma*W.
            sr <- survreg(Surv(t, d) ~ trt, data = df, dist = "weibull")
            # coefficients(sr) = c((Intercept)=mu, trt=alpha); sr$scale = sigma.
            emit("sr_intercept", as.numeric(coef(sr)[["(Intercept)"]]))
            emit("sr_log_scale", log(as.numeric(sr$scale)))
            emit("sr_trt", as.numeric(coef(sr)[["trt"]]))

            # flexsurvreg Weibull: survival curves on the shared grid for both arms.
            fs <- flexsurvreg(Surv(t, d) ~ trt, data = df, dist = "weibull")
            s0 <- summary(fs, newdata = data.frame(trt = 0), t = grid, ci = FALSE)[[1]]
            s1 <- summary(fs, newdata = data.frame(trt = 1), t = grid, ci = FALSE)[[1]]
            emit("surv", c(s0$est, s1$est))
            "#
        ),
    );

    let r_aft = [
        r.scalar("sr_intercept"),
        r.scalar("sr_log_scale"),
        r.scalar("sr_trt"),
    ];
    let r_surv = r.vector("surv");
    assert_eq!(
        r_surv.len(),
        gam_surv.len(),
        "flexsurv survival grid length mismatch: gam={} r={}",
        gam_surv.len(),
        r_surv.len()
    );

    // ---- compare ----------------------------------------------------------
    let coef_diff = max_abs_diff(&gam_aft, &r_aft);
    let surv_corr = pearson(&gam_surv, r_surv);

    eprintln!(
        "bone Weibull AFT: n={n} gam(scale={scale:.4}, shape={shape:.4}, gamma={gamma:.4}) \
         gam_aft=[mu={:.4}, log_sigma={:.4}, trt={:.4}] \
         survreg_aft=[mu={:.4}, log_sigma={:.4}, trt={:.4}] \
         max_abs_coef_diff={coef_diff:.4} surv_pearson={surv_corr:.5}",
        gam_aft[0], gam_aft[1], gam_aft[2], r_aft[0], r_aft[1], r_aft[2]
    );

    // AFT coefficients (location + log-scale + covariate) are parametric and
    // smoothing-invariant; the three engines should agree to within optimizer
    // noise on n=24. 0.05 is a tight, principled bound.
    assert!(
        coef_diff <= 0.05,
        "AFT coefficients diverge from survreg: max_abs_diff={coef_diff:.4} \
         (gam={gam_aft:?}, survreg={r_aft:?})"
    );
    // The fitted S(t|x) over both arms must essentially coincide with
    // flexsurvreg's; >= 0.995 Pearson catches any real divergence in shape,
    // scale, or covariate effect while tolerating tiny grid-level differences.
    assert!(
        surv_corr >= 0.995,
        "fitted survival curves diverge from flexsurvreg: pearson={surv_corr:.5}"
    );
}
