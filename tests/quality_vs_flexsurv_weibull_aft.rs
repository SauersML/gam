//! End-to-end quality: gam's parametric Weibull survival baseline
//! (`survival_likelihood = "weibull"`, the Royston-Parmar net model with a
//! *linear* `[1, log t]` time basis seeded by scale/shape) must reproduce the
//! mature R reference for parametric Weibull survival:
//! `flexsurv::flexsurvreg(dist = "weibull")`.
//!
//! Why flexsurv. `flexsurvreg` is the gold-standard parametric survival fitter
//! in R and reports a *proportional-hazards* Weibull directly: it parameterizes
//! the model by `shape` and `scale` and exposes the covariate effect as a
//! log-hazard-ratio on the PH scale via `summary(..., type = "survival")` /
//! `type = "cumhaz"`. The Weibull distribution is simultaneously PH and AFT, so
//! a Weibull fitted by gam (which works on the log-cumulative-hazard / net-
//! survival scale) is directly comparable.
//!
//! What gam actually fits — and why we compare the FITTED FUNCTION, not the
//! `(scale, shape)` struct. In `survival_likelihood = "weibull"` mode with no
//! `timewiggle`, gam's `baseline_cfg.target` stays `Linear` (the Weibull enters
//! only as the *seed* of a 2-column `[1, log t]` time basis); `baseline_cfg`
//! never carries the fitted `scale`/`shape` (they are `None`). The fitted model
//! lives entirely in the coefficient vector
//!
//!     beta = [ beta_time(2) | gamma_cov ],
//!     log Λ(t | x) = Σ_k (b_k(t) − b_k(anchor)) · beta_time_k + gamma·x,
//!     S(t | x)     = exp(−Λ(t | x)),
//!
//! exactly as the sibling `quality_vs_flexsurv_rp_baseline` test reconstructs a
//! Royston-Parmar fit. The constant basis column is anchor-centred to zero, so
//! the *free* estimands gam reports are the log-time slope (Weibull shape p =
//! beta_time[1]) and the covariate PH log-hazard-ratio gamma; the baseline
//! location is fixed by the time anchor. We therefore compare the two quantities
//! gam genuinely produces — the fitted survival curve S(t|x) on a shared time
//! grid (both arms), and the covariate PH log-hazard-ratio — against flexsurv,
//! reconstructing gam's curve from the frozen time basis + beta (no reliance on
//! the unset `baseline_cfg.scale/shape`).
//!
//! Bounds:
//!   1. relative-L2 of S(t|x) over both arms on the grid ≤ 0.03, and
//!   2. |gam gamma − flexsurv PH log-HR| ≤ 0.05.
//! The covariate PH log-HR is a parametric, smoothing-invariant target; gam and
//! flexsurv maximize (penalized) likelihood on identical n=23 data, so it must
//! agree to optimizer noise (0.05). S(t|x) is the full fitted survival function;
//! 0.03 relative-L2 catches any real divergence in shape or covariate effect
//! while tolerating the small-sample/anchor differences between the two fitters.
//! We never weaken them and never edit gam to pass.

use csv::StringRecord;
use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs;
use std::path::Path;

const BONE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/bone.csv");

#[test]
fn gam_weibull_matches_flexsurv_on_bone() {
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
            assert!(
                line.starts_with("\"t\""),
                "unexpected bone.csv header: {line}"
            );
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            3,
            "bone.csv row {i} should have 3 fields: {line}"
        );
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
    // `survival_likelihood = "weibull"` is gam's parametric Weibull baseline: a
    // 2-column `[1, log t]` time basis seeded by scale/shape, with the single
    // linear covariate appended. The fit lives in the coefficient vector
    // (beta = [time(2) | gamma]); baseline_cfg stays Linear and does NOT carry
    // the fitted (scale, shape).
    let headers = vec!["t".to_string(), "d".to_string(), "trt".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", t[i]),
                format!("{:.1}", d[i]),
                format!("{:.1}", trt[i]),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode bone data");

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, d) ~ trt", &data, &cfg).expect("gam Weibull fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit result for survival_likelihood=weibull");
    };

    // beta = [β_time(2) | gamma]; the linear time block is a strict prefix.
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert_eq!(
        p_time, 2,
        "Weibull linear time basis must have 2 columns, got {p_time}"
    );
    assert_eq!(
        beta.len(),
        3,
        "Weibull beta = [time0, time1, trt] expected length 3, got {}",
        beta.len()
    );
    let beta_time = beta.slice(ndarray::s![..p_time]).to_owned();
    let gamma = beta[beta.len() - 1]; // covariate PH log-hazard-ratio

    // Resolved (knot-frozen) time-basis config + anchor row, mirroring the
    // engine's anchor-centred `[1, log t]` rows that produced `beta_time`.
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
        "anchor row width must equal the Weibull time block width"
    );

    // Covariate linear predictor gamma·trt, rebuilt from the frozen spec so the
    // column order/coding match `beta` exactly. trt enters linearly, so the PH
    // log-hazard-ratio is the finite difference of the covariate predictor.
    let trt_idx = data.column_map()["trt"];
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();
    let cov_eta = |trt_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, data.headers.len()));
        grid[[0, trt_idx]] = trt_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };
    let gam_gamma = cov_eta(1.0) - cov_eta(0.0); // PH log-HR (allo -> auto)
    assert!(
        (gam_gamma - gamma).abs() < 1e-9,
        "single linear covariate: finite-difference PH log-HR must equal the last beta"
    );

    // gam fitted survival curve on a shared time grid, reconstructed from the
    // frozen time basis + beta exactly as the engine evaluates log Λ:
    //   log Λ(t|x) = Σ_k (b_k(t) − anchor_k)·beta_time_k + gamma·x,
    //   S(t|x) = exp(−Λ).
    let t_max = t.iter().cloned().fold(f64::MIN, f64::max);
    let grid: Vec<f64> = (1..=40).map(|k| t_max * (k as f64) / 40.0).collect();
    let mut gam_surv: Vec<f64> = Vec::with_capacity(grid.len() * 2);
    for &x in &[0.0_f64, 1.0_f64] {
        let cov_contrib = cov_eta(x);
        for &tt in &grid {
            let b = evaluate_survival_time_basis_row(tt, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut log_cumhaz = cov_contrib;
            for k in 0..p_time {
                log_cumhaz += (b[k] - anchor_row[k]) * beta_time[k];
            }
            gam_surv.push((-log_cumhaz.exp()).exp());
        }
    }

    // ---- fit the SAME data with flexsurvreg(dist = "weibull") -------------
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
            suppressPackageStartupMessages(library(flexsurv))
            grid <- c({grid_csv})

            # flexsurvreg Weibull: PH-comparable parametric fit on identical data.
            fs <- flexsurvreg(Surv(t, d) ~ trt, data = df, dist = "weibull")

            # Proportional-hazards log-hazard-ratio for the covariate. flexsurv's
            # Weibull is AFT-parameterized (coef on the log-scale / AFT time
            # ratio); convert the AFT covariate coefficient to the PH log-HR via
            # the exact Weibull map  log-HR = -shape * alpha_AFT.
            shape <- as.numeric(fs$res["shape", "est"])
            alpha_aft <- as.numeric(coef(fs)[["trt"]])
            emit("ph_loghr", -shape * alpha_aft)

            # Survival curves on the shared grid for both arms.
            s0 <- summary(fs, newdata = data.frame(trt = 0), t = grid, ci = FALSE)[[1]]
            s1 <- summary(fs, newdata = data.frame(trt = 1), t = grid, ci = FALSE)[[1]]
            emit("surv", c(s0$est, s1$est))
            "#
        ),
    );

    let r_ph_loghr = r.scalar("ph_loghr");
    let r_surv = r.vector("surv");
    assert_eq!(
        r_surv.len(),
        gam_surv.len(),
        "flexsurv survival grid length mismatch: gam={} r={}",
        gam_surv.len(),
        r_surv.len()
    );

    // ---- compare on the quantities gam genuinely fits ---------------------
    let surv_rel = relative_l2(&gam_surv, r_surv);
    let loghr_diff = max_abs_diff(&[gam_gamma], &[r_ph_loghr]);

    eprintln!(
        "bone Weibull vs flexsurv: n={n} gam(shape=beta_time1={:.4}, gamma={gam_gamma:.4}) \
         flexsurv_ph_loghr={r_ph_loghr:.4} loghr_diff={loghr_diff:.4} surv_rel_l2={surv_rel:.5}",
        beta_time[1]
    );

    // The covariate PH log-hazard-ratio is a parametric, smoothing-invariant
    // target; both engines maximize (penalized) likelihood on identical n=23
    // data, so it must agree to optimizer noise. 0.05 is a tight, principled
    // bound.
    assert!(
        loghr_diff <= 0.05,
        "PH log-hazard-ratio diverges from flexsurv: |gam={gam_gamma:.4} - flexsurv={r_ph_loghr:.4}| = {loghr_diff:.4}"
    );
    // The fitted S(t|x) over both arms must essentially coincide with
    // flexsurvreg's; 0.03 relative-L2 catches any real divergence in shape,
    // baseline, or covariate effect while tolerating tiny grid-level / small-n
    // differences between the two parametric fitters.
    assert!(
        surv_rel <= 0.03,
        "fitted survival curves diverge from flexsurvreg: rel_l2={surv_rel:.5}"
    );
}
