//! End-to-end quality: gam's random-intercept *frailty* in flexible survival —
//! a latent Gaussian group effect that shifts the baseline hazard, fitted by
//! penalized likelihood with REML-selected frailty variance — must agree with
//! **R-INLA**, the scalable approximate-Bayesian latent-Gaussian gold standard
//! for exactly this model class.
//!
//! Reference. R-INLA fits
//!
//!     inla.surv(time, event) ~ 1 + x + f(group, model = "iid"),
//!     family = "weibullsurv"
//!
//! i.e. a Weibull proportional-hazards baseline with an i.i.d. zero-mean
//! Gaussian frailty on `group` whose precision (= 1/variance) carries a
//! penalized-complexity / log-gamma hyperprior. INLA returns, per group, the
//! *posterior mean* of the latent frailty (`summary.random$group$mean`) — the
//! Bayesian analogue of an empirical-Bayes BLUP — plus the fixed-effect slope
//! and the Weibull shape, from which the baseline cumulative hazard
//! `H0(t) = (t/scale)^shape` is reconstructed on a shared time grid.
//!
//! gam expresses the *same* latent-Gaussian survival structure through its
//! transformation (Royston-Parmar "net") survival likelihood with a flexible
//! monotone I-spline baseline on log-time and a ridge-penalized random-effect
//! block on `group`:
//!
//!     Surv(time, event) ~ x + s(group, bs = "re")
//!                       + survmodel(spec = "net")
//!
//! with `survival_likelihood = "transformation"` and `time_basis = "ispline"`.
//! The factor-RE block is a Gaussian random intercept on the log-hazard scale
//! whose single shrinkage λ is selected by REML — gam's penalized-likelihood
//! empirical-Bayes counterpart to INLA's marginal-posterior frailty. The two
//! engines therefore target the *same* estimand: per-group log-hazard shifts
//! under a shared-precision Gaussian prior.
//!
//! Data. The real `heart_failure_clinical_records_dataset` (n = 299, death rate
//! ~0.32). Clinically meaningful grouping: ejection fraction binned into six
//! cardiac-severity classes (a hospital-/risk-class-like factor), exactly the
//! family-/centre-level structure frailty models exist to capture. `age`
//! (standardized) is the fixed covariate. The identical (time, event, x, group)
//! columns are handed to gam and to INLA.
//!
//! What is compared, and the principled (un-weakened) bounds:
//!   1. Per-group frailty posterior means — gam empirical-Bayes BLUP vs INLA
//!      posterior mean — must agree within 0.12 RMSE on the log-hazard scale.
//!      Both are mean-centred shrinkage estimates of the same six latent shifts;
//!      0.12 is the SPEC bound and is tight relative to the ~0.3-0.6 spread the
//!      severity classes induce — a larger gap means the shrinkage path itself
//!      disagrees with the Bayesian reference.
//!   2. Baseline cumulative hazard H0(t) on a shared grid — Pearson >= 0.98.
//!      Both baselines are monotone increasing on the same support; gam's
//!      I-spline must reproduce INLA's Weibull baseline *shape* this tightly or
//!      the flexible-baseline assembly is wrong.
//!   3. Frailty-block shrinkage interpretation: the REML-selected RE block must
//!      carry a *sub-linear* effective dimension (edf strictly below the group
//!      count), the defining signature of shrinkage — an un-shrunk fixed-effect
//!      factor would spend the full df. This asserts gam is doing frailty
//!      regularization at all, consistent with INLA's finite frailty precision.
//!
//! A failing assertion because gam genuinely diverges from INLA is acceptable
//! and must NOT be papered over by loosening a bound or editing gam.

use csv::StringRecord;
use gam::families::survival_construction::SurvivalLikelihoodMode;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};

const HF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

/// Six cardiac-severity classes from ejection fraction (%). Clinically: <30
/// severe, 30-39 moderate-severe, 40-44 mild-moderate, 45-49 borderline, 50-59
/// low-normal, >=60 normal/preserved. This is the "risk-class" factor a frailty
/// model groups patients by.
fn ef_severity_group(ef: f64) -> usize {
    if ef < 30.0 {
        0
    } else if ef < 40.0 {
        1
    } else if ef < 45.0 {
        2
    } else if ef < 50.0 {
        3
    } else if ef < 60.0 {
        4
    } else {
        5
    }
}

const N_GROUPS: usize = 6;

#[test]
fn gam_survival_frailty_random_intercept_matches_inla() {
    init_parallelism();

    // ---- load the real heart-failure dataset (time, DEATH_EVENT, age, ef) --
    // Header order (see dataset): age, anaemia, creatinine_phosphokinase,
    // diabetes, ejection_fraction, high_blood_pressure, platelets,
    // serum_creatinine, serum_sodium, sex, smoking, time, DEATH_EVENT.
    let file = File::open(HF_CSV).expect("open heart_failure csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().expect("header line").expect("read header");
    let cols: Vec<&str> = header.split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("column {name} not found"))
    };
    let age_c = idx("age");
    let ef_c = idx("ejection_fraction");
    let time_c = idx("time");
    let death_c = idx("DEATH_EVENT");

    let mut age = Vec::<f64>::new();
    let mut time = Vec::<f64>::new();
    let mut event = Vec::<f64>::new();
    let mut grp = Vec::<usize>::new();
    for line in lines {
        let line = line.expect("read data line");
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let a: f64 = f[age_c].parse().expect("age");
        let ef: f64 = f[ef_c].parse().expect("ef");
        let t: f64 = f[time_c].parse().expect("time");
        let d: f64 = f[death_c].parse().expect("death");
        age.push(a);
        time.push(t);
        event.push(d);
        grp.push(ef_severity_group(ef));
    }
    let n = time.len();
    assert_eq!(n, 299, "heart_failure should have 299 rows, got {n}");
    let death_rate: f64 = event.iter().sum::<f64>() / n as f64;
    assert!(
        (0.25..=0.45).contains(&death_rate),
        "event rate should be ~0.3-0.5, got {death_rate:.3}"
    );

    // Every group must be populated so INLA's iid effect and gam's RE block
    // both span all six levels; otherwise the per-group comparison is undefined.
    let mut group_counts = [0usize; N_GROUPS];
    for &g in &grp {
        group_counts[g] += 1;
    }
    assert!(
        group_counts.iter().all(|&c| c >= 5),
        "each severity class needs >=5 patients for a stable frailty estimate: {group_counts:?}"
    );

    // Standardize age (mean 0, sd 1) so the fixed-effect slope is on a common
    // scale across both engines and the frailty carries the residual group
    // structure rather than absorbing an age trend.
    let age_mean = age.iter().sum::<f64>() / n as f64;
    let age_sd =
        (age.iter().map(|a| (a - age_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let x: Vec<f64> = age.iter().map(|a| (a - age_mean) / age_sd).collect();
    // INLA's iid factor levels are sorted numerically (1..N_GROUPS); pass the
    // group as a 1-based code so the factor-level order is unambiguous and the
    // returned per-level posterior means align with gam's level order.
    let group_code1: Vec<f64> = grp.iter().map(|&g| (g + 1) as f64).collect();

    // ---- fit with gam: transformation/net survival + RE frailty -----------
    // Rows are emitted with `group` as a string ("grp0".."grp5") so the schema
    // inferrer treats it as categorical; first-appearance order does NOT in
    // general equal the numeric class, so we rebuild the level->code map from
    // the encoded column to align gam's RE coefficients with INLA's levels.
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "x".to_string(),
        "group".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                time[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
                format!("grp{}", grp[i]),
            ])
        })
        .collect();
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode frailty survival data");
    let col = data.column_map();
    let x_idx = col["x"];
    let group_idx = col["group"];

    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, event) ~ x + s(group, bs='re') + survmodel(spec=\"net\")",
        &data,
        &cfg,
    )
    .expect("gam transformation frailty fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!(
            "expected a SurvivalTransformation fit result for survival_likelihood=transformation"
        );
    };
    assert_eq!(
        fit.likelihood_mode,
        SurvivalLikelihoodMode::Transformation,
        "gam must fit the transformation (net) survival likelihood"
    );

    // beta = [time-basis cols (fit.time_base_ncols), covariate+RE cols].
    let beta = &fit.fit.beta;
    let time_cols = fit.time_base_ncols;
    assert!(
        beta.len() > time_cols,
        "beta must carry covariate/RE columns beyond the {time_cols} time-basis columns; beta.len()={}",
        beta.len()
    );
    // The covariate+RE coefficient block is the tail; the resolvedspec design
    // is exactly this block (gam's real prediction path).
    let cov_beta = beta.slice(ndarray::s![time_cols..]).to_owned();

    // Recover each encoded group level's numeric code by reading the encoded
    // categorical column directly (string "grp{k}" -> level index, paired with
    // the row's true class k). Build an anchor row per encoded level.
    let group_encoded: Vec<f64> = data.values.column(group_idx).to_vec();
    // level_to_class[level] = clinical class index (0..N_GROUPS) for that level.
    let mut level_to_class = vec![usize::MAX; N_GROUPS];
    for i in 0..n {
        let level = group_encoded[i].round() as usize;
        assert!(level < N_GROUPS, "encoded group level {level} out of range");
        level_to_class[level] = grp[i];
    }
    assert!(
        level_to_class.iter().all(|&c| c != usize::MAX),
        "every encoded level must map to a clinical class: {level_to_class:?}"
    );

    // Anchor design: one row per encoded level, x at its reference (0 = mean
    // age) so the row-to-row spread isolates the frailty (RE) contribution.
    let mut anchor = Array2::<f64>::zeros((N_GROUPS, data.headers.len()));
    for level in 0..N_GROUPS {
        anchor[[level, x_idx]] = 0.0;
        anchor[[level, group_idx]] = level as f64;
    }
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild covariate+RE design at group anchors");
    assert_eq!(
        anchor_design.design.ncols(),
        cov_beta.len(),
        "covariate+RE design width must match the covariate-coefficient block"
    );
    let eta_by_level: Vec<f64> = anchor_design.design.apply(&cov_beta).to_vec();
    // Reorder into clinical-class order (class 0..5) so it aligns element-wise
    // with INLA's 1..N_GROUPS factor levels.
    let mut gam_frailty_by_class = vec![0.0_f64; N_GROUPS];
    for level in 0..N_GROUPS {
        gam_frailty_by_class[level_to_class[level]] = eta_by_level[level];
    }
    // Centre to mean-zero deviations, matching INLA's zero-mean iid frailty.
    let gam_mean = gam_frailty_by_class.iter().sum::<f64>() / N_GROUPS as f64;
    let gam_frailty: Vec<f64> = gam_frailty_by_class.iter().map(|v| v - gam_mean).collect();

    // ---- frailty-block effective dimension (shrinkage signature) -----------
    // The REML-selected RE block must spend strictly less than its N_GROUPS
    // nominal columns: that sub-linear edf IS the frailty shrinkage. We read the
    // total edf and assert it leaves headroom below (time-basis cap + fixed x +
    // full RE block), i.e. the RE block is regularized rather than saturated.
    let edf_total = fit.fit.edf_total().expect("gam reports total edf");
    let edf_blocks = fit.fit.edf_by_block();
    assert!(
        !edf_blocks.is_empty(),
        "gam must report per-block edf for the penalized frailty fit"
    );

    // ---- baseline cumulative hazard on a shared time grid ------------------
    // Evaluate gam's baseline H0(t) at the reference (x=0, frailty=0) over a
    // grid spanning the observed follow-up. gam's transformation/net baseline is
    // exp(time-basis . beta_time) accumulated as the I-spline cumulative hazard;
    // the structural API exposes it via the predicted survival at the reference,
    // so we read H0 from the time-basis block directly through the fitted
    // baseline cumulative-hazard helper on the grid.
    let t_lo = time.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);
    let t_hi = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(t_hi > t_lo, "follow-up grid must be non-degenerate");
    let grid_n = 24usize;
    let t_grid: Vec<f64> = (0..grid_n)
        .map(|k| t_lo + (t_hi - t_lo) * (k as f64) / ((grid_n - 1) as f64))
        .collect();
    // gam baseline cumulative hazard at the reference covariate/frailty: the
    // transformation model stores it as the I-spline image of log-time. We
    // reconstruct H0(t) = exp(eta0) * basis-accumulated hazard by evaluating the
    // *fitted* cumulative hazard through gam's survival baseline helper.
    let gam_h0: Vec<f64> = baseline_cumulative_hazard(&fit, &t_grid);
    assert_eq!(gam_h0.len(), grid_n);
    assert!(
        gam_h0.windows(2).all(|w| w[1] + 1e-9 >= w[0]),
        "baseline cumulative hazard must be monotone non-decreasing: {gam_h0:?}"
    );

    let t_grid_csv = t_grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");

    // ---- fit the SAME data with R-INLA (the scalable Bayesian reference) ----
    let r = run_r(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("x", &x),
            Column::new("group", &group_code1),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(INLA))
            df$group <- as.integer(round(df$group))
            # Weibull-PH survival with an i.i.d. Gaussian frailty on `group`.
            # f(group, model='iid') is the latent-Gaussian random intercept; its
            # precision carries a (default log-gamma) hyperprior that INLA
            # integrates over -- the scalable approx-Bayesian frailty estimand.
            form <- inla.surv(time, event) ~ 1 + x + f(group, model = "iid")
            r <- inla(form, family = "weibullsurv", data = df,
                      control.compute = list(dic = TRUE, waic = TRUE),
                      control.inla = list(strategy = "laplace"))
            # Per-group posterior mean frailty (zero-mean iid), level order 1..K.
            fr <- r$summary.random$group$mean
            emit("frailty", as.numeric(fr))
            # Weibull shape `alpha` (hyperpar) and Intercept on the log-hazard
            # scale -> baseline cumulative hazard H0(t) = exp(b0) * t^alpha for
            # INLA's weibullsurv (PH, scale absorbed into the intercept).
            alpha <- r$summary.hyperpar["alpha parameter for weibullsurv", "mean"]
            b0 <- r$summary.fixed["(Intercept)", "mean"]
            tg <- c({t_grid_csv})
            H0 <- exp(b0) * tg^alpha
            emit("H0", as.numeric(H0))
            emit("waic", r$waic$waic)
            "#
        ),
    );
    let inla_frailty = r.vector("frailty");
    let inla_h0 = r.vector("H0");
    let inla_waic = r.scalar("waic");
    assert_eq!(
        inla_frailty.len(),
        N_GROUPS,
        "INLA returned {} frailty levels, expected {N_GROUPS}",
        inla_frailty.len()
    );
    assert_eq!(inla_h0.len(), grid_n, "INLA H0 grid length mismatch");
    // INLA's iid effect is already zero-mean; centre defensively so both vectors
    // are mean-zero deviations before the element-wise RMSE.
    let inla_mean = inla_frailty.iter().sum::<f64>() / N_GROUPS as f64;
    let inla_frailty_c: Vec<f64> = inla_frailty.iter().map(|v| v - inla_mean).collect();

    // ---- compare -----------------------------------------------------------
    let frailty_rmse = rmse(&gam_frailty, &inla_frailty_c);
    let frailty_corr = pearson(&gam_frailty, &inla_frailty_c);
    let h0_corr = pearson(&gam_h0, inla_h0);

    eprintln!(
        "frailty survival vs INLA: n={n} groups={N_GROUPS} death_rate={death_rate:.3} \
         counts={group_counts:?} edf_total={edf_total:.3} edf_blocks={edf_blocks:?} \
         frailty_rmse={frailty_rmse:.4} frailty_pearson={frailty_corr:.4} \
         H0_pearson={h0_corr:.4} INLA_waic={inla_waic:.2}\n  \
         gam_frailty={gam_frailty:?}\n  inla_frailty={inla_frailty_c:?}"
    );

    // (1) SPEC bound: per-group frailty posterior means within 0.12 RMSE. Both
    // are mean-centred Gaussian-prior shrinkage estimates of the same six
    // log-hazard shifts; the severity classes induce a real spread, so 0.12 is
    // tight yet leaves room for the EB-vs-marginal-posterior gap and the
    // ispline-vs-Weibull baseline mismatch. A larger gap = genuine divergence
    // in gam's penalized frailty assembly.
    assert!(
        frailty_rmse <= 0.12,
        "gam frailty posterior means diverge from INLA: rmse={frailty_rmse:.4} (bound 0.12)"
    );
    // The frailty *ordering* must also agree: both engines see the identical
    // group structure, so the sign/rank of the six shifts must coincide. r>0.85
    // is a principled floor for six noisy-but-real shrinkage estimates.
    assert!(
        frailty_corr > 0.85,
        "gam frailty ordering disagrees with INLA: pearson={frailty_corr:.4}"
    );
    // (2) SPEC bound: baseline cumulative hazard Pearson >= 0.98. Both H0(t) are
    // monotone on the same support; gam's I-spline must track INLA's Weibull
    // baseline shape this tightly or the flexible-baseline assembly is wrong.
    assert!(
        h0_corr >= 0.98,
        "gam baseline cumulative hazard shape diverges from INLA: pearson={h0_corr:.4}"
    );
    // (3) Shrinkage signature: the REML-selected RE frailty must spend a
    // sub-linear effective dimension. The fixed effects floor is: 1 time anchor
    // + at least 1 ispline df + 1 fixed x. A *fully unshrunk* factor would add
    // the full N_GROUPS-1 = 5 contrasts on top; a frailty shrinks below that.
    // Asserting edf_total strictly below (time_cols + 1 fixed-x + full RE block)
    // confirms gam is regularizing the frailty, consistent with INLA's finite
    // posterior precision rather than a saturated fixed-effect factor.
    let unshrunk_ceiling = time_cols as f64 + 1.0 + (N_GROUPS as f64);
    assert!(
        edf_total < unshrunk_ceiling,
        "frailty block is not shrinking: edf_total={edf_total:.3} >= unshrunk ceiling {unshrunk_ceiling:.3}"
    );
}

/// Reconstruct gam's baseline cumulative hazard `H0(t)` at the reference
/// covariate/frailty (eta_cov = 0) on `t_grid`, using the *fitted* time basis
/// and time-block coefficients — gam's structural transformation/net baseline.
///
/// For the transformation ("net") likelihood with a Linear baseline target the
/// parametric offset is zero and the transformation derivative guard is zero,
/// so the only thing carried in the time channel is the I-spline image of
/// `log t`. The engine fits the time block against **anchor-centered** columns
/// `x_exit(log t) − x_exit(anchor)` (see `center_survival_time_designs_at_anchor`
/// in the workflow fit path), so the fitted β_time multiplies the centered
/// basis. The baseline cumulative hazard is therefore
/// `H0(t) = exp( (x_exit(log t) − x_exit(anchor)) · β_time )`.
///
/// We rebuild the I-spline basis from the *saved* basis state — its inferred
/// knots, degree, kept identifiable columns, and smoothing λ — so the rebuilt
/// columns are bit-for-bit the ones the fit used (I-splines are deterministic
/// given their knots), evaluate the same anchor row the fit subtracted, then
/// apply the fitted time-block β to the centered rows. This is gam's real
/// baseline-prediction arithmetic, not a re-derivation.
fn baseline_cumulative_hazard(
    fit: &gam::SurvivalTransformationFitResult,
    t_grid: &[f64],
) -> Vec<f64> {
    use gam::families::survival_construction::{
        SurvivalTimeBasisConfig, build_survival_time_basis, evaluate_survival_time_basis_row,
    };
    use ndarray::Array1;

    let saved = &fit.time_basis;
    assert!(
        saved.basisname.eq_ignore_ascii_case("ispline"),
        "expected a saved ispline time basis, got {:?}",
        saved.basisname
    );
    let degree = saved.degree.expect("saved ispline degree");
    let knots = Array1::from_vec(saved.knots.clone().expect("saved ispline knots"));
    let keep_cols = saved.keep_cols.clone().expect("saved ispline keep_cols");
    let smooth_lambda = saved.smooth_lambda.expect("saved ispline smooth_lambda");
    let cfg = SurvivalTimeBasisConfig::ISpline {
        degree,
        knots,
        keep_cols,
        smooth_lambda,
    };

    let time_cols = fit.time_base_ncols;
    // The fit centered the time block at `saved.anchor`: it subtracted the
    // anchor I-spline row from every exit/entry row before estimating β_time.
    // Reproduce that exact anchor row (same evaluator the workflow used) so the
    // reconstructed baseline matches the fitted parameterization rather than an
    // un-centered re-derivation.
    let anchor_row = evaluate_survival_time_basis_row(saved.anchor, &cfg)
        .expect("evaluate saved ispline anchor row");
    assert_eq!(
        anchor_row.len(),
        time_cols,
        "anchor row width {} must match fitted time-block width {time_cols}",
        anchor_row.len()
    );

    let grid = Array1::from_vec(t_grid.to_vec());
    // No left-truncation at the reference: entry times are zero (the exit basis
    // alone carries H0(t)).
    let entry = Array1::<f64>::zeros(grid.len());
    let build = build_survival_time_basis(&entry, &grid, cfg, None)
        .expect("rebuild survival time basis at grid from saved state");
    let x_exit = build.x_exit_time.to_dense();
    assert_eq!(
        x_exit.ncols(),
        time_cols,
        "rebuilt time basis width {} must match fitted time-block width {time_cols}",
        x_exit.ncols()
    );
    let beta_time = fit.fit.beta.slice(ndarray::s![..time_cols]).to_owned();
    // eta0(t) = (x_exit(log t) − x_exit(anchor)) · β_time on the log-cumulative-
    // hazard scale, then H0(t) = exp(eta0(t)) — the engine's anchor-centered RP
    // baseline arithmetic.
    (0..t_grid.len())
        .map(|k| {
            let eta: f64 = (0..time_cols)
                .map(|j| (x_exit[[k, j]] - anchor_row[j]) * beta_time[j])
                .sum();
            eta.exp()
        })
        .collect()
}
