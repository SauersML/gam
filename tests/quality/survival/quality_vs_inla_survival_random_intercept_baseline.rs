//! End-to-end OBJECTIVE quality: gam's random-intercept *frailty* in flexible
//! survival — a latent Gaussian group effect that shifts the baseline hazard,
//! fitted by penalized likelihood with REML-selected frailty variance.
//!
//! OBJECTIVE METRIC ASSERTED (this is the pass/fail claim):
//!   * PREDICTIVE DISCRIMINATION. On a deterministic, fixed train/test split of
//!     the real heart-failure data, gam is fit on the TRAIN rows, its per-subject
//!     proportional-hazards risk score (the covariate + frailty linear predictor
//!     `eta_cov`, which under the transformation/net model is monotone in the
//!     hazard) is evaluated on the held-out TEST rows, and Harrell's concordance
//!     index `C` between that risk and the held-out (time, event) outcome must
//!     clear an ABSOLUTE bar `C >= 0.55` (real, above-chance discrimination on a
//!     hard clinical endpoint). The C-index is computed on gam's OWN held-out
//!     predictions — not on agreement with any tool.
//!   * STRUCTURE. gam's baseline cumulative hazard `H0(t)` is finite, positive,
//!     and monotone non-decreasing, and the implied baseline survival
//!     `S0(t) = exp(-H0(t))` lies in [0,1] and is non-increasing — the defining
//!     mathematical axioms of a survival baseline.
//!   * SHRINKAGE SIGNATURE. The REML-selected RE block spends a strictly
//!     sub-linear effective dimension (edf below the unshrunk fixed-factor
//!     ceiling): the defining signature that gam is regularizing the frailty
//!     rather than spending the full per-group df.
//!
//! BASELINE TO MATCH-OR-BEAT: R-INLA (the scalable approximate-Bayesian
//! latent-Gaussian frailty gold standard) is fit on the SAME train rows and its
//! own held-out PH risk score `b0 + slope*x + frailty[group]` is scored with the
//! IDENTICAL C-index routine on the IDENTICAL test rows. gam must MATCH-OR-BEAT
//! INLA on discrimination: `C_gam >= C_inla - 0.02`. INLA is a baseline on the
//! objective metric, NOT a fitted-output target — we never assert gam reproduces
//! INLA's coefficients, frailty values, or baseline shape; we only require gam to
//! discriminate held-out risk at least as well. (rel-context diagnostics — the
//! per-group frailty agreement and H0 correlation — are still printed via
//! eprintln! for human inspection, but are NOT pass/fail criteria.)
//!
//! Reference model (INLA): `inla.surv(time, event) ~ 1 + x + f(group, iid)`,
//! family `weibullsurv` — a Weibull-PH baseline with an i.i.d. zero-mean Gaussian
//! frailty on `group`. gam expresses the same latent-Gaussian survival structure
//! through its transformation ("net") survival likelihood with a flexible
//! monotone I-spline baseline on log-time and a ridge-penalized random-effect
//! block on `group`:
//!
//!     Surv(time, event) ~ x + s(group, bs = "re") + survmodel(spec = "net")
//!
//! with `survival_likelihood = "transformation"` and `time_basis = "ispline"`.
//!
//! Data. The real `heart_failure_clinical_records_dataset` (n = 299, death rate
//! ~0.32). Ejection fraction binned into six cardiac-severity classes is the
//! risk-class factor frailty models exist to capture; standardized `age` is the
//! fixed covariate. The identical (time, event, x, group) columns and the
//! identical train/test partition are handed to gam and to INLA.
//!
//! A failing assertion because gam genuinely under-discriminates relative to the
//! absolute bar or to INLA is acceptable and must NOT be papered over by
//! loosening a bound or editing gam.

use csv::StringRecord;
use gam::families::survival::construction::SurvivalLikelihoodMode;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, r_package_available, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const VETERAN_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/veteran_lung.csv"
);

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

/// Harrell's concordance index between a per-subject risk score (HIGHER = more
/// hazardous => expected to fail SOONER) and a right-censored survival outcome.
///
/// A pair `(i, j)` is *comparable* when the one that fails first does so at a
/// strictly smaller time and that earlier event is observed (event == 1): then
/// we know subject i (the earlier failure) truly out-ranks j in risk. The pair
/// is *concordant* when the higher risk score belongs to the earlier failure,
/// *tied* when the risk scores are equal, and *discordant* otherwise. Pairs
/// where both are censored, or where the earlier of the two is censored (so the
/// order is unknown), are not comparable and excluded. `C = (concordant +
/// 0.5*tied) / comparable`. This is the standard Harrell estimator; it needs
/// only the per-subject risk score and is invariant to any monotone baseline
/// (so it is exactly right for a proportional-hazards risk).
fn harrell_concordance(risk: &[f64], time: &[f64], event: &[f64]) -> f64 {
    assert_eq!(
        risk.len(),
        time.len(),
        "concordance risk/time length mismatch"
    );
    assert_eq!(
        risk.len(),
        event.len(),
        "concordance risk/event length mismatch"
    );
    let n = risk.len();
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Identify the earlier-failing subject of the pair.
            let (early, late) = if time[i] < time[j] {
                (i, j)
            } else if time[j] < time[i] {
                (j, i)
            } else {
                // Equal event times carry no order information about who failed
                // first, so the pair is not comparable.
                continue;
            };
            // The earlier subject must be an observed event for the order to be
            // known; otherwise (early censored) we cannot say `late` outlived it.
            if event[early] < 0.5 {
                continue;
            }
            comparable += 1.0;
            let r_early = risk[early];
            let r_late = risk[late];
            if (r_early - r_late).abs() <= 1e-12 {
                concordant += 0.5;
            } else if r_early > r_late {
                // Earlier failure carries higher risk => concordant.
                concordant += 1.0;
            }
        }
    }
    assert!(
        comparable > 0.0,
        "no comparable (ordered, observed-earlier) survival pairs in the test split"
    );
    concordant / comparable
}

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

    // Every group must be populated so INLA's iid effect and gam's RE block both
    // span all six levels.
    let mut group_counts = [0usize; N_GROUPS];
    for &g in &grp {
        group_counts[g] += 1;
    }
    assert!(
        group_counts.iter().all(|&c| c >= 5),
        "each severity class needs >=5 patients for a stable frailty estimate: {group_counts:?}"
    );

    // Standardize age (mean 0, sd 1) so the fixed-effect slope is on a common
    // scale and the frailty carries the residual group structure.
    let age_mean = age.iter().sum::<f64>() / n as f64;
    let age_sd =
        (age.iter().map(|a| (a - age_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let x: Vec<f64> = age.iter().map(|a| (a - age_mean) / age_sd).collect();
    let group_code1: Vec<f64> = grp.iter().map(|&g| (g + 1) as f64).collect();

    // ---- deterministic train/test split ------------------------------------
    // Fixed, reproducible partition: every 4th row (index % 4 == 0) is held out
    // for testing, the rest train. This is data-independent and identical for
    // gam and for INLA, so neither tool sees the held-out outcomes and the
    // comparison is apples-to-apples. ~25% test (75 rows) keeps enough comparable
    // event pairs for a stable concordance estimate while leaving a substantial
    // training set.
    let is_test: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test[i]).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test[i]).collect();
    assert!(
        train_idx.len() > 200 && test_idx.len() >= 70,
        "split sizes off: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    // Both splits must still carry observed events (for comparable pairs) and
    // span groups (so the frailty levels are learnable and predictable).
    let train_events: f64 = train_idx.iter().map(|&i| event[i]).sum();
    let test_events: f64 = test_idx.iter().map(|&i| event[i]).sum();
    assert!(
        train_events >= 30.0 && test_events >= 10.0,
        "need observed events in both splits: train_events={train_events} test_events={test_events}"
    );

    // ---- fit gam on the TRAIN rows: transformation/net survival + RE frailty
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "x".to_string(),
        "group".to_string(),
    ];
    let train_rows: Vec<StringRecord> = train_idx
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                time[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
                format!("grp{}", grp[i]),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, train_rows)
        .expect("encode train frailty survival data");
    let col = data.column_map();
    let x_idx = col["x"];
    let group_idx = col["group"];

    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
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
    let cov_beta = beta.slice(ndarray::s![time_cols..]).to_owned();

    // Map each encoded group level to its clinical class (the encoder orders
    // levels by first appearance, not by numeric class).
    let group_encoded: Vec<f64> = data.values.column(group_idx).to_vec();
    let mut level_to_class = vec![usize::MAX; N_GROUPS];
    let mut class_to_level = vec![usize::MAX; N_GROUPS];
    for (row, &train_i) in train_idx.iter().enumerate() {
        let level = group_encoded[row].round() as usize;
        assert!(level < N_GROUPS, "encoded group level {level} out of range");
        level_to_class[level] = grp[train_i];
        class_to_level[grp[train_i]] = level;
    }
    assert!(
        level_to_class.iter().all(|&c| c != usize::MAX),
        "every encoded level must map to a clinical class: {level_to_class:?}"
    );
    assert!(
        class_to_level.iter().all(|&l| l != usize::MAX),
        "every clinical class must map to an encoded level (each group must appear in TRAIN)"
    );

    // ---- gam's held-out PH risk score on the TEST rows ----------------------
    // Build the covariate+RE design at the TEST subjects (their standardized x
    // and their group's encoded level) and apply the fitted covariate block.
    // Under the transformation/net model log H(t) = log H0(t) + eta_cov, so
    // eta_cov is the proportional-hazards risk score: monotone in the hazard,
    // baseline-shape invariant, exactly what concordance ranks.
    let mut test_design_in = Array2::<f64>::zeros((test_idx.len(), data.headers.len()));
    for (row, &i) in test_idx.iter().enumerate() {
        test_design_in[[row, x_idx]] = x[i];
        test_design_in[[row, group_idx]] = class_to_level[grp[i]] as f64;
    }
    let test_design = build_term_collection_design(test_design_in.view(), &fit.resolvedspec)
        .expect("build covariate+RE design at TEST rows");
    assert_eq!(
        test_design.design.ncols(),
        cov_beta.len(),
        "covariate+RE design width must match the covariate-coefficient block"
    );
    let gam_risk: Vec<f64> = test_design.design.apply(&cov_beta).to_vec();
    assert_eq!(gam_risk.len(), test_idx.len());

    let test_time: Vec<f64> = test_idx.iter().map(|&i| time[i]).collect();
    let test_event: Vec<f64> = test_idx.iter().map(|&i| event[i]).collect();
    let gam_c = harrell_concordance(&gam_risk, &test_time, &test_event);

    // ---- per-group frailty (context diagnostic only, NOT a pass criterion) --
    // Anchor row per encoded level at x = 0 isolates the frailty contribution.
    let mut anchor = Array2::<f64>::zeros((N_GROUPS, data.headers.len()));
    for level in 0..N_GROUPS {
        anchor[[level, x_idx]] = 0.0;
        anchor[[level, group_idx]] = level as f64;
    }
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild covariate+RE design at group anchors");
    let eta_by_level: Vec<f64> = anchor_design.design.apply(&cov_beta).to_vec();
    let mut gam_frailty_by_class = vec![0.0_f64; N_GROUPS];
    for level in 0..N_GROUPS {
        gam_frailty_by_class[level_to_class[level]] = eta_by_level[level];
    }
    let gam_mean = gam_frailty_by_class.iter().sum::<f64>() / N_GROUPS as f64;
    let gam_frailty: Vec<f64> = gam_frailty_by_class.iter().map(|v| v - gam_mean).collect();

    // ---- frailty-block effective dimension (shrinkage signature) -----------
    let edf_total = fit.fit.edf_total().expect("gam reports total edf");
    let edf_blocks = fit.fit.edf_by_block();
    assert!(
        !edf_blocks.is_empty(),
        "gam must report per-block edf for the penalized frailty fit"
    );

    // ---- baseline cumulative hazard on a shared time grid (structure check) -
    let t_lo = time.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);
    let t_hi = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(t_hi > t_lo, "follow-up grid must be non-degenerate");
    let grid_n = 24usize;
    let t_grid: Vec<f64> = (0..grid_n)
        .map(|k| t_lo + (t_hi - t_lo) * (k as f64) / ((grid_n - 1) as f64))
        .collect();
    let gam_h0: Vec<f64> = baseline_cumulative_hazard(&fit, &t_grid);
    assert_eq!(gam_h0.len(), grid_n);

    // STRUCTURE: H0(t) finite, positive, monotone non-decreasing; baseline
    // survival S0(t) = exp(-H0(t)) in [0,1] and non-increasing. These are the
    // mathematical axioms of a survival baseline — an objective correctness
    // claim, independent of any reference tool.
    assert!(
        gam_h0.iter().all(|v| v.is_finite() && *v > 0.0),
        "baseline cumulative hazard must be finite and positive: {gam_h0:?}"
    );
    assert!(
        gam_h0.windows(2).all(|w| w[1] + 1e-9 >= w[0]),
        "baseline cumulative hazard must be monotone non-decreasing: {gam_h0:?}"
    );
    let s0: Vec<f64> = gam_h0.iter().map(|h| (-h).exp()).collect();
    assert!(
        s0.iter().all(|s| (0.0..=1.0).contains(s)),
        "baseline survival S0(t)=exp(-H0) must lie in [0,1]: {s0:?}"
    );
    assert!(
        s0.windows(2).all(|w| w[1] <= w[0] + 1e-9),
        "baseline survival must be non-increasing: {s0:?}"
    );

    let t_grid_csv = t_grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");

    // ---- fit the SAME TRAIN rows with R-INLA (the match-or-beat baseline) ----
    // INLA is fit on TRAIN and emits the components of its held-out PH risk
    // score: the `x` slope, the per-group frailty posterior means, and the per-
    // TEST-row group code. We assemble INLA's test risk = slope*x + frailty[grp]
    // in Rust and score it with the IDENTICAL concordance routine on the
    // IDENTICAL test rows, so the only thing compared is held-out discrimination.
    let train_time: Vec<f64> = train_idx.iter().map(|&i| time[i]).collect();
    let train_event: Vec<f64> = train_idx.iter().map(|&i| event[i]).collect();
    let train_x: Vec<f64> = train_idx.iter().map(|&i| x[i]).collect();
    let train_group1: Vec<f64> = train_idx.iter().map(|&i| group_code1[i]).collect();
    // SANCTIONED ENVIRONMENTAL GATE (CUDA/DoubleML category): R-INLA is provisioned
    // best-effort in CI and is frequently absent. When `library(INLA)` would fail to
    // load, we cannot run the match-or-beat baseline arm — but gam's OWN held-out
    // discrimination is already computed (gam_c, above), so we still assert gam's
    // TOOL-FREE absolute concordance bar (C >= 0.55) and skip ONLY the gam-vs-INLA
    // comparison. We never weaken or drop the absolute gam-side claim.
    if !r_package_available("INLA") {
        eprintln!(
            "R-INLA unavailable — asserting gam's tool-free absolute quality only \
             (skipping match-or-beat arm): gam_C={gam_c:.4}"
        );
        assert!(
            gam_c >= 0.55,
            "gam under-discriminates held-out survival: C={gam_c:.4} (absolute bar 0.55)"
        );
        return;
    }
    let r = run_r(
        &[
            Column::new("time", &train_time),
            Column::new("event", &train_event),
            Column::new("x", &train_x),
            Column::new("group", &train_group1),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(INLA))
            df$group <- as.integer(round(df$group))
            # Weibull-PH survival with an i.i.d. Gaussian frailty on `group`.
            form <- inla.surv(time, event) ~ 1 + x + f(group, model = "iid")
            r <- inla(form, family = "weibullsurv", data = df,
                      control.compute = list(dic = TRUE, waic = TRUE),
                      control.inla = list(strategy = "laplace"))
            # Per-group posterior-mean frailty (zero-mean iid), level order 1..K.
            fr <- r$summary.random$group$mean
            emit("frailty", as.numeric(fr))
            # Fixed-effect slope on x (the covariate part of the PH risk score).
            slope <- r$summary.fixed["x", "mean"]
            emit("slope", as.numeric(slope))
            # Diagnostics-only: Weibull shape + intercept -> H0(t)=exp(b0)*t^alpha.
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
    let inla_slope = r.scalar("slope");
    let inla_h0 = r.vector("H0");
    let inla_waic = r.scalar("waic");
    assert_eq!(
        inla_frailty.len(),
        N_GROUPS,
        "INLA returned {} frailty levels, expected {N_GROUPS}",
        inla_frailty.len()
    );
    assert_eq!(inla_h0.len(), grid_n, "INLA H0 grid length mismatch");

    // INLA's held-out PH risk score on the TEST rows. INLA's iid frailty levels
    // are in numeric group order 1..K, i.e. frailty[k] for clinical class k.
    let inla_risk: Vec<f64> = test_idx
        .iter()
        .map(|&i| inla_slope * x[i] + inla_frailty[grp[i]])
        .collect();
    let inla_c = harrell_concordance(&inla_risk, &test_time, &test_event);

    // ---- context-only diagnostics (NOT pass/fail) --------------------------
    let inla_mean = inla_frailty.iter().sum::<f64>() / N_GROUPS as f64;
    let inla_frailty_c: Vec<f64> = inla_frailty.iter().map(|v| v - inla_mean).collect();
    let frailty_rmse = rmse(&gam_frailty, &inla_frailty_c);
    let frailty_corr = pearson(&gam_frailty, &inla_frailty_c);
    let h0_corr = pearson(&gam_h0, inla_h0);

    eprintln!(
        "frailty survival (held-out): n={n} train={} test={} groups={N_GROUPS} \
         death_rate={death_rate:.3} counts={group_counts:?} edf_total={edf_total:.3} \
         edf_blocks={edf_blocks:?} gam_C={gam_c:.4} inla_C={inla_c:.4} \
         [context-only] frailty_rmse={frailty_rmse:.4} frailty_pearson={frailty_corr:.4} \
         H0_pearson={h0_corr:.4} INLA_waic={inla_waic:.2}\n  \
         gam_frailty={gam_frailty:?}\n  inla_frailty={inla_frailty_c:?}",
        train_idx.len(),
        test_idx.len()
    );

    // (1) PRIMARY OBJECTIVE: held-out concordance clears an absolute bar. A
    // C-index of 0.5 is random guessing; 0.55 is a principled floor for real,
    // above-chance discrimination of a hard clinical endpoint from age + a
    // six-level severity frailty alone. This is gam's OWN held-out predictive
    // quality, not agreement with any tool.
    assert!(
        gam_c >= 0.55,
        "gam under-discriminates held-out survival: C={gam_c:.4} (absolute bar 0.55)"
    );
    // (1b) MATCH-OR-BEAT: gam must discriminate held-out risk at least as well
    // as the INLA baseline (within a 0.02 estimation-noise margin). INLA is a
    // baseline on the objective metric, never a fitted-output target.
    assert!(
        gam_c >= inla_c - 0.02,
        "gam loses to INLA on held-out discrimination: gam_C={gam_c:.4} inla_C={inla_c:.4} (margin 0.02)"
    );
    // (2) Structure assertions on H0/S0 are above (axioms, no reference needed).
    // (3) SHRINKAGE SIGNATURE: the REML-selected RE frailty must spend a
    // sub-linear effective dimension. A fully unshrunk factor would add the full
    // N_GROUPS contrasts on top of the time basis + fixed x; a frailty shrinks
    // below that. edf_total strictly below (time_cols + 1 fixed-x + full RE
    // block) confirms gam is regularizing the frailty.
    let unshrunk_ceiling = time_cols as f64 + 1.0 + (N_GROUPS as f64);
    assert!(
        edf_total < unshrunk_ceiling,
        "frailty block is not shrinking: edf_total={edf_total:.3} >= unshrunk ceiling {unshrunk_ceiling:.3}"
    );
}

/// REAL-DATA ARM of the random-intercept survival-frailty capability.
///
/// Same gam capability as `gam_survival_frailty_random_intercept_matches_inla`
/// (transformation/net survival likelihood + monotone I-spline baseline on
/// log-time + a ridge-penalized random-effect frailty block on a grouping
/// factor), exercised on a DIFFERENT real clinical dataset so the accuracy claim
/// generalizes beyond one cohort.
///
/// Dataset SOURCE: the Veterans' Administration lung-cancer trial of Kalbfleisch
/// & Prentice (1980), "The Statistical Analysis of Failure Time Data" — shipped
/// as `survival::veteran` in R and committed here as
/// `bench/datasets/veteran_lung.csv` (n = 137 men with advanced inoperable lung
/// cancer; columns: trt, celltype, time (days), status, karno (Karnofsky
/// performance score 0-100), diagtime, age, prior). Tumor histology `celltype`
/// (squamous / smallcell / adeno / large) is the natural risk-class factor a
/// frailty model groups patients by; the Karnofsky performance score `karno`
/// (standardized) is the strongest single fixed prognostic covariate.
///
/// OBJECTIVE METRIC ASSERTED (pass/fail): held-out PROPORTIONAL-HAZARDS
/// DISCRIMINATION. gam is fit on a deterministic train split, its covariate+RE
/// linear predictor `eta_cov` (monotone in the hazard under the net model) is
/// evaluated on the held-out test rows, and Harrell's concordance `C` between
/// that risk and the held-out (time, status) outcome must clear an ABSOLUTE bar
/// `C >= 0.60` (real, above-chance discrimination; Karnofsky score is a famously
/// strong predictor in this cohort) AND match-or-beat the R-INLA Weibull-PH
/// frailty baseline on the SAME metric and rows: `C_gam >= C_inla - 0.02`. INLA
/// is a baseline on the objective metric, never a fitted-output target. The
/// same baseline-survival STRUCTURE axioms (H0 finite/positive/monotone, S0 in
/// [0,1] non-increasing) and the sub-linear-edf SHRINKAGE SIGNATURE are also
/// asserted. A failing assertion from genuine under-discrimination is acceptable
/// and must NOT be papered over by loosening a bound.
#[test]
fn gam_survival_frailty_random_intercept_matches_inla_on_real_data() {
    init_parallelism();

    // Four tumor-histology classes; the frailty (and INLA's iid effect) span all.
    const N_CELL: usize = 4;

    // ---- load the real veteran lung-cancer trial -----------------------------
    let ds = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    // `celltype` is a string factor; the inferred-schema encoder stores its level
    // code in the numeric matrix. We read raw celltype strings from the CSV to
    // build a stable, name-keyed class index (independent of encoder level order)
    // so gam and INLA agree on which class each subject belongs to.
    let file = File::open(VETERAN_CSV).expect("open veteran_lung csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().expect("header line").expect("read header");
    let hcols: Vec<&str> = header.split(',').collect();
    let celltype_col = hcols
        .iter()
        .position(|c| *c == "celltype")
        .expect("celltype column");
    let mut celltype_str = Vec::<String>::new();
    for line in lines {
        let line = line.expect("read data line");
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        celltype_str.push(f[celltype_col].to_string());
    }

    let time: Vec<f64> = ds.values.column(time_idx).to_vec();
    let event: Vec<f64> = ds.values.column(status_idx).to_vec();
    let karno: Vec<f64> = ds.values.column(karno_idx).to_vec();
    let n = time.len();
    assert_eq!(
        celltype_str.len(),
        n,
        "raw celltype rows {} must match encoded rows {n}",
        celltype_str.len()
    );
    assert_eq!(n, 137, "veteran_lung should have 137 rows, got {n}");

    // Map celltype names -> stable class index 0..N_CELL in first-appearance
    // order, and also a numeric 1-based code for INLA's iid factor.
    let mut class_names = Vec::<String>::new();
    let cell_class: Vec<usize> = celltype_str
        .iter()
        .map(|name| {
            if let Some(p) = class_names.iter().position(|c| c == name) {
                p
            } else {
                class_names.push(name.clone());
                class_names.len() - 1
            }
        })
        .collect();
    assert_eq!(
        class_names.len(),
        N_CELL,
        "veteran has {N_CELL} histology classes, found {}: {class_names:?}",
        class_names.len()
    );
    let mut class_counts = [0usize; N_CELL];
    for &c in &cell_class {
        class_counts[c] += 1;
    }
    assert!(
        class_counts.iter().all(|&c| c >= 20),
        "each histology class needs >=20 patients for a stable frailty: {class_counts:?}"
    );
    let cell_code1: Vec<f64> = cell_class.iter().map(|&c| (c + 1) as f64).collect();

    let event_rate: f64 = event.iter().sum::<f64>() / n as f64;
    assert!(
        event_rate >= 0.8,
        "veteran is lightly censored; event rate should be high, got {event_rate:.3}"
    );

    // Standardize the Karnofsky score so the fixed slope is on a common scale.
    let karno_mean = karno.iter().sum::<f64>() / n as f64;
    let karno_sd =
        (karno.iter().map(|k| (k - karno_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let x: Vec<f64> = karno.iter().map(|k| (k - karno_mean) / karno_sd).collect();

    // ---- deterministic train/test split: every 4th row held out -------------
    let is_test: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test[i]).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test[i]).collect();
    assert!(
        train_idx.len() > 95 && test_idx.len() >= 30,
        "split sizes off: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    let train_events: f64 = train_idx.iter().map(|&i| event[i]).sum();
    let test_events: f64 = test_idx.iter().map(|&i| event[i]).sum();
    assert!(
        train_events >= 40.0 && test_events >= 15.0,
        "need observed events in both splits: train_events={train_events} test_events={test_events}"
    );
    // Every histology class must appear in TRAIN so its frailty level is learnable.
    let mut train_class_counts = [0usize; N_CELL];
    for &i in &train_idx {
        train_class_counts[cell_class[i]] += 1;
    }
    assert!(
        train_class_counts.iter().all(|&c| c >= 5),
        "each histology class needs training rows to estimate its frailty: {train_class_counts:?}"
    );

    // ---- fit gam on the TRAIN rows: transformation/net survival + RE frailty -
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "x".to_string(),
        "group".to_string(),
    ];
    let train_rows: Vec<StringRecord> = train_idx
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                time[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
                format!("cell{}", cell_class[i]),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, train_rows)
        .expect("encode train frailty survival data");
    let dcol = data.column_map();
    let x_idx = dcol["x"];
    let group_idx = dcol["group"];

    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
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

    let beta = &fit.fit.beta;
    let time_cols = fit.time_base_ncols;
    assert!(
        beta.len() > time_cols,
        "beta must carry covariate/RE columns beyond the {time_cols} time-basis columns; beta.len()={}",
        beta.len()
    );
    let cov_beta = beta.slice(ndarray::s![time_cols..]).to_owned();

    // Map each encoded group level back to its histology class.
    let group_encoded: Vec<f64> = data.values.column(group_idx).to_vec();
    let mut level_to_class = vec![usize::MAX; N_CELL];
    let mut class_to_level = vec![usize::MAX; N_CELL];
    for (row, &train_i) in train_idx.iter().enumerate() {
        let level = group_encoded[row].round() as usize;
        assert!(level < N_CELL, "encoded group level {level} out of range");
        level_to_class[level] = cell_class[train_i];
        class_to_level[cell_class[train_i]] = level;
    }
    assert!(
        level_to_class.iter().all(|&c| c != usize::MAX)
            && class_to_level.iter().all(|&l| l != usize::MAX),
        "every histology class must map to an encoded level (each appears in TRAIN): {level_to_class:?}"
    );

    // ---- gam's held-out PH risk score on the TEST rows ----------------------
    let mut test_design_in = Array2::<f64>::zeros((test_idx.len(), data.headers.len()));
    for (row, &i) in test_idx.iter().enumerate() {
        test_design_in[[row, x_idx]] = x[i];
        test_design_in[[row, group_idx]] = class_to_level[cell_class[i]] as f64;
    }
    let test_design = build_term_collection_design(test_design_in.view(), &fit.resolvedspec)
        .expect("build covariate+RE design at TEST rows");
    assert_eq!(
        test_design.design.ncols(),
        cov_beta.len(),
        "covariate+RE design width must match the covariate-coefficient block"
    );
    let gam_risk: Vec<f64> = test_design.design.apply(&cov_beta).to_vec();
    assert_eq!(gam_risk.len(), test_idx.len());

    let test_time: Vec<f64> = test_idx.iter().map(|&i| time[i]).collect();
    let test_event: Vec<f64> = test_idx.iter().map(|&i| event[i]).collect();
    let gam_c = harrell_concordance(&gam_risk, &test_time, &test_event);

    // ---- per-class frailty (context diagnostic only) ------------------------
    let mut anchor = Array2::<f64>::zeros((N_CELL, data.headers.len()));
    for level in 0..N_CELL {
        anchor[[level, x_idx]] = 0.0;
        anchor[[level, group_idx]] = level as f64;
    }
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild covariate+RE design at class anchors");
    let eta_by_level: Vec<f64> = anchor_design.design.apply(&cov_beta).to_vec();
    let mut gam_frailty_by_class = vec![0.0_f64; N_CELL];
    for level in 0..N_CELL {
        gam_frailty_by_class[level_to_class[level]] = eta_by_level[level];
    }
    let gam_mean = gam_frailty_by_class.iter().sum::<f64>() / N_CELL as f64;
    let gam_frailty: Vec<f64> = gam_frailty_by_class.iter().map(|v| v - gam_mean).collect();

    // ---- shrinkage-signature edf -------------------------------------------
    let edf_total = fit.fit.edf_total().expect("gam reports total edf");
    let edf_blocks = fit.fit.edf_by_block();
    assert!(
        !edf_blocks.is_empty(),
        "gam must report per-block edf for the penalized frailty fit"
    );

    // ---- baseline cumulative hazard on a shared time grid (structure) -------
    let t_lo = time.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);
    let t_hi = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(t_hi > t_lo, "follow-up grid must be non-degenerate");
    let grid_n = 24usize;
    let t_grid: Vec<f64> = (0..grid_n)
        .map(|k| t_lo + (t_hi - t_lo) * (k as f64) / ((grid_n - 1) as f64))
        .collect();
    let gam_h0: Vec<f64> = baseline_cumulative_hazard(&fit, &t_grid);
    assert_eq!(gam_h0.len(), grid_n);
    assert!(
        gam_h0.iter().all(|v| v.is_finite() && *v > 0.0),
        "baseline cumulative hazard must be finite and positive: {gam_h0:?}"
    );
    assert!(
        gam_h0.windows(2).all(|w| w[1] + 1e-9 >= w[0]),
        "baseline cumulative hazard must be monotone non-decreasing: {gam_h0:?}"
    );
    let s0: Vec<f64> = gam_h0.iter().map(|h| (-h).exp()).collect();
    assert!(
        s0.iter().all(|s| (0.0..=1.0).contains(s)),
        "baseline survival S0(t)=exp(-H0) must lie in [0,1]: {s0:?}"
    );
    assert!(
        s0.windows(2).all(|w| w[1] <= w[0] + 1e-9),
        "baseline survival must be non-increasing: {s0:?}"
    );
    let t_grid_csv = t_grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");

    // ---- fit the SAME TRAIN rows with R-INLA (the match-or-beat baseline) ----
    let train_time: Vec<f64> = train_idx.iter().map(|&i| time[i]).collect();
    let train_event: Vec<f64> = train_idx.iter().map(|&i| event[i]).collect();
    let train_x: Vec<f64> = train_idx.iter().map(|&i| x[i]).collect();
    let train_group1: Vec<f64> = train_idx.iter().map(|&i| cell_code1[i]).collect();
    // SANCTIONED ENVIRONMENTAL GATE (CUDA/DoubleML category): R-INLA is provisioned
    // best-effort in CI and is frequently absent. When `library(INLA)` would fail to
    // load, the match-or-beat baseline arm cannot run — but gam's OWN held-out
    // discrimination is already computed (gam_c, above), so we still assert gam's
    // TOOL-FREE absolute concordance bar (C >= 0.60) and skip ONLY the gam-vs-INLA
    // comparison. We never weaken or drop the absolute gam-side claim.
    if !r_package_available("INLA") {
        eprintln!(
            "R-INLA unavailable — asserting gam's tool-free absolute quality only \
             (skipping match-or-beat arm): gam_C={gam_c:.4}"
        );
        assert!(
            gam_c >= 0.60,
            "gam under-discriminates held-out survival: C={gam_c:.4} (absolute bar 0.60)"
        );
        return;
    }
    let r = run_r(
        &[
            Column::new("time", &train_time),
            Column::new("event", &train_event),
            Column::new("x", &train_x),
            Column::new("group", &train_group1),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(INLA))
            df$group <- as.integer(round(df$group))
            form <- inla.surv(time, event) ~ 1 + x + f(group, model = "iid")
            r <- inla(form, family = "weibullsurv", data = df,
                      control.compute = list(dic = TRUE, waic = TRUE),
                      control.inla = list(strategy = "laplace"))
            fr <- r$summary.random$group$mean
            emit("frailty", as.numeric(fr))
            slope <- r$summary.fixed["x", "mean"]
            emit("slope", as.numeric(slope))
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
    let inla_slope = r.scalar("slope");
    let inla_h0 = r.vector("H0");
    let inla_waic = r.scalar("waic");
    assert_eq!(
        inla_frailty.len(),
        N_CELL,
        "INLA returned {} frailty levels, expected {N_CELL}",
        inla_frailty.len()
    );
    assert_eq!(inla_h0.len(), grid_n, "INLA H0 grid length mismatch");

    // INLA's held-out PH risk on TEST rows; iid levels are in numeric code order
    // 1..K, i.e. frailty[c] for histology class c.
    let inla_risk: Vec<f64> = test_idx
        .iter()
        .map(|&i| inla_slope * x[i] + inla_frailty[cell_class[i]])
        .collect();
    let inla_c = harrell_concordance(&inla_risk, &test_time, &test_event);

    // ---- context-only diagnostics ------------------------------------------
    let inla_mean = inla_frailty.iter().sum::<f64>() / N_CELL as f64;
    let inla_frailty_c: Vec<f64> = inla_frailty.iter().map(|v| v - inla_mean).collect();
    let frailty_rmse = rmse(&gam_frailty, &inla_frailty_c);
    let frailty_corr = pearson(&gam_frailty, &inla_frailty_c);
    let h0_corr = pearson(&gam_h0, inla_h0);

    eprintln!(
        "veteran frailty survival (held-out): n={n} train={} test={} classes={N_CELL} \
         event_rate={event_rate:.3} counts={class_counts:?} edf_total={edf_total:.3} \
         edf_blocks={edf_blocks:?} gam_C={gam_c:.4} inla_C={inla_c:.4} \
         [context-only] frailty_rmse={frailty_rmse:.4} frailty_pearson={frailty_corr:.4} \
         H0_pearson={h0_corr:.4} INLA_waic={inla_waic:.2}\n  classes={class_names:?}\n  \
         gam_frailty={gam_frailty:?}\n  inla_frailty={inla_frailty_c:?}",
        train_idx.len(),
        test_idx.len()
    );

    // (1) PRIMARY OBJECTIVE: held-out concordance clears an absolute bar. The
    // Karnofsky performance score is a famously strong prognostic covariate in
    // this cohort, so a competent PH frailty model discriminates well above
    // chance; C >= 0.60 is a principled floor (0.5 is random) and is gam's OWN
    // held-out predictive quality, not agreement with any tool.
    assert!(
        gam_c >= 0.60,
        "gam under-discriminates held-out survival: C={gam_c:.4} (absolute bar 0.60)"
    );
    // (1b) MATCH-OR-BEAT: gam discriminates held-out risk at least as well as the
    // R-INLA baseline (within a 0.02 estimation-noise margin).
    assert!(
        gam_c >= inla_c - 0.02,
        "gam loses to INLA on held-out discrimination: gam_C={gam_c:.4} inla_C={inla_c:.4} (margin 0.02)"
    );
    // (2) STRUCTURE axioms on H0/S0 are asserted above.
    // (3) SHRINKAGE SIGNATURE: the REML-selected RE frailty spends a sub-linear
    // effective dimension below the unshrunk ceiling (time basis + fixed x + full
    // per-class block).
    let unshrunk_ceiling = time_cols as f64 + 1.0 + (N_CELL as f64);
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
    use gam::families::survival::construction::{
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
    let anchor_row = evaluate_survival_time_basis_row(saved.anchor, &cfg)
        .expect("evaluate saved ispline anchor row");
    assert_eq!(
        anchor_row.len(),
        time_cols,
        "anchor row width {} must match fitted time-block width {time_cols}",
        anchor_row.len()
    );

    let grid = Array1::from_vec(t_grid.to_vec());
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
    (0..t_grid.len())
        .map(|k| {
            let eta: f64 = (0..time_cols)
                .map(|j| (x_exit[[k, j]] - anchor_row[j]) * beta_time[j])
                .sum();
            eta.exp()
        })
        .collect()
}
