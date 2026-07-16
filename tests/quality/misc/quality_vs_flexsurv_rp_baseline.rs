//! End-to-end OBJECTIVE quality of gam's Royston-Parmar flexible-parametric
//! survival baseline (the `survival_likelihood="transformation"` family with an
//! explicit net-survival hazard spec, `survmodel(spec=net)`).
//!
//! OBJECTIVE METRIC ASSERTED — held-out concordance (Harrell's C-index).
//! A quality test must assert gam's objective quality, never that gam reproduces
//! a reference tool's fitted numbers. "We get the same covariate slope / log Λ as
//! flexsurv" proves nothing — matching a reference optimizer can still mean both
//! fits rank patients badly. So instead of comparing fitted coefficients, this
//! test measures something with an unambiguous right answer: *does the fitted
//! model rank unseen subjects' survival ordering correctly?*
//!
//! We make a deterministic train/test split of a real survival cohort, fit
//! gam's Royston-Parmar model on the TRAIN rows only, then on the held-out TEST
//! rows compute, for each subject, the proportional-hazards linear predictor
//! η = c(cov)·β_cov (a monotone relative-risk score: higher η ⇒ higher hazard
//! ⇒ shorter survival under PH). Harrell's concordance is the fraction of
//! comparable, orderable test pairs whose risk ranking agrees with the observed
//! (possibly right-censored) survival ordering. C = 0.5 is random ranking, C = 1.0
//! is a perfect ranker.
//!
//! The PRIMARY claim is absolute: gam's held-out C clears a bar strictly above
//! chance (the model has learned a real survival ordering, not noise). flexsurv is
//! KEPT only as a BASELINE-TO-MATCH-OR-BEAT on the *same* objective metric: fit on
//! the identical train rows, score the identical test rows, and require gam's
//! held-out C to be at least flexsurv's minus a small margin. We still compute
//! flexsurv's fitted covariate slope and print its closeness for context with
//! eprintln!, but "close to flexsurv's fit" is no longer a pass/fail criterion.
//!
//! Why flexsurv is the right baseline: `flexsurvspline(..., scale = "hazard")` *is*
//! the textbook Royston-Parmar model — log Λ(t | covariates) as a restricted-cubic
//! spline in log t plus proportional linear covariate effects — exactly the
//! estimand gam's transformation family targets. `survmodel(spec=net)` selects
//! gam's net-survival Royston-Parmar working model (`SurvivalSpec::Net`). Both
//! engines therefore fit the same PH-on-log-Λ structure and produce a comparable
//! relative-risk score for concordance.
//!
//! Comparable wiggliness: gam fits a cubic (degree-3) monotone I-spline time basis
//! with two interior knots (`k = 2`, the minimum its degree-5 knot validation
//! admits), and the flexsurv reference fits the SAME two interior knots
//! (`FLEXSURV_REF_KNOTS == N_INTERNAL_KNOTS`). Both arms now run on large real
//! cohorts (heart-failure n=299, veteran n=137) where a `k = 2` `flexsurvspline`
//! is well-determined and fits stably, so the comparison is MATCHED-complexity
//! (k = 2 vs k = 2): both engines fit the SAME flexible PH-on-log-Λ structure plus
//! the SAME linear covariates and are scored on the SAME held-out concordance
//! metric. (This replaces the original 23-row `bone.csv` arm, which was too small
//! for a meaningful match-or-beat — issue #725: the RP spline was underdetermined
//! at k = 2 on ~12 train rows, forcing a k = 2 vs k = 1 mismatch and a held-out
//! concordance indistinguishable from chance on ~11 test rows.)
//!
//! Data — FIRST arm: `heart_failure_clinical_records_dataset.csv` — the UCI
//! heart-failure clinical-records cohort (299 subjects; Chicco & Jurman, BMC Med
//! Inform Decis Mak 2020). `time` = follow-up days, `DEATH_EVENT` = death
//! indicator (1 = died, 0 = right-censored), `ejection_fraction` = left-ventricular
//! ejection fraction (%), a strong, well-established continuous prognostic factor
//! (lower EF ⇒ higher mortality). SECOND arm: `veteran_lung.csv` (137 subjects).
//! Both splits are fixed (even original-row index ⇒ train, odd ⇒ test) so gam and
//! flexsurv see byte-identical train and test subsets.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

const HEART_FAILURE_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

// Two interior knots (k = 2) plus the two boundary knots flexsurv always places
// (flexsurv's own `df = k + 1`, i.e. df = 3). gam's cubic (degree-3) I-spline gets
// the same two interior knots so the baselines have matched flexibility.
//
// Why 2 and not 1: a gam degree-3 I-spline is the integral of a degree-4 M-spline,
// and its knot validation runs at degree-5 (the I-spline raises the underlying
// B-spline degree once for the M-spline and once more for the antiderivative).
// A well-posed open knot vector at validation degree 5 needs 2*(5+1) = 12 knots;
// the automatic placement emits `2*(4+1) + k = 10 + k` knots, so k must be >= 2.
// k = 1 yields only 11 knots and the fit is rejected as under-knotted. k = 2 is the
// minimum admissible interior-knot count for this cubic monotone baseline, and the
// flexsurv comparator below reads the same constant so the two stay matched.
const N_INTERNAL_KNOTS: usize = 2;
const TIME_DEGREE: usize = 3;

// Interior-knot count for the flexsurv REFERENCE Royston-Parmar baseline. Both
// arms of this file now fit large real survival cohorts (heart-failure n=299,
// veteran n=137), so the reference uses the SAME two interior knots as gam
// (`N_INTERNAL_KNOTS`). On these cohorts a k = 2 `flexsurvspline` (df = 3) is
// well-determined and fits stably, so the two engines compare MATCHED model
// complexity (gam k = 2 vs flexsurv k = 2) rather than the historical k = 2 vs
// k = 1 mismatch that the tiny 23-row bone split forced. Matched knots make the
// held-out concordance match-or-beat an honest like-for-like comparison.
const FLEXSURV_REF_KNOTS: usize = N_INTERNAL_KNOTS;

/// Load `heart_failure_clinical_records_dataset.csv` into
/// `(time, event, ejection_fraction)` rows: `time` = follow-up days,
/// `event` = `DEATH_EVENT` (1 = died, 0 = right-censored), `ef` = left-ventricular
/// ejection fraction (%), the continuous prognostic covariate.
fn load_heart_failure() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw = load_csvwith_inferred_schema(Path::new(HEART_FAILURE_CSV))
        .expect("load heart_failure_clinical_records_dataset.csv");
    let col = raw.column_map();
    let time = raw.values.column(col["time"]).to_vec();
    let event = raw.values.column(col["DEATH_EVENT"]).to_vec();
    let ef = raw.values.column(col["ejection_fraction"]).to_vec();
    (time, event, ef)
}

/// Harrell's concordance (C-index) for right-censored data.
///
/// `risk[i]` is a relative-risk score that increases with hazard (higher risk ⇒
/// expected shorter survival). A pair (i, j) is *comparable* when the one with the
/// shorter observed time experienced an event (so its survival is known to be
/// strictly shorter); a tie in observed time is not orderable and is skipped. The
/// pair is *concordant* when the subject with the shorter survival has the higher
/// risk score. Ties in risk score count as half-concordant. Returns
/// concordant_weight / comparable_pairs; with no comparable pairs returns NaN.
fn concordance(times: &[f64], events: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(times.len(), events.len(), "concordance length mismatch");
    assert_eq!(times.len(), risk.len(), "concordance length mismatch");
    let n = times.len();
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Identify which subject of the pair has the shorter observed time;
            // the pair is orderable only if that earlier subject had an event.
            let (early, late) = if times[i] < times[j] {
                (i, j)
            } else if times[j] < times[i] {
                (j, i)
            } else {
                // Equal observed time: survival ordering undetermined, skip.
                continue;
            };
            if events[early] != 1.0 {
                // Earlier subject was censored: we cannot assert it outlived the
                // later one, so the pair is not comparable.
                continue;
            }
            comparable += 1.0;
            // The earlier-event subject has the SHORTER survival, so a correct
            // ranker assigns it the HIGHER risk score.
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if risk[early] == risk[late] {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        f64::NAN
    } else {
        concordant / comparable
    }
}

#[test]
fn gam_rp_baseline_holdout_concordance_matches_or_beats_flexsurvspline_on_heart_failure() {
    init_parallelism();

    // ---- load the real heart-failure clinical-records survival data ---------
    let (time, event, ef) = load_heart_failure();
    let n = time.len();
    assert!(n > 250, "heart-failure should have ~299 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events >= 80,
        "heart-failure should carry many deaths, got {n_events}"
    );
    // Ejection fraction is a genuine continuous prognostic score spanning a wide
    // range (clinically ~14-80%); lower EF ⇒ higher mortality.
    let ef_min = ef.iter().cloned().fold(f64::INFINITY, f64::min);
    let ef_max = ef.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        ef_max - ef_min > 40.0,
        "ejection fraction should span a wide range: [{ef_min}, {ef_max}]"
    );

    // Standardize EF so the spline-time and covariate blocks are on a comparable
    // numeric scale (a monotone affine reparametrization; it leaves the PH
    // ordering, and hence concordance, unchanged for both engines). The identical
    // standardized values are written to gam and to flexsurv.
    let ef_mean = ef.iter().sum::<f64>() / n as f64;
    let ef_sd = {
        let v = ef
            .iter()
            .map(|e| (e - ef_mean) * (e - ef_mean))
            .sum::<f64>()
            / (n as f64 - 1.0);
        v.sqrt().max(1e-12)
    };
    let ef_z: Vec<f64> = ef.iter().map(|e| (e - ef_mean) / ef_sd).collect();

    // Deterministic split: even original-row index -> train, odd -> test. Fixed,
    // reproducible, and applied identically before either engine sees the data.
    let train_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let test_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    assert!(
        train_idx.len() >= 100 && test_idx.len() >= 100,
        "split too small: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    // The held-out set must contain enough events for an informative concordance.
    let test_events: usize = test_idx.iter().filter(|&&i| event[i] == 1.0).count();
    assert!(
        test_events >= 30,
        "held-out set needs events to score concordance, got {test_events}"
    );

    let pick = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let (train_time, train_event, train_ef) = (
        pick(&time, &train_idx),
        pick(&event, &train_idx),
        pick(&ef_z, &train_idx),
    );
    let (test_time, test_event, test_ef) = (
        pick(&time, &test_idx),
        pick(&event, &test_idx),
        pick(&ef_z, &test_idx),
    );
    let n_train = train_idx.len();

    // ---- encode the numeric TRAIN survival frame for gam --------------------
    let headers = ["t", "event", "ef"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n_train)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", train_time[i]),
                format!("{:.1}", train_event[i]),
                format!("{:.17e}", train_ef[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode heart-failure train frame");

    // ---- fit gam on TRAIN: Royston-Parmar net-survival baseline + linear cov -
    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ ef + survmodel(spec=net)", &ds, &cfg)
        .expect("gam RP-baseline net-survival fit on heart-failure");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    // beta = [β_time | β_cov]; the I-spline time block is a strict prefix.
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // Proportional-hazards covariate linear predictor η = c(ef)·β_cov, rebuilt from
    // the frozen spec so column order/basis match β_cov exactly. Under PH the
    // baseline log-Λ is shared across subjects, so η alone is a sufficient
    // relative-risk score for ranking survival (higher η ⇒ higher hazard ⇒ shorter
    // survival). This is the held-out prediction gam itself makes.
    let ef_idx = ds.column_map()["ef"];
    let cov_eta = |ef_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, ef_idx]] = ef_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    // Held-out risk score on every TEST subject.
    let gam_risk: Vec<f64> = (0..test_idx.len()).map(|i| cov_eta(test_ef[i])).collect();
    let gam_c = concordance(&test_time, &test_event, &gam_risk);
    assert!(
        gam_c.is_finite(),
        "gam held-out concordance is undefined (no comparable pairs)"
    );

    // For context only (NOT a pass criterion): gam's fitted EF slope, the finite
    // difference of η in ef (linear ⇒ exact). Higher EF (healthier) should LOWER
    // hazard, so we expect β_ef < 0.
    let gam_beta_ef = cov_eta(1.0) - cov_eta(0.0);

    // ---- fit the SAME model with flexsurv on the SAME TRAIN rows, score TEST --
    // flexsurvspline(scale="hazard") is the textbook Royston-Parmar model. The
    // reference uses k = FLEXSURV_REF_KNOTS = N_INTERNAL_KNOTS = 2 interior knots,
    // MATCHING gam's basis: on this ~150-row train split a k = 2 RP spline is
    // well-determined and fits stably, so both engines compare like-for-like model
    // complexity. We fit on the train subset and emit, for each held-out test
    // subject, the log-cumulative-hazard at a fixed reference time as its PH
    // relative-risk score (under PH the ordering of log Λ across subjects is the
    // ordering of the linear predictor, at any fixed t).
    let train_flag: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let ref_time = {
        // A reference time inside the observed range; the median observed time.
        let mut sorted = time.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
        sorted[sorted.len() / 2]
    };
    let r = run_r(
        &[
            Column::new("t", &time),
            Column::new("event", &event),
            Column::new("ef", &ef_z),
            Column::new("train", &train_flag),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            tr <- df[df$train == 1, ]
            te <- df[df$train == 0, ]
            m <- flexsurvspline(Surv(t, event) ~ ef, data = tr,
                                k = {k}, scale = "hazard")
            # Fitted continuous-covariate slope (context only).
            emit("beta_ef", as.numeric(coef(m)["ef"]))
            # PH relative-risk score per held-out subject: log Lambda at a fixed
            # reference time. summary(type="cumhaz") returns Lambda(ref | newdata).
            nd <- data.frame(ef = te$ef)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = c({ref_time}), ci = FALSE)
            risk <- sapply(ch, function(s) log(s$est[1]))
            emit("risk", as.numeric(risk))
            "#,
            k = FLEXSURV_REF_KNOTS,
            ref_time = format!("{ref_time:.10e}"),
        ),
    );
    let flex_beta_ef = r.scalar("beta_ef");
    let flex_risk = r.vector("risk");
    assert_eq!(
        flex_risk.len(),
        test_idx.len(),
        "flexsurv emitted {} held-out risk scores, expected {}",
        flex_risk.len(),
        test_idx.len()
    );
    let flex_c = concordance(&test_time, &test_event, flex_risk);
    assert!(
        flex_c.is_finite(),
        "flexsurv baseline concordance is undefined (no comparable pairs)"
    );

    // Closeness of the fitted slopes, printed for context but NOT asserted.
    let rel_beta_ef = relative_l2(&[gam_beta_ef], &[flex_beta_ef]);
    eprintln!(
        "heart-failure RP-baseline held-out concordance: n={n} (train={n_train} test={} test_events={test_events}) \
         gam_C={gam_c:.4} flex_C={flex_c:.4} | context: gam_beta_ef={gam_beta_ef:.4} \
         flex_beta_ef={flex_beta_ef:.4} rel_l2(beta_ef)={rel_beta_ef:.4}",
        test_idx.len()
    );

    // ---- OBJECTIVE assertion 1 (PRIMARY): gam ranks unseen survival ----------
    // Ejection fraction is a strong, well-established prognostic factor in heart
    // failure, and the held-out set is large (~149 subjects, >=30 events), so a
    // competent ranker clears a bar comfortably above chance. C >= 0.58 is a
    // genuine signal on real data, far from the 0.5 coin flip and statistically
    // informative at this sample size (unlike the original 23-row bone arm).
    assert!(
        gam_c >= 0.58,
        "gam's held-out concordance {gam_c:.4} is no better than chance — the fitted \
         Royston-Parmar model does not rank unseen heart-failure survival ordering"
    );

    // ---- OBJECTIVE assertion 2 (BASELINE-TO-BEAT): match-or-beat flexsurv -----
    // Same metric, same train rows, same held-out subjects, MATCHED knots
    // (k = 2 vs k = 2): gam's ranking quality must be at least the mature
    // reference's, up to a small tolerance for residual basis differences.
    assert!(
        gam_c >= flex_c - 0.03,
        "gam's held-out concordance {gam_c:.4} trails flexsurv {flex_c:.4} by more than \
         the basis-difference margin (0.03): gam ranks unseen survival worse than the reference"
    );
}

// ===========================================================================
// SECOND COHORT ARM (veteran lung cancer)
// ===========================================================================
//
// Same gam capability (the Royston-Parmar net-survival flexible-parametric
// baseline), exercised on a second, independent real clinical survival cohort
// where the truth is unknown. Because there is no ground-truth function to
// recover, the OBJECTIVE metric is again held-out Harrell's concordance: fit
// gam's RP model on the TRAIN rows, score the identical held-out TEST rows, and
// require (PRIMARY) gam's held-out C to clear a bar strictly above chance, and
// (BASELINE-TO-BEAT) gam's held-out C to be at least flexsurvspline's minus a
// small basis-difference margin on that SAME metric.
//
// Data SOURCE: `veteran_lung.csv` — the Veterans' Administration lung-cancer
// randomized trial (137 subjects), the canonical `veteran` dataset shipped with
// R's `survival` package (Kalbfleisch & Prentice, "The Statistical Analysis of
// Failure Time Data"). Columns used: `time` = survival time in days,
// `status` = death indicator (1 = died, 0 = right-censored), `karno` =
// Karnofsky performance score (0-100; higher = healthier, a strong continuous
// prognostic factor). The PH covariate is the continuous `karno` score, so the
// fitted model has the same linear-PH-on-log-Λ structure as the heart-failure
// arm above, on a second independent cohort. Split is fixed (even original-row
// index ⇒ train, odd ⇒ test) so gam and flexsurv see byte-identical train and
// test subsets.

const VETERAN_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/veteran_lung.csv"
);

#[test]
fn gam_rp_baseline_holdout_concordance_matches_or_beats_flexsurvspline_on_veteran() {
    init_parallelism();

    // ---- load the real veteran lung-cancer survival data --------------------
    let raw = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let col = raw.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let time: Vec<f64> = raw.values.column(time_idx).to_vec();
    let event: Vec<f64> = raw.values.column(status_idx).to_vec();
    let karno: Vec<f64> = raw.values.column(karno_idx).to_vec();
    let n = time.len();
    assert!(n > 120, "veteran should have ~137 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events >= 100,
        "veteran is mostly uncensored; expected many deaths, got {n_events}"
    );
    // karno is a genuine continuous prognostic score spanning a wide range.
    let karno_min = karno.iter().cloned().fold(f64::INFINITY, f64::min);
    let karno_max = karno.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        karno_max - karno_min > 50.0,
        "karno score should span a wide range: [{karno_min}, {karno_max}]"
    );

    // Standardize karno so the spline-time and covariate blocks are on a
    // comparable numeric scale (a monotone affine reparametrization; it leaves
    // the PH ordering, and hence concordance, unchanged for both engines). The
    // identical standardized values are written to gam and to flexsurv.
    let karno_mean = karno.iter().sum::<f64>() / n as f64;
    let karno_sd = {
        let v = karno
            .iter()
            .map(|k| (k - karno_mean) * (k - karno_mean))
            .sum::<f64>()
            / (n as f64 - 1.0);
        v.sqrt().max(1e-12)
    };
    let karno_z: Vec<f64> = karno.iter().map(|k| (k - karno_mean) / karno_sd).collect();

    // ---- deterministic split: even original-row index -> train, odd -> test -
    let train_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let test_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    assert!(
        train_idx.len() >= 50 && test_idx.len() >= 50,
        "split too small: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    let test_events: usize = test_idx.iter().filter(|&&i| event[i] == 1.0).count();
    assert!(
        test_events >= 20,
        "held-out set needs events to score concordance, got {test_events}"
    );

    let pick = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let (train_time, train_event, train_karno) = (
        pick(&time, &train_idx),
        pick(&event, &train_idx),
        pick(&karno_z, &train_idx),
    );
    let (test_time, test_event, test_karno) = (
        pick(&time, &test_idx),
        pick(&event, &test_idx),
        pick(&karno_z, &test_idx),
    );
    let n_train = train_idx.len();

    // ---- encode the numeric TRAIN survival frame for gam --------------------
    let headers = ["t", "event", "karno"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n_train)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", train_time[i]),
                format!("{:.1}", train_event[i]),
                format!("{:.17e}", train_karno[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode veteran train frame");

    // ---- fit gam on TRAIN: Royston-Parmar net-survival baseline + linear cov -
    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ karno + survmodel(spec=net)", &ds, &cfg)
        .expect("gam RP-baseline net-survival fit on veteran");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // PH covariate linear predictor η = c(karno)·β_cov rebuilt from the frozen
    // spec; under PH the baseline log-Λ is shared, so η alone is a sufficient
    // relative-risk score for ranking held-out survival.
    let karno_col = ds.column_map()["karno"];
    let cov_eta = |karno_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, karno_col]] = karno_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    let gam_risk: Vec<f64> = (0..test_idx.len())
        .map(|i| cov_eta(test_karno[i]))
        .collect();
    let gam_c = concordance(&test_time, &test_event, &gam_risk);
    assert!(
        gam_c.is_finite(),
        "gam held-out concordance is undefined (no comparable pairs)"
    );

    // Context only (NOT a pass criterion): gam's fitted karno slope. Higher
    // Karnofsky (healthier) should LOWER hazard, so we expect β_karno < 0.
    let gam_beta_karno = cov_eta(1.0) - cov_eta(0.0);

    // ---- fit the SAME model with flexsurv on the SAME TRAIN rows, score TEST --
    let train_flag: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let ref_time = {
        let mut sorted = time.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
        sorted[sorted.len() / 2]
    };
    let r = run_r(
        &[
            Column::new("t", &time),
            Column::new("event", &event),
            Column::new("karno", &karno_z),
            Column::new("train", &train_flag),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            tr <- df[df$train == 1, ]
            te <- df[df$train == 0, ]
            m <- flexsurvspline(Surv(t, event) ~ karno, data = tr,
                                k = {k}, scale = "hazard")
            emit("beta_karno", as.numeric(coef(m)["karno"]))
            nd <- data.frame(karno = te$karno)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = c({ref_time}), ci = FALSE)
            risk <- sapply(ch, function(s) log(s$est[1]))
            emit("risk", as.numeric(risk))
            "#,
            k = FLEXSURV_REF_KNOTS,
            ref_time = format!("{ref_time:.10e}"),
        ),
    );
    let flex_beta_karno = r.scalar("beta_karno");
    let flex_risk = r.vector("risk");
    assert_eq!(
        flex_risk.len(),
        test_idx.len(),
        "flexsurv emitted {} held-out risk scores, expected {}",
        flex_risk.len(),
        test_idx.len()
    );
    let flex_c = concordance(&test_time, &test_event, flex_risk);
    assert!(
        flex_c.is_finite(),
        "flexsurv baseline concordance is undefined (no comparable pairs)"
    );

    let rel_beta = relative_l2(&[gam_beta_karno], &[flex_beta_karno]);
    eprintln!(
        "veteran RP-baseline held-out concordance: n={n} (train={n_train} test={} test_events={test_events}) \
         gam_C={gam_c:.4} flex_C={flex_c:.4} | context: gam_beta_karno={gam_beta_karno:.4} \
         flex_beta_karno={flex_beta_karno:.4} rel_l2(beta_karno)={rel_beta:.4}",
        test_idx.len()
    );

    // ---- OBJECTIVE assertion 1 (PRIMARY): gam ranks unseen survival ----------
    // Karnofsky score is a strong, well-established prognostic factor in the
    // veteran trial, and the held-out set is large (~68 subjects, >=20 events),
    // so a competent ranker clears a bar comfortably above chance. C >= 0.58 is
    // a genuine signal on real data, far from the 0.5 coin flip.
    assert!(
        gam_c >= 0.58,
        "gam's held-out concordance {gam_c:.4} is no better than chance — the fitted \
         Royston-Parmar model does not rank unseen veteran survival ordering"
    );

    // ---- OBJECTIVE assertion 2 (BASELINE-TO-BEAT): match-or-beat flexsurv -----
    assert!(
        gam_c >= flex_c - 0.05,
        "gam's held-out concordance {gam_c:.4} trails flexsurv {flex_c:.4} by more than \
         the basis-difference margin (0.05): gam ranks unseen survival worse than the reference"
    );
}
