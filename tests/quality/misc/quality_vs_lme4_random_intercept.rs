//! End-to-end quality: gam's random-intercept term must RECOVER THE TRUTH on a
//! synthetic random-intercept design — not merely reproduce another tool's fit.
//!
//! The data is generated from a KNOWN structure:
//!
//!     y = 2π·sin(x) + μ_g + ε,   ε ~ N(0, σ_ε²=0.25),   μ_g = MU_G[g]
//!
//! so the ground-truth random-intercept deviations (the centred `MU_G`), the
//! true random-intercept variance σ_g² (the population sample-variance of the
//! centred `MU_G`), and the true residual variance σ_ε² = 0.25 are all known in
//! closed form. gam expresses the structure via `group(g)` / `s(g, bs="re")`:
//! the random intercept is a ridge-penalized factor block whose penalty is
//! selected by REML. The fitted gam model is
//!
//!     y ~ s(x) + group(g)
//!
//! exercising the cross-feature combination that matters: a penalized smooth
//! additive with a random intercept (penalized blocks side by side).
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, never "matches lme4"):
//!   1. ACCURACY of the per-group effects: RMSE(gam group deviations, centred
//!      true μ_g) is a small fraction of the group-effect signal range, AND
//!      gam's RMSE ≤ lme4's RMSE × 1.10 (match-or-beat lme4 on accuracy).
//!   2. RESIDUAL VARIANCE recovery: |σ̂_ε² − 0.25| small, AND gam at least as
//!      accurate as lme4 (its |error| ≤ lme4's |error| × 1.10).
//!   3. RANDOM-INTERCEPT VARIANCE recovery: σ̂_g² recovers the true population
//!      sample variance of the centred μ_g, AND gam ≤ lme4 error × 1.10.
//!
//! lme4 is fit on the IDENTICAL data and retained only as a BASELINE TO
//! MATCH-OR-BEAT on each accuracy metric — never as the pass criterion. The
//! primary claim is that gam recovers the data-generating truth. A genuine
//! shortfall here is a real bug in gam's random-effect machinery, not something
//! to paper over by loosening the bound.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pad_to, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

// Real-data benchmark dataset. SOURCE: the `sleepstudy` data from R's `lme4`
// package (Belenky et al. 2003 sleep-deprivation study; Reaction time in ms by
// Days of deprivation for 18 Subjects, 10 days each = 180 rows), exported to
// CSV at bench/datasets/sleepstudy.csv.
const SLEEPSTUDY_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/sleepstudy.csv");

const N_GROUPS: usize = 8;
const PER_GROUP: usize = 60;
// True group intercepts μ_g (population draws fixed by the spec). Their sample
// variance is the target σ_g² that lme4 estimates from the data.
const MU_G: [f64; N_GROUPS] = [-1.5, -0.9, -0.4, 0.0, 0.3, 0.7, 1.1, 1.5];
// Residual noise: ε ~ N(mean=0, variance=0.25) ⇒ standard deviation 0.5.
// rand_distr::Normal is parameterised by (mean, std_dev), so we pass 0.5; the
// residual *variance* σ_ε² we compare against is therefore 0.25.
const RESID_SD: f64 = 0.5;
const RESID_VAR: f64 = RESID_SD * RESID_SD;
const SEED: u64 = 42;

#[test]
fn gam_random_intercept_matches_lme4() {
    init_parallelism();

    // ---- synthesize the random-intercept dataset --------------------------
    // Rows are emitted group-blocked (all of g0, then g1, …). Group labels are
    // strings ("g0".."g7") so the schema inferrer treats `g` as categorical;
    // first-appearance order then makes the encoded level index equal the group
    // number, which we rely on when rebuilding the design at group anchors.
    let n = N_GROUPS * PER_GROUP;
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, RESID_SD).expect("normal");
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut x = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    let mut g_code = Vec::<f64>::with_capacity(n); // numeric group index for lme4
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for grp in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi: f64 = ux.sample(&mut rng);
            let yi = two_pi * xi.sin() + MU_G[grp] + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            g_code.push(grp as f64);
            rows.push(StringRecord::from(vec![
                format!("{xi}"),
                format!("g{grp}"),
                format!("{yi}"),
            ]));
        }
    }

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode RE dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    // ---- fit with gam: y ~ s(x) + group(g), REML --------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x) + group(g)", &ds, &cfg).expect("gam RE fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian random-intercept model");
    };
    // gam's residual standard deviation on the response scale (Gaussian identity
    // stores σ_ε here per the UnifiedFitResult contract).
    let gam_resid_var = fit.fit.standard_deviation * fit.fit.standard_deviation;

    // Predicted per-group intercept: evaluate the fitted model at a *single*
    // common x reference for every group. The s(x) contribution and the global
    // intercept are then identical across the 8 rows, so the row-to-row spread
    // isolates the estimated group effect (the gam BLUP). We use the mean x as
    // the reference (well inside the data support, no extrapolation).
    let x_ref = x.iter().sum::<f64>() / n as f64;
    let mut anchor = Array2::<f64>::zeros((N_GROUPS, ds.headers.len()));
    for grp in 0..N_GROUPS {
        anchor[[grp, x_idx]] = x_ref;
        // Encoded categorical level index equals the group number (see above).
        anchor[[grp, g_idx]] = grp as f64;
    }
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild design at group anchors");
    let gam_group_pred: Vec<f64> = anchor_design.design.apply(&fit.fit.beta).to_vec();
    // Centre to per-group *deviations* (mean-zero), matching lme4 conditional
    // modes which are deviations from the fixed-effect intercept.
    let gam_mean = gam_group_pred.iter().sum::<f64>() / N_GROUPS as f64;
    let gam_dev: Vec<f64> = gam_group_pred.iter().map(|v| v - gam_mean).collect();
    // Sample variance of the predicted group deviations is gam's estimate of
    // σ_g². With 60 observations/group and σ_ε²=0.25 the BLUP shrinkage factor
    // σ_g²/(σ_g²+σ_ε²/60) ≈ 0.99, so this is an essentially unbiased read of the
    // variance component and the apples-to-apples match for lme4's VarCorr.
    let gam_sigma_g2 = gam_dev.iter().map(|v| v * v).sum::<f64>() / (N_GROUPS as f64 - 1.0);

    // ---- fit the SAME model with lme4 (the mature reference) ---------------
    // lme4 separates fixed and random parts: the smooth main effect of x is a
    // fixed-effect natural spline (df matched to a typical s(x) edf), and the
    // random intercept is (1|g). VarCorr gives σ_g², sigma² gives σ_ε², and
    // ranef gives the per-group conditional modes in factor-level order g0..g7.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("g", &g_code),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(lme4))
        suppressPackageStartupMessages(library(splines))
        df$g <- factor(df$g, levels = as.character(sort(unique(df$g))))
        m <- lmer(y ~ ns(x, df = 6) + (1 | g), data = df, REML = TRUE)
        vc <- as.data.frame(VarCorr(m))
        sigma_g2 <- vc$vcov[vc$grp == "g"]
        sigma_e2 <- vc$vcov[vc$grp == "Residual"]
        re <- ranef(m)$g[, "(Intercept)"]
        emit("sigma_g2", sigma_g2)
        emit("sigma_e2", sigma_e2)
        emit("ranef", as.numeric(re))
        "#,
    );
    let lme4_sigma_g2 = r.scalar("sigma_g2");
    let lme4_sigma_e2 = r.scalar("sigma_e2");
    let lme4_ranef = r.vector("ranef");
    assert_eq!(
        lme4_ranef.len(),
        N_GROUPS,
        "lme4 returned {} conditional modes, expected {N_GROUPS}",
        lme4_ranef.len()
    );

    // ---- ground truth (closed form from the data-generating process) ------
    // True per-group deviations: the population intercepts MU_G centred to
    // mean-zero, matching the conditional-mode convention used for gam_dev and
    // lme4's ranef.
    let true_mu_mean = MU_G.iter().sum::<f64>() / N_GROUPS as f64;
    let true_dev: Vec<f64> = MU_G.iter().map(|m| m - true_mu_mean).collect();
    // True random-intercept variance in the same (n-1) sample-variance
    // convention used to compute gam_sigma_g2 from the predicted deviations.
    let true_sigma_g2 = true_dev.iter().map(|d| d * d).sum::<f64>() / (N_GROUPS as f64 - 1.0);
    // Signal range of the group effects, used as the scale for the accuracy bar.
    let dev_range = true_dev.iter().cloned().fold(f64::MIN, f64::max)
        - true_dev.iter().cloned().fold(f64::MAX, f64::min);

    // ---- accuracy vs truth (the OBJECTIVE metric) -------------------------
    // (1) Per-group effect accuracy: RMSE of the predicted group deviations
    // against the true centred μ_g. lme4's conditional modes are also scored
    // against the same truth, purely as a baseline to match-or-beat.
    let gam_dev_rmse = rmse(&gam_dev, &true_dev);
    let lme4_dev_rmse = rmse(lme4_ranef, &true_dev);

    // (2) Residual-variance accuracy: absolute error against the true σ_ε²=0.25.
    let gam_e_err = (gam_resid_var - RESID_VAR).abs();
    let lme4_e_err = (lme4_sigma_e2 - RESID_VAR).abs();

    // (3) Random-intercept-variance accuracy: absolute error against the true
    // population sample variance of the centred μ_g.
    let gam_g_err = (gam_sigma_g2 - true_sigma_g2).abs();
    let lme4_g_err = (lme4_sigma_g2 - true_sigma_g2).abs();

    eprintln!(
        "random-intercept truth recovery: n={n} groups={N_GROUPS} \
         dev_rmse gam={gam_dev_rmse:.4} lme4={lme4_dev_rmse:.4} (range {dev_range:.3}) | \
         sigma_e2 gam={gam_resid_var:.4} lme4={lme4_sigma_e2:.4} truth={RESID_VAR:.4} \
         (err gam={gam_e_err:.4} lme4={lme4_e_err:.4}) | \
         sigma_g2 gam={gam_sigma_g2:.4} lme4={lme4_sigma_g2:.4} truth={true_sigma_g2:.4} \
         (err gam={gam_g_err:.4} lme4={lme4_g_err:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_lme4_random_intercept::deviations",
            "dev_rmse",
            gam_dev_rmse,
            "lme4",
            lme4_dev_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_lme4_random_intercept::residual_variance",
            "abs_err_sigma_e2",
            gam_e_err,
            "lme4",
            lme4_e_err,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_lme4_random_intercept::intercept_variance",
            "abs_err_sigma_g2",
            gam_g_err,
            "lme4",
            lme4_g_err,
        )
        .line()
    );

    // (1) The 8 group effects span [-1.5, 1.5] (range ≈ 3.0) and sit far above
    // the per-group noise floor (σ_ε²/60 ≈ 0.004), so a faithful random-intercept
    // estimator pins them tightly. RMSE ≤ 10% of the signal range is the
    // principled accuracy bar; gam must additionally do no worse than lme4 ×1.10.
    assert!(
        gam_dev_rmse <= 0.10 * dev_range,
        "gam group effects miss the truth: rmse={gam_dev_rmse:.4} > {:.4} (10% of range {dev_range:.3})",
        0.10 * dev_range
    );
    assert!(
        gam_dev_rmse <= lme4_dev_rmse * 1.10,
        "gam group-effect accuracy worse than lme4 baseline: gam_rmse={gam_dev_rmse:.4} vs lme4_rmse={lme4_dev_rmse:.4}"
    );

    // (2) Residual variance is the best-determined component (≈480 residual
    // d.f.); a faithful REML fit nails σ_ε²=0.25 to within 10% absolute-relative
    // of the truth. gam must also match-or-beat lme4's accuracy.
    assert!(
        gam_e_err <= 0.10 * RESID_VAR,
        "gam residual variance misses truth 0.25: sigma_e2={gam_resid_var:.4} (err={gam_e_err:.4} > {:.4})",
        0.10 * RESID_VAR
    );
    assert!(
        gam_e_err <= lme4_e_err * 1.10 + 1e-6,
        "gam residual-variance accuracy worse than lme4 baseline: gam_err={gam_e_err:.4} vs lme4_err={lme4_e_err:.4}"
    );

    // (3) The random-intercept variance is estimated from only 8 groups and so
    // is the noisiest component, but with shrinkage ≈1% at 60 obs/group the
    // BLUP-derived estimate should recover the true population variance within
    // 25% of its value. gam must also do no worse than lme4 ×1.10.
    assert!(
        gam_g_err <= 0.25 * true_sigma_g2,
        "gam random-intercept variance misses truth: sigma_g2={gam_sigma_g2:.4} truth={true_sigma_g2:.4} (err={gam_g_err:.4} > {:.4})",
        0.25 * true_sigma_g2
    );
    assert!(
        gam_g_err <= lme4_g_err * 1.10 + 1e-6,
        "gam random-intercept-variance accuracy worse than lme4 baseline: gam_err={gam_g_err:.4} vs lme4_err={lme4_g_err:.4}"
    );
}

/// Real-data arm of the random-intercept quality test.
///
/// The synthetic test above proves truth-RECOVERY against a known data-
/// generating process. This arm exercises the SAME gam capability — a penalized
/// smooth additive with a per-group random intercept, `Reaction ~ s(Days) +
/// group(Subject)` — on REAL data (lme4's `sleepstudy`), where the truth is
/// unknown. With no known truth the only honest quality measure is OBJECTIVE
/// out-of-sample predictive accuracy, so we make a deterministic train/test
/// split, fit gam on the training rows, predict the held-out rows, and assert:
///
///   PRIMARY (objective, tool-free): held-out per-observation prediction RMSE
///     on Reaction is below an absolute bar far under the response spread —
///     proof the fitted subject intercepts + Days smooth genuinely transfer to
///     unseen rows.
///
///   BASELINE (match-or-beat): lme4 fits the IDENTICAL training rows
///     (`Reaction ~ Days + (1 | Subject)`) and predicts the IDENTICAL held-out
///     rows; gam's held-out RMSE must be no worse than `lme4_rmse * 1.10`. lme4
///     is a baseline to match-or-beat on accuracy, never an output to copy.
///
/// The sleepstudy rows are Subject-blocked with Days 0..9 repeating, so holding
/// out every 5th row (Days 0 and 5 of each subject) keeps every subject present
/// in the training data — the random intercept is estimable for every held-out
/// row — and the held-out Days sit inside the [0, 9] training support.
#[test]
fn gam_random_intercept_matches_lme4_on_real_data() {
    init_parallelism();

    // ---- load sleepstudy (Days, Subject -> Reaction) ----------------------
    let ds = load_csvwith_inferred_schema(Path::new(SLEEPSTUDY_CSV)).expect("load sleepstudy.csv");
    let col = ds.column_map();
    let days_idx = col["Days"];
    let subject_idx = col["Subject"];
    let reaction_idx = col["Reaction"];
    let days: Vec<f64> = ds.values.column(days_idx).to_vec();
    let subject: Vec<f64> = ds.values.column(subject_idx).to_vec();
    let reaction: Vec<f64> = ds.values.column(reaction_idx).to_vec();
    let n = reaction.len();
    assert_eq!(n, 180, "sleepstudy should have 180 rows, got {n}");

    // ---- deterministic train/test split: every 5th row held out ----------
    // Rows are subject-blocked with Days 0..9, so i % 5 == 0 holds out Days 0
    // and 5 of every subject (each subject keeps 8 training rows).
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert_eq!(train_rows.len(), 144, "expected 144 training rows");
    assert_eq!(test_rows.len(), 36, "expected 36 held-out rows");

    let train_days: Vec<f64> = train_rows.iter().map(|&i| days[i]).collect();
    let train_subject: Vec<f64> = train_rows.iter().map(|&i| subject[i]).collect();
    let train_reaction: Vec<f64> = train_rows.iter().map(|&i| reaction[i]).collect();
    let test_days: Vec<f64> = test_rows.iter().map(|&i| days[i]).collect();
    let test_subject: Vec<f64> = test_rows.iter().map(|&i| subject[i]).collect();
    let test_reaction: Vec<f64> = test_rows.iter().map(|&i| reaction[i]).collect();

    // Training-only dataset: sub-set the encoded rows. Headers, schema and
    // column kinds are unchanged so the formula resolves identically, and the
    // random-effect levels are frozen from the training Subject values (every
    // subject appears in training, so every held-out row maps to a known level).
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: Reaction ~ s(Days) + group(Subject), REML ------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Reaction ~ s(Days) + group(Subject)", &train_ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Gaussian sleepstudy model");
    };

    // gam predictions at the held-out (Days, Subject) pairs: rebuild the design
    // from the frozen spec (identity link => design*beta = predicted mean). Both
    // the Days smooth and the Subject random intercept are evaluated at the real
    // held-out covariate values, in held-out row order.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, days_idx]] = test_days[i];
        test_grid[[i, subject_idx]] = test_subject[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(
        gam_test_pred.len(),
        test_rows.len(),
        "gam held-out prediction length mismatch"
    );

    // ---- fit the SAME model on TRAIN with lme4, predict the SAME TEST -----
    // lme4 fits a linear Days fixed effect plus a Subject random intercept; the
    // held-out Days/Subject pairs are passed inside the SAME call as parallel
    // padded columns (one data.frame, every column equal length = train length)
    // and re-attached to a newdata frame of the true held-out length inside R.
    let k = test_rows.len();
    let train_len = train_rows.len();
    let r = run_r(
        &[
            Column::new("Days", &train_days),
            Column::new("Subject", &train_subject),
            Column::new("Reaction", &train_reaction),
            Column::new("test_Days", &pad_to(&test_days, train_len)),
            Column::new("test_Subject", &pad_to(&test_subject, train_len)),
            Column::new("test_n", &vec![k as f64; train_len]),
        ],
        r#"
        suppressPackageStartupMessages(library(lme4))
        df$Subject <- factor(df$Subject)
        m <- lmer(Reaction ~ Days + (1 | Subject), data = df, REML = TRUE)
        k <- df$test_n[1]
        newd <- data.frame(
          Days = df$test_Days[1:k],
          Subject = factor(df$test_Subject[1:k], levels = levels(df$Subject))
        )
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        "#,
    );
    let lme4_test_pred = r.vector("test_pred");
    assert_eq!(
        lme4_test_pred.len(),
        test_rows.len(),
        "lme4 held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions --------------
    let gam_test_rmse = rmse(&gam_test_pred, &test_reaction);
    let lme4_test_rmse = rmse(lme4_test_pred, &test_reaction);

    // Response spread, for context on the absolute bar.
    let resp_mean = test_reaction.iter().sum::<f64>() / k as f64;
    let resp_sd = (test_reaction
        .iter()
        .map(|y| (y - resp_mean) * (y - resp_mean))
        .sum::<f64>()
        / (k as f64 - 1.0))
        .sqrt();

    eprintln!(
        "sleepstudy random-intercept held-out: n_train={train_len} n_test={k} \
         gam_test_rmse={gam_test_rmse:.3} lme4_test_rmse={lme4_test_rmse:.3} \
         (held-out Reaction sd={resp_sd:.3})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_lme4_random_intercept::test",
            "test_rmse",
            gam_test_rmse,
            "lme4",
            lme4_test_rmse,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: gam predicts held-out Reaction ------
    // Reaction times in sleepstudy span ~190..460 ms with a held-out spread
    // (sd) near 50 ms. A model that captures the per-subject level and the Days
    // trend predicts held-out reaction times to well within ~40 ms RMSE — far
    // below the response spread and the naive constant-mean error. 40 ms is the
    // principled absolute bar; a genuine shortfall is a real bug in gam's
    // smooth+random-intercept machinery, not something to loosen.
    assert!(
        gam_test_rmse <= 40.0,
        "gam held-out Reaction RMSE too high: {gam_test_rmse:.3} ms (> 40 ms; held-out sd {resp_sd:.3})"
    );

    // ---- BASELINE (match-or-beat): no worse than lme4 on held-out RMSE ----
    assert!(
        gam_test_rmse <= lme4_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.3} exceeds lme4 {lme4_test_rmse:.3} * 1.10"
    );
}
