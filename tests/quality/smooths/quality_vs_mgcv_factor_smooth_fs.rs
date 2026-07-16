//! End-to-end OBJECTIVE quality: gam's factor-smooth term `s(x, group, bs="fs")`
//! must RECOVER THE KNOWN GENERATING FUNCTION, not merely echo mgcv's fit.
//!
//! The data are simulated from a known per-group truth
//!     y = μ_g + A·sin(4πx) + N(0, σ²),
//! i.e. one common sinusoidal shape shared across groups plus a distinct
//! per-group mean offset μ_g. The `fs` ("factor smooth") penalty is exactly the
//! right structure for this: a single shared marginal smooth in `x` replicated
//! per factor level, with a shared smoothing parameter and a null-space penalty
//! that shrinks the per-level offsets (the random-effect flavour of a smooth).
//!
//! OBJECTIVE METRIC (the pass/fail claim): on a dense (x-grid × group) lattice we
//! evaluate gam's fitted per-group curves and compare them to the NOISELESS truth
//! `μ_g + A·sin(4πx)`. We assert
//!   * RMSE(gam_fit, truth) <= 0.5·σ — recovering the mean function to well below
//!     the per-observation noise level is the defining property of a correct
//!     smoother (≈400 points / 4 groups give it ample data to average out σ);
//!   * each per-group fit is genuinely wiggly (its sample std over the x-grid is a
//!     sizeable fraction of the true amplitude A), so a degenerate flat/collapsed
//!     fit cannot pass.
//!
//! mgcv (`gam(..., bs="fs", method="REML")`) is fit on the IDENTICAL data and
//! kept ONLY as a match-or-beat ACCURACY BASELINE: gam's truth-recovery RMSE must
//! be no worse than 1.10× mgcv's truth-recovery RMSE. We are not asserting gam
//! reproduces mgcv's (itself noisy) fitted output — both are scored against the
//! same ground-truth function, and gam must recover it at least as well.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, QualityPair, held_out_r2, pad_to, relative_l2, rmse, run_r,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

const N: usize = 400;
const N_GROUPS: usize = 4;
const SEED: u64 = 33;
const SIGMA: f64 = 0.15;
/// Amplitude of the shared sinusoidal shape; per-group offsets sit on top.
const AMP: f64 = 1.0;
/// Distinct per-group mean offsets μ_g — these live in the marginal null space
/// that the fs penalty shrinks, so they exercise the null-space penalty path.
const MU: [f64; N_GROUPS] = [-0.8, -0.2, 0.4, 1.0];

/// Noiseless per-group truth `μ_g + A·sin(4πx)` — the function gam must recover.
fn truth(x: f64, g: usize) -> f64 {
    MU[g] + AMP * (4.0 * std::f64::consts::PI * x).sin()
}

fn build_data() -> (gam::data::EncodedDataset, Vec<f64>, Vec<String>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let mut x = Vec::with_capacity(N);
    let mut grp = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let xi = ux.sample(&mut rng);
        let g = i % N_GROUPS; // round-robin keeps all groups well-populated
        let yi = truth(xi, g) + noise.sample(&mut rng);
        x.push(xi);
        grp.push(g);
        y.push(yi);
    }

    // Group labels "g0".."g3"; since rows 0..3 introduce g0,g1,g2,g3 in order,
    // gam's categorical encoder assigns level codes 0,1,2,3 in that order, which
    // we mirror on the prediction grid below.
    let headers = vec!["x".to_string(), "group".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                x[i].to_string(),
                format!("g{}", grp[i]),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode factor-smooth data");

    // String columns handed to R (so mgcv sees the IDENTICAL factor).
    let group_labels: Vec<String> = grp.iter().map(|&g| format!("g{g}")).collect();
    (ds, y, group_labels)
}

#[test]
fn gam_factor_smooth_fs_recovers_truth() {
    init_parallelism();

    let (ds, y, group_labels) = build_data();
    let colmap = ds.column_map();
    let x_idx = colmap["x"];
    let group_idx = colmap["group"];
    let x_vals: Vec<f64> = ds.values.column(x_idx).to_vec();

    // ---- fit with gam: y ~ s(x, group, bs="fs"), REML, gaussian -----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, group, bs=\"fs\")", &ds, &cfg).expect("gam factor-smooth fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian factor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- shared evaluation lattice: dense x in [0,1] × each group ---------
    let n_x = 41usize;
    let x_grid: Vec<f64> = (0..n_x).map(|k| k as f64 / (n_x as f64 - 1.0)).collect();
    let n_eval = x_grid.len() * N_GROUPS;
    // gam side: build a design at the lattice. The group column carries the
    // categorical level CODE (0.0..3.0 as f64), matching the encoder above.
    let mut grid = Array2::<f64>::zeros((n_eval, ds.headers.len()));
    // R side: feed mgcv the IDENTICAL lattice as (x, group-label) pairs.
    let mut grid_x = Vec::with_capacity(n_eval);
    let mut grid_group = Vec::with_capacity(n_eval);
    // Noiseless truth at every lattice node — the objective target.
    let mut truth_curve = Vec::with_capacity(n_eval);
    let mut row = 0;
    for g in 0..N_GROUPS {
        for &xv in &x_grid {
            grid[[row, x_idx]] = xv;
            grid[[row, group_idx]] = g as f64;
            grid_x.push(xv);
            grid_group.push(format!("g{g}"));
            truth_curve.push(truth(xv, g));
            row += 1;
        }
    }
    assert_eq!(row, n_eval);

    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild factor-smooth design at lattice");
    let gam_curves: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_curves.len(), n_eval, "gam lattice prediction length");

    // ---- fit the SAME model with mgcv (match-or-beat accuracy baseline) ---
    // `group` must be a factor; bs="fs" with method="REML" reproduces the shared
    // smoothing parameter + null-space penalty. We predict the smooth on the
    // identical lattice; mgcv emits the lattice columns row-major in the order we
    // send them, so the returned `fitted` aligns elementwise with gam_curves and
    // with truth_curve. mgcv is scored against the SAME ground truth as gam.
    let r = run_r(
        &[
            Column::new("x", &x_vals),
            // group passed as the integer code; R reconstructs the same factor.
            Column::new(
                "group_code",
                &group_labels
                    .iter()
                    .map(|s| s[1..].parse::<f64>().expect("group code"))
                    .collect::<Vec<f64>>(),
            ),
            Column::new("y", &y),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            df$group <- factor(paste0("g", df$group_code), levels = c({levels}))
            m <- gam(y ~ s(x, group, bs = "fs"), data = df, method = "REML")
            gx <- c({gx})
            gg <- factor(c({gg}), levels = c({levels}))
            nd <- data.frame(x = gx, group = gg)
            emit("fitted", as.numeric(predict(m, newdata = nd)))
            emit("edf", sum(m$edf))
            "#,
            levels = (0..N_GROUPS)
                .map(|g| format!("\"g{g}\""))
                .collect::<Vec<_>>()
                .join(", "),
            gx = grid_x
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(", "),
            gg = grid_group
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    );
    let mgcv_curves = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_curves.len(),
        n_eval,
        "mgcv lattice prediction length mismatch"
    );

    // ---- OBJECTIVE truth-recovery scoring ---------------------------------
    let gam_truth_rmse = rmse(&gam_curves, &truth_curve);
    let mgcv_truth_rmse = rmse(mgcv_curves, &truth_curve);
    // Context only: how close the two fits happen to be (NOT a pass criterion).
    let rel_to_mgcv = relative_l2(&gam_curves, mgcv_curves);

    // Per-group diagnostics + a wiggliness guard: each fit must reproduce a real
    // sinusoid, not collapse to a flat line. The true per-group curve has sample
    // std AMP/√2 ≈ 0.707 over a uniform x-grid spanning two full periods.
    let mut min_group_std = f64::INFINITY;
    for g in 0..N_GROUPS {
        let lo = g * x_grid.len();
        let hi = lo + x_grid.len();
        let seg = &gam_curves[lo..hi];
        let mean = seg.iter().sum::<f64>() / seg.len() as f64;
        let var = seg.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / seg.len() as f64;
        let std = var.sqrt();
        min_group_std = min_group_std.min(std);
        let rg = rmse(seg, &truth_curve[lo..hi]);
        eprintln!("[fs] group g{g}: truth_rmse={rg:.4} fit_std={std:.4}");
    }

    eprintln!(
        "[fs] s(x,group,bs=fs): n={N} groups={N_GROUPS} \
         gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4} \
         (gam/mgcv={:.3}) min_group_std={min_group_std:.4} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} rel_l2_to_mgcv={rel_to_mgcv:.4} (context only)",
        gam_truth_rmse / mgcv_truth_rmse.max(1e-12)
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_factor_smooth_fs",
            "truth_rmse",
            gam_truth_rmse,
            "mgcv",
            mgcv_truth_rmse,
        )
        .line()
    );

    // PRIMARY CLAIM: gam recovers the noiseless generating function. Averaging
    // ~100 observations per group beats down σ; recovering the mean to half the
    // per-observation noise is the correct-smoother bar (a fit that merely
    // interpolated the noise would sit near σ, and an oversmoothed/flat fit would
    // miss the sinusoid by ~AMP/√2 ≫ σ).
    assert!(
        gam_truth_rmse <= 0.5 * SIGMA,
        "factor-smooth fit fails to recover the truth: rmse={gam_truth_rmse:.4} > {:.4}",
        0.5 * SIGMA
    );

    // STRUCTURE GUARD: every per-group fit must be genuinely wiggly — at least
    // half the true sinusoid's std (AMP/√2). Rejects a degenerate flat/collapsed
    // solution that could otherwise sneak under an RMSE bar via low variance.
    let min_required_std = 0.5 * AMP / std::f64::consts::SQRT_2;
    assert!(
        min_group_std >= min_required_std,
        "a factor-smooth group fit collapsed (too flat): min_group_std={min_group_std:.4} < {min_required_std:.4}"
    );

    // MATCH-OR-BEAT BASELINE: gam's truth-recovery accuracy must be no worse than
    // mgcv's by more than 10%. mgcv is the mature fs reference; scored on the same
    // ground truth, gam must be at least as accurate (within margin) — NOT merely
    // "close to mgcv's output".
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam less accurate than mgcv at recovering truth: gam_rmse={gam_truth_rmse:.4} > 1.10*mgcv_rmse={:.4}",
        mgcv_truth_rmse * 1.10
    );
}

// ============================================================================
// REAL-DATA ARM
// ============================================================================
//
// Dataset SOURCE: the classic `sleepstudy` data from the R `lme4` package
// (Belenky et al. 2003 sleep-deprivation study; 18 subjects, average reaction
// time on a psychomotor-vigilance task measured daily over 10 days of
// restricted sleep). Shipped here as `bench/datasets/sleepstudy.csv` with
// columns `Reaction` (ms), `Days` (0..9), `Subject` (id). Reaction time drifts
// upward with sleep deprivation but the slope and baseline differ subject to
// subject — exactly the per-group-curve structure the factor smooth
// `s(Days, Subject, bs="fs")` is built for: one shared marginal smooth in Days
// replicated per subject, with a shared smoothing parameter shrinking the
// per-subject deviations toward the population curve.
//
// Truth is unknown on real data, so quality is OUT-OF-SAMPLE predictive
// accuracy. We hold out two days per subject (Days 3 and 8, fixed indices),
// fit the per-subject smooth on the remaining 8 days/subject, predict the
// held-out days, and assert OBJECTIVE held-out metrics on gam's OWN
// predictions:
//
//   PRIMARY (objective, tool-free): held-out R^2 >= 0.55. The fs model must
//     explain held-out reaction-time variance well above the global-mean
//     predictor (R^2 = 0) — i.e. it genuinely recovers each subject's distinct
//     drift, not just the grand mean.
//
//   BASELINE (match-or-beat): mgcv fits the IDENTICAL train rows with
//     `gam(Reaction ~ s(Days, Subject, bs="fs"), method="REML")` and predicts
//     the IDENTICAL held-out rows; gam's held-out RMSE must be no worse than
//     mgcv_test_rmse * 1.10. mgcv is the mature baseline to match-or-beat, not
//     an output to replicate.

const SLEEPSTUDY_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/sleepstudy.csv");

#[test]
fn gam_factor_smooth_fs_recovers_truth_on_real_data() {
    init_parallelism();

    // ---- load the real sleepstudy dataset (Days, Subject -> Reaction) -----
    let ds = load_csvwith_inferred_schema(Path::new(SLEEPSTUDY_CSV)).expect("load sleepstudy.csv");
    let col = ds.column_map();
    let days_idx = col["Days"];
    let subject_idx = col["Subject"];
    let reaction_idx = col["Reaction"];
    // `Subject` parses as all-numeric, so CSV inference codes it as a continuous
    // column; the fs term needs it as a FACTOR. We read the raw values here and
    // re-encode Subject as a string label ("S<id>") below so mgcv and gam see
    // the IDENTICAL factor — the numbers themselves are the untouched real data.
    let days: Vec<f64> = ds.values.column(days_idx).to_vec();
    let subject_raw: Vec<f64> = ds.values.column(subject_idx).to_vec();
    let reaction: Vec<f64> = ds.values.column(reaction_idx).to_vec();
    let n = days.len();
    assert!(n > 150, "sleepstudy should have 180 rows, got {n}");

    let subject_label: Vec<String> = subject_raw
        .iter()
        .map(|&s| format!("S{}", s.round() as i64))
        .collect();

    // ---- deterministic train/test split: hold out Days 3 and 8 -------------
    // Every subject is measured on Days 0..9; holding out two interior days per
    // subject keeps each subject well-populated in train (8 days) while forcing
    // the model to PREDICT (interpolate) the missing days from the subject's own
    // curve plus the shared population trend.
    let is_test = |i: usize| {
        let d = days[i].round() as i64;
        d == 3 || d == 8
    };
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 130 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // ---- build the TRAIN dataset with Subject as a categorical factor ------
    let headers = vec![
        "Days".to_string(),
        "Subject".to_string(),
        "Reaction".to_string(),
    ];
    let train_records: Vec<StringRecord> = train_rows
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                days[i].to_string(),
                subject_label[i].clone(),
                reaction[i].to_string(),
            ])
        })
        .collect();
    let train_ds = encode_recordswith_inferred_schema(headers, train_records)
        .expect("encode sleepstudy train data");
    let train_col = train_ds.column_map();
    let t_days_idx = train_col["Days"];
    let t_subject_idx = train_col["Subject"];

    // Map each subject label to the integer level code the encoder assigned (by
    // first-appearance order in the train rows), so the prediction grid can put
    // the matching code in the Subject column for every held-out row.
    let mut subject_code: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        let code = train_ds.values[[out_row, t_subject_idx]];
        subject_code
            .entry(subject_label[src_row].clone())
            .or_insert(code);
    }

    // ---- fit gam on TRAIN: Reaction ~ s(Days, Subject, bs="fs"), REML ------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Reaction ~ s(Days, Subject, bs=\"fs\")", &train_ds, &cfg)
        .expect("gam sleepstudy factor-smooth fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian factor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- gam predictions at the held-out (Days, Subject) rows --------------
    let p = train_ds.headers.len();
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (out_row, &src_row) in test_rows.iter().enumerate() {
        let code = *subject_code
            .get(&subject_label[src_row])
            .expect("held-out subject seen in train");
        test_grid[[out_row, t_days_idx]] = days[src_row];
        test_grid[[out_row, t_subject_idx]] = code;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild factor-smooth design at held-out rows");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_test_pred.len(), test_rows.len(), "gam held-out length");

    let test_reaction: Vec<f64> = test_rows.iter().map(|&i| reaction[i]).collect();

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST ------
    // One run_r call exposes one data.frame, and every column must be equal
    // length, so we ride the (shorter) held-out columns along as right-padded
    // parallel columns and read back only their first `test_n` entries inside R.
    let train_days: Vec<f64> = train_rows.iter().map(|&i| days[i]).collect();
    let train_subject: Vec<f64> = train_rows.iter().map(|&i| subject_raw[i]).collect();
    let train_reaction: Vec<f64> = train_rows.iter().map(|&i| reaction[i]).collect();
    let test_days: Vec<f64> = test_rows.iter().map(|&i| days[i]).collect();
    let test_subject: Vec<f64> = test_rows.iter().map(|&i| subject_raw[i]).collect();
    let train_len = train_rows.len();

    let r = run_r(
        &[
            Column::new("Days", &train_days),
            Column::new("Subject", &train_subject),
            Column::new("Reaction", &train_reaction),
            Column::new("test_Days", &pad_to(&test_days, train_len)),
            Column::new("test_Subject", &pad_to(&test_subject, train_len)),
            Column::new("test_n", &vec![test_rows.len() as f64; train_len]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        df$Subject <- factor(df$Subject)
        # Holding out Days 3 and 8 leaves only 8 distinct Days in train, so the
        # fs marginal smooth's default basis dim (k = 10) exceeds the unique
        # covariate values and mgcv errors ("fewer unique covariate combinations
        # than specified maximum degrees of freedom"). Cap k at the 8 the split
        # supports; the shared per-subject Days curve is still a rich smooth.
        m <- gam(Reaction ~ s(Days, Subject, bs = "fs", k = 8), data = df, method = "REML")
        k <- df$test_n[1]
        newd <- data.frame(
          Days = df$test_Days[1:k],
          Subject = factor(df$test_Subject[1:k], levels = levels(df$Subject))
        )
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions ---------------
    let gam_test_r2 = held_out_r2(&gam_test_pred, &test_reaction);
    let gam_test_rmse = rmse(&gam_test_pred, &test_reaction);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_reaction);
    // Context only (NOT a pass criterion): closeness of the two held-out fits.
    let rel_to_mgcv = relative_l2(&gam_test_pred, mgcv_test_pred);

    eprintln!(
        "[fs-real] sleepstudy s(Days,Subject,bs=fs) held-out: n_train={train_len} \
         n_test={} gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         (context: rel_l2 vs mgcv={rel_to_mgcv:.4})",
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_factor_smooth_fs::test",
            "test_rmse",
            gam_test_rmse,
            "mgcv",
            mgcv_test_rmse,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out signal -----
    // Reaction time has a clear per-subject upward drift; recovering each
    // subject's curve explains well over half the held-out variance. R^2 >= 0.55
    // is far above the grand-mean baseline (0) and would catch a collapsed fit
    // that ignored the per-subject structure.
    assert!(
        gam_test_r2 >= 0.55,
        "gam's held-out predictive R2 too low on sleepstudy: {gam_test_r2:.4} (< 0.55)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out RMSE -----
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 1.0 && gam_edf < 120.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
