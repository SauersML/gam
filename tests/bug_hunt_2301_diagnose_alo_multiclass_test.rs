//! Regression for #2301: `gam diagnose --alo` covered only Standard-class
//! fits and rejected every other model class with a bare "unavailable in
//! this binary" message that named nothing and explained nothing.
//!
//! The fix generalized the saved-model ALO core
//! (`gam_predict::compute_saved_model_alo`) to dispatch location-scale
//! (Gaussian / binomial / dispersion), Bernoulli marginal-slope, and
//! transformation-normal and every survival likelihood through the shared
//! parameter-aligned, saved-Hessian ALO machinery, and `gam diagnose` now calls
//! that one dispatcher instead of duplicating a Standard-only replay.
//!
//! This test drives the real CLI (`gam fit` + `gam diagnose --alo`) on:
//! 1. a Gaussian location-scale fit (`--predict-noise`) — must now SUCCEED
//!    and print an ALO diagnostics table;
//! 2. a survival fit (`Surv(...) ~ x`) — must also succeed through its typed
//!    entry/exit/derivative replay and print the same ALO diagnostics contract.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::resolve_termspec_for_prediction;
use gam::gam_binary;
use gam::inference::model::FittedModel;
use gam::predict::input::build_predict_input_for_model;
use gam::predict::{
    PredictInput, SavedAloObservations, SavedModelAloDiagnostics, SavedModelAloInput,
    compute_saved_model_alo,
};
use gam::smooth::build_term_collection_design;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::{Array1, Array2};

fn write_csv(path: &Path, header: &[&str], rows: &[Vec<f64>]) {
    let mut writer = csv::Writer::from_path(path).expect("create csv");
    writer.write_record(header).expect("write header");
    for row in rows {
        let record: Vec<String> = row.iter().map(|value| format!("{value:.10}")).collect();
        writer.write_record(&record).expect("write row");
    }
    writer.flush().expect("flush csv");
}

#[test]
fn diagnose_alo_supports_location_scale_and_survival_2301() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // --- Part 1: Gaussian location-scale fit gets real ALO diagnostics. ---
    let mut rng_state: u64 = 0x2301_2301_2301_2301;
    let mut next = || {
        // Small xorshift so this test has no external RNG crate dependency.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 150;
    let mut ls_rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.0 + 4.0 * next();
        let noise_scale = (0.2 + 0.3 * x).exp();
        // Box-Muller for an approximately Gaussian residual.
        let u1 = next().max(1e-12);
        let u2 = next();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let y = 1.0 + 2.0 * x + noise_scale * z;
        ls_rows.push(vec![y, x]);
    }
    let ls_data = dir.path().join("ls.csv");
    let ls_model = dir.path().join("ls.gam");
    write_csv(&ls_data, &["y", "x"], &ls_rows);

    let mut fit_ls = Command::new(gam_binary!());
    fit_ls
        .arg("fit")
        .args(["--family", "gaussian"])
        .arg("--predict-noise")
        .arg("x")
        .arg("--out")
        .arg(&ls_model)
        .arg(&ls_data)
        .arg("y ~ x");
    run_or_panic(fit_ls, "gam fit gaussian location-scale (#2301)");
    assert!(ls_model.is_file(), "gam fit did not write {ls_model:?}");

    let mut diagnose_ls = Command::new(gam_binary!());
    diagnose_ls
        .arg("diagnose")
        .arg("--alo")
        .arg(&ls_model)
        .arg(&ls_data);
    let ls_output = diagnose_ls
        .output()
        .expect("spawn gam diagnose --alo (location-scale)");
    let ls_stdout = String::from_utf8_lossy(&ls_output.stdout);
    let ls_stderr = String::from_utf8_lossy(&ls_output.stderr);
    assert!(
        ls_output.status.success(),
        "diagnose --alo must now succeed for a location-scale fit (#2301):\n\
         --- stdout ---\n{ls_stdout}\n--- stderr ---\n{ls_stderr}"
    );
    assert!(
        ls_stdout.contains("ALO diagnostics"),
        "location-scale diagnose must print an ALO table: {ls_stdout}"
    );
    assert!(
        ls_stdout.contains("ALO coordinates"),
        "location-scale diagnose must name its multicoordinate frame: {ls_stdout}"
    );

    // --- Part 2: survival uses typed entry/exit/derivative row replay. ---
    let mut surv_rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -1.5 + 3.0 * next();
        let u = next().clamp(1e-9, 1.0 - 1e-12);
        let scale = (0.5 + 0.4 * x).exp();
        let t_latent = scale * (-u.ln());
        let censor = 4.0;
        let exit = t_latent.min(censor);
        let event = if t_latent <= censor { 1.0 } else { 0.0 };
        surv_rows.push(vec![0.0, exit, event, x]);
    }
    let surv_data = dir.path().join("surv.csv");
    let surv_model = dir.path().join("surv.gam");
    write_csv(&surv_data, &["entry", "exit", "event", "x"], &surv_rows);

    let mut fit_surv = Command::new(gam_binary!());
    fit_surv
        .arg("fit")
        .arg(&surv_data)
        .arg("Surv(entry, exit, event) ~ x")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&surv_model);
    run_or_panic(fit_surv, "gam fit Weibull survival (#2301)");
    assert!(surv_model.is_file(), "gam fit did not write {surv_model:?}");

    let mut diagnose_surv = Command::new(gam_binary!());
    diagnose_surv
        .arg("diagnose")
        .arg("--alo")
        .arg(&surv_model)
        .arg(&surv_data);
    let surv_output = diagnose_surv
        .output()
        .expect("spawn gam diagnose --alo (survival)");
    let surv_stdout = String::from_utf8_lossy(&surv_output.stdout);
    let surv_stderr = String::from_utf8_lossy(&surv_output.stderr);
    assert!(
        surv_output.status.success(),
        "diagnose --alo must succeed for a survival fit (#2301):\n\
         --- stdout ---\n{surv_stdout}\n--- stderr ---\n{surv_stderr}"
    );
    assert!(
        surv_stdout.contains("ALO diagnostics"),
        "survival diagnose must print an ALO table: {surv_stdout}"
    );
    assert!(
        surv_stdout.contains("ALO coordinates"),
        "survival diagnose must name its typed multicoordinate frame: {surv_stdout}"
    );
}

// ---------------------------------------------------------------------------
// Numerical-correctness arm for the marginal-slope and transformation-normal
// ALO paths (#2301). The two prints-only arms above never checked that the
// production ALO *numbers* are right for these classes. Here the oracle is an
// INDEPENDENT brute-force leave-one-out refit: for a held-out row i we refit
// the whole model on the remaining n-1 rows and evaluate the model's fitted
// leading local coordinate (marginal-eta / location-gamma) at row i. That
// exact-refit coordinate is what production ALO approximates in one Newton
// step at the full-data smoothing, so the two must agree.
//
// Both sides are driven through PRODUCTION code and stay in the identical
// coordinate frame by construction:
//   * production ALO      = `gam_predict::compute_saved_model_alo` over the
//     input built by the same `build_predict_input_for_model` the CLI uses;
//   * brute-force LOO      = refit via the `gam` CLI, then the leading
//     coordinate is `design(i) · beta[0..design.ncols()]`, where `design` is
//     the very coordinate-0 design that production ALO consumes.
// No reference tool and no replay of production values against themselves.

/// Encode in-memory numeric rows exactly as the fitted CSV was encoded, so the
/// ALO input matches the training data row-for-row.
fn encode_numeric_dataset(headers: &[&str], rows: &[Vec<f64>]) -> (Array2<f64>, HashMap<String, usize>) {
    let header_owned: Vec<String> = headers.iter().map(|name| (*name).to_string()).collect();
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|row| StringRecord::from(row.iter().map(|value| format!("{value:.10}")).collect::<Vec<_>>()))
        .collect();
    let dataset =
        encode_recordswith_inferred_schema(header_owned, records).expect("encode ALO dataset");
    (dataset.values, dataset.column_map())
}

/// Dense coordinate-0 design of `model` on `data` — the same covariate design
/// production ALO consumes for the leading local coordinate.
///
/// The frozen training spec is remapped to `data`'s column layout by name (the
/// same `resolve_termspec_for_prediction` the CLI predict/diagnose paths use),
/// so the reconstruction is column-order-independent.
fn leading_coordinate_design(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
) -> Array2<f64> {
    let spec = resolve_termspec_for_prediction(
        &model.payload().resolved_termspec,
        model.payload().training_headers.as_ref(),
        col_map,
        "resolved_termspec",
    )
    .expect("resolve frozen covariate termspec for the leading ALO coordinate");
    let design = build_term_collection_design(data.view(), &spec)
        .expect("build the covariate design for the leading ALO coordinate");
    design.design.to_dense()
}

/// Production saved-model ALO over the full fit.
///
/// Marginal-slope routes through the exact CLI input builder (which also
/// assembles the log-slope and latent-score channels). Transformation-normal
/// cannot: its predict input carries the response-scale conditional mean, not
/// the covariate design ALO needs, so we assemble the covariate-design affine
/// carrier here exactly as `gam diagnose --alo` does.
fn production_saved_alo(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    response: &Array1<f64>,
) -> SavedModelAloDiagnostics {
    let n = data.nrows();
    let weights = Array1::<f64>::ones(n);
    let training_headers = model.payload().training_headers.as_ref();
    let input = if model.predict_model_class().name() == "transformation-normal" {
        let spec = resolve_termspec_for_prediction(
            &model.payload().resolved_termspec,
            training_headers,
            col_map,
            "resolved_termspec",
        )
        .expect("resolve transformation-normal covariate termspec");
        let design = build_term_collection_design(data.view(), &spec)
            .expect("build transformation-normal covariate design");
        let base = Array1::<f64>::zeros(n);
        let offset = design
            .compose_offset(base.view(), "saved transformation-normal ALO design")
            .expect("compose transformation-normal ALO offset");
        SavedModelAloInput::affine(PredictInput {
            design: design.design,
            offset,
            design_noise: None,
            offset_noise: None,
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        })
    } else {
        let offset = Array1::<f64>::zeros(n);
        let offset_noise = Array1::<f64>::zeros(n);
        let predict_input = build_predict_input_for_model(
            model,
            data.view(),
            col_map,
            training_headers,
            &offset,
            &offset_noise,
            false,
        )
        .expect("build production saved-model ALO predict input");
        SavedModelAloInput::affine(predict_input)
    };
    compute_saved_model_alo(
        model,
        &input,
        SavedAloObservations {
            response,
            prior_weights: &weights,
        },
    )
    .expect("production saved-model ALO must succeed")
}

/// Exact fitted leading local coordinate of `model` evaluated at `row`.
///
/// For every affine saved class the first ALO coordinate is
/// `design · beta[0..design.ncols()]` (marginal-eta for marginal-slope,
/// location-gamma for transformation-normal). Applying this to a model refit
/// without `row` yields the exact leave-one-out coordinate — the brute-force
/// oracle that production ALO approximates.
fn fitted_leading_coordinate_at(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    row: usize,
) -> f64 {
    let dense = leading_coordinate_design(model, data, col_map);
    let leading_width = dense.ncols();
    let fit = model
        .payload()
        .fit_result
        .as_ref()
        .expect("refit carries a canonical fit result");
    let beta = &fit.beta;
    assert!(
        beta.len() >= leading_width,
        "refit beta ({}) shorter than coordinate-0 design width ({leading_width})",
        beta.len()
    );
    (0..leading_width).map(|column| dense[[row, column]] * beta[column]).sum()
}

/// The `leave_out` highest-leverage rows — where the ALO one-step correction is
/// most stressed and any leave-one-out inaccuracy would show up first.
fn highest_leverage_rows(diag: &SavedModelAloDiagnostics, leave_out: usize) -> Vec<usize> {
    let leverage = &diag.diagnostics.leverage;
    let mut order: Vec<usize> = (0..leverage.len()).collect();
    order.sort_by(|&left, &right| {
        leverage[right]
            .partial_cmp(&leverage[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(leave_out);
    order
}

/// Assert the production ALO leading coordinate tracks the brute-force refit
/// LOO coordinate across `fold_rows`, on the class named `class_label` with
/// leading coordinate `coordinate_name`.
fn assert_alo_matches_brute_force_loo(
    class_label: &str,
    coordinate_name: &str,
    diag: &SavedModelAloDiagnostics,
    fold_rows: &[usize],
    // (row, brute-force leave-one-out leading coordinate)
    brute_force: &[(usize, f64)],
) {
    assert_eq!(
        diag.model_class.name(),
        class_label,
        "production ALO reported the wrong model class"
    );
    assert_eq!(
        diag.coordinate_names[0], coordinate_name,
        "leading ALO coordinate must be the fit's primary linear predictor"
    );
    assert_eq!(brute_force.len(), fold_rows.len());

    let mut alo_values = Vec::with_capacity(fold_rows.len());
    let mut gaps = Vec::with_capacity(fold_rows.len());
    let mut standard_errors = Vec::with_capacity(fold_rows.len());
    for &(row, brute) in brute_force {
        let alo = diag.diagnostics.eta_tilde[row][0];
        let se = diag.diagnostics.alo_variance[row][0].sqrt();
        let gap = (alo - brute).abs();
        // Per-row bound: ALO is a first-order LOO approximation at the
        // full-data smoothing, so its deviation from an exact refit (which also
        // reselects the smoothing on n-1 rows) must sit within a few standard
        // errors of the coordinate's own posterior uncertainty; a small
        // absolute floor covers finite-precision fits and near-zero SEs. A
        // wrong coordinate frame or a logic bug would blow far past this — the
        // gap would be of order the coordinate magnitude, not a few sigma.
        assert!(
            gap <= f64::max(5.0e-2, 4.0 * se),
            "{class_label} row {row}: production ALO {coordinate_name}={alo:.6} vs brute-force \
             LOO refit {brute:.6} (|gap|={gap:.6}) exceeds max(5.0e-2, 4sigma={:.6})",
            4.0 * se
        );
        alo_values.push(alo);
        gaps.push(gap);
        standard_errors.push(se);
    }

    let alo_min = alo_values.iter().copied().fold(f64::INFINITY, f64::min);
    let alo_max = alo_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread = alo_max - alo_min;
    // Non-vacuity: the leading coordinate must genuinely vary across the
    // held-out rows, otherwise agreement would be trivial.
    assert!(
        spread > 5.0e-2,
        "{class_label}: leading ALO coordinate barely varies across held-out rows \
         (spread={spread:.6}); the numerical check would be vacuous"
    );
    let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
    // Aggregate: ALO must reproduce the exact leave-one-out coordinate to well
    // within a small fraction of its own dynamic range — it tracks the LOO
    // signal, not just its noise floor.
    assert!(
        mean_gap <= 0.25 * spread + 2.0e-3,
        "{class_label}: mean |ALO - brute-force LOO| = {mean_gap:.6} is not small relative to the \
         coordinate spread {spread:.6}; production ALO does not track exact leave-one-out"
    );
}

#[test]
fn diagnose_alo_marginal_slope_matches_brute_force_loo_2301() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // Deterministic Bernoulli marginal-slope data: a monotone disease risk in
    // x, plus a standardized latent score z the family calibrates its slope
    // against. Small n keeps the brute-force LOO refits fast.
    let mut rng_state: u64 = 0x2301_A105_BEEF_0001;
    let mut next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 80;
    let mut xs = Vec::with_capacity(n);
    let mut raw_z = Vec::with_capacity(n);
    let mut disease = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.2 + 4.4 * next();
        let z = {
            let u1 = next().max(1e-12);
            let u2 = next();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let p = 1.0 / (1.0 + (-(0.4 + 0.9 * x)).exp());
        let y = if next() < p { 1.0 } else { 0.0 };
        xs.push(x);
        raw_z.push(z);
        disease.push(y);
    }
    // Standardize the latent score, as the marginal-slope family expects.
    let z_mean = raw_z.iter().sum::<f64>() / n as f64;
    let z_sd = (raw_z.iter().map(|z| (z - z_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let zs: Vec<f64> = raw_z.iter().map(|z| (z - z_mean) / z_sd).collect();

    let headers = ["disease", "x", "z"];
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![disease[i], xs[i], zs[i]])
        .collect();

    let fit_formula = "disease ~ s(x)";
    let full_csv = dir.path().join("ms_full.csv");
    write_csv(&full_csv, &headers, &rows);
    let full_model = dir.path().join("ms_full.gam");
    let mut fit = Command::new(gam_binary!());
    fit.arg("fit")
        .arg(&full_csv)
        .arg(fit_formula)
        .args(["--z-column", "z"])
        .args(["--logslope-formula", "1"])
        .arg("--out")
        .arg(&full_model);
    run_or_panic(fit, "gam fit Bernoulli marginal-slope (#2301 ALO numeric)");
    assert!(full_model.is_file(), "gam fit did not write {full_model:?}");

    let model = FittedModel::load_from_path(&full_model).expect("load full marginal-slope model");
    let (data, col_map) = encode_numeric_dataset(&headers, &rows);
    let response = Array1::from_iter(disease.iter().copied());
    let diag = production_saved_alo(&model, &data, &col_map, &response);

    let fold_rows = highest_leverage_rows(&diag, 10);
    let mut brute_force = Vec::with_capacity(fold_rows.len());
    for &i in &fold_rows {
        let fold: Vec<Vec<f64>> = rows
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, row)| row.clone())
            .collect();
        let fold_csv = dir.path().join(format!("ms_fold_{i}.csv"));
        write_csv(&fold_csv, &headers, &fold);
        let refit_model = dir.path().join(format!("ms_refit_{i}.gam"));
        let mut refit = Command::new(gam_binary!());
        refit
            .arg("fit")
            .arg(&fold_csv)
            .arg(fit_formula)
            .args(["--z-column", "z"])
            .args(["--logslope-formula", "1"])
            .arg("--out")
            .arg(&refit_model);
        run_or_panic(refit, "gam refit marginal-slope leave-one-out fold (#2301)");
        let refit = FittedModel::load_from_path(&refit_model).expect("load marginal-slope refit");
        // Evaluate the refit's marginal-eta at the held-out row on the full
        // data (row i aligns with the full-data ALO row i).
        let brute = fitted_leading_coordinate_at(&refit, &data, &col_map, i);
        brute_force.push((i, brute));
    }

    assert_alo_matches_brute_force_loo(
        "bernoulli marginal-slope",
        "marginal-eta",
        &diag,
        &fold_rows,
        &brute_force,
    );
}

#[test]
fn diagnose_alo_transformation_normal_matches_brute_force_loo_2301() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // Deterministic bounded response with an x-dependent location so the
    // conditional transformation genuinely varies with the covariate.
    let mut rng_state: u64 = 0x2301_C7A0_FACE_0002;
    let mut next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 80;
    let mut rows = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = next();
        let noise = 0.16 * (next() - 0.5);
        let y = (0.20 + 0.55 * x + noise).clamp(0.02, 0.98);
        rows.push(vec![y, x]);
        ys.push(y);
    }

    let headers = ["y", "x"];
    let fit_formula = "y ~ s(x, k=6)";
    let full_csv = dir.path().join("ctn_full.csv");
    write_csv(&full_csv, &headers, &rows);
    let full_model = dir.path().join("ctn_full.gam");
    let mut fit = Command::new(gam_binary!());
    fit.arg("fit")
        .arg(&full_csv)
        .arg(fit_formula)
        .arg("--transformation-normal")
        .arg("--out")
        .arg(&full_model);
    run_or_panic(fit, "gam fit transformation-normal (#2301 ALO numeric)");
    assert!(full_model.is_file(), "gam fit did not write {full_model:?}");

    let model = FittedModel::load_from_path(&full_model).expect("load full transformation-normal");
    let (data, col_map) = encode_numeric_dataset(&headers, &rows);
    let response = Array1::from_iter(ys.iter().copied());
    let diag = production_saved_alo(&model, &data, &col_map, &response);

    let fold_rows = highest_leverage_rows(&diag, 10);
    let mut brute_force = Vec::with_capacity(fold_rows.len());
    for &i in &fold_rows {
        let fold: Vec<Vec<f64>> = rows
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, row)| row.clone())
            .collect();
        let fold_csv = dir.path().join(format!("ctn_fold_{i}.csv"));
        write_csv(&fold_csv, &headers, &fold);
        let refit_model = dir.path().join(format!("ctn_refit_{i}.gam"));
        let mut refit = Command::new(gam_binary!());
        refit
            .arg("fit")
            .arg(&fold_csv)
            .arg(fit_formula)
            .arg("--transformation-normal")
            .arg("--out")
            .arg(&refit_model);
        run_or_panic(refit, "gam refit transformation-normal leave-one-out fold (#2301)");
        let refit =
            FittedModel::load_from_path(&refit_model).expect("load transformation-normal refit");
        let brute = fitted_leading_coordinate_at(&refit, &data, &col_map, i);
        brute_force.push((i, brute));
    }

    assert_alo_matches_brute_force_loo(
        "transformation-normal",
        "location-gamma",
        &diag,
        &fold_rows,
        &brute_force,
    );
}
