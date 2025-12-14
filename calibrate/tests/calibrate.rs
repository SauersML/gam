use gnomon::calibrate::calibrator::{
    CalibratorFeatures, CalibratorModel, CalibratorSpec, auc, build_calibrator_design,
    fit_calibrator, predict_calibrator,
};
use gnomon::calibrate::model::{
    BasisConfig, LinkFunction, calibrator_enabled, reset_calibrator_flag, set_calibrator_enabled,
};
use gnomon::calibrate::survival::{
    CholeskyFactor, HessianFactor, delta_method_standard_errors, survival_calibrator_features,
};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

fn auc_lower_confidence_bound(
    num_positive: usize,
    num_negative: usize,
    z_value: f64,
) -> f64 {
    assert!(num_positive > 0 && num_negative > 0);
    let n_pos = num_positive as f64;
    let n_neg = num_negative as f64;
    let variance = (n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg);
    let standard_error = variance.sqrt();
    let lower = 0.5 - z_value * standard_error;
    lower.max(0.0)
}

#[test]
fn calibration_toggle_round_trip() {
    reset_calibrator_flag();
    assert!(calibrator_enabled());

    set_calibrator_enabled(false);
    assert!(!calibrator_enabled());

    reset_calibrator_flag();
    assert!(calibrator_enabled());
}

fn make_survival_factor() -> HessianFactor {
    let lower = array![[2.0, 0.0], [0.5, (2.75_f64).sqrt()]];
    HessianFactor::Expected {
        factor: CholeskyFactor { lower },
    }
}

#[test]
fn logit_risk_calibration_improves_log_loss() {
    let predictions = array![0.12, 0.22, 0.55, 0.72, 0.81, 0.35];
    let outcomes = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let weights = Array1::ones(predictions.len());

    let design = Array2::from_shape_vec(
        (predictions.len(), 2),
        vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.2, 0.4, 0.6, 0.8, 0.3, 0.3, 0.7],
    )
    .unwrap();

    let factor = make_survival_factor();
    let se = delta_method_standard_errors(&factor, &design).unwrap();
    assert_eq!(se.len(), predictions.len());

    let features_matrix =
        survival_calibrator_features(&predictions, &design, Some(&factor)).unwrap();

    let features = CalibratorFeatures {
        pred: features_matrix.column(0).to_owned(),
        se: features_matrix.column(1).to_owned(),
        dist: Array1::zeros(predictions.len()),
        pred_identity: features_matrix.column(0).to_owned(),
        fisher_weights: weights.clone(),
    };

    let spec = CalibratorSpec {
        link: LinkFunction::Logit,
        pred_basis: BasisConfig {
            degree: 1,
            num_knots: 4,
        },
        se_basis: BasisConfig {
            degree: 1,
            num_knots: 4,
        },
        dist_basis: BasisConfig {
            degree: 1,
            num_knots: 4,
        },
        penalty_order_pred: 2,
        penalty_order_se: 2,
        penalty_order_dist: 2,
        distance_enabled: true,
        distance_hinge: false,
        prior_weights: Some(weights.clone()),
        firth: CalibratorSpec::firth_default_for_link(LinkFunction::Logit),
    };

    let (design_matrix, penalties, schema, offset) =
        build_calibrator_design(&features, &spec).expect("design build");
    assert!(design_matrix.ncols() > 0);

    let null_dims = penalties
        .iter()
        .zip(
            [
                schema.penalty_nullspace_dims.0,
                schema.penalty_nullspace_dims.1,
                schema.penalty_nullspace_dims.2,
                schema.penalty_nullspace_dims.3,
            ]
            .into_iter(),
        )
        .filter_map(|(penalty, dim)| {
            let max_abs = penalty
                .iter()
                .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
            (max_abs > 1e-12).then_some(dim)
        })
        .collect::<Vec<_>>();
    let (beta, lambdas, _, _, _) = fit_calibrator(
        outcomes.view(),
        weights.view(),
        design_matrix.view(),
        offset.view(),
        &penalties,
        &null_dims,
        LinkFunction::Logit,
        &spec,
    )
    .expect("fit calibrator");

    let mut saved_spec = spec.clone();
    saved_spec.prior_weights = None;

    let model = CalibratorModel {
        spec: saved_spec,
        knots_pred: schema.knots_pred,
        knots_se: schema.knots_se,
        knots_dist: schema.knots_dist,
        pred_constraint_transform: schema.pred_constraint_transform,
        stz_se: schema.stz_se,
        stz_dist: schema.stz_dist,
        penalty_nullspace_dims: schema.penalty_nullspace_dims,
        standardize_pred: schema.standardize_pred,
        standardize_se: schema.standardize_se,
        standardize_dist: schema.standardize_dist,
        interaction_center_pred: Some(schema.interaction_center_pred),
        se_wiggle_only_drop: schema.se_wiggle_only_drop,
        dist_wiggle_only_drop: schema.dist_wiggle_only_drop,
        lambda_pred: lambdas[0],
        lambda_pred_param: lambdas[1],
        lambda_se: lambdas[2],
        lambda_dist: lambdas[3],
        coefficients: beta,
        column_spans: schema.column_spans,
        pred_param_range: schema.pred_param_range,
        scale: None,
        assumes_frequency_weights: true,
    };

    let base_loss = outcomes
        .iter()
        .zip(predictions.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(1e-6, 1.0 - 1e-6);
            -((y * clipped.ln()) + ((1.0 - y) * (1.0 - clipped).ln()))
        })
        .sum::<f64>()
        / (predictions.len() as f64);

    let calibrated = predict_calibrator(
        &model,
        features.pred.view(),
        features.se.view(),
        features.dist.view(),
    )
    .expect("predict calibrated");

    let calibrated_loss = outcomes
        .iter()
        .zip(calibrated.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(1e-6, 1.0 - 1e-6);
            -((y * clipped.ln()) + ((1.0 - y) * (1.0 - clipped).ln()))
        })
        .sum::<f64>()
        / (predictions.len() as f64);

    assert!(calibrated_loss < base_loss);
}

struct SimulatedSample {
    phenotype: f64,
    score: f64,
    sex: f64,
}

fn simulate_cohort(
    rng: &mut StdRng,
    sex_value: f64,
    n_samples: usize,
    case_count: usize,
    score_effect: f64,
) -> Vec<SimulatedSample> {
    let mut scores = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let draw: f64 = rng.sample(StandardNormal);
        scores.push(draw);
    }

    let mean = scores.iter().sum::<f64>() / (n_samples as f64);
    for value in &mut scores {
        *value -= mean;
    }

    let mut liabilities = Vec::with_capacity(n_samples);
    for &score in &scores {
        let noise: f64 = rng.sample(StandardNormal);
        liabilities.push(score_effect * score + noise);
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.sort_unstable_by(|&a, &b| liabilities[b].total_cmp(&liabilities[a]));

    let mut phenotypes = vec![0.0; n_samples];
    for index in indices.into_iter().take(case_count) {
        phenotypes[index] = 1.0;
    }

    scores
        .into_iter()
        .zip(phenotypes.into_iter())
        .map(|(score, phenotype)| SimulatedSample {
            phenotype,
            score,
            sex: sex_value,
        })
        .collect()
}

fn write_training_file(path: &Path, samples: &[SimulatedSample]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "phenotype\tscore\tsex")?;
    for sample in samples {
        writeln!(
            writer,
            "{}\t{}\t{}",
            sample.phenotype, sample.score, sample.sex
        )?;
    }
    writer.flush()
}

fn write_prediction_file(path: &Path, samples: &[SimulatedSample]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "sample_id\tscore\tsex")?;
    for (index, sample) in samples.iter().enumerate() {
        writeln!(writer, "{}\t{}\t{}", index + 1, sample.score, sample.sex)?;
    }
    writer.flush()
}

fn extract_predictions(
    path: &Path,
    expected_len: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let mut lines = contents.lines();
    let header = lines
        .next()
        .ok_or_else(|| "prediction file missing header".to_string())?;
    let columns: Vec<&str> = header.split('\t').collect();
    let prediction_index = columns
        .iter()
        .position(|&column| column == "prediction")
        .ok_or_else(|| "prediction column missing from predictions.tsv".to_string())?;
    let preferred_index = columns
        .iter()
        .position(|&column| column == "uncalibrated_prediction")
        .unwrap_or(prediction_index);

    let mut predictions = Vec::with_capacity(expected_len);
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let target_index = if preferred_index < fields.len() {
            preferred_index
        } else {
            prediction_index
        };
        if target_index >= fields.len() {
            return Err(format!(
                "prediction column index {target_index} out of bounds for line: {line}"
            )
            .into());
        }
        let value: f64 = fields[target_index]
            .parse()
            .map_err(|e| format!("invalid prediction value '{}': {e}", fields[target_index]))?;
        predictions.push(value);
    }

    if predictions.len() != expected_len {
        return Err(format!(
            "expected {expected_len} predictions but found {}",
            predictions.len()
        )
        .into());
    }

    Ok(predictions)
}

fn cli_train_large_zero_pc_dataset_produces_model_impl() -> Result<(), Box<dyn std::error::Error>> {
    const TOTAL_ROWS: usize = 33_000;
    const MALE_ROWS: usize = 19_800;
    const FEMALE_ROWS: usize = 13_200;
    const MALE_CASES: usize = 206;
    const FEMALE_CASES: usize = 98;
    // Calculate exact target prevalence
    const TARGET_PREVALENCE: f64 = (MALE_CASES + FEMALE_CASES) as f64 / TOTAL_ROWS as f64;
    // Calculate exact ratio based on integer counts
    const MALE_TO_FEMALE_RATIO: f64 = ((MALE_CASES as f64) / (MALE_ROWS as f64)) / ((FEMALE_CASES as f64) / (FEMALE_ROWS as f64));
    const SCORE_EFFECT: f64 = 0.9;

    let mut rng = StdRng::seed_from_u64(7_316_511);

    let male_samples = simulate_cohort(&mut rng, 1.0, MALE_ROWS, MALE_CASES, SCORE_EFFECT);
    let female_samples = simulate_cohort(&mut rng, 0.0, FEMALE_ROWS, FEMALE_CASES, SCORE_EFFECT);

    assert_eq!(male_samples.len(), MALE_ROWS);
    assert_eq!(female_samples.len(), FEMALE_ROWS);

    let male_case_count = male_samples
        .iter()
        .filter(|sample| sample.phenotype == 1.0)
        .count();
    let female_case_count = female_samples
        .iter()
        .filter(|sample| sample.phenotype == 1.0)
        .count();
    assert_eq!(male_case_count, MALE_CASES);
    assert_eq!(female_case_count, FEMALE_CASES);

    let male_mean_score =
        male_samples.iter().map(|sample| sample.score).sum::<f64>() / (MALE_ROWS as f64);
    let female_mean_score = female_samples
        .iter()
        .map(|sample| sample.score)
        .sum::<f64>()
        / (FEMALE_ROWS as f64);
    assert!(
        (male_mean_score - female_mean_score).abs() < 1.0e-12,
        "sex-specific score means diverged: male={male_mean_score}, female={female_mean_score}"
    );

    let male_prevalence = (male_case_count as f64) / (MALE_ROWS as f64);
    let female_prevalence = (female_case_count as f64) / (FEMALE_ROWS as f64);
    let prevalence_ratio = male_prevalence / female_prevalence;
    assert!(
        (prevalence_ratio - MALE_TO_FEMALE_RATIO).abs() < 2.0e-3,
        "prevalence ratio deviates: {prevalence_ratio}"
    );
    let overall_prevalence = (male_case_count + female_case_count) as f64 / (TOTAL_ROWS as f64);
    assert!(
        (overall_prevalence - TARGET_PREVALENCE).abs() < 1.0e-12,
        "overall prevalence mismatch: {overall_prevalence}"
    );

    let tmp = tempdir()?;
    let training_path = tmp.path().join("large_train.tsv");

    let mut samples = male_samples;
    samples.extend(female_samples);
    assert_eq!(samples.len(), TOTAL_ROWS);
    write_training_file(training_path.as_path(), &samples)?;

    let exe = env!("CARGO_BIN_EXE_gnomon");
    let output = Command::new(exe)
        .current_dir(tmp.path())
        .args([
            "train",
            training_path
                .to_str()
                .expect("training path should be valid UTF-8"),
            "--num-pcs",
            "0",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "gnomon CLI failed: status={:?}\nstdout:{}\nstderr:{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let model_path = tmp.path().join("model.toml");
    assert!(model_path.exists(), "expected model.toml to be created");
    let metadata = fs::metadata(&model_path)?;
    assert!(metadata.len() > 0, "model.toml should not be empty");

    let prediction_input_path = tmp.path().join("large_predict.tsv");
    write_prediction_file(prediction_input_path.as_path(), &samples)?;

    let infer_output = Command::new(exe)
        .current_dir(tmp.path())
        .args([
            "infer",
            prediction_input_path
                .to_str()
                .expect("prediction path should be valid UTF-8"),
            "--model",
            model_path
                .to_str()
                .expect("model path should be valid UTF-8"),
        ])
        .output()?;

    assert!(
        infer_output.status.success(),
        "gnomon CLI inference failed: status={:?}\nstdout:{}\nstderr:{}",
        infer_output.status,
        String::from_utf8_lossy(&infer_output.stdout),
        String::from_utf8_lossy(&infer_output.stderr)
    );

    let predictions_path = tmp.path().join("predictions.tsv");
    assert!(
        predictions_path.exists(),
        "expected predictions.tsv to be created"
    );

    let predictions = extract_predictions(predictions_path.as_path(), TOTAL_ROWS)?;
    let observed = Array1::from_vec(samples.iter().map(|sample| sample.phenotype).collect());
    let predicted = Array1::from_vec(predictions);
    let auc_value = auc(&observed, &predicted);
    let auc_floor = auc_lower_confidence_bound(
        MALE_CASES + FEMALE_CASES,
        TOTAL_ROWS - (MALE_CASES + FEMALE_CASES),
        1.959963984540054,
    );
    assert!(
        auc_value > auc_floor,
        "expected inference AUC to exceed the 95% null bound ({auc_floor:.4}), got {auc_value}"
    );

    Ok(())
}

fn cli_train_small_zero_pc_dataset_produces_model_impl() -> Result<(), Box<dyn std::error::Error>> {
    const TOTAL_ROWS: usize = 3_300;
    const MALE_ROWS: usize = 1_980;
    const FEMALE_ROWS: usize = 1_320;
    const MALE_CASES: usize = 54;
    const FEMALE_CASES: usize = 26;
    const TARGET_PREVALENCE: f64 = 80.0 / 3_300.0;
    const MALE_TO_FEMALE_RATIO: f64 = (MALE_CASES as f64 / MALE_ROWS as f64)
        / (FEMALE_CASES as f64 / FEMALE_ROWS as f64);
    const SCORE_EFFECT: f64 = 1.4;

    let mut rng = StdRng::seed_from_u64(7_316_511);

    let male_samples = simulate_cohort(&mut rng, 1.0, MALE_ROWS, MALE_CASES, SCORE_EFFECT);
    let female_samples = simulate_cohort(&mut rng, 0.0, FEMALE_ROWS, FEMALE_CASES, SCORE_EFFECT);

    assert_eq!(male_samples.len(), MALE_ROWS);
    assert_eq!(female_samples.len(), FEMALE_ROWS);

    let male_case_count = male_samples
        .iter()
        .filter(|sample| sample.phenotype == 1.0)
        .count();
    let female_case_count = female_samples
        .iter()
        .filter(|sample| sample.phenotype == 1.0)
        .count();
    assert_eq!(male_case_count, MALE_CASES);
    assert_eq!(female_case_count, FEMALE_CASES);

    let male_mean_score =
        male_samples.iter().map(|sample| sample.score).sum::<f64>() / (MALE_ROWS as f64);
    let female_mean_score = female_samples
        .iter()
        .map(|sample| sample.score)
        .sum::<f64>()
        / (FEMALE_ROWS as f64);
    assert!(
        (male_mean_score - female_mean_score).abs() < 1.0e-12,
        "sex-specific score means diverged: male={male_mean_score}, female={female_mean_score}"
    );

    let male_prevalence = (male_case_count as f64) / (MALE_ROWS as f64);
    let female_prevalence = (female_case_count as f64) / (FEMALE_ROWS as f64);
    let prevalence_ratio = male_prevalence / female_prevalence;
    assert!(
        (prevalence_ratio - MALE_TO_FEMALE_RATIO).abs() < 2.0e-3,
        "prevalence ratio deviates: {prevalence_ratio}"
    );
    let overall_prevalence = (male_case_count + female_case_count) as f64 / (TOTAL_ROWS as f64);
    assert!(
        (overall_prevalence - TARGET_PREVALENCE).abs() < 1.0e-12,
        "overall prevalence mismatch: {overall_prevalence}"
    );

    let tmp = tempdir()?;
    let training_path = tmp.path().join("small_train.tsv");

    let mut samples = male_samples;
    samples.extend(female_samples);
    assert_eq!(samples.len(), TOTAL_ROWS);
    write_training_file(training_path.as_path(), &samples)?;

    let exe = env!("CARGO_BIN_EXE_gnomon");
    let output = Command::new(exe)
        .current_dir(tmp.path())
        .args([
            "train",
            training_path
                .to_str()
                .expect("training path should be valid UTF-8"),
            "--num-pcs",
            "0",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "gnomon CLI failed: status={:?}\nstdout:{}\nstderr:{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let model_path = tmp.path().join("model.toml");
    assert!(model_path.exists(), "expected model.toml to be created");
    let metadata = fs::metadata(&model_path)?;
    assert!(metadata.len() > 0, "model.toml should not be empty");

    let prediction_input_path = tmp.path().join("small_predict.tsv");
    write_prediction_file(prediction_input_path.as_path(), &samples)?;

    let infer_output = Command::new(exe)
        .current_dir(tmp.path())
        .args([
            "infer",
            prediction_input_path
                .to_str()
                .expect("prediction path should be valid UTF-8"),
            "--model",
            model_path
                .to_str()
                .expect("model path should be valid UTF-8"),
        ])
        .output()?;

    assert!(
        infer_output.status.success(),
        "gnomon CLI inference failed: status={:?}\nstdout:{}\nstderr:{}",
        infer_output.status,
        String::from_utf8_lossy(&infer_output.stdout),
        String::from_utf8_lossy(&infer_output.stderr)
    );

    let predictions_path = tmp.path().join("predictions.tsv");
    assert!(
        predictions_path.exists(),
        "expected predictions.tsv to be created"
    );

    let predictions = extract_predictions(predictions_path.as_path(), TOTAL_ROWS)?;
    let observed = Array1::from_vec(samples.iter().map(|sample| sample.phenotype).collect());
    let predicted = Array1::from_vec(predictions);
    let auc_value = auc(&observed, &predicted);
    let auc_floor = auc_lower_confidence_bound(
        MALE_CASES + FEMALE_CASES,
        TOTAL_ROWS - (MALE_CASES + FEMALE_CASES),
        1.959963984540054,
    );
    assert!(
        auc_value > auc_floor,
        "expected inference AUC to exceed the 95% null bound ({auc_floor:.4}), got {auc_value}"
    );

    Ok(())
}

mod calibrate {
    #[test]
    fn cli_train_large_zero_pc_dataset_produces_model(
    ) -> Result<(), Box<dyn std::error::Error>> {
        super::cli_train_large_zero_pc_dataset_produces_model_impl()
    }

    #[test]
    fn cli_train_small_zero_pc_dataset_produces_model(
    ) -> Result<(), Box<dyn std::error::Error>> {
        super::cli_train_small_zero_pc_dataset_produces_model_impl()
    }
}
