//! Regression for the binary-outcome Bernoulli marginal-slope audit refusal.
//!
//! The production failure had the marginal formula
//!
//!   event ~ matern(PC1, PC2, PC3, centers=10)
//!       + sex + entry_age_z
//!       + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4
//!
//! with a Matérn log-slope surface over the same PCs. The pre-fit
//! identifiability audit reported `marginal_surface` rank 14/15 and attributed
//! the dropped local column to the scalar prefix (`local column 3`), before the
//! Matérn basis columns. In the real pipeline `current_age_ns_1` can be the
//! constant natural-spline basis column, duplicating the implicit intercept
//! inside the BMS marginal block.
//!
//! This test drives the full formula-to-fit path so the pre-fit audit runs; it
//! is not just a materialization check.

use gam::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::{FitConfig, FitResult, fit_from_formula};
use ndarray::Array2;

fn binary_outcome_shape_dataset() -> EncodedDataset {
    let n = 96usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (i as f64 - 47.5) / 18.0;
        let t = ((i % 19) as f64 - 9.0) / 9.0;
        let current_age_ns_1 = 1.0;
        let current_age_ns_2 = t;
        let current_age_ns_3 = t * t;
        let current_age_ns_4 = t * t * t;
        let prs_z = (((i * 37) % 101) as f64 - 50.0) / 22.0;
        let pc1 = ((i as f64) * 0.17).sin();
        let pc2 = ((i as f64) * 0.23).cos();
        let pc3 = ((i as f64) * 0.31).sin() * ((i as f64) * 0.07).cos();
        let eta = -0.15 + 0.25 * sex + 0.18 * entry_age_z + 0.16 * t + 0.10 * prs_z + 0.08 * pc1;
        let deterministic_noise = (((i * 13) % 17) as f64 - 8.0) / 11.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            current_age_ns_1,
            current_age_ns_2,
            current_age_ns_3,
            current_age_ns_4,
            prs_z,
            pc1,
            pc2,
            pc3,
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("binary-outcome-shape BMS data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "entry_age_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

fn duplicate_pc_binary_outcome_shape_dataset() -> EncodedDataset {
    let n = 160usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let pc_cloud = [
        [-1.0, -0.5, 0.25],
        [-0.25, 0.75, -0.5],
        [0.5, -0.75, 0.75],
        [1.0, 0.5, -0.25],
    ];
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (i as f64 - 79.5) / 30.0;
        let t = ((i % 29) as f64 - 14.0) / 14.0;
        let prs_z = (((i * 17) % 131) as f64 - 65.0) / 31.0;
        let pc = pc_cloud[i % pc_cloud.len()];
        let eta = -0.05 + 0.12 * sex + 0.08 * entry_age_z + 0.06 * t + 0.05 * prs_z + 0.04 * pc[0];
        let deterministic_noise = (((i * 11) % 23) as f64 - 11.0) / 10.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            t,
            t * t,
            t * t * t,
            t * t * t * t,
            prs_z,
            pc[0],
            pc[1],
            pc[2],
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("duplicate-PC binary-outcome-shape BMS data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "entry_age_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

fn production_like_pc_confound_dataset() -> EncodedDataset {
    let n = 384usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let phase = i as f64;
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (phase - 191.5) / 95.0;
        let t = ((i % 73) as f64 - 36.0) / 36.0;
        let current_age_ns_1 = t;
        let current_age_ns_2 = t * t;
        let current_age_ns_3 = t * t * t;
        let current_age_ns_4 = t * t * t * t;
        let pc1 = (phase * 0.071).sin() + 0.15 * (phase * 0.017).cos();
        let pc2 = (phase * 0.097).cos() + 0.10 * (phase * 0.023).sin();
        let pc3 = (phase * 0.043).sin() * (phase * 0.031).cos() + 0.05 * t;
        let prs_z = 0.92 * pc1 - 0.37 * pc2 + 0.18 * pc3 + 0.04 * ((i % 11) as f64 - 5.0);
        let eta = -0.22 + 0.12 * sex + 0.24 * entry_age_z + 0.18 * current_age_ns_1
            - 0.07 * current_age_ns_2
            + 0.11 * prs_z
            + 0.16 * pc1
            - 0.08 * pc2;
        let deterministic_noise = (((i * 29) % 41) as f64 - 20.0) / 12.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            current_age_ns_1,
            current_age_ns_2,
            current_age_ns_3,
            current_age_ns_4,
            prs_z,
            pc1,
            pc2,
            pc3,
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("production-like binary-outcome BMS data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "entry_age_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

#[test]
fn binary_outcome_shape_bms_matern_fit_is_not_refused_by_identifiability_audit() {
    gam::init_parallelism();
    let data = binary_outcome_shape_dataset();
    let cfg = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=4)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "event ~ matern(PC1, PC2, PC3, centers=4) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
        &data,
        &cfg,
    );
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            assert!(
                out.fit.beta.iter().all(|coef| coef.is_finite()),
                "binary-outcome-shape BMS fit should produce finite coefficients, got {:?}",
                out.fit.beta
            );
        }
        Ok(_) => panic!("binary-outcome-shape fit returned the wrong family variant"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("identifiability audit refused")
                    && !msg.contains("pre-fit identifiability audit"),
                "binary-outcome-shape BMS fit was still refused by the identifiability audit: {msg}"
            );
            panic!("binary-outcome-shape BMS fit failed after passing the audit: {msg}");
        }
    }
}

#[test]
fn binary_outcome_shape_bms_matern_centers60_are_rank_reduced() {
    gam::init_parallelism();
    let data = duplicate_pc_binary_outcome_shape_dataset();
    let cfg = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=60, length_scale=1.0)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "event ~ matern(PC1, PC2, PC3, centers=60, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
        &data,
        &cfg,
    );
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            assert!(
                out.fit.beta.iter().all(|coef| coef.is_finite()),
                "rank-reduced Matérn BMS fit should produce finite coefficients, got {:?}",
                out.fit.beta
            );
        }
        Ok(_) => panic!("rank-reduced Matérn fit returned the wrong family variant"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("identifiability audit refused")
                    && !msg.contains("joint rank")
                    && !msg.contains("dropped column"),
                "redundant Matérn centers should be rank-reduced before the joint audit: {msg}"
            );
            panic!("rank-reduced Matérn BMS fit failed after passing the audit: {msg}");
        }
    }
}

#[test]
fn binary_outcome_shape_bms_shared_matern_prs_pc_confound_starts_outer_solver() {
    gam::init_parallelism();
    let mut data = binary_outcome_shape_dataset();
    let prs_idx = data
        .headers
        .iter()
        .position(|h| h == "prs_z")
        .expect("prs_z column");
    let pc1_idx = data
        .headers
        .iter()
        .position(|h| h == "PC1")
        .expect("PC1 column");
    let pc2_idx = data
        .headers
        .iter()
        .position(|h| h == "PC2")
        .expect("PC2 column");
    for row in 0..data.values.nrows() {
        data.values[[row, prs_idx]] =
            data.values[[row, pc1_idx]] + 0.1 * data.values[[row, pc2_idx]];
    }

    let cfg = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=6, length_scale=1.0)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "event ~ matern(PC1, PC2, PC3, centers=6, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
        &data,
        &cfg,
    );
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            assert!(
                out.fit.beta.iter().all(|coef| coef.is_finite()),
                "PC-confounded Matérn BMS fit should produce finite coefficients, got {:?}",
                out.fit.beta
            );
        }
        Ok(_) => panic!("PC-confounded Matérn fit returned the wrong family variant"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("CertRefused")
                    && !msg.contains("phantom_multiplier")
                    && !msg.contains("outer smoothing optimization did not converge")
                    && !msg.contains("no candidate seeds passed outer startup validation"),
                "PC-confounded Matérn BMS fit still hit the #754 startup failure: {msg}"
            );
            panic!("PC-confounded Matérn BMS fit failed: {msg}");
        }
    }
}

#[test]
fn production_like_binary_outcome_shared_matern_centers10_confound_starts_outer_solver() {
    gam::init_parallelism();
    let data = production_like_pc_confound_dataset();
    let cfg = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=10, length_scale=1.0)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "event ~ matern(PC1, PC2, PC3, centers=10, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
        &data,
        &cfg,
    );
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            assert!(
                out.fit.beta.iter().all(|coef| coef.is_finite()),
                "production-like PC-confounded Matérn BMS fit should produce finite coefficients, got {:?}",
                out.fit.beta
            );
            assert!(
                out.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
                "production-like BMS fit must report finite smoothing parameters after the Jeffreys/orthogonalization cure; got log_lambdas={:?}",
                out.fit.log_lambdas
            );
        }
        Ok(_) => panic!("production-like Matérn fit returned the wrong family variant"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("identifiability audit refused")
                    && !msg.contains("CertRefused")
                    && !msg.contains("phantom_multiplier")
                    && !msg.contains("outer smoothing optimization did not converge")
                    && !msg.contains("no candidate seeds passed outer startup validation"),
                "production-like Matérn BMS fit still hit the #754 failure signature: {msg}"
            );
            panic!("production-like Matérn BMS fit failed: {msg}");
        }
    }
}

#[test]
fn production_like_binary_outcome_shared_matern_learned_kappa_starts_outer_solver() {
    gam::init_parallelism();
    let data = production_like_pc_confound_dataset();
    let cfg = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=6)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "event ~ matern(PC1, PC2, PC3, centers=6) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
        &data,
        &cfg,
    );
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            assert!(
                out.fit.beta.iter().all(|coef| coef.is_finite()),
                "learned-kappa production-like PC-confounded Matérn BMS fit should produce finite coefficients, got {:?}",
                out.fit.beta
            );
            assert!(
                out.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
                "learned-kappa BMS fit must report finite smoothing parameters after the Jeffreys/orthogonalization cure; got log_lambdas={:?}",
                out.fit.log_lambdas
            );
        }
        Ok(_) => {
            panic!("learned-kappa production-like Matérn fit returned the wrong family variant")
        }
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("identifiability audit refused")
                    && !msg.contains("CertRefused")
                    && !msg.contains("phantom_multiplier")
                    && !msg.contains("outer smoothing optimization did not converge")
                    && !msg.contains("joint hyper rho dimension mismatch")
                    && !msg.contains("no candidate seeds passed outer startup validation"),
                "learned-kappa production-like Matérn BMS fit still hit the #754 failure signature: {msg}"
            );
            panic!("learned-kappa production-like Matérn BMS fit failed: {msg}");
        }
    }
}
