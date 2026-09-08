//! #2714 probe: the witness's own fit, with the solver's diagnostics turned on.
//!
//! The issue records that the quality binary installs no `log` backend, so
//! everything the joint-Newton inner emits about WHY it refuses — the
//! per-attempt trust ladder, the `TrustRatioRefinementWitness` model-consistency
//! fault, the objective's measured resolution — is dropped. Every lane that has
//! read this fixture has read it blind on that half, and turning it on costs one
//! line.
//!
//! This is the same fit as `tests/repro_2714_latent_frailty_inner_solve.rs`
//! (veteran lung, every 4th row held out, latent likelihood, Weibull baseline,
//! I-spline time basis, `HazardMultiplier` frailty with a learned scale) —
//! that file carries the GRADE, this one carries the READING.
//!
//! ```text
//! cargo run --release --example probe_2714_frailty_trust_ladder
//! ```

use csv::StringRecord;
use gam::families::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use gam::{
    FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use std::path::Path;

const VETERAN_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/veteran_lung.csv"
);

fn main() {
    gam::test_support::install_diagnostic_logger();
    init_parallelism();

    let raw = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let rcol = raw.column_map();
    let (r_time, r_status, r_karno, r_celltype) = (
        rcol["time"],
        rcol["status"],
        rcol["karno"],
        rcol["celltype"],
    );
    let celltype_levels = &raw.schema.columns[r_celltype].levels;
    let n = raw.values.nrows();

    let time: Vec<f64> = raw.values.column(r_time).to_vec();
    let status: Vec<f64> = raw.values.column(r_status).to_vec();
    let karno: Vec<f64> = raw.values.column(r_karno).to_vec();
    let celltype_label: Vec<String> = raw
        .values
        .column(r_celltype)
        .iter()
        .map(|&code| celltype_levels[code as usize].clone())
        .collect();

    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let headers = vec![
        "time".to_string(),
        "status".to_string(),
        "karno".to_string(),
        "celltype".to_string(),
    ];
    let train_records: Vec<StringRecord> = train_rows
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                time[i].to_string(),
                status[i].to_string(),
                karno[i].to_string(),
                celltype_label[i].clone(),
            ])
        })
        .collect();
    let train_ds = encode_recordswith_inferred_schema(headers, train_records)
        .expect("encode veteran train survival data");

    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "weibull".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
            loading: HazardLoading::Full,
        },
        ..FitConfig::default()
    };

    match fit_from_formula("Surv(time, status) ~ karno", &train_ds, &cfg) {
        Ok(_) => eprintln!("[2714-probe] the fit returned Ok"),
        Err(error) => eprintln!("[2714-probe] the fit returned Err:\n{error}"),
    }
    // A run of zeros is the easiest wrong answer to produce by accident: a
    // backend that silently failed to write is indistinguishable from a solver
    // that emitted nothing.
    let dropped = gam::test_support::diagnostic_write_failures();
    assert_eq!(
        dropped, 0,
        "#2714 probe: {dropped} diagnostic records were dropped, so this run measured nothing"
    );
}
