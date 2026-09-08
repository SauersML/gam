//! Replay the prostate holdout fit that previously refused fold 3 in #1561.
//! Usage: binomial_holdout_1561 PROSTATE_CSV PREDICTIONS_CSV [FOLD]
//! The default runs all five folds, starting with the previously failing fold.
//! The predictions reproduce the quality fixture's frozen design and eta.

use gam_data::{EncodedDataset, load_csvwith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use ndarray::Array2;
use std::path::Path;
use std::time::Instant;

fn subset(full: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let mut values = Array2::zeros((rows.len(), full.headers.len()));
    for (out, &source) in rows.iter().enumerate() {
        values.row_mut(out).assign(&full.values.row(source));
    }
    EncodedDataset {
        headers: full.headers.clone(),
        values,
        schema: full.schema.clone(),
        column_kinds: full.column_kinds.clone(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("expected PROSTATE_CSV PREDICTIONS_CSV [FOLD]".into());
    }
    let folds = match args.get(3) {
        Some(fold) => vec![fold.parse::<usize>()?],
        None => vec![3, 0, 1, 2, 4],
    };
    if folds.iter().any(|&fold| fold >= 5) {
        return Err("fold must be in 0..5".into());
    }
    let data = load_csvwith_inferred_schema(Path::new(&args[1]))?;
    let columns = data.column_map();
    let config = FitConfig {
        family: Some("binomial".into()),
        link: Some("logit".into()),
        ..FitConfig::default()
    };
    let mut writer = csv::Writer::from_path(&args[2])?;
    writer.write_record(["fold", "row", "y", "eta"])?;
    for fold in folds {
        let training: Vec<_> = (0..data.values.nrows()).filter(|i| i % 5 != fold).collect();
        let held_out: Vec<_> = (0..data.values.nrows()).filter(|i| i % 5 == fold).collect();
        let started = Instant::now();
        eprintln!("fold={fold} start training_rows={}", training.len());
        let fitted = fit_from_formula(
            "y ~ s(pc1, k=5) + s(pc2, k=5)",
            &subset(&data, &training),
            &config,
        )
        .map_err(|error| {
            format!(
                "fold={fold} elapsed={:.3}s fit_refused={error}",
                started.elapsed().as_secs_f64()
            )
        })?;
        let FitResult::Standard(fit) = fitted else {
            return Err("binomial logit must return a standard fit".into());
        };
        let frozen = freeze_term_collection_from_design(&fit.resolvedspec, &fit.design)?;
        let test = subset(&data, &held_out);
        let design = build_term_collection_design(test.values.view(), &frozen)?;
        let eta = design.design.to_dense().dot(&fit.fit.beta);
        for (&row, &linear) in held_out.iter().zip(&eta) {
            writer.write_record([
                fold.to_string(),
                row.to_string(),
                data.values[[row, columns["y"]]].to_string(),
                linear.to_string(),
            ])?;
        }
        writer.flush()?;
        eprintln!(
            "fold={fold} elapsed={:.3}s edf={:?} rho={:?}",
            started.elapsed().as_secs_f64(),
            fit.fit.edf_total(),
            fit.fit.log_lambdas
        );
    }
    Ok(())
}
