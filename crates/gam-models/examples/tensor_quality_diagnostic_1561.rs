//! Export the two fixed tensor fixtures from #1561 for independent model audits.
//! Usage: tensor_quality_diagnostic_1561 OUTPUT_DIRECTORY
//! The CSVs retain observations, known truth, and the fitted response together.

use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::smooth::build_term_collection_design;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use std::path::Path;

fn fixture(output: &Path, poisson: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(345);
    let (nx, nz, family, formula) = if poisson {
        (15, 20, "poisson", "y ~ te(x, z, k=[6,6])")
    } else {
        (18, 18, "gaussian", "y ~ ti(x, z, k=6)")
    };
    let mut values = Vec::new();
    for i in 0..nx {
        for j in 0..nz {
            let mut x = i as f64 / (nx - 1) as f64;
            let mut z = j as f64 / (nz - 1) as f64;
            let truth = if poisson {
                x *= std::f64::consts::TAU;
                z = 2.0 * z - 1.0;
                (0.8 + 0.3 * x.sin() + 0.2 * z * z).exp()
            } else {
                (3.0 * x).sin() * (3.0 * z).cos()
            };
            let y = if poisson {
                Poisson::new(truth)?.sample(&mut rng)
            } else {
                truth
            };
            values.push([x, z, y, truth]);
        }
    }
    let records = values
        .iter()
        .map(|row| {
            csv::StringRecord::from(row[..3].iter().map(ToString::to_string).collect::<Vec<_>>())
        })
        .collect();
    let data = encode_recordswith_inferred_schema(
        ["x", "z", "y"].into_iter().map(String::from).collect(),
        records,
    )?;
    // Retain the common observations even when a model refuses to fit.
    let csv_path = output.join(format!("{family}-tensor.csv"));
    let mut writer = csv::Writer::from_path(&csv_path)?;
    writer.write_record(["x", "z", "y", "truth"])?;
    for row in &values {
        writer.write_record(row.iter().map(ToString::to_string))?;
    }
    writer.flush()?;
    let config = FitConfig {
        family: Some(family.into()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula(formula, &data, &config)? else {
        return Err("expected standard tensor fit".into());
    };
    let design = build_term_collection_design(data.values.view(), &fit.resolvedspec)?;
    let x = design.design.to_dense();
    let eta = x.dot(&fit.fit.beta);
    let problem = serde_json::json!({
        "family": family,
        "x": x.rows().into_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
        "y": values.iter().map(|row| row[2]).collect::<Vec<_>>(),
        "truth": values.iter().map(|row| row[3]).collect::<Vec<_>>(),
        "beta": fit.fit.beta.to_vec(),
        "rho": fit.fit.log_lambdas.to_vec(),
        "phi": fit.fit.dispersion_phi()?,
        "criterion": fit.fit.reml_score(),
        "log_likelihood": fit.fit.log_likelihood,
        "stable_penalty_term": fit.fit.stable_penalty_term,
        "penalized_objective": fit.fit.penalized_objective(),
        "penalties": fit.design.penalties.iter().map(|penalty| serde_json::json!({
            "start": penalty.col_range.start,
            "end": penalty.col_range.end,
            "matrix": penalty.local.rows().into_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "nullspace_dims": fit.design.nullspace_dims,
        "conditional_covariance": fit.fit.beta_covariance(),
        "marginal_covariance": fit.fit.beta_covariance_corrected(),
    });
    serde_json::to_writer_pretty(
        std::fs::File::create(output.join(format!("{family}-tensor.json")))?,
        &problem,
    )?;
    let mut writer = csv::Writer::from_path(csv_path)?;
    writer.write_record(["x", "z", "y", "truth", "gam"])?;
    for (row, &linear) in values.iter().zip(&eta) {
        let mean = if poisson { linear.exp() } else { linear };
        writer.write_record(row.iter().copied().chain([mean]).map(|v| v.to_string()))?;
    }
    writer.flush()?;
    eprintln!(
        "family={family} formula={formula} edf={:?} rho={:?} objective={:?}",
        fit.fit.edf_total(),
        fit.fit.log_lambdas,
        fit.fit.reml_score()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err("expected OUTPUT_DIRECTORY".into());
    }
    let output = Path::new(&args[1]);
    std::fs::create_dir_all(output)?;
    let mut failed = false;
    for poisson in [true, false] {
        if let Err(error) = fixture(output, poisson) {
            eprintln!("poisson={poisson} fit_refused={error}");
            failed = true;
        }
    }
    if failed {
        Err("at least one tensor fit refused; observations retained for reference audit".into())
    } else {
        Ok(())
    }
}
