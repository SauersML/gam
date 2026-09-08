//! Reproduce #1561's fixed-design Gaussian interval experiment through the
//! model service, without linking the unrelated quality families or SAE.
//! Run with `cargo run -p gam-models --example gaussian_interval_calibration_1561 --profile test`.
//! This reports empirical coverage and width ordering; total covariance alone
//! does not imply an ordering against covariance conditional at the rho mode.

use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::smooth::build_term_collection_design;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Uniform};

fn main() {
    const N: usize = 300;
    const REPLICATES: usize = 30;
    let mut rng = StdRng::seed_from_u64(20_260_530);
    let uniform = Uniform::new(0.0_f64, 1.0).expect("valid design distribution");
    let mut pc1: Vec<f64> = (0..N).map(|_| uniform.sample(&mut rng)).collect();
    pc1.sort_by(f64::total_cmp);
    let pc2: Vec<f64> = (0..N)
        .map(|_| uniform.sample(&mut rng) * 2.0 - 1.0)
        .collect();
    let truth: Vec<f64> = pc1
        .iter()
        .zip(&pc2)
        .map(|(&x, &z)| (std::f64::consts::TAU * x).sin() + 0.5 * x + 0.8 * z)
        .collect();
    let noise = Normal::new(0.0, 0.3).expect("valid observation distribution");
    let config = FitConfig {
        family: Some("gaussian".into()),
        ..FitConfig::default()
    };
    let mut conditional_covered = 0;
    let mut marginal_covered = 0;
    let mut narrower = 0;
    let mut wider = 0;
    let mut min_ratio = f64::INFINITY;
    let mut max_ratio = f64::NEG_INFINITY;
    for replicate in 0..REPLICATES {
        let mut rng = StdRng::seed_from_u64(100 + replicate as u64);
        let rows = (0..N)
            .map(|row| {
                csv::StringRecord::from(vec![
                    pc1[row].to_string(),
                    pc2[row].to_string(),
                    (truth[row] + noise.sample(&mut rng)).to_string(),
                ])
            })
            .collect();
        let data = encode_recordswith_inferred_schema(
            ["pc1", "pc2", "y"].into_iter().map(String::from).collect(),
            rows,
        )
        .expect("encode fixed-design replicate");
        let FitResult::Standard(fit) = fit_from_formula("y ~ s(pc1) + pc2", &data, &config)
            .unwrap_or_else(|error| panic!("replicate {replicate}: {error}"))
        else {
            panic!("Gaussian model must produce a standard fit");
        };
        let design = build_term_collection_design(data.values.view(), &fit.resolvedspec)
            .expect("realize fitted Gaussian design")
            .design
            .to_dense();
        let conditional = fit
            .fit
            .beta_covariance()
            .expect("conditional coefficient covariance");
        let marginal = fit
            .fit
            .beta_covariance_corrected()
            .expect("marginal coefficient covariance");
        for (row, &mean) in design.rows().into_iter().zip(&truth) {
            let fitted = row.dot(&fit.fit.beta);
            let conditional_variance = row.dot(&conditional.dot(&row));
            let marginal_variance = row.dot(&marginal.dot(&row));
            assert!(conditional_variance.is_finite() && conditional_variance > 0.0);
            assert!(marginal_variance.is_finite() && marginal_variance > 0.0);
            let conditional_se = conditional_variance.sqrt();
            let marginal_se = marginal_variance.sqrt();
            conditional_covered +=
                usize::from((fitted - mean).abs() <= 1.959963984540054 * conditional_se);
            marginal_covered +=
                usize::from((fitted - mean).abs() <= 1.959963984540054 * marginal_se);
            narrower += usize::from(marginal_se + 1e-10 < conditional_se);
            wider += usize::from(marginal_se > conditional_se + 1e-10);
            min_ratio = min_ratio.min(marginal_se / conditional_se);
            max_ratio = max_ratio.max(marginal_se / conditional_se);
        }
        eprintln!(
            "replicate={} method={:?}",
            replicate + 1,
            fit.fit
                .inference
                .as_ref()
                .expect("Gaussian inference metadata")
                .smoothing_correction_method
        );
    }
    let count = (N * REPLICATES) as f64;
    println!(
        "rows={} conditional_coverage={:.6} marginal_coverage={:.6} narrower={narrower} wider={wider} ratio_min={min_ratio:.8} ratio_max={max_ratio:.8}",
        N * REPLICATES,
        conditional_covered as f64 / count,
        marginal_covered as f64 / count
    );
    assert!(
        (marginal_covered as f64 / count - 0.95).abs() <= 0.06,
        "marginal intervals must meet the existing empirical calibration bar"
    );
}
