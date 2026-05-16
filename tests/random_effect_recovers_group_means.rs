//! FAILING TEST (potentially) — ticket: `group(g)` random effects should
//! recover known per-group means when n_per_group is large.
//!
//! Generate G=8 groups with distinct true means in [-1.5, 1.5], 50 samples per
//! group with σ=0.20. Fit `y ~ group(g)`. Predict for each group's "anchor"
//! sample; assert the predicted per-group mean is within 0.25 of the true
//! mean (much larger than σ/√50 ≈ 0.028, so this only fails on real bugs).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[test]
fn group_random_effect_recovers_per_group_means() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(23);
    let noise = Normal::new(0.0, 0.20).expect("normal");

    let true_means: [f64; 8] = [-1.4, -0.9, -0.4, 0.0, 0.3, 0.7, 1.1, 1.5];
    let n_per = 50usize;
    let mut g_col = Vec::with_capacity(true_means.len() * n_per);
    let mut y_col = Vec::with_capacity(true_means.len() * n_per);
    for (gi, mu) in true_means.iter().enumerate() {
        for _ in 0..n_per {
            g_col.push(gi as f64);
            y_col.push(mu + noise.sample(&mut rng));
        }
    }

    let headers = ["g", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = g_col
        .iter()
        .zip(y_col.iter())
        .map(|(a, b)| StringRecord::from(vec![format!("{}", *a as i64), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ group(g)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    // Predict for one anchor per group.
    let mut nd = Array2::<f64>::zeros((true_means.len(), 2));
    for gi in 0..true_means.len() {
        nd[[gi, 0]] = gi as f64;
        nd[[gi, 1]] = 0.0;
    }
    let design = build_term_collection_design(nd.view(), &fit.resolvedspec)
        .expect("rebuild predict design for group");
    let pred = design.design.apply(&fit.fit.beta);

    let mut worst = 0.0_f64;
    for (gi, mu) in true_means.iter().enumerate() {
        let e = (pred[gi] - mu).abs();
        eprintln!(
            "  group {gi}: predicted={:.4}  truth={:.4}  |err|={e:.4}",
            pred[gi], mu
        );
        if e > worst {
            worst = e;
        }
    }
    eprintln!("[group-recover] worst |err| = {worst:.4} (budget 0.25)");
    assert!(
        worst < 0.25,
        "group(g) random-effect predictions deviate from true per-group means by up to {worst:.4} (budget 0.25)",
    );
}
