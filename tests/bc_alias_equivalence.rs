//! BC option aliases: `bc_left` / `left_bc` / `start_bc` (and the
//! corresponding right variants) should map to the same internal
//! enum value and produce identical fits.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|t| (std::f64::consts::PI * t).sin() + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str) -> Vec<f64> {
    let data = make_dataset();
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let probes: Vec<f64> = (0..20).map(|i| 0.05 + 0.9 * (i as f64) / 19.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() { m[[i, 0]] = v; m[[i, 1]] = 0.0; }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn assert_equiv(reference: &str, alternatives: &[&str]) {
    let ref_pred = fit_predict(reference);
    for alt in alternatives {
        let pred = fit_predict(alt);
        let max_diff: f64 = ref_pred.iter().zip(pred.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff < 1e-10,
            "alias `{alt}` differs from reference `{reference}`: max_diff={max_diff:.3e}",
        );
    }
    eprintln!("[bc-alias] equiv: `{reference}` ≡ {alternatives:?}");
}

#[test]
fn bc_left_aliases_equivalent() {
    init_parallelism();
    assert_equiv(
        "y ~ s(x, bc_left=clamped, k=10)",
        &[
            "y ~ s(x, left_bc=clamped, k=10)",
            "y ~ s(x, start_bc=clamped, k=10)",
        ],
    );
}

#[test]
fn bc_right_aliases_equivalent() {
    init_parallelism();
    assert_equiv(
        "y ~ s(x, bc_right=clamped, k=10)",
        &[
            "y ~ s(x, right_bc=clamped, k=10)",
            "y ~ s(x, end_bc=clamped, k=10)",
        ],
    );
}

#[test]
fn bc_global_vs_explicit_both_sides_equivalent() {
    init_parallelism();
    assert_equiv(
        "y ~ s(x, bc=anchored, k=10)",
        &[
            "y ~ s(x, bc_left=anchored, bc_right=anchored, k=10)",
            "y ~ s(x, start_bc=anchored, end_bc=anchored, k=10)",
        ],
    );
}
