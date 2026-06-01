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

fn truth(a: f64, b: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let a = ux.sample(&mut rng);
            let b = ux.sample(&mut rng);
            let y = truth(a, b) + noise.sample(&mut rng);
            StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn matern_formula_fit_succeeds_on_ordinary_2d_data() {
    init_parallelism();
    let data = build_dataset(150, 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ matern(x1, x2)", &data, &cfg)
        .unwrap_or_else(|e| panic!("matern(x1, x2) failed to fit ordinary 150-row 2-D data: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for matern(x1, x2)");
    };

    let g: Vec<f64> = (0..20).map(|i| 0.05 + 0.90 * i as f64 / 19.0).collect();
    let m = g.len();
    let mut design_in = Array2::<f64>::zeros((m * m, 3));
    let mut truth_vals = Vec::with_capacity(m * m);
    let mut row = 0;
    for &a in &g {
        for &b in &g {
            design_in[[row, 0]] = a;
            design_in[[row, 1]] = b;
            design_in[[row, 2]] = 0.0;
            truth_vals.push(truth(a, b));
            row += 1;
        }
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild matern design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();

    assert!(
        pred.iter().all(|v| v.is_finite()),
        "matern predictions must all be finite"
    );
    let span = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - pred.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        span > 0.20,
        "matern fit collapsed to a near-constant surface (span={span:.4})"
    );
    let rmse = (pred
        .iter()
        .zip(truth_vals.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / (m * m) as f64)
        .sqrt();
    assert!(
        rmse < 0.35,
        "matern interior RMSE {rmse:.4} exceeds the 0.35 budget"
    );
}
