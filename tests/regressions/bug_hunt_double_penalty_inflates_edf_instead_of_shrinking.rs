use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn linear_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.15).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = i as f64 / (n.saturating_sub(1).max(1)) as f64;
            let y = 2.0 + 3.0 * x + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(["x", "y"].into_iter().map(String::from).collect(), rows)
        .expect("encode")
}

fn fit_edf(formula: &str, data: &gam::data::EncodedDataset) -> f64 {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("fit ok");
    match result {
        FitResult::Standard(fit) => {
            fit.fit
                .inference
                .as_ref()
                .expect("default fit must compute inference")
                .edf_total
        }
        FitResult::SplineScan(scan) => scan.edf(),
        _ => panic!("expected a standard Gaussian or spline-scan fit"),
    }
}

#[test]
fn bspline_double_penalty_does_not_inflate_linear_edf() {
    init_parallelism();

    let mut edf_on = Vec::new();
    let mut edf_off = Vec::new();
    for seed in 0..5 {
        let data = linear_dataset(seed, 800);
        edf_on.push(fit_edf("y ~ s(x, k=20, bs=ps, double_penalty=True)", &data));
        edf_off.push(fit_edf(
            "y ~ s(x, k=20, bs=ps, double_penalty=False)",
            &data,
        ));
    }

    let mean_on = edf_on.iter().sum::<f64>() / edf_on.len() as f64;
    let mean_off = edf_off.iter().sum::<f64>() / edf_off.len() as f64;
    assert!(
        mean_on <= 2.35,
        "B-spline double penalty did not reach the mgcv linear-data EDF target \
         (~2.10): double_penalty=true mean={mean_on:.6}, values={edf_on:?}"
    );
    assert!(
        mean_on <= mean_off + 1.0e-8,
        "enabling B-spline double penalty inflated EDF on linear data: \
         double_penalty=true mean={mean_on:.6}, values={edf_on:?}; \
         double_penalty=false mean={mean_off:.6}, values={edf_off:?}"
    );
}
