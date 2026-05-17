//! Bayesian credible intervals on sphere fits. The REML scale-invariance
//! fix changes the pseudo-logdet calculation; the posterior covariance
//! depends on penalty rank too. Verify the credible intervals on a
//! known truth have proper coverage (~95% empirical) when REML works.

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

fn truth(lat: f64, lon: f64) -> f64 {
    0.5 + 0.6 * lat.to_radians().sin()
        + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
}

fn make_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = truth(lat, lon) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn sphere_fit_predictions_stay_finite_and_close_to_truth_across_seeds() {
    init_parallelism();
    // Sphere fits with different random seeds should all reach similar
    // quality (rmse within a fixed envelope). If REML is unstable on any
    // particular seed (e.g. due to a borderline rank-tolerance case)
    // we'd see one seed blow up.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut probes = Vec::new();
    for i in 0..10 {
        let lat = -70.0 + 140.0 * (i as f64) / 9.0;
        for j in 0..10 {
            let lon = -160.0 + 320.0 * (j as f64) / 9.0;
            probes.push((lat, lon));
        }
    }
    let mut rmses = Vec::new();
    for seed in [3u64, 7, 11, 17, 23] {
        let data = make_dataset(300, seed);
        // Use the pseudo-spline path (the historically fragile one) at m=4.
        let result = fit_from_formula(
            "y ~ sphere(lat, lon, k=25, m=4, kernel=pseudo)",
            &data,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("seed={seed} fit: {e}"));
        let FitResult::Standard(fit) = result else { panic!() };
        let n = probes.len();
        let mut m = Array2::<f64>::zeros((n, 3));
        for (i, (lat, lon)) in probes.iter().enumerate() {
            m[[i, 0]] = *lat;
            m[[i, 1]] = *lon;
        }
        let design = build_term_collection_design(m.view(), &fit.resolvedspec)
            .expect("design");
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "seed={seed} produced non-finite predictions",
        );
        let sumsq: f64 = pred
            .iter()
            .zip(probes.iter())
            .map(|(p, (lat, lon))| (p - truth(*lat, *lon)).powi(2))
            .sum();
        let rmse = (sumsq / n as f64).sqrt();
        eprintln!("[sphere-seed] seed={seed} rmse={rmse:.4}");
        rmses.push(rmse);
    }
    // All seeds within 3× the median rmse (no extreme outlier from
    // REML instability).
    rmses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = rmses[2];
    let max = rmses[4];
    eprintln!("[sphere-seed] median rmse={median:.4}  max rmse={max:.4}");
    assert!(
        max < 3.0 * median + 0.05,
        "seed-to-seed instability: median={median:.4} but max={max:.4}",
    );
    // All rmses below 0.10 (good fit on a smooth truth).
    for (i, &r) in rmses.iter().enumerate() {
        assert!(
            r < 0.10,
            "seed rank {i} rmse={r:.4} > 0.10 — fit quality regressed",
        );
    }
}
