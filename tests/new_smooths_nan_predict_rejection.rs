//! Predict-time NaN/Inf handling for the new smooth families.
//!
//! When the user passes NaN/Inf in the predict matrix, the system must
//! produce a clean actionable error rather than silently propagating
//! NaN into the predicted vector (which corrupts downstream analyses).

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

const TAU: f64 = std::f64::consts::TAU;

fn make_sphere_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.9_f64, 179.9).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.6 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn make_periodic_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..150).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| 1.0 + 0.6 * theta.cos() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn make_cylinder_dataset() -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let mut records = Vec::with_capacity(20 * 6);
    for i in 0..20 {
        let theta = TAU * (i as f64) / 20.0;
        for j in 0..6 {
            let h = -1.0 + 2.0 * (j as f64) / 5.0;
            let y = 1.0 + 0.6 * theta.cos() + 0.4 * h;
            records.push(StringRecord::from(vec![
                theta.to_string(),
                h.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

fn predict_with_nan(formula: &str, data: &gam::data::EncodedDataset, nan_inputs: Vec<(f64, f64)>) -> Result<Vec<f64>, String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        return Err("expected standard".to_string());
    };
    let n = nan_inputs.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (a, b)) in nan_inputs.iter().enumerate() {
        m[[i, 0]] = *a;
        m[[i, 1]] = *b;
        m[[i, 2]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e}"))?;
    Ok(design.design.apply(&fit.fit.beta).to_vec())
}

#[test]
fn sphere_wahba_predict_nan_rejected_or_nonpropagated() {
    init_parallelism();
    let data = make_sphere_dataset();
    let r = predict_with_nan(
        "y ~ sphere(lat, lon, k=20)",
        &data,
        vec![(f64::NAN, 0.0), (45.0, f64::NAN), (0.0, 0.0)],
    );
    match r {
        Ok(pred) => {
            // If accepted, must NOT silently NaN the OK row.
            assert!(
                pred[2].is_finite(),
                "Wahba propagated NaN to OK row: {pred:?}",
            );
        }
        Err(e) => {
            let lower = e.to_lowercase();
            assert!(
                lower.contains("nan") || lower.contains("finite") || lower.contains("latitude") || lower.contains("longitude"),
                "Wahba rejected NaN without a clear message: {e}",
            );
        }
    }
}

#[test]
fn sphere_harmonic_predict_nan_rejected_or_nonpropagated() {
    init_parallelism();
    let data = make_sphere_dataset();
    let r = predict_with_nan(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        &data,
        vec![(f64::NAN, 0.0), (45.0, f64::NAN), (0.0, 0.0)],
    );
    match r {
        Ok(pred) => {
            assert!(pred[2].is_finite(), "harmonic propagated NaN: {pred:?}");
        }
        Err(e) => {
            let lower = e.to_lowercase();
            assert!(
                lower.contains("nan") || lower.contains("finite") || lower.contains("latitude") || lower.contains("longitude"),
                "harmonic rejected NaN without clear message: {e}",
            );
        }
    }
}

#[test]
fn periodic_1d_predict_nan_rejected_or_nonpropagated() {
    init_parallelism();
    let data = make_periodic_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &data,
        &cfg,
    )
    .expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((3, 2));
    m[[0, 0]] = f64::NAN;
    m[[1, 0]] = f64::INFINITY;
    m[[2, 0]] = 1.5;
    let design = build_term_collection_design(m.view(), &fit.resolvedspec);
    match design {
        Ok(d) => {
            let pred = d.design.apply(&fit.fit.beta);
            assert!(pred[2].is_finite(), "periodic 1D propagated NaN: {pred:?}");
        }
        Err(e) => {
            let lower = e.to_string().to_lowercase();
            assert!(
                lower.contains("nan") || lower.contains("inf") || lower.contains("finite"),
                "periodic 1D rejected without clear message: {e}",
            );
        }
    }
}

#[test]
fn cylinder_te_predict_nan_rejected_or_nonpropagated() {
    init_parallelism();
    let data = make_cylinder_dataset();
    let r = predict_with_nan(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        &data,
        vec![(f64::NAN, 0.0), (1.5, f64::NAN), (1.5, 0.0)],
    );
    match r {
        Ok(pred) => {
            assert!(pred[2].is_finite(), "cylinder te propagated NaN: {pred:?}");
        }
        Err(e) => {
            let lower = e.to_lowercase();
            assert!(
                lower.contains("nan") || lower.contains("finite") || lower.contains("inf"),
                "cylinder rejected NaN without clear message: {e}",
            );
        }
    }
}
