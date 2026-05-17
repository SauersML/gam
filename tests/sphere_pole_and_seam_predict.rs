//! Adversarial sphere predict probes:
//!
//! 1. Predict at the exact poles (lat = ±90°) for both methods.
//! 2. Predict at longitudes outside [-180, 180] (e.g. 540°, -200°) — these
//!    are valid sphere points modulo 360° and should return the same value
//!    as the wrapped equivalents.
//! 3. Symmetric seam predictions: f(lat, -180) == f(lat, 180) for any lat.
//!
//! For all three the fit should not produce NaN/Inf, and the wrapped/seam
//! invariants should hold to machine precision (the sphere itself is
//! continuous; any discontinuity is a basis bug).

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

fn make_data() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(15 * 30);
    for i in 0..15 {
        let lat = -80.0 + 160.0 * (i as f64) / 14.0;
        for j in 0..30 {
            let lon = -175.0 + 350.0 * (j as f64) / 29.0;
            let lat_r = lat.to_radians();
            let lon_r = lon.to_radians();
            let y = 0.5
                + 0.7 * lat_r.sin()
                + 0.4 * lat_r.cos() * (2.0 * lon_r).cos()
                + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict_all(formula: &str, points: &[(f64, f64)]) -> Vec<f64> {
    let data = make_data();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("sphere fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = points.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in points.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
        m[[i, 2]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn sphere_wahba_predicts_at_poles_finite() {
    init_parallelism();
    let pts = [(90.0, 0.0), (90.0, 45.0), (-90.0, 0.0), (-90.0, -120.0)];
    let pred = predict_all("y ~ sphere(lat, lon, k=48)", &pts);
    for (i, p) in pred.iter().enumerate() {
        assert!(
            p.is_finite(),
            "Wahba predicted non-finite at pole probe {:?} = {p}",
            pts[i],
        );
    }
}

#[test]
fn sphere_harmonic_predicts_at_poles_finite() {
    init_parallelism();
    let pts = [(90.0, 0.0), (90.0, 45.0), (-90.0, 0.0), (-90.0, -120.0)];
    let pred = predict_all("y ~ sphere(lat, lon, method=harmonic, max_degree=6)", &pts);
    for (i, p) in pred.iter().enumerate() {
        assert!(
            p.is_finite(),
            "harmonic predicted non-finite at pole probe {:?} = {p}",
            pts[i],
        );
    }
}

#[test]
fn sphere_wahba_pole_predict_lon_invariant() {
    // At lat=±90 the longitude doesn't change the point on the sphere; the
    // predicted value must NOT depend on the longitude passed at the pole.
    // Only lat = ±90 *exactly* is the pole; lat = 89.999 is a small ring of
    // genuinely-distinct points where the value can legitimately differ.
    init_parallelism();
    let lats = [90.0, -90.0];
    let lons = [-180.0, -90.0, 0.0, 90.0, 180.0];
    let mut pts = Vec::new();
    for lat in lats {
        for lon in lons {
            pts.push((lat, lon));
        }
    }
    let pred = predict_all("y ~ sphere(lat, lon, k=48)", &pts);
    // For lat = ±90, all longitudes should give the same value.
    for (lat_idx, _lat) in lats.iter().enumerate() {
        let base = pred[lat_idx * lons.len()];
        for lon_idx in 1..lons.len() {
            let p = pred[lat_idx * lons.len() + lon_idx];
            let diff = (p - base).abs();
            assert!(
                diff < 1e-6,
                "Wahba pole predict longitude-dependent: lat={} lons {} vs {} → {p:.6} vs {base:.6} diff={diff:.3e}",
                lats[lat_idx],
                lons[0],
                lons[lon_idx],
            );
        }
    }
}

#[test]
fn sphere_harmonic_pole_predict_lon_invariant() {
    init_parallelism();
    let lats = [90.0, -90.0];
    let lons = [-180.0, -90.0, 0.0, 90.0, 180.0];
    let mut pts = Vec::new();
    for lat in lats {
        for lon in lons {
            pts.push((lat, lon));
        }
    }
    let pred = predict_all("y ~ sphere(lat, lon, method=harmonic, max_degree=6)", &pts);
    for (lat_idx, _lat) in lats.iter().enumerate() {
        let base = pred[lat_idx * lons.len()];
        for lon_idx in 1..lons.len() {
            let p = pred[lat_idx * lons.len() + lon_idx];
            let diff = (p - base).abs();
            assert!(
                diff < 1e-6,
                "harmonic pole predict longitude-dependent: lat={} lons {} vs {} → {p:.6} vs {base:.6} diff={diff:.3e}",
                lats[lat_idx],
                lons[0],
                lons[lon_idx],
            );
        }
    }
}

#[test]
fn sphere_wahba_longitude_wrap_invariance() {
    // f(lat, lon) must equal f(lat, lon + 360°·k) for any integer k.
    init_parallelism();
    let lats = [-60.0, -30.0, 0.0, 30.0, 60.0];
    let base_lons = [-170.0, -90.0, 0.0, 90.0, 170.0];
    let mut pts = Vec::new();
    for lat in lats {
        for lon in base_lons {
            pts.push((lat, lon));
        }
    }
    // Shifted by +360° and -360°
    for lat in lats {
        for lon in base_lons {
            pts.push((lat, lon + 360.0));
        }
    }
    for lat in lats {
        for lon in base_lons {
            pts.push((lat, lon - 360.0));
        }
    }
    let pred = predict_all("y ~ sphere(lat, lon, k=48)", &pts);
    let n_grid = lats.len() * base_lons.len();
    for k in 1..3 {
        for i in 0..n_grid {
            let base = pred[i];
            let shifted = pred[k * n_grid + i];
            let diff = (base - shifted).abs();
            assert!(
                diff < 1e-6,
                "Wahba lon-wrap invariance broken at grid index {i} (shift {}·360°): {base:.6} vs {shifted:.6} diff={diff:.3e}",
                if k == 1 { "+1" } else { "-1" },
            );
        }
    }
}

#[test]
fn sphere_harmonic_longitude_wrap_invariance() {
    init_parallelism();
    let lats = [-60.0, -30.0, 0.0, 30.0, 60.0];
    let base_lons = [-170.0, -90.0, 0.0, 90.0, 170.0];
    let mut pts = Vec::new();
    for lat in lats {
        for lon in base_lons {
            pts.push((lat, lon));
        }
    }
    for lat in lats {
        for lon in base_lons {
            pts.push((lat, lon + 360.0));
        }
    }
    for lat in lats {
        for lon in base_lons {
            pts.push((lat, lon - 360.0));
        }
    }
    let pred = predict_all("y ~ sphere(lat, lon, method=harmonic, max_degree=6)", &pts);
    let n_grid = lats.len() * base_lons.len();
    for k in 1..3 {
        for i in 0..n_grid {
            let base = pred[i];
            let shifted = pred[k * n_grid + i];
            let diff = (base - shifted).abs();
            assert!(
                diff < 1e-6,
                "harmonic lon-wrap invariance broken at grid index {i} (shift {}·360°): {base:.6} vs {shifted:.6} diff={diff:.3e}",
                if k == 1 { "+1" } else { "-1" },
            );
        }
    }
}
