//! One-binary adversarial probe battery for the new smooth families.
//! Many small probes share one compile cycle so we can cover dozens of
//! corner cases quickly. Each probe is reported (not panicked) and at
//! the end we hard-assert that no probe produced NaN/Inf.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn n(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.u() - 1.0;
            let v = 2.0 * self.u() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 { return u * (-2.0 * s.ln() / s).sqrt(); }
        }
    }
}

#[derive(Debug)]
struct Outcome {
    probe: String,
    status: &'static str, // OK, CLEAN_ERR, NAN, PANIC
    detail: String,
}

fn run_probe(
    probe: &str,
    formula: &str,
    data: gam::data::EncodedDataset,
    probes: Vec<Vec<f64>>,
    ncols: usize,
) -> Outcome {
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let fit_r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_from_formula(formula, &data, &cfg)
    }));
    let fit_r = match fit_r {
        Ok(r) => r,
        Err(_) => return Outcome { probe: probe.into(), status: "PANIC", detail: "fit panicked".into() },
    };
    let result = match fit_r {
        Ok(r) => r,
        Err(e) => return Outcome { probe: probe.into(), status: "CLEAN_ERR", detail: format!("fit: {e}") },
    };
    let FitResult::Standard(fit) = result else {
        return Outcome { probe: probe.into(), status: "CLEAN_ERR", detail: "non-standard fit".into() };
    };
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, ncols));
    for (i, row) in probes.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m[[i, j]] = v;
        }
    }
    let predict_r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build_term_collection_design(m.view(), &fit.resolvedspec)
    }));
    let predict_r = match predict_r {
        Ok(r) => r,
        Err(_) => return Outcome { probe: probe.into(), status: "PANIC", detail: "predict panicked".into() },
    };
    let design = match predict_r {
        Ok(d) => d,
        Err(e) => return Outcome { probe: probe.into(), status: "CLEAN_ERR", detail: format!("design: {e:?}") },
    };
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Outcome { probe: probe.into(), status: "NAN", detail: format!("pred: {pred:?}") };
    }
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Outcome { probe: probe.into(), status: "OK", detail: format!("range=[{mn:.3}, {mx:.3}]") }
}

fn mk1d(name: &str, vals: Vec<(f64, f64)>) -> gam::data::EncodedDataset {
    let rows: Vec<StringRecord> = vals.into_iter()
        .map(|(x, y)| StringRecord::from(vec![x.to_string(), y.to_string()]))
        .collect();
    let hs = vec![name.to_string(), "y".to_string()];
    encode_recordswith_inferred_schema(hs, rows).expect("encode")
}

fn mk2d(headers_in: Vec<&str>, vals: Vec<(f64, f64, f64)>) -> gam::data::EncodedDataset {
    let headers: Vec<String> = headers_in.iter().map(|s| s.to_string()).collect();
    let rows: Vec<StringRecord> = vals.into_iter()
        .map(|(a, b, c)| StringRecord::from(vec![a.to_string(), b.to_string(), c.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn adversarial_battery() {
    init_parallelism();
    let mut out = Vec::<Outcome>::new();

    // ---- PERIODIC 1D ----
    let mut rng = Lcg::new(1);
    // 1. Very large period (100.0)
    let pts: Vec<(f64, f64)> = (0..200).map(|_| {
        let t = 100.0 * rng.u();
        (t, (t * TAU / 100.0).sin() + 0.05 * rng.n())
    }).collect();
    out.push(run_probe("per1d_large_period", "y ~ s(x, periodic=true, period=100.0)",
        mk1d("x", pts),
        (0..5).map(|i| vec![10.0 + 20.0 * i as f64, 0.0]).collect(), 2));

    // 2. Very small period (0.1)
    let pts: Vec<(f64, f64)> = (0..200).map(|_| {
        let t = 0.1 * rng.u();
        (t, (t * TAU / 0.1).sin() + 0.05 * rng.n())
    }).collect();
    out.push(run_probe("per1d_small_period", "y ~ s(x, periodic=true, period=0.1)",
        mk1d("x", pts),
        (0..5).map(|i| vec![0.01 * i as f64 + 0.005, 0.0]).collect(), 2));

    // 3. Data at single point (degenerate)
    let pts: Vec<(f64, f64)> = (0..200).map(|_| (1.5_f64, rng.n() * 0.1)).collect();
    out.push(run_probe("per1d_degenerate_constant_x", "y ~ s(x, periodic=true, period=6.283185307179586)",
        mk1d("x", pts), vec![vec![0.0, 0.0]], 2));

    // 4. Heavily concentrated data + few outliers
    let mut pts: Vec<(f64, f64)> = (0..195).map(|_| {
        let t = 3.0 + 0.1 * rng.n();
        (t.rem_euclid(TAU), 0.5 * t.cos() + 0.05 * rng.n())
    }).collect();
    pts.extend((0..5).map(|_| (TAU * rng.u(), 5.0 * rng.n())));
    out.push(run_probe("per1d_concentrated_plus_outliers", "y ~ s(x, periodic=true, period=6.283185307179586)",
        mk1d("x", pts), (0..5).map(|i| vec![1.5 * i as f64, 0.0]).collect(), 2));

    // ---- SPHERE (Wahba) ----
    // 5. Single latitude band
    let pts: Vec<(f64, f64, f64)> = (0..150).map(|_| {
        let lat = 30.0_f64 + 1.0 * rng.n();
        let lon = -180.0 + 360.0 * rng.u();
        let y = lon.to_radians().cos() + 0.1 * rng.n();
        (lat, lon, y)
    }).collect();
    out.push(run_probe("sphere_w_single_lat_band", "y ~ sphere(lat, lon, k=20)",
        mk2d(vec!["lat", "lon", "y"], pts), vec![
            vec![30.0, 0.0, 0.0], vec![-30.0, 0.0, 0.0], vec![0.0, 90.0, 0.0],
        ], 3));

    // 6. Single longitude band
    let pts: Vec<(f64, f64, f64)> = (0..150).map(|_| {
        let lat = -80.0 + 160.0 * rng.u();
        let lon = 45.0 + 1.0 * rng.n();
        let y = lat.to_radians().sin() + 0.1 * rng.n();
        (lat, lon, y)
    }).collect();
    out.push(run_probe("sphere_w_single_lon_band", "y ~ sphere(lat, lon, k=20)",
        mk2d(vec!["lat", "lon", "y"], pts), vec![
            vec![45.0, 45.0, 0.0], vec![45.0, -45.0, 0.0],
        ], 3));

    // 7. Heavy polar concentration
    let pts: Vec<(f64, f64, f64)> = (0..200).map(|_| {
        let lat = 85.0 + 5.0 * rng.u();
        let lon = -180.0 + 360.0 * rng.u();
        let y = 0.5 + 0.1 * rng.n();
        (lat, lon, y)
    }).collect();
    out.push(run_probe("sphere_w_polar_only", "y ~ sphere(lat, lon, k=20)",
        mk2d(vec!["lat", "lon", "y"], pts), vec![vec![-45.0, 0.0, 0.0], vec![0.0, 90.0, 0.0]], 3));

    // ---- SPHERE (Harmonic) ----
    // 8. Same single-lat-band on harmonic
    let pts: Vec<(f64, f64, f64)> = (0..150).map(|_| {
        let lat = 30.0_f64 + 1.0 * rng.n();
        let lon = -180.0 + 360.0 * rng.u();
        let y = lon.to_radians().cos() + 0.1 * rng.n();
        (lat, lon, y)
    }).collect();
    out.push(run_probe("sphere_h_single_lat_band", "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        mk2d(vec!["lat", "lon", "y"], pts), vec![
            vec![30.0, 0.0, 0.0], vec![-30.0, 0.0, 0.0], vec![0.0, 90.0, 0.0],
        ], 3));

    // 9. Harmonic with very small dataset
    let pts: Vec<(f64, f64, f64)> = (0..15).map(|_| {
        let lat = -75.0 + 150.0 * rng.u();
        let lon = -179.0 + 358.0 * rng.u();
        let y = 0.5 * lat.to_radians().sin() + 0.1 * rng.n();
        (lat, lon, y)
    }).collect();
    out.push(run_probe("sphere_h_tiny_n", "y ~ sphere(lat, lon, method=harmonic, max_degree=2)",
        mk2d(vec!["lat", "lon", "y"], pts), vec![vec![0.0, 0.0, 0.0]], 3));

    // ---- CYLINDER ----
    // 10. Cylinder with very narrow height band
    let mut pts = Vec::<(f64, f64, f64)>::new();
    for i in 0..20 {
        let theta = TAU * (i as f64) / 20.0;
        for _ in 0..5 {
            let h = 0.5 + 0.01 * rng.n();
            let y = theta.cos() + 0.1 * rng.n();
            pts.push((theta, h, y));
        }
    }
    out.push(run_probe("cylinder_narrow_h", "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        mk2d(vec!["theta", "h", "y"], pts), vec![
            vec![0.0, 0.5, 0.0], vec![PI, 0.5, 0.0], vec![PI, -0.5, 0.0],
        ], 3));

    // 11. Cylinder with huge height range
    let mut pts = Vec::<(f64, f64, f64)>::new();
    for i in 0..20 {
        let theta = TAU * (i as f64) / 20.0;
        for j in 0..6 {
            let h = -1000.0 + 2000.0 * (j as f64) / 5.0;
            let y = theta.cos() + 0.001 * h + 0.1 * rng.n();
            pts.push((theta, h, y));
        }
    }
    out.push(run_probe("cylinder_huge_h_range", "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        mk2d(vec!["theta", "h", "y"], pts), vec![
            vec![0.0, -500.0, 0.0], vec![PI, 500.0, 0.0],
        ], 3));

    // ---- BC ----
    // 12. BC anchored with truth that has zero value AND zero slope at boundary
    let pts: Vec<(f64, f64)> = (0..200).map(|_| {
        let x = rng.u();
        let y = (PI * x).sin().powi(2) + 0.05 * rng.n();
        (x, y)
    }).collect();
    out.push(run_probe("bc_anchored_zero_slope_truth", "y ~ s(x, bc=anchored, k=15)",
        mk1d("x", pts), (0..5).map(|i| vec![0.1 + 0.2 * i as f64, 0.0]).collect(), 2));

    // 13. BC clamped with truth that has nonzero slope at boundary (mismatch)
    let pts: Vec<(f64, f64)> = (0..200).map(|_| {
        let x = rng.u();
        let y = 0.5 + 0.3 * x + 0.05 * rng.n();
        (x, y)
    }).collect();
    out.push(run_probe("bc_clamped_nonzero_slope_truth", "y ~ s(x, bc=clamped, k=15)",
        mk1d("x", pts), (0..5).map(|i| vec![0.1 + 0.2 * i as f64, 0.0]).collect(), 2));

    // 14. BC anchored at one end, free at other
    let pts: Vec<(f64, f64)> = (0..200).map(|_| {
        let x = rng.u();
        let y = x.powi(3) + 0.05 * rng.n();
        (x, y)
    }).collect();
    out.push(run_probe("bc_anchored_left_only", "y ~ s(x, bc_left=anchored, k=15)",
        mk1d("x", pts), (0..5).map(|i| vec![0.05 + 0.225 * i as f64, 0.0]).collect(), 2));

    // ---- MATERN ----
    // 15. Matern on adversarial high-frequency truth
    let pts: Vec<(f64, f64)> = (0..200).map(|_| {
        let x = rng.u();
        let y = (TAU * 10.0 * x).sin() + 0.1 * rng.n();
        (x, y)
    }).collect();
    for nu in ["1/2", "3/2", "5/2", "7/2", "9/2"] {
        out.push(run_probe(&format!("matern_hf_nu_{}", nu.replace('/', "_")),
            &format!("y ~ matern(x, nu={nu})"),
            mk1d("x", pts.clone()), (0..5).map(|i| vec![0.1 + 0.2 * i as f64, 0.0]).collect(), 2));
    }

    // ---- Report ----
    for o in &out {
        eprintln!("[battery] {} {} -- {}", o.status, o.probe, o.detail);
    }
    let bad: Vec<&Outcome> = out.iter().filter(|o| o.status == "NAN" || o.status == "PANIC").collect();
    assert!(
        bad.is_empty(),
        "Adversarial battery NaN/PANIC failures:\n  - {}",
        bad.iter().map(|o| format!("{} {} -- {}", o.status, o.probe, o.detail)).collect::<Vec<_>>().join("\n  - "),
    );
}
