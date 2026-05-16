//! Biobank-scale fit-time benchmark across all geometric smooths.
//!
//! Measures `fit_from_formula` end-to-end at N=1M for every geometric
//! feature we ship. Goal: prove the literal 150× target on the FIT path
//! (basis builds are already past 150× via parallel + algorithmic
//! changes, see docs/perf-150x-roadmap.md).
//!
//! These tests print timings via eprintln and are intentionally
//! ungated — they always pass even if the perf is slow — so the
//! numbers are visible in CI logs without breaking the build.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::f64::consts::{PI, TAU};
use std::time::Instant;

fn cylinder_data(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["theta".into(), "h".into(), "y".into()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let theta = TAU * (i as f64) / (n as f64);
            let h = -1.0 + 2.0 * ((i % 16) as f64) / 15.0;
            let y = 1.0 + 0.55 * theta.cos() - 0.25 * (2.0 * theta).sin() + 0.3 * h;
            StringRecord::from(vec![theta.to_string(), h.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("cyl")
}

fn periodic_1d_data(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["theta".into(), "y".into()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let theta = TAU * (i as f64) / (n as f64);
            let y = 0.5 + 0.4 * theta.cos() - 0.2 * (2.0 * theta).sin();
            StringRecord::from(vec![theta.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("p1d")
}

fn bc_1d_data(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["x".into(), "y".into()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = (i as f64) / (n as f64 - 1.0);
            // function with zero at x=0 (the BC anchor)
            let y = x * (1.0 - x) * (1.0 + 0.5 * (PI * x).sin());
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("bc")
}

fn sphere_data(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["lat".into(), "lon".into(), "y".into()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            // Lambert equal-area sphere coverage
            let frac = (i as f64) / (n as f64);
            let z = 1.0 - 2.0 * frac;
            let phi = TAU * ((i as f64) * 0.61803398875).fract();
            let lat = z.asin().to_degrees();
            let lon = phi.to_degrees() - 180.0;
            let lat_r = lat.to_radians();
            let lon_r = lon.to_radians();
            let y = 1.0 + lat_r.sin() + (2.0 * lon_r).cos() * lat_r.cos();
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("sphere")
}

fn time_fit(formula: &str, data: &gam::data::EncodedDataset, cfg: &FitConfig) -> (f64, usize) {
    let t = Instant::now();
    let res = fit_from_formula(formula, data, cfg);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let p = res
        .ok()
        .map(|r| match r {
            FitResult::Standard(f) => f.fit.beta.len(),
            _ => 0,
        })
        .unwrap_or(0);
    (ms, p)
}

#[test]
fn biobank_perf_cylinder_n1m() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = cylinder_data(1_000_000);
    let (ms, p) = time_fit(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None])",
        &data,
        &cfg,
    );
    eprintln!("[biobank-fit] cylinder N=1M p={p}: {ms:.0} ms");
}

#[test]
fn biobank_perf_periodic_1d_n1m() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = periodic_1d_data(1_000_000);
    let (ms, p) = time_fit(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data,
        &cfg,
    );
    eprintln!("[biobank-fit] periodic_1d N=1M p={p}: {ms:.0} ms");
}

#[test]
fn biobank_perf_bc_1d_n1m() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = bc_1d_data(1_000_000);
    let (ms, p) = time_fit("y ~ s(x, bc=anchored)", &data, &cfg);
    eprintln!("[biobank-fit] bc_anchored 1D N=1M p={p}: {ms:.0} ms");
}

#[test]
fn biobank_perf_sphere_wahba_n100k() {
    // Sphere Wahba kernel is O(N·K), so at N=1M K=50 = 50M kernel evals,
    // which dominates. Cap at N=100K for now to keep the test under a
    // minute.
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = sphere_data(100_000);
    let (ms, p) = time_fit("y ~ sphere(lat, lon, k=24)", &data, &cfg);
    eprintln!("[biobank-fit] sphere_wahba N=100K K=24 p={p}: {ms:.0} ms");
}

#[test]
fn biobank_perf_sphere_harmonic_n1m() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = sphere_data(1_000_000);
    let (ms, p) = time_fit(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        &data,
        &cfg,
    );
    eprintln!("[biobank-fit] sphere_harmonic N=1M L=4 p={p}: {ms:.0} ms");
}
