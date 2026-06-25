//! #1464 diagnostic: print the production full-fit κ̂ for both mirror datasets
//! (spherical κ⋆=+2 and hyperbolic κ⋆=−2) so the κ-sign behaviour of the joint
//! spatial solve is directly observable. This is a DIAGNOSTIC (plain `eprintln!`,
//! no `{:?}`), not a pass/fail contract — the contract lives in
//! `bug_hunt_1464_curv_sign_identifiable.rs`. It exists so a maintainer can see,
//! per dataset, the final κ̂ the fit reports after the fixed-κ sign-basin scan +
//! hard sign-pin land, and whether the two mirror datasets are now separated.
//!
//! The scan's selected κ_seed per term is emitted by the production
//! `log::info!("[spatial-kappa] #1464 fixed-κ sign-basin scan selected …")`; run
//! with `RUST_LOG=info` to see it alongside this κ̂ output.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::smooth::get_constant_curvature_kappa;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

use csv::StringRecord;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn curved_dataset(kappa_star: f64, seed: u64) -> gam::data::EncodedDataset {
    let radius = 0.68_f64;
    let noise = 0.02_f64;
    let n = 600usize;
    let manifold = ConstantCurvature::new(2, kappa_star);
    let origin = ndarray::array![0.0_f64, 0.0_f64];
    let mut st = seed;
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let mut records = Vec::with_capacity(n);
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        let x1 = a * radius;
        let x2 = b * radius;
        let pt = ndarray::array![x1, x2];
        let d = manifold
            .distance(pt.view(), origin.view())
            .expect("in-chart geodesic distance");
        let y = 2.0 * (-d).exp() - 1.0 + noise * next_gauss(&mut st);
        records.push(StringRecord::from(vec![
            y.to_string(),
            x1.to_string(),
            x2.to_string(),
        ]));
        filled += 1;
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode curved dataset")
}

fn fit_kappa_hat(kappa_star: f64, seed: u64) -> f64 {
    let data = curved_dataset(kappa_star, seed);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ curv(x1, x2, centers=10)", &data, &config)
        .expect("curv formula fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };
    get_constant_curvature_kappa(&fit.resolvedspec, 0)
        .expect("fitted spec must carry a constant-curvature κ̂")
}

#[test]
fn curv_full_fit_sign_diagnostic() {
    init_parallelism();

    let kappa_spherical = fit_kappa_hat(2.0, 0x5151_0001);
    let kappa_hyperbolic = fit_kappa_hat(-2.0, 0x5151_0003);

    eprintln!(
        "[#1464-diag] spherical  (planted kappa* = +2.0): final kappa_hat = {kappa_spherical}"
    );
    eprintln!(
        "[#1464-diag] hyperbolic (planted kappa* = -2.0): final kappa_hat = {kappa_hyperbolic}"
    );
    eprintln!(
        "[#1464-diag] separation |spherical - hyperbolic| = {}",
        (kappa_spherical - kappa_hyperbolic).abs()
    );
    eprintln!(
        "[#1464-diag] signs: spherical positive = {}, hyperbolic negative = {}",
        kappa_spherical > 0.0,
        kappa_hyperbolic < 0.0
    );
}
