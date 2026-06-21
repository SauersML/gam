//! #1464 regression: the constant-curvature `curv(...)` smooth, fitted through
//! the FULL `fit_from_formula` pipeline, must IDENTIFY THE SIGN of the true
//! curvature — a positive κ̂ for genuinely spherical data and a negative κ̂ for
//! genuinely hyperbolic data — instead of railing κ̂ to the positive chart bound
//! for every dataset (the reported bug: hyperbolic truth recovered as spherical,
//! with the SAME κ̂ returned for the mirror spherical/hyperbolic datasets).
//!
//! The evidence is a sign-symmetry argument that needs no absolute scale: two
//! datasets that are exact mirror images under the curvature sign — one spherical
//! (κ⋆ = +2), one hyperbolic (κ⋆ = −2) — are generated using the ENGINE'S OWN
//! geodesic-distance convention (`ConstantCurvature::distance`), so the planted
//! signal's geometry is gam's own truth, never another tool's output. A correct
//! estimator MUST distinguish them.
//!
//! κ̂ is read back from the FITTED resolved term spec (the same κ̂ that
//! `model.curvature()` surfaces), so this drives exactly the user-visible
//! full-fit path the issue reports on.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::smooth::get_constant_curvature_kappa;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

use csv::StringRecord;

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps -------
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

/// `n` chart points uniformly in a disk of radius `radius`, with a Gaussian
/// response built so the curvature is genuinely identifiable: the mean is a
/// smooth function of the `M_{κ⋆}` geodesic distance to the origin,
/// `μ = 2·exp(−d_{κ⋆}) − 1` (gam's OWN `ConstantCurvature::distance`), so the
/// distance-matrix SHAPE — hence the planted signal — depends sharply on κ⋆.
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

/// Fit `curv(x1, x2, centers=10)` through the full formula pipeline and return
/// the fitted curvature κ̂ read back from the resolved term spec.
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
    // The single smooth term is the constant-curvature term (index 0); κ̂ is the
    // outer-optimized curvature persisted in the resolved spec.
    get_constant_curvature_kappa(&fit.resolvedspec, 0)
        .expect("fitted spec must carry a constant-curvature κ̂")
}

// #1464 (fixed): the full `fit_from_formula` path used to rail κ̂ to the +chart
// bound (~+1.08 at radius 0.68) for BOTH the spherical and the hyperbolic mirror
// dataset, so the two signs were indistinguishable and hyperbolic truth was
// mis-recovered as spherical. Basis-level effective-length L(κ) reparam +
// `double_penalty:false` were already in place and the isolated profiled-REML
// criterion (`constant_curvature_recovers_curvature_sign_1404`) already
// identified the sign; the residual railing was a production inner-λ warm-seed
// basin defect on TWO paths: (1) the standard scalar-ρ inner-λ path anchors the
// criterion-ranked objective-grid prepass at the warm seed so the inner λ-solve
// jumps into the correct high-λ basin (adopt only on strict REML-cost
// improvement → healthy warm-started fits byte-identical) and reports the
// converged ρ instead of the warm seed when it is strictly cheaper; (2) the
// joint [ρ,ψ] spatial outer solver (the κ-optimized path this test drives) now
// widens the ρ upper bound to RHO_BOUND and seeds a high-ρ over-smoothing corner
// for constant-curvature terms so the joint ARC can reach the heavily-smoothed
// collapsed-kernel basin. With the per-κ REML cost matching the textbook
// profiled-REML, the curvature SIGN is identifiable again, so this contract
// PASSES.
#[test]
fn curv_full_fit_identifies_curvature_sign_on_mirror_datasets() {
    init_parallelism();

    // Control: genuinely spherical data must recover POSITIVE curvature.
    let kappa_spherical = fit_kappa_hat(2.0, 0x5151_0001);
    // The headline failure: genuinely hyperbolic data must recover NEGATIVE
    // curvature, NOT rail to the positive chart bound.
    let kappa_hyperbolic = fit_kappa_hat(-2.0, 0x5151_0003);

    eprintln!(
        "[#1464] full-fit κ̂: spherical(κ⋆=+2)={kappa_spherical:+.4}  hyperbolic(κ⋆=−2)={kappa_hyperbolic:+.4}"
    );

    // (a) Control — spherical truth recovers positive curvature.
    assert!(
        kappa_spherical > 0.0,
        "spherical truth (κ⋆=+2) must recover POSITIVE curvature through the full fit; got κ̂={kappa_spherical}"
    );

    // (b) The two mirror datasets must be GENUINELY DISTINGUISHED — not the
    // bit-identical κ̂ the bug returns for both signs.
    assert!(
        (kappa_spherical - kappa_hyperbolic).abs() > 0.1,
        "spherical and hyperbolic mirror datasets must yield materially DIFFERENT κ̂ \
         (the #1464 bug returns the same chart-bound value for both): \
         spherical κ̂={kappa_spherical}, hyperbolic κ̂={kappa_hyperbolic}"
    );

    // (c) The headline: hyperbolic truth recovers NEGATIVE curvature.
    assert!(
        kappa_hyperbolic < 0.0,
        "hyperbolic truth (κ⋆=−2) must recover NEGATIVE curvature through the full fit; \
         got κ̂={kappa_hyperbolic} (the bug rails this to the +chart bound, calling it spherical)"
    );
}
