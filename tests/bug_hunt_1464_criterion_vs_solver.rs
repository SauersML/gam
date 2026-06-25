//! #1464 decisive experiment: evaluate the EXACT production fixed-κ
//! profiled-REML criterion at +κ vs −κ for the genuinely HYPERBOLIC dataset.
//!
//! This settles solver-vs-criterion. The full fit rails κ̂ to ~+1.08 (the +chart
//! bound) for BOTH the spherical and hyperbolic mirror datasets. Either:
//!   (A) the criterion itself prefers the collapsed +κ corner — `V_p(+κ) < V_p(−κ)`
//!       for hyperbolic data — in which case the bug is in the constant-curvature
//!       REML/Occam term (same CLASS as #1426's ½log|S|₊ inversion), NOT the
//!       optimiser; or
//!   (B) `V_p(−κ) < V_p(+κ)` for hyperbolic data, yet the full fit returns +κ — in
//!       which case the criterion is sign-correct and the bug is the solver/readback.
//!
//! `constant_curvature_profiled_reml_scores` calls the SAME
//! `fixed_kappa_profiled_reml_score` the production joint-fit κ-sign scan uses, on
//! the data/spec/family/options materialised exactly like the full fit. So this is
//! the production criterion, not a re-derivation.
//!
//! Diagnostic only (plain `eprintln!`, no `{:?}`). It asserts nothing about which
//! way the answer falls — it PRINTS the scores so the maintainer reads the verdict.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::{
    FitConfig, constant_curvature_profiled_reml_scores, encode_recordswith_inferred_schema,
    init_parallelism,
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

/// Identical generator to the #1464 contract test, so the criterion is probed on
/// exactly the dataset the full fit rails on.
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

fn print_scores(label: &str, kappa_star: f64, seed: u64) -> f64 {
    let data = curved_dataset(kappa_star, seed);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // Probe symmetric κ corners: ±chart-bound (~±1.08 at radius 0.68), the planted
    // truth ±2 (clamped into the chart by the production fit), an interior ±0.5,
    // and flat 0. Lower V_p = preferred (negative-log-evidence the outer loop
    // minimises).
    let kappas = [-2.0_f64, -1.08, -0.5, 0.0, 0.5, 1.08, 2.0];
    let scores = constant_curvature_profiled_reml_scores(
        "y ~ curv(x1, x2, centers=10)",
        &data,
        &config,
        &kappas,
    )
    .expect("fixed-κ profiled-REML scan should succeed");

    eprintln!("[#1464-crit] === {label} (planted kappa* = {kappa_star:+}) ===");
    let mut best_k = f64::NAN;
    let mut best_v = f64::INFINITY;
    for (k, v) in &scores {
        eprintln!("[#1464-crit]   V_p(kappa={k:+.4}) = {v}");
        if *v < best_v {
            best_v = *v;
            best_k = *k;
        }
    }
    eprintln!(
        "[#1464-crit]   --> argmin V_p over probed grid: kappa = {best_k:+.4} (V_p = {best_v})"
    );
    eprintln!(
        "[#1464-crit]   verdict: criterion prefers {} curvature for this {label} dataset",
        if best_k < 0.0 {
            "NEGATIVE (hyperbolic)"
        } else if best_k > 0.0 {
            "POSITIVE (spherical)"
        } else {
            "FLAT"
        }
    );
    best_k
}

#[test]
fn curv_criterion_prefers_which_sign_per_mirror_dataset() {
    init_parallelism();
    // The headline: does the production criterion prefer +κ or −κ for genuinely
    // hyperbolic data? If +κ wins, the bug is the criterion (Occam term), not the
    // solver.
    let hyp_k = print_scores("HYPERBOLIC", -2.0, 0x5151_0003);
    // Control: the spherical mirror must prefer +κ.
    let sph_k = print_scores("SPHERICAL", 2.0, 0x5151_0001);
    // Correct contract: the production criterion must prefer NEGATIVE curvature for
    // genuinely hyperbolic data and POSITIVE for spherical. If this assertion fails
    // (e.g. hyp_k >= 0), the criterion is sign-blind — that IS the #1464 bug, and the
    // two V_p grids printed above are the diagnostic payload pinpointing it.
    assert!(
        hyp_k < 0.0 && sph_k > 0.0,
        "constant-curvature REML criterion is sign-blind: hyperbolic argmin kappa = {hyp_k:+.4} (want < 0), spherical argmin kappa = {sph_k:+.4} (want > 0)"
    );
}
