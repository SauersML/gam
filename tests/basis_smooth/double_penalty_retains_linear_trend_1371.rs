//! #1371 GUARD: default `s(x)` (bs=ps, `double_penalty = True` — the mgcv
//! `select = TRUE` analogue) must NOT annihilate a GENUINE strong linear trend.
//!
//! This is the DUAL of #1266. #1266 guards that a double-penalty smooth on an
//! UNSUPPORTED term shrinks OUT (EDF → 0). #1371 guards the opposite failure on a
//! SUPPORTED null-space direction: on clean `y = 2 + 3x + N(0, σ)` the
//! Marra & Wood null-space shrinkage ridge `Z Zᵀ` must NOT be driven to
//! `λ_nullspace → ∞`, which would force the real slope to EXACTLY 0 (`edf ≈ 1`,
//! RMSE ≈ 0.87) and report a strong effect as non-significant.
//!
//! Root cause this guards against (gam#1371): the outer REML loop FALSE-CONVERGES
//! on the high-λ_nullspace shelf. The continuation pre-warm starts from an
//! over-smoothing ρ₀ that pushes the null-space coordinate up; there the
//! null-space coefficients are already shrunk to 0, so the analytic ρ-gradient
//! vanishes (∂deviance/∂ρ_null → 0 and the Occam terms cancel) and the optimizer
//! certifies a stationary point whose REML cost is FAR WORSE than the basin it
//! seeded from (which retains the trend). The fit must instead keep the
//! lower-cost basin (slope ≈ 3, EDF ≈ 2).
//!
//! Path: this exercises the FORMULA path (`fit_from_formula`), which is the one
//! that drives the continuation pre-warm / outer-seed machinery the bug lives in
//! — the lower-level term-collection path in
//! `double_penalty_edf_inflation_repro_1266.rs` does not reproduce it.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// The #1371 DGP: `y = 2 + 3x + N(0, σ)` on uniform `x ∈ [0,1]`. The slope lives
/// entirely in the `{1, x}` null space of the 2nd-difference bending penalty, so
/// the null-space shrinkage ridge can annihilate it if `λ_nullspace → ∞`.
fn linear_trend_dataset(seed: u64, n: usize, sd: f64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sd).expect("normal");
    let mut xs = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            xs.push(x);
            let y = 2.0 + 3.0 * x + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(
        ["x", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode");
    (ds, xs)
}

/// Slope of the fitted mean via cov(x, μ)/var(x), μ = Xβ̂.
fn fitted_slope(fit: &FitResult, xs: &[f64]) -> (f64, f64) {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected Standard fit (double_penalty s(x) must stay on the dense path)");
    };
    let mu = std_fit.design.design.dot(&std_fit.fit.beta);
    let n = xs.len() as f64;
    let xbar = xs.iter().sum::<f64>() / n;
    let mbar = mu.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (xi, mi) in xs.iter().zip(mu.iter()) {
        sxy += (xi - xbar) * (mi - mbar);
        sxx += (xi - xbar) * (xi - xbar);
    }
    (sxy / sxx, std_fit.fit.edf_total().unwrap_or(f64::NAN))
}

#[test]
fn double_penalty_retains_linear_trend_1371() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // Pre-fix this collapsed on ~5/16 of these seeds at σ=0.15 (slope EXACTLY 0,
    // EDF = 1). The fix must restore slope ≈ 3 on EVERY seed.
    let n = 800usize;
    let sd = 0.15_f64;
    let mut collapses = 0usize;
    let mut worst_slope_err = 0.0_f64;
    for seed in 7000u64..7016 {
        let (data, xs) = linear_trend_dataset(seed, n, sd);
        let fit = fit_from_formula("y ~ s(x)", &data, &cfg).expect("gam fit");
        let (slope, edf) = fitted_slope(&fit, &xs);
        eprintln!("[1371] seed={seed} slope={slope:.4} edf={edf:.4}");
        if slope.abs() < 0.5 {
            collapses += 1;
        }
        worst_slope_err = worst_slope_err.max((slope - 3.0).abs());
    }
    assert_eq!(
        collapses, 0,
        "#1371: default s(x) double-penalty ANNIHILATED a genuine slope=3 linear trend on \
         {collapses}/16 seeds (slope ≈ 0, EDF ≈ 1) — the null-space shrinkage ridge ran to \
         λ_nullspace → ∞ on a DATA-SUPPORTED null direction. A real strong effect must survive."
    );
    assert!(
        worst_slope_err < 0.3,
        "#1371: recovered slope drifted from the truth 3.0 by {worst_slope_err:.4} on some seed; \
         the double penalty must leave a genuinely-supported linear trend essentially unshrunk."
    );
}
