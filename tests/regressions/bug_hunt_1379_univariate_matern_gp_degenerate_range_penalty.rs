//! #1379 — univariate `matern(x)` / `s(x, bs="gp")` deterministically aborted at
//! n=200 on >50% of ordinary 1-D datasets with
//! "range penalty block contains non-finite entries (max finite magnitude
//! 0.000e0)".
//!
//! Root cause: during the REML / spatial-κ optimization the per-penalty
//! smoothing weight `λ_k = exp(ρ_k)` of the redundant Matérn *stiffness*
//! operator is driven past `ρ ≈ 709` (the Matérn kernel already controls the
//! smoothness that operator also penalizes, so REML wants `λ_stiffness → ∞`).
//! `exp(ρ)` then overflows to `+∞`; the range penalty block assembled as
//! `Σ_k λ_k S_k` hits `∞ · 0 = NaN` wherever a transformed `S_k` entry is `0.0`,
//! so the whole block comes back non-finite and the eigensolve aborts — and the
//! final fit-result validation rejects the non-finite stored λ outright.
//! n=400/800 happened to avoid the overflowing λ; `bs="cr"/"ps"/"tp"` and
//! `duchon(x)` were never affected.
//!
//! Fix: finite-ceiling λ wherever ρ is exponentiated into a penalty weight (the
//! inner PIRLS eval, the reparam range-block assembly, and the stored fit
//! result), so a fully-smoothed direction is `λ ≤ ~1e304` instead of `+∞`. The
//! direction is pinned exactly as hard for every finite-arithmetic consumer
//! while `λ · 0 = 0`, so the block stays a well-formed PSD matrix and the result
//! validates. Ordinary finite λ are untouched.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn build_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = ux.sample(&mut rng);
            let truth = (2.0 * std::f64::consts::PI * x).sin();
            let y = truth + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn univariate_matern_smooth_fits_ordinary_1d_data_n200() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // Several seeds at n=200 — before the fix the κ optimizer drove a redundant
    // operator's λ to +∞ and the range-penalty eigensolve aborted on a subset of
    // these. They must all fit now.
    for seed in [3u64, 4, 6, 7, 11, 13] {
        let data = build_dataset(200, seed);
        fit_from_formula("y ~ matern(x)", &data, &cfg).unwrap_or_else(|e| {
            panic!("matern(x) failed to fit ordinary 200-row 1-D data at seed {seed}: {e}")
        });
    }
}
