//! Regression for #1427: a factor-level `by=` wrapper must center its inner
//! smooth EXACTLY ONCE (against the gated level indicator), not twice (pooled
//! sum-to-zero in the inner B-spline build PLUS the level-indicator centering
//! in the global identifiability pass).
//!
//! The double centering imposed two generically-independent constraints — the
//! pooled column moment `m = Σ_h m_h` and the per-level moment `m_g` — so a raw
//! `k`-column B-spline collapsed to `k-2` columns per level instead of `k-1`,
//! deleting one genuine nonconstant spline direction before REML ever runs. The
//! group main effect carries only the constant and cannot restore it, so the
//! joint fit is bias-floored regardless of λ.
//!
//! This test uses INDEPENDENT per-group `x` support, where `m_g` is not
//! proportional to `m` and the lost column is unambiguous (on a shared identical
//! grid the two constraints can be numerically redundant and mask the bug).

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const K: usize = 10;

fn three_group_dataset(seed: u64, n_per_group: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(3 * n_per_group);
    for (label, truth) in [("a", 0u8), ("b", 1u8), ("c", 2u8)] {
        for _ in 0..n_per_group {
            // Independent x draws per group => m_g not proportional to pooled m.
            let x: f64 = unit.sample(&mut rng);
            let f = match truth {
                0 => (2.0 * std::f64::consts::PI * x).sin(),
                1 => (2.0 * std::f64::consts::PI * x).cos(),
                _ => 4.0 * (x - 0.5) * (x - 0.5),
            };
            let y = f + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![
                x.to_string(),
                label.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(
        ["x", "g", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode")
}

/// Total design column count = length of the fitted coefficient vector
/// (representation-agnostic: works whether the design is stored dense or sparse).
fn ncols(fit: &FitResult) -> usize {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a Standard Gaussian fit");
    };
    std_fit.fit.beta_flat().len()
}

#[test]
fn factor_by_level_keeps_k_minus_one_smooth_columns() {
    init_parallelism();
    let data = three_group_dataset(1427, 200);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // `y ~ s(x, by=g)` auto-adds a treatment-coded factor main effect, so the
    // non-smooth part is `intercept + (L-1) contrasts = L` columns, and each of
    // the L level blocks contributes its centered smooth. Hence
    //   total = L + L*(smooth columns per level).
    // Correct centering (one constraint) gives K-1 smooth columns per level, so
    //   total = L + L*(K-1) = L*K.
    // The #1427 double-centering bug gives K-2 per level, i.e. total = L*(K-1).
    const L: usize = 3;
    let byfit = fit_from_formula(&format!("y ~ s(x, by=g, k={K})"), &data, &cfg)
        .expect("by-factor fit ok");
    let total = ncols(&byfit);

    let non_smooth = L; // intercept + (L-1) treatment contrasts
    let smooth_cols = total
        .checked_sub(non_smooth)
        .expect("by-factor design has fewer columns than the factor main effect");
    assert_eq!(
        smooth_cols % L,
        0,
        "the {L} level blocks should contribute equally; got {smooth_cols} smooth columns \
         (total={total})"
    );
    let per_level = smooth_cols / L;

    assert_eq!(
        per_level,
        K - 1,
        "factor-by level block must keep k-1={} smooth columns, got {per_level} \
         (double-centering bug gives k-2={}); total={total} (fixed=L*K={}, buggy=L*(K-1)={})",
        K - 1,
        K - 2,
        L * K,
        L * (K - 1),
    );
}
