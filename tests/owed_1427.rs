//! Owed-work regression gate for issue #1427.
//!
//! #1427: `s(x, by=g)` (factor `by=`) joint REML under-recovered versus gamfit's
//! OWN independent per-group `s(x)` fits — ~3× worse at n=300, ~16× at n=2000,
//! and the gap GREW with data (a structural defect, not noise). The defect had
//! two coupled roots, both now fixed on `main`:
//!
//!   1. Shared-λ (commit 044fb7596): the `ByVarKind::Factor` arm of
//!      `build_by_smooth_local` (`term_specs.rs`) tiled ONE inner penalty across
//!      every level's diagonal block, so all factor levels shared a SINGLE
//!      smoothing parameter. The design is block-diagonal and block-separable,
//!      so a correct REML must reproduce the independent per-group fits — which
//!      one shared λ cannot when per-group smoothness is uneven. The fix emits
//!      `n_levels * n_penalties` INDEPENDENT penalty blocks (one λ coordinate per
//!      (level, inner-penalty) pair), restoring block-separable λ-selection.
//!
//!   2. Double-centering (k-2 columns): pinned separately by
//!      `tests/issue_1427_factor_by_single_centering.rs` (k-1 columns per level).
//!
//! This file pins ROOT 1: the per-level INDEPENDENT λ structure. The number of
//! λ coordinates for `s(x, by=g)` must scale with the number of factor levels —
//! it must equal `n_levels` times the λ count of a single standalone `s(x)` of
//! the same basis. The shared-λ regression emits exactly the single-smooth count
//! regardless of `n_levels`, which this test forbids. Deriving the per-smooth
//! count from a standalone fit keeps the assertion robust to `double_penalty`
//! (1 vs 2 penalties per spline): what is pinned is that the count GROWS with the
//! level count, the load-bearing block-separability property.
//!
//! Uses only the public crate API. INDEPENDENT per-group `x` support (matching
//! the issue's own repro) so the construction is genuinely block-separable and
//! the per-level λ's are distinct.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const K: usize = 10;

/// `labels`-many groups, each with its OWN independent `x` draws (so the per-
/// level column moments are not proportional to the pooled moment — the generic
/// block-separable case the issue specifies). Each group gets a distinct true
/// curve so its optimal smoothness genuinely differs — the case a single shared
/// λ cannot represent.
fn grouped_dataset(seed: u64, labels: &[&str], n_per_group: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(labels.len() * n_per_group);
    for (g, label) in labels.iter().enumerate() {
        for _ in 0..n_per_group {
            let x: f64 = unit.sample(&mut rng);
            let f = match g % 3 {
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

/// Number of smoothing-parameter (λ) coordinates of a fit — one per penalty
/// block in the joint penalized least-squares system.
fn n_lambdas(fit: &FitResult) -> usize {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a Standard Gaussian fit");
    };
    std_fit.fit.lambdas.len()
}

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// #1427 (root 1): `s(x, by=g)` must emit INDEPENDENT per-level penalties — the
/// λ-coordinate count scales with the number of factor levels. It must equal
/// `n_levels` × (the λ count of a single standalone `s(x)` of the same basis).
/// The shared-λ regression emits exactly the single-smooth count regardless of
/// `n_levels` (all groups forced onto one λ), which this test rejects.
#[test]
fn factor_by_emits_independent_per_level_lambda_1427() {
    init_parallelism();
    let cfg = gaussian_cfg();

    // Per-smooth λ count: fit a single standalone `s(x)` on one group's data.
    // (Robust to `double_penalty`: this is 1 or 2 depending on config, but the
    // by-factor count must be an exact multiple of it.)
    let one_group = grouped_dataset(1427, &["a"], 200);
    let single = fit_from_formula(&format!("y ~ s(x, k={K})"), &one_group, &cfg)
        .expect("standalone s(x) fit ok");
    let per_smooth_lambdas = n_lambdas(&single);
    assert!(
        per_smooth_lambdas >= 1,
        "a standalone s(x) must have at least one smoothing parameter"
    );

    // Now the by-factor smooths at L=2 and L=3 levels.
    let data2 = grouped_dataset(1427, &["a", "b"], 200);
    let data3 = grouped_dataset(1427, &["a", "b", "c"], 200);
    let by2 = fit_from_formula(&format!("y ~ s(x, by=g, k={K})"), &data2, &cfg)
        .expect("by-factor L=2 fit ok");
    let by3 = fit_from_formula(&format!("y ~ s(x, by=g, k={K})"), &data3, &cfg)
        .expect("by-factor L=3 fit ok");

    let n_lambda2 = n_lambdas(&by2);
    let n_lambda3 = n_lambdas(&by3);

    // The auto-added treatment-coded factor main effect is UNPENALIZED, so the
    // only λ coordinates come from the L level-blocks' smooths. Independent
    // per-level emission ⇒ count = per_smooth_lambdas * n_levels.
    assert_eq!(
        n_lambda2,
        per_smooth_lambdas * 2,
        "s(x, by=g) with 2 levels must emit per-level independent λ \
         ({per_smooth_lambdas}×2={}), not the shared-λ count {per_smooth_lambdas}; got {n_lambda2}",
        per_smooth_lambdas * 2,
    );
    assert_eq!(
        n_lambda3,
        per_smooth_lambdas * 3,
        "s(x, by=g) with 3 levels must emit per-level independent λ \
         ({per_smooth_lambdas}×3={}), not the shared-λ count {per_smooth_lambdas}; got {n_lambda3}",
        per_smooth_lambdas * 3,
    );

    // The decisive regression signature: the λ count GROWS with the level count.
    // The shared-λ bug would give n_lambda2 == n_lambda3 == per_smooth_lambdas.
    assert!(
        n_lambda3 > n_lambda2,
        "λ-coordinate count must grow with the number of factor levels \
         (independent per-level λ); shared-λ regression gives equal counts \
         (L=2 ⇒ {n_lambda2}, L=3 ⇒ {n_lambda3})"
    );
    assert_eq!(
        n_lambda3 - n_lambda2,
        per_smooth_lambdas,
        "adding one factor level must add exactly one level's worth of λ \
         coordinates ({per_smooth_lambdas}); got {}",
        n_lambda3 - n_lambda2,
    );
}
