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
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;
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

// ---------------------------------------------------------------------------
// Root-1 OBJECTIVE arm: the issue is not "are there N blocks" but "does joint
// `s(x, by=g)` recover each group's curve as well as fitting that group ALONE".
// The structural test above proves independent λ COORDINATES exist; this test
// proves the joint REML actually USES them to smooth each group independently,
// by measuring per-group truth recovery against the independent per-group
// `s(x)` baseline the issue compares to — at n=300 AND n=2000, because the
// shared-λ defect GREW with n (3× at n=300, 16× at n=2000). A shared λ forces
// one global smoothness, so the smooth group is over-fit and the wiggly group
// is over-smoothed; independent per-level λ must close that gap and KEEP it
// closed as n grows.
// ---------------------------------------------------------------------------

/// Two groups whose OPTIMAL smoothing parameters are sharply different, so a
/// single shared λ cannot serve both: group `a` is essentially linear (wants a
/// large λ / very smooth fit), group `b` is a high-frequency sinusoid (wants a
/// small λ / wiggly fit). With one shared λ the REML must compromise and at
/// least one group is mis-smoothed; independent per-level λ lets each group
/// pick its own.
fn truth(level: usize, x: f64) -> f64 {
    use std::f64::consts::PI;
    match level {
        0 => 0.7 * x - 0.3,        // group a: near-linear → wants high λ
        1 => (5.0 * PI * x).sin(), // group b: fast sinusoid → wants low λ
        _ => unreachable!(),
    }
}

const LABELS: [&str; 2] = ["a", "b"];

/// Build a two-group dataset with `n_per_group` rows per group, each group on
/// its OWN independent `x` draws and its own true curve `truth(level, x)`.
fn recovery_dataset(seed: u64, n_per_group: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.15).expect("normal");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(LABELS.len() * n_per_group);
    for (g, label) in LABELS.iter().enumerate() {
        for _ in 0..n_per_group {
            let x: f64 = unit.sample(&mut rng);
            let y = truth(g, x) + noise.sample(&mut rng);
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

/// Predict a Standard Gaussian fit on a fresh single-feature grid via the
/// frozen resolved spec (identity link ⇒ η == μ).
fn predict_standard(fit: &StandardFitResult, n_cols: usize, x_idx: usize, grid: &[f64]) -> Vec<f64> {
    let mut g = Array2::<f64>::zeros((grid.len(), n_cols));
    for (r, &xj) in grid.iter().enumerate() {
        g[[r, x_idx]] = xj;
    }
    build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild standalone design at grid")
        .design
        .apply(&fit.fit.beta)
        .to_vec()
}

/// Predict ONE level of a by-factor fit on a fresh grid (set the factor column
/// to that level's exact encoded bits, the `x` column to the grid).
fn predict_byfactor_level(
    fit: &StandardFitResult,
    n_cols: usize,
    x_idx: usize,
    g_idx: usize,
    lvl_bits: f64,
    grid: &[f64],
) -> Vec<f64> {
    let mut g = Array2::<f64>::zeros((grid.len(), n_cols));
    for (r, &xj) in grid.iter().enumerate() {
        g[[r, x_idx]] = xj;
        g[[r, g_idx]] = lvl_bits;
    }
    build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild by-factor design on a fresh single-level grid")
        .design
        .apply(&fit.fit.beta)
        .to_vec()
}

fn rmse_to_truth(pred: &[f64], level: usize, grid: &[f64]) -> f64 {
    let sse: f64 = pred
        .iter()
        .zip(grid)
        .map(|(&p, &x)| {
            let d = p - truth(level, x);
            d * d
        })
        .sum();
    (sse / grid.len() as f64).sqrt()
}

/// At one sample size: fit each group's `s(x)` ALONE (the issue's reference),
/// fit the joint `s(x, by=g)`, and return, per group, the joint RMSE and the
/// independent RMSE on a shared fresh grid. The joint fit must recover each
/// group nearly as well as its own standalone fit; the shared-λ defect made the
/// joint fit much worse, increasingly so with n.
fn joint_vs_independent_rmse(seed: u64, n_per_group: usize) -> [(f64, f64); 2] {
    let cfg = gaussian_cfg();

    // --- joint by-factor fit ---
    let ds = recovery_dataset(seed, n_per_group);
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];
    let n_cols = ds.headers.len();
    // Rows are emitted group-by-group, so level k's encoded value is the
    // g-column of any row in its block; read it from the first such row.
    let lvl_bits: Vec<f64> = (0..LABELS.len())
        .map(|k| ds.values[[k * n_per_group, g_idx]])
        .collect();

    let joint = match fit_from_formula(&format!("y ~ s(x, by=g, k={K})"), &ds, &cfg)
        .expect("joint by-factor fit ok")
    {
        FitResult::Standard(f) => f,
        _ => panic!("expected Standard joint fit"),
    };

    // --- independent per-group fits ---
    const N_GRID: usize = 200;
    let grid: Vec<f64> = (0..N_GRID)
        .map(|j| j as f64 / (N_GRID as f64 - 1.0))
        .collect();

    let mut out = [(0.0_f64, 0.0_f64); 2];
    for level in 0..LABELS.len() {
        // Independent fit: this group's data alone, plain s(x).
        let solo = recovery_dataset_single(seed.wrapping_add(level as u64 + 1), n_per_group, level);
        let solo_col = solo.column_map();
        let solo_x = solo_col["x"];
        let solo_fit = match fit_from_formula(&format!("y ~ s(x, k={K})"), &solo, &cfg)
            .expect("standalone per-group fit ok")
        {
            FitResult::Standard(f) => f,
            // single-penalty 1-D PS scan may route differently; force a Standard
            // fit by keeping the default basis (cr/tp → Standard).
            _ => panic!("expected Standard standalone fit"),
        };
        let indep_pred = predict_standard(&solo_fit, solo.headers.len(), solo_x, &grid);
        let indep_rmse = rmse_to_truth(&indep_pred, level, &grid);

        let joint_pred =
            predict_byfactor_level(&joint, n_cols, x_idx, g_idx, lvl_bits[level], &grid);
        // Joint prediction carries the global intercept; remove the per-curve
        // mean offset relative to truth so we measure SHAPE/smoothing recovery,
        // matching how the independent fit (its own intercept) is scored.
        let joint_rmse = rmse_to_truth_centered(&joint_pred, level, &grid);

        out[level] = (joint_rmse, indep_rmse);
    }
    out
}

/// One group's data alone (level `which`), with column layout `x, y`.
fn recovery_dataset_single(seed: u64, n: usize, which: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.15).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let y = truth(which, x) + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(["x", "y"].into_iter().map(String::from).collect(), rows)
        .expect("encode single")
}

/// RMSE to truth after removing the best constant offset (the global-intercept
/// reference the by-factor design shares across groups), so the joint and the
/// independent fits are scored on the same shape/baseline footing.
fn rmse_to_truth_centered(pred: &[f64], level: usize, grid: &[f64]) -> f64 {
    let gap: f64 = pred
        .iter()
        .zip(grid)
        .map(|(&p, &x)| p - truth(level, x))
        .sum::<f64>()
        / grid.len() as f64;
    let sse: f64 = pred
        .iter()
        .zip(grid)
        .map(|(&p, &x)| {
            let d = (p - gap) - truth(level, x);
            d * d
        })
        .sum();
    (sse / grid.len() as f64).sqrt()
}

/// #1427 (root 1, OBJECTIVE): joint `s(x, by=g)` must recover each group nearly
/// as well as the independent per-group `s(x)` fit the issue benchmarks against
/// — at n=300 AND n=2000. The shared-λ defect made the joint fit 3× worse at
/// n=300 and 16× worse at n=2000 because one global λ over-smoothed the wiggly
/// group while over-fitting the smooth one. Independent per-level λ closes that
/// gap and, crucially, KEEPS it closed as n grows.
#[test]
fn factor_by_recovers_per_group_like_independent_fits_1427() {
    init_parallelism();

    // n=300 total (150/group) and n=2000 total (1000/group): the two sample
    // sizes the issue reports, where the gap was 3× and 16× respectively.
    let small = joint_vs_independent_rmse(424_242, 150);
    let large = joint_vs_independent_rmse(909_090, 1000);

    // The decisive bound: the joint by-factor per-group RMSE must stay within a
    // small multiple of the independent per-group RMSE. The shared-λ bug blew
    // this ratio to 3×/16×; independent per-level λ keeps it modest. We allow a
    // generous 2.5× headroom (the joint fit also estimates a shared error scale
    // + per-group baselines, so it is not expected to be byte-identical to the
    // solo fits) — far below the 3×/16× shared-λ signature.
    const RATIO_BAR: f64 = 2.5;
    for (tag, sizes) in [("n=300", &small), ("n=2000", &large)] {
        for (level, &(joint_rmse, indep_rmse)) in sizes.iter().enumerate() {
            // Independent fit must itself recover the truth (sanity floor): if
            // the baseline is garbage the ratio is meaningless.
            assert!(
                indep_rmse < 0.5 && indep_rmse.is_finite(),
                "{tag} grp{level}: independent per-group s(x) baseline failed to recover \
                 truth (RMSE={indep_rmse:.4}); cannot compare"
            );
            let ratio = joint_rmse / indep_rmse.max(1e-6);
            assert!(
                ratio < RATIO_BAR,
                "{tag} grp{level}: joint s(x, by=g) under-recovers vs the independent \
                 per-group s(x) fit (joint RMSE={joint_rmse:.4}, independent RMSE={indep_rmse:.4}, \
                 ratio={ratio:.2} >= {RATIO_BAR}). The shared-λ defect (#1427) forces one global \
                 smoothness across groups with different optimal λ; independent per-level λ must \
                 keep the joint fit close to the per-group fits."
            );
        }
    }

    // The defect GREW with n: the worst-group ratio at n=2000 must NOT be much
    // larger than at n=300. (Shared-λ went 3×→16×; independent per-level λ must
    // be roughly flat in n.) Compare the worst per-group ratio at each size.
    let worst = |s: &[(f64, f64); 2]| {
        s.iter()
            .map(|&(j, i)| j / i.max(1e-6))
            .fold(0.0_f64, f64::max)
    };
    let worst_small = worst(&small);
    let worst_large = worst(&large);
    assert!(
        worst_large < worst_small * 1.8 + 0.5,
        "the joint-vs-independent recovery gap must NOT blow up with n (the shared-λ \
         signature went 3× at n=300 to 16× at n=2000): worst ratio n=300={worst_small:.2}, \
         n=2000={worst_large:.2}"
    );
}
