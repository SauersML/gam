//! Rolled-sheet recovery regression for the intrinsic-metric seeder (#2240/#2280).
//!
//! The thesis (validated in numpy by the geometry owner): on a FOLDED manifold —
//! a swiss roll, a flat 2-D sheet rolled up in 3-D so that geodesically-distant
//! points are ambient-close — a LINEAR (PCA) seed keeps the two highest-variance
//! ambient directions and drops the axis the fold hides, so a flexible decoder
//! reading the 2-D PCA chart cannot reconstruct the manifold's third ambient
//! coordinate (its held-out R² caps at the retained variance fraction). The
//! geodesic (Isomap) seed UNROLLS the sheet into a faithful 2-D chart that carries
//! every intrinsic coordinate, so the same decoder reconstructs the full ambient
//! image (held-out R² → 1). On a NON-fold the two seeds are equivalent charts and
//! tie.
//!
//! The reconstruction proxy is a k-nearest-neighbor-in-latent regressor: predict a
//! held-out row's ambient image from the mean ambient image of its `k` nearest
//! neighbors IN THE SEED CHART. This measures exactly the property that matters —
//! whether the chart is a single-valued parameterization of the manifold (latent
//! neighbors are manifold neighbors). It needs no fit pipeline and is fully
//! deterministic. Everything here is RNG-free: a fixed lattice, index-strided
//! held-out split, index-tie-broken neighbor search.

use super::*;
use ndarray::{Array2, ArrayView2};

/// Deterministic swiss roll on a `n_t × n_h` lattice: a flat `(arclength, height)`
/// sheet rolled into 3-D as `(t·cos t, height, t·sin t)` over `turns` revolutions.
/// The height axis carries a MINORITY of the ambient variance, so a PCA-2 seed
/// (which keeps the two roll-plane directions) drops it — the fold pathology.
fn swiss_roll(n_t: usize, n_h: usize, turns: f64, height: f64) -> Array2<f64> {
    let n = n_t * n_h;
    let mut z = Array2::<f64>::zeros((n, 3));
    let t0 = 1.5 * std::f64::consts::PI;
    let t_span = turns * std::f64::consts::TAU;
    for i in 0..n_t {
        let t = t0 + t_span * (i as f64) / ((n_t - 1) as f64);
        for j in 0..n_h {
            let h = height * (j as f64) / ((n_h - 1) as f64);
            let row = i * n_h + j;
            z[[row, 0]] = t * t.cos();
            z[[row, 1]] = h;
            z[[row, 2]] = t * t.sin();
        }
    }
    z
}

/// A gentle (nearly flat) sheet: `(u, v, curvature·sin u)` — a genuine 2-D chart
/// with mild ambient curvature, NOT folded. Both PCA and intrinsic seeds recover
/// it, so their reconstruction R² ties.
fn gentle_sheet(n_u: usize, n_v: usize, curvature: f64) -> Array2<f64> {
    let n = n_u * n_v;
    let mut z = Array2::<f64>::zeros((n, 3));
    for i in 0..n_u {
        let u = 2.0 * (i as f64) / ((n_u - 1) as f64) - 1.0;
        for j in 0..n_v {
            let v = 2.0 * (j as f64) / ((n_v - 1) as f64) - 1.0;
            let row = i * n_v + j;
            z[[row, 0]] = u;
            z[[row, 1]] = v;
            z[[row, 2]] = curvature * (std::f64::consts::PI * u).sin();
        }
    }
    z
}

/// Held-out reconstruction R² of the ambient image from a 2-D latent chart, via a
/// deterministic kNN-in-latent regressor. Rows with `i % 5 == 0` are held out;
/// each held-out row is predicted by the mean ambient image of its `k` nearest
/// TRAIN rows in the chart (ties broken by ascending index). R² is over the
/// held-out rows, pooled across ambient channels.
fn knn_latent_holdout_r2(coords: ArrayView2<'_, f64>, ambient: ArrayView2<'_, f64>, k: usize) -> f64 {
    let n = ambient.nrows();
    let p = ambient.ncols();
    let train: Vec<usize> = (0..n).filter(|i| i % 5 != 0).collect();
    let test: Vec<usize> = (0..n).filter(|i| i % 5 == 0).collect();
    let kk = k.min(train.len()).max(1);

    // Held-out ambient mean (per channel) for SS_tot.
    let mut mean = vec![0.0_f64; p];
    for &t in &test {
        for c in 0..p {
            mean[c] += ambient[[t, c]];
        }
    }
    for m in mean.iter_mut() {
        *m /= test.len() as f64;
    }

    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for &t in &test {
        // k nearest train rows in the latent chart (ascending distance, then index).
        let mut d: Vec<(f64, usize)> = train
            .iter()
            .map(|&tr| {
                let mut acc = 0.0;
                for c in 0..coords.ncols() {
                    let diff = coords[[t, c]] - coords[[tr, c]];
                    acc += diff * diff;
                }
                (acc, tr)
            })
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for c in 0..p {
            let mut pred = 0.0;
            for &(_, tr) in d.iter().take(kk) {
                pred += ambient[[tr, c]];
            }
            pred /= kk as f64;
            let truth = ambient[[t, c]];
            ss_res += (truth - pred) * (truth - pred);
            ss_tot += (truth - mean[c]) * (truth - mean[c]);
        }
    }
    if ss_tot <= 0.0 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Extract an atom's `(n, 2)` latent chart from a `(1, n, d_max)` seed array.
fn chart_of(seed: &ndarray::Array3<f64>) -> Array2<f64> {
    let n = seed.shape()[1];
    let mut out = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        out[[row, 0]] = seed[[0, row, 0]];
        out[[row, 1]] = seed[[0, row, 1]];
    }
    out
}

/// ROLLED-SHEET REGRESSION: on a 3-turn swiss roll the intrinsic (geodesic) seed
/// unrolls the fold into a faithful 2-D chart whose held-out reconstruction R²
/// clears 0.99, while the PCA-2 seed drops the fold-hidden height axis and caps
/// well below it. The gap is the whole point of the seeder.
#[test]
fn swiss_roll_intrinsic_seed_reconstructs_where_pca_folds() {
    let z = swiss_roll(40, 14, 3.0, 13.0);
    let kinds = vec![SaeAtomBasisKind::Linear];
    let dims = vec![2usize];

    let intrinsic = sae_intrinsic_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
    let pca = sae_pca_seed_initial_coords(z.view(), &kinds, &dims).unwrap();

    let r2_intrinsic = knn_latent_holdout_r2(chart_of(&intrinsic).view(), z.view(), 6);
    let r2_pca = knn_latent_holdout_r2(chart_of(&pca).view(), z.view(), 6);

    assert!(
        r2_intrinsic >= 0.99,
        "intrinsic geodesic seed must reconstruct the rolled sheet (held-out R² = {r2_intrinsic:.4}, need >= 0.99)"
    );
    assert!(
        r2_pca <= 0.95,
        "PCA-2 seed must cap below the intrinsic seed on a fold (it drops the \
         fold-hidden axis); held-out R² = {r2_pca:.4}, expected <= 0.95"
    );
    assert!(
        r2_intrinsic - r2_pca >= 0.04,
        "intrinsic seed must clearly beat PCA on the fold: intrinsic {r2_intrinsic:.4} vs PCA {r2_pca:.4}"
    );
}

/// PARITY ON A NON-FOLD: on a gentle (unfolded) sheet the intrinsic and PCA seeds
/// are equivalent charts — both reconstruct with high held-out R² and tie within a
/// small band. The intrinsic seed must not REGRESS the easy case it is not needed
/// for (the race would pick either; here we assert both charts' quality directly).
#[test]
fn gentle_sheet_intrinsic_and_pca_seeds_tie() {
    let z = gentle_sheet(28, 20, 0.15);
    let kinds = vec![SaeAtomBasisKind::Linear];
    let dims = vec![2usize];

    let intrinsic = sae_intrinsic_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
    let pca = sae_pca_seed_initial_coords(z.view(), &kinds, &dims).unwrap();

    let r2_intrinsic = knn_latent_holdout_r2(chart_of(&intrinsic).view(), z.view(), 6);
    let r2_pca = knn_latent_holdout_r2(chart_of(&pca).view(), z.view(), 6);

    assert!(
        r2_intrinsic >= 0.95 && r2_pca >= 0.95,
        "both seeds must reconstruct a gentle sheet well (intrinsic {r2_intrinsic:.4}, PCA {r2_pca:.4})"
    );
    assert!(
        (r2_intrinsic - r2_pca).abs() <= 0.05,
        "on a non-fold the intrinsic and PCA seeds must tie (intrinsic {r2_intrinsic:.4}, PCA {r2_pca:.4})"
    );
}

/// The rolled-sheet seed is deterministic run-to-run at the full Array3 contract
/// (the module core has its own bit-identity test; this pins the production entry
/// the seed race consumes).
#[test]
fn swiss_roll_intrinsic_seed_is_deterministic() {
    let z = swiss_roll(30, 12, 3.0, 13.0);
    let kinds = vec![SaeAtomBasisKind::Linear];
    let dims = vec![2usize];
    let a = sae_intrinsic_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
    let b = sae_intrinsic_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
    assert_eq!(a, b, "intrinsic swiss-roll seed must be bit-identical run-to-run");
}
