//! Regression (#1654): a CONVEX shape-constrained univariate smooth parks in the
//! "linear corner" — collapses to a near-straight line through a clearly convex
//! truth — for a seed/basis-dimension–specific subset of fits at the default and
//! several explicit `k`. Convexity is still ENFORCED (zero second-difference
//! violations), so the constraint is not dropped; instead the convex-constrained
//! outer REML/λ selection lands on a degenerate operating point (the boundary of
//! the convex cone where every second divided difference is zero, i.e. an exactly
//! linear fit) and never escapes.
//!
//! ## What was wrong (double-penalty ridge broke the box-reparam congruence)
//!
//! A convex/concave shape constraint puts the smooth on the box-reparameterized
//! PIRLS path: the coefficients are written `β = Tγ` where `T` is the order-2
//! Greville-scaled second *divided*-difference transform
//! (`convex_divided_difference_transform_matrix`). A reparameterization must
//! leave the penalized REML fit invariant, which requires EVERY penalty block to
//! transform by the SAME congruence `S ↦ TᵀST`. The wiggliness penalty did; but
//! the double-penalty nullspace ridge (`PenaltySource::DoublePenaltyNullspace`,
//! which shrinks the unpenalized level/slope null space) was instead rebuilt
//! from scratch in `gam-terms::smooth::term_specs::build_single_local_smooth_term`
//! as the orthonormal null-space projector of `TᵀST` — a γ-space ridge whose
//! null face matches in subspace but is measured in the γ inner product, not the
//! congruence image of the β-space ridge. Because the two penalty blocks are then
//! independently Frobenius-normalized, the from-scratch projector silently
//! re-weights the level/slope shrinkage relative to the wiggliness penalty,
//! decoupling their scales. The distorted REML λ landscape drove the
//! curvature-constrained smooth into the flat linear corner (curvature ≈ 0,
//! EDF ≈ 1.5) for a seed/basis-dimension–specific subset of fits at the default
//! and several explicit `k`, even though an unconstrained `s(x)` on the same data
//! recovers the convex truth at EDF ≈ 4.
//!
//! The fix restores the exact congruence `Tᵀ R_β T` for the order-2 (curvature)
//! ridge so the box reparameterization stays a true invertible change of
//! coordinates and both penalty blocks live in one inner product. The order-1
//! (monotone) cumulative-sum transform keeps the from-scratch projector rebuild,
//! which the #509 over-smoothing fix introduced for its fast-growing
//! conditioning.
//!
//! ## The assertion
//!
//! On a strongly convex truth `f(x) = 4·(x − 0.5)²` (f'' = 8 > 0 everywhere),
//! a default-style convex-constrained smooth at a basis dimension that collapses
//! today (`k = 20`, seed 7 — one of the issue's named failing seeds) must NOT
//! return a degenerate near-linear fit: its fitted curve's second-difference
//! (curvature) energy on a dense grid must clear a floor that the collapsed
//! corner (curvature ≈ 0, a straight line) cannot, AND its truth-recovery RMSE
//! must sit well below the corner's ~0.30 and near the good ~0.05 interior fit.
//! Before the fix the fit collapses (rmse ≈ 0.31, curv_rms ≈ 0.15); after, it
//! recovers (rmse ≈ 0.05, curv_rms ≈ 6–8).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Deterministic convex-truth dataset. Mirrors the issue's generator exactly
/// (StdRng seed, x ~ U(0,1), y = 4·(x−0.5)² + N(0, sigma)); this `(seed, k)`
/// is one of the configurations that collapses to the linear corner pre-fix.
fn make_convex_data(seed: u64, n: usize, sigma: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let truth = |x: f64| 4.0 * (x - 0.5) * (x - 0.5);
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, sigma).expect("normal noise");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let fi = truth(xi);
        x.push(xi);
        y.push(fi + noise.sample(&mut rng));
        t.push(fi);
    }
    (x, y, t)
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let s: f64 = a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
    (s / a.len() as f64).sqrt()
}

#[test]
fn convex_shape_smooth_does_not_park_in_linear_corner_at_collapsing_k() {
    init_parallelism();

    // Seed 7 / k = 20 / sigma = 0.4 is a confirmed pre-fix collapse (one of the
    // issue's named failing seeds). n is kept small (100) so the single fit is
    // cheap while still exhibiting the degeneracy.
    let n = 100usize;
    let sigma = 0.4_f64;
    let (x, y, t) = make_convex_data(7, n, sigma);

    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode convex dataset");
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=20, shape=convex)", &ds, &cfg)
        .expect("gam convex smooth fit");
    let FitResult::Standard(fit) = result else {
        panic!("Gaussian convex smooth is a scalar family => expected FitResult::Standard");
    };

    // Fitted mean at the training points => truth-recovery RMSE.
    let mut train_grid = Array2::<f64>::zeros((n, width));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild convex design at training points");
    let fitted: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();
    let err = rmse(&fitted, &t);

    // Curvature energy of the fitted curve on a dense sorted grid. The collapsed
    // linear corner is a straight line (second differences ≈ 0); a genuine
    // convex recovery of f'' = 8 has large, strictly positive curvature.
    const M: usize = 200;
    let mut dense_grid = Array2::<f64>::zeros((M, width));
    for i in 0..M {
        dense_grid[[i, x_idx]] = i as f64 / (M as f64 - 1.0);
    }
    let dense_design = build_term_collection_design(dense_grid.view(), &fit.resolvedspec)
        .expect("rebuild convex design on dense grid");
    let dv: Vec<f64> = dense_design.design.apply(&fit.fit.beta).to_vec();
    let h = 1.0 / (M as f64 - 1.0);
    let curv_rms = ((1..M - 1)
        .map(|i| {
            let d2 = (dv[i + 1] - 2.0 * dv[i] + dv[i - 1]) / (h * h);
            d2 * d2
        })
        .sum::<f64>()
        / (M - 2) as f64)
        .sqrt();

    // Constraint must STILL be enforced: no concave dip on the fitted curve.
    let worst_violation = (1..M - 1)
        .map(|i| dv[i + 1] - 2.0 * dv[i] + dv[i - 1])
        .fold(f64::INFINITY, f64::min);
    let h2 = h * h;
    let convexity_eps = 0.01 * 8.0 * h2; // 1% of the true grid second difference (f''=8)
    assert!(
        worst_violation >= -convexity_eps,
        "convex constraint dropped: worst fitted second difference {worst_violation:.3e} < -eps {:.3e}",
        -convexity_eps
    );

    eprintln!(
        "[#1654] seed=7 k=20 sigma={sigma} convex fit: rmse={err:.4} curv_rms={curv_rms:.4}"
    );

    // PRIMARY: not the linear corner. Pre-fix this fit collapses to
    // rmse ≈ 0.31, curv_rms ≈ 0.15 (a straight line). Post-fix it recovers the
    // convex bowl: rmse ≈ 0.05, curv_rms ≈ 6–8. The thresholds sit in the wide
    // gap between the two regimes.
    assert!(
        curv_rms > 1.0,
        "convex smooth parked in the linear corner: curvature energy {curv_rms:.4} \
         is near zero (a straight line); expected a strongly convex fit (> 1.0)"
    );
    assert!(
        err < 0.12,
        "convex smooth truth-recovery RMSE {err:.4} is in the collapsed-corner regime \
         (~0.30); expected the good interior fit (< 0.12)"
    );
}
