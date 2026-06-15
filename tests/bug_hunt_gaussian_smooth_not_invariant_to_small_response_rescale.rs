//! Bug hunt: a Gaussian identity-link GAM's fitted smooth must be *exactly*
//! equivariant to a multiplicative rescaling of the response — and it is for
//! large factors but NOT for small ones.
//!
//! Mathematical fact. For `y ~ s(x)` with Gaussian identity link, replacing the
//! response `y` by `a·y` (`a > 0`) is an exact linear relabeling of the
//! problem:
//!
//!   * the penalized normal equations `(XᵀWX + S_λ) β = Xᵀ W y` are linear in
//!     `y`, so the penalized coefficients scale exactly: `β̂(a·y) = a·β̂(y)` at
//!     any fixed `λ`;
//!   * the profiled Gaussian REML criterion for the smoothing parameter is
//!     `a`-invariant up to an additive constant `−(n−p)·ln a` (the dispersion
//!     `σ̂²` absorbs the `a²`), so the **selected `λ̂` is unchanged**, and with
//!     it the effective degrees of freedom;
//!   * therefore the fitted smooth scales exactly: `ŝ(x; a·y) = a·ŝ(x; y)`.
//!
//! A user whose response is measured in micro-units (`a = 1e-6`: strain, volts,
//! mole fractions, returns) must get the *same* shape — scaled by `a` — as a
//! user who first rescales the response to O(1).
//!
//! Observed (this crate, `fit_from_formula` Gaussian REML, default `s(x)` with
//! the double penalty). Up-scaling is exact, but **down-scaling to a small
//! response magnitude grossly over-smooths**. With the identical covariate and
//! noise realization, only the response scale `a` differing:
//!
//!   a = 1e0 ... 1e-4   selected λ̂ identical to ~1e-9 (correct)
//!   a = 1e-5           λ̂ inflates ~4.5x
//!   a = 1e-6           λ̂ inflates ~40x  (null-space penalty ~440x), the
//!                      recovered shape collapses toward a straight line
//!   a = 1e-8           λ̂ saturates ~54x larger
//!
//! Concretely the `a = 1e-6` fit's smooth, divided back by `a`, differs from the
//! unit-scale smooth by ~O(1) over a signal of range ~2 — the fit is a
//! *different, over-smoothed* function, not the same one rescaled.
//!
//! This is the small-magnitude (down-scale) sibling of the constant-offset
//! defect in `bug_hunt_gaussian_smooth_shape_not_response_shift_invariant.rs`
//! (#1000): that one noted up-scaling `y → a·y` is invariant to ~1e-15 even at
//! `a = 1e6`, which is true — but only checked the *large* direction. The
//! *small* direction is broken.
//!
//! Likely cause. The REML smoothing-parameter selection's stopping rule is keyed
//! to an **absolute** objective/gradient scale rather than one that tracks the
//! response magnitude. The inner-solve plateau band floors the objective scale
//! at `1.0` (`objective_scale = state.deviance.abs().max(...).max(1.0)`,
//! `src/solver/pirls/newton_solve.rs:903`, and the `.max(1.0)` in
//! `src/solver/latent_inner.rs:316`), and the outer ρ-gradient tolerance is the
//! bare `config.tolerance` whenever no `objective_scale` is supplied
//! (`outer_gradient_tolerance`, `src/solver/outer_strategy/run.rs:1724-1735`).
//! When the whole Gaussian objective is `O(a²) ≪ 1`, those absolute floors swamp
//! the real signal, so the outer optimizer declares premature convergence at an
//! over-smoothed λ instead of descending to the (scale-invariant) optimum. The
//! exact O(n) state-space spline scan (`double_penalty=false`), which selects λ
//! by a scale-free golden section on `log λ`, is by contrast invariant to ~1e-15
//! at `a = 1e-6` — isolating the defect to the general double-penalty REML path.
//!
//! When the selection is made scale-equivariant (e.g. by normalizing the
//! convergence scale by the weighted response variance, mirroring the existing
//! column conditioning), this test passes unchanged.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Fixed synthetic Gaussian dataset `y0 = sin(2π x) + N(0, 0.3)`, with every
/// response multiplied by `a`. The covariate column and the noise realization
/// are identical for every `a` (same seed), so the only difference between two
/// datasets is the multiplicative response scale.
fn dataset_with_scale(a: f64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(20240615);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = u.sample(&mut rng);
            let y0 = (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
            let y = y0 * a;
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `y ~ s(x)` (Gaussian) and return the fitted linear predictor on a fixed
/// grid of `x` values.
fn fit_grid_predictions(data: &gam::data::EncodedDataset) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };
    let probes: Vec<f64> = (0..25).map(|i| 0.02 + 0.96 * (i as f64) / 24.0).collect();
    let mut grid = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() {
        grid[[i, 0]] = v;
        grid[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Subtract the mean: isolate the smooth shape from the intercept.
fn centered(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| x - mean).collect()
}

#[test]
fn gaussian_smooth_is_equivariant_to_a_small_response_rescale() {
    init_parallelism();

    // Same data, response scaled by a = 1 vs a = 1e-6 (a realistic micro-unit
    // response). The unit-scale smooth is the ground truth; the small-scale
    // smooth, divided back by a, must reproduce it.
    let a: f64 = 1.0e-6;
    let base_shape = centered(&fit_grid_predictions(&dataset_with_scale(1.0, 400)));
    let small_pred = fit_grid_predictions(&dataset_with_scale(a, 400));
    let small_shape_rescaled: Vec<f64> = centered(&small_pred).iter().map(|v| v / a).collect();

    let drift: f64 = base_shape
        .iter()
        .zip(&small_shape_rescaled)
        .map(|(b, s)| (b - s).abs())
        .fold(0.0, f64::max);

    // The smooth has range ~2 (a full sine over [0, 1]). Up-scaling is exact to
    // ~1e-13; the down-scaling path's observed defect is ~O(1). A scale of
    // 1e-3 is the tightest the linear normal equations should ever justify — and
    // far below the ~1e-1..1e0 over-smoothing seen at a = 1e-6. The fit must be
    // a pure linear rescaling of the response.
    assert!(
        drift < 1.0e-3,
        "Gaussian smooth changed when the response was rescaled by a={a:.0e}: \
         max |shape_base - shape_small/a| = {drift:.3e} over a signal of range ~2 \
         (must be ~1e-13, as it is for up-scaling). A Gaussian identity-link GAM's \
         penalized normal equations are exactly linear in y and its REML λ̂ is \
         scale-invariant, so rescaling the response may only rescale the smooth, \
         never reshape it."
    );
}
