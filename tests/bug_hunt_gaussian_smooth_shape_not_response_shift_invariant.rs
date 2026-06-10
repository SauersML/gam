//! Bug hunt: a Gaussian identity-link GAM's fitted smooth **shape** must be
//! exactly invariant to a constant added to the response.
//!
//! Mathematical fact. For `y ~ s(x)` with an unpenalized intercept and Gaussian
//! identity link, replacing `y` by `y + c·1` leaves the entire fit unchanged
//! except for the intercept, which increases by `c`:
//!
//!   * the constant `1` lies in the column space (the intercept fits it
//!     exactly), so the hat matrix `A` satisfies `(I − A)·1 = 0`;
//!   * therefore the residuals `(I − A)(y + c·1) = (I − A)y` are unchanged;
//!   * the profiled REML criterion depends on `y` only through those residuals
//!     and `(I − A)`, so the selected smoothing parameter `λ̂` is unchanged;
//!   * with `λ̂` and the residuals unchanged, the penalized coefficient block
//!     for the smooth is unchanged, so the fitted smooth **shape**
//!     `s(x) − mean(s)` is identical.
//!
//! In other words: shifting the response by a constant is a pure relabeling of
//! the intercept. A user whose response happens to have a large mean relative
//! to its variation (temperatures in Kelvin, financial levels, sensor
//! baselines, calendar years, …) must get the *same* smooth as a user who first
//! mean-centers the response.
//!
//! Observed (this crate, `fit_from_formula` Gaussian REML). The fitted smooth
//! shape drifts grossly as the response offset `c` grows, scaling like `c²`:
//!
//!   c = 1      shape drift ≈ 1.2e-10   (correct: machine precision)
//!   c = 1e2    shape drift ≈ 5.0e-7
//!   c = 1e4    shape drift ≈ 3.7e-3    ← this test (off by 0.0037)
//!   c = 1e6    shape drift ≈ 1.9e-1    (the smooth is wrong by 0.2)
//!
//! By contrast, *scaling* the response (`y → a·y`, which the same algebra
//! predicts should scale the smooth by `a`) is invariant to ~1e-15 even at
//! `a = 1e6` — the normal equations are exactly linear in `y`, so the scale
//! path is exact, isolating the defect to the additive-offset path.
//!
//! Likely cause. The design *columns* are mean-centered and scaled before the
//! solve (the `ColumnConditioner` change-of-basis in
//! `src/solver/estimate.rs:475-700`, which absorbs the column means into the
//! intercept), but the **response is never centered**. With an uncentered
//! response of magnitude `c`, a `yᵀy`-magnitude quantity (which scales as `c²`,
//! matching the observed drift exponent) on the REML evaluation path loses
//! ~`c²·ε` of precision relative to the true residual sum of squares (~`n·σ²`),
//! mis-locating `λ̂` and corrupting the smooth shape. The unit deviance itself
//! is formed safely as `Σ wᵢ (yᵢ − μᵢ)²` (`calculate_deviance`,
//! `src/solver/pirls/mod.rs:6446-6463`), so the defect is upstream of it, in
//! the smoothing-parameter selection / its sufficient statistics.
//!
//! The fix is to center the response by its (weighted) mean before fitting and
//! fold that constant into the intercept, mirroring the existing column
//! conditioning. This test then passes unchanged.

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

/// Fixed synthetic Gaussian dataset `y0 = sin(pi x) + N(0, 0.1)`, with a
/// constant `c` added to every response. The covariate column and the noise
/// realization are identical for every `c` (same seed), so the only difference
/// between two datasets is the additive response offset.
fn dataset_with_offset(c: f64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(20240611);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = u.sample(&mut rng);
            let y = (std::f64::consts::PI * x).sin() + noise.sample(&mut rng) + c;
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

/// Subtract the mean: isolate the smooth shape from the (legitimate) intercept
/// shift that absorbs the response offset.
fn centered(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| x - mean).collect()
}

#[test]
fn gaussian_smooth_shape_is_invariant_to_a_constant_response_offset() {
    init_parallelism();

    // Same data, response offset by 0 vs c = 1e4. A realistic large-mean
    // response (e.g. values in the thousands).
    let base_shape = centered(&fit_grid_predictions(&dataset_with_offset(0.0, 400)));
    let shifted_shape = centered(&fit_grid_predictions(&dataset_with_offset(1.0e4, 400)));

    let drift: f64 = base_shape
        .iter()
        .zip(&shifted_shape)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    // The smooth has range ~2 (sin over [0, pi]); a drift of 1e-6 is already
    // far above the ~1e-10 floor the c = 1 case achieves, and far below the
    // observed ~3.7e-3 defect at c = 1e4. The fit must be a pure relabeling of
    // the intercept, so this difference has to be at floating-point noise.
    assert!(
        drift < 1.0e-6,
        "Gaussian smooth shape changed when the response was offset by a \
         constant c=1e4: max |shape_base - shape_shifted| = {drift:.3e} \
         (must be ~1e-10; adding a constant to the response of an \
         identity-link GAM may only move the intercept, never the smooth)."
    );
}
