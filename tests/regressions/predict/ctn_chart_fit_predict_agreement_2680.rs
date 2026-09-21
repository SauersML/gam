//! Regression (#2680): a fitted CTN and its own replay paths must read the
//! coefficient block through **one** chart.
//!
//! ## The defect
//!
//! `#2306` moved the conditional-transformation-normal likelihood onto the
//! direct-α chart
//!
//! ```text
//! h(y, x) = α₀(x) + Σ_{k≥1} I_k(y)·α_k(x) + offset + ε·(y − median),
//! α_k(x) = ψ(x)ᵀ A[k, :],   α_k(x_i) ≥ 0 by the Khatri-Rao monotonicity cone,
//! ```
//!
//! and every replay consumer kept reading the same `blocks[0].beta` through the
//! pre-cutover squared latent chart `Σ_{k≥1} I_k(y)·γ_k(x)²`. Because the lower
//! endpoint basis is `[1, 0, …, 0]` it reads only the (unsquared) location
//! column, while the upper endpoint `Σ_k α_k` becomes `Σ_k α_k²`; with the shape
//! coordinates near a common `c` the reported score is
//!
//! ```text
//! z_reported ≈ c·z + (1 − c)·L,     c = (U_squared − L) / (U_direct − L),
//! ```
//!
//! i.e. correct **exactly at `c = 1`** and wrong in proportion to how far the
//! fitted support width is from the shape-coordinate count. `c ≈ range(h)/p_shape`
//! grows with the sample range of the response, so the error is invisible on
//! small fixtures and severe at production `n` — the shape #2680 was filed on
//! (`sd(z)` measured `1.16` at `n = 128` and `1.79` at `n = 800`, against a fit
//! whose own score was `N(0, 1.01…1.05)` throughout).
//!
//! ## What this pins, and why it is the right invariant
//!
//! Not "the score is calibrated" — that is a statistical bar with a tolerance.
//! **The fit's own latent score and the predict path's latent score are the same
//! numbers.** `calibrate_transformation_scores` writes `block_states[0].eta` from
//! `row_quantities`, the exact quantity the likelihood was maximized on;
//! `build_transformation_normal_observed_scores` is the production function
//! behind `gamfit`'s `model.transformation_score(df)`. Evaluated on the training
//! rows of the model that produced them, those two vectors are the same
//! function of the same coefficients and must agree to round-off.
//!
//! That statement is **chart-agnostic**: it does not encode which chart is
//! right, only that there is one. It therefore catches this defect, its mirror
//! image (a fit that moves without its consumers), and any future divergence in
//! how the two sides rebuild the response basis, prepend the location column, or
//! locate the support endpoints — none of which a calibration bar would notice
//! until it happened to cross a tolerance.
//!
//! The second assertion is the statistical consequence, at the size where the
//! defect bites: the score's first two moments must clear the very gate
//! `bms/gradient_paths.rs` applies before it will consume `z` as a generated
//! regressor (`|mean| ≤ 4/√n`, `|sd − 1| ≤ 4/√(2(n−1))`). On the squared chart
//! this fixture reports `sd ≈ 1.4`, roughly ten times its own bound.
//!
//! Both live in ONE test on ONE fit. A CTN fit at this size is ~35 s, and the
//! two claims are about the same score vector, so a second fit would buy
//! nothing but wall-clock and a second chance to trip over an unrelated
//! optimizer refusal. The identity is asserted first: if the two sides disagree
//! about what `β` means, the moments of either one are not the interesting
//! number.

use gam::data::EncodedDataset;
use gam::inference::model::{FittedModel, PredictModelClass};
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::predict::input::build_transformation_normal_observed_scores;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array1;

/// Tiny deterministic PRNG (SplitMix64) so the fixture is identical on every
/// platform without pulling in an RNG crate.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        gam_linalg::utils::splitmix64(&mut self.state)
    }
    fn next_uniform(&mut self) -> f64 {
        let u = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// #2680's fixture shape: correlated Gaussian covariates, a response linear in
/// them plus homoscedastic noise, at a size where the squared chart's `c` is
/// well away from 1. A conditionally-Gaussian response is the *easiest* possible
/// conditional transformation (`h` is affine in `y`), which is what makes a
/// miscalibrated score here unambiguous.
fn build_fixture(n: usize, seed: u64) -> (Vec<String>, Vec<csv::StringRecord>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "x3".to_string(),
        "y".to_string(),
    ];
    let mut records = Vec::with_capacity(n);
    let mut responses = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = rng.next_normal();
        let x2 = 0.3 * x1 + (1.0 - 0.09_f64).sqrt() * rng.next_normal();
        let x3 = rng.next_normal();
        let y = 0.4 * x1 - 0.2 * x2 + 0.15 * x3 + 0.9 * rng.next_normal();
        records.push(csv::StringRecord::from(vec![
            format!("{x1:.17e}"),
            format!("{x2:.17e}"),
            format!("{x3:.17e}"),
            format!("{y:.17e}"),
        ]));
        responses.push(y);
    }
    (headers, records, responses)
}

fn fit_ctn(dataset: &EncodedDataset) -> (FittedModel, Array1<f64>) {
    let config = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(
        "y ~ s(x1, k=5) + s(x2, k=5)".to_string(),
        dataset,
        &config,
    )
    .expect("transformation-normal fit + payload");
    let fitted = payload
        .fit_result
        .as_ref()
        .expect("transformation-normal payload carries a unified fit");
    // `calibrate_transformation_scores` overwrites the single block's `eta` with
    // `Φ⁻¹(F̂(y_i|x_i))` computed from `row_quantities` — the fit's own latent
    // score, on the chart the likelihood was maximized on.
    let fit_eta = fitted
        .block_states
        .first()
        .expect("transformation-normal fit has one coefficient block")
        .eta
        .clone();
    let model = FittedModel::from_payload(payload);
    assert_eq!(
        model.predict_model_class(),
        PredictModelClass::TransformationNormal,
        "fixture must produce a transformation-normal model"
    );
    (model, fit_eta)
}

#[test]
fn ctn_predict_score_reproduces_the_fitted_score_2680() {
    init_parallelism();
    const N: usize = 400;
    let (headers, records, _) = build_fixture(N, 4271);
    let dataset =
        encode_recordswith_inferred_schema(headers, records).expect("encode fixture dataset");
    let (model, fit_eta) = fit_ctn(&dataset);

    let col_map = dataset.column_map();
    let response_column = *col_map.get("y").expect("response column present");
    let response = dataset.values.column(response_column).to_owned();
    let offset = Array1::<f64>::zeros(N);

    let predicted = build_transformation_normal_observed_scores(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &response,
        &offset,
    )
    .expect("observed CTN scores on the training rows");

    assert_eq!(predicted.len(), fit_eta.len(), "score row count");

    // ---- (1) PRIMARY: fit and predict are on ONE chart --------------------
    let mut max_gap = 0.0_f64;
    let mut worst = 0usize;
    for i in 0..predicted.len() {
        let gap = (predicted[i] - fit_eta[i]).abs();
        if gap > max_gap {
            max_gap = gap;
            worst = i;
        }
    }
    eprintln!(
        "#2680 CTN fit-vs-predict: n={N} max|predict − fit eta|={max_gap:.6e} at row {worst}"
    );

    // Both sides evaluate the same affine chart on the same coefficients, so the
    // only admissible difference is accumulation round-off on `p_resp` terms —
    // orders of magnitude below this bound. On the squared chart the gap is
    // O(1): #2680 measured a reported score of `N(+0.96, 1.47)` against a fitted
    // `N(+0.001, 1.011)` on the issue's own `n = 400` cell.
    assert!(
        max_gap < 1.0e-9,
        "the predict path and the fit disagree about the CTN latent score: \
         max|predict − fit eta| = {max_gap:.6e} at row {worst} \
         (predict={:.6}, fit={:.6}). They read the SAME blocks[0].beta, so a \
         non-round-off gap means the two sides are on different coefficient charts.",
        predicted[worst],
        fit_eta[worst]
    );

    // ---- (2) the statistical consequence, on the same score ---------------
    let n = predicted.len() as f64;
    let mean = predicted.iter().sum::<f64>() / n;
    let sd = (predicted.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt();

    // The bars are the ones `bms::gradient_paths` applies before it will consume
    // `z` as a generated regressor: four standard errors of the mean and of the
    // sd of a standard normal sample. Derived from `n`, not tuned.
    let mean_tol = 4.0 / n.sqrt();
    let sd_tol = 4.0 / (2.0 * (n - 1.0)).sqrt();
    eprintln!(
        "#2680 CTN observed score: n={N} mean={mean:+.6} (tol {mean_tol:.4}) \
         sd={sd:.6} (tol {sd_tol:.4})"
    );
    assert!(
        mean.abs() <= mean_tol,
        "CTN observed score is not centred: mean={mean:+.6} exceeds {mean_tol:.4}"
    );
    assert!(
        (sd - 1.0).abs() <= sd_tol,
        "CTN observed score is not unit-scale: sd={sd:.6} misses 1 by more than {sd_tol:.4}. \
         A score read through the wrong coefficient chart is `c·z + (1−c)·L`, so its sd IS `c` \
         — #2680's `1.79` at n=800."
    );
}
