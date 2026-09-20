//! gam#2600, the predictive half: a conditional-transformation-normal model has
//! to be able to answer a question about its own tails.
//!
//! ## The defect
//!
//! gam#2600 removed the endpoint renormalizer, so the fitted CTN density is the
//! most-likely-transformation density `φ(h)·h'` and the model's CDF is
//! `F(y|x) = Φ(h(y|x))` on the whole real line. Nothing gave `h` anything to be
//! out there. `ctn_response_bases_at` built the response basis from the shared
//! I-spline evaluator, which SATURATES outside its knots (`I_k` constant, and
//! gam#2695 zeroed `M_k` to match), so the entire exterior of the fitted
//! transformation was
//!
//! ```text
//! h(y) = h(y_b) + ε·(y − y_b),    ε = TRANSFORMATION_MONOTONICITY_EPS = 1e-8,
//! ```
//!
//! the monotonicity floor and nothing else. Both inverse-transform consumers
//! then hid that by returning the SUPPORT ENDPOINT for any latent target off the
//! end of the tabulated transform — a truncation the likelihood does not
//! perform. Measured on this fixture before the fix: `Φ(h(y_lo)) = 2.4e-2` of
//! the model's own predictive mass sat below the table and was collapsed onto
//! the single point `y_lo`, so the 2.3 %, the 0.13 % and the 0.003 % predictive
//! quantiles were all the same number.
//!
//! ## What is asserted, and why these claims and not a calibration bar
//!
//! Every assertion here is an identity of the fitted model with itself, so none
//! of them needs a statistical tolerance or can be argued down:
//!
//! 1. **The reported quantile is a quantile.** `h(h⁻¹(z)|x) = z` for every `z` on
//!    the predictive ladder, evaluated by feeding the reported response back
//!    through the production observed-score path. Under the clamp this failed by
//!    the whole distance from the ladder node to the support endpoint's latent.
//! 2. **The transform is C¹ across the support boundary.** A finite difference of
//!    the score just outside the fitted support matches one just inside. This is
//!    the root cause stated directly: the old exterior slope was `1e-8` against
//!    an interior slope of order one.
//! 3. **`Φ(h)` is a usable CDF.** The response at which the model's own CDF
//!    reaches `1 − 1e-6` is within one support width of `y_hi`. With the floor it
//!    was `≈1.5e8` response units out, i.e. ten million support widths.
//! 4. **The sampler has no atoms.** Inverse-transform draws never land exactly on
//!    a support endpoint, and the share that lands outside the fitted support is
//!    the share the model itself claims.
//!
//! The law is chosen so the truth is closed-form: `Y = exp(Z)`, `Z ~ N(0,1)`
//! gives `F(y) = Φ(ln y)`, and the model's own definition `F = Φ(h)` pins
//! `h(y) = ln y` exactly, with no location/scale freedom left to quotient out.
//! Truth values are printed for context; the pass criteria are the four
//! identities above.

use gam::generative::sampleobservation_seeded_replicates;
use gam::inference::model::{FittedModel, PredictModelClass};
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::predict::input::build_predict_input_for_model;
use gam::predict::input::{
    build_transformation_normal_observed_scores, build_transformation_normal_quantile_grid,
};
use gam::predict::{
    FittedModelPredictExt, InferenceCovarianceMode, PosteriorMeanOptions, SavedGenerativeInput,
    generative_spec_for_saved_model,
};
use gam::probability::{normal_cdf, standard_normal_quantile};
use gam::test_support::synthetic::SplitMixNormalRng;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Sample size: large enough that the fitted support genuinely excludes a
/// percent-scale slice of the predictive mass (which is what the clamp used to
/// swallow), small enough that the intercept-only solve is a second.
const N: usize = 256;

/// The fitted model plus what a caller needs to drive its replay paths.
struct Fixture {
    model: FittedModel,
    col_map: HashMap<String, usize>,
    y: Vec<f64>,
}

fn fit_lognormal_ctn() -> Fixture {
    init_parallelism();
    let mut rng = SplitMixNormalRng::new(0x2600_C7Du64);
    let y: Vec<f64> = (0..N).map(|_| rng.standard_normal().exp()).collect();
    assert!(
        y.iter().all(|value| value.is_finite() && *value > 0.0),
        "the lognormal fixture must be strictly positive and finite"
    );
    let headers = vec!["y".to_string()];
    let rows: Vec<csv::StringRecord> = y
        .iter()
        .map(|value| csv::StringRecord::from(vec![format!("{value:.17e}")]))
        .collect();
    let dataset =
        encode_recordswith_inferred_schema(headers, rows).expect("encode the lognormal column");
    let config = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ 1".to_string(), &dataset, &config)
        .expect("intercept-only transformation-normal fit");
    let model = FittedModel::from_payload(payload);
    assert_eq!(
        model.predict_model_class(),
        PredictModelClass::TransformationNormal,
        "the fixture must produce a transformation-normal model"
    );
    let col_map = dataset.column_map();
    Fixture { model, col_map, y }
}

impl Fixture {
    /// A frame with `rows` rows carrying the training response column. An
    /// intercept-only covariate design does not read it, so the only thing that
    /// matters here is the row count — but a frame built from the training
    /// column keeps that count honest against the schema.
    fn frame(&self, rows: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, 1), |(i, _)| self.y[i])
    }

    /// `h(y | x)` at arbitrary responses through the PRODUCTION observed-score
    /// path — the same function `gamfit`'s `model.transformation_score(df)`
    /// calls, so a round trip through it is a statement about shipped behaviour
    /// rather than about a test's own re-derivation of the chart.
    fn scores_at(&self, y: &[f64]) -> Array1<f64> {
        let m = y.len();
        let mut frame = Array2::<f64>::zeros((m, 1));
        for (row, &value) in y.iter().enumerate() {
            frame[[row, 0]] = value;
        }
        let mut col_map: HashMap<String, usize> = HashMap::new();
        col_map.insert("y".to_string(), 0);
        build_transformation_normal_observed_scores(
            &self.model,
            frame.view(),
            &col_map,
            self.model.training_headers.as_ref(),
            &Array1::from_vec(y.to_vec()),
            &Array1::<f64>::zeros(m),
        )
        .expect("observed CTN scores")
    }
}

#[test]
fn ctn_predictive_quantiles_invert_the_models_own_transform_2600() {
    let fixture = fit_lognormal_ctn();
    let frame = fixture.frame(N);
    let grid = build_transformation_normal_quantile_grid(
        &fixture.model,
        frame.view(),
        &fixture.col_map,
        fixture.model.training_headers.as_ref(),
        &Array1::<f64>::zeros(N),
    )
    .expect("CTN quantile table");

    let grid_y = grid.table.grid_y().to_owned();
    let g = grid_y.len();
    let (y_lo, y_hi) = (grid_y[0], grid_y[g - 1]);
    let latent = grid.table.latent();
    let (lower, upper) = (latent[[0, 0]], latent[[0, g - 1]]);
    let clamped_mass = normal_cdf(lower) + (1.0 - normal_cdf(upper));
    // The table's exterior is affine at its end slopes (`CtnTransformTable::evaluate`),
    // so the secant one support width past each end is that end's slope.
    let width = y_hi - y_lo;
    let tail_lo = (lower - grid.table.evaluate(0, y_lo - width)) / width;
    let tail_hi = (grid.table.evaluate(0, y_hi + width) - upper) / width;
    eprintln!(
        "#2600 tails: support [{y_lo:.6e}, {y_hi:.6e}] L={lower:+.6} U={upper:+.6} \
         Phi(L)+1-Phi(U)={clamped_mass:.6e} tail_slopes=({tail_lo:.6e}, {tail_hi:.6e})"
    );
    assert!(
        clamped_mass > 1.0e-3,
        "this fixture no longer exercises the defect: only {clamped_mass:.3e} of the \
         predictive mass falls outside the tabulated support, so a clamp there would be \
         invisible and the round trip below would prove nothing"
    );

    // ---- (1) the reported quantile is a quantile ---------------------------
    // The production ladder: `TRANSFORMATION_NORMAL_BAND_Z_NODES = 65` nodes
    // evenly spaced over `±TRANSFORMATION_NORMAL_BAND_Z_MAX = 4`, which is
    // exactly what the predictor interpolates to build response-scale bands.
    const LADDER_NODES: usize = 65;
    const LADDER_Z_MAX: f64 = 4.0;
    let ladder_step = 2.0 * LADDER_Z_MAX / ((LADDER_NODES - 1) as f64);
    let z_ladder: Vec<f64> = (0..LADDER_NODES)
        .map(|j| -LADDER_Z_MAX + ladder_step * (j as f64))
        .collect();
    let quantiles: Vec<f64> = z_ladder.iter().map(|&z| grid.table.invert(0, z)).collect();
    let round_trip = fixture.scores_at(&quantiles);
    let mut worst = 0.0_f64;
    let mut worst_at = 0.0_f64;
    for (j, &z) in z_ladder.iter().enumerate() {
        let gap = (round_trip[j] - z).abs();
        if gap > worst {
            worst = gap;
            worst_at = z;
        }
        if j % 8 == 0 {
            eprintln!(
                "#2600 tails: z={z:+.2} h^-1(z)={:.6e} h(h^-1(z))={:+.8} truth exp(z)={:.6e}",
                quantiles[j],
                round_trip[j],
                z.exp()
            );
        }
    }
    eprintln!(
        "#2600 tails: max|h(h^-1(z)) - z| = {worst:.6e} at z={worst_at:+.3} \
         (bar: 1% of the ladder step {ladder_step:.4})"
    );
    // The bar is a property of the reported object, not a chosen tolerance: a
    // ladder whose nodes are `ladder_step` apart in the latent has to be
    // resolved to well inside one step, or two adjacent reported quantiles are
    // not distinguishable from each other's error. One percent of the step is
    // the statement "resolved"; the two affine tails invert exactly and the
    // interior is a cubic Hermite of a 257-node table, which lands an order
    // below it. The chord that preceded the Hermite measured 2.1e-3 here, and
    // the endpoint clamp measured 2.03 — the whole distance from z = -4 to
    // L = -1.971548.
    assert!(
        worst < 0.01 * ladder_step,
        "the reported predictive quantile is not the model's own quantile: \
         max|h(h^-1(z)) - z| = {worst:.6e} at z={worst_at:+.3}, against 1% of the ladder \
         step ({:.6e}). Under the endpoint clamp every ladder node past [L, U] returned a \
         support endpoint, so this gap was the whole distance from the node to \
         L={lower:+.6} / U={upper:+.6}.",
        0.01 * ladder_step
    );

    // Distinctness: a clamped ladder is CONSTANT past the support, which is how
    // the 2.3 %, 0.13 % and 0.003 % quantiles became one number.
    for j in 1..quantiles.len() {
        assert!(
            quantiles[j] > quantiles[j - 1],
            "the predictive quantile ladder is not strictly increasing at z={:+.2} -> {:+.2} \
             ({:.6e} -> {:.6e}); a flat run means the ladder saturated at a support endpoint",
            z_ladder[j - 1],
            z_ladder[j],
            quantiles[j - 1],
            quantiles[j]
        );
    }
}

#[test]
fn ctn_transform_is_c1_across_the_fitted_support_boundary_2600() {
    let fixture = fit_lognormal_ctn();
    let frame = fixture.frame(N);
    let grid = build_transformation_normal_quantile_grid(
        &fixture.model,
        frame.view(),
        &fixture.col_map,
        fixture.model.training_headers.as_ref(),
        &Array1::<f64>::zeros(N),
    )
    .expect("CTN quantile table");
    let grid_y = grid.table.grid_y().to_owned();
    let g = grid_y.len();
    let (y_lo, y_hi) = (grid_y[0], grid_y[g - 1]);
    let width = y_hi - y_lo;

    // ---- (2) no derivative jump at either boundary knot --------------------
    // The two secants are taken over the same short step on either side, so the
    // only thing separating their ratio from 1 is the curvature of `h` over that
    // step (second order in it) — six orders of magnitude away from the `1e-8`
    // the saturating exterior produced.
    let step = 1.0e-6 * width;
    for (name, boundary) in [("lower", y_lo), ("upper", y_hi)] {
        // Two secants of the same width, one wholly inside the fitted support
        // and one wholly outside it, straddling the boundary knot.
        let probes = vec![
            boundary - 2.0 * step,
            boundary - step,
            boundary + step,
            boundary + 2.0 * step,
        ];
        let s = fixture.scores_at(&probes);
        let (inside, outside) = if name == "lower" {
            ((s[3] - s[2]) / step, (s[1] - s[0]) / step)
        } else {
            ((s[1] - s[0]) / step, (s[3] - s[2]) / step)
        };
        eprintln!(
            "#2600 tails: {name} boundary y={boundary:.6e} h' inside={inside:.6e} \
             outside={outside:.6e} ratio={:.6e}",
            outside / inside
        );
        assert!(
            inside > 1.0e-3,
            "the {name} interior slope is already degenerate ({inside:.6e}); there is nothing \
             for the continuation to match"
        );
        assert!(
            (outside / inside - 1.0).abs() < 1.0e-3,
            "h' jumps across the {name} support boundary: inside={inside:.6e}, \
             outside={outside:.6e}. Before gam#2600's affine continuation the exterior slope \
             was the monotonicity floor 1e-8, so this ratio was ~1e-8 and the model's entire \
             tail was a numerical artefact."
        );
    }

    // ---- (3) Phi(h) is a usable CDF ----------------------------------------
    // The response at which the model's own CDF reaches 1 - 1e-6, bisected
    // inside a bracket one support width wide. With the floor the answer was
    // ~1.5e8 response units past y_hi — ten million support widths — so the
    // bracket below simply would not have contained it.
    let target = 4.753_424_308_822_899_f64; // Phi^{-1}(1 - 1e-6)
    let (mut lo, mut hi) = (y_hi, y_hi + width);
    let bracket_top = fixture.scores_at(&[hi])[0];
    eprintln!(
        "#2600 tails: h(y_hi + width) = {bracket_top:+.6} (needs >= {target:.6} for the \
         1-1e-6 quantile to sit within one support width of y_hi)"
    );
    assert!(
        bracket_top >= target,
        "the fitted model cannot reach its own 1-1e-6 quantile within one support width \
         of y_hi: h({hi:.6e}) = {bracket_top:+.6} < {target:.6}. A transformation that is \
         flat outside its knots needs O(1/eps) response units to spend its tail mass."
    );
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if fixture.scores_at(&[mid])[0] < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    eprintln!(
        "#2600 tails: 1-1e-6 quantile = {hi:.6e} = y_hi + {:.4} support widths \
         (truth exp(4.7534) = {:.6e})",
        (hi - y_hi) / width,
        target.exp()
    );
}

#[test]
fn ctn_inverse_transform_draws_have_no_atoms_at_the_support_2600() {
    let fixture = fit_lognormal_ctn();
    let n_rows = 64usize;
    let frame = fixture.frame(n_rows);
    let zero = Array1::<f64>::zeros(n_rows);
    let spec = generative_spec_for_saved_model(
        &fixture.model,
        SavedGenerativeInput {
            data: frame.view(),
            col_map: &fixture.col_map,
            training_headers: fixture.model.training_headers.as_ref(),
            offset: &zero,
            offset_noise: &zero,
            noise_offset_supplied: false,
            prior_weights: None,
        },
    )
    .expect("a saved CTN model exposes its inverse-transform sampler");

    let grid = build_transformation_normal_quantile_grid(
        &fixture.model,
        frame.view(),
        &fixture.col_map,
        fixture.model.training_headers.as_ref(),
        &zero,
    )
    .expect("CTN quantile table");
    let grid_y = grid.table.grid_y().to_owned();
    let g = grid_y.len();
    let (y_lo, y_hi) = (grid_y[0], grid_y[g - 1]);
    let latent = grid.table.latent();
    // The model's OWN statement of how much mass lies outside the tabulated
    // support, averaged over the rows drawn — this is the bar, not a constant.
    let expected_outside: f64 = (0..n_rows)
        .map(|i| normal_cdf(latent[[i, 0]]) + (1.0 - normal_cdf(latent[[i, g - 1]])))
        .sum::<f64>()
        / (n_rows as f64);

    const DRAWS: usize = 400;
    let draws = sampleobservation_seeded_replicates(&spec, 0, DRAWS, 0x2600_D7A)
        .expect("inverse-transform replicates");
    assert_eq!(draws.shape(), [DRAWS, n_rows]);
    let total = DRAWS * n_rows;
    let atoms = draws
        .iter()
        .filter(|value| **value == y_lo || **value == y_hi)
        .count();
    let outside = draws
        .iter()
        .filter(|value| **value < y_lo || **value > y_hi)
        .count();
    let observed = (outside as f64) / (total as f64);
    eprintln!(
        "#2600 tails: {DRAWS}x{n_rows} draws — {atoms} sit EXACTLY on a support endpoint, \
         {outside} land outside [{y_lo:.6e}, {y_hi:.6e}] ({observed:.5} of the sample against \
         the model's own {expected_outside:.5})"
    );
    assert_eq!(
        atoms, 0,
        "the inverse-transform sampler put {atoms}/{total} draws EXACTLY on a fitted support \
         endpoint. Those are the atoms the endpoint clamp created: a latent draw past \
         h(y_hi|x) has probability 1 - Phi(h(y_hi|x)) under the model, and returning y_hi \
         for it turns a continuous predictive law into one with two point masses."
    );
    // The sampler must spend the tail mass the model claims, not a tuned share
    // of it: four binomial standard errors around the model's own figure.
    let se = (expected_outside * (1.0 - expected_outside) / (total as f64)).sqrt();
    assert!(
        (observed - expected_outside).abs() <= 4.0 * se,
        "the fraction of draws outside the fitted support is {observed:.5}, against the \
         model's own {expected_outside:.5} +/- {:.5}",
        4.0 * se
    );
}

#[test]
fn ctn_observation_bands_are_the_models_own_quantiles_at_every_level_2600() {
    // The band a user actually receives, through the production predictor, at
    // three levels that bracket the two places the ladder used to stop being a
    // quantile: inside the tabulated support, past the fitted support (where the
    // grid inversion clamped), and past the ladder's own `z_max = 4` (where the
    // ladder clamped).
    let fixture = fit_lognormal_ctn();
    let frame = fixture.frame(N);
    let zero = Array1::<f64>::zeros(N);
    let input = build_predict_input_for_model(
        &fixture.model,
        frame.view(),
        &fixture.col_map,
        fixture.model.training_headers.as_ref(),
        &zero,
        &zero,
        false,
    )
    .expect("CTN predict input");
    let predictor = fixture
        .model
        .predictor()
        .expect("a saved CTN model exposes a predictor");

    let mut previous: Option<(f64, f64)> = None;
    for &level in &[0.95_f64, 0.9995, 0.999_999] {
        let result = predictor
            .predict_posterior_mean(
                &input,
                fixture
                    .model
                    .unified()
                    .expect("a saved CTN model carries a unified fit"),
                &PosteriorMeanOptions {
                    confidence_level: Some(level),
                    covariance_mode: InferenceCovarianceMode::Conditional,
                    include_observation_interval: true,
                    extrapolation_variance: None,
                    observation_prior_weights: None,
                },
            )
            .expect("posterior-mean prediction with an observation band");
        let lower = result
            .observation_lower
            .as_ref()
            .expect("CTN reports a response-scale observation band")[0];
        let upper = result
            .observation_upper
            .as_ref()
            .expect("CTN reports a response-scale observation band")[0];
        // The band limits are posterior-predictive quantiles, `g(limit) = ±z` with
        // `g = ĥ/√(1 + s²)`, so the plug-in transform the production score path
        // reports sits at or past `±Φ⁻¹((1+level)/2)`: coefficient uncertainty only
        // widens a band (SPEC rule 3).
        let z = standard_normal_quantile(0.5 * (1.0 + level)).expect("central z");
        let scores = fixture.scores_at(&[lower, upper]);
        eprintln!(
            "#2600 tails: level={level} band=[{lower:.6e}, {upper:.6e}] \
             h(band)=[{:+.6}, {:+.6}] against z=±{z:.6}",
            scores[0], scores[1]
        );
        assert!(
            lower < upper,
            "the observation band is not ordered at level {level}: [{lower}, {upper}]"
        );
        // One percent of the ladder's own step (0.125) is the same "resolved" bar
        // the table round trip uses.
        for (side, score, target) in [("lower", scores[0], -z), ("upper", scores[1], z)] {
            assert!(
                score * target.signum() >= z - 0.01 * 0.125,
                "the {side} observation limit at level {level} is narrower than the plug-in \
                 quantile: h(limit)={score:+.8} against {target:+.8}"
            );
        }
        if let Some((previous_lower, previous_upper)) = previous {
            assert!(
                lower < previous_lower && upper > previous_upper,
                "the band did not widen from the previous level: \
                 [{previous_lower:.6e}, {previous_upper:.6e}] -> [{lower:.6e}, {upper:.6e}]. \
                 Both clamps this pins showed up exactly here — as two levels reporting \
                 the same interval."
            );
        }
        previous = Some((lower, upper));
    }
}
