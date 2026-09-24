//! gam#2612: the multinomial's covariance-definition axis, gated from an angle
//! no coverage sweep can reach.
//!
//! The defect this pins is that `fit_penalized_multinomial_formula` read
//! `fit.covariance_conditional` and dropped the first-order ρ-uncertainty
//! correction `C = J·Var(ρ̂)·Jᵀ` the SAME fit had already computed, so every
//! multinomial band was conditional-on-λ̂ while every other family in the library
//! defaults to the corrected definition. A coverage sweep sees that only as
//! "under-covers", which is also what a wrong Jacobian, a mis-ordered covariance
//! or a bad centre look like. These assertions separate them:
//!
//!   1. the correction SURVIVES the fit (it is `Some`, symmetric, PSD-diagonal),
//!   2. the correction REACHES the response scale — the corrected band is
//!      strictly wider than the conditional one, which is the step that was
//!      missing and that a stored-but-unused matrix would not produce,
//!   3. the factorised `gᵀ C g` kernel equals the literal `d`-dimensional
//!      contraction assembled entry by entry, and
//!   4. asking for a correction a model does not carry is an ERROR, never a
//!      silent downgrade to the narrower band.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    InferenceCovarianceMode, MultinomialFitRequest, MultinomialIntervalDecline,
    MultinomialSavedModel, MultinomialSpreadDecline, fit_penalized_multinomial_formula,
    predict_multinomial_formula, predict_multinomial_formula_with_intervals,
};

const N: usize = 220;

/// Deterministic LCG → `U[0,1)`; no external RNG so the fixture is
/// byte-identical run to run.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next_u01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A three-class smooth softmax sample: every class is well represented and the
/// boundaries are curved, so REML has a genuine interior optimum in ρ and the
/// outer solve has curvature to propagate.
fn smooth_three_class(seed: u64) -> Vec<StringRecord> {
    let mut rng = Lcg::new(seed);
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        let x = -2.0 + 4.0 * (i as f64) / ((N - 1) as f64);
        let eta = [0.9 * (x + 0.4).sin(), -0.4 + 0.7 * x.cos(), 0.0];
        let shift = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = eta.iter().map(|e| (e - shift).exp()).collect();
        let total: f64 = weights.iter().sum();
        let u = rng.next_u01();
        let mut cumulative = 0.0;
        let mut drawn = 2usize;
        for (class, weight) in weights.iter().enumerate() {
            cumulative += weight / total;
            if u < cumulative {
                drawn = class;
                break;
            }
        }
        rows.push(StringRecord::from(vec![
            format!("{x:.8}"),
            format!("c{drawn}"),
        ]));
    }
    rows
}

fn fit_smooth_three_class(seed: u64) -> MultinomialSavedModel {
    let data = encode_recordswith_inferred_schema(
        vec!["x".to_string(), "y".to_string()],
        smooth_three_class(seed),
    )
    .expect("encode three-class dataset");
    let config = FitConfig::default();
    fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: usize::MAX,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tps', k=8)", &config)
    })
    .expect("three-class smooth multinomial fit")
}

/// `softmax` over the active logits with the reference class pinned at `η = 0`.
fn softmax_with_reference(eta: &[f64]) -> Vec<f64> {
    let shift = eta.iter().copied().fold(0.0_f64, f64::max);
    let mut out: Vec<f64> = eta.iter().map(|e| (e - shift).exp()).collect();
    out.push((-shift).exp());
    let total: f64 = out.iter().sum();
    for value in out.iter_mut() {
        *value /= total;
    }
    out
}

/// The per-row standard errors as one `(R, K)` array. Every row of this fixture
/// publishes its spread, so a declined row is a failure here, named with its reason.
fn published_standard_errors(
    rows: Vec<Result<ndarray::Array1<f64>, MultinomialSpreadDecline>>,
) -> ndarray::Array2<f64> {
    let published: Vec<ndarray::Array1<f64>> = rows
        .into_iter()
        .enumerate()
        .map(|(row, spread)| {
            spread.unwrap_or_else(|decline| {
                panic!("row {row} published no standard error on this fixture: {decline}")
            })
        })
        .collect();
    let classes = published.first().map_or(0, |spread| spread.len());
    ndarray::Array2::from_shape_fn((published.len(), classes), |(row, class)| {
        published[row][class]
    })
}

#[test]
fn the_correction_reaches_the_response_scale_and_widens_the_band_2612() {
    let model = fit_smooth_three_class(11);
    let x = model.training_design().expect("training design");
    let correction = model.smoothing_correction().expect("retained correction");

    let (mean_conditional, se_conditional) = model
        .predict_probabilities_with_se_in_mode(x.view(), InferenceCovarianceMode::Conditional)
        .expect("conditional band");
    let se_conditional = published_standard_errors(se_conditional);
    let (mean_corrected, se_corrected) = model
        .predict_probabilities_with_se_in_mode(
            x.view(),
            InferenceCovarianceMode::SmoothingCorrected,
        )
        .expect("corrected band");
    let se_corrected = published_standard_errors(se_corrected);

    // The CENTRE is the same estimand under both modes — only the spread carries
    // the smoothing uncertainty. A mode that also moved the mean would be a
    // different fit, not a different interval.
    for ((row, class), &value) in mean_conditional.indexed_iter() {
        assert_eq!(
            value,
            mean_corrected[[row, class]],
            "the covariance mode moved the posterior MEAN at ({row}, {class})"
        );
    }

    let mut strictly_wider = 0usize;
    for ((row, class), &narrow) in se_conditional.indexed_iter() {
        let wide = se_corrected[[row, class]];
        assert!(
            wide >= narrow - 1e-15,
            "the corrected band is NARROWER than the conditional one at \
             ({row}, {class}): {wide} < {narrow} — a variance component cannot subtract"
        );
        if wide > narrow {
            strictly_wider += 1;
        }
    }
    assert!(
        strictly_wider * 2 > se_conditional.len(),
        "the retained correction reached fewer than half the published \
         (row, class) spreads ({strictly_wider} of {}); a correction that is \
         stored but never contracted through the response Jacobian is the \
         original defect wearing a new field",
        se_conditional.len(),
    );

    // The factorised kernel against the literal contraction. The shipped path
    // never forms `g`: it exploits `g_c = u_c ⊗ x` and contracts an `M × M`
    // Gram. This rebuilds the full length-`d` Jacobian and evaluates `gᵀ C g`
    // entry by entry, which is the definition the shortcut has to reproduce.
    let p = model.p_per_class;
    let m = model.n_active_classes;
    let k = model.class_levels.len();
    let d = p * m;
    let beta = model.coefficients_active().expect("coefficients");
    let mut worst_relative = 0.0_f64;
    for row in 0..x.nrows() {
        let design_row = x.row(row);
        let eta: Vec<f64> = (0..m)
            .map(|a| (0..p).map(|i| design_row[i] * beta[[i, a]]).sum::<f64>())
            .collect();
        let probabilities = softmax_with_reference(&eta);
        for class in 0..k {
            let mut jacobian = vec![0.0_f64; d];
            for a in 0..m {
                let delta = if class == a { 1.0 } else { 0.0 };
                let weight = probabilities[class] * (delta - probabilities[a]);
                for i in 0..p {
                    jacobian[a * p + i] = weight * design_row[i];
                }
            }
            let mut expected = 0.0_f64;
            for i in 0..d {
                if jacobian[i] == 0.0 {
                    continue;
                }
                let mut acc = 0.0_f64;
                for j in 0..d {
                    acc += correction[[i, j]] * jacobian[j];
                }
                expected += jacobian[i] * acc;
            }
            let published =
                se_corrected[[row, class]].powi(2) - se_conditional[[row, class]].powi(2);
            let denominator = expected.abs().max(published.abs()).max(1e-300);
            worst_relative = worst_relative.max((published - expected).abs() / denominator);
        }
    }
    assert!(
        worst_relative < 1e-8,
        "the factorised response-scale correction disagrees with the literal \
         gᵀCg contraction by relative {worst_relative:e}"
    );
}

#[test]
fn the_multinomial_fit_retains_the_rho_uncertainty_correction_2612() {
    let model = fit_smooth_three_class(11);

    let correction = model.smoothing_correction().expect(
        "a converged REML fit with outer ρ curvature must retain the correction; \
                 its absence is exactly the defect #2612 names",
    );
    let conditional = model
        .coefficient_covariance()
        .expect("conditional covariance");
    assert_eq!(
        correction.dim(),
        conditional.dim(),
        "the correction and the conditional covariance must live in the same frame"
    );

    // Symmetry and a non-negative diagonal: `C = J V_ρ Jᵀ` with `V_ρ` PSD, so a
    // negative variance on any coordinate would mean the assembly, not the
    // model.
    let d = conditional.nrows();
    let scale = correction
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        .max(f64::MIN_POSITIVE);
    for i in 0..d {
        assert!(
            correction[[i, i]] >= -1e-12 * scale,
            "smoothing correction has a negative variance at coordinate {i}: {}",
            correction[[i, i]]
        );
        for j in 0..d {
            assert!(
                (correction[[i, j]] - correction[[j, i]]).abs() <= 1e-10 * scale,
                "smoothing correction is not symmetric at ({i}, {j}): {} vs {}",
                correction[[i, j]],
                correction[[j, i]],
            );
        }
    }

    // The correction is not the zero matrix. A zero correction would make the
    // response-scale widening gate vacuously true while reproducing the defect
    // exactly.
    assert!(
        correction.iter().any(|value| value.abs() > 0.0),
        "the retained correction is identically zero, so the corrected and \
         conditional definitions cannot differ and this gate proves nothing"
    );
}

#[test]
fn a_requested_correction_a_model_does_not_carry_is_an_error_2612() {
    let mut model = fit_smooth_three_class(11);
    let x = model.training_design().expect("training design");

    // Strip the correction, exactly as a fit whose outer solve retained no ρ
    // curvature would arrive.
    model.smoothing_correction_flat = None;
    assert!(
        model.smoothing_correction().is_none(),
        "stripping the payload must strip the accessor"
    );

    let refused = model.predict_probabilities_with_se_in_mode(
        x.view(),
        InferenceCovarianceMode::SmoothingCorrected,
    );
    assert!(
        refused.is_err(),
        "SmoothingCorrected on a model with no correction must REFUSE; silently \
         serving the conditional band is how a caller ends up with a narrower \
         interval than it asked for and no way to know"
    );

    // ... and the conditional definition still works, and announces itself.
    let (_, _, source) = model
        .predict_probabilities_with_se_and_source(x.view())
        .expect("the conditional band is still publishable");
    assert_eq!(
        source,
        InferenceCovarianceMode::Conditional,
        "a model without a correction must report the definition it actually used"
    );
}

/// #1082: decisions about the predictive's missing mass are per row. Every row
/// publishes its renormalized mean and its measured defect `d_row`, and a
/// `(1 − α)` interval declines, typed, exactly on the rows with `d_row > α`,
/// because that much of the row's posterior is unaccounted for. The worst row
/// declines at `α = d_max/2` and publishes at `α = 2·d_max`, and every other row
/// follows the same rule at both levels.
#[test]
fn a_row_whose_missing_mass_exceeds_alpha_declines_only_its_own_interval_1082() {
    let model = fit_smooth_three_class(11);
    let data = encode_recordswith_inferred_schema(
        vec!["x".to_string(), "y".to_string()],
        smooth_three_class(11),
    )
    .expect("encode three-class dataset");
    let means =
        predict_multinomial_formula(&model, &data).expect("point means publish for every row");
    let measured = predict_multinomial_formula_with_intervals(&model, &data, 0.95)
        .expect("the interval surface publishes on this fixture");
    assert_eq!(
        measured.mass_defect.len(),
        means.nrows(),
        "one measured defect per row"
    );
    let mut worst_row = 0usize;
    let mut worst = 0.0_f64;
    for (row, &defect) in measured.mass_defect.iter().enumerate() {
        if defect > worst {
            worst = defect;
            worst_row = row;
        }
    }
    assert!(
        worst > 0.0 && worst < 0.5,
        "premise: the fixture's worst row has a resolvable defect below one half, got {worst:e}"
    );

    for requested_alpha in [0.5 * worst, 2.0 * worst] {
        let intervals =
            predict_multinomial_formula_with_intervals(&model, &data, 1.0 - requested_alpha)
                .expect("per-row intervals");
        assert_eq!(
            intervals.mean, means,
            "the interval route must publish the point route's means on every row"
        );
        // The α production judges at is `1 − level`, which is `requested_alpha` only
        // to rounding.
        let alpha = 1.0 - intervals.level;
        let mut published = 0usize;
        for (row, decline) in intervals.declined.iter().enumerate() {
            let defect = intervals.mass_defect[row];
            match decline {
                Some(MultinomialIntervalDecline::MassDefectExceedsAlpha {
                    mass_defect,
                    alpha: carried,
                }) => {
                    assert!(
                        defect > alpha,
                        "row {row} declined with defect {defect:e} ≤ α = {alpha:e}"
                    );
                    assert_eq!(
                        (*mass_defect, *carried),
                        (defect, alpha),
                        "row {row}'s decline must carry its own defect and the α it was judged at"
                    );
                    for values in [
                        &intervals.mean_lower,
                        &intervals.mean_upper,
                        &intervals.standard_error,
                    ] {
                        assert!(
                            values.row(row).iter().all(|value| value.is_nan()),
                            "declined row {row} must publish no interval numbers"
                        );
                    }
                }
                Some(other) => panic!(
                    "row {row} declined for a reason other than its missing mass: {other:?}"
                ),
                None => {
                    assert!(
                        defect <= alpha,
                        "row {row} published with defect {defect:e} > α = {alpha:e}"
                    );
                    for values in [&intervals.mean_lower, &intervals.mean_upper] {
                        assert!(
                            values.row(row).iter().all(|value| value.is_finite()),
                            "published row {row} must carry a finite interval"
                        );
                    }
                    published += 1;
                }
            }
        }
        if requested_alpha < worst {
            assert!(
                intervals.declined[worst_row].is_some(),
                "the worst row (defect {worst:e}) must decline at α = {alpha:e}"
            );
            assert!(
                published > 0,
                "premise: some row's defect is at most half the worst, so the decline is per row"
            );
        } else {
            assert_eq!(
                published,
                means.nrows(),
                "no row's defect exceeds twice the worst, so every interval publishes at α = {alpha:e}"
            );
        }
    }
}

