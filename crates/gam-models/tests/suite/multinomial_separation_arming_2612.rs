//! #2612: the Jeffreys/Firth proper prior must be armed by **separation**, not
//! by the ordinary shape of a penalized smooth basis.
//!
//! `fit_penalized_multinomial_formula` engages the Firth/Jeffreys prior
//! CONDITIONALLY (#715 arm (b) / #753) because the prior is not free: it pulls
//! fitted class probabilities toward the uniform simplex `1/K`, which #715
//! measured as a real truth-RMSE cost on interior data, and it routes the fit
//! through a lane whose outer curvature certificate is deliberately weaker (the
//! armed Hessian omits `D²_β H_Φ` by construction). So a false positive costs a
//! whole re-solve on a biased objective and a certificate that cannot be as
//! strong.
//!
//! The decision is taken from the reduced conditioning gate at the certified
//! mode, whose absolute arm fires when the worst-determined direction holds less
//! than **one observation-equivalent** of curvature. Until #2612 that gate was
//! handed the bare Fisher information `H`. A `k`-dimensional spline basis has
//! high-frequency directions the data barely resolve BY CONSTRUCTION — that is
//! precisely why they are penalized — so on labels DRAWN from a smooth softmax
//! truth, with every class keeping appreciable probability everywhere and
//! nothing separating anywhere, the certificate read
//!
//! ```text
//!   lambda_min = 4.051e-1   lambda_max = 1.575e2   ratio = 2.572e-3
//!   Jeffreys gate weight = 1   =>  "separation evidence"  =>  Firth/Jeffreys refit
//! ```
//!
//! The relative arm was nowhere near firing (`2.6e-3` against a `1e-6` clear
//! knot); the verdict was the absolute arm reading a *penalized* direction as if
//! nothing were holding it. "Arm only on separation evidence" was therefore
//! unconditional in practice on any multinomial GAM carrying a smooth.
//!
//! The distinction #715 derives is exactly the one that was missing: a direction
//! `v` is beyond `λ`'s reach only when `S v = 0`, because `(H + S_λ)v = Hv +
//! λSv`. The certificate now forms `H + S_λ` at the λ the mode was certified at.
//!
//! The three arms below are the discriminator, and a repair that satisfies any
//! two of them is wrong in a way those two could not see:
//!
//! 1. data that do not separate must be fitted with the prior **disarmed**;
//! 2. data that genuinely do separate must still **arm** it;
//! 3. and on data that separate *and* carry a smooth, the published
//!    probabilities must be **calibrated** — because a verdict can be right and
//!    the fit still wrong.
//!
//! Arm 2 uses a design with NO penalized term at all — `S_λ = 0`, so `H + S_λ =
//! H` identically — which is the sharpest available statement that the repair
//! did not simply make the certificate quieter: on that geometry the two
//! certificates are the same matrix and the verdict must be unchanged.
//!
//! Arm 3 exists because arms 1 and 2 assert a VERDICT and the estimand is a
//! PROBABILITY. On a quasi-separated fit carrying a smooth the certificate was
//! right — a direction really is undetermined — and the term it armed then
//! acted on the whole basis anyway, because `jeffreys_antiderivative` acts
//! wherever a reduced eigenvalue is under `CONDITIONING_GATE_ABSOLUTE_CLEAR =
//! 16` observation-equivalents and a quasi-separated softmax puts `W =
//! diag(p) − ppᵀ ≈ 0.005` on every row, so NO direction reaches that scale.
//! `#2612`'s repair is therefore about the SUBSPACE the verdict is taken over —
//! `ker(S_λ)`, the directions no `λ` reaches — and the calibration gap is what
//! says whether that landed.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, MultinomialSavedModel, fit_penalized_multinomial_formula,
    predict_multinomial_formula,
};

const CLASS_NAMES: [&str; 3] = ["a", "b", "c"];

/// The true class probabilities of the generating softmax at `(x1, x2)`.
fn truth(x1: f64, x2: f64) -> [f64; 3] {
    let scores = [
        0.6 * x1 - 0.3 * x2,
        -0.4 * x1 + 0.5 * x2,
        0.2 * (x1 * x1 - x2 * x2) * 0.25,
    ];
    let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - shift).exp()).collect();
    let total: f64 = weights.iter().sum();
    [
        weights[0] / total,
        weights[1] / total,
        weights[2] / total,
    ]
}

fn covariates(index: usize) -> (f64, f64) {
    // Two coprime golden-ratio-style strides: the covariates stay deterministic
    // and well spread, so only the labels carry randomness.
    (
        -2.0 + 4.0 * (((index as f64) * 0.618_033_988_749_894_8) % 1.0),
        -2.0 + 4.0 * (((index as f64) * 0.414_213_562_373_095_1) % 1.0),
    )
}

/// Labels DRAWN from the smooth softmax truth above.
///
/// Drawing rather than taking the argmax is load-bearing: an argmax label is a
/// deterministic function of `x`, so the classes would be exactly separated by
/// their own decision boundaries and this fixture would be on the separation
/// lane by construction — testing nothing. The draw uses a small deterministic
/// LCG: reproducible, no RNG dependency, no seed to choose.
fn drawn_records(n: usize) -> Vec<StringRecord> {
    let mut rows = Vec::with_capacity(n);
    let mut lcg: u64 = 0x2612_2612_2612_2612;
    let mut next_unit = || -> f64 {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for index in 0..n {
        let (x1, x2) = covariates(index);
        let probabilities = truth(x1, x2);
        let mut draw = next_unit();
        let mut label = CLASS_NAMES.len() - 1;
        for (class, probability) in probabilities.iter().enumerate() {
            if draw < *probability {
                label = class;
                break;
            }
            draw -= probability;
        }
        rows.push(StringRecord::from(vec![
            x1.to_string(),
            x2.to_string(),
            CLASS_NAMES[label].to_string(),
        ]));
    }
    rows
}

/// A QUASI-separated three-class design with NO penalized term.
///
/// `y` is a step function of `x1` — so the softmax MLE is pushed toward
/// `|η| → ∞` and the identifiable-span information collapses — except for a
/// handful of deliberately mislabelled rows near each boundary, which keep the
/// MLE finite. That is the geometry the penguins witness has and the geometry
/// #715 was written for: quasi-complete separation, not complete separation.
///
/// With no smooth in the formula there is no penalty at all, so `S_λ` is exactly
/// the zero operator and `H + S_λ` IS `H`. Nothing about the #2612 repair can
/// reach this fixture, which is the point: the two certificates are the same
/// matrix here and the verdict must be unchanged.
///
/// (PERFECT separation — every row on the correct side — is deliberately NOT
/// used: on this formula path the Firth-armed refit of a fully separated
/// unpenalized 3-class design does not converge, reporting "no-smoothing inner
/// solve: coefficient optimization did not converge after 80 cycles". That is a
/// pre-existing defect on the arming path and not the subject of this file — the
/// arming decision it is downstream of is identical before and after #2612,
/// because `S_λ = 0` makes the two certificates the same matrix — so it is
/// recorded here rather than used as this test's fixture.)
fn separated_records(n: usize) -> Vec<StringRecord> {
    let mut rows = Vec::with_capacity(n);
    for index in 0..n {
        let x1 = -3.0 + 6.0 * (index as f64) / ((n - 1) as f64);
        // A second covariate that is NOT a multiple of the first — an aliased
        // column would be dropped by the identifiability audit and the fixture
        // would be testing a one-covariate model without saying so.
        let x2 = (2.7 * x1).sin();
        let mut class = if x1 < -1.0 {
            0
        } else if x1 < 1.0 {
            1
        } else {
            2
        };
        // Two overlap rows per boundary, at fixed indices: enough that no
        // hyperplane separates the classes exactly, few enough that the
        // identifiable-span information stays far below one observation-
        // equivalent.
        if index % 23 == 7 {
            class = (class + 1) % CLASS_NAMES.len();
        }
        rows.push(StringRecord::from(vec![
            x1.to_string(),
            x2.to_string(),
            CLASS_NAMES[class].to_string(),
        ]));
    }
    rows
}

fn fit(records: Vec<StringRecord>, formula: &str) -> MultinomialSavedModel {
    let headers = ["x1", "x2", "y"].into_iter().map(str::to_string).collect();
    let data =
        encode_recordswith_inferred_schema(headers, records).expect("encode multinomial dataset");
    let config = FitConfig::default();
    fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &data,
        formula,
        config: &config,
        init_lambda: 1.0,
        max_iter: usize::MAX,
        tol: 1e-8,
    })
    .unwrap_or_else(|error| panic!("multinomial formula fit `{formula}`: {error}"))
}

/// Arm 1 — a smooth multinomial on data that does not separate keeps the prior
/// disarmed, and the fitted surface recovers the truth it was drawn from.
#[test]
fn a_smooth_multinomial_that_does_not_separate_keeps_the_prior_disarmed_2612() {
    let model = fit(drawn_records(600), "y ~ s(x1, k=6) + s(x2, k=6)");

    assert_eq!(
        model.separation_evidence, None,
        "labels drawn from a smooth softmax truth separate nowhere, so the Jeffreys/Firth \
         proper prior must stay disarmed and the published coefficients must be the unbiased \
         penalized-REML mode. Armed with: {:?}",
        model.separation_evidence,
    );

    // The estimand bar, from a different direction: the fit is compared to the
    // probabilities it was DRAWN from, so this is reference-free and it is the
    // quantity the prior would move. An unnecessary proper prior pulls every row
    // toward `1/K`, which shows up here as bias, not as noise.
    let headers = ["x1", "x2", "y"].into_iter().map(str::to_string).collect();
    let holdout_rows: Vec<StringRecord> = (0..600)
        .map(|index| {
            // Interleaved, so the held-out covariates are inside the training
            // range but are not training points.
            let (x1, x2) = covariates(index + 100_000);
            StringRecord::from(vec![x1.to_string(), x2.to_string(), "a".to_string()])
        })
        .collect();
    let holdout = encode_recordswith_inferred_schema(headers, holdout_rows)
        .expect("encode held-out frame");
    let predicted = predict_multinomial_formula(&model, &holdout).expect("multinomial predict");
    assert_eq!(predicted.nrows(), 600, "held-out prediction rows");

    let level_of = |name: &str| {
        model
            .class_levels
            .iter()
            .position(|level| level == name)
            .unwrap_or_else(|| panic!("class {name} missing from {:?}", model.class_levels))
    };
    let columns: Vec<usize> = CLASS_NAMES.iter().map(|name| level_of(name)).collect();

    let mut squared = 0.0_f64;
    // Shrinkage toward `1/K` is a BIAS: it pulls the largest true probability
    // down on every row. So the statistic that detects it is the SIGNED MEAN of
    // the argmax-class gap `truth − fitted` over the held-out rows, with its own
    // standard error — not a one-sided maximum, which is positive by
    // construction for any estimator and grows with the row count. Both are
    // collected; only the mean is asserted, and the extremes are printed so a
    // reader can see which one a number is about (#2612).
    let mut argmax_gaps: Vec<f64> = Vec::with_capacity(600);
    for (row, index) in (0..600).enumerate() {
        let (x1, x2) = covariates(index + 100_000);
        let expected = truth(x1, x2);
        let argmax = expected
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (c, v)| {
                if *v > best.1 { (c, *v) } else { best }
            })
            .0;
        let mut mass = 0.0_f64;
        for class in 0..3 {
            let p = predicted[[row, columns[class]]];
            mass += p;
            squared += (p - expected[class]).powi(2);
        }
        argmax_gaps.push(expected[argmax] - predicted[[row, columns[argmax]]]);
        assert!(
            (mass - 1.0).abs() <= 1e-6,
            "published probabilities must be a simplex, row {row} sums to {mass}"
        );
    }
    let rmse = (squared / (600.0 * 3.0)).sqrt();
    let rows = argmax_gaps.len() as f64;
    let mean_gap = argmax_gaps.iter().sum::<f64>() / rows;
    let gap_variance =
        argmax_gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / (rows - 1.0);
    let gap_standard_error = (gap_variance / rows).sqrt();
    let worst_shrinkage = argmax_gaps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let worst_inflation = argmax_gaps.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "#2612 disarmed smooth fit: held-out truth-RMSE = {rmse:.5}; argmax gap (truth-fitted) \
         mean = {mean_gap:+.5} (s.e. {gap_standard_error:.5}, t = {:+.2}), worst shrink = \
         {worst_shrinkage:+.5}, worst inflate = {worst_inflation:+.5}",
        mean_gap / gap_standard_error,
    );
    // The generating softmax keeps every class between roughly 0.15 and 0.6, so
    // an intercept-only fit scores about 0.13 here and a fit shrunk to the
    // uniform simplex scores worse still. The bar is loose enough to be about
    // the estimand and not about the sampling noise of one draw.
    assert!(
        rmse <= 0.05,
        "the fitted surface must recover the softmax truth it was drawn from; \
         truth-RMSE = {rmse:.4}"
    );
    // THE BIAS BAR, and why it is this quantity (#2612).
    //
    // This assertion used to read `worst_shrinkage <= 0.10`, justified as "that
    // one-sided gap is what a proper prior armed on non-separated data costs".
    // A one-sided maximum over 600 rows cannot support that reading: the largest
    // of many mean-zero errors is positive by construction, and it grows with
    // the row count, so the statistic conflates "the estimator is biased
    // downward on the winning class" with "the estimator is unbiased and 600
    // rows have a tail". Measured on this fixture with the prior correctly
    // DISARMED, the gap is symmetric —
    //
    // ```text
    //   mean = -0.00294 (s.e. 0.00179, t = -1.64)
    //   worst shrink = +0.11414     worst inflate = -0.13552
    // ```
    //
    // — the largest excursion is an INFLATION, larger than the shrinkage the old
    // bar refused on, and the mean says the winning class is if anything
    // slightly sharper than the truth. So the max was reading the tail of
    // ordinary estimation noise at truth-RMSE 0.037.
    //
    // `0.02` is not a widened `0.10`; it is a bar on a different and much
    // tighter quantity. It is eleven standard errors of the measured mean, and
    // it sits far below what a false-positive arming costs: the same one-sided
    // gap on the Firth-armed penguins witness is `0.137` (mean argmax
    // probability 0.828 against accuracy 0.965), i.e. seven times this bar. A
    // prior armed on data that does not separate cannot hide under it.
    assert!(
        mean_gap.abs() <= 0.02,
        "the disarmed fit must be UNBIASED on the winning class: mean argmax gap \
         (truth − fitted) = {mean_gap:+.5} over {rows} held-out rows (s.e. \
         {gap_standard_error:.5}). A systematic pull toward 1/K is what an unnecessary proper \
         prior costs, and it shows up here as a nonzero mean, not as a large maximum \
         (worst shrink {worst_shrinkage:+.5}, worst inflate {worst_inflation:+.5})"
    );
}

/// Arm 2 — genuine quasi-separation still arms the prior, on a design where the
/// repair provably cannot act (`S_λ = 0`, so the certificate reads the same
/// matrix it always did).
#[test]
fn genuine_separation_still_arms_the_prior_2612() {
    const ROWS: usize = 300;
    // b7b874a2a gave formula linear effects the null-recovery ridge by default (SPEC rules 12
    // and 14), so a bare `x1 + x2` now carries one penalty per effect. This arm needs a design
    // the #2612 repair provably cannot act on, `S_lambda = 0`, so it opts both effects out.
    let model = fit(
        separated_records(ROWS),
        "y ~ linear(x1, double_penalty=false) + linear(x2, double_penalty=false)",
    );

    assert!(
        model.smooth_term_spans.is_empty() && model.lambdas.is_empty(),
        "arm 2 must carry no penalty, so that `H + S_lambda` is `H` identically; got {} span(s) \
         and {} lambda",
        model.smooth_term_spans.len(),
        model.lambdas.len(),
    );
    let evidence = model
        .separation_evidence
        .as_deref()
        .expect("a quasi-separated three-class design must arm the Jeffreys/Firth prior");
    assert!(
        evidence.contains("lambda_min"),
        "the recorded evidence must be the certificate it was taken from, not a flag: {evidence}"
    );

    // And the armed fit is a fit: finite, interior, and classifying its own
    // near-separable training rows correctly. A prior that bounded the runaway
    // MLE by destroying the signal would pass the arming assertion above and
    // fail here.
    let headers = ["x1", "x2", "y"].into_iter().map(str::to_string).collect();
    let frame = encode_recordswith_inferred_schema(headers, separated_records(ROWS))
        .expect("encode separated frame");
    let predicted = predict_multinomial_formula(&model, &frame).expect("multinomial predict");
    let mut correct = 0usize;
    for (row, record) in separated_records(ROWS).iter().enumerate() {
        let mut best = (0usize, f64::NEG_INFINITY);
        for class in 0..model.class_levels.len() {
            let p = predicted[[row, class]];
            assert!(
                p > 0.0 && p < 1.0,
                "the Firth-armed fit must stay strictly interior, got p={p} at row {row}"
            );
            if p > best.1 {
                best = (class, p);
            }
        }
        if model.class_levels[best.0] == record[2] {
            correct += 1;
        }
    }
    // 13 of the 300 rows are the deliberate overlap that keeps the MLE finite,
    // so a fit that recovered the step boundary exactly scores 287/300. The bar
    // sits below that and far above chance (100/300).
    eprintln!("#2612 armed quasi-separated fit: {correct}/{ROWS} training rows classified");
    assert!(
        correct >= 270,
        "the armed fit must still classify its own near-separable training rows: \
         {correct}/{ROWS}"
    );
}

/// A genuinely QUASI-SEPARATED design that carries a SMOOTH, banded along one
/// covariate so `W = diag(p) − ppᵀ → 0` on almost every row — the penguins
/// geometry at a size that fits in seconds.
///
/// The two boundaries are perturbed inside a thin overlap band, so a handful of
/// rows sit on the wrong side of their own boundary and the maximiser is finite
/// (quasi-, not complete, separation).
fn banded_records(indices: &[usize]) -> (Vec<StringRecord>, Vec<usize>) {
    /// Overlap half-width of the two class boundaries, in covariate units.
    const OVERLAP: f64 = 0.06;
    // The van der Corput sequence in base 2 on `[-1, 1]`: deterministic and low
    // discrepancy, so the fixture is bit-reproducible and every re-measurement
    // is of the code rather than of a draw.
    fn covariate(index: usize) -> f64 {
        let mut numerator = 0.0_f64;
        let mut denominator = 1.0_f64;
        let mut n = index + 1;
        while n > 0 {
            denominator *= 2.0;
            numerator += ((n % 2) as f64) / denominator;
            n /= 2;
        }
        2.0 * numerator - 1.0
    }
    let mut labels = Vec::with_capacity(indices.len());
    let records = indices
        .iter()
        .map(|&index| {
            let x = covariate(index);
            let wobble = if index % 7 == 0 { OVERLAP } else { -OVERLAP };
            let class = if x < -1.0 / 3.0 + wobble {
                0
            } else if x < 1.0 / 3.0 + wobble {
                1
            } else {
                2
            };
            labels.push(class);
            StringRecord::from(vec![x.to_string(), CLASS_NAMES[class].to_string()])
        })
        .collect();
    (records, labels)
}

fn banded_frame(indices: &[usize]) -> (gam_data::EncodedDataset, Vec<usize>) {
    let (records, labels) = banded_records(indices);
    let headers = ["x", "y"].into_iter().map(str::to_string).collect();
    (
        encode_recordswith_inferred_schema(headers, records).expect("encode banded frame"),
        labels,
    )
}

/// Arm 3 — the estimand bar, from the direction neither arming assertion can
/// see: a quasi-separated fit that carries a SMOOTH must be CALIBRATED.
///
/// Arms 1 and 2 assert the arming VERDICT. A verdict can be right and the
/// published probabilities still wrong, and on this geometry that is exactly
/// what happened: the certificate correctly identified a direction the data do
/// not determine, and the term it armed then acted on the WHOLE basis, because
/// `jeffreys_antiderivative` acts wherever a reduced eigenvalue is under
/// `CONDITIONING_GATE_ABSOLUTE_CLEAR = 16` observation-equivalents and a
/// quasi-separated softmax has `W = diag(p) − ppᵀ ≈ 0.005` per row, so NO
/// direction reaches that scale. Measured on this fixture before #2612's
/// subspace repair: held-out accuracy `0.9833` against a mean argmax
/// probability of `0.9032` — the fit says "90% sure" and is right 98% of the
/// time.
///
/// The bar is the CALIBRATION GAP, `mean argmax probability − accuracy`, which
/// is reference-free (no comparator implementation appears in it) and is the
/// defining property of a probabilistic classifier: a calibrated one predicts
/// its own accuracy. On 120 held-out rows at `p ≈ 0.98` the binomial standard
/// error of the accuracy is `sqrt(0.98·0.02/120) = 0.0128`, so `0.05` is about
/// four standard errors — loose enough that one draw's noise cannot fail it,
/// and six times tighter than the `-0.080` a full-span prior costs here.
#[test]
fn a_quasi_separated_smooth_fit_is_calibrated_2612() {
    let (train, _) = banded_frame(&(0..180).collect::<Vec<_>>());
    let (test, test_labels) = banded_frame(&(180..300).collect::<Vec<_>>());
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train,
        formula: "y ~ s(x, k=8)",
        config: &config,
        init_lambda: 1.0,
        max_iter: usize::MAX,
        tol: 1e-8,
    })
    .expect(
        "a quasi-separated three-class fit carrying one smooth must produce a fit; this fixture \
         produced none at all until the #2612 saddle-escape repair",
    );

    let columns: Vec<usize> = CLASS_NAMES
        .iter()
        .map(|name| {
            model
                .class_levels
                .iter()
                .position(|level| level == name)
                .unwrap_or_else(|| panic!("class {name} missing from {:?}", model.class_levels))
        })
        .collect();
    let predicted = predict_multinomial_formula(&model, &test).expect("held-out predict");
    assert_eq!(predicted.nrows(), test_labels.len(), "held-out rows");

    let mut correct = 0usize;
    let mut argmax_mass = 0.0_f64;
    for (row, &truth) in test_labels.iter().enumerate() {
        let mut best = (0usize, f64::NEG_INFINITY);
        for (class, &column) in columns.iter().enumerate() {
            let p = predicted[[row, column]];
            if p > best.1 {
                best = (class, p);
            }
        }
        argmax_mass += best.1;
        if best.0 == truth {
            correct += 1;
        }
    }
    let rows = test_labels.len() as f64;
    let accuracy = correct as f64 / rows;
    let mean_argmax = argmax_mass / rows;
    let calibration_gap = mean_argmax - accuracy;
    let accuracy_standard_error = (accuracy * (1.0 - accuracy) / rows).sqrt().max(1e-12);
    eprintln!(
        "#2612 quasi-separated smooth fit: separation_evidence={:?}\n#2612 quasi-separated \
         smooth fit: held-out accuracy={accuracy:.4} mean_argmax_p={mean_argmax:.4} \
         calibration_gap={calibration_gap:+.4} ({:+.1} s.e.)",
        model.separation_evidence,
        calibration_gap / accuracy_standard_error,
    );
    assert!(
        accuracy >= 0.90,
        "the banded classes are decided by `x` except inside a thin overlap band, so a competent \
         fit classifies well above 0.90; got {accuracy:.4}"
    );
    assert!(
        calibration_gap.abs() <= 0.05,
        "a probabilistic classifier must predict its own accuracy: mean argmax probability \
         {mean_argmax:.4} against held-out accuracy {accuracy:.4} is a calibration gap of \
         {calibration_gap:+.4} ({:+.1} binomial s.e.). A proper prior armed over the whole \
         identifiable span costs exactly this, and it is invisible to an arming-verdict \
         assertion",
        calibration_gap / accuracy_standard_error,
    );
}
