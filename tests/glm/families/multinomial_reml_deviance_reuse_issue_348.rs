//! Regression test for issue #348.
//!
//! `fit_penalized_multinomial_formula` used to rebuild `η = Xβ` row-by-row and
//! recompute the softmax deviance from scratch after the REML driver returned.
//! That post-fit work was replaced by reading the unpenalized deviance straight
//! off `UnifiedFitResult.log_likelihood` (`deviance = −2 · log_lik`), and the
//! orphaned `block_states` arity guard was dropped.
//!
//! The contract that makes the reuse legitimate is that the recorded
//! `log_likelihood` is the *unpenalized* softmax log-likelihood at `β̂` — i.e.
//! the stored `model.deviance` must equal `−2 · Σ_i log p(y_i | x_i)` computed
//! independently from the fitted softmax probabilities. This test fits a real
//! 3-class formula through the same entry point the FFI uses and checks that
//! identity, so any future change that lets `log_likelihood` drift away from the
//! unpenalized deviance (e.g. accidentally folding in the penalty term) is
//! caught.
//!
//! # Which probabilities the oracle reads, and why that had to change (gam#2612)
//!
//! `p(y_i | x_i)` in the contract above is `softmax(η̂)_{y_i}` — the MODE's own
//! probability, the quantity `−2 · log L(β̂)` is defined against. This test read
//! it from `predict_multinomial_formula`, which was that function when #348 was
//! written and is not any more: gam#2612 made the default predict path publish
//! the posterior MEAN `E[softmax(η)]`, computed as a ratio of normalising
//! constants. The mode's own `softmax(η̂)` is recomputed below from the training
//! design and active coefficients the payload publishes.
//!
//! So the assertion became a comparison of two different estimands and failed on
//! their difference — `7.502289` against `7.110750` on this fixture — which is
//! not the drift it exists to catch. Nothing about `log_likelihood` had moved;
//! the oracle had. The estimand-matched pair is asserted below, and the
//! MISMATCHED pair is asserted to differ, so this test can never quietly go back
//! to comparing a mode against a posterior mean and reporting the gap as a
//! defect in the deviance.

use csv::StringRecord;
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// 3-class categorical response with one numeric feature that separates the
/// classes along `x`, so the softmax fit converges to a non-degenerate optimum.
fn categorical_dataset() -> (gam::data::EncodedDataset, Vec<String>) {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut labels = Vec::new();
    let rows = (0..30)
        .map(|i| {
            let x = -3.0 + 6.0 * i as f64 / 29.0;
            // Three contiguous bands in x → classes "a" / "b" / "c".
            let label = if x < -1.0 {
                "a"
            } else if x < 1.0 {
                "b"
            } else {
                "c"
            };
            labels.push(label.to_string());
            StringRecord::from(vec![x.to_string(), label.to_string()])
        })
        .collect::<Vec<_>>();
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode categorical dataset");
    (data, labels)
}

#[test]
fn multinomial_formula_deviance_equals_independent_softmax_recompute() {
    init_parallelism();
    let (data, labels) = categorical_dataset();

    // Same entry point + tuning the Python FFI uses (fit_multinomial_formula_pyfunc).
    let config = FitConfig::default();
    let model =
        fit_penalized_multinomial_formula(&MultinomialFitRequest::new(&data, "y ~ x", &config))
            .expect("multinomial formula fit must succeed");

    // Independently recompute the unpenalized deviance from the fitted softmax
    // probabilities on the training rows — exactly the quantity the deleted
    // row-by-row `η = Xβ` + softmax rebuild used to produce, which is
    // `softmax(η̂)` and therefore the PLUG-IN path (see the module doc).
    let deviance_from = |probs: &ndarray::Array2<f64>, what: &str| -> f64 {
        assert_eq!(probs.nrows(), labels.len(), "{what}: row count");
        assert_eq!(probs.ncols(), model.class_levels.len(), "{what}: col count");
        let mut total = 0.0_f64;
        for (i, label) in labels.iter().enumerate() {
            let class_idx = model
                .class_levels
                .iter()
                .position(|lvl| lvl == label)
                .expect("training label must be one of the fitted class levels");
            let p = probs[[i, class_idx]];
            assert!(
                p.is_finite() && p > 0.0,
                "{what}: prob p[{i}] must be finite & positive (got {p})"
            );
            total += p.ln();
        }
        -2.0 * total
    };

    // The mode's own probability `softmax(η̂)` on the training rows, rebuilt from
    // the payload's training design and active coefficients with the reference
    // class's `η = 0` last.
    let plugin = {
        let design = model.training_design().expect("training design");
        let beta = model.coefficients_active().expect("active coefficients");
        let eta = design.dot(&beta);
        let active = eta.ncols();
        let mut probs = ndarray::Array2::<f64>::zeros((eta.nrows(), active + 1));
        for (row, mut out) in eta.rows().into_iter().zip(probs.rows_mut()) {
            let shift = row.iter().copied().fold(0.0_f64, f64::max);
            let partition =
                (-shift).exp() + row.iter().map(|&value| (value - shift).exp()).sum::<f64>();
            for (class, &value) in row.iter().enumerate() {
                out[class] = (value - shift).exp() / partition;
            }
            out[active] = (-shift).exp() / partition;
        }
        probs
    };
    let recomputed_deviance = deviance_from(&plugin, "plug-in");

    assert!(
        model.deviance.is_finite(),
        "stored deviance must be finite (got {})",
        model.deviance
    );
    // The reuse (`−2 · log_lik`) must equal the from-scratch softmax recompute.
    assert!(
        (model.deviance - recomputed_deviance).abs() <= 1.0e-7 * (1.0 + recomputed_deviance.abs()),
        "model.deviance ({}) must equal the independent −2·Σ log p(y_i) recompute ({}); \
         a mismatch means UnifiedFitResult.log_likelihood is no longer the unpenalized softmax \
         log-likelihood the post-fit reuse depends on (issue #348)",
        model.deviance,
        recomputed_deviance
    );

    // The teeth on the oracle itself (gam#2612): the DEFAULT predict path
    // publishes the posterior mean `E[softmax(η)]`, which is a different
    // estimand, and reading it here is what made this gate red on a difference
    // that is not a defect. Assert the two estimands are genuinely distinct on
    // this fixture, so a future edit that points the oracle back at the default
    // path fails HERE, naming the estimand, instead of failing above as if the
    // deviance had drifted.
    let posterior_mean =
        predict_multinomial_formula(&model, &data).expect("posterior-mean probabilities");
    let posterior_mean_deviance = deviance_from(&posterior_mean, "posterior-mean");
    assert!(
        (posterior_mean_deviance - recomputed_deviance).abs()
            > 1.0e-7 * (1.0 + recomputed_deviance.abs()),
        "the posterior-mean recompute ({posterior_mean_deviance}) and the plug-in recompute \
         ({recomputed_deviance}) are indistinguishable on this fixture, so the assertion above \
         no longer discriminates between the mode's likelihood and a posterior average — pick a \
         fixture where posterior width is visible, or this gate proves nothing"
    );
}
