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
    // row-by-row `η = Xβ` + softmax rebuild used to produce.
    let probs = predict_multinomial_formula(&model, &data).expect("predict probabilities");
    assert_eq!(probs.nrows(), labels.len());
    assert_eq!(probs.ncols(), model.class_levels.len());

    let mut recomputed_deviance = 0.0_f64;
    for (i, label) in labels.iter().enumerate() {
        let class_idx = model
            .class_levels
            .iter()
            .position(|lvl| lvl == label)
            .expect("training label must be one of the fitted class levels");
        let p = probs[[i, class_idx]];
        assert!(
            p.is_finite() && p > 0.0,
            "softmax prob p[{i}] must be finite & positive (got {p})"
        );
        recomputed_deviance += p.ln();
    }
    recomputed_deviance *= -2.0;

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
}
