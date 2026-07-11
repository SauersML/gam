//! Regression for #1544: the multinomial saved model must expose one λ label
//! per *penalty component*, parallel to a single class block's λ slice.
//!
//! `MultinomialModel.summary()` paired smooth-term labels with the per-class λ
//! via `zip(term_labels, lam_chunk)`, assuming a 1:1 term↔λ mapping. The default
//! Marra–Wood double penalty emits TWO penalty components — hence two λ — per
//! smooth term per class (a primary wiggliness penalty and a null-space
//! shrinkage penalty), so the summary both raised a "lambda metadata mismatch"
//! guard error and, with the guard removed, would silently drop the null-space
//! λ. The fix records `MultinomialSavedModel::lambda_labels`, one descriptive
//! label per penalty component, exactly parallel to a single block's λ. This
//! test pins that cardinality contract at the Rust core, independent of the
//! Python renderer.

use csv::StringRecord;
use gam::families::multinomial::{MultinomialFitRequest, fit_penalized_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// 3-class categorical response with TWO numeric features, each smoothed. Two
/// double-penalty smooths give 4 penalty components per class — strictly more
/// than the 2 smooth terms — which is exactly the configuration that exposed the
/// truncation bug.
fn two_smooth_multinomial_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let n = 240usize;
    let rows = (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            let x = -3.0 + 6.0 * t;
            // A second, phase-shifted coordinate so s(z) is a genuine smooth.
            let z = (6.28 * t).sin() * 2.0;
            let label = if x + 0.5 * z < -1.0 {
                "a"
            } else if x + 0.5 * z < 1.0 {
                "b"
            } else {
                "c"
            };
            StringRecord::from(vec![
                format!("{x:.6}"),
                format!("{z:.6}"),
                label.to_string(),
            ])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode two-smooth multinomial dataset")
}

#[test]
fn multinomial_lambda_labels_are_one_per_penalty_component() {
    init_parallelism();
    let data = two_smooth_multinomial_dataset();
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        max_iter: 100,
        ..MultinomialFitRequest::new(&data, "y ~ s(x) + s(z)", &config)
    })
    .expect("two-smooth multinomial fit must succeed");

    let m = model.n_active_classes;
    assert!(m >= 1, "fixture must fit at least one active class");

    // Premise guard: the double penalty must give each block MORE λ than there
    // are smooth term spans, otherwise the test would not exercise the 1:1
    // assumption that #1544 broke.
    let per_block = &model.lambdas_per_block;
    assert_eq!(per_block.len(), m, "one λ-block per active class");
    let n_term_spans = model.smooth_term_spans.len();
    assert_eq!(n_term_spans, 2, "two smooth terms expected");
    assert!(
        per_block.iter().all(|&b| b > n_term_spans),
        "double penalty must emit >1 λ per smooth term per class; got \
         lambdas_per_block={per_block:?} for {n_term_spans} term spans"
    );

    // (1) CARDINALITY: exactly one label per penalty component in a class block.
    assert_eq!(
        model.lambda_labels.len(),
        per_block[0],
        "lambda_labels must have one entry per penalty component in a class block \
         (lambdas_per_block[0]={}); got {} labels: {:?}",
        per_block[0],
        model.lambda_labels.len(),
        model.lambda_labels
    );

    // Every active class block carries the SAME number of components in the
    // shared-design architecture, so the per-block label set covers each.
    assert!(
        per_block.iter().all(|&b| b == model.lambda_labels.len()),
        "every class block must carry one λ per recorded label; lambdas_per_block={per_block:?}, \
         labels={:?}",
        model.lambda_labels
    );

    // (2) CONTENT: both smooth terms must be named, and the double penalty's
    //     null-space shrinkage component must be distinguishable from the
    //     primary wiggliness component so neither is dropped in the rollup.
    let joined = model.lambda_labels.join(" | ");
    assert!(joined.contains("s(x)"), "labels must name s(x): {joined}");
    assert!(joined.contains("s(z)"), "labels must name s(z): {joined}");
    assert!(
        model.lambda_labels.iter().any(|l| l.contains("null space")),
        "the double penalty's null-space λ must carry a distinguishing role label: {joined}"
    );

    // (3) NO DROPS: pairing labels with the per-block λ slice must consume the
    //     whole slice — the truncation bug would leave labels (or λ) unmatched.
    let block0 = &model.lambdas[..per_block[0]];
    assert_eq!(
        block0.len(),
        model.lambda_labels.len(),
        "the first class block's λ slice and its labels must be the same length"
    );
    assert!(
        block0.iter().all(|v| v.is_finite() && *v >= 0.0),
        "selected λ must be finite and non-negative: {block0:?}"
    );
}
