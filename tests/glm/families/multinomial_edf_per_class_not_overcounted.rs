//! Regression: multinomial `edf_per_class` must be a genuine length-(K-1)
//! per-class effective-degrees-of-freedom vector, not the raw per-penalty-block
//! EDF list (which over-counts).
//!
//! `MultinomialSavedModel::edf_per_class` is documented as "per-active-class
//! effective degrees of freedom (hat-matrix trace), length K-1". It used to be
//! assigned `info.edf_by_block` verbatim — the REML inference block's per-PENALTY
//! vector, with one entry per (class, term, penalty) and each entry computed as
//! `rank(S_kk) − tr(H⁻¹ λ_kk S_kk)`. That has the wrong LENGTH (Σ_a n_blocks_a,
//! not K-1) and OVER-COUNTS the model EDF whenever several penalties share one
//! coefficient range: a default `s(x)` smooth carries a Marra-Wood double penalty
//! (≥2 penalty blocks over the same columns), so `Σ rank(S_kk) > p` and the
//! reported per-class EDF can blow past the class's own coefficient count (the
//! observed ~79 EDF for a ~24-coefficient model).
//!
//! The honest per-class EDF is `tr(F)` over each class's disjoint coefficient
//! block, `p_per_class − Σ_{kk∈class} tr_kk`, which is bounded by `p_per_class`
//! and whose K-1 entries sum to the model `edf_total`. This test pins the
//! decidable contract on the public saved model and rejects the over-count.

use csv::StringRecord;
use gam::families::multinomial::{MultinomialFitRequest, fit_penalized_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// 3-class categorical response with a numeric feature, fit with a penalized
/// SMOOTH of `x` per class. The default `s(x)` carries a double penalty, so each
/// class block has ≥2 penalty blocks over one coefficient range — exactly the
/// shared-range configuration that made the old per-block EDF over-count.
fn smooth_multinomial_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let n = 150usize;
    let rows = (0..n)
        .map(|i| {
            let x = -3.0 + 6.0 * i as f64 / (n as f64 - 1.0);
            // Smoothly-varying class bands so the per-class smooths are real.
            let label = if x < -1.0 {
                "a"
            } else if x < 1.0 {
                "b"
            } else {
                "c"
            };
            StringRecord::from(vec![format!("{x:.6}"), label.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode smooth multinomial dataset")
}

#[test]
fn multinomial_edf_per_class_is_per_class_not_per_block_overcount() {
    init_parallelism();
    let data = smooth_multinomial_dataset();
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        max_iter: 100,
        ..MultinomialFitRequest::new(&data, "y ~ s(x)", &config)
    })
    .expect("multinomial smooth fit must succeed");

    let m = model.n_active_classes; // K - 1
    let p_per_class = model.p_per_class;
    assert!(
        m >= 1 && p_per_class >= 1,
        "fixture must fit a non-trivial model"
    );

    // Premise guard: the default `s(x)` double penalty must give each class block
    // MORE than one penalty block, otherwise the per-block list would coincide
    // with the per-class list and the test would not exercise the over-count.
    let max_blocks_per_class = model.lambdas_per_block.iter().copied().max().unwrap_or(0);
    assert!(
        max_blocks_per_class >= 2,
        "fixture must produce ≥2 penalty blocks per class (double-penalty smooth) to \
         exercise the per-block over-count; got lambdas_per_block={:?}",
        model.lambdas_per_block
    );

    let edf = model
        .edf_per_class
        .as_ref()
        .expect("REML multinomial fit must report per-class EDF");

    // (1) LENGTH: genuinely per-class (K-1), NOT per-(class,term,penalty) block.
    //     The old code returned a vector of length Σ lambdas_per_block > K-1.
    assert_eq!(
        edf.len(),
        m,
        "edf_per_class must have one entry per active class (K-1={m}), not one per \
         penalty block (Σ lambdas_per_block={}). edf_per_class={edf:?}",
        model.lambdas_per_block.iter().sum::<usize>()
    );

    // (2) PER-CLASS BOUND: a class's EDF is tr(F) over its own p_per_class-wide
    //     coefficient block, so it lies in [0, p_per_class]. The over-count
    //     violated this (≈79 for p_per_class≈8). A small floor (>0) confirms the
    //     unpenalized null space (intercept) is always counted.
    for (a, &e) in edf.iter().enumerate() {
        assert!(
            e.is_finite() && e >= 0.0 && e <= p_per_class as f64 + 1e-6,
            "class {a} EDF {e} out of the structural range [0, p_per_class={p_per_class}] — \
             a per-class EDF cannot exceed the class's own coefficient count (the per-block \
             over-count bug)."
        );
    }

    // (3) TOTAL: the per-class EDF sums to the model EDF, which is bounded by the
    //     total coefficient count m·p_per_class and far below the old per-block
    //     over-count. (The honest fixed point is Σ_a edf_a = edf_total = tr(F).)
    let total: f64 = edf.iter().sum();
    let p_total = (m * p_per_class) as f64;
    assert!(
        total.is_finite() && total >= 0.0 && total <= p_total + 1e-6,
        "Σ edf_per_class = {total} must be a valid model EDF in [0, m·p_per_class={p_total}]; \
         the per-block sum over-counted well past this."
    );
    // A real penalized smooth spends more than just the unpenalized null space,
    // so the model EDF clears the bare per-class intercept count.
    assert!(
        total > m as f64,
        "Σ edf_per_class = {total} should exceed the m={m} unpenalized intercepts for a \
         genuine per-class smooth fit"
    );

    // (4) PER-PENALTY companion vector: distinct from the per-class total above.
    //     `edf_per_penalty` carries ONE entry per smoothing parameter (length =
    //     Σ lambdas_per_block = lambdas.len()), each the clamped per-block trace
    //     EDF `rank(S_k) − λ_k·tr(H⁻¹ S_k)` ∈ [0, rank(S_k)]. For this
    //     double-penalty fixture (≥2 blocks per class) it is STRICTLY LONGER than
    //     the per-class vector — that length gap is exactly what makes it a
    //     separate field rather than a reshaping of `edf_per_class`.
    let edf_pen = model
        .edf_per_penalty
        .as_ref()
        .expect("REML multinomial fit must report per-penalty EDF");
    let n_pen: usize = model.lambdas_per_block.iter().sum();
    assert_eq!(
        edf_pen.len(),
        n_pen,
        "edf_per_penalty must carry one entry per smoothing parameter \
         (Σ lambdas_per_block={n_pen}), not the per-class count K-1={m}. \
         edf_per_penalty={edf_pen:?}"
    );
    assert_eq!(
        edf_pen.len(),
        model.lambdas.len(),
        "edf_per_penalty must be aligned 1:1 with the flat lambdas vector"
    );
    assert!(
        edf_pen.len() > edf.len(),
        "with a double-penalty smooth the per-penalty vector ({}) must be strictly \
         longer than the per-class vector ({}) — they are genuinely different shapes",
        edf_pen.len(),
        edf.len()
    );
    // Each per-penalty EDF is a single penalty block's trace EDF, bounded by the
    // block rank, which cannot exceed the class's own coefficient count.
    for (k, &e) in edf_pen.iter().enumerate() {
        assert!(
            e.is_finite() && e >= 0.0 && e <= p_per_class as f64 + 1e-6,
            "penalty {k} EDF {e} out of [0, p_per_class={p_per_class}] — a single penalty \
             block's trace EDF cannot exceed the class coefficient count"
        );
    }
}
