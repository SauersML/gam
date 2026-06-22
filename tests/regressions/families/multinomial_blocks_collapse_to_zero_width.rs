//! Regression test for issue #363.
//!
//! A multinomial-logit fit used to collapse **every class block after the
//! first** to zero width before the joint Newton solve ran, so no model with
//! `K ≥ 3` classes (or a 2-class model with covariates) could be fit.
//!
//! Root cause: `MultinomialFamily::build_block_specs` builds `K − 1` blocks
//! that all share the identical `P`-column design `X`, but it left
//! `jacobian_callback = None`. With every block reporting a single output
//! channel, `canonicalize_for_identifiability` took the *flat* audit, which
//! assembles the joint design by placing every block's columns side by side
//! over the same `N` rows — `[X | X | … | X]`. RRQR then declared the repeated
//! columns redundant aliases and, by gauge priority, stripped every block past
//! `class_0` to width 0. Those zero-width blocks entered the joint solve with
//! `block_residual_inf = NaN`, so the fit aborted with "no candidate seeds
//! passed outer startup validation".
//!
//! The blocks are **not** gauge-redundant: each feeds a separate softmax
//! channel `η_a = X β_a`, so the true joint Jacobian is the block-diagonal
//! `blkdiag(X, …, X)` of full rank `(K − 1)·P`. The fix installs an
//! `AdditiveBlockJacobian` on each block (own output channel `a`, `K − 1`
//! total channels), routing canonicalisation through the channel-aware audit.
//!
//! This test reproduces the issue from a meaningful angle: a deterministic,
//! RNG-free 3-class multinomial-logit sample where `x` is genuinely
//! informative (true slopes `+1.2`, `−0.6`, `0` over classes A/B/C). It asserts
//! the fit succeeds with `n_active_classes == 2`, `p_per_class == 2`, all 4
//! coefficients finite, and — the load-bearing check that the second block was
//! not collapsed — that `class_1`'s slope is non-trivially nonzero. A
//! zero-width `class_1` (the pre-fix behaviour) cannot produce a nonzero slope.

use csv::StringRecord;
use gam::families::multinomial::fit_penalized_multinomial_formula;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// Deterministic 3-class multinomial-logit sample.
///
/// The class probabilities follow `softmax([1.2·x, −0.6·x, 0])`; each row is
/// assigned its class by inverse-CDF against a fixed, evenly-spaced quantile
/// grid (no RNG), so every class is populated and the fit is reproducible.
fn three_class_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let n = 600usize;
    let rows = (0..n)
        .map(|i| {
            // x spans roughly [-3, 3], deterministic.
            let x = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            let lin = [1.2 * x, -0.6 * x, 0.0];
            let m = lin.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = lin.iter().map(|v| (v - m).exp()).collect();
            let denom: f64 = exps.iter().sum();
            let p = [exps[0] / denom, exps[1] / denom, exps[2] / denom];
            // Inverse-CDF assignment against an evenly-spaced quantile in (0,1).
            let u = ((i as f64) + 0.5) / (n as f64);
            let label = if u < p[0] {
                "A"
            } else if u < p[0] + p[1] {
                "B"
            } else {
                "C"
            };
            StringRecord::from(vec![x.to_string(), label.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 3-class dataset")
}

#[test]
fn multinomial_three_class_fit_keeps_all_blocks_full_width() {
    init_parallelism();
    let data = three_class_dataset();

    // Same entry point the Python FFI uses for family="multinomial".
    let model =
        fit_penalized_multinomial_formula(&data, "y ~ x", &FitConfig::default(), 1.0, 200, 1.0e-8)
            .expect(
                "3-class multinomial fit must succeed: before the #363 fix it aborted with \
         'no candidate seeds passed outer startup validation' because every block past \
         class_0 was stripped to zero width by the flat identifiability audit",
            );

    // K − 1 = 2 active class blocks, each carrying the full P = 2 design
    // (intercept + slope). A collapsed block would report p_per_class for the
    // surviving block but a zero-width second block — the repack in
    // fit_penalized_multinomial_formula would then have erred out above.
    assert_eq!(
        model.n_active_classes, 2,
        "expected K − 1 = 2 active class blocks, got {}",
        model.n_active_classes
    );
    assert_eq!(
        model.p_per_class, 2,
        "each active class block must carry the full P = 2 design (intercept + slope), got {}",
        model.p_per_class
    );

    // coefficients_flat is row-major (P, K−1): [β_0[0], β_0[1], β_1[0], β_1[1]].
    assert_eq!(
        model.coefficients_flat.len(),
        4,
        "expected 4 coefficients for a 2-class × 2-covariate fit, got {}",
        model.coefficients_flat.len()
    );
    for (idx, c) in model.coefficients_flat.iter().enumerate() {
        assert!(
            c.is_finite(),
            "coefficient[{idx}] must be finite (got {c}); a NaN here is the \
             zero-width-block stationarity-residual signature from #363"
        );
    }

    // Load-bearing check that class_1 was NOT collapsed to width 0: its slope
    // must be a genuine nonzero estimate. With the true generating slopes
    // (+1.2 for A, −0.6 for B against reference C), the fitted class blocks
    // must both carry appreciable slopes. The flat-audit failure would have
    // forced class_1 to width 0, making any nonzero slope impossible.
    //
    // coefficients_active[i, a] = coefficients_flat[i * (K−1) + a].
    let k_minus_1 = model.n_active_classes;
    // Intercept rows: i = 0; slope rows: i = 1 (constant term + "x").
    // Slope of block a: coefficients_flat[1 * k_minus_1 + a].
    let slope_class0 = model.coefficients_flat[k_minus_1];
    let slope_class1 = model.coefficients_flat[k_minus_1 + 1];
    assert!(
        slope_class0.abs() > 1.0e-2,
        "class_0 slope must be a non-trivial estimate (got {slope_class0})"
    );
    assert!(
        slope_class1.abs() > 1.0e-2,
        "class_1 slope must be a non-trivial estimate (got {slope_class1}); a (near-)zero \
         slope is the fingerprint of the collapsed zero-width second block from #363"
    );
}
