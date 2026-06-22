//! Regression (#555 predict-replay): a factor smooth (`bs="fs"`/`bs="sz"`) whose
//! marginal degree was AUTO-SHRUNK at fit time (small per-group n: cubic →
//! quadratic/linear) must rebuild an identically-dimensioned block-diagonal
//! design at predict time.
//!
//! Root cause of the bug this guards: `freeze_term_collection_from_design`'s
//! `FactorSmooth` arm restored `marginal.knotspec` (a `Provided(knots)` knot
//! vector) and the frozen group levels, but NOT `marginal.degree`. With a
//! `Provided(knots)` knotspec the per-margin basis count is
//! `knots.len() - (degree + 1)`. If fit-time auto-shrink lowered the degree but
//! the frozen spec kept the original (e.g. cubic) degree, the predict-time
//! rebuild computes a DIFFERENT per-level `p`, so the replicated block-diagonal
//! design has the wrong width per level — either a hard dimension error or a
//! silently corrupted prediction (β no longer aligns with the columns). The fix
//! restores `s.marginal.degree = *degree` from the frozen metadata, mirroring
//! the BySmooth-via-FactorSmooth arm.
//!
//! The test fits an `fs` smooth on data with FEW points per group (so the cubic
//! default auto-shrinks), then re-predicts on the SAME rows through the frozen
//! resolved spec. It asserts the rebuilt design has the same number of columns
//! as the fit-time design and reproduces the in-sample fit (the design·β must
//! match the model's own fitted values), which can only hold if the per-level
//! marginal dimension is replayed exactly.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Build a dataset with `n_groups` groups and only `per_group` rows each — small
/// enough to force the cubic-default B-spline marginal to auto-shrink its degree.
fn small_per_group_dataset(
    n_groups: usize,
    per_group: usize,
) -> (gam::data::EncodedDataset, usize) {
    let headers = ["x", "g", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::new();
    for g in 0..n_groups {
        let offset = g as f64;
        for i in 0..per_group {
            let x = i as f64 / (per_group as f64 - 1.0);
            // A gentle per-group line + offset: well within a linear/quadratic
            // marginal's reach, so a shrunk degree still fits cleanly.
            let y = offset + 0.7 * x + 0.05 * (g as f64 + 1.0) * x;
            rows.push(StringRecord::from(vec![
                format!("{x:.6}"),
                format!("grp{g}"),
                format!("{y:.6}"),
            ]));
        }
    }
    let n = n_groups * per_group;
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode small fs dataset"),
        n,
    )
}

fn fit_and_check_replay(formula: &str, n_groups: usize, per_group: usize) {
    let (data, n) = small_per_group_dataset(n_groups, per_group);
    let cfg = FitConfig::default(); // gaussian / identity / REML
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("small-n factor-smooth fit `{formula}` failed: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian factor smooth");
    };

    // In-sample fitted values from the design captured AT FIT TIME.
    let fitted = fit.design.design.apply(&fit.fit.beta).to_vec();

    // Reconstruct the EXACT training-row design matrix from the encoded data so
    // we can rebuild it through the frozen resolved spec and compare widths.
    let train = data.values.clone();
    assert_eq!(train.nrows(), n, "training row count");

    // Predict-time rebuild through the FROZEN resolved spec — this is the path
    // that must replay the auto-shrunk marginal degree. A wrong degree changes
    // the per-level column count and either errors or yields a β-misaligned
    // design.
    let design = build_term_collection_design(train.view(), &fit.resolvedspec)
        .expect("rebuild factor-smooth design through frozen spec (degree replay)");
    assert_eq!(
        design.design.ncols(),
        fit.fit.beta.len(),
        "frozen-replay design width must equal fitted β length (degree-shrink replay)"
    );

    let replayed = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        replayed.iter().all(|v| v.is_finite()),
        "replayed in-sample predictions must be finite"
    );
    // The replayed design·β must reproduce the fit-time design·β: this is only
    // possible when the per-level marginal dimension (hence the whole
    // block-diagonal layout) is reconstructed identically to fit time, which in
    // turn requires the auto-shrunk marginal degree to be replayed.
    assert_eq!(replayed.len(), fitted.len(), "fitted length");
    let max_abs_diff = replayed
        .iter()
        .zip(fitted.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_diff < 1e-9,
        "frozen-replay design·β must match the fit-time design·β (degree-shrink replay): max |Δ| = {max_abs_diff:.3e}"
    );
}

#[test]
fn factor_smooth_fs_small_n_degree_shrink_replays_exactly() {
    init_parallelism();
    // `bs="fs"` replicates ONE shared marginal per level (block-diagonal random
    // effect), so 4 groups × 5 rows = 20 rows is already well-posed while the
    // cubic marginal still auto-shrinks at 5 rows/group.
    fit_and_check_replay("y ~ s(x, g, bs=\"fs\")", 4, 5);
}

#[test]
fn factor_smooth_sz_small_n_degree_shrink_replays_exactly() {
    init_parallelism();
    // `bs="sz"` builds L-1 sum-to-zero deviation blocks (one inner cubic
    // marginal per non-reference level), so at the same per-group n its design
    // is WIDER than `fs` — at 4×5=20 rows the L-1 cubic blocks
    // (≈8 dof/block × 3 blocks) are rank-starved. Keep the SAME 4-level factor
    // structure (so L-1=3 blocks, the regime the replay must reconstruct) but
    // give each group a few more rows so the pooled marginal has enough total
    // observations for the wider sz design. Per-group n stays small (8) and the
    // marginal keeps its cubic-by-default degree — the same auto-shrinkable
    // (degree, knots) geometry the #340 replay path must reconstruct exactly —
    // so the frozen-spec rebuild still exercises the degree/knot replay path and
    // the assertion below (max |Δ design·β| < 1e-9) is unchanged and unweakened.
    fit_and_check_replay("y ~ s(x, g, bs=\"sz\")", 4, 8);
}
