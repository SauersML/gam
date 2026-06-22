//! Regression: a `by=factor` smooth's Marra-Wood double-penalty null-space
//! ridge must be rebuilt PER LEVEL in the constrained chart — each level's ridge
//! shrinks exactly that level's own bending null space.
//!
//! A factor-`by` smooth emits one `Primary` (bending) + one `DoublePenaltyNullspace`
//! (ridge) penalty PER LEVEL, each confined to that level's disjoint
//! `[lvl·p .. lvl·p + p]` diagonal coefficient block (#1427, independent per-level
//! λ). The central #1476 chart-rebuild in `design_construction` (commit 2afdb5ca1)
//! originally rebuilt EVERY ridge from the FIRST `Primary` it found — correct for a
//! single smooth term, but for a `by=factor` term it placed every level's ridge in
//! level 0's coefficient block (commit 1ea40395e pairs each ridge with the Primary
//! sharing its support instead).
//!
//! Contract pinned on the realized constrained `penalties_local`: for EVERY level
//! block, the double-penalty ridge living in that block annihilates the bending
//! penalty living in the SAME block (`ridge·bending ≈ 0`) — complementary
//! subspaces. The wrong-block rebuild leaves a ridge in the wrong coefficient
//! range, so it does NOT annihilate the bending penalty co-located with it.

use csv::StringRecord;
use gam::basis::PenaltySource;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const K: usize = 10;

fn three_group_dataset() -> gam::data::EncodedDataset {
    // Deterministic; the chart geometry under test does not depend on noise.
    let mut rows: Vec<StringRecord> = Vec::new();
    let n_per_group = 120usize;
    for (label, kind) in [("a", 0u8), ("b", 1u8), ("c", 2u8)] {
        for i in 0..n_per_group {
            let x = (i as f64 + 0.5) / n_per_group as f64;
            let f = match kind {
                0 => (2.0 * std::f64::consts::PI * x).sin(),
                1 => (2.0 * std::f64::consts::PI * x).cos(),
                _ => 4.0 * (x - 0.5) * (x - 0.5),
            };
            rows.push(StringRecord::from(vec![
                format!("{x:.6}"),
                label.to_string(),
                format!("{f:.6}"),
            ]));
        }
    }
    encode_recordswith_inferred_schema(
        ["x", "g", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode by-factor dataset")
}

fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Inclusive-exclusive nonzero-row support of a penalty matrix.
fn support(m: &Array2<f64>) -> (usize, usize) {
    let n = m.nrows();
    let (mut lo, mut hi) = (n, 0usize);
    for i in 0..n {
        if (0..m.ncols()).any(|j| m[[i, j]] != 0.0) {
            lo = lo.min(i);
            hi = hi.max(i + 1);
        }
    }
    (lo, hi)
}

#[test]
fn factor_by_double_penalty_ridge_is_per_level_in_constrained_chart() {
    init_parallelism();
    let data = three_group_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(&format!("y ~ s(x, by=g, k={K})"), &data, &cfg)
        .expect("by-factor double-penalty fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };

    // The s(x, by=g) smooth is the term carrying the per-level penalty blocks.
    let term = fit
        .design
        .smooth
        .terms
        .iter()
        .find(|t| {
            t.penaltyinfo_local
                .iter()
                .any(|i| matches!(i.source, PenaltySource::DoublePenaltyNullspace))
        })
        .expect("a by-factor double-penalty smooth term with a null-space ridge");

    let primaries: Vec<&Array2<f64>> = term
        .penalties_local
        .iter()
        .zip(term.penaltyinfo_local.iter())
        .filter(|(_, i)| matches!(i.source, PenaltySource::Primary))
        .map(|(s, _)| s)
        .collect();
    let ridges: Vec<&Array2<f64>> = term
        .penalties_local
        .iter()
        .zip(term.penaltyinfo_local.iter())
        .filter(|(_, i)| matches!(i.source, PenaltySource::DoublePenaltyNullspace))
        .map(|(s, _)| s)
        .collect();

    // Premise: a 3-level by-factor double-penalty smooth must emit ≥2 Primary
    // blocks (one per level) so the per-level pairing is actually exercised.
    assert!(
        primaries.len() >= 2 && ridges.len() >= 2,
        "fixture must produce ≥2 per-level Primary+ridge pairs; got {} primaries, {} ridges",
        primaries.len(),
        ridges.len()
    );

    // For EACH ridge, the bending penalty co-located in the SAME coefficient
    // block must be annihilated by it (complementary subspaces in the constrained
    // chart). The wrong-block rebuild puts a ridge where its support does not
    // overlap the co-located primary, so this product is large.
    for (r, ridge) in ridges.iter().enumerate() {
        let (rlo, rhi) = support(ridge);
        let owner = primaries
            .iter()
            .find(|p| {
                let (plo, phi) = support(p);
                plo <= rlo && rhi <= phi
            })
            .unwrap_or_else(|| {
                panic!(
                    "ridge {r} (support [{rlo},{rhi})) has no co-located Primary bending block — \
                     it was rebuilt in the wrong coefficient block (#1476 by-factor regression)."
                )
            });
        let rn = frob(ridge);
        let pn = frob(owner);
        assert!(rn > 0.0 && pn > 0.0, "ridge {r}: degenerate ridge/primary");
        let rel = frob(&(**ridge).dot(&**owner)) / (rn * pn);
        assert!(
            rel < 1e-8,
            "ridge {r}: per-level double-penalty ridge does not annihilate its co-located \
             bending block (‖ridge·bending‖/(‖ridge‖‖bending‖) = {rel:.3e} ≥ 1e-8); the ridge \
             is in the wrong constrained chart / coefficient block (#1476 by-factor regression)."
        );
    }
}
