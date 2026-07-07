//! Regression: a `by=factor` smooth's Marra-Wood double-penalty null-space
//! ridge must be rebuilt PER LEVEL in the constrained chart — each level's ridge
//! shrinks exactly that level's own bending null space.
//!
//! A factor-`by` smooth emits one `Primary` (bending) + one `DoublePenaltyNullspace`
//! (ridge) penalty PER LEVEL, each shrinking exactly that level's own bending null
//! space (#1427, independent per-level λ). The central #1476 chart-rebuild in
//! `design_construction` (commit 2afdb5ca1) originally rebuilt EVERY ridge from the
//! FIRST `Primary` it found — correct for a single smooth term, but for a
//! `by=factor` term it placed every level's ridge in level 0's coefficient block
//! (commit 1ea40395e pairs each ridge with the Primary sharing its support).
//!
//! Since #1981 (3b110d9bc) a factor-`by` smooth is realized as an INDEPENDENT
//! smooth term PER LEVEL (`term_builder.rs`: "Unordered factor-by smooths are
//! independent level-specific smooths"), each carrying its own single Primary +
//! ridge pair and its own λ — so the per-level pairs live in DISTINCT terms rather
//! than as disjoint blocks in one term. This test tracks that architecture: it
//! gathers every by-factor level term and pins, PER TERM, that the double-penalty
//! ridge annihilates the co-located bending penalty (`ridge·bending ≈ 0`) —
//! complementary subspaces. A wrong-block rebuild would leave a ridge whose support
//! misses its co-located bending block; a regression that collapsed or dropped
//! levels would fail the ≥2-per-level-terms premise.

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

    // #1981 changed a factor-`by` smooth from ONE term carrying per-level
    // coefficient blocks to a SEPARATE per-level smooth term (term_builder.rs,
    // "Unordered factor-by smooths are independent level-specific smooths"), each
    // with its own independent λ (#1427). So the per-level Primary+ridge pairs
    // now live in DISTINCT terms — one per factor level — not as ≥2 blocks inside
    // a single term. Gather every by-factor level term (each carries exactly one
    // Primary bending penalty and one DoublePenaltyNullspace ridge) and check the
    // #1476 contract PER LEVEL: the ridge annihilates the co-located bending.
    // In this fixture the `s(x, by=g)` smooth is the only penalized smooth, so
    // every double-penalty term IS one of its level replicas.
    let level_terms: Vec<_> = fit
        .design
        .smooth
        .terms
        .iter()
        .filter(|t| {
            t.penaltyinfo_local
                .iter()
                .any(|i| matches!(i.source, PenaltySource::DoublePenaltyNullspace))
        })
        .collect();

    // Premise: the 3-level by-factor smooth must emit ≥2 INDEPENDENT per-level
    // double-penalty terms so the per-level pairing is actually exercised. A
    // regression that collapsed the levels into one term — or dropped levels —
    // trips this rather than silently passing.
    assert!(
        level_terms.len() >= 2,
        "fixture must produce ≥2 independent per-level by-factor double-penalty terms; got {}",
        level_terms.len()
    );

    // For EACH per-level term, the double-penalty ridge must annihilate the
    // bending penalty co-located in the SAME coefficient block (complementary
    // subspaces in the constrained chart). The #1476 wrong-block rebuild puts a
    // ridge where its support does not overlap the co-located primary, so the
    // product is large.
    for (t_idx, term) in level_terms.iter().enumerate() {
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
        assert!(
            !primaries.is_empty() && !ridges.is_empty(),
            "per-level term {t_idx} must carry both a Primary bending block and a ridge; \
             got {} primaries, {} ridges",
            primaries.len(),
            ridges.len()
        );
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
                        "per-level term {t_idx} ridge {r} (support [{rlo},{rhi})) has no co-located \
                         Primary bending block — it was rebuilt in the wrong coefficient block \
                         (#1476 by-factor regression)."
                    )
                });
            let rn = frob(ridge);
            let pn = frob(owner);
            assert!(
                rn > 0.0 && pn > 0.0,
                "per-level term {t_idx} ridge {r}: degenerate ridge/primary"
            );
            let rel = frob(&(**ridge).dot(&**owner)) / (rn * pn);
            assert!(
                rel < 1e-8,
                "per-level term {t_idx} ridge {r}: per-level double-penalty ridge does not annihilate \
                 its co-located bending block (‖ridge·bending‖/(‖ridge‖‖bending‖) = {rel:.3e} ≥ 1e-8); \
                 the ridge is in the wrong constrained chart / coefficient block (#1476 by-factor regression)."
            );
        }
    }
}
