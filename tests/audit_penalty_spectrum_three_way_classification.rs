//! Audit: a symmetric penalty's spectrum is partitioned into exactly THREE
//! structural classes — range (`ev > tol`), null (`|ev| <= tol`), and negative
//! curvature (`ev < -tol`) — and a genuinely negative eigendirection is NEVER
//! misclassified as null space.
//!
//! Root-cause class behind #1425: the penalty pipeline used to split the
//! spectrum with a binary `ev <= tol` test, which has no name for the negative
//! class and so folded genuine O(1) negative-curvature directions into the null
//! space. Downstream, the canonical penalty root and the joint-null absorption
//! then treated those negative-curvature directions as *unpenalized* and
//! absorbed them into the parametric block — silently truncating real negative
//! curvature out of the model.
//!
//! `CanonicalPenaltyBlock` now exposes the full partition (`rank` / `nullity` /
//! `negative_dim`). This test certifies, directly on a hand-built indefinite
//! penalty, that:
//!   1. a genuine negative eigenvalue is counted in `negative_dim`, NOT in
//!      `nullity`;
//!   2. `rank + nullity + negative_dim == dim` (the partition is total); and
//!   3. a PSD penalty has `negative_dim == 0` and `rank + nullity == dim`
//!      (the negative class is empty exactly when the penalty is PSD).

use gam::terms::basis::analyze_penalty_block;
use ndarray::Array2;

/// A symmetric matrix with a chosen diagonal spectrum (already diagonal, so the
/// eigenvalues ARE the diagonal entries). Builds an `n × n` diagonal penalty.
fn diag_penalty(diag: &[f64]) -> Array2<f64> {
    let n = diag.len();
    let mut m = Array2::<f64>::zeros((n, n));
    for (i, &d) in diag.iter().enumerate() {
        m[[i, i]] = d;
    }
    m
}

#[test]
fn negative_eigendirection_is_negative_not_null() {
    // Spectrum: two strong positive (range), one exact zero (null), one strong
    // negative (negative curvature). max|ev| = 5 so tol ≈ 4·1e-10·5 ≪ 1: the
    // ±5/+3 are unambiguously range/negative and the 0 is unambiguously null.
    let penalty = diag_penalty(&[5.0, 3.0, 0.0, -5.0]);
    let block = analyze_penalty_block(&penalty).expect("analyze");

    assert_eq!(
        block.rank, 2,
        "two positive eigenvalues span range(S); got rank={}",
        block.rank
    );
    assert_eq!(
        block.nullity, 1,
        "exactly ONE eigenvalue is within tol of zero; the negative one must NOT \
         be counted as null. got nullity={} (the #1425 defect would report 2)",
        block.nullity
    );
    assert_eq!(
        block.negative_dim, 1,
        "the single genuinely-negative eigendirection must be surfaced as \
         negative curvature, not silently dropped. got negative_dim={}",
        block.negative_dim
    );

    // The partition is total: every eigendirection is in exactly one class.
    assert_eq!(
        block.rank + block.nullity + block.negative_dim,
        penalty.nrows(),
        "range + null + negative must cover the whole spectrum"
    );
}

#[test]
fn psd_penalty_has_empty_negative_class() {
    // A PSD penalty (one zero null direction, the rest positive) has NO negative
    // class, and then — and only then — does the classic `rank + nullity == dim`
    // identity hold.
    let penalty = diag_penalty(&[4.0, 2.0, 1.0, 0.0]);
    let block = analyze_penalty_block(&penalty).expect("analyze");

    assert_eq!(
        block.negative_dim, 0,
        "a PSD penalty has no negative curvature"
    );
    assert_eq!(block.rank, 3, "three positive eigenvalues");
    assert_eq!(block.nullity, 1, "one null direction");
    assert_eq!(
        block.rank + block.nullity,
        penalty.nrows(),
        "with an empty negative class the spectrum is exactly range ⊕ null"
    );
}

#[test]
fn full_rank_psd_penalty_has_no_null_or_negative() {
    let penalty = diag_penalty(&[3.0, 2.0, 1.0]);
    let block = analyze_penalty_block(&penalty).expect("analyze");
    assert_eq!(block.rank, 3);
    assert_eq!(block.nullity, 0);
    assert_eq!(block.negative_dim, 0);
    assert!(!block.iszero);
}
