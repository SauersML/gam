//! Pure-data identifiability-audit result types.
//!
//! These structs are the family-facing results of the pre-fit cross-block
//! identifiability audit and the MAP-uniqueness check. They carry only plain
//! data (`Vec`/`String`/`f64`/`bool`/`usize`) with no `faer`/`ndarray`/solver
//! dependency, so they live in `gam-problem` (below the monolith) where the
//! `CustomFamilyError` cone and other low-level consumers can name them. The
//! compute code that BUILDS these audits stays in the monolith
//! (`crate::identifiability::audit`) and constructs them through these public
//! fields.

/// Per-block accounting record. `original_dim` is the spec's column
/// count at audit entry (post `joint_null_rotation` absorption — the
/// audit is contractually run on the rotated specs). `effective_dim`
/// is what remains after the audit drops aliased columns. Equal values
/// mean the block carried no redundant directions w.r.t. earlier
/// blocks.
#[derive(Debug, Clone)]
pub struct BlockIdentity {
    pub block_name: String,
    pub original_dim: usize,
    pub effective_dim: usize,
    /// Numerical rank of the block's column space at the n training
    /// rows, computed by penalty-aware column-pivoted RRQR on `[J; S]`
    /// (so penalty-covered design-null directions count as identified).
    /// Equal to `original_dim` for any well-posed block; smaller values
    /// flag a within-block rank deficiency that escaped within-smooth
    /// nullspace absorption.
    pub design_range_rank: usize,
}

/// A pair `(block_a.column → block_b.column)` whose normalised
/// inner product exceeds the alias-overlap reporting threshold.
/// Reported once per audited pair, in block-order (`block_a` index
/// strictly less than `block_b` index in the spec list, so the
/// "earlier block carries the image" attribution is well-defined).
#[derive(Debug, Clone)]
pub struct AliasedPair {
    pub block_a: String,
    pub block_b: String,
    pub direction_a: usize,
    pub direction_b: usize,
    /// `|aᵀb| / (‖a‖·‖b‖)`. Always in `[0, 1]`. Values at or near 1.0
    /// indicate near-perfect collinearity; values in `(threshold, 1.0)`
    /// indicate partial overlap that the column-pivoted QR will still
    /// preserve (only fully redundant directions get pivoted out).
    pub overlap: f64,
    /// Bias shift applied to the null-distribution mean for this pair,
    /// equal to `bias_shift_for_pair(z_a, z_b, s2_a, s2_b)`.
    /// Non-zero when exactly one block carries a `RowScaledJacobian` callback
    /// (or the two scalings differ) and the row-scaling vector is skewed.
    /// Stored so that the halt-threshold check can apply the same
    /// directional correction as the report-threshold check.
    /// Zero for all pairs arising from the channel-aware audit path,
    /// and for pairs from the flat path when both blocks have symmetric
    /// (or absent) row scaling.
    pub bias_shift: f64,
}

#[derive(Debug, Clone)]
pub struct DroppedColumn {
    pub block: String,
    pub column: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct IdentifiabilityAudit {
    pub blocks: Vec<BlockIdentity>,
    pub aliased_pairs: Vec<AliasedPair>,
    pub dropped_columns: Vec<DroppedColumn>,
    /// `true` when at least one dropped column's attribution to an
    /// earlier block is ambiguous (overlap distributed across multiple
    /// earlier blocks above tolerance) or the drop would silently
    /// change model semantics. Callers must refuse the fit in that
    /// case rather than silently proceed with a different model.
    pub fatal: bool,
    pub summary: String,
}

/// Error produced when the MAP uniqueness condition
/// `ker(J^T W J) ∩ ker(S) = {0}` is violated.
///
/// A null direction `n` of `J^T W J` with `n^T S n = 0` means the posterior
/// is flat along `n`: no likelihood curvature AND no penalty curvature,
/// so the MAP estimate is non-unique.  The error names the offending
/// direction and the dominant block (the block whose columns have the
/// largest component in `n`) so the caller can trace which smooth term
/// contributed the unpenalised null direction.
#[derive(Debug, Clone)]
pub struct MapUniquenessError {
    /// Human-readable description of the failure, including the dominant block.
    pub message: String,
    /// Name of the block whose columns dominate the null direction.
    pub dominant_block: String,
    /// Index of the null direction (0-based among directions below tolerance).
    pub null_direction_index: usize,
    /// `n^T S n` for the offending null direction (≈ 0.0).
    pub penalty_quadratic_form: f64,
}

impl std::fmt::Display for MapUniquenessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
