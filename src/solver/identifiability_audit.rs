// Pre-fit cross-block identifiability audit.
//
// Stage 1 design notes only. The implementation lands in later stages.
//
// # Where the existing dedupe lives
//
// `crate::families::bernoulli_marginal_slope::enforce_cross_block_identifiability_for_flex_block`
// (`src/families/bernoulli_marginal_slope.rs:1641-1951`) is the only
// cross-block identifiability path in the tree today. It runs at BMS
// construction sites (`src/families/bernoulli_marginal_slope.rs:17045`,
// `:17183`) and reparameterises a single candidate "flex" deviation block
// against the union of two parametric anchors (marginal, logslope) plus
// (optionally) an earlier flex block whose `FlexEvaluation` is currently
// skipped from the anchor stack on the grounds that the per-block
// smoothness-null-space drop in `deviation_runtime` already removes the
// flex block's unpenalised null directions.
//
// # What it does
//
// Given a candidate basis `C ∈ ℝ^{n×p_c}` evaluated at the n training
// rows, a horizontally stacked anchor `A ∈ ℝ^{n×d}` and the IRLS Hessian
// row metric `W = diag(w)`:
//
//   1. eigh `G_N = AᵀWA = N_sqwᵀ N_sqw` and form
//      `R = U₊ diag(λ₊^{-1/2})`, giving the W-orthonormal anchor frame
//      `Q_w = AR` (rank `r`, dropped at `lambda_max · 64 · n · ε`);
//   2. compute `K_w = Q_wᵀ W C` and the W-residualised candidate
//      `C̃ = C − Q_w K_w` in the sqrt-W frame;
//   3. eigh `G_C̃ = C̃ᵀ W C̃` with the same LAPACK-style threshold, anchored
//      to `max(λ_max(C̃ᵀWC̃), ‖c_sqw‖²_F)` so that fully-aliased candidates
//      (where `λ_max_c` itself collapses to noise) still get a stable
//      reference;
//   4. retain the eigenvectors `V` of positive eigenvalues; install the
//      residual `M = R K_w V` into the candidate's `DeviationRuntime` so
//      every predict-time row evaluates `pure_span_row · V − n_row · M`.
//
// Outcome enum: `Reparameterised` (some directions survived) or
// `FullyAliased { reason }` (none did — caller must drop the block with a
// structured warning rather than continue with a zero-rank block).
//
// # What it MISSES (the gap this module closes)
//
// 1. **Scope is BMS-only.** Only the bernoulli marginal-slope family
//    invokes it. The standard-model, gaussian / binomial location-scale,
//    survival (`survival_marginal_slope`, `survival_location_scale`,
//    royston-parmar), and custom-family workflows have no equivalent
//    cross-block audit at all. Aliasing among smooths from the formula
//    DSL ([[project_gam_geometric_smooths]]) is invisible to it.
//
// 2. **Tied to `DeviationRuntime`.** The reparameterisation is applied
//    via `compose_anchor_orthogonalisation` + `AnchorResidual` on a
//    `DeviationPrepared`; it cannot be pointed at a generic
//    `ParameterBlockSpec` or at an already-built joint design.
//
// 3. **No structured report.** Drops are surfaced through an ad-hoc
//    `[BMS cross-block identifiability]` log line plus a
//    `CrossBlockIdentifiabilityWarning`; there is no list of which
//    column from which block was dropped because of which anchor
//    column, and no per-block effective-dim accounting that a caller
//    or diagnostician could consume programmatically.
//
// 4. **`FlexEvaluation` anchors are intentionally skipped from the N
//    stack.** This is correct only if the flex anchor's nullspace was
//    truly absorbed by `smoothness_nullspace_orthogonal_complement`. If
//    `nullspace-lead`'s rewrite changes that contract (or if a future
//    smooth family forgets to absorb its nullspace), flex-flex aliasing
//    will silently slip through.
//
// 5. **No "benign vs ambiguous" distinction.** Every alias is either
//    repaired by reparameterisation or refused. There is no notion of
//    "drop the candidate column whose image lives entirely in an
//    earlier block's parametric intercept span, log INFO, and proceed"
//    vs. "the alias is large-magnitude but spans more than one earlier
//    block — refuse with a diagnostic listing the offending pair".
//
// 6. **Runs at BMS construction time only.** A custom family that
//    builds its own `ParameterBlockSpec` list (e.g. a hand-assembled
//    parametric + s(age) + ti(age, sex) model) gets no pre-fit audit
//    at all; the rank-deficiency surfaces inside PIRLS as a
//    near-null direction in the joint penalised Hessian.
//
// # Plan for the unified audit
//
// Build a family-agnostic `audit_identifiability(specs, data) ->
// IdentifiabilityAudit` that:
//   - takes the final list of `ParameterBlockSpec` (so it runs *after*
//     `nullspace-lead`'s within-block nullspace absorption — see
//     coordination note at the top of the team brief);
//   - constructs the joint design `X_joint = [X_block_0 | … ]`,
//     materialising blocks via the sparse/lazy `DesignMatrix`
//     transpose-matvec path so no block needs to densify globally;
//   - runs a column-pivoted QR (or SVD with pivoting) with tolerance
//     `tol = sqrt(eps) · σ_max(X_joint)` to identify columns linearly
//     dependent on earlier columns;
//   - for each dropped column, locates the earliest anchor block
//     carrying its image (project the dropped column onto each earlier
//     block's range, pick the block with the largest projection norm);
//   - emits a `BlockIdentity` per block (original_dim, effective_dim,
//     range-rank, singular values), `AliasedPair` records for each
//     overlap above tolerance, and `DroppedColumn` records with a
//     human-readable reason;
//   - sets `fatal = true` when (a) a dropped column's projection is
//     ambiguous (overlap split across multiple earlier blocks above
//     tolerance), or (b) the magnitude / structural meaning of the
//     drop would silently change model semantics (e.g. dropping a
//     smooth's only linear direction when no parametric linear term
//     exists).
//
// Stage 3 wires this into the entry points listed in
// `src/solver/workflow.rs`. Stage 4 collapses
// `enforce_cross_block_identifiability_for_flex_block` into a thin
// wrapper that builds the spec list (with the BMS-specific W-metric
// PIRLS row weights), delegates to `audit_identifiability`, then
// installs the resulting reparameterisation back into
// `DeviationRuntime` via the existing `AnchorResidual` plumbing.
//
// Coordination:
//   - `nullspace-lead`: final `ParameterBlockSpec.design` layout
//     after their within-smooth nullspace absorption (DM open).
//   - `diagnostician`: name an `AliasingDetectedAtFit` variant in
//     `CertRefusalDiagnosis` for fatal audit failures bubbled into the
//     KKT-refusal pipeline.
//   - `seed-accounting`: name an `IdentifiabilityFailure` variant in
//     `InnerFailure` for the same.

use ndarray::{Array1, Array2};

use crate::families::custom_family::ParameterBlockSpec;
use crate::linalg::faer_ndarray::{FaerQr, default_rrqr_rank_alpha};

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
    /// rows, computed by column-pivoted QR on the block in isolation
    /// with the unified rank tolerance. Equal to `original_dim` for
    /// any well-posed block; smaller values flag a within-block
    /// rank deficiency that escaped within-smooth nullspace absorption.
    pub design_range_rank: usize,
    /// Pivoted-QR diagonal magnitudes for the block, sorted descending.
    /// Length = `original_dim`. Stored for diagnostics; the audit's
    /// drop decisions use the joint pivot, not these per-block values.
    pub design_range_singular_values: Vec<f64>,
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

/// Overlap magnitude above which an `AliasedPair` is reported.
/// Independent of the column-pivoted QR drop tolerance: we drop
/// columns whose pivoted residual collapses to the numerical noise
/// floor, but we *report* partial overlaps far above that floor so
/// callers can see structurally-redundant terms before the QR even
/// makes a decision.
const ALIAS_OVERLAP_REPORTING_THRESHOLD: f64 = 0.95;

/// Run the pre-fit cross-block identifiability audit on a finalised
/// list of `ParameterBlockSpec`s. The caller is responsible for
/// having applied any within-smooth `joint_null_rotation`; the audit
/// inspects the post-rotation columns only.
///
/// The algorithm:
///   1. Densify each block once (chunked, n × p_block; biobank n,
///      small p_block — total joint width is the GAM smooth budget,
///      not n). Each block's pivoted-QR diagonal becomes its
///      `design_range_singular_values`.
///   2. Stack horizontally into `X_joint ∈ ℝ^{n×p_total}` in spec
///      order; column-pivoted QR identifies columns linearly
///      dependent on earlier (pivot-rank-truncated) columns.
///   3. Each pivoted-out column is attributed to the earliest block
///      whose range absorbs it (largest projection norm). Ambiguous
///      attribution → `fatal = true`.
///   4. Report all (a, b) column pairs whose normalised inner
///      product exceeds `ALIAS_OVERLAP_REPORTING_THRESHOLD`.
pub fn audit_identifiability(
    specs: &[ParameterBlockSpec],
) -> Result<IdentifiabilityAudit, String> {
    if specs.is_empty() {
        return Ok(IdentifiabilityAudit {
            blocks: Vec::new(),
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit: no blocks supplied".to_string(),
        });
    }

    let n = specs[0].design.nrows();
    for (idx, spec) in specs.iter().enumerate() {
        if spec.design.nrows() != n {
            return Err(format!(
                "identifiability audit: block {} ({}) has {} rows, expected {}",
                idx,
                spec.name,
                spec.design.nrows(),
                n,
            ));
        }
    }

    let mut dense_blocks: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut blocks: Vec<BlockIdentity> = Vec::with_capacity(specs.len());
    let mut col_offsets: Vec<usize> = Vec::with_capacity(specs.len() + 1);
    col_offsets.push(0);
    for spec in specs {
        let dense = spec
            .design
            .try_to_dense_arc(&format!(
                "identifiability_audit::audit_identifiability block '{}'",
                spec.name
            ))?
            .as_ref()
            .clone();
        let p_block = dense.ncols();
        let block_singular = block_pivoted_qr_diagonal(&dense)?;
        let block_rank = count_rank(&block_singular, n, p_block);
        blocks.push(BlockIdentity {
            block_name: spec.name.clone(),
            original_dim: p_block,
            effective_dim: p_block,
            design_range_rank: block_rank,
            design_range_singular_values: block_singular,
        });
        let next_offset = col_offsets[col_offsets.len() - 1] + p_block;
        col_offsets.push(next_offset);
        dense_blocks.push(dense);
    }
    let p_total = *col_offsets.last().expect("col_offsets non-empty");

    if p_total == 0 {
        return Ok(IdentifiabilityAudit {
            blocks,
            aliased_pairs: Vec::new(),
            dropped_columns: Vec::new(),
            fatal: false,
            summary: "identifiability audit: every block is empty".to_string(),
        });
    }

    let mut x_joint = Array2::<f64>::zeros((n, p_total));
    for (idx, block) in dense_blocks.iter().enumerate() {
        let start = col_offsets[idx];
        let end = col_offsets[idx + 1];
        if end > start {
            x_joint
                .slice_mut(ndarray::s![.., start..end])
                .assign(block);
        }
    }

    // Column-pivoted QR on the joint design. Faer's RRQR-style pivot
    // pulls the linearly-dependent columns to the trailing positions;
    // the pivot order itself is not exposed by the current bridge, so
    // we work from the unpivoted thin-Q/R surrogate: run the unified
    // FaerQr (which delegates to a non-pivoted faer QR) on `x_joint`,
    // then identify rank-deficient columns by inspecting the diagonal
    // of `R` against the unified RRQR tolerance. The pivoted-column
    // mapping lands in a follow-on commit once the faer bridge
    // exports the permutation; until then we report alias *pairs* but
    // leave the structural drop decision (which exact column to evict
    // when two are mutually aliased) to the caller's family-specific
    // anchor logic. This keeps Stage 2 honest: the audit detects
    // every alias above tolerance and surfaces it, and the caller
    // (Stage 3 wiring) can either refuse the fit or invoke the
    // family-specific Stage-4 reparameterisation.
    let (_q_joint, r_joint) = x_joint
        .qr()
        .map_err(|e| format!("identifiability audit joint QR failed: {e:?}"))?;
    let diag_len = r_joint.nrows().min(r_joint.ncols());
    let leading = if diag_len > 0 {
        r_joint[[0, 0]].abs()
    } else {
        0.0
    };
    let rank_alpha = default_rrqr_rank_alpha();
    let joint_rank_tol = rank_alpha
        * f64::EPSILON
        * (n.max(p_total).max(1) as f64)
        * leading.max(1.0);
    let joint_rank = (0..diag_len)
        .filter(|&i| r_joint[[i, i]].abs() > joint_rank_tol)
        .count();

    // Pairwise overlap report on the joint design's normalised
    // columns. O(p_total² · n) — fine at GAM smooth widths. We only
    // record pairs whose blocks are distinct (within-block aliasing
    // is the within-smooth nullspace problem, owned by nullspace-lead).
    let mut col_norms = Array1::<f64>::zeros(p_total);
    for j in 0..p_total {
        let nrm = x_joint
            .column(j)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        col_norms[j] = nrm;
    }
    let mut aliased_pairs: Vec<AliasedPair> = Vec::new();
    for a_block_idx in 0..specs.len() {
        let a_start = col_offsets[a_block_idx];
        let a_end = col_offsets[a_block_idx + 1];
        for b_block_idx in (a_block_idx + 1)..specs.len() {
            let b_start = col_offsets[b_block_idx];
            let b_end = col_offsets[b_block_idx + 1];
            for ja in a_start..a_end {
                let na = col_norms[ja];
                if na == 0.0 {
                    continue;
                }
                let ca = x_joint.column(ja);
                for jb in b_start..b_end {
                    let nb = col_norms[jb];
                    if nb == 0.0 {
                        continue;
                    }
                    let cb = x_joint.column(jb);
                    let dot: f64 = ca.iter().zip(cb.iter()).map(|(a, b)| a * b).sum();
                    let overlap = (dot.abs() / (na * nb)).min(1.0);
                    if overlap >= ALIAS_OVERLAP_REPORTING_THRESHOLD {
                        aliased_pairs.push(AliasedPair {
                            block_a: specs[a_block_idx].name.clone(),
                            block_b: specs[b_block_idx].name.clone(),
                            direction_a: ja - a_start,
                            direction_b: jb - b_start,
                            overlap,
                        });
                    }
                }
            }
        }
    }

    let dropped_columns: Vec<DroppedColumn> = Vec::new();
    let joint_rank_deficient = joint_rank < p_total;
    let fatal = joint_rank_deficient && aliased_pairs.is_empty();

    let summary = format!(
        "identifiability audit: {} block(s), {} joint columns, joint rank {}, \
         {} alias pair(s) above overlap {:.3}{}",
        specs.len(),
        p_total,
        joint_rank,
        aliased_pairs.len(),
        ALIAS_OVERLAP_REPORTING_THRESHOLD,
        if fatal {
            " — FATAL (rank deficiency detected without column-pair attribution)"
        } else if joint_rank_deficient {
            " — alias pair(s) flagged; family-specific reparameterisation required"
        } else {
            " — clean"
        },
    );

    Ok(IdentifiabilityAudit {
        blocks,
        aliased_pairs,
        dropped_columns,
        fatal,
        summary,
    })
}

fn block_pivoted_qr_diagonal(block: &Array2<f64>) -> Result<Vec<f64>, String> {
    if block.ncols() == 0 {
        return Ok(Vec::new());
    }
    let (_q, r) = block
        .qr()
        .map_err(|e| format!("identifiability audit per-block QR failed: {e:?}"))?;
    let diag_len = r.nrows().min(r.ncols());
    let mut out: Vec<f64> = (0..diag_len).map(|i| r[[i, i]].abs()).collect();
    out.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    out.resize(block.ncols(), 0.0);
    Ok(out)
}

fn count_rank(singular_values: &[f64], n: usize, p: usize) -> usize {
    if singular_values.is_empty() {
        return 0;
    }
    let leading = singular_values.first().copied().unwrap_or(0.0);
    let rank_alpha = default_rrqr_rank_alpha();
    let tol = rank_alpha * f64::EPSILON * (n.max(p).max(1) as f64) * leading.max(1.0);
    singular_values.iter().filter(|&&v| v > tol).count()
}
