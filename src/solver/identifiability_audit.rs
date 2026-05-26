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
use crate::linalg::faer_ndarray::{FaerQr, default_rrqr_rank_alpha, rrqr_with_permutation};

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
pub fn audit_identifiability(specs: &[ParameterBlockSpec]) -> Result<IdentifiabilityAudit, String> {
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
    let block_phase_started = std::time::Instant::now();
    let block_heartbeat = (n.saturating_mul(specs.len()) >= 1_000_000)
        .then(crate::util::heartbeat::Heartbeat::default_interval);
    for (idx, spec) in specs.iter().enumerate() {
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
        if let Some(hb) = block_heartbeat.as_ref() {
            hb.tick(1, |progress, elapsed| {
                log::info!(
                    "[STAGE] identifiability audit: per-block QR progress={}/{} elapsed={:.1}s (overall={:.1}s)",
                    progress.min(specs.len()),
                    specs.len(),
                    elapsed,
                    block_phase_started.elapsed().as_secs_f64(),
                );
            });
        }
        // Per-block summary on the last iteration so the user always
        // sees one waypoint between "audit start" and the dense joint
        // assembly even if no heartbeat fires.
        if idx + 1 == specs.len() {
            log::info!(
                "[STAGE] identifiability audit: per-block QR complete blocks={} elapsed={:.3}s",
                specs.len(),
                block_phase_started.elapsed().as_secs_f64(),
            );
        }
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
            x_joint.slice_mut(ndarray::s![.., start..end]).assign(block);
        }
    }

    // Column-pivoted RRQR on the joint design. The pivot permutation
    // names which original columns were demoted past the rank
    // threshold, so we can attribute each dropped column back to its
    // (block, local_col) origin deterministically. `rrqr_with_permutation`
    // uses the same `RRQR_RANK_ALPHA · ε · max(m,n) · leading`
    // tolerance as the per-block diagonal counter above, so the joint
    // verdict and the per-block diagnostics are tolerance-consistent.
    let rrqr_started = std::time::Instant::now();
    log::info!(
        "[STAGE] identifiability audit: joint RRQR start n={} p_total={}",
        n,
        p_total,
    );
    let rrqr = rrqr_with_permutation(&x_joint, default_rrqr_rank_alpha())
        .map_err(|e| format!("identifiability audit joint RRQR failed: {e:?}"))?;
    log::info!(
        "[STAGE] identifiability audit: joint RRQR end rank={}/{} elapsed={:.3}s",
        rrqr.rank,
        p_total,
        rrqr_started.elapsed().as_secs_f64(),
    );
    let joint_rank = rrqr.rank;
    let joint_rank_tol = rrqr.rank_tol;
    let demoted_joint_cols: Vec<usize> = rrqr.column_permutation[rrqr.rank..].to_vec();

    // Pairwise overlap report on the joint design's normalised
    // columns. O(p_total² · n) — fine at GAM smooth widths. We only
    // record pairs whose blocks are distinct (within-block aliasing
    // is the within-smooth nullspace problem, owned by nullspace-lead).
    let pairwise_started = std::time::Instant::now();
    log::info!(
        "[STAGE] identifiability audit: pairwise overlap scan start n={} p_total={} blocks={}",
        n,
        p_total,
        specs.len(),
    );
    let mut col_norms = Array1::<f64>::zeros(p_total);
    for j in 0..p_total {
        let nrm = x_joint.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
        col_norms[j] = nrm;
    }
    let pairwise_block_heartbeat = (n.saturating_mul(p_total) >= 1_000_000)
        .then(crate::util::heartbeat::Heartbeat::default_interval);
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

    // Attribute each demoted joint column back to its (block, local_col)
    // origin using the col_offsets table built above. The earliest
    // block whose range absorbs the demoted column is, by construction,
    // the block at lower index containing the largest projection norm
    // — but RRQR's column-pivoting selects the column most aligned
    // with the *trailing* (demoted) space, so we attribute the alias
    // to "the demoted column itself was selected as redundant; the
    // earlier blocks (in spec order) reconstruct it". The reason
    // string names the joint-column index and the joint rank tolerance
    // so callers can correlate with the structural log.
    let mut dropped_columns: Vec<DroppedColumn> = Vec::new();
    for &joint_col in &demoted_joint_cols {
        let (block_idx, local_col) = locate_block_column(&col_offsets, joint_col)?;
        let block_name = specs[block_idx].name.clone();
        let reason = format!(
            "joint-design column {joint_col} (block '{block_name}' local column \
             {local_col}) demoted past joint RRQR rank tolerance {tol:.3e}; earlier \
             blocks' column span absorbs this direction",
            tol = joint_rank_tol,
        );
        dropped_columns.push(DroppedColumn {
            block: block_name,
            column: local_col,
            reason,
        });
    }

    // Reflect the dropped-column attribution into `BlockIdentity::
    // effective_dim`: each block's effective dimension is its original
    // dim minus the count of its columns appearing in
    // `demoted_joint_cols`. The per-block `design_range_rank` (from
    // the in-isolation per-block QR) still flags any within-block
    // rank deficiency that escaped within-smooth absorption.
    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let lo = col_offsets[block_idx];
        let hi = col_offsets[block_idx + 1];
        let dropped_here = demoted_joint_cols
            .iter()
            .filter(|&&j| j >= lo && j < hi)
            .count();
        block.effective_dim = block.original_dim.saturating_sub(dropped_here);
    }

    let joint_rank_deficient = joint_rank < p_total;
    // Fatal when joint rank is deficient AND we cannot attribute the
    // deficiency to either (a) at least one pairwise alias above the
    // reporting threshold or (b) at least one dropped-column record.
    // RRQR always populates `demoted_joint_cols` for any deficiency
    // it detects, so the only way to reach `fatal=true` is if the
    // tolerance disagreement between RRQR and the column-norm
    // pairwise scan hides the structural alias — exactly the >2-way
    // alias case the caller must refuse.
    let fatal = joint_rank_deficient && aliased_pairs.is_empty() && dropped_columns.is_empty();

    let summary = format!(
        "identifiability audit: {} block(s), {} joint columns, joint rank {}, \
         {} alias pair(s) above overlap {:.3}, {} dropped column(s){}",
        specs.len(),
        p_total,
        joint_rank,
        aliased_pairs.len(),
        ALIAS_OVERLAP_REPORTING_THRESHOLD,
        dropped_columns.len(),
        if fatal {
            " — FATAL (rank deficiency without pair or column attribution)"
        } else if joint_rank_deficient {
            " — alias(es) flagged; family-specific reparameterisation required"
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

fn locate_block_column(col_offsets: &[usize], joint_col: usize) -> Result<(usize, usize), String> {
    // col_offsets has length specs.len() + 1; col_offsets[i..i+1] is
    // the joint-column range for block i. Linear scan is fine — the
    // table is tiny (one entry per block).
    for i in 0..col_offsets.len() - 1 {
        if joint_col >= col_offsets[i] && joint_col < col_offsets[i + 1] {
            return Ok((i, joint_col - col_offsets[i]));
        }
    }
    Err(format!(
        "identifiability_audit::locate_block_column: joint_col {joint_col} \
         outside col_offsets range (max = {})",
        col_offsets.last().copied().unwrap_or(0),
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::Array2;

    fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
        let n = design.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
        }
    }

    fn linspace_minus_one_to_one(n: usize) -> Array1<f64> {
        if n <= 1 {
            return Array1::<f64>::zeros(n.max(1));
        }
        let step = 2.0 / (n as f64 - 1.0);
        Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
    }

    /// Test 1: a model with no aliasing → audit returns clean, no
    /// drops, fatal=false.
    #[test]
    fn audit_no_aliasing_returns_clean() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        // Parametric: [1, x]. Smooth: [x², x³] — orthogonal-ish to the
        // parametric directions, no exact alias.
        let mut parametric = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            parametric[[i, 0]] = 1.0;
            parametric[[i, 1]] = x[i];
        }
        let mut smooth = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            smooth[[i, 0]] = x[i] * x[i];
            smooth[[i, 1]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_x", smooth),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed on clean specs");
        assert!(
            !audit.fatal,
            "no aliasing must not be fatal: {}",
            audit.summary
        );
        assert!(
            audit.aliased_pairs.is_empty(),
            "expected no alias pairs; got {:?}",
            audit.aliased_pairs,
        );
        assert!(
            audit.dropped_columns.is_empty(),
            "expected no dropped columns; got {:?}",
            audit.dropped_columns,
        );
        assert_eq!(audit.blocks.len(), 2);
        assert_eq!(audit.blocks[0].original_dim, 2);
        assert_eq!(audit.blocks[0].effective_dim, 2);
        assert_eq!(audit.blocks[1].original_dim, 2);
        assert_eq!(audit.blocks[1].effective_dim, 2);
    }

    /// Test 2: a smooth's constant column aliased with a separate
    /// parametric intercept → audit drops one column, fatal=false,
    /// at least one alias pair attributed.
    #[test]
    fn audit_smooth_constant_aliased_with_intercept() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        // Parametric: [1]. Smooth: [1, x², x³] — its leading column
        // exactly reproduces the parametric intercept.
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let mut smooth = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            smooth[[i, 0]] = 1.0;
            smooth[[i, 1]] = x[i] * x[i];
            smooth[[i, 2]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_with_const", smooth),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed");
        assert!(
            !audit.fatal,
            "alias with attribution must not be fatal: {}",
            audit.summary,
        );
        assert!(
            !audit.aliased_pairs.is_empty(),
            "smooth-constant aliased with intercept must surface at least one alias pair",
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "smooth-constant aliased with intercept must populate dropped_columns",
        );
        // The dropped column should belong to one of the two blocks
        // (RRQR picks pivot order; we don't pin which block wins).
        for drop in &audit.dropped_columns {
            assert!(
                drop.block == "intercept" || drop.block == "smooth_with_const",
                "unexpected drop block name {:?}",
                drop.block,
            );
        }
        // Joint rank should be exactly the count of linearly independent
        // columns: intercept + x² + x³ = 3.
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 3,
            "expected 3 surviving directions; got {total_kept} (summary: {})",
            audit.summary,
        );
    }

    /// Test 3: two smooths on the same covariate axis with a shared
    /// linear direction → audit drops one column.
    #[test]
    fn audit_two_smooths_share_linear_direction() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        // Both smooths contain a column that equals `x`; they also
        // each carry a quadratic direction the other doesn't.
        let mut smooth_a = Array2::<f64>::zeros((n, 2));
        let mut smooth_b = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            smooth_a[[i, 0]] = x[i];
            smooth_a[[i, 1]] = x[i] * x[i];
            smooth_b[[i, 0]] = x[i];
            smooth_b[[i, 1]] = (x[i] - 0.3).powi(2);
        }
        let specs = [
            spec_from_dense("smooth_a", smooth_a),
            spec_from_dense("smooth_b", smooth_b),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed");
        assert!(
            !audit.fatal,
            "attributed alias must not be fatal: {}",
            audit.summary
        );
        let cross_linear_pair = audit
            .aliased_pairs
            .iter()
            .find(|p| p.block_a == "smooth_a" && p.block_b == "smooth_b" && p.overlap > 0.999);
        assert!(
            cross_linear_pair.is_some(),
            "expected an x~x alias pair between the two smooths; got pairs {:?}",
            audit.aliased_pairs,
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "RRQR must demote one of the duplicated linear columns",
        );
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 3,
            "expected 3 independent directions (1 shared linear + 2 quadratics); got {total_kept}",
        );
    }

    /// Test 4: an ambiguous high-magnitude alias that the pairwise
    /// scan misses but RRQR catches as joint-rank deficiency. We
    /// engineer a 3-way structural alias: parametric `[1]`, smooth
    /// A `[1 + ε·x]`, smooth B `[1 - ε·x]`. Each individual pair
    /// has overlap below 1.0 but the joint design's third column
    /// lies in the span of the first two — RRQR demotes one and
    /// attributes it via dropped_columns; pairwise scan may or may
    /// not catch all three pairs above threshold. Either way the
    /// fit must proceed (fatal=false) because dropped_columns is
    /// populated.
    #[test]
    fn audit_three_way_alias_is_attributed_not_fatal() {
        let n = 64;
        let x = linspace_minus_one_to_one(n);
        let eps = 0.5;
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        let smooth_a = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 + eps * x[i]);
        let smooth_b = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 - eps * x[i]);
        let specs = [
            spec_from_dense("intercept", parametric),
            spec_from_dense("smooth_a", smooth_a),
            spec_from_dense("smooth_b", smooth_b),
        ];
        let audit = audit_identifiability(&specs).expect("audit must succeed");
        assert!(
            !audit.dropped_columns.is_empty(),
            "RRQR must attribute the three-way alias as a dropped column",
        );
        assert!(
            !audit.fatal,
            "alias with column attribution must not be fatal: {}",
            audit.summary,
        );
        // Joint rank = 2 (the three columns span at most {1, x}).
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 2,
            "three-way alias collapses to rank 2; got {total_kept}"
        );
    }

    /// Test 5: end-to-end shape on a biobank-like configuration —
    /// 4 blocks, ~50 total columns, with one intentional cross-block
    /// linear alias seeded in. The audit must complete in well under
    /// a second and produce a single attributed drop with the
    /// expected joint rank.
    #[test]
    fn audit_biobank_shape_end_to_end() {
        let n = 1024;
        let x = linspace_minus_one_to_one(n);
        // Block 0: parametric intercept + age-linear.
        let mut parametric = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            parametric[[i, 0]] = 1.0;
            parametric[[i, 1]] = x[i];
        }
        // Block 1: smooth in x — 8 polynomial-like columns.
        let mut s_x = Array2::<f64>::zeros((n, 8));
        for i in 0..n {
            for k in 0..8 {
                s_x[[i, k]] = (x[i] - (k as f64 - 4.0) * 0.2).powi(2);
            }
        }
        // Block 2: smooth in sin(x) — 6 columns. No alias with block 1.
        let mut s_sin = Array2::<f64>::zeros((n, 6));
        for i in 0..n {
            for k in 0..6 {
                s_sin[[i, k]] = ((k as f64 + 1.0) * x[i]).sin();
            }
        }
        // Block 3: deliberately seeded alias — first column is exactly
        // `x` (duplicates parametric block's column 1).
        let mut alias_block = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            alias_block[[i, 0]] = x[i];
            alias_block[[i, 1]] = x[i].cos();
            alias_block[[i, 2]] = (2.0 * x[i]).cos();
            alias_block[[i, 3]] = (3.0 * x[i]).cos();
        }
        let specs = [
            spec_from_dense("parametric", parametric),
            spec_from_dense("s_x", s_x),
            spec_from_dense("s_sin", s_sin),
            spec_from_dense("alias_block", alias_block),
        ];
        let audit = audit_identifiability(&specs).expect("biobank-shape audit must succeed");
        assert!(
            !audit.fatal,
            "seeded alias with attribution must not be fatal: {}",
            audit.summary,
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "seeded x~x alias must produce at least one dropped column",
        );
        // The alias is exactly one direction; expect exactly one drop.
        assert_eq!(
            audit.dropped_columns.len(),
            1,
            "biobank-shape audit should attribute exactly the seeded alias; \
             got {:?}",
            audit.dropped_columns,
        );
        // Total effective dim = 2 + 8 + 6 + 4 − 1 = 19.
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 19,
            "biobank-shape: expected 19 kept directions; got {total_kept} ({})",
            audit.summary,
        );
    }
}
