// Pre-fit cross-block identifiability audit.
//
// Stage 1 design notes only. The implementation lands in later stages.
//
// # Where the existing dedupe lives
//
// `crate::families::bernoulli_marginal_slope::install_compiled_flex_block_into_runtime`
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
//    via `install_compiled_flex_block` + `InstalledFlexBlock` on a
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
// `install_compiled_flex_block_into_runtime` into a thin
// wrapper that builds the spec list (with the BMS-specific W-metric
// PIRLS row weights), delegates to `audit_identifiability`, then
// installs the resulting reparameterisation back into
// `DeviationRuntime` via the existing `InstalledFlexBlock` plumbing.
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

/// Overlap magnitude above which the audit halts the fit unconditionally.
///
/// At or above this threshold, two named blocks contribute the *same*
/// direction up to 1% relative noise. The joint penalised-Newton inner
/// solve has no unique minimiser in that direction — the inner KKT
/// system is structurally rank-deficient regardless of penalty value,
/// the outer Hessian inherits the same null space, and the outer
/// optimiser silently spins on the unattributed direction. This is the
/// "auto-subsample eval=1/12 hung for hours" failure mode the audit-gate
/// task (#5) installs the safety net for: the audit must refuse, with
/// an actionable message naming the offending blocks/columns and a
/// reparameterisation hint, BEFORE the outer solver ever enters.
///
/// Note this is strictly stricter than
/// `ALIAS_OVERLAP_REPORTING_THRESHOLD` (0.95). The 0.95 threshold is
/// for *reporting* — partial overlap that the penalty and line search
/// can still resolve. The 0.99 threshold is for *halting* — overlap so
/// close to 1.0 that no penalty value can distinguish the two
/// directions and no warm-start can untangle them. The aliasing-fix
/// peer's basis reparameterisation should keep biobank specs below
/// this threshold; if this gate ever fires on production specs after
/// their fix lands, that's a real bug surfaced rather than silently
/// absorbed.
const HARD_HALT_OVERLAP_THRESHOLD: f64 = 0.99;

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

    // Per-joint-column gauge priority, inherited from the owning block.
    // RRQR uses greedy column pivoting: at each step it picks the
    // remaining column with the largest residual norm. When two
    // columns carry the same direction (alias) the residual norms are
    // identical after the earlier of the two enters the kept set, so
    // RRQR's choice of "which one to keep" is determined by the order
    // it scans columns. We exploit that by presenting columns in
    // descending-priority order: high-priority columns are scanned
    // first and enter the kept set first, so when an alias collapses
    // it is the lower-priority block's column that gets demoted into
    // the trailing rank-deficient space.
    //
    // For survival marginal-slope this realises the gauge-ownership
    // contract: a shared affine direction between time_surface (high
    // priority) and marginal_surface (lower) is dropped from
    // marginal_surface, not from time_surface; a shared deviation
    // direction between marginal_surface and score_warp_dev (lowest)
    // is dropped from score_warp_dev. With all priorities equal (the
    // default = 100) `priority_perm` is the identity (stable sort
    // preserves spec order), so legacy callers see no behaviour
    // change.
    let mut priority_perm: Vec<usize> = (0..p_total).collect();
    let col_block_idx: Vec<usize> = (0..specs.len())
        .flat_map(|i| std::iter::repeat(i).take(col_offsets[i + 1] - col_offsets[i]))
        .collect();
    priority_perm.sort_by(|&a, &b| {
        let pa = specs[col_block_idx[a]].gauge_priority;
        let pb = specs[col_block_idx[b]].gauge_priority;
        pb.cmp(&pa).then_with(|| a.cmp(&b))
    });
    let priority_perm_is_identity = priority_perm
        .iter()
        .enumerate()
        .all(|(new_j, &old_j)| new_j == old_j);

    // Column-pivoted RRQR on the joint design. The pivot permutation
    // names which original columns were demoted past the rank
    // threshold, so we can attribute each dropped column back to its
    // (block, local_col) origin deterministically. `rrqr_with_permutation`
    // uses the same `RRQR_RANK_ALPHA · ε · max(m,n) · leading`
    // tolerance as the per-block diagonal counter above, so the joint
    // verdict and the per-block diagnostics are tolerance-consistent.
    let rrqr_started = std::time::Instant::now();
    log::info!(
        "[STAGE] identifiability audit: joint RRQR start n={} p_total={} priority_reorder={}",
        n,
        p_total,
        !priority_perm_is_identity,
    );
    let rrqr = if priority_perm_is_identity {
        rrqr_with_permutation(&x_joint, default_rrqr_rank_alpha())
            .map_err(|e| format!("identifiability audit joint RRQR failed: {e:?}"))?
    } else {
        let mut x_priority = Array2::<f64>::zeros((n, p_total));
        for (new_j, &old_j) in priority_perm.iter().enumerate() {
            x_priority.column_mut(new_j).assign(&x_joint.column(old_j));
        }
        rrqr_with_permutation(&x_priority, default_rrqr_rank_alpha()).map_err(|e| {
            format!("identifiability audit joint RRQR (priority-ordered) failed: {e:?}")
        })?
    };
    log::info!(
        "[STAGE] identifiability audit: joint RRQR end rank={}/{} elapsed={:.3}s",
        rrqr.rank,
        p_total,
        rrqr_started.elapsed().as_secs_f64(),
    );
    let joint_rank = rrqr.rank;
    let joint_rank_tol = rrqr.rank_tol;
    // RRQR's `column_permutation` indexes into the matrix it actually
    // saw. If we reordered by priority, map those back to original
    // joint-column indices so downstream block-attribution stays
    // correct.
    let map_to_original = |reordered_idx: usize| -> usize {
        if priority_perm_is_identity {
            reordered_idx
        } else {
            priority_perm[reordered_idx]
        }
    };
    let demoted_joint_cols: Vec<usize> = rrqr.column_permutation[rrqr.rank..]
        .iter()
        .map(|&j| map_to_original(j))
        .collect();

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
    let n_block_pairs = specs.len().saturating_mul(specs.len().saturating_sub(1)) / 2;
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
            if let Some(hb) = pairwise_block_heartbeat.as_ref() {
                hb.tick(1, |done, secs| {
                    log::info!(
                        "[STAGE] identifiability audit: pairwise overlap progress {done}/{n_block_pairs} block pairs in {secs:.1}s",
                    );
                });
            }
        }
    }
    log::info!(
        "[STAGE] identifiability audit: pairwise overlap scan done in {:.3}s ({} aliased pairs)",
        pairwise_started.elapsed().as_secs_f64(),
        aliased_pairs.len(),
    );

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
    // Hard-halt cases (the audit-gate from task #5 — the safety net that
    // would have prevented the biobank `eval=1/12` hours-long hang):
    //
    //   (a) `joint_rank < p_total` — the joint design is structurally
    //       rank-deficient. Whether RRQR attributed the deficiency via
    //       `dropped_columns` does NOT change the fact that the inner
    //       penalised-Newton KKT system has a non-trivial null space
    //       inherited by the outer Hessian: the outer optimiser will
    //       silently spin on the unattributed direction. The previous
    //       gate (`deficient && pairs.empty() && drops.empty()`) only
    //       fired when attribution was IMPOSSIBLE; that's the wrong
    //       boundary because "attributed" does not imply "fittable".
    //
    //   (b) Any pairwise overlap `>= HARD_HALT_OVERLAP_THRESHOLD` (0.99)
    //       between two named blocks. Two distinct blocks contributing
    //       the same direction to within 1% noise floor is structurally
    //       unfittable regardless of penalty values, regardless of
    //       whether the joint rank happens to be full at this n.
    //
    // RRQR's tolerance can disagree with the column-norm pairwise scan
    // on edge cases (e.g. a high-overlap pair that RRQR keeps because
    // its residual is just above the rank threshold), so the two
    // conditions are kept independent: either one is sufficient to halt.
    //
    // For multi-channel families (e.g. survival marginal-slope, K=4),
    // identical raw-X columns may live in orthogonal K-channels of the
    // row Jacobian and BE separately identifiable through the
    // likelihood. Those families must NOT route through this flat
    // audit; they invoke
    // `audit_identifiability_channel_aware` with their per-block row
    // Jacobian operators + structural row Hessian.
    let hard_alias_pair = aliased_pairs
        .iter()
        .max_by(|a, b| {
            a.overlap
                .partial_cmp(&b.overlap)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|p| p.overlap >= HARD_HALT_OVERLAP_THRESHOLD)
        .cloned();
    let fatal = joint_rank_deficient || hard_alias_pair.is_some();

    let fatal_detail = if fatal {
        let mut parts: Vec<String> = Vec::new();
        if joint_rank_deficient {
            // Name the worst-attributed dropped column (if any) so the
            // caller sees which (block, local_col) to reparameterise
            // first. When attribution is empty, surface that fact —
            // it indicates a >2-way structural alias that the pairwise
            // scan didn't catch and is the hardest case to fix.
            let attribution = if let Some(first_drop) = dropped_columns.first() {
                format!(
                    "first attributed drop: block '{}' local column {} \
                     (reparam: replace this column with a sum-to-zero or \
                     orthogonal-complement projection against earlier blocks, \
                     or remove the redundant term entirely)",
                    first_drop.block, first_drop.column,
                )
            } else {
                "no per-column attribution (>2-way structural alias); \
                 audit the joint design by eye and absorb the shared null \
                 subspace into a single parametric block"
                    .to_string()
            };
            parts.push(format!(
                "joint rank {} < joint columns {} ({} dropped column(s); {})",
                joint_rank,
                p_total,
                dropped_columns.len(),
                attribution,
            ));
        }
        if let Some(pair) = hard_alias_pair.as_ref() {
            parts.push(format!(
                "alias pair: '{}'[{}] ~ '{}'[{}] overlap={:.4} >= halt threshold {:.2} \
                 (reparam: orthogonalise one block's column {} against the other \
                 via sum-to-zero, or absorb the shared direction into a single \
                 parametric block)",
                pair.block_a,
                pair.direction_a,
                pair.block_b,
                pair.direction_b,
                pair.overlap,
                HARD_HALT_OVERLAP_THRESHOLD,
                pair.direction_b,
            ));
        }
        format!(" — FATAL: {}", parts.join("; "))
    } else if !aliased_pairs.is_empty() {
        " — partial alias(es) below halt threshold; penalty + line search will resolve".to_string()
    } else {
        " — clean".to_string()
    };

    let summary = format!(
        "identifiability audit: {} block(s), {} joint columns, joint rank {}, \
         {} alias pair(s) above overlap {:.3}, {} dropped column(s){}",
        specs.len(),
        p_total,
        joint_rank,
        aliased_pairs.len(),
        ALIAS_OVERLAP_REPORTING_THRESHOLD,
        dropped_columns.len(),
        fatal_detail,
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
            gauge_priority: 100,
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
    /// parametric intercept → audit drops one column AND now flags
    /// the configuration as fatal under the task #5 halt-or-repair gate.
    /// The two blocks contribute the same direction up to numerical
    /// noise (overlap == 1.0): the inner KKT system is structurally
    /// rank-deficient regardless of penalty, so the audit must refuse
    /// the fit with an actionable message rather than warn-and-proceed.
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
            audit.fatal,
            "exact intercept~smooth-constant alias must be fatal under the halt gate: {}",
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
        // Actionable message must name both offending blocks and the
        // reparam suggestion so the caller can act on the failure.
        assert!(
            audit.summary.contains("intercept") && audit.summary.contains("smooth_with_const"),
            "fatal summary must name both blocks; got {:?}",
            audit.summary,
        );
        assert!(
            audit.summary.contains("reparam")
                || audit.summary.contains("sum-to-zero")
                || audit.summary.contains("orthogonal")
                || audit.summary.contains("absorb"),
            "fatal summary must include a reparameterisation suggestion; got {:?}",
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
        // Under the task #5 halt gate, an x~x alias at overlap ~ 1.0 is
        // fatal — two distinct blocks contributing the same direction is
        // structurally unfittable regardless of attribution.
        assert!(
            audit.fatal,
            "exact x~x alias across two smooth blocks must be fatal under the halt gate: {}",
            audit.summary,
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
        // Under the task #5 halt gate, joint_rank < joint_cols is fatal
        // even with attribution: the inner KKT system inherits the
        // unattributed null direction and the outer optimiser will spin.
        assert!(
            audit.fatal,
            "three-way alias with joint rank < joint cols must be fatal under the halt gate: {}",
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
        // The seeded x~x alias is exactly the biobank failure shape the
        // task #5 halt gate exists to refuse: two distinct blocks
        // contributing the same direction at overlap 1.0. Must be fatal.
        assert!(
            audit.fatal,
            "seeded biobank-shape exact x~x alias must be fatal under the halt gate: {}",
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
