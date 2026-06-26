// Cross-block identifiability canonicalisation.
//
// The pre-fit `audit_identifiability` (see `crate::audit`)
// runs a joint RRQR on `[X_block_0 | X_block_1 | ...]` and reports per-
// block (block_idx, local_col) drops attributing each demoted joint
// column back to its origin. This module **previously** converted that
// report into a concrete coordinate transform that reduced the inner
// solve to a (p_raw → r_reduced) subspace via selection-T.
//
// Naively substituting a column-reduced `ParameterBlockSpec` under a family
// that captures its design at construction time was unsafe: blockwise families
// (e.g. `SurvivalMarginalSlopeFamily::marginal_design`, `::logslope_design`)
// run `evaluate_blockwise_exact_newton` row-Hessian assembly through
// `DesignMatrix::syr_row_into_view` / `::row_outer_into_view`, which assert that
// the target slice's column count equals the captured design's column count.
// A raw-width callback fed a reduced spec produces `DesignMatrix::syr_row_into
// shape mismatch` (matrix.rs:6529), a panic later in the pipeline that masks the
// audit's diagnostic.
//
// #933 inverts that ownership so the family never sees raw coordinates: a
// reduced callback block carries a [`GaugeComposedJacobian`] that post-multiplies
// the family's raw-width effective Jacobian by the block's selection `T_b`, so
// the effective Jacobian emerges at the reduced width and the column-count
// assertions hold BY CONSTRUCTION.
//
// This module's safe behaviors:
//   - plain single-channel dense blocks are exactly orthogonalized with a
//     per-block transform, then lifted back to raw coordinates after fitting;
//   - `jacobian_callback`-only blocks (no `stacked_design`) are column-reduced
//     with their callback wrapped in `GaugeComposedJacobian`, so the family fits
//     in the reduced section and the one Gauge lifts back through the same `T_b`;
//   - `stacked_design` blocks keep raw block widths even when the audit
//     attributes weak columns, because their `3·n`-row eta operator carries the
//     family's hidden z-lift / monotonicity / leading-fixed-column layout
//     (#892, #1068) that a raw-column drop would desynchronise from the reduced β.
//
// Fatal audit results still fail closed with an immediate
// `CustomFamilyError::IdentifiabilityFailure`, naming the offending blocks and a
// reparameterization hint in milliseconds rather than after a singular Newton
// solve. Clean or safely canonicalized results carry explicit per-block
// transforms so `lift_block_states_to_raw` and `lift_fit_geometry_to_raw` map
// the fit back to the raw coordinate system.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::audit::{
    IdentifiabilityAudit, audit_identifiability, audit_identifiability_channel_aware,
    block_structural_penalty_dense, priority_tiered_rank_from_gram, rank_of_gram,
};
use crate::families::compiler::{
    IdentityRowHessian, RowJacobianOperator, orthogonalize_design_blocks, symmetric_sqrt_into,
};
use gam_linalg::faer_ndarray::{
    default_rrqr_rank_alpha, fast_ata, fast_atb, rrqr_with_permutation,
};
use gam_linalg::matrix::{CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix};
use gam_problem::Gauge;
use gam_problem::{
    BlockEffectiveJacobian, CustomFamilyError, FamilyLinearizationState, ParameterBlockSpec,
    PenaltyMatrix,
};

enum BlockJacobianSource {
    Callback(Arc<dyn BlockEffectiveJacobian>),
    FlatDesign(DesignMatrix),
}

/// A lazy [`RowJacobianOperator`] for identifiability audit blocks.
///
/// Callback blocks stream `BlockEffectiveJacobian::effective_jacobian_rows`;
/// plain blocks stream `DesignMatrix::row_chunk_into` and embed the flat rows in
/// channel 0. No `(n, p, K)` tensor is held across blocks.
struct BlockJacobianAsRowOp {
    source: BlockJacobianSource,
    n: usize,
    p: usize,
    k_block: usize,
    k_target: usize,
    block_name: String,
}

impl BlockJacobianAsRowOp {
    /// Build from a `BlockEffectiveJacobian` callback.
    ///
    /// `n_rows` is the number of training observations; `k_target` is the
    /// channel count of the emitted tensor. A callback whose `n_outputs()`
    /// is smaller than `k_target` has its Jacobian embedded in the leading
    /// channels with the trailing channels zero — built directly into the
    /// padded tensor, so no intermediate `(n, p, k_block)` tensor plus copy
    /// is ever materialized (at biobank scale every avoided `(n, p, k)`
    /// duplicate is hundreds of MiB, #979). The `effective_jacobian_at`
    /// call uses `beta = 0` and `family_scalars = None`.
    fn from_callback(
        cb: Arc<dyn BlockEffectiveJacobian>,
        n_rows: usize,
        p_block: usize,
        k_target: usize,
        block_name: &str,
    ) -> Result<Self, String> {
        let k = cb.n_outputs();
        if k == 0 {
            return Err(format!(
                "BlockJacobianAsRowOp block '{block_name}': n_outputs=0 is invalid"
            ));
        }
        if k > k_target {
            return Err(format!(
                "BlockJacobianAsRowOp block '{block_name}': n_outputs({k}) exceeds the \
                 audit channel count k_target({k_target})"
            ));
        }
        Ok(Self {
            source: BlockJacobianSource::Callback(cb),
            n: n_rows,
            p: p_block,
            k_block: k,
            k_target,
            block_name: block_name.to_string(),
        })
    }

    fn from_flat_design(
        design: DesignMatrix,
        n_rows: usize,
        k_target: usize,
        block_name: &str,
    ) -> Self {
        let p = design.ncols();
        Self {
            source: BlockJacobianSource::FlatDesign(design),
            n: n_rows,
            p,
            k_block: 1,
            k_target,
            block_name: block_name.to_string(),
        }
    }

    fn zero_state() -> FamilyLinearizationState<'static> {
        FamilyLinearizationState {
            beta: &[],
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        }
    }

    fn stacked_rows(&self, start: usize, end: usize) -> Result<Array2<f64>, String> {
        match &self.source {
            BlockJacobianSource::Callback(cb) => {
                let state = Self::zero_state();
                let stacked = cb
                    .effective_jacobian_rows(&state, start..end)
                    .map_err(|e| {
                        format!("BlockJacobianAsRowOp block '{}': {e}", self.block_name)
                    })?;
                let chunk = end - start;
                if stacked.nrows() != self.k_block * chunk || stacked.ncols() != self.p {
                    return Err(format!(
                        "BlockJacobianAsRowOp block '{}': effective_jacobian_rows returned \
                         shape {:?}, expected [{}, {}]",
                        self.block_name,
                        stacked.shape(),
                        self.k_block * chunk,
                        self.p,
                    ));
                }
                Ok(stacked)
            }
            BlockJacobianSource::FlatDesign(design) => {
                let chunk = end - start;
                let mut out = Array2::<f64>::zeros((chunk, self.p));
                design
                    .row_chunk_into(start..end, out.view_mut())
                    .map_err(|e| {
                        format!(
                            "BlockJacobianAsRowOp block '{}': flat design row chunk failed: {e}",
                            self.block_name
                        )
                    })?;
                Ok(out)
            }
        }
    }
}

impl RowJacobianOperator for BlockJacobianAsRowOp {
    fn k(&self) -> usize {
        self.k_target
    }
    fn ncols(&self) -> usize {
        self.p
    }
    fn nrows(&self) -> usize {
        self.n
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        let k = self.k();
        assert_eq!(out.len(), k);
        assert_eq!(delta_beta.len(), self.ncols());
        for r in 0..k {
            out[r] = 0.0;
        }
        let stacked = self
            .stacked_rows(row, row + 1)
            .expect("BlockJacobianAsRowOp::apply_row failed to read row");
        for r in 0..self.k_block {
            for (j, &b) in delta_beta.iter().enumerate() {
                out[r] += stacked[[r, j]] * b;
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let entries = self.n.saturating_mul(self.p).saturating_mul(self.k_target);
        const MAX_EVALUATE_FULL_ENTRIES: usize = 10_000_000;
        assert!(
            entries <= MAX_EVALUATE_FULL_ENTRIES,
            "BlockJacobianAsRowOp::evaluate_full refused to materialize {entries} entries"
        );
        let mut out = Array3::<f64>::zeros((self.n, self.p, self.k_target));
        for start in (0..self.n).step_by(4096) {
            let end = (start + 4096).min(self.n);
            let stacked = self
                .stacked_rows(start, end)
                .expect("BlockJacobianAsRowOp::evaluate_full failed to read row chunk");
            let chunk = end - start;
            for ch in 0..self.k_block {
                for local_i in 0..chunk {
                    for col in 0..self.p {
                        out[[start + local_i, col, ch]] = stacked[[ch * chunk + local_i, col]];
                    }
                }
            }
        }
        out
    }
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        let n = self.nrows();
        let p = self.ncols();
        let k = self.k();
        assert_eq!(h_full.shape(), &[n, k, k]);
        let mut out = Array2::<f64>::zeros((n * k, p));
        let mut sqrt_h = Array2::<f64>::zeros((k, k));
        let mut h_i = Array2::<f64>::zeros((k, k));
        for start in (0..n).step_by(4096) {
            let end = (start + 4096).min(n);
            let chunk = end - start;
            let mut rows = Array2::<f64>::zeros((chunk * k, p));
            self.channel_flattened_rows(start..end, &mut rows);
            for local_i in 0..chunk {
                let row = start + local_i;
                for a in 0..k {
                    for b in 0..k {
                        h_i[[a, b]] = h_full[[row, a, b]];
                    }
                }
                symmetric_sqrt_into(&h_i, &mut sqrt_h);
                for ch in 0..k {
                    let dst = row * k + ch;
                    for col in 0..p {
                        let mut acc = 0.0;
                        for cp in 0..k {
                            acc += sqrt_h[[ch, cp]] * rows[[local_i * k + cp, col]];
                        }
                        out[[dst, col]] = acc;
                    }
                }
            }
        }
        out
    }
    fn channel_flattened_column(&self, col: usize, out: &mut [f64]) {
        let n = self.nrows();
        let k = self.k();
        assert!(
            col < self.ncols(),
            "BlockJacobianAsRowOp::channel_flattened_column col {col} out of range {}",
            self.ncols()
        );
        assert_eq!(out.len(), n * k);
        let mut offset = 0usize;
        for start in (0..n).step_by(4096) {
            let end = (start + 4096).min(n);
            let chunk = end - start;
            let mut rows = Array2::<f64>::zeros((chunk * k, self.p));
            self.channel_flattened_rows(start..end, &mut rows);
            for local_i in 0..chunk {
                for ch in 0..k {
                    out[offset + local_i * k + ch] = rows[[local_i * k + ch, col]];
                }
            }
            offset += chunk * k;
        }
    }

    fn channel_flattened_rows(&self, rows: std::ops::Range<usize>, out: &mut Array2<f64>) {
        let start = rows.start.min(self.n);
        let end = rows.end.min(self.n);
        let chunk = end - start;
        assert_eq!(out.shape(), &[chunk * self.k_target, self.p]);
        out.fill(0.0);
        let stacked = self
            .stacked_rows(start, end)
            .expect("BlockJacobianAsRowOp::channel_flattened_rows failed to read rows");
        for ch in 0..self.k_block {
            for local_i in 0..chunk {
                for col in 0..self.p {
                    out[[local_i * self.k_target + ch, col]] = stacked[[ch * chunk + local_i, col]];
                }
            }
        }
    }
}

/// Specs after pre-fit cross-block identifiability canonicalisation.
///
/// `reduced_specs[i]` carries an `r_i`-column design wrapping the raw
/// `p_i`-column design via `CoefficientTransformOperator`. Penalties
/// are pulled back as `T_iᵀ S_k T_i`. `gauge` is the block-diagonal
/// [`Gauge`] whose block `i` slab is the raw-to-reduced transform
/// `T_i` of shape `(p_i_raw, r_i)`; it owns every lift back to raw
/// coordinates (`lift_block_betas` for β, `lift_covariance` for the
/// joint covariance / penalized Hessian).
///
/// `used_channel_aware_audit` is `true` when the multi-channel path was
/// taken (i.e. at least one block declared `n_outputs > 1` via its
/// `jacobian_callback`).  Tests that assert routing correctness inspect
/// this field directly.
#[derive(Debug)]
pub struct CanonicalSpecs {
    pub reduced_specs: Vec<ParameterBlockSpec>,
    pub gauge: Gauge,
    pub audit: IdentifiabilityAudit,
    /// `true` iff the audit was routed through `audit_identifiability_channel_aware`
    /// (multi-channel families such as survival marginal-slope).
    pub used_channel_aware_audit: bool,
}

/// Run the pre-fit cross-block identifiability audit. Fail-closed
/// safety gate (see module docs).
///
/// Behaviour:
///   - If the audit cleanly passes (`!fatal`), each `T_i` is the
///     identity and the reduced specs are clones of the raw specs.
///     The lift/sandwich machinery downstream becomes a no-op.
///   - If the audit is `fatal=true` for **any** cause (joint rank
///     deficiency with attributed drops, joint rank deficiency
///     without attribution, or a hard-overlap alias pair), we refuse
///     the fit with `CustomFamilyError::IdentifiabilityFailure`. The
///     audit summary names the offending blocks and a reparameterisation
///     hint, giving the caller a millisecond-scale diagnostic instead
///     of a downstream `syr_row_into shape mismatch` panic when the
///     family captures raw-width designs.
///
/// # Multi-channel routing
///
/// When any spec's `jacobian_callback` reports `n_outputs > 1` (i.e.
/// the block contributes to multiple stacked output channels — as in
/// survival marginal-slope where marginal and logslope blocks target
/// orthogonal channels of the per-row Jacobian), this function routes
/// through [`audit_identifiability_channel_aware`] instead of the flat
/// [`audit_identifiability`].
///
/// The routing decision is principled: for each spec, call
/// `effective_jacobian_at(beta=0)` and check whether the returned
/// matrix has `nrows > n` (i.e. `nrows == n * k` for some `k > 1`).
/// If any block satisfies this, all blocks are treated as multi-channel
/// and [`BlockJacobianAsRowOp`] adapters are built from each spec's
/// callback to feed the channel-aware audit.
///
/// For specs without a `jacobian_callback`, the flat design with a
/// single-channel identity operator is used.
///
/// # Invariant assertion
///
/// After building the transform `T`, the post-T joint Jacobian
/// `J_can = J · T_full` is materialised and RRQR-checked.  If
/// `rank(J_can) != rank(J)`, the transform `T` is defective (a bug in
/// its construction) and the function returns
/// `CustomFamilyError::DimensionMismatch` with a diagnostic naming
/// `rank(J)`, `rank(J_can)`, and all per-block `T_i` shapes.
pub fn canonicalize_for_identifiability(
    specs: &[ParameterBlockSpec],
) -> Result<CanonicalSpecs, CustomFamilyError> {
    // Robustness is unconditional: always attempt the exact W-metric
    // orthogonalisation pass before the fail-closed audit. `try_orthogonalize_
    // blocks` is self-gating (it returns `None` — and the audit runs unchanged —
    // unless there are ≥2 plain single-channel dense blocks with an actual
    // structural overlap to remove, deferring on any family-owned-geometry block,
    // any multi-channel `stacked_design` block, and on clean designs), so this is
    // byte-identical wherever there is nothing
    // to orthogonalise.
    canonicalize_for_identifiability_inner(specs, true)
}

/// Core canonicalisation worker.
///
/// `orthogonalize` is an INTERNAL recursion-control flag (NOT a user knob): the
/// public entry passes `true` to attempt the exact orthogonalisation pass; the
/// post-orthogonalisation recursion below passes `false` so the already-reduced
/// specs are audited without re-orthogonalising.
///
/// When orthogonalisation runs, a general exact W-metric pass reparameterises
/// overlapping design blocks (e.g. a logslope surface confounded with the
/// marginal surface) so the lower-priority block's overlap with the
/// higher-priority anchor is removed exactly, rather than being penalised by a
/// hand-tuned ridge. The reparam `V_b` is folded into each block's design via
/// [`CoefficientTransformOperator`], penalties are pulled back as `V_bᵀ S V_b`,
/// and the block-diagonal [`Gauge`] carries `V_b` so the shared
/// [`Gauge::lift_block_betas`] / [`Gauge::lift_covariance`] machinery maps the
/// reduced fit back to raw coordinates unchanged (the lift is `β_raw = V_b · θ`,
/// already supported for dense transforms).
/// Rank of `j` under the SAME σ²-Gram eigenvalue convention the
/// channel-aware audit uses to decide which columns to drop
/// (`families::compiler::keep_positive_eigenspace`).
///
/// The post-T rank invariant must compare `rank(J)` and `rank(J_can)` with the
/// exact threshold/operator under which the audit declared the dropped columns
/// gauge — otherwise it compares two non-comparable ranks. The audit ranks a
/// design by the positive eigenspace of its Gram `G = JᵀJ`, retaining an
/// eigenvalue `λ_i` iff `λ_i > scale · 64 · (n·k) · ε` with
/// `scale = max(λ_max, tr(G))`. Because `λ = σ²`, that is a σ²-space cutoff:
/// near-separable directions with `σ_i/σ_max` between ~`√ε` and the RRQR floor
/// (~`ε`) are gauge under THIS metric but full-rank under a fresh σ-space RRQR.
/// On the near-separable penguin multinomial the two metrics disagreed by 4
/// directions (164 vs 160; gam#1220) — the audit (correctly, by its own
/// convention) dropped 4 weak softmax-channel directions that the post-T RRQR
/// still counted, tripping the invariant. Ranking BOTH `J` and `J_can` here
/// makes the invariant hold by construction for a pure column-selection `T`
/// (dropping audit-gauge columns), while still catching a genuine `T` bug:
/// dropping a column the audit deemed identifiable lowers `rank_audit(J_can)`
/// below `rank_audit(J)`.
///
/// `nk_scale` is the channel-major design row count `n·K` (= `j.nrows()`), the
/// `n_total` term `rank_of_gram`/`count_rank` use to size their tolerance.
///
/// This verification ranks the PENALTY-AUGMENTED reduced Gram
/// `J_canᵀJ_can + Σ_block SᵀS` with the AUDIT'S OWN rank function
/// (`rank_of_gram` → `count_rank`), so the post-T check and the audit's keep
/// decision use the identical augmentation AND the identical tolerance by
/// construction — they can no longer disagree on a faithful column-selection `T`.
/// Two ingredients that the earlier bare-eigenvalue verification dropped, each of
/// which produced a FALSE "post-T rank invariant violated":
///   • PENALTY AUGMENTATION (channel-aware analogue of the flat #1391 fix). The
///     channel-aware audit keeps a direction that is data-WEAK but penalty-
///     ANCHORED, because it ranks `JᵀJ + Σ SᵀS`
///     (`channel_aware_penalty_aware_joint_rank`). The bare `J_can` Gram
///     undercounts exactly those kept directions — e.g. the wiggliness-penalised
///     null modes a softmax channel shares once the cross-class σ²-coupling thins
///     their data signal (multinomial `s(x) + s(x, by=g)`: 28 vs p_red 30).
///   • TOLERANCE MATCH. `count_rank` keeps a singular value `σ=√λ` down to
///     `rank_alpha·ε·n·σ_max`; the old eigenvalue cutoff `λ > scale·64·n·ε` was
///     ~ε larger and demoted penalty-covered modes whose `λ` sits between
///     `ε²λ_max` and `ε·λ_max` (Gaussian survival location-scale: 16 vs p_red 18).
/// The bare data Gram is recovered when no block is penalised (or the block
/// layout does not tile the columns), so unpenalised channel-aware fits are
/// unaffected.
fn audit_convention_rank(j: &Array2<f64>, nk_scale: usize, blocks: &[FlatRankBlock]) -> usize {
    let p = j.ncols();
    if p == 0 || j.nrows() == 0 {
        return 0;
    }
    // G = JᵀJ (p × p), symmetric PSD — same Gram the audit eigendecomposes.
    let mut gram = fast_ata(j);
    // Augment block-diagonally with each block's SᵀS, exactly as the audit's
    // `channel_aware_penalty_aware_joint_rank` builds its augmented Gram. Only
    // applied when the declared block widths tile the design columns; otherwise
    // fall back to the bare Gram (no spurious augmentation).
    let declared: usize = blocks.iter().map(|b| b.width).sum();
    let mut n_penalty_rows = 0usize;
    if declared == p {
        let mut col_off = 0usize;
        for b in blocks {
            if let Some(s) = b.structural_penalty.as_ref()
                && s.nrows() == b.width
                && s.ncols() == b.width
                && b.width > 0
            {
                let sts = fast_atb(s, s);
                let mut sub = gram.slice_mut(ndarray::s![
                    col_off..col_off + b.width,
                    col_off..col_off + b.width
                ]);
                sub += &sts;
                n_penalty_rows += b.width;
            }
            col_off += b.width;
        }
    }
    // Rank the augmented Gram with the AUDIT'S OWN singular-value convention
    // (`rank_of_gram` → `count_rank`), so the post-T verification and the audit's
    // keep decision share the identical tolerance. The penalty rows enlarge the
    // tall-design row count exactly as the audit's
    // `rank_of_gram(.., n_design_rows + n_penalty_rows)` does. The earlier
    // eigenvalue cutoff `λ > scale·64·n·ε` was ~ε larger than `count_rank`'s
    // σ-space `rank_alpha·ε·n·σ_max` floor and demoted penalty-covered modes whose
    // `λ` sits between `ε²λ_max` and `ε·λ_max` (Gaussian survival location-scale:
    // 16 vs p_red 18). On eigendecomposition failure fall back to the structural
    // column count (no demotion), so a numerical hiccup never becomes a spurious
    // violation.
    rank_of_gram(&gram, nk_scale.saturating_add(n_penalty_rows)).unwrap_or(p)
}

/// Per-block descriptor for the penalty-augmented priority-tiered rank: the
/// block's column WIDTH in the joint design, its structural penalty `S` (dense,
/// `width × width`, `None` when unpenalised), and its `gauge_priority`.
struct FlatRankBlock {
    width: usize,
    structural_penalty: Option<Array2<f64>>,
    priority: u8,
}

/// Numerical rank of the joint design `j` under the EXACT convention the FLAT
/// identifiability audit (`audit_identifiability`) uses to decide which joint
/// columns to demote.
///
/// The flat audit does NOT rank the bare design with a plain column-pivoted RRQR.
/// It ranks the PENALTY-AUGMENTED joint design `[X_joint; S_blockdiag]` via a
/// GAUGE-PRIORITY-TIERED pivoted Cholesky on its Gram
/// `Gₐ = JᵀJ + Σ_block SᵀS` (`priority_tiered_rank_from_gram` on
/// `joint_gram_aug`). Two ingredients of that convention were missing from the
/// earlier #1391 fix, which re-ranked the bare `J` with `rrqr_with_permutation`:
///
///   • PENALTY AUGMENTATION. For penalized (P-spline / PS) bases the structural
///     penalty `S` annihilates the smooth's affine null space, so the audit's
///     `JᵀJ + SᵀS` rank can differ from `rank(J)`. A direction that is
///     data-full-rank but penalty-null — e.g. a constant shared between a GAMLSS
///     mu-PS and sigma-PS block — is counted by the bare RRQR yet is exactly the
///     kind of direction the augmented, priority-tiered audit demotes from the
///     lower-priority block. The bare RRQR therefore over-counted `rank(J_pre)`,
///     so a faithful column-selection `T` reported `rank(J_pre) > rank(J_can)`
///     and tripped a false invariant violation (GAMLSS-PS, gam#1391; survival
///     marginal-slope, gam#1388).
///   • PRIORITY TIERING. The audit demotes the alias from the LOWER-priority
///     block; a norm-pivoted RRQR ignores priority and can keep/drop a different
///     column, so its rank verdict need not match the audit's drop set.
///
/// The post-T invariant ranks the REDUCED design `J_can` with this convention
/// and certifies it is FULL column rank (`== p_red`): a faithful
/// column-selection `T` (which removes exactly the audit-demoted columns) leaves
/// the kept columns independent under this metric, while a defective `T` that
/// drops a column the audit kept makes `J_can` rank-deficient and is caught.
fn flat_audit_convention_rank(j: &Array2<f64>, blocks: &[FlatRankBlock]) -> usize {
    let p = j.ncols();
    if p == 0 || j.nrows() == 0 {
        return 0;
    }
    // Sum of declared block widths must cover the design columns; if the layout
    // is inconsistent, fall back to the bare RRQR rank (no spurious demotion).
    let declared: usize = blocks.iter().map(|b| b.width).sum();
    if declared != p {
        return match rrqr_with_permutation(j, default_rrqr_rank_alpha()) {
            Ok(rrqr) => rrqr.rank,
            Err(_) => p,
        };
    }

    // Gram G = JᵀJ (p × p), the same column inner products the audit's
    // priority-tiered pivoted Cholesky consumes.
    let mut gram = fast_ata(j);

    // Augment block-diagonally with each block's SᵀS, exactly as
    // `audit_identifiability` builds `joint_gram_aug`. `n_penalty_rows` mirrors
    // the audit's tall row count for the tolerance scaling.
    let mut col_off = 0usize;
    let mut n_penalty_rows = 0usize;
    let mut col_priority: Vec<u8> = Vec::with_capacity(p);
    for b in blocks {
        for _ in 0..b.width {
            col_priority.push(b.priority);
        }
        if let Some(s) = b.structural_penalty.as_ref() {
            if s.nrows() == b.width && s.ncols() == b.width && b.width > 0 {
                let sts = fast_atb(s, s);
                let mut sub = gram.slice_mut(ndarray::s![
                    col_off..col_off + b.width,
                    col_off..col_off + b.width
                ]);
                sub += &sts;
                n_penalty_rows += b.width;
            }
        }
        col_off += b.width;
    }

    let m_rows = j.nrows() + n_penalty_rows;
    let tiered =
        priority_tiered_rank_from_gram(&gram, &col_priority, m_rows, default_rrqr_rank_alpha());
    tiered.rank
}

fn canonicalize_for_identifiability_inner(
    specs: &[ParameterBlockSpec],
    orthogonalize: bool,
) -> Result<CanonicalSpecs, CustomFamilyError> {
    // Exact orthogonalisation of structural confounds. Runs only on the top-
    // level entry AND only where the design is single-channel dense (the general
    // multi-channel coupled path is handled by the Tier-B joint-Newton Jeffreys
    // term, not by a per-block design reparam). On any structural condition that
    // the orthogonaliser cannot express as a per-block transform, it falls
    // through to the unmodified audit gate below — never worse than today.
    if orthogonalize {
        if let Some(canon) = try_orthogonalize_blocks(specs)? {
            return Ok(canon);
        }
    }
    if specs.is_empty() {
        return Ok(CanonicalSpecs {
            reduced_specs: Vec::new(),
            gauge: Gauge::identity(&[]),
            audit: audit_identifiability(specs).map_err(|r| {
                CustomFamilyError::DimensionMismatch {
                    reason: format!("pre-fit identifiability audit failed: {r}"),
                }
            })?,
            used_channel_aware_audit: false,
        });
    }

    // `design.nrows() == n_obs` is a struct invariant for every
    // ParameterBlockSpec: the canonical n-row operator is what the audit
    // and shape policy read.  Multi-channel survival LS stacked operators
    // live in `stacked_design` and are deliberately invisible here.
    let n_rows = specs[0].design.nrows();

    // ── Multi-channel routing decision ───────────────────────────────────
    //
    // Probe each spec's effective Jacobian at beta=0 to detect
    // multi-output blocks.  A block is multi-output when the returned
    // matrix has nrows > n_rows — i.e. nrows == n_rows * k for k > 1.
    // We use the spec-level n_outputs() shortcut when a jacobian_callback
    // is present; otherwise the block is always single-output (k=1).
    let max_n_outputs = specs
        .iter()
        .map(|s| {
            s.jacobian_callback
                .as_ref()
                .map(|cb| cb.n_outputs())
                .unwrap_or(1)
        })
        .max()
        .unwrap_or(1);
    let use_channel_aware = max_n_outputs > 1;

    log::debug!(
        "[CANON] canonicalize_for_identifiability: blocks={} n_rows={} \
         max_n_outputs={} route={}",
        specs.len(),
        n_rows,
        max_n_outputs,
        if use_channel_aware {
            "channel-aware"
        } else {
            "flat"
        },
    );

    // ── Per-block Jacobian Frobenius-norm logging (instrumentation) ──────
    //
    // Log the Frobenius norm and row-count of each block's effective
    // Jacobian before the audit so discrepancies between pilot and outer-fit
    // audits are visible in the log stream.
    //
    // This is purely diagnostic: the only consumer of `frob_sq` is the
    // `log::debug!` below.  A full `effective_jacobian_at` probe materialises
    // the block's entire `(n·k, p)` effective Jacobian — an `(n, p, k)`-class
    // transient that at biobank scale is hundreds of MiB per block, paid every
    // canonicalisation even when debug logging is OFF (#979).  Gate the whole
    // loop behind the log level so production fits (info/warn) pay nothing, and
    // when it does run, accumulate the Frobenius norm by streaming 4096-row
    // chunks instead of holding the full Jacobian.
    if log::log_enabled!(log::Level::Debug) {
        const FROB_CHUNK: usize = 4096;
        for spec in specs.iter() {
            let k = spec
                .jacobian_callback
                .as_ref()
                .map(|cb| cb.n_outputs())
                .unwrap_or(1);
            let jac_nrows = if use_channel_aware {
                n_rows * k
            } else {
                n_rows
            };
            let p = spec.design.ncols();
            let zeros = vec![0.0f64; p];
            let state = FamilyLinearizationState {
                beta: &zeros,
                family_scalars: None,
                channel_hessian: None,
                probit_frailty_scale: 1.0,
            };
            let mut frob_sq = 0.0_f64;
            let mut probe_err: Option<String> = None;
            for start in (0..n_rows).step_by(FROB_CHUNK) {
                let end = (start + FROB_CHUNK).min(n_rows);
                let chunk = match spec.jacobian_callback.as_ref() {
                    Some(cb) => cb.effective_jacobian_rows(&state, start..end),
                    None => {
                        let mut out = Array2::<f64>::zeros((end - start, p));
                        spec.design
                            .row_chunk_into(start..end, out.view_mut())
                            .map(|()| out)
                            .map_err(|e| e.to_string())
                    }
                };
                match chunk {
                    Ok(rows) => frob_sq += rows.iter().map(|v| v * v).sum::<f64>(),
                    Err(e) => {
                        probe_err = Some(e);
                        break;
                    }
                }
            }
            match probe_err {
                Some(e) => log::debug!(
                    "[CANON]   block '{}': effective_jacobian probe failed: {e}",
                    spec.name,
                ),
                None => log::debug!(
                    "[CANON]   block '{}': p={} jac_nrows={} frob_norm={:.4e}",
                    spec.name,
                    p,
                    jac_nrows,
                    frob_sq.sqrt(),
                ),
            }
        }
    }

    // ── Run the audit ─────────────────────────────────────────────────────
    let audit = if use_channel_aware {
        // Determine the common k (all blocks must agree on the channel count;
        // blocks without a jacobian_callback get a single-channel identity
        // adapter at k = max_n_outputs).
        let k = max_n_outputs;
        let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::with_capacity(specs.len());
        for spec in specs.iter() {
            let op: Arc<dyn RowJacobianOperator> = match spec.jacobian_callback.as_ref() {
                Some(cb) => {
                    // `from_callback` zero-pads the trailing channels for
                    // blocks with fewer outputs than the audit's common k,
                    // building the padded tensor directly.
                    let row_op = BlockJacobianAsRowOp::from_callback(
                        Arc::clone(cb),
                        n_rows,
                        spec.design.ncols(),
                        k,
                        &spec.name,
                    )
                    .map_err(|e| CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "canonicalize_for_identifiability: build \
                                         BlockJacobianAsRowOp for block '{}': {e}",
                            spec.name,
                        ),
                    })?;
                    Arc::new(row_op)
                }
                None => Arc::new(BlockJacobianAsRowOp::from_flat_design(
                    spec.design.clone(),
                    n_rows,
                    k,
                    &spec.name,
                )),
            };
            operators.push(op);
        }
        let row_hess = IdentityRowHessian::new(n_rows, k);
        let audit_result = audit_identifiability_channel_aware(specs, &operators, &row_hess)
            .map_err(|reason| CustomFamilyError::DimensionMismatch {
                reason: format!("pre-fit channel-aware identifiability audit failed: {reason}"),
            })?;

        log::info!(
            "[CANON] channel-aware audit: {} blocks, joint_rank={}/{} (flat audit NOT used)",
            specs.len(),
            audit_result
                .blocks
                .iter()
                .map(|b| b.effective_dim)
                .sum::<usize>(),
            specs.iter().map(|s| s.design.ncols()).sum::<usize>(),
        );

        // NOTE: the flat audit (`audit_identifiability`) must NOT be run on
        // multi-output blocks. It re-materialises each block's effective
        // Jacobian — which the channel-aware path has already proven is
        // `(n·k)`-row — and then the surrounding reduced-design / post-T
        // reconstruction conflates those `(n·k)`-row operators with the
        // `n`-row placeholder designs, broadcasting `(n·k, p)` into `(n, p)`
        // and panicking inside ndarray. The channel-aware `audit_result` is
        // the authoritative verdict (consumed below); no comparison audit is
        // sound or needed here.
        audit_result
    } else {
        let audit_result = audit_identifiability(specs).map_err(|reason| {
            CustomFamilyError::DimensionMismatch {
                reason: format!("pre-fit identifiability audit failed: {reason}"),
            }
        })?;
        log::debug!(
            "[CANON] flat audit: {} blocks, joint_rank={}",
            specs.len(),
            audit_result
                .blocks
                .iter()
                .map(|b| b.effective_dim)
                .sum::<usize>(),
        );
        audit_result
    };

    if audit.fatal {
        return Err(CustomFamilyError::IdentifiabilityFailure { audit });
    }

    // A block owns its effective geometry — and therefore its own raw-coordinate
    // identifiability bookkeeping — when it carries EITHER a `jacobian_callback`
    // (BMS marginal-slope, …: the family reconstructs the predictor from internal
    // full-width designs the block `design` only seeds) OR a `stacked_design`
    // (the survival location-scale / latent-survival time, threshold, and
    // log-sigma blocks, whose solver eta is the `3·n`-row `[exit; entry; deriv]·β`
    // operator and whose raw I-spline coordinates carry the family's warp-gauge
    // lift `β_raw = z·β_reduced + affine_shift` (#892), its leading-fixed-column
    // expansion, and its monotonicity constraint matrix `A·β ≥ b`). For both
    // kinds the audit's per-column DROP is unrepresentable: dropping a raw column
    // desynchronises the family's internal `z`/constraint/fixed-col layout (and
    // the prediction-time design rebuilt at raw width) from the reduced β,
    // tripping the family's own `eta length mismatch: got n, expected 3n`,
    // `time-warp design mismatch`, and `violates represented linear constraint`
    // shape checks (gam#1068). Keep every such block at its raw width with an
    // identity gauge and let the Tier-B joint-Newton Jeffreys/Firth term add
    // curvature on the weak directions WITHOUT any design surgery.
    // The veto is needed ONLY for the column that the audit would remove from a
    // block that *owns* its geometry: dropping a raw column from such a block
    // (a `jacobian_callback` block, or a `stacked_design` block carrying the
    // family's `3·n`-row z-lift / leading-fixed-column / monotonicity layout)
    // desynchronises that internal layout from the reduced β (#1068). A column
    // dropped from a PLAIN-DENSE block — e.g. the survival location-scale
    // threshold / log-σ blocks for a time-INVARIANT covariate, where
    // `stacked_design` is `None` and the design is an ordinary covariate matrix —
    // has no such hidden layout; the per-block selection-`T_i` reduction below
    // already column-reduces `design` (and any `stacked_design`) consistently, so
    // applying that drop is exactly the right move. Gating the whole pass on a
    // GLOBAL "any block owns geometry" flag was too broad: with three aliased
    // constants across the time/threshold/log_sigma blocks, the audit attributes
    // the two surplus-constant drops to the lower-priority PLAIN threshold and
    // log_sigma blocks, but the presence of the time block's `stacked_design`
    // vetoed BOTH — leaving the joint design rank-deficient by two and letting the
    // downstream solve collapse a genuine covariate direction (a time-invariant
    // covariate's coefficient pinned to exactly 0; gam#1110). So veto only when a
    // dropped column is attributed to a block that actually owns its geometry.
    // The veto now applies ONLY to `stacked_design` blocks (#933). A
    // `jacobian_callback`-only block can be column-reduced safely: the per-block
    // reduction loop below wraps its callback in `GaugeComposedJacobian` with the
    // block's selection-`T_i`, so the family's effective Jacobian emerges at the
    // REDUCED width by construction and its row-Hessian column-count assertions
    // hold — the family captures the reduced design instead of asserting raw
    // widths. A `stacked_design` block still carries the family's hidden
    // `3·n`-row z-lift / leading-fixed-column / monotonicity layout (#892, #1068)
    // that a raw-column drop would desynchronise from the reduced β, so those
    // remain width-preserving and defer curvature on weak directions to the
    // robust/Firth path.
    let owns_stacked_geometry = |name: &str| -> bool {
        specs
            .iter()
            .any(|spec| spec.name == name && spec.stacked_design.is_some())
    };
    let dropped_on_owned_block = audit
        .dropped_columns
        .iter()
        .any(|drop| owns_stacked_geometry(&drop.block));
    if dropped_on_owned_block {
        let raw_widths: Vec<usize> = specs.iter().map(|spec| spec.design.ncols()).collect();
        let dropped_summary = audit
            .dropped_columns
            .iter()
            .map(|drop| format!("{}[{}]", drop.block, drop.column))
            .collect::<Vec<_>>()
            .join(", ");
        log::info!(
            "[CANON] width-preserving family-owned geometry path: audit attributed \
             dropped columns [{dropped_summary}], at least one of which falls on a block \
             that owns its effective geometry via jacobian_callback or a multi-channel \
             stacked_design; keeping raw block widths and deferring curvature on the weak \
             directions to the robust/Firth path"
        );
        return Ok(CanonicalSpecs {
            reduced_specs: specs.to_vec(),
            gauge: Gauge::identity(&raw_widths),
            audit,
            used_channel_aware_audit: use_channel_aware,
        });
    }

    let mut per_block_transform: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    let mut reduced_specs: Vec<ParameterBlockSpec> = Vec::with_capacity(specs.len());

    for spec in specs.iter() {
        let p_raw = spec.design.ncols();
        let dropped_locals: Vec<usize> = audit
            .dropped_columns
            .iter()
            .filter(|drop| drop.block == spec.name)
            .map(|drop| drop.column)
            .collect();
        let mut dropped_sorted = dropped_locals.clone();
        dropped_sorted.sort_unstable();
        dropped_sorted.dedup();
        for &col in &dropped_sorted {
            if col >= p_raw {
                crate::bail_dim_custom!(
                    "canonicalize_for_identifiability: audit reported dropped column \
                         {col} for block '{}' which has only {} columns",
                    spec.name,
                    p_raw,
                );
            }
        }
        let kept: Vec<usize> = (0..p_raw)
            .filter(|c| dropped_sorted.binary_search(c).is_err())
            .collect();
        let r_block = kept.len();

        let mut t_i = Array2::<f64>::zeros((p_raw, r_block));
        for (col_out, &raw_col) in kept.iter().enumerate() {
            t_i[[raw_col, col_out]] = 1.0;
        }

        let reduced_design = if dropped_sorted.is_empty() {
            spec.design.clone()
        } else {
            build_reduced_design(&spec.design, &kept, &spec.name, &t_i)?
        };

        // Column-reduce the optional stacked solver operator alongside
        // `design` so the post-canonical β still indexes the same
        // surviving columns in both views.
        let reduced_stacked_design: Option<DesignMatrix> = match spec.stacked_design.as_ref() {
            Some(stacked) if !dropped_sorted.is_empty() => {
                Some(build_reduced_design(stacked, &kept, &spec.name, &t_i)?)
            }
            Some(stacked) => Some(stacked.clone()),
            None => None,
        };
        let reduced_stacked_offset = spec.stacked_offset.clone();

        let reduced_penalties: Vec<PenaltyMatrix> = spec
            .penalties
            .iter()
            .map(|p| pull_back_penalty(p, &kept))
            .collect();

        let reduced_initial_beta = match &spec.initial_beta {
            Some(beta_raw) => {
                if beta_raw.len() != p_raw {
                    crate::bail_dim_custom!(
                        "canonicalize_for_identifiability: block '{}' initial_beta \
                             length {} != design ncols {}",
                        spec.name,
                        beta_raw.len(),
                        p_raw,
                    );
                }
                let mut theta = Array1::<f64>::zeros(r_block);
                for (out_idx, &raw_col) in kept.iter().enumerate() {
                    theta[out_idx] = beta_raw[raw_col];
                }
                Some(theta)
            }
            None => None,
        };

        // Compose the gauge INTO the callback (#933): when columns are dropped
        // from a `jacobian_callback` block, the family must see the reduced
        // width. `GaugeComposedJacobian` post-multiplies the callback's raw-width
        // effective Jacobian by `T_i` (p_raw × r_block) so the effective
        // Jacobian emerges at `r_block` columns — the family captures the reduced
        // design and its row-Hessian column-count assertions hold by
        // construction, instead of asserting the raw width against a reduced
        // spec. With no drops `T_i` is the identity and the callback is forwarded
        // unchanged. (`stacked_design` blocks never reach this width-reducing
        // branch — they are vetoed above.)
        let reduced_jacobian_callback = match spec.jacobian_callback.as_ref() {
            Some(cb) if !dropped_sorted.is_empty() => Some(Arc::new(
                gam_problem::GaugeComposedJacobian::new(Arc::clone(cb), Arc::new(t_i.clone())),
            )
                as Arc<dyn BlockEffectiveJacobian>),
            other => other.cloned(),
        };

        reduced_specs.push(ParameterBlockSpec {
            name: spec.name.clone(),
            design: reduced_design,
            offset: spec.offset.clone(),
            penalties: reduced_penalties,
            // Pulled-back penalties may carry an enlarged structural
            // nullspace (a column dropped from a smooth's pure-span
            // basis adds that direction to the penalty kernel).
            // Falling back to eigenvalue-based rank detection in the
            // pseudo-logdet path is the safe choice when the
            // selection-T pullback changes the kernel structurally.
            nullspace_dims: Vec::new(),
            initial_log_lambdas: spec.initial_log_lambdas.clone(),
            initial_beta: reduced_initial_beta,
            gauge_priority: spec.gauge_priority,
            jacobian_callback: reduced_jacobian_callback,
            stacked_design: reduced_stacked_design,
            stacked_offset: reduced_stacked_offset,
        });
        per_block_transform.push(t_i);
    }

    // ── Post-T invariant check + MAP uniqueness check ────────────────────
    //
    // Materialise the joint post-T Jacobian J_can = J · T_full where
    // J is the (n*k × p_total) stacked Jacobian and T_full is block-diagonal
    // of the per-block T_i.  Assert rank(J_can) == rank(J_pre_T).
    //
    // After confirming the rank invariant, run the MAP uniqueness check:
    //   ker(J^T W J) ∩ ker(S) = {0}
    // where S = blockdiag of the reduced-spec joint penalty.  If any null
    // direction of J^T W J also lies in ker(S), the MAP is non-unique —
    // refuse with MapUniquenessFailure naming the dominant block.
    {
        let p_total_raw: usize = specs.iter().map(|s| s.design.ncols()).sum();
        let p_total_red: usize = per_block_transform.iter().map(|t| t.ncols()).sum();
        let k = if use_channel_aware { max_n_outputs } else { 1 };
        let nk = n_rows * k;

        // A `stacked_design` block owns its effective geometry through a
        // `k_s·n`-row `[entry; exit; deriv]·β` operator whose TRUE column span
        // lives across all channels — not in the `n`-row `design` that
        // `effective_jacobian_at` returns for it. The MAP-uniqueness check must
        // see that full span, or a stacked column collinear with another block's
        // covariate in the first `n` rows (e.g. a monotone time column vs an
        // `age` covariate that are affine-equal on the observation grid) is
        // falsely flagged as a flat null direction (gam#1197). Mirror the joint
        // audit's geometry exactly: size `j_pre` to the tallest stacked
        // operator, pack stacked blocks at their native `k_s·n` rows, and
        // replicate plain per-observation blocks across the stacked operator's
        // OBSERVATION bands (the bands carrying an ~constant intercept column;
        // the derivative band annihilates constants and is excluded), so a plain
        // block's intercept aligns with the stacked additive constant while a
        // genuine covariate stays distinguished by the derivative band.
        let stacked_rows = specs
            .iter()
            .filter_map(|s| s.stacked_design.as_ref().map(|d| d.nrows()))
            .max()
            .unwrap_or(0);
        let r_map = nk.max(stacked_rows);
        // Dense stacked designs (k_s·n × p_b) for the observation-band detection
        // and stacked packing.
        let stacked_dense: Vec<Option<Array2<f64>>> = specs
            .iter()
            .map(|s| {
                s.stacked_design.as_ref().and_then(|d| {
                    d.try_to_dense_arc("canonicalize_rank_check stacked")
                        .ok()
                        .map(|a| a.as_ref().clone())
                })
            })
            .collect();
        let k_bands = (r_map / n_rows).max(1);
        let observation_bands: Vec<usize> = if k_bands <= 1 {
            vec![0]
        } else if let Some(stacked_ref) = stacked_dense.iter().flatten().max_by_key(|d| d.nrows()) {
            let mut bands: Vec<usize> = (0..k_bands)
                .filter(|&b| {
                    let lo = b * n_rows;
                    let hi = ((b + 1) * n_rows).min(stacked_ref.nrows());
                    if hi <= lo {
                        return false;
                    }
                    // Observation band iff the INTERCEPT column (column 0, the
                    // additive constant) is ~constant non-zero there; a pure
                    // derivative band maps the intercept coefficient to ~0 and is
                    // excluded (see the matching audit.rs comment, gam#1197).
                    if stacked_ref.ncols() == 0 {
                        return false;
                    }
                    let first = stacked_ref[[lo, 0]];
                    first.abs() > 1e-12
                        && (lo..hi).all(|r| {
                            (stacked_ref[[r, 0]] - first).abs() <= 1e-9 * first.abs().max(1.0)
                        })
                })
                .collect();
            if bands.is_empty() {
                bands.push(0);
            }
            bands
        } else {
            vec![0]
        };

        // Build J_pre_T: (r_map, p_total_raw) by row-stacking per-block Jacobians.
        let mut j_pre = Array2::<f64>::zeros((r_map, p_total_raw));
        let mut col_off = 0usize;
        for (idx, spec) in specs.iter().enumerate() {
            let p_b = spec.design.ncols();
            // Stacked-geometry blocks: pack the full k_s·n-row stacked operator
            // directly (rows 0..stacked.nrows()), so their column span carries the
            // entry/exit/deriv channels the n-row flat Jacobian omits.
            if let Some(dense) = stacked_dense[idx].as_ref() {
                let br = dense.nrows().min(r_map);
                for i in 0..br {
                    for j in 0..p_b.min(dense.ncols()) {
                        j_pre[[i, col_off + j]] = dense[[i, j]];
                    }
                }
                col_off += p_b;
                continue;
            }
            let zeros = vec![0.0f64; p_b];
            let state = FamilyLinearizationState {
                beta: &zeros,
                family_scalars: None,
                channel_hessian: None,
                probit_frailty_scale: 1.0,
            };
            match spec.effective_jacobian_at("canonicalize_rank_check", &state) {
                Ok(j_b) => {
                    // j_b is channel-major (k_b·n_rows, p_b): row `r·n_rows + i`
                    // carries observation `i`'s channel-`r` row Jacobian. For a
                    // single-channel plain block (k_b == 1) replicate it across
                    // the observation bands so its intercept aligns with the
                    // stacked additive constant (gam#1197); a genuinely multi-
                    // channel block keeps its own channel-major rows.
                    let k_b = j_b.nrows() / n_rows;
                    if k_b <= 1 {
                        for &b in &observation_bands {
                            let base = b * n_rows;
                            for i in 0..n_rows {
                                let dst_row = base + i;
                                if dst_row >= r_map {
                                    break;
                                }
                                for j in 0..p_b {
                                    j_pre[[dst_row, col_off + j]] = j_b[[i, j]];
                                }
                            }
                        }
                    } else {
                        let r_max = k_b.min(k);
                        for r in 0..r_max {
                            let src_row_base = r * n_rows;
                            for i in 0..n_rows {
                                let dst_row = i * k + r;
                                let src_row = src_row_base + i;
                                for j in 0..p_b {
                                    j_pre[[dst_row, col_off + j]] = j_b[[src_row, j]];
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // Fall back: embed flat design across the observation bands.
                    if let Ok(flat) = spec
                        .design
                        .try_to_dense_arc("canonicalize_rank_check")
                        .map(|a| a.as_ref().clone())
                    {
                        for &b in &observation_bands {
                            let base = b * n_rows;
                            for i in 0..n_rows.min(flat.nrows()) {
                                let dst_row = base + i;
                                if dst_row >= r_map {
                                    break;
                                }
                                for j in 0..p_b.min(flat.ncols()) {
                                    j_pre[[dst_row, col_off + j]] = flat[[i, j]];
                                }
                            }
                        }
                    }
                }
            }
            col_off += p_b;
        }

        // ── Identity-T fast path (exact, data-independent) ──────────────────
        //
        // Each `T_i` is a column-SELECTION matrix built from the surviving
        // columns `kept`: when no column was dropped from block `i`, `kept`
        // is `0..p_i` and `T_i` is the `p_i × p_i` identity. When *every*
        // block kept all its columns AND no orthogonalisation ran, the joint
        // `T_full = blockdiag(T_i)` is the `p_total_raw × p_total_raw`
        // identity, hence `J_can = J_pre · I = J_pre` IDENTICALLY for any
        // data. The post-T rank invariant `rank(J_can) == rank(J_pre)` is then
        // satisfied by construction — `I` preserves rank for every matrix — so
        // the two ~O(nk·p²) RRQRs and the J_can materialisation are pure
        // redundant work. This is the biobank common case (clean full-rank
        // design, no aliases): at n=326k, p=85 the skipped J_can build + double
        // RRQR is the ~10s that dominated the canonicalise stage (#1110 perf).
        //
        // Correctness is INDEPENDENT of the row data: the skip is gated only on
        // the structural shape of T (every block square). When ANY block
        // dropped a column the transform genuinely reduces width and we fall
        // through to the full materialise-and-RRQR verification below.
        let t_is_identity = per_block_transform.iter().all(|t| t.nrows() == t.ncols());

        // On the identity path `J_can ≡ J_pre`, so the MAP uniqueness check
        // (which consumes the reduced-coordinate Jacobian) reads `j_pre`
        // directly; on the reducing path it reads the materialised `j_can`.
        let mut j_can_reduced: Option<Array2<f64>> = None;

        if t_is_identity {
            log::info!(
                "[CANON] post-T invariant: T=identity (all blocks full-width) — \
                 J_can≡J_pre, rank preserved by construction; skipping J_can \
                 materialise + double RRQR (p_raw={p_total_raw} p_red={p_total_red} k={k})",
            );
        } else {
            // Build J_can = J_pre · T_full where T_full = blockdiag(T_i).
            let mut j_can = Array2::<f64>::zeros((r_map, p_total_red));
            let mut raw_col_off = 0usize;
            let mut red_col_off = 0usize;
            for t_i in per_block_transform.iter() {
                let p_i = t_i.nrows();
                let r_i = t_i.ncols();
                if p_i > 0 && r_i > 0 {
                    // J_can[:, red_col_off .. red_col_off+r_i]
                    //   = J_pre[:, raw_col_off .. raw_col_off+p_i] · T_i
                    for row in 0..r_map {
                        for out_col in 0..r_i {
                            let mut acc = 0.0_f64;
                            for in_col in 0..p_i {
                                acc += j_pre[[row, raw_col_off + in_col]] * t_i[[in_col, out_col]];
                            }
                            j_can[[row, red_col_off + out_col]] = acc;
                        }
                    }
                }
                raw_col_off += p_i;
                red_col_off += r_i;
            }

            // ── Post-T rank invariant, ANCHORED to the audit certificate ──────
            //
            // What the invariant must certify: the column-selection `T` did not
            // drop a direction the audit deemed identifiable. The audit already
            // produced its verdict — `audit_kept_rank`, the joint rank it kept
            // (= sum of per-block `effective_dim`). By construction every dropped
            // column comes from `audit.dropped_columns`, so the surviving columns
            // ARE exactly the audit's kept basis and `p_total_red ==
            // audit_kept_rank`. The faithful-`T` certificate is therefore simply
            // that the REDUCED design `J_can` is FULL COLUMN RANK under the same
            // convention that decided the drops:
            //
            //     rank_drop_convention(J_can) == p_total_red (== audit_kept_rank)
            //
            // A defective `T` that drops a genuinely identifiable direction, or
            // leaves a redundant one, makes `J_can` rank-DEFICIENT and is caught.
            //
            // Why this replaces the earlier `rank(J_pre) == rank(J_can)` compare:
            // the previous gate ranked the FULL-WIDTH `J_pre` (which still carries
            // the about-to-be-dropped columns) and compared it to the reduced
            // `J_can`. That only holds if the drop convention agrees the dropped
            // columns are redundant — but the convention used in this gate is a
            // PROXY for the audit's true drop rule, not the rule itself:
            //
            //   • CHANNEL-AWARE path — the audit drops via a SEQUENTIAL, per-block
            //     RESIDUALISED σ²-Gram eigenspace walk (`keep_positive_eigenspace`
            //     on each block's residual Gram after projecting out the cumulative
            //     higher-priority anchor, in a structural + curvature two-pass).
            //     `audit_convention_rank` instead does ONE eigendecomposition of
            //     the joint `JᵀJ`. A direction that is residual-WEAK (absorbed by a
            //     higher-priority anchor) but still carries mass in the joint Gram
            //     is dropped by the walk yet COUNTED by the single joint
            //     eigendecomposition — so `audit_convention_rank(J_pre)`
            //     over-counted and a faithful `T` tripped a false `rank(J_pre) >
            //     rank(J_can)` (GAMLSS-PS wine_gamair/wine_price_vs_temp, gam#1391).
            //   • FLAT path — the audit drops via the PENALTY-AUGMENTED,
            //     gauge-priority-tiered rank of `[X_joint; S_blockdiag]`; a bare
            //     RRQR on the un-augmented `J_pre` over-counted penalty-null shared
            //     directions (gam#1388/#1391).
            //
            // Anchoring to `J_can`'s full-rank certificate sidesteps `J_pre`'s
            // over-count entirely while keeping the defective-`T` guarantee, and it
            // uses ONE consistent convention applied to ONE matrix (`J_can`), so
            // there is no cross-convention comparison left to disagree.
            //
            // `nk_scale` mirrors the audit's tolerance size term: r_map is the
            // (n·k)-stacked row count.
            let nk_scale = r_map;
            // Per-block descriptors for the flat path's penalty-augmented,
            // priority-tiered rank of `J_can` — built from the REDUCED specs so
            // each reduced design is augmented with its pulled-back structural
            // penalty (VᵀSV) and the gauge priority that drove the audit's drops.
            let flat_blocks_can: Vec<FlatRankBlock> = reduced_specs
                .iter()
                .map(|s| FlatRankBlock {
                    width: s.design.ncols(),
                    structural_penalty: block_structural_penalty_dense(s),
                    priority: s.gauge_priority,
                })
                .collect();
            let rank_j_can = if use_channel_aware {
                audit_convention_rank(&j_can, nk_scale, &flat_blocks_can)
            } else {
                flat_audit_convention_rank(&j_can, &flat_blocks_can)
            };

            // Same convention applied to the FULL pre-reduction design `J_pre`,
            // built from the RAW specs so each raw block is augmented with its
            // own (un-pulled-back) structural penalty. This is the rank ceiling a
            // faithful column-selection `T` can possibly leave behind: `T` removes
            // exactly the audit-demoted columns, so the surviving columns are as
            // independent as the full design ever was — no more, no less.
            //
            // On an UNDER-DETERMINED joint (`p_total_raw > r_map`, the
            // cirrhosis/heart_failure `p_joint > n` regime of gam#1388, where every
            // categorical level expands into its own column) the unaugmented data
            // Gram of `J_pre` is row-capped at `r_map` independent directions, so
            // `rank_j_pre < p_total_raw`. The same cap then limits `J_can`: it
            // CANNOT reach `p_total_red` independent columns when the rows (and the
            // penalty anchors) simply do not span that many directions — that is
            // legitimate penalty-rested under-determination, NOT a defective `T`.
            // Certifying against `p_total_red` directly therefore tripped a FALSE
            // "post-T rank invariant violated" on exactly these benchmarks.
            let flat_blocks_pre: Vec<FlatRankBlock> = specs
                .iter()
                .map(|s| FlatRankBlock {
                    width: s.design.ncols(),
                    structural_penalty: block_structural_penalty_dense(s),
                    priority: s.gauge_priority,
                })
                .collect();
            let rank_j_pre = if use_channel_aware {
                audit_convention_rank(&j_pre, nk_scale, &flat_blocks_pre)
            } else {
                flat_audit_convention_rank(&j_pre, &flat_blocks_pre)
            };
            // The achievable target: a faithful `T` leaves `J_can` at the FULL
            // design's rank when the design is over-determined (`rank_j_pre ==
            // p_total_raw ⇒ target == p_total_red`, the original strict invariant),
            // and at the row-/penalty-limited rank when it is under-determined. A
            // defective `T` that drops a direction the audit kept makes `J_can`
            // rank-deficient BELOW this target and is still caught.
            let rank_target = rank_j_pre.min(p_total_red);

            // Threaded certificate from the audit itself: the per-block
            // `effective_dim` sums to the rank the drop convention kept — the
            // audit's own joint_rank verdict, which equals `p_total_red` for a
            // column-selection `T`.
            let audit_kept_rank: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();

            log::info!(
                "[CANON] post-T invariant ({} convention): \
                 rank(J_can)={rank_j_can} rank(J_pre)={rank_j_pre} \
                 rank_target={rank_target} p_red={p_total_red} \
                 audit_kept_rank={audit_kept_rank} \
                 (p_raw={p_total_raw} k={k})",
                if use_channel_aware {
                    "σ²-Gram"
                } else {
                    "σ-space RRQR (penalty-augmented, priority-tiered)"
                },
            );

            // The faithful-`T` certificate: `J_can` retains every independent
            // direction the FULL design carried into the kept columns —
            // `rank(J_can) == min(rank(J_pre), p_total_red)`. When the design is
            // over-determined `rank(J_pre) == p_total_raw ≥ p_total_red`, so the
            // target collapses to `p_total_red` and this is the original strict
            // full-column-rank invariant. When it is under-determined the target is
            // the row-/penalty-limited `rank(J_pre)`, so legitimate `p_joint > n`
            // joints pass. A rank-deficient `J_can` BELOW the target still means `T`
            // dropped an identifiable direction (or left a redundant one) — a bug
            // in `T` construction, which remains caught.
            if rank_j_can != rank_target {
                let block_shapes: Vec<String> = per_block_transform
                    .iter()
                    .zip(specs.iter())
                    .map(|(t, s)| format!("{}:({},{})", s.name, t.nrows(), t.ncols()))
                    .collect();
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "canonicalize_for_identifiability: post-T rank invariant violated — \
                         under the drop-deciding {} convention the reduced design J_can is \
                         rank-deficient: rank(J_can)={rank_j_can} but rank_target=\
                         min(rank(J_pre)={rank_j_pre}, p_red={p_total_red})={rank_target} \
                         (audit_kept_rank={audit_kept_rank}, p_raw={p_total_raw}, k={k}); the \
                         column-selection T dropped a direction the FULL design carried into \
                         the kept columns — a bug in T construction; per-block T shapes: [{}]",
                        if use_channel_aware {
                            "σ²-Gram"
                        } else {
                            "σ-space RRQR (penalty-augmented, priority-tiered)"
                        },
                        block_shapes.join(", "),
                    ),
                });
            }
            j_can_reduced = Some(j_can);
        }

        // ── MAP uniqueness check ──────────────────────────────────────────
        //
        // Build the joint penalty S = blockdiag(sum_of_reduced_penalties_block_i)
        // in the reduced parameter space (p_total_red × p_total_red).
        // Each block's total penalty is the sum of its per-lambda penalty
        // matrices (all with equal weight 1.0 — the uniqueness condition is
        // independent of the specific λ values since we only need to know
        // whether ANY penalty covers the direction, not the magnitude).
        if p_total_red > 0 {
            let mut s_joint = Array2::<f64>::zeros((p_total_red, p_total_red));
            let mut red_off = 0usize;
            for spec in reduced_specs.iter() {
                let r_i = spec.design.ncols();
                for pen in spec.penalties.iter() {
                    let s_dense = pen.as_dense_cow();
                    // s_dense is (r_i, r_i).  Add it into the diagonal block.
                    if s_dense.nrows() == r_i && s_dense.ncols() == r_i {
                        for ii in 0..r_i {
                            for jj in 0..r_i {
                                s_joint[[red_off + ii, red_off + jj]] += s_dense[[ii, jj]];
                            }
                        }
                    }
                }
                red_off += r_i;
            }

            // Build col_offsets for the reduced specs.
            let mut red_col_offsets: Vec<usize> = Vec::with_capacity(reduced_specs.len() + 1);
            red_col_offsets.push(0);
            for spec in reduced_specs.iter() {
                let prev = *red_col_offsets.last().unwrap();
                red_col_offsets.push(prev + spec.design.ncols());
            }

            // The MAP uniqueness check operates on the flat (n_rows, p_total_red)
            // view of J_can.  For multi-channel families (k > 1), the channel
            // stacking increases the effective row count but the penalty still
            // lives in p_total_red dimensions.  We use the (nk, p_total_red)
            // J_can directly: the additional channel rows only help — if J^T W J
            // (with J being the full nk-row matrix) already has a non-trivial
            // null space, those extra rows could only shrink it relative to the
            // flat view.  Using the full J_can gives the tightest (most
            // conservative) null-space detection.
            // When `T = identity` (the fast path above did not materialise
            // `j_can`) the reduced Jacobian equals `j_pre` exactly, so the
            // check reads `j_pre`; otherwise it reads the reduced `j_can`.
            let j_for_map = j_can_reduced.as_ref().unwrap_or(&j_pre);
            crate::audit::check_map_uniqueness(
                j_for_map,
                &[],
                &s_joint,
                &reduced_specs,
                &red_col_offsets,
            )
            .map_err(|error| {
                log::warn!("[CANON] MAP uniqueness check failed: {}", error.message,);
                CustomFamilyError::MapUniquenessFailure { error }
            })?;

            log::debug!(
                "[CANON] MAP uniqueness check passed \
                 (p_red={p_total_red} penalty_blocks={})",
                reduced_specs
                    .iter()
                    .map(|s| s.penalties.len())
                    .sum::<usize>(),
            );
        }
    }

    Ok(CanonicalSpecs {
        reduced_specs,
        gauge: Gauge::from_block_transforms(&per_block_transform),
        audit,
        used_channel_aware_audit: use_channel_aware,
    })
}

/// Flag-gated exact orthogonalisation of structural confounds across blocks.
///
/// Returns `Ok(Some(canon))` when an overlap was found and removed by exact
/// W-metric reparameterisation; `Ok(None)` when orthogonalisation is not
/// applicable (multi-channel families, sparse/operator-backed designs, or no
/// cross-block overlap detected) so the caller falls through to the unmodified
/// audit gate. Never returns a *worse* result than today: a clean design
/// yields `None` (byte-identical fall-through), and an unrepresentable
/// structural condition also yields `None`.
fn try_orthogonalize_blocks(
    specs: &[ParameterBlockSpec],
) -> Result<Option<CanonicalSpecs>, CustomFamilyError> {
    if specs.len() < 2 {
        return Ok(None);
    }
    // Families whose blocks carry a `jacobian_callback` (BMS marginal-slope,
    // survival LS, …) own their effective geometry: the family reconstructs the
    // additive predictor from internal full-width designs (e.g. BMS reads its
    // own `marginal_design`/`logslope_design` per row), and the block `design`
    // here is only the *raw* basis the callback consumes. A per-block reparam
    // `X_b · V_b` that drops the callback does NOT change what the family
    // computes, but it shrinks the block coefficient width below the family's
    // internal design width — leaving the inner solve's reduced β (e.g. 8) out
    // of sync with the family's full design (e.g. 12) and tripping the family's
    // own shape validation. Such families are robustified by the Tier-B
    // joint-Newton Jeffreys/Firth term, which adds curvature on the
    // under-identified span WITHOUT any design surgery and keeps every block β at
    // full width. Defer here so they take that path. (Single-channel, plain-design
    // blocks — `jacobian_callback: None` — are reparam'd here; a clean design with
    // no overlap to remove falls through to the audit gate byte-identically.)
    let family_owned_geometry = specs.iter().any(|s| s.jacobian_callback.is_some());
    if family_owned_geometry {
        return Ok(None);
    }

    // Multi-channel eta blocks (`stacked_design = Some(_)`, e.g. the latent-
    // survival / survival location-scale time-transform block whose solver eta
    // is `[entry; exit; deriv] · β`, a `3·n`-row operator) cannot be expressed
    // by the single-channel `X_b · V_b` reparam this path builds: that reparam
    // only rewrites the n-row `design`, and the orthogonalised-spec builder
    // below deliberately sets `stacked_design = None`. Dropping the stacked
    // operator silently collapses the family's eta from `3·n` rows to `n`,
    // which trips the family's own `eta length mismatch: got n, expected 3n`
    // shape check (gam#1068). Defer to the audit gate so the multi-channel
    // operator survives untouched; the audit's column-selection reduction
    // (`canonicalize_for_identifiability_inner`) does pull `stacked_design`
    // through correctly.
    let multi_channel_geometry = specs.iter().any(|s| s.stacked_design.is_some());
    if multi_channel_geometry {
        return Ok(None);
    }

    // Densify every block design. Any non-densifiable (large/lazy operator)
    // block makes the per-block reparam non-representable here → defer.
    let n_rows = specs[0].design.nrows();
    let mut block_designs: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    for spec in specs.iter() {
        if spec.design.nrows() != n_rows {
            return Ok(None);
        }
        let dense = match spec
            .design
            .try_to_dense_arc("orthogonalize_design_blocks densify")
        {
            Ok(arc) => arc.as_ref().clone(),
            Err(_) => return Ok(None),
        };
        block_designs.push(dense);
    }

    // Pilot W-metric: the released structural audit runs in the unweighted
    // (Euclidean) row metric, and structural rank-overlap removal is exact in
    // that metric. Use uniform weights so the reparam matches the audit's
    // geometry; the family-curvature W-metric refinement belongs to the
    // Tier-B coupled path.
    let weight = vec![1.0_f64; n_rows];
    let priority: Vec<u32> = specs.iter().map(|s| s.gauge_priority as u32).collect();

    let ortho = orthogonalize_design_blocks(&block_designs, &priority, &weight).map_err(|e| {
        CustomFamilyError::DimensionMismatch {
            reason: format!("orthogonalize_design_blocks failed: {e}"),
        }
    })?;

    // No overlap removed ⇒ nothing to do; fall through to the standard gate so
    // behaviour is byte-identical to today on clean / square-rotation designs.
    if ortho.dropped.is_empty() {
        return Ok(None);
    }

    // Equal-priority gauge-ambiguity guard. Orthogonalisation removes a block's
    // overlap by residualising it against the cumulative anchor of all
    // higher-or-equal-priority blocks already visited (descending-priority,
    // stable-on-ties order — identical to `orthogonalize_design_blocks`). When a
    // direction is absorbed *into an equal-priority anchor block* there is NO
    // gauge ordering to decide which block loses the shared column: the inner
    // KKT system is structurally rank-deficient regardless of penalty, exactly
    // the contract the flat audit gate encodes (`all_priorities_equal` forces
    // `gauge_resolves_rank_deficiency = false`, so the alias is fatal). Reducing
    // here would silently drop the later block's column instead of refusing.
    // Defer to the audit gate, which raises `IdentifiabilityFailure`, whenever
    // an absorbed block shares its priority with any earlier-visited block.
    let mut visit_order: Vec<usize> = (0..specs.len()).collect();
    visit_order.sort_by(|&a, &b| priority[b].cmp(&priority[a]));
    let visit_rank: Vec<usize> = {
        let mut rank = vec![0usize; specs.len()];
        for (r, &b) in visit_order.iter().enumerate() {
            rank[b] = r;
        }
        rank
    };
    for annotation in ortho
        .direction_annotations
        .iter()
        .filter(|annotation| annotation.absorbed_width > 0)
    {
        let absorbed = annotation.block_idx;
        let equal_priority_anchor_exists = (0..specs.len()).any(|other| {
            other != absorbed
                && visit_rank[other] < visit_rank[absorbed]
                && priority[other] == priority[absorbed]
        });
        if equal_priority_anchor_exists {
            log::info!(
                "[CANON] orthogonalisation declined: block {} (priority {}) was absorbed into an \
                 equal-priority anchor — exact alias has no gauge ordering; deferring to the fatal \
                 audit gate instead of arbitrarily dropping the later block's column",
                absorbed,
                priority[absorbed],
            );
            return Ok(None);
        }
    }

    for annotation in ortho
        .direction_annotations
        .iter()
        .filter(|annotation| annotation.absorbed_width > 0)
    {
        log::info!(
            "[IDENT] structural direction annotation: block={} raw_width={} kept_width={} absorbed_width={} kind={:?}",
            annotation.block_idx,
            annotation.raw_width,
            annotation.kept_width,
            annotation.absorbed_width,
            annotation.kind,
        );
    }

    // Build orthogonalised specs: design ← X_b · V_b (via
    // CoefficientTransformOperator), penalties ← V_bᵀ S V_b, initial_beta ←
    // V_bᵀ β₀ (least-squares image; V_b has orthonormal columns so V_bᵀ is the
    // pseudo-inverse), and remember V_b for the round-trip composition.
    let mut ortho_specs: Vec<ParameterBlockSpec> = Vec::with_capacity(specs.len());
    for (spec, v_b) in specs.iter().zip(ortho.block_transforms.iter()) {
        let p_b = spec.design.ncols();
        if v_b.nrows() != p_b {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "orthogonalize: block '{}' transform has {} rows but design has {p_b} columns",
                    spec.name,
                    v_b.nrows(),
                ),
            });
        }
        let inner_dense = match &spec.design {
            DesignMatrix::Dense(d) => d.clone(),
            DesignMatrix::Sparse(_) => {
                let dense = spec
                    .design
                    .try_to_dense_arc("orthogonalize reduced-design densify")
                    .map_err(|reason| CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "orthogonalize: densify block '{}' failed: {reason}",
                            spec.name,
                        ),
                    })?;
                DenseDesignMatrix::from(dense)
            }
        };
        let op = CoefficientTransformOperator::new(inner_dense, v_b.clone()).map_err(|reason| {
            CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "orthogonalize: build CoefficientTransformOperator for block '{}': {reason}",
                    spec.name,
                ),
            }
        })?;
        let reduced_design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op)));

        let reduced_penalties: Vec<PenaltyMatrix> = spec
            .penalties
            .iter()
            .map(|p| pull_back_penalty_dense(p, v_b))
            .collect();

        let reduced_initial_beta = match &spec.initial_beta {
            Some(beta_raw) => {
                if beta_raw.len() != p_b {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "orthogonalize: block '{}' initial_beta length {} != design ncols {p_b}",
                            spec.name,
                            beta_raw.len(),
                        ),
                    });
                }
                Some(v_b.t().dot(beta_raw))
            }
            None => None,
        };

        // The orthogonalised design no longer matches the raw-width
        // jacobian_callback / stacked_design; single-channel blocks here carry
        // neither (multi-channel was deferred above), so drop them defensively.
        ortho_specs.push(ParameterBlockSpec {
            name: spec.name.clone(),
            design: reduced_design,
            offset: spec.offset.clone(),
            penalties: reduced_penalties,
            nullspace_dims: Vec::new(),
            initial_log_lambdas: spec.initial_log_lambdas.clone(),
            initial_beta: reduced_initial_beta,
            gauge_priority: spec.gauge_priority,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        });
    }

    // Run the standard (non-orthogonalising) audit + post-T invariant + MAP
    // uniqueness checks on the orthogonalised specs. With the overlap removed,
    // this should produce a clean (identity-T) verdict; if a *residual* rank
    // deficiency survives orthogonalisation, the fail-closed gate still
    // refuses with an actionable diagnostic.
    let inner = canonicalize_for_identifiability_inner(&ortho_specs, false)?;

    // Compose the round-trip transform: β_raw = V_b · (T_inner · θ).
    // `inner.gauge.block_transform(b)` is T_inner (selection/identity from
    // the audit gate); the full raw lift is `V_b · T_inner`.
    let mut composed_transform: Vec<Array2<f64>> = Vec::with_capacity(specs.len());
    for (b, v_b) in ortho.block_transforms.iter().enumerate() {
        let t_inner = inner.gauge.block_transform(b);
        if v_b.ncols() != t_inner.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "orthogonalize: transform composition shape mismatch — V_b is {:?}, \
                     T_inner is {:?}",
                    v_b.dim(),
                    t_inner.dim(),
                ),
            });
        }
        composed_transform.push(v_b.dot(&t_inner));
    }

    log::info!(
        "[CANON] orthogonalisation applied: {} block(s) shed overlap directions {:?}; \
         p_raw={} → p_reduced={}",
        ortho.dropped.len(),
        ortho.dropped,
        specs.iter().map(|s| s.design.ncols()).sum::<usize>(),
        composed_transform.iter().map(|t| t.ncols()).sum::<usize>(),
    );

    Ok(Some(CanonicalSpecs {
        reduced_specs: inner.reduced_specs,
        gauge: Gauge::from_block_transforms(&composed_transform),
        audit: inner.audit,
        used_channel_aware_audit: inner.used_channel_aware_audit,
    }))
}

/// Pull a penalty back through a dense reparam `V` as `Vᵀ S V`, preserving the
/// precision label and fixed-λ pin. Companion to [`pull_back_penalty`] (which
/// pulls back through a column *selection*); this handles the dense
/// orthogonalisation transform.
fn pull_back_penalty_dense(penalty: &PenaltyMatrix, v: &Array2<f64>) -> PenaltyMatrix {
    let label = penalty.precision_label().map(|s| s.to_string());
    let fixed_log_lambda = penalty.fixed_log_lambda();
    let dense = penalty.as_dense_cow();
    // Vᵀ S V  (r × r).
    let s_v = dense.dot(v);
    let reduced = v.t().dot(&s_v);
    let mut base = PenaltyMatrix::Dense(reduced);
    if let Some(lbl) = label {
        base = base.with_precision_label(lbl);
    }
    if let Some(value) = fixed_log_lambda {
        base = base.with_fixed_log_lambda(value);
    }
    base
}

fn build_reduced_design(
    raw: &DesignMatrix,
    kept: &[usize],
    block_name: &str,
    t_i: &Array2<f64>,
) -> Result<DesignMatrix, CustomFamilyError> {
    let inner_dense = match raw {
        DesignMatrix::Dense(d) => d.clone(),
        DesignMatrix::Sparse(_) => {
            let dense = raw
                .try_to_dense_by_chunks(&format!(
                    "canonicalize_for_identifiability sparse->dense block '{block_name}'"
                ))
                .map_err(|reason| CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "canonicalize_for_identifiability: densify sparse block '{block_name}' \
                         failed: {reason}"
                    ),
                })?;
            DenseDesignMatrix::from(dense)
        }
    };
    // Hot path: when the inner is already a materialised dense Array2,
    // slice the kept columns directly. This avoids carrying the full
    // raw-width inner through every PIRLS iteration when many columns
    // were dropped.
    if let Some(arr) = inner_dense.as_dense_ref() {
        let reduced =
            Array2::<f64>::from_shape_fn((arr.nrows(), kept.len()), |(i, j)| arr[[i, kept[j]]]);
        return Ok(DesignMatrix::Dense(DenseDesignMatrix::from(reduced)));
    }
    // Operator-backed inner (Lazy): preserve the operator structure
    // by wrapping with CoefficientTransformOperator on the selection T.
    let op = CoefficientTransformOperator::new(inner_dense, t_i.clone()).map_err(|reason| {
        CustomFamilyError::DimensionMismatch {
            reason: format!(
                "canonicalize_for_identifiability: build CoefficientTransformOperator \
                 for block '{block_name}': {reason}"
            ),
        }
    })?;
    Ok(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op))))
}

fn pull_back_penalty(penalty: &PenaltyMatrix, kept: &[usize]) -> PenaltyMatrix {
    let label = penalty.precision_label().map(|s| s.to_string());
    let fixed_log_lambda = penalty.fixed_log_lambda();
    let dense = penalty.as_dense_cow();
    let reduced =
        Array2::<f64>::from_shape_fn((kept.len(), kept.len()), |(i, j)| dense[[kept[i], kept[j]]]);
    let mut base = PenaltyMatrix::Dense(reduced);
    if let Some(lbl) = label {
        base = base.with_precision_label(lbl);
    }
    if let Some(value) = fixed_log_lambda {
        base = base.with_fixed_log_lambda(value);
    }
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::DenseDesignMatrix;
    use gam_problem::AdditiveBlockJacobian;

    use ndarray::Array2;

    fn linspace(n: usize) -> ndarray::Array1<f64> {
        if n <= 1 {
            return ndarray::Array1::<f64>::zeros(n.max(1));
        }
        ndarray::Array1::linspace(-1.0, 1.0, n)
    }

    #[test]
    fn canonical_clean_specs_identity_transform() {
        let n = 32;
        let x = linspace(n);
        let mut p = Array2::<f64>::zeros((n, 2));
        let mut s = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            p[[i, 0]] = 1.0;
            p[[i, 1]] = x[i];
            s[[i, 0]] = x[i] * x[i];
            s[[i, 1]] = x[i] * x[i] * x[i];
        }
        let specs = [spec_from_dense("p", p), spec_from_dense("s", s)];
        let canon = canonicalize_for_identifiability(&specs).expect("clean canonical must succeed");
        assert_eq!(canon.reduced_specs.len(), 2);
        assert_eq!(canon.gauge.block_transform(0).dim(), (2, 2));
        assert_eq!(canon.gauge.block_transform(1).dim(), (2, 2));
        let theta = vec![Array1::from(vec![0.5, -0.25]), Array1::from(vec![1.0, 2.0])];
        let raw = canon.gauge.lift_block_betas(&theta);
        assert_eq!(raw[0].as_slice().unwrap(), &[0.5, -0.25]);
        assert_eq!(raw[1].as_slice().unwrap(), &[1.0, 2.0]);
    }

    /// Fail-closed contract: an aliased smooth-constant ~ intercept
    /// configuration must produce `CustomFamilyError::IdentifiabilityFailure`,
    /// not a silently-reduced spec list. Substituting a column-reduced
    /// `ParameterBlockSpec` under a family that captures raw-width
    /// designs (the current `CustomFamily` contract) panics inside
    /// `DesignMatrix::syr_row_into_view`; the safe behaviour is to
    /// refuse the fit with an actionable diagnostic in milliseconds.
    #[test]
    fn canonical_refuses_aliased_smooth_constant_with_intercept() {
        let n = 64;
        let x = linspace(n);
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
        let err = canonicalize_for_identifiability(&specs)
            .expect_err("aliased smooth-constant + intercept must refuse, not reduce");
        match err {
            CustomFamilyError::IdentifiabilityFailure { audit } => {
                assert!(
                    audit.fatal,
                    "audit attached to IdentifiabilityFailure must be fatal; got {}",
                    audit.summary,
                );
                assert!(
                    audit.summary.contains("intercept")
                        && audit.summary.contains("smooth_with_const"),
                    "refusal summary must name both offending blocks; got {:?}",
                    audit.summary,
                );
            }
            other => panic!("expected IdentifiabilityFailure, got {other:?}"),
        }
    }

    /// Five-block large-scale aliasing repro. Each block carries an
    /// intercept-like constant column; gauge_priority is set per the
    /// survival marginal-slope ownership policy (time=200 > marginal=150
    /// > logslope=120 > score_warp=80 > link_dev=60). The joint design
    /// has a 4-D null space among the five constants.
    ///
    /// Under the gauge-aware contract the canonicalisation must SUCCEED
    /// (not refuse) because the distinct `gauge_priority` values provide
    /// an unambiguous ordering for which columns to drop. The
    /// priority-ordered RRQR demotes the lowest-priority participants;
    /// the canonical-gauge pipeline applies the column-selection matrices
    /// and proceeds with reduced specs.  `time_surface` (highest priority)
    /// must retain all its columns; every attributed drop must belong to
    /// one of the four lower-priority blocks.
    #[test]
    fn canonical_five_block_gauge_ownership_succeeds_with_attribution() {
        let n = 96;
        let x = linspace(n);
        // Each block carries `ones(n)` in column 0 (the shared-constant
        // direction) plus per-block-unique content in columns 1..p.
        // The joint design therefore has a 4-D null space among the
        // five constant columns; priority decides which four are
        // dropped. We expect canonicalize to drop one constant from
        // each of (marginal, logslope, score_warp, link_dev) and keep
        // the time block's constant.
        let mut time = Array2::<f64>::zeros((n, 3));
        let mut marginal = Array2::<f64>::zeros((n, 3));
        let mut logslope = Array2::<f64>::zeros((n, 3));
        let mut score_warp = Array2::<f64>::zeros((n, 2));
        let mut link_dev = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            time[[i, 0]] = 1.0;
            time[[i, 1]] = x[i];
            time[[i, 2]] = x[i] * x[i] * x[i];
            marginal[[i, 0]] = 1.0;
            marginal[[i, 1]] = x[i] * x[i];
            marginal[[i, 2]] = x[i].sin();
            logslope[[i, 0]] = 1.0;
            logslope[[i, 1]] = (3.0 * x[i]).sin();
            logslope[[i, 2]] = (6.0 * x[i]).sin();
            score_warp[[i, 0]] = 1.0;
            score_warp[[i, 1]] = (5.0 * x[i]).cos();
            link_dev[[i, 0]] = 1.0;
            link_dev[[i, 1]] = (7.0 * x[i]).tanh();
        }
        let mut t_spec = spec_from_dense("time_surface", time);
        t_spec.gauge_priority = 200;
        let mut m_spec = spec_from_dense("marginal_surface", marginal);
        m_spec.gauge_priority = 150;
        let mut g_spec = spec_from_dense("logslope_surface", logslope);
        g_spec.gauge_priority = 120;
        let mut w_spec = spec_from_dense("score_warp_dev", score_warp);
        w_spec.gauge_priority = 80;
        let mut l_spec = spec_from_dense("link_dev", link_dev);
        l_spec.gauge_priority = 60;
        let specs = [t_spec, m_spec, g_spec, w_spec, l_spec];

        // With distinct gauge_priority values, the audit recognises that
        // the rank deficiency is gauge-resolvable and returns Ok (non-fatal).
        // The canonical-gauge pipeline proceeds with the column reductions.
        let canon = canonicalize_for_identifiability(&specs).expect(
            "five-block aliased joint with distinct gauge_priority must succeed (gauge-resolved)",
        );

        // The audit stored in canon.audit must be non-fatal.
        assert!(
            !canon.audit.fatal,
            "audit must be non-fatal when gauge_priority is non-trivial and \
             all drops are attributed to lower-priority blocks; got {}",
            canon.audit.summary,
        );

        // Raw p_total = 3+3+3+2+2 = 13; expected rank = 13 − 4 = 9.
        // Audit's per-block effective_dim sum equals joint_rank.
        let total_kept: usize = canon.audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept,
            9,
            "expected joint rank = 13 − 4 = 9 reported by audit; got {total_kept} \
             (per-block effective_dim {:?})",
            canon
                .audit
                .blocks
                .iter()
                .map(|b| (b.block_name.clone(), b.effective_dim))
                .collect::<Vec<_>>(),
        );

        // Gauge-priority attribution: with priority-descending column ordering,
        // the time_surface block (highest priority = 200) must NOT appear among
        // the attributed drops — every attributed drop belongs to one of the
        // four lower-priority blocks.
        for drop in &canon.audit.dropped_columns {
            assert_ne!(
                drop.block, "time_surface",
                "highest-priority block must never be the attributed drop \
                 origin under priority-aware RRQR; got drop on time_surface \
                 ({drop:?})",
            );
        }

        // The reduced specs must have 9 total columns (rank = 9).
        let reduced_total: usize = canon.reduced_specs.iter().map(|s| s.design.ncols()).sum();
        assert_eq!(
            reduced_total, 9,
            "reduced specs must have joint rank = 9 total columns; got {reduced_total}",
        );

        // time_surface retains all 3 of its columns (highest priority, never dropped).
        assert_eq!(
            canon.reduced_specs[0].design.ncols(),
            3,
            "time_surface must retain all 3 columns after gauge canonicalisation",
        );
    }

    /// #1220: the post-T rank invariant must rank `J` and `J_can` with the
    /// SAME threshold/operator the audit used to decide the drops, so a column
    /// the audit (σ²-Gram eigenvalue convention) demotes on near-separable data
    /// does not trip the invariant (which previously re-ranked `J` with a
    /// σ-space RRQR whose cutoff is ~6 orders of magnitude lower, counting the
    /// demoted direction and reporting `rank(J) > rank(J_can)`).
    ///
    /// Fixture mirrors the multinomial Firth/Jeffreys penguin geometry: two
    /// softmax channel blocks share an identical design, each routed through an
    /// `AdditiveBlockJacobian` (`n_family_outputs = 2`, `own_output = a`) so
    /// canonicalisation runs the CHANNEL-AWARE audit (block-diagonal
    /// `blkdiag(X, X)`, not a false `[X | X]` alias). `X` carries a
    /// near-separable direction: a column whose relative singular value sits in
    /// the gap between the σ-space RRQR floor (~ε) and the σ²-Gram cutoff
    /// (~√ε) — full-rank under RRQR, gauge under the audit. The invariant must
    /// HOLD (no `DimensionMismatch` rank-invariant panic) for either audit
    /// verdict.
    #[test]
    fn canonical_multinomial_near_separable_post_t_rank_invariant_holds() {
        let n = 120;
        let x = linspace(n);
        // Shared per-channel design X (n × 4): three well-conditioned smooth
        // basis columns plus one NEAR-SEPARABLE column. The weak column is
        // `c3 = c1 + ε_rel · w`, where `w` is an independent oscillation and
        // `ε_rel` places the residual singular value of `c3` (after projecting
        // out c0..c2) at ~`1e-7 · σ_max` — comfortably above the σ-space RRQR
        // floor (~`1e-13 · σ_max`) yet below the σ²-Gram retain cutoff
        // (~`8·√(nk·ε)·σ_max` ≈ `4e-6 · σ_max`). RRQR ⇒ full rank 4; the
        // channel-aware audit may demote the weak column.
        let eps_rel = 1e-7_f64;
        let p = 4usize;
        let mut xmat = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let xi = x[i];
            xmat[[i, 0]] = 1.0; // constant
            xmat[[i, 1]] = xi; // linear
            xmat[[i, 2]] = (3.0 * xi).sin(); // oscillation
            // Near-separable: ~equal to a fixed combination of c0..c1 plus a
            // tiny independent wiggle.
            let w = (7.0 * xi).cos();
            xmat[[i, 3]] = 1.0 + 0.5 * xi + eps_rel * w;
        }

        let m = 2usize; // K-1 active softmax channels
        let specs: Vec<ParameterBlockSpec> = (0..m)
            .map(|a| {
                let mut spec = spec_from_dense_with_priority(
                    &format!("class_{a}"),
                    xmat.clone(),
                    100u8.saturating_add((m - a) as u8),
                );
                spec.jacobian_callback = Some(std::sync::Arc::new(AdditiveBlockJacobian {
                    design: xmat.clone(),
                    own_output: a,
                    n_family_outputs: m,
                }));
                spec
            })
            .collect();

        // The contract: canonicalisation must NOT fail with the post-T
        // rank-invariant `DimensionMismatch`. It may legitimately reduce
        // (dropping audit-gauge directions) or keep identity — either is fine;
        // what must never happen is the rank(J) != rank(J_can) panic.
        match canonicalize_for_identifiability(&specs) {
            Ok(canon) => {
                // If it reduced, the gauge must round-trip the reduced fit back
                // to raw width without shape mismatch.
                let theta: Vec<Array1<f64>> = canon
                    .reduced_specs
                    .iter()
                    .map(|s| Array1::<f64>::zeros(s.design.ncols()))
                    .collect();
                let raw = canon.gauge.lift_block_betas(&theta);
                assert_eq!(raw.len(), m, "lift must return one beta per channel block");
                for (a, b) in raw.iter().enumerate() {
                    assert_eq!(
                        b.len(),
                        p,
                        "lifted channel {a} beta must be raw width {p}, got {}",
                        b.len()
                    );
                }
            }
            Err(CustomFamilyError::DimensionMismatch { reason })
                if reason.contains("post-T rank invariant violated") =>
            {
                panic!(
                    "post-T rank invariant must hold under the shared audit rank \
                     convention on near-separable multinomial data (#1220); got: {reason}"
                );
            }
            // A correct, separation-aware refusal (e.g. fatal identifiability
            // failure) is acceptable per the issue — only the rank-invariant
            // panic is the regression we pin against.
            Err(other) => {
                let msg = format!("{other:?}");
                assert!(
                    !msg.contains("post-T rank invariant"),
                    "must not be a post-T rank-invariant failure; got {msg}"
                );
            }
        }
    }

    /// #1391 regression (function level): the FLAT-path post-T rank convention
    /// (the σ²-Gram positive-eigenspace metric `audit_convention_rank`) is the
    /// convention GAMLSS-PS canonicalisation runs (location-scale blocks carry an
    /// `AdditiveBlockJacobian` with `n_family_outputs = 2`, so the route is
    /// CHANNEL-AWARE).
    ///
    /// The earlier #1391 fix compared `audit_convention_rank(J_pre)` (the FULL,
    /// pre-reduction joint design) against `audit_convention_rank(J_can)`. But
    /// `audit_convention_rank` does ONE eigendecomposition of the joint `JᵀJ`,
    /// whereas the audit drops via a SEQUENTIAL per-block RESIDUALISED eigenspace
    /// walk. A direction that is residual-weak (absorbed by a higher-priority
    /// anchor) but still carries mass in the joint Gram is DROPPED by the walk yet
    /// COUNTED by the single joint eigendecomposition — so `audit_convention_rank`
    /// over-counts `J_pre` and a faithful column-selection `T` tripped a false
    /// `rank(J_pre) > rank(J_can)` (wine_gamair trial 22 60→59, wine_price_vs_temp
    /// trial 26 20→18; gam#1391).
    ///
    /// This fixture reproduces exactly that geometry: a near-separable direction
    /// whose joint-Gram eigenvalue sits ABOVE the σ²-Gram retain cutoff (so the
    /// single joint eigendecomposition counts it in `J_pre`) but which a faithful
    /// `T` drops. The test asserts (a) the OLD full-`J_pre` compare WOULD have
    /// tripped (`rank(J_pre) > rank(J_can)`), and (b) the NEW certificate-anchored
    /// invariant HOLDS: the reduced `J_can` is full column rank
    /// (`audit_convention_rank(J_can) == p_red`).
    #[test]
    fn post_t_invariant_jcan_full_rank_under_channel_aware_convention_1391() {
        let n = 96;
        let x = linspace(n);

        // Joint stacked Jacobian J_pre (n × 6): 5 well-conditioned columns plus
        // one NEAR-SEPARABLE column whose residual (after projecting the other 5)
        // sits just above the σ²-Gram retain cutoff, so the single joint
        // eigendecomposition counts it as full rank (rank 6).
        let p_total = 6usize;
        let mut j_pre = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            let xi = x[i];
            j_pre[[i, 0]] = 1.0;
            j_pre[[i, 1]] = xi;
            j_pre[[i, 2]] = (2.0 * xi).sin();
            j_pre[[i, 3]] = (3.0 * xi).cos();
            j_pre[[i, 4]] = xi * xi;
            // Near-separable: ~equal to column 0 plus a tiny independent wiggle.
            j_pre[[i, 5]] = 1.0 + 1e-5 * (7.0 * xi).sin();
        }

        let nk_scale = n;

        // The single joint eigendecomposition counts the near-separable column:
        // J_pre is full rank under `audit_convention_rank`.
        let rank_pre = audit_convention_rank(&j_pre, nk_scale, &[]);
        assert_eq!(
            rank_pre, p_total,
            "fixture must make the joint eigendecomposition count the near-separable \
             column in J_pre (rank {rank_pre} != {p_total})"
        );

        // A faithful column-selection T drops the near-separable column 5 (the
        // direction the sequential residual walk would absorb). J_can = kept cols.
        let kept: [usize; 5] = [0, 1, 2, 3, 4];
        let p_red = kept.len();
        let mut j_can = Array2::<f64>::zeros((n, p_red));
        for i in 0..n {
            for (out, &src) in kept.iter().enumerate() {
                j_can[[i, out]] = j_pre[[i, src]];
            }
        }

        let rank_can = audit_convention_rank(&j_can, nk_scale, &[]);

        // OLD compare (full J_pre vs reduced J_can) WOULD have tripped: the joint
        // eigendecomposition over-counts J_pre relative to the reduced design.
        assert!(
            rank_pre > rank_can,
            "fixture must reproduce the over-count: rank(J_pre)={rank_pre} must \
             exceed rank(J_can)={rank_can} under the single-eigendecomposition convention"
        );

        // NEW certificate-anchored invariant: the reduced J_can is FULL column
        // rank under the drop-deciding convention — the faithful-T guarantee that
        // no identifiable direction was dropped.
        assert_eq!(
            rank_can, p_red,
            "post-T rank invariant must hold: reduced J_can must be full rank \
             ({rank_can} != p_red {p_red}) under the channel-aware σ²-Gram convention (#1391)"
        );
    }

    /// #1391 (flat path / survival-marginal-slope #1388): the FLAT-path post-T
    /// rank convention (`flat_audit_convention_rank`) must rank the reduced design
    /// `J_can` under the SAME penalty-augmented, gauge-priority-tiered metric the
    /// flat audit used to decide drops — NOT a bare column-pivoted RRQR. A faithful
    /// column-selection `T` must leave `J_can` full column rank under that metric.
    #[test]
    fn flat_audit_convention_rank_jcan_full_rank_penalty_augmented() {
        let n = 64;
        let x = linspace(n);

        // Two penalized P-spline blocks (mu, sigma). A 2nd-difference penalty
        // annihilates the affine part; the reduced sigma block has its (aliased)
        // constant column dropped. The reduced J_can must be full rank under the
        // penalty-augmented, priority-tiered convention. mu and sigma use
        // DISTINCT covariates (`x` vs `z`) so the kept columns are genuinely
        // independent — the only aliased direction was sigma's constant, already
        // dropped by the faithful T.
        let z = Array1::from_iter((0..n).map(|i| (1.7 * x[i]).tanh() + 0.3 * x[i] * x[i]));
        // mu: constant, linear(x), sin(3x), cos(3x).
        let mut mu_d = Array2::<f64>::zeros((n, 4));
        // sigma: constant (DROPPED), linear(z), sin(5z), cos(5z).
        let mut sigma_d = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let xi = x[i];
            let zi = z[i];
            mu_d[[i, 0]] = 1.0;
            mu_d[[i, 1]] = xi;
            mu_d[[i, 2]] = (3.0 * xi).sin();
            mu_d[[i, 3]] = (3.0 * xi).cos();
            sigma_d[[i, 0]] = 1.0;
            sigma_d[[i, 1]] = zi;
            sigma_d[[i, 2]] = (5.0 * zi).sin();
            sigma_d[[i, 3]] = (5.0 * zi).cos();
        }

        // J_can = mu (4 cols) | sigma minus its constant (3 cols): cols
        // [linear(z), sin(5z), cos(5z)] of sigma.
        let p_red = 7usize;
        let mut j_can = Array2::<f64>::zeros((n, p_red));
        for i in 0..n {
            for j in 0..4 {
                j_can[[i, j]] = mu_d[[i, j]];
            }
            j_can[[i, 4]] = sigma_d[[i, 1]];
            j_can[[i, 5]] = sigma_d[[i, 2]];
            j_can[[i, 6]] = sigma_d[[i, 3]];
        }

        let mut s = Array2::<f64>::zeros((4, 4));
        s[[2, 2]] = 1.0;
        s[[3, 3]] = 1.0;
        s[[2, 3]] = -1.0;
        s[[3, 2]] = -1.0;
        let mut s_red = Array2::<f64>::zeros((3, 3));
        s_red[[1, 1]] = 1.0;
        s_red[[2, 2]] = 1.0;
        s_red[[1, 2]] = -1.0;
        s_red[[2, 1]] = -1.0;

        let blocks_can = [
            FlatRankBlock {
                width: 4,
                structural_penalty: Some(s),
                priority: 120,
            },
            FlatRankBlock {
                width: 3,
                structural_penalty: Some(s_red),
                priority: 100,
            },
        ];

        let rank_can = flat_audit_convention_rank(&j_can, &blocks_can);
        assert_eq!(
            rank_can, p_red,
            "reduced J_can must be full rank ({rank_can} != p_red {p_red}) under the \
             penalty-augmented, priority-tiered flat convention (#1391/#1388)"
        );
    }

    /// #1388 root cause (post-T rank invariant): on the cirrhosis/heart_failure
    /// survival benchmarks the joint marginal-slope design is UNDER-DETERMINED —
    /// once every categorical level becomes its own column, `p_joint > n`, so the
    /// joint design has FEWER independent rows than columns and its
    /// convention-rank is capped strictly below `p_joint`. A faithful
    /// column-selection `T` (which removes exactly the audit-demoted aliases)
    /// then leaves a reduced `J_can` whose rank is STILL row-capped below
    /// `p_total_red`: that is legitimate penalty-rested under-determination, NOT a
    /// defective `T`. The OLD strict invariant `rank(J_can) == p_total_red` tripped
    /// a FALSE "post-T rank invariant violated: rank(J_can) != p_red" abort here
    /// (the issue's "rank(T)=53 != dim=31"); the corrected invariant certifies
    /// `rank(J_can) == min(rank(J_pre), p_total_red)`, which a faithful `T`
    /// satisfies while a defective `T` (rank below the target) is still caught.
    ///
    /// This fixture is the exact geometry at unit scale: an under-determined flat
    /// joint (`p_raw = 6 > n = 4`) whose full-design convention-rank `rank(J_pre)`
    /// is row-capped at 4 < `p_total_red = 5`. The test asserts (a) the OLD
    /// `rank(J_can) == p_total_red` invariant WOULD have tripped
    /// (`rank(J_can) < p_total_red`), and (b) the NEW target-anchored invariant
    /// HOLDS (`rank(J_can) == min(rank(J_pre), p_total_red)`).
    #[test]
    fn post_t_invariant_underdetermined_jcan_meets_min_target_1388() {
        // n = 4 rows, p_raw = 6 columns split across two blocks → UNDER-DETERMINED.
        // Block A: a single shared constant (the aliased intercept, dropped by a
        // faithful T). Block B: 5 columns spanning at most the 4-row column space.
        let n = 4usize;
        let p_total_raw = 6usize;
        let mut j_pre = Array2::<f64>::zeros((n, p_total_raw));
        // Block A col 0: constant (aliases block B col 0).
        for i in 0..n {
            j_pre[[i, 0]] = 1.0;
        }
        // Block B cols 1..6: constant (= alias of A), then a 4-row-rank basis whose
        // 5 columns can carry at most 4 independent directions.
        for i in 0..n {
            let xi = i as f64;
            j_pre[[i, 1]] = 1.0; // alias of block-A constant
            j_pre[[i, 2]] = xi;
            j_pre[[i, 3]] = xi * xi;
            j_pre[[i, 4]] = xi * xi * xi;
            // col 5 is an EXACT duplicate of col 4, so block B's 5 columns carry
            // only 4 independent directions (an exactly rank-deficient Gram,
            // tolerance-independent — the post-T re-rank cannot reach 5).
            j_pre[[i, 5]] = xi * xi * xi;
        }

        // No penalties: pure data-rank regime. Under the flat convention the full
        // design's rank is row-capped.
        let blocks_pre = [
            FlatRankBlock {
                width: 1,
                structural_penalty: None,
                priority: 200,
            },
            FlatRankBlock {
                width: 5,
                structural_penalty: None,
                priority: 100,
            },
        ];
        let rank_j_pre = flat_audit_convention_rank(&j_pre, &blocks_pre);
        assert!(
            rank_j_pre < p_total_raw,
            "fixture must be under-determined: rank(J_pre)={rank_j_pre} must be < \
             p_raw={p_total_raw}",
        );

        // A faithful T drops block-A's aliased constant (column 0). The reduced
        // design keeps block B's 5 columns.
        let kept: [usize; 5] = [1, 2, 3, 4, 5];
        let p_total_red = kept.len();
        let mut j_can = Array2::<f64>::zeros((n, p_total_red));
        for i in 0..n {
            for (out, &src) in kept.iter().enumerate() {
                j_can[[i, out]] = j_pre[[i, src]];
            }
        }
        let blocks_can = [FlatRankBlock {
            width: 5,
            structural_penalty: None,
            priority: 100,
        }];
        let rank_j_can = flat_audit_convention_rank(&j_can, &blocks_can);

        // (a) The OLD strict invariant `rank(J_can) == p_total_red` WOULD have
        // tripped: the reduced design is row-capped below its column count.
        assert!(
            rank_j_can < p_total_red,
            "fixture must reproduce the under-determined deficit: rank(J_can)=\
             {rank_j_can} must be < p_red={p_total_red} (the OLD invariant would abort)",
        );

        // (b) The NEW target-anchored invariant HOLDS: a faithful T leaves J_can at
        // exactly the achievable rank `min(rank(J_pre), p_total_red)`.
        let rank_target = rank_j_pre.min(p_total_red);
        assert_eq!(
            rank_j_can, rank_target,
            "post-T rank invariant must hold under the corrected target: \
             rank(J_can)={rank_j_can} != min(rank(J_pre)={rank_j_pre}, \
             p_red={p_total_red})={rank_target} (#1388)",
        );
    }

    /// On a clean (non-fatal) configuration with a non-trivial penalty,
    /// canonicalisation must succeed with **identity** transforms (the
    /// fail-closed contract makes reduction unreachable, but the
    /// pull-back-on-identity path is still exercised by the lift
    /// machinery downstream). The reduced penalty equals the raw
    /// penalty modulo cloning.
    #[test]
    fn canonical_clean_specs_with_penalty_round_trip() {
        let n = 32;
        let x = linspace(n);
        let parametric = Array2::<f64>::from_shape_fn((n, 1), |(_, _)| 1.0);
        // Smooth WITHOUT a duplicate constant column — clean joint design.
        let mut smooth = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            smooth[[i, 0]] = x[i] * x[i];
            smooth[[i, 1]] = x[i] * x[i] * x[i];
        }
        let mut smooth_spec = spec_from_dense("smooth_only", smooth);
        let mut s = Array2::<f64>::zeros((2, 2));
        s[[0, 0]] = 4.0;
        s[[1, 1]] = 9.0;
        s[[0, 1]] = 1.5;
        s[[1, 0]] = 1.5;
        smooth_spec.penalties = vec![PenaltyMatrix::Dense(s.clone())];
        smooth_spec.initial_log_lambdas = Array1::from(vec![0.0]);
        let specs = [spec_from_dense("intercept", parametric), smooth_spec];
        let canon = canonicalize_for_identifiability(&specs)
            .expect("clean canonical must succeed with identity transforms");
        let smooth_reduced = &canon.reduced_specs[1];
        assert_eq!(smooth_reduced.penalties.len(), 1);
        let dense_red = smooth_reduced.penalties[0].as_dense_cow().into_owned();
        // Identity pullback: penalty dimensions equal raw design width.
        assert_eq!(dense_red.dim(), (2, 2));
        // Identity transform: per-block transform is the 2×2 identity.
        let t_smooth = canon.gauge.block_transform(1);
        assert_eq!(t_smooth.dim(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t_smooth[[i, j]], expected, "T_smooth must be identity");
            }
        }
    }

    use gam_test_support::{spec_from_dense, spec_from_dense_with_priority};

    /// #933: a `jacobian_callback`-only block (no `stacked_design`) whose audit
    /// attributes a dropped column is now SAFELY REDUCED rather than kept at raw
    /// width. The callback is wrapped in `GaugeComposedJacobian` so its effective
    /// Jacobian emerges at the reduced width, and the one Gauge lifts the reduced
    /// fit back through the SAME selection `T_i`. The round-trip invariant is η:
    /// the wrapped (reduced) callback applied to θ must equal the raw callback
    /// applied to the lifted β_raw = T_i · θ.
    #[test]
    fn callback_owned_geometry_reduces_and_round_trips_via_gauge() {
        use gam_problem::FamilyLinearizationState;

        let n = 48;
        let x = linspace(n);
        let mut anchor = Array2::<f64>::zeros((n, 2));
        let mut callback_owned = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            anchor[[i, 0]] = 1.0;
            anchor[[i, 1]] = x[i];
            // Block b column 0 aliases anchor's x direction; column 1 is x².
            callback_owned[[i, 0]] = x[i];
            callback_owned[[i, 1]] = x[i] * x[i];
        }

        let anchor_spec = spec_from_dense_with_priority("marginal_surface", anchor, 150);
        let mut callback_spec =
            spec_from_dense_with_priority("logslope_surface", callback_owned.clone(), 120);
        let raw_callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(AdditiveBlockJacobian {
            design: callback_owned.clone(),
            own_output: 0,
            n_family_outputs: 1,
        });
        callback_spec.jacobian_callback = Some(Arc::clone(&raw_callback));
        let specs = [anchor_spec, callback_spec];

        let canon = canonicalize_for_identifiability(&specs)
            .expect("callback-only overlap must now reduce safely (#933)");

        assert!(
            !canon.audit.fatal,
            "priority-owned overlap should be non-fatal; got {}",
            canon.audit.summary,
        );
        assert!(
            canon
                .audit
                .dropped_columns
                .iter()
                .any(|drop| drop.block == "logslope_surface"),
            "test must exercise an attributed logslope drop; got {:?}",
            canon.audit.dropped_columns,
        );
        // Anchor (higher priority) keeps both columns; the aliased callback block
        // now sheds exactly one column — the reduction the width-preserving path
        // used to refuse.
        assert_eq!(
            canon.reduced_specs[0].design.ncols(),
            2,
            "anchor block keeps raw width"
        );
        assert_eq!(
            canon.reduced_specs[1].design.ncols(),
            1,
            "callback-only block is now column-reduced via the composed gauge (#933)",
        );

        // The block's gauge transform is the 2×1 selection of the surviving
        // column, and the reduced spec carries a wrapped callback that emits the
        // reduced width.
        let t_b = canon.gauge.block_transform(1);
        assert_eq!(t_b.dim(), (2, 1), "reduced callback block transform is 2×1");
        let reduced_cb = canon.reduced_specs[1]
            .jacobian_callback
            .as_ref()
            .expect("reduced callback-only block must still carry a callback");

        // η round-trip: (J_raw · T_b) · θ == J_raw · (T_b · θ) for an arbitrary θ.
        let theta = Array1::from(vec![0.73_f64]);
        let beta_raw = t_b.dot(&theta); // length-2 lifted coefficient
        let zeros_red = vec![0.0_f64; 1];
        let state_red = FamilyLinearizationState {
            beta: &zeros_red,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        };
        let j_reduced = reduced_cb
            .effective_jacobian_at(&state_red)
            .expect("reduced callback Jacobian");
        assert_eq!(j_reduced.dim(), (n, 1), "reduced Jacobian is n×1");
        let eta_reduced = j_reduced.dot(&theta);

        let zeros_raw = vec![0.0_f64; 2];
        let state_raw = FamilyLinearizationState {
            beta: &zeros_raw,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        };
        let j_raw = raw_callback
            .effective_jacobian_at(&state_raw)
            .expect("raw callback Jacobian");
        let eta_raw = j_raw.dot(&beta_raw);
        for i in 0..n {
            assert!(
                (eta_reduced[i] - eta_raw[i]).abs() < 1e-12,
                "η must be invariant under the composed-gauge reduction at row {i}: \
                 reduced {} vs raw {}",
                eta_reduced[i],
                eta_raw[i],
            );
        }
    }

    /// #933 assembly-safety isolation: the η round-trip above proves the reduced
    /// callback emits the right Jacobian, but the panic the gauge cutover was
    /// meant to retire is a *shape* panic during per-row Hessian assembly —
    /// `DesignMatrix::syr_row_into shape mismatch` / `row_outer_into shape
    /// mismatch` — which fires when the family's captured reduced `design` and
    /// its effective Jacobian disagree on column count (the family was fed a
    /// column-reduced spec but still asserts the raw width). This test pins the
    /// precondition that makes that panic impossible by construction: after the
    /// canonical reduction, the reduced callback block's `design.ncols()` equals
    /// the wrapped callback's emitted reduced width, and an actual `syr_row_into`
    /// / `row_outer_into` against an assembly target sized from the reduced
    /// design SUCCEEDS (returns `Ok`, no shape-mismatch error) for every row.
    #[test]
    fn callback_owned_reduced_block_assembles_row_hessian_without_shape_panic() {
        use gam_problem::FamilyLinearizationState;

        let n = 48;
        let x = linspace(n);
        let mut anchor = Array2::<f64>::zeros((n, 2));
        let mut callback_owned = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            anchor[[i, 0]] = 1.0;
            anchor[[i, 1]] = x[i];
            // Column 0 aliases anchor's x; column 1 (x²) is genuinely new.
            callback_owned[[i, 0]] = x[i];
            callback_owned[[i, 1]] = x[i] * x[i];
        }

        // Leave the shared direction unpenalised so the audit sees the real
        // `ker(J) ∩ ker(S)` gauge direction and must quotient it through the
        // callback wrapper. A full-rank ridge would make `[J; S]` full rank and
        // let this test pass without exercising the reduced callback path.
        let anchor_spec = spec_from_dense_with_priority("marginal_surface", anchor, 150);
        let mut callback_spec =
            spec_from_dense_with_priority("logslope_surface", callback_owned.clone(), 120);
        let raw_callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(AdditiveBlockJacobian {
            design: callback_owned.clone(),
            own_output: 0,
            n_family_outputs: 1,
        });
        callback_spec.jacobian_callback = Some(Arc::clone(&raw_callback));

        let canon = canonicalize_for_identifiability(&[anchor_spec, callback_spec])
            .expect("callback-only overlap must reduce safely (#933)");

        let reduced_block = &canon.reduced_specs[1];
        let reduced_design = &reduced_block.design;
        assert!(
            canon
                .audit
                .dropped_columns
                .iter()
                .any(|drop| drop.block == "logslope_surface"),
            "test must exercise a real callback-owned gauge drop; got {:?}",
            canon.audit.dropped_columns,
        );
        assert_eq!(
            reduced_design.ncols(),
            1,
            "callback-owned block must be reduced to the non-aliased column",
        );
        let reduced_cb = reduced_block
            .jacobian_callback
            .as_ref()
            .expect("reduced callback-only block must still carry a callback");

        // The width the family's row-Hessian assembly asserts on is exactly the
        // reduced `design`'s column count. The wrapped callback must emit that
        // SAME width — the equality that retires the shape panic.
        let zeros_red = vec![0.0_f64; reduced_design.ncols()];
        let state_red = FamilyLinearizationState {
            beta: &zeros_red,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        };
        let j_reduced = reduced_cb
            .effective_jacobian_at(&state_red)
            .expect("reduced callback Jacobian");
        assert_eq!(
            reduced_design.ncols(),
            j_reduced.ncols(),
            "reduced design width {} must match the wrapped callback's emitted width {} \
             — the precondition that makes syr_row_into / row_outer_into shape-match",
            reduced_design.ncols(),
            j_reduced.ncols(),
        );

        // Drive the real assembly primitives. A Hessian-assembly target is sized
        // from the family's reduced design (`p × p` for the symmetric per-block
        // block, `p × p` for the self cross-block). Every row's update must
        // return Ok — a raw/reduced width desync would surface here as the named
        // `syr_row_into shape mismatch` / `row_outer_into shape mismatch` Err.
        let p = reduced_design.ncols();
        let mut syr_target = Array2::<f64>::zeros((p, p));
        let mut outer_target = Array2::<f64>::zeros((p, p));
        for row in 0..n {
            reduced_design
                .syr_row_into(row, 1.0, &mut syr_target)
                .unwrap_or_else(|e| {
                    panic!("row {row}: reduced design must syr_row_into without shape panic: {e}")
                });
            reduced_design
                .row_outer_into(row, reduced_design, 1.0, &mut outer_target)
                .unwrap_or_else(|e| {
                    panic!("row {row}: reduced design must row_outer_into without shape panic: {e}")
                });
        }
        // syr accumulates Σ_row xxᵀ; the (0,0) entry is Σ_row x_row² > 0 for a
        // non-degenerate surviving column, confirming the assembly ran (not a
        // silent no-op on a zero-width target).
        assert!(
            syr_target[[0, 0]] > 0.0,
            "syr assembly must accumulate a positive diagonal on the surviving column",
        );
    }

    /// #933 sweep coverage: the single-channel assembly-safety test above
    /// guards the GLM/marginal-slope shape of callback-owned block. A
    /// location-scale family emits a MULTI-channel effective Jacobian
    /// (`n_family_outputs > 1`), which drives the channel-aware audit and the
    /// `BlockJacobianAsRowOp::from_callback` zero-padding path. This test pins
    /// that the channel-aware reduction is ALSO assembly-safe: a `k = 2`
    /// callback block whose own column aliases a higher-priority anchor sheds
    /// the aliased column, the reduced design width equals the wrapped callback's
    /// per-channel emitted width, and real `syr_row_into` / `row_outer_into`
    /// against a reduced-width target returns Ok for every row.
    #[test]
    fn multichannel_callback_owned_reduced_block_assembles_without_shape_panic() {
        use gam_problem::FamilyLinearizationState;

        let n = 40;
        let x = linspace(n);
        let mut anchor = Array2::<f64>::zeros((n, 2));
        let mut callback_owned = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            anchor[[i, 0]] = 1.0;
            anchor[[i, 1]] = x[i];
            // Column 0 aliases anchor's x; column 1 (x²) is genuinely new.
            callback_owned[[i, 0]] = x[i];
            callback_owned[[i, 1]] = x[i] * x[i];
        }

        let anchor_spec = spec_from_dense_with_priority("location_anchor", anchor, 150);
        let mut callback_spec =
            spec_from_dense_with_priority("scale_logslope", callback_owned.clone(), 120);
        // A 2-output (location-scale) callback: this block drives output 1 (the
        // scale channel); its raw Jacobian is (2·n × 2) with the block design in
        // the channel-1 row band and zeros in channel 0.
        let raw_callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(AdditiveBlockJacobian {
            design: callback_owned.clone(),
            own_output: 1,
            n_family_outputs: 2,
        });
        callback_spec.jacobian_callback = Some(Arc::clone(&raw_callback));

        let canon = canonicalize_for_identifiability(&[anchor_spec, callback_spec])
            .expect("multi-channel callback overlap must reduce safely (#933)");

        let reduced_block = &canon.reduced_specs[1];
        let reduced_design = &reduced_block.design;
        let reduced_cb = reduced_block
            .jacobian_callback
            .as_ref()
            .expect("reduced multi-channel callback block must still carry a callback");
        assert_eq!(reduced_cb.n_outputs(), 2, "channel count is preserved");

        // The reduced effective Jacobian is (2·n × r_reduced); its column count
        // is the width the family's per-block row-Hessian assembly asserts on and
        // must equal the reduced design's column count.
        let zeros_red = vec![0.0_f64; reduced_design.ncols()];
        let state_red = FamilyLinearizationState {
            beta: &zeros_red,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        };
        let j_reduced = reduced_cb
            .effective_jacobian_at(&state_red)
            .expect("reduced multi-channel callback Jacobian");
        assert_eq!(
            j_reduced.nrows(),
            2 * n,
            "two-channel Jacobian stacks both output bands",
        );
        assert_eq!(
            reduced_design.ncols(),
            j_reduced.ncols(),
            "reduced design width {} must match the wrapped multi-channel callback's \
             emitted width {} — the shape-match precondition for row-Hessian assembly",
            reduced_design.ncols(),
            j_reduced.ncols(),
        );

        // Drive the real assembly primitives at the reduced width.
        let p = reduced_design.ncols();
        let mut syr_target = Array2::<f64>::zeros((p, p));
        let mut outer_target = Array2::<f64>::zeros((p, p));
        for row in 0..n {
            reduced_design
                .syr_row_into(row, 1.0, &mut syr_target)
                .unwrap_or_else(|e| {
                    panic!("row {row}: reduced design must syr_row_into without shape panic: {e}")
                });
            reduced_design
                .row_outer_into(row, reduced_design, 1.0, &mut outer_target)
                .unwrap_or_else(|e| {
                    panic!("row {row}: reduced design must row_outer_into without shape panic: {e}")
                });
        }
        assert!(
            syr_target[[0, 0]] > 0.0,
            "multi-channel reduced assembly must accumulate a positive diagonal",
        );
    }

    /// Two single-channel blocks with an exact shared column (anchor block
    /// `a` has column [1, x]; block `b` has [x, x²]). The `x` direction is
    /// shared. Orthogonalisation is unconditional, so block `b` (lower priority)
    /// must shed exactly one direction, the joint reduced design must be
    /// full-rank, and the round-trip lift must reproduce the raw prediction.
    #[test]
    fn orthogonalize_removes_exact_cross_block_overlap_and_round_trips() {
        let n = 48;
        let x = linspace(n);
        // Block a (high priority): [1, x].
        let mut a = Array2::<f64>::zeros((n, 2));
        // Block b (low priority): [x, x²]  → its first column aliases a's x.
        let mut b = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            a[[i, 0]] = 1.0;
            a[[i, 1]] = x[i];
            b[[i, 0]] = x[i];
            b[[i, 1]] = x[i] * x[i];
        }
        let specs = [
            spec_from_dense_with_priority("anchor", a.clone(), 150),
            spec_from_dense_with_priority("overlap", b.clone(), 120),
        ];
        let canon = canonicalize_for_identifiability(&specs)
            .expect("orthogonalisation must resolve the overlap, not refuse");

        // Block b shed exactly one direction (the x alias): V_b is 2×1.
        let v_b = canon.gauge.block_transform(1);
        assert_eq!(
            v_b.ncols(),
            1,
            "overlap block must keep exactly one direction"
        );
        assert_eq!(
            v_b.nrows(),
            2,
            "overlap block transform maps from raw width 2"
        );
        // Anchor block keeps both directions (square rotation).
        assert_eq!(canon.gauge.block_transform(0).ncols(), 2);

        // Round-trip: a reduced fit θ lifts to raw β = V·θ and predicts
        // identically through the raw designs.
        let theta = vec![Array1::from(vec![0.7, -0.3]), Array1::from(vec![1.4])];
        let raw = canon.gauge.lift_block_betas(&theta);
        assert_eq!(raw[0].len(), 2);
        assert_eq!(raw[1].len(), 2);
        // Raw prediction = a·β_a + b·β_b.
        let pred_a = a.dot(&raw[0]);
        let pred_b = b.dot(&raw[1]);
        // Reduced prediction = (a·V_a)·θ_a + (b·V_b)·θ_b must equal it.
        let v_a = canon.gauge.block_transform(0);
        let red_a = a.dot(&v_a).dot(&theta[0]);
        let red_b = b.dot(&v_b).dot(&theta[1]);
        for i in 0..n {
            let raw_pred = pred_a[i] + pred_b[i];
            let red_pred = red_a[i] + red_b[i];
            assert!(
                (raw_pred - red_pred).abs() < 1e-9,
                "row {i}: raw prediction {raw_pred} != reduced prediction {red_pred}",
            );
        }
    }

    /// A clean (non-overlapping) two-block design must canonicalise to identity
    /// transforms even though orthogonalisation runs unconditionally: with no
    /// cross-block overlap to remove, `try_orthogonalize_blocks` finds nothing to
    /// drop and the audit gate produces raw-width identity transforms.
    #[test]
    fn orthogonalize_clean_design_yields_identity_transforms() {
        let n = 32;
        let x = linspace(n);
        let mut p = Array2::<f64>::zeros((n, 2));
        let mut s = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            p[[i, 0]] = 1.0;
            p[[i, 1]] = x[i];
            s[[i, 0]] = x[i] * x[i];
            s[[i, 1]] = x[i] * x[i] * x[i];
        }
        let specs = [
            spec_from_dense_with_priority("p", p, 150),
            spec_from_dense_with_priority("s", s, 120),
        ];
        let canon = canonicalize_for_identifiability(&specs).expect("clean design canonicalises");
        // Identity transforms (nothing to orthogonalise) on the clean design.
        assert_eq!(canon.gauge.block_transform(0).dim(), (2, 2));
        assert_eq!(canon.gauge.block_transform(1).dim(), (2, 2));
    }

    /// Direct unit test of the compiler primitive: a block whose columns are
    /// fully spanned by a higher-priority anchor must shed all overlapping
    /// directions, and a non-overlapping configuration must keep full width.
    #[test]
    fn orthogonalize_design_blocks_drops_only_overlap() {
        use crate::families::compiler::orthogonalize_design_blocks;
        let n = 40;
        let x = linspace(n);
        let mut anchor = Array2::<f64>::zeros((n, 2));
        let mut overlap = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            anchor[[i, 0]] = 1.0;
            anchor[[i, 1]] = x[i];
            // First column is exactly anchor's x; second is genuinely new.
            overlap[[i, 0]] = x[i];
            overlap[[i, 1]] = x[i] * x[i] * x[i];
        }
        let weight = vec![1.0_f64; n];
        let res =
            orthogonalize_design_blocks(&[anchor.clone(), overlap.clone()], &[150, 120], &weight)
                .expect("orthogonalisation must succeed");
        assert_eq!(
            res.block_transforms[0].ncols(),
            2,
            "anchor keeps full width"
        );
        assert_eq!(
            res.block_transforms[1].ncols(),
            1,
            "overlap block sheds exactly the aliased direction",
        );
        assert_eq!(
            res.dropped,
            vec![(1, 1)],
            "one direction dropped from block 1"
        );
    }

    /// Regression for gam#1068: a multi-channel block (`stacked_design = Some`,
    /// the latent-survival / survival-LS time-transform layout whose solver eta
    /// is `[entry; exit; deriv] · β` — a `3·n`-row operator) must NOT lose its
    /// stacked operator through canonicalisation, even when it shares a column
    /// (here the constant) with a higher-priority block that would otherwise
    /// trigger the single-channel orthogonalisation reparam. The
    /// orthogonalisation path sets `stacked_design = None`; if it ran on this
    /// block the family's eta would collapse from `3·n` rows to `n` and trip
    /// `latent survival time eta length mismatch: got n, expected 3n`. The fix
    /// defers any spec set carrying a `stacked_design` to the column-selection
    /// audit, which preserves the stacked operator.
    #[test]
    fn canonical_preserves_stacked_design_for_multichannel_block() {
        let n = 24;
        let x = linspace(n);
        // Mean block (higher priority) carries the constant + a linear term.
        let mut mean = Array2::<f64>::zeros((n, 2));
        // Time block (lower priority) single-channel `design` is the n-row exit
        // view; its eta-producing `stacked_design` is the 3n-row [entry; exit;
        // deriv] stack. Both carry the shared constant column (the alias the
        // orthogonaliser would try to remove).
        let mut time_exit = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            mean[[i, 0]] = 1.0;
            mean[[i, 1]] = x[i];
            time_exit[[i, 0]] = 1.0;
            time_exit[[i, 1]] = x[i] * x[i];
        }
        let mut stacked = Array2::<f64>::zeros((3 * n, 2));
        for i in 0..n {
            // entry channel (zeros baseline), exit channel (= design), deriv.
            stacked[[i, 0]] = 1.0;
            stacked[[i, 1]] = x[i] * x[i] * 0.5;
            stacked[[n + i, 0]] = 1.0;
            stacked[[n + i, 1]] = x[i] * x[i];
            stacked[[2 * n + i, 0]] = 0.0;
            stacked[[2 * n + i, 1]] = 2.0 * x[i];
        }

        let mean_spec = spec_from_dense_with_priority("mean", mean, 150);
        let mut time_spec = spec_from_dense_with_priority("time_transform", time_exit, 200);
        time_spec.stacked_design = Some(DesignMatrix::Dense(DenseDesignMatrix::from(stacked)));
        time_spec.stacked_offset = Some(Array1::<f64>::zeros(3 * n));

        let specs = [mean_spec, time_spec];
        let canon = canonicalize_for_identifiability(&specs)
            .expect("multi-channel canonicalisation must succeed (defer to audit gate)");

        let time_reduced = canon
            .reduced_specs
            .iter()
            .find(|s| s.name == "time_transform")
            .expect("time block survives canonicalisation");
        let stacked_after = time_reduced
            .stacked_design
            .as_ref()
            .expect("stacked_design must survive canonicalisation (gam#1068)");
        assert_eq!(
            stacked_after.nrows(),
            3 * n,
            "stacked eta operator must keep its 3·n rows; collapsing to n is the #1068 bug",
        );
        assert!(
            time_reduced.stacked_offset.is_some(),
            "stacked_offset must survive alongside stacked_design",
        );
    }

    /// Regression for gam#1110: the survival location-scale parametric-AFT joint
    /// pairs a geometry-OWNING `time_transform` block (`stacked_design = Some`,
    /// gauge priority 200 — the 3·n-row `[entry; exit; deriv]` eta operator the
    /// family's `z`-lift / monotonicity layout depends on, #1068) with a PLAIN
    /// `threshold` (location) block `[1, age]` at lower priority 150. The
    /// constant columns of the two blocks are mutually aliased, so the flat
    /// joint design is rank-deficient by one.
    ///
    /// The old veto was all-or-nothing: ANY block owning its geometry (the time
    /// block does, via `stacked_design`) forced EVERY block to keep its raw
    /// width, leaving the surplus constant in the joint design. The downstream
    /// robust/ridge solve then resolved that residual rank deficiency by
    /// collapsing the genuine `age` covariate direction to EXACTLY zero
    /// (`gam_a_age = 0.00000`, the #1110 tell).
    ///
    /// The fix vetoes the width-preserving path ONLY when a dropped column is
    /// attributed to a block that actually owns its geometry. The surplus
    /// constant is attributed by gauge priority to the LOWER-priority PLAIN
    /// `threshold` block (its intercept), so the column-selection reduction
    /// drops it while:
    ///   * the `time_transform` block stays at raw width with its `stacked_design`
    ///     intact (the #1068 layout that would otherwise desync the z-lift), and
    ///   * the genuine non-constant `age` covariate direction is KEPT — a covariate
    ///     is never an aliased-constant partner, so it can never be the attributed
    ///     drop. After the fix the threshold block retains its `age` column.
    #[test]
    fn canonical_survival_ls_aft_keeps_covariate_when_time_owns_geometry() {
        let n = 96;
        let x = linspace(n);
        // age covariate, centred and clearly non-constant.
        let age: Vec<f64> = (0..n)
            .map(|i| (i as f64) - (n as f64 - 1.0) / 2.0)
            .collect();

        // time_transform: owns geometry via a 3·n-row stacked operator. Its
        // n-row canonical `design` carries the shared constant (col 0) plus a
        // monotone time column (col 1) — the additive location baseline that
        // gauge ownership pins to the highest-priority block.
        let mut time_exit = Array2::<f64>::zeros((n, 2));
        let mut time_stacked = Array2::<f64>::zeros((3 * n, 2));
        for i in 0..n {
            time_exit[[i, 0]] = 1.0;
            time_exit[[i, 1]] = x[i];
            // [entry; exit; deriv] stack.
            time_stacked[[i, 0]] = 1.0;
            time_stacked[[i, 1]] = 0.5 * x[i];
            time_stacked[[n + i, 0]] = 1.0;
            time_stacked[[n + i, 1]] = x[i];
            time_stacked[[2 * n + i, 0]] = 0.0;
            time_stacked[[2 * n + i, 1]] = 1.0;
        }

        // threshold (location covariate) block: PLAIN `[1, age]`. The intercept
        // (col 0) is aliased with the time block's constant; `age` (col 1) is a
        // genuine covariate direction that MUST SURVIVE.
        let mut threshold = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            threshold[[i, 0]] = 1.0;
            threshold[[i, 1]] = age[i];
        }

        let mut t_spec = spec_from_dense("time_transform", time_exit);
        t_spec.gauge_priority = 200;
        t_spec.stacked_design = Some(DesignMatrix::Dense(DenseDesignMatrix::from(time_stacked)));
        t_spec.stacked_offset = Some(Array1::<f64>::zeros(3 * n));
        let mut th_spec = spec_from_dense("threshold", threshold);
        th_spec.gauge_priority = 150;

        let specs = [t_spec, th_spec];
        let canon = canonicalize_for_identifiability(&specs).expect(
            "survival-LS AFT aliased-constant joint with distinct gauge_priority must \
             succeed (gauge-resolved, gam#1110)",
        );

        assert!(
            !canon.audit.fatal,
            "audit must be non-fatal: the surplus constant is gauge-resolvable; got {}",
            canon.audit.summary,
        );

        // No drop may be attributed to the geometry-owning time_transform block:
        // its raw width must be preserved so the #1068 stacked layout stays valid.
        for drop in &canon.audit.dropped_columns {
            assert_ne!(
                drop.block, "time_transform",
                "geometry-owning time_transform block must never be the attributed \
                 drop origin (gam#1110/#1068); got {drop:?}",
            );
        }

        // The time block keeps both raw columns AND its 3·n-row stacked operator
        // (the all-or-nothing exemption's ONE correct effect, preserved here).
        let time_reduced = canon
            .reduced_specs
            .iter()
            .find(|s| s.name == "time_transform")
            .expect("time_transform survives canonicalisation");
        assert_eq!(
            time_reduced.design.ncols(),
            2,
            "time_transform must keep both raw columns (raw-width owned geometry)",
        );
        assert_eq!(
            time_reduced
                .stacked_design
                .as_ref()
                .expect("time_transform stacked_design must survive (gam#1068)")
                .nrows(),
            3 * n,
            "time_transform stacked eta operator must keep its 3·n rows",
        );

        // The crux of #1110: the threshold block must KEEP its genuine `age`
        // covariate direction. After dropping its redundant intercept it has one
        // surviving column, and that column must VARY across rows (it is `age`,
        // not a constant). The old all-or-nothing veto left the joint
        // rank-deficient and the downstream solve pinned `age` to exactly 0.
        let threshold_reduced = canon
            .reduced_specs
            .iter()
            .find(|s| s.name == "threshold")
            .expect("threshold block survives canonicalisation");
        let th_dense = threshold_reduced
            .design
            .try_to_dense_arc("threshold reduced densify")
            .expect("dense threshold design");
        let th_view = th_dense.as_ref();
        assert!(
            th_view.ncols() >= 1,
            "threshold block must retain its `age` covariate column",
        );
        let has_nonconstant_column = (0..th_view.ncols()).any(|c| {
            let col = th_view.column(c);
            let first = col[0];
            col.iter().any(|&v| (v - first).abs() > 1e-9)
        });
        assert!(
            has_nonconstant_column,
            "threshold block must KEEP the genuine non-constant `age` covariate \
             direction after canonicalisation — the #1110 bug dropped/pinned it, \
             leaving only constant columns (gam_a_age = 0.00000)",
        );

        // Joint rank: raw 2 + 2 = 4 columns, minus the one redundant constant = 3.
        // time keeps 2, threshold keeps 1 (`age`).
        let reduced_total: usize = canon.reduced_specs.iter().map(|s| s.design.ncols()).sum();
        assert_eq!(
            reduced_total,
            3,
            "joint rank = 4 raw columns − 1 surplus constant = 3; got {reduced_total} \
             (per-block reduced ncols {:?})",
            canon
                .reduced_specs
                .iter()
                .map(|s| (s.name.clone(), s.design.ncols()))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            threshold_reduced.design.ncols(),
            1,
            "threshold keeps exactly its `age` column after the intercept drop",
        );
    }
}
