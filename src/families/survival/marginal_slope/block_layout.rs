//! Coefficient-block layout: the canonical block ordering (`HessBlock`),
//! the per-block coefficient slices (`BlockSlices`), the flex primary-space
//! index schema (`FlexPrimarySlices`), the GPU row-batch descriptor, and the
//! per-z score-warp striping that direct-sums scalar warps into a blockspec.

use super::*;

#[derive(Clone)]
pub(crate) struct PerZScoreWarpPrepared {
    pub(crate) block: ParameterBlockInput,
    pub(crate) runtime: DeviationRuntime,
    pub(crate) score_dim: usize,
}

impl PerZScoreWarpPrepared {
    #[inline]
    pub(crate) fn basis_dim(&self) -> usize {
        self.runtime.basis_dim()
    }

    #[inline]
    pub(crate) fn total_basis_dim(&self) -> usize {
        self.basis_dim() * self.score_dim
    }
}

pub(crate) fn score_warp_component_range(
    runtime: &DeviationRuntime,
    coord: usize,
) -> std::ops::Range<usize> {
    let p = runtime.basis_dim();
    coord * p..(coord + 1) * p
}

pub(crate) fn score_warp_component_beta(
    runtime: &DeviationRuntime,
    beta: &Array1<f64>,
    coord: usize,
) -> Result<Array1<f64>, String> {
    let range = score_warp_component_range(runtime, coord);
    if range.end > beta.len() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival score-warp coefficient block is too short for z coordinate {coord}: need {}, got {}",
                range.end,
                beta.len()
            ),
        }
        .into());
    }
    Ok(beta.slice(s![range]).to_owned())
}

/// Stripe a (post-reparam) scalar score-warp `base` across K z coordinates
/// to produce the direct-sum block `Φ_total = [Φ(z_1) | Φ(z_2) | ... | Φ(z_K)]`.
///
/// Caller is responsible for any cross-block reparameterisation on
/// `base.runtime` BEFORE calling this — the install path has already
/// updated `base.runtime.basis_dim()` to the kept dimension `p_kept`,
/// and `base.block.design / penalties / nullspace_dims` reflect that
/// reparam. Each per-z stripe then evaluates `runtime.design_at_training_with_residual`
/// at `z[:, k]` so the cached parametric anchor rows are folded into every
/// stripe, giving a striped design that is jointly orthogonal (in the W-
/// metric used during reparam) to span(anchors) at training rows.
pub(crate) fn stripe_score_warp_across_z_coords(
    base: ParameterBlockInput,
    base_runtime: DeviationRuntime,
    z: &Array2<f64>,
) -> Result<PerZScoreWarpPrepared, String> {
    let score_dim = z.ncols();
    if score_dim == 0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival score-warp requires at least one z coordinate".to_string(),
        }
        .into());
    }
    if score_dim == 1 {
        return Ok(PerZScoreWarpPrepared {
            block: base,
            runtime: base_runtime,
            score_dim,
        });
    }

    // Vector-z score warp is the direct sum of K scalar warp spaces:
    //
    //     h_i = sum_{k=1}^K W_k(z_{ik}) beta_k .
    //
    // The coefficient vector is laid out as [beta_1 | ... | beta_K],
    // and the row design is the horizontal concatenation of the K scalar
    // designs. Per-z stripes go through `design_at_training_with_residual`
    // so that any installed anchor residual (from cross-block reparam) is
    // subtracted uniformly across all K stripes — each stripe lives in
    // the SAME orthogonal complement of the parametric anchor span as the
    // primary z coordinate's basis. Penalties are block-local on each
    // coordinate slice, giving each W_k its own smoothing parameter unless
    // a later grouping layer intentionally ties precision labels.
    let p = base_runtime.basis_dim();
    let n = z.nrows();
    let mut design = Array2::<f64>::zeros((n, p * score_dim));
    for coord in 0..score_dim {
        let z_coord = z.column(coord).to_owned();
        let coord_design = base_runtime.design_at_training_with_residual(&z_coord)?;
        if coord_design.nrows() != n || coord_design.ncols() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp design shape mismatch for z coordinate {coord}: got {}x{}, expected {n}x{p}",
                    coord_design.nrows(),
                    coord_design.ncols()
                ),
            }
            .into());
        }
        design
            .slice_mut(s![.., coord * p..(coord + 1) * p])
            .assign(&coord_design);
    }

    let mut block = base.clone();
    block.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design));
    block.offset = Array1::zeros(n);
    block.initial_beta = Some(Array1::zeros(p * score_dim));
    block.initial_log_lambdas = None;
    let base_penalties = base.penalties.clone();
    let base_nullspaces = base.nullspace_dims.clone();
    block.penalties.clear();
    block.nullspace_dims.clear();
    for coord in 0..score_dim {
        let col_range = coord * p..(coord + 1) * p;
        for (penalty_idx, penalty) in base_penalties.iter().enumerate() {
            let local = match penalty {
                crate::model_types::PenaltySpec::Dense(matrix)
                | crate::model_types::PenaltySpec::DenseWithMean { matrix, .. } => matrix.clone(),
                crate::model_types::PenaltySpec::Block { local, .. } => local.clone(),
            };
            block
                .penalties
                .push(crate::model_types::PenaltySpec::Block {
                    local,
                    col_range: col_range.clone(),
                    prior_mean: crate::model_types::CoefficientPriorMean::Zero,
                    structure_hint: None,
                    op: None,
                });
            block.nullspace_dims.push(
                base_nullspaces
                    .get(penalty_idx)
                    .copied()
                    .unwrap_or_default(),
            );
        }
    }

    Ok(PerZScoreWarpPrepared {
        block,
        runtime: base_runtime,
        score_dim,
    })
}

pub(crate) fn build_per_z_score_warp_aux_blockspec(
    prepared: &PerZScoreWarpPrepared,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let mut block = prepared.block.clone();
    block.initial_log_lambdas = Some(rho);
    let total_p = prepared.total_basis_dim();
    let candidate = beta_hint.unwrap_or_else(|| Array1::<f64>::zeros(total_p));
    if candidate.len() != total_p {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival score-warp beta hint length mismatch: got {}, expected {total_p}",
                candidate.len()
            ),
        }
        .into());
    }
    let mut projected = Array1::<f64>::zeros(total_p);
    for coord in 0..prepared.score_dim {
        let range = score_warp_component_range(&prepared.runtime, coord);
        let proposed = candidate.slice(s![range.clone()]).to_owned();
        let zero = Array1::<f64>::zeros(prepared.basis_dim());
        let local = project_monotone_feasible_beta(
            &prepared.runtime,
            &zero,
            &proposed,
            &format!("score_warp_dev[z{coord}]"),
        )?;
        projected.slice_mut(s![range]).assign(&local);
    }
    block.initial_beta = Some(projected);
    let mut spec = block.intospec("score_warp_dev")?;
    // Survival marginal-slope gauge ownership: score-warp deviations
    // are pure shape modifications around the latent score axis and
    // should never own a shared affine direction with time, marginal,
    // or logslope blocks. Set a low priority so the canonical-gauge
    // selector drops shared directions from score_warp_dev before any
    // parametric block loses a column.
    spec.gauge_priority = 80;
    if prepared.score_dim > 1 {
        // The physical penalty order mirrors the direct-sum coefficient
        // layout: all penalties for beta_1, then beta_2, ..., beta_K.  Giving
        // each coordinate-local penalty a distinct precision label makes the
        // default MAP problem one-lambda-per-W_k; task-04 coefficient-group
        // penalties can still introduce intentionally shared precision
        // factors without accidentally tying these base smoothness penalties.
        if spec.penalties.len() % prepared.score_dim != 0 {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp penalty count {} is not divisible by K={}",
                    spec.penalties.len(),
                    prepared.score_dim
                ),
            }
            .into());
        }
        let penalties_per_coord = spec.penalties.len() / prepared.score_dim;
        for coord in 0..prepared.score_dim {
            for penalty_idx in 0..penalties_per_coord {
                let flat_idx = coord * penalties_per_coord + penalty_idx;
                let label = format!("score_warp_dev[z{coord}].penalty{penalty_idx}");
                spec.penalties[flat_idx] =
                    spec.penalties[flat_idx].clone().with_precision_label(label);
            }
        }
    }
    Ok(spec)
}

// ── Block layout ──────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct BlockSlices {
    pub(crate) time: std::ops::Range<usize>,
    pub(crate) marginal: std::ops::Range<usize>,
    pub(crate) logslope: std::ops::Range<usize>,
    pub(crate) score_warp: Option<std::ops::Range<usize>>,
    pub(crate) link_dev: Option<std::ops::Range<usize>>,
    /// Absorbed Stage-1 influence block (#461), trailing the flex blocks. Its
    /// width is `p₁` (the Stage-1 coefficient count), `None` when no CTN Stage-1
    /// chain produced an influence Jacobian.
    pub(crate) influence: Option<std::ops::Range<usize>>,
    pub(crate) total: usize,
}

/// Identifies one coefficient block of the survival marginal-slope joint
/// Hessian. The discriminant order *is* the coordinate layout order
/// (`time | marginal | logslope | score_warp? | link_dev?`), so it doubles as
/// the upper-triangle ordering used to pick the stored half of each symmetric
/// off-diagonal block. `ScoreWarp` and `LinkDev` are optional: they are absent
/// from the layout unless the corresponding flex deviation block is active.
///
/// This enum and [`BlockHessianAccumulator::block_view`] are the single source
/// of truth for the block layout. Every Hessian read-out (dense scatter,
/// matvec, bilinear form, diagonal extraction) is driven by them rather than
/// re-listing the fifteen blocks and their transpose relationships by hand, so
/// a layout change lands in exactly one place.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum HessBlock {
    Time,
    Marginal,
    Logslope,
    ScoreWarp,
    LinkDev,
    /// Absorbed Stage-1 influence block (#461). Placed LAST in the canonical
    /// layout so the existing `score_warp` (index 3) / `link_dev` (index 4)
    /// block-state positions are undisturbed; the absorber's coordinate range
    /// trails them and its β is dropped at predict.
    Influence,
}

impl HessBlock {
    /// Blocks in canonical coordinate-layout order. Iterating this array is how
    /// every assembler visits blocks; the order fixes floating-point
    /// accumulation order in the matvec / bilinear paths.
    pub(crate) const ALL: [HessBlock; 6] = [
        HessBlock::Time,
        HessBlock::Marginal,
        HessBlock::Logslope,
        HessBlock::ScoreWarp,
        HessBlock::LinkDev,
        HessBlock::Influence,
    ];
}

impl BlockSlices {
    /// Coordinate range occupied by `block`, or `None` when the (optional)
    /// flex block is inactive. `time`/`marginal`/`logslope` are always present.
    #[inline]
    pub(crate) fn range_of(&self, block: HessBlock) -> Option<std::ops::Range<usize>> {
        match block {
            HessBlock::Time => Some(self.time.clone()),
            HessBlock::Marginal => Some(self.marginal.clone()),
            HessBlock::Logslope => Some(self.logslope.clone()),
            HessBlock::ScoreWarp => self.score_warp.clone(),
            HessBlock::LinkDev => self.link_dev.clone(),
            HessBlock::Influence => self.influence.clone(),
        }
    }
}

pub(crate) fn block_slices(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
) -> BlockSlices {
    if !block_states.is_empty() {
        let expected_blocks = 3
            + usize::from(family.score_warp.is_some())
            + usize::from(family.link_dev.is_some())
            + usize::from(family.influence_absorber.is_some());
        assert_eq!(
            block_states.len(),
            expected_blocks,
            "survival marginal-slope block layout mismatch: expected {expected_blocks} blocks, got {}",
            block_states.len()
        );
    }
    let time = 0..family.design_entry.ncols();
    let marginal = time.end..time.end + family.marginal_design.ncols();
    let logslope = marginal.end..marginal.end + family.logslope_design.ncols();
    let mut cursor = logslope.end;
    let score_warp = family.score_warp.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim() * family.score_dim();
        cursor = range.end;
        range
    });
    let link_dev = family.link_dev.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    // Absorbed influence block trails the flex blocks: its width is the
    // Stage-1 coefficient count `p₁` (= `Z̃_infl.ncols()`).
    let influence = family.influence_absorber.as_ref().map(|z_tilde| {
        let range = cursor..cursor + z_tilde.ncols();
        cursor = range.end;
        range
    });
    let total = cursor;
    BlockSlices {
        time,
        marginal,
        logslope,
        score_warp,
        link_dev,
        influence,
        total,
    }
}

/// Owned scratch buffers backing a
/// [`crate::families::survival::marginal_slope::gpu::SurvivalFlexGpuRowInputs`] descriptor.
///
/// Built per-call by
/// [`SurvivalMarginalSlopeFamily::build_survival_flex_gpu_row_batch`];
/// callers hold the batch by value across the GPU `try_*` entry so the
/// borrowed slices returned by [`Self::as_inputs`] live for the dispatch.
pub(crate) struct SurvivalFlexGpuRowBatch {
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) q0: Vec<f64>,
    pub(crate) q1: Vec<f64>,
    pub(crate) qd1: Vec<f64>,
    pub(crate) z: Vec<f64>,
    pub(crate) g: Vec<f64>,
    pub(crate) beta: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    pub(crate) event: Vec<f64>,
}

impl SurvivalFlexGpuRowBatch {
    /// Borrow the buffers as a
    /// [`crate::families::survival::marginal_slope::gpu::SurvivalFlexGpuRowInputs`] descriptor.
    pub(crate) fn as_inputs<'a>(
        &'a self,
        family: &SurvivalMarginalSlopeFamily,
    ) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexGpuRowInputs<'a> {
        crate::families::survival::marginal_slope::gpu::SurvivalFlexGpuRowInputs {
            n: self.n,
            r: N_PRIMARY,
            p: self.p,
            score_dim: family.score_dim(),
            beta: &self.beta,
            q0: &self.q0,
            q1: &self.q1,
            qd1: &self.qd1,
            z: &self.z,
            g: &self.g,
            weights: &self.weights,
            event: &self.event,
            derivative_guard: family.derivative_guard,
            probit_scale: family.probit_frailty_scale(),
        }
    }
}

// ── Primary-space helpers ─────────────────────────────────────────────

// Primary scalar indices: 0=q0, 1=q1, 2=qd1, 3=g
pub(crate) const N_PRIMARY: usize = 4;

#[derive(Clone)]
pub(crate) struct FlexPrimarySlices {
    pub(crate) q0: usize,
    pub(crate) q1: usize,
    pub(crate) qd1: usize,
    pub(crate) g: usize,
    pub(crate) h: Option<std::ops::Range<usize>>,
    pub(crate) w: Option<std::ops::Range<usize>>,
    /// Single trailing primary index for the absorbed Stage-1 influence offset
    /// `o_infl` (#461). Unlike `g`/`h`/`w`, `o_infl` does NOT enter the de-nested
    /// calibration cells — it is a pure additive shift of the OBSERVED index η₁,
    /// so its only non-zero primary partial is `∂η₁/∂o_infl = 1` injected at the
    /// observed-timepoint reconstruction (cell-coefficient partials stay zero).
    pub(crate) infl: Option<usize>,
    pub(crate) total: usize,
}

/// Pack a private `SurvivalFlexTimepointExact` into the Block 10
/// pub-substrate input type so the shared CPU/GPU pure assembler in
/// `crate::families::survival::marginal_slope::gpu` can consume it without taking a
/// dependency on the family's private jet structs.
pub(crate) fn block10_pack_base(
    base: &SurvivalFlexTimepointExact,
) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBase {
    crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBase {
        eta: base.eta,
        chi: base.chi,
        d: base.d,
        eta_u: base.eta_u.to_vec(),
        eta_uv: base.eta_uv.iter().copied().collect(),
        chi_u: base.chi_u.to_vec(),
        chi_uv: base.chi_uv.iter().copied().collect(),
        d_u: base.d_u.to_vec(),
        d_uv: base.d_uv.iter().copied().collect(),
    }
}

pub(crate) fn block10_pack_dir(
    ext: &SurvivalFlexTimepointDirectionalExact,
) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointDirectional {
    crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointDirectional {
        eta_uv_dir: ext.eta_uv_dir.iter().copied().collect(),
        chi_u_dir: ext.chi_u_dir.to_vec(),
        chi_uv_dir: ext.chi_uv_dir.iter().copied().collect(),
        d_u_dir: ext.d_u_dir.to_vec(),
        d_uv_dir: ext.d_uv_dir.iter().copied().collect(),
    }
}

pub(crate) fn block10_pack_bi(
    bi: &SurvivalFlexTimepointBiDirectionalExact,
) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBiDirectional {
    crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBiDirectional {
        eta_uv_uv: bi.eta_uv_uv.iter().copied().collect(),
        chi_uv_uv: bi.chi_uv_uv.iter().copied().collect(),
        d_uv_uv: bi.d_uv_uv.iter().copied().collect(),
    }
}

pub(crate) fn flex_primary_slices(family: &SurvivalMarginalSlopeFamily) -> FlexPrimarySlices {
    let q0 = 0usize;
    let q1 = 1usize;
    let qd1 = 2usize;
    let g = 3usize;
    let mut cursor = 4usize;
    let h = family.score_warp.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim() * family.score_dim();
        cursor = range.end;
        range
    });
    let w = family.link_dev.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    // The absorber contributes a single primary scalar `o_infl` (trailing all
    // flex bases). Its full coefficient block lives in the `Influence`
    // ParameterBlockSpec; here it is one primary channel whose row-design is
    // `Z̃_infl[row,:]`, projected by `add_pullback`.
    let infl = family.influence_absorber.as_ref().map(|_| {
        let idx = cursor;
        cursor += 1;
        idx
    });
    FlexPrimarySlices {
        q0,
        q1,
        qd1,
        g,
        h,
        w,
        infl,
        total: cursor,
    }
}

pub(crate) fn flex_identity_block_pairs(
    primary: &FlexPrimarySlices,
    slices: &BlockSlices,
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let mut pairs = Vec::with_capacity(2);
    if let (Some(primary_range), Some(block_range)) =
        (primary.h.as_ref(), slices.score_warp.as_ref())
    {
        pairs.push((primary_range.clone(), block_range.clone()));
    }
    if let (Some(primary_range), Some(block_range)) = (primary.w.as_ref(), slices.link_dev.as_ref())
    {
        pairs.push((primary_range.clone(), block_range.clone()));
    }
    pairs
}
