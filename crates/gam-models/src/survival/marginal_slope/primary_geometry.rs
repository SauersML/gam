//! Primary-space geometry support: the dynamic-q blockwise accumulator, the
//! denested-cell fixed-partial layout, the exact time-wiggle / dynamic-row
//! geometry carriers, and the primary-direction / bilinear contraction
//! helpers that map psi rows into the four-axis primary frame.

use super::*;

#[derive(Clone)]
pub(crate) struct DynamicQBlockwiseAccumulator {
    pub(crate) log_likelihood: f64,
    pub(crate) grad_time: Array1<f64>,
    pub(crate) grad_marginal: Array1<f64>,
    pub(crate) grad_slope: Array1<f64>,
    pub(crate) hess_time: Array2<f64>,
    pub(crate) hess_marginal: Array2<f64>,
    pub(crate) hess_slope: Array2<f64>,
    pub(crate) grad_score_warp: Option<Array1<f64>>,
    pub(crate) hess_score_warp: Option<Array2<f64>>,
    pub(crate) grad_link_dev: Option<Array1<f64>>,
    pub(crate) hess_link_dev: Option<Array2<f64>>,
    /// Absorbed Stage-1 influence block (#461): the trailing block-diagonal
    /// grad/Hess over the `p₁` absorber coefficients `γ`, projected from the
    /// single `o_infl` primary scalar through the `Z̃_infl` design row.
    pub(crate) grad_influence: Option<Array1<f64>>,
    pub(crate) hess_influence: Option<Array2<f64>>,
}

impl DynamicQBlockwiseAccumulator {
    pub(crate) fn new(slices: &BlockSlices) -> Self {
        Self {
            log_likelihood: 0.0,
            grad_time: Array1::zeros(slices.time.len()),
            grad_marginal: Array1::zeros(slices.marginal.len()),
            grad_slope: Array1::zeros(slices.slope.len()),
            hess_time: Array2::zeros((slices.time.len(), slices.time.len())),
            hess_marginal: Array2::zeros((slices.marginal.len(), slices.marginal.len())),
            hess_slope: Array2::zeros((slices.slope.len(), slices.slope.len())),
            grad_score_warp: slices
                .score_warp
                .as_ref()
                .map(|range| Array1::zeros(range.len())),
            hess_score_warp: slices
                .score_warp
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
            grad_link_dev: slices
                .link_dev
                .as_ref()
                .map(|range| Array1::zeros(range.len())),
            hess_link_dev: slices
                .link_dev
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
            grad_influence: slices
                .influence
                .as_ref()
                .map(|range| Array1::zeros(range.len())),
            hess_influence: slices
                .influence
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
        }
    }

    pub(crate) fn add_assign(&mut self, other: &Self) {
        self.log_likelihood += other.log_likelihood;
        self.grad_time += &other.grad_time;
        self.grad_marginal += &other.grad_marginal;
        self.grad_slope += &other.grad_slope;
        self.hess_time += &other.hess_time;
        self.hess_marginal += &other.hess_marginal;
        self.hess_slope += &other.hess_slope;
        add_optional_vector(&mut self.grad_score_warp, &other.grad_score_warp);
        add_optional_vector(&mut self.grad_link_dev, &other.grad_link_dev);
        add_optional_vector(&mut self.grad_influence, &other.grad_influence);
        add_optional_matrix(&mut self.hess_score_warp, &other.hess_score_warp);
        add_optional_matrix(&mut self.hess_link_dev, &other.hess_link_dev);
        add_optional_matrix(&mut self.hess_influence, &other.hess_influence);
    }

    pub(crate) fn into_family_evaluation(self) -> FamilyEvaluation {
        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: self.grad_time,
                hessian: SymmetricMatrix::Dense(self.hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: self.grad_marginal,
                hessian: SymmetricMatrix::Dense(self.hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: self.grad_slope,
                hessian: SymmetricMatrix::Dense(self.hess_slope),
            },
        ];
        if let (Some(gradient), Some(hessian)) = (self.grad_score_warp, self.hess_score_warp) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (self.grad_link_dev, self.hess_link_dev) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (self.grad_influence, self.hess_influence) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        FamilyEvaluation {
            log_likelihood: self.log_likelihood,
            blockworking_sets,
        }
    }
}

pub(crate) struct DenestedCellPrimaryFixedPartials {
    pub(crate) dc_da: [f64; 4],
    pub(crate) dc_daa: [f64; 4],
    pub(crate) dc_daaa: [f64; 4],
    pub(crate) coeff_u: Vec<[f64; 4]>,
    pub(crate) coeff_au: Vec<[f64; 4]>,
    pub(crate) coeff_bu: Vec<[f64; 4]>,
    pub(crate) coeff_aau: Vec<[f64; 4]>,
    pub(crate) coeff_abu: Vec<[f64; 4]>,
    pub(crate) coeff_bbu: Vec<[f64; 4]>,
    pub(crate) coeff_aaau: Vec<[f64; 4]>,
    pub(crate) coeff_aabu: Vec<[f64; 4]>,
    pub(crate) coeff_abbu: Vec<[f64; 4]>,
    pub(crate) coeff_bbbu: Vec<[f64; 4]>,
}

/// Pre-computed calibration data for a single timepoint evaluation, built once
/// per (a, b, β_h, β_w) and reused across the three passes (F, D, D_uv) that
/// previously each rebuilt it independently: the Gaussian law's partition cells,
/// or the nodes of the declared finite law the fit anchors on (gam#2948).
pub(crate) enum CachedPartitionCells {
    Gaussian(Vec<CachedCellEntry>),
    /// The law's nodes, with the probit-frailty scale `s` the index carries.
    Law {
        nodes: Vec<CachedLawNode>,
        scale: f64,
    },
}

/// Direction-independent per-row state for the flex third-order contraction.
///
/// Built once per row by `build_row_flex_third_base_with_states` and reused
/// across every coefficient axis of a Jeffreys all-axes sweep so the intercept
/// solves, cached partitions, and exact base timepoints are paid once instead
/// of `p` times. See `row_flex_third_contract_from_base`.
pub(crate) struct FlexThirdRowBase {
    pub(crate) row: usize,
    pub(crate) p: usize,
    pub(crate) qd1: f64,
    pub(crate) q0: f64,
    pub(crate) q1: f64,
    pub(crate) q0_index: usize,
    pub(crate) q1_index: usize,
    pub(crate) a0: f64,
    pub(crate) a1: f64,
    pub(crate) g: f64,
    /// The absorbed-influence index offset `o_infl[row]` that the directional
    /// timepoints read, exactly as the single-direction contraction reads it.
    pub(crate) o_infl: f64,
    pub(crate) beta_h: Option<Array1<f64>>,
    pub(crate) beta_w: Option<Array1<f64>>,
    pub(crate) entry_cached: CachedPartitionCells,
    pub(crate) exit_cached: CachedPartitionCells,
    pub(crate) entry_base:
        crate::survival::marginal_slope::timepoint_exact::flex_jet::FlexTimepointBasePack,
    pub(crate) exit_base:
        crate::survival::marginal_slope::timepoint_exact::flex_jet::FlexTimepointBasePack,
}

pub(crate) struct CachedCellEntry {
    pub(crate) partition_cell: exact_kernel::DenestedPartitionCell,
    pub(crate) state: exact_kernel::CellMomentState,
    pub(crate) fixed: DenestedCellPrimaryFixedPartials,
}

/// One node `u_k` of a declared finite law at a timepoint's solved intercept `a`
/// and slope `b` (gam#2948). The de-nested index there is
/// `η_k = s·(U + b·h(u_k) + w(U))` with `U = a + b·u_k`: the score warp is read at
/// the fixed node, the link deviation at `U`, and each basis function enters
/// linearly in its own coefficient.
pub(crate) struct CachedLawNode {
    pub(crate) node: f64,
    pub(crate) weight: f64,
    /// `h(u_k)`, zero without a score warp.
    pub(crate) score_value: f64,
    /// `(primary axis, H_j(u_k))` for each score-warp basis function not zero at the node.
    pub(crate) score_basis: Vec<(usize, f64)>,
    /// `[w, w′, w″, w‴]` at `U`, zero without a link deviation.
    pub(crate) link_stack: [f64; 4],
    /// `(primary axis, [W_j, W_j′, W_j″, W_j‴])` at `U` for each link basis function
    /// not identically zero there.
    pub(crate) link_basis: Vec<(usize, [f64; 4])>,
}

pub(crate) struct SurvivalFlexTimepointExact {
    pub(crate) eta: f64,
    pub(crate) chi: f64,
    pub(crate) d: f64,
    pub(crate) eta_u: Array1<f64>,
    pub(crate) eta_uv: Array2<f64>,
    pub(crate) chi_u: Array1<f64>,
    pub(crate) chi_uv: Array2<f64>,
    pub(crate) d_u: Array1<f64>,
    pub(crate) d_uv: Array2<f64>,
}

pub(crate) struct SurvivalFlexTimepointFirstOrderExact {
    pub(crate) eta: f64,
    pub(crate) chi: f64,
    pub(crate) d: f64,
    pub(crate) eta_u: Array1<f64>,
    pub(crate) chi_u: Array1<f64>,
    pub(crate) d_u: Array1<f64>,
}

#[derive(Clone)]
pub(crate) struct SurvivalTimeWiggleGeometry {
    pub(crate) basis: Array2<f64>,
    pub(crate) basis_d1: Array2<f64>,
    pub(crate) basis_d2: Array2<f64>,
    pub(crate) basis_d3: Array2<f64>,
    pub(crate) basis_d4: Array2<f64>,
    pub(crate) dq_dq0: Array1<f64>,
    pub(crate) d2q_dq02: Array1<f64>,
    pub(crate) d3q_dq03: Array1<f64>,
    pub(crate) d4q_dq04: Array1<f64>,
    pub(crate) d5q_dq05: Array1<f64>,
}

#[derive(Clone)]
pub(crate) struct SurvivalTimeWiggleFirstOrderGeometry {
    pub(crate) basis: Array2<f64>,
    pub(crate) basis_d1: Array2<f64>,
    pub(crate) basis_d2: Array2<f64>,
    pub(crate) dq_dq0: Array1<f64>,
    pub(crate) d2q_dq02: Array1<f64>,
}

#[derive(Clone)]
pub(crate) struct SurvivalMarginalSlopeDynamicRowValues {
    pub(crate) q0: f64,
    pub(crate) q1: f64,
    pub(crate) qd1: f64,
}

#[derive(Clone)]
pub(crate) struct SurvivalMarginalSlopeDynamicRowGradient {
    pub(crate) q0: f64,
    pub(crate) q1: f64,
    pub(crate) qd1: f64,
    pub(crate) dq0_time: Array1<f64>,
    pub(crate) dq1_time: Array1<f64>,
    pub(crate) dqd1_time: Array1<f64>,
    pub(crate) dq0_marginal: Array1<f64>,
    pub(crate) dq1_marginal: Array1<f64>,
    pub(crate) dqd1_marginal: Array1<f64>,
}

#[derive(Clone)]
pub(crate) struct SurvivalMarginalSlopeDynamicRow {
    pub(crate) q0: f64,
    pub(crate) q1: f64,
    pub(crate) qd1: f64,
    pub(crate) dq0_time: Array1<f64>,
    pub(crate) dq1_time: Array1<f64>,
    pub(crate) dqd1_time: Array1<f64>,
    pub(crate) dq0_marginal: Array1<f64>,
    pub(crate) dq1_marginal: Array1<f64>,
    pub(crate) dqd1_marginal: Array1<f64>,
    pub(crate) d2q0_time_time: Array2<f64>,
    pub(crate) d2q1_time_time: Array2<f64>,
    pub(crate) d2qd1_time_time: Array2<f64>,
    pub(crate) d2q0_time_marginal: Array2<f64>,
    pub(crate) d2q1_time_marginal: Array2<f64>,
    pub(crate) d2qd1_time_marginal: Array2<f64>,
    pub(crate) d2q0_marginal_marginal: Array2<f64>,
    pub(crate) d2q1_marginal_marginal: Array2<f64>,
    pub(crate) d2qd1_marginal_marginal: Array2<f64>,
}

impl SurvivalMarginalSlopeDynamicRow {
    /// Construct a zero-sized workspace. Sizes are filled in lazily by
    /// [`reset`] on the first call to [`row_dynamic_q_geometry_into`].
    pub(crate) fn empty_workspace() -> Self {
        Self {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(0),
            dq1_time: Array1::zeros(0),
            dqd1_time: Array1::zeros(0),
            dq0_marginal: Array1::zeros(0),
            dq1_marginal: Array1::zeros(0),
            dqd1_marginal: Array1::zeros(0),
            d2q0_time_time: Array2::zeros((0, 0)),
            d2q1_time_time: Array2::zeros((0, 0)),
            d2qd1_time_time: Array2::zeros((0, 0)),
            d2q0_time_marginal: Array2::zeros((0, 0)),
            d2q1_time_marginal: Array2::zeros((0, 0)),
            d2qd1_time_marginal: Array2::zeros((0, 0)),
            d2q0_marginal_marginal: Array2::zeros((0, 0)),
            d2q1_marginal_marginal: Array2::zeros((0, 0)),
            d2qd1_marginal_marginal: Array2::zeros((0, 0)),
        }
    }

    /// Resize buffers to `(p_time, p_marginal)` and zero them in place.
    /// Reallocates only when the existing buffer shape differs from the
    /// requested shape; otherwise reuses the existing storage with
    /// `fill(0.0)` to keep the per-row allocator pressure flat.
    pub(crate) fn reset(&mut self, p_time: usize, p_marginal: usize) {
        self.q0 = 0.0;
        self.q1 = 0.0;
        self.qd1 = 0.0;
        reset_array1(&mut self.dq0_time, p_time);
        reset_array1(&mut self.dq1_time, p_time);
        reset_array1(&mut self.dqd1_time, p_time);
        reset_array1(&mut self.dq0_marginal, p_marginal);
        reset_array1(&mut self.dq1_marginal, p_marginal);
        reset_array1(&mut self.dqd1_marginal, p_marginal);
        reset_array2(&mut self.d2q0_time_time, p_time, p_time);
        reset_array2(&mut self.d2q1_time_time, p_time, p_time);
        reset_array2(&mut self.d2qd1_time_time, p_time, p_time);
        reset_array2(&mut self.d2q0_time_marginal, p_time, p_marginal);
        reset_array2(&mut self.d2q1_time_marginal, p_time, p_marginal);
        reset_array2(&mut self.d2qd1_time_marginal, p_time, p_marginal);
        reset_array2(&mut self.d2q0_marginal_marginal, p_marginal, p_marginal);
        reset_array2(&mut self.d2q1_marginal_marginal, p_marginal, p_marginal);
        reset_array2(&mut self.d2qd1_marginal_marginal, p_marginal, p_marginal);
    }
}

#[inline]
pub(crate) fn reset_array1(arr: &mut Array1<f64>, len: usize) {
    if arr.len() == len {
        arr.fill(0.0);
    } else {
        *arr = Array1::zeros(len);
    }
}

#[inline]
pub(crate) fn reset_array2(arr: &mut Array2<f64>, rows: usize, cols: usize) {
    if arr.shape() == [rows, cols] {
        arr.fill(0.0);
    } else {
        *arr = Array2::zeros((rows, cols));
    }
}

/// A design-moving ψ's loading onto the row program's primary space.
///
/// Sized by the family's own frame (`core_primary_dimension`), not by the
/// `N_PRIMARY` constant: a follow-up-varying slope runs the six-primary frame
/// `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)`, and a length-4 loading contracted against a
/// length-6 primary gradient is a shape error, not an approximation (#2765).
///
/// The marginal block loads onto `q₀` and `q₁` in EITHER frame — the location
/// index is what the frame does not change. The slope block is the one that
/// gains channels: with a time margin its three channel designs are
/// `X_cov ⊗ B_entry`, `X_cov ⊗ B_exit` and `X_cov ⊗ B′_exit`, and one loading
/// cannot represent a ψ that moves `X_cov`. [`psi_row_channels`] lifts such a ψ
/// onto the three channels from the layout's stored margin. A follow-up layout
/// without a stored margin has nothing to lift from, so a slope ψ is refused
/// here by name rather than lowered through one channel.
pub(crate) fn spatial_block_primary_loading(
    family: &SurvivalMarginalSlopeFamily,
    block_idx: usize,
) -> Result<Array1<f64>, String> {
    let mut out = Array1::<f64>::zeros(family.core_primary_dimension());
    match block_idx {
        1 => {
            out[PRIMARY_Q0] = 1.0;
            out[PRIMARY_Q1] = 1.0;
            Ok(out)
        }
        2 => {
            refuse_follow_up_varying_design_psi(family)?;
            out[PRIMARY_SLOPE] = 1.0;
            Ok(out)
        }
        _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival marginal-slope spatial psi loading requested for unsupported block {block_idx}"
            ),
        }
        .into()),
    }
}

/// A ψ that moves the SLOPE design cannot be lowered through a
/// follow-up-varying frame from a single `X_ψ`. See
/// [`spatial_block_primary_loading`].
fn refuse_follow_up_varying_design_psi(
    family: &SurvivalMarginalSlopeFamily,
) -> Result<(), String> {
    if family.slope_layout.is_follow_up_varying() {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: "a follow-up-varying slope carries three channel designs \
                     (X_cov ⊗ B_entry, X_cov ⊗ B_exit, X_cov ⊗ B′_exit), and this \
                     layout records no time margin to lift a covariate derivative \
                     onto them, so a spatial length scale on the slope surface \
                     cannot be lowered through this frame"
                .to_string(),
        }
        .into());
    }
    Ok(())
}

/// Derive a primary-space direction from a precomputed psi design row and beta,
/// avoiding a redundant psi design row build inside `row_primary_psi_direction`.
pub(crate) fn primary_direction_from_psi_row(
    family: &SurvivalMarginalSlopeFamily,
    block_idx: usize,
    psi_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let mut out = Array1::<f64>::zeros(family.core_primary_dimension());
    let value = psi_row.dot(beta_block);
    // Only blocks 1 and 2 carry a loading onto primary space (see
    // `spatial_block_primary_loading`); every other block leaves the direction
    // at zero.
    if block_idx == 1 {
        out[PRIMARY_Q0] = value;
        out[PRIMARY_Q1] = value;
    } else if block_idx == 2 {
        refuse_follow_up_varying_design_psi(family)?;
        out[PRIMARY_SLOPE] = value;
    }
    Ok(out)
}

/// A design-moving ψ's motion on one row: the design rows it moves, each with
/// the primary loading that row moves.
///
/// The marginal block moves `q₀` and `q₁` through one row and a time-constant
/// slope moves `g` through one row, so every configuration today carries exactly
/// one channel. Callers contract and pull back channel by channel, which is the
/// shape a slope whose covariate factor is tensored against a follow-up margin
/// needs: there one covariate derivative row moves three slope primaries through
/// three different design rows (gam#2767).
pub(crate) struct PsiRowChannels(Vec<(Array1<f64>, Array1<f64>)>);

impl PsiRowChannels {
    /// The `(primary loading, design row)` pairs, in primary order.
    pub(crate) fn channels(&self) -> &[(Array1<f64>, Array1<f64>)] {
        &self.0
    }

    /// `Σ_c L_c (x_c · v)`: the primary-space motion of a block coefficient
    /// vector `v` along this ψ axis.
    pub(crate) fn direction(&self, block_vector: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.0[0].0.len());
        for (loading, design_row) in &self.0 {
            out.scaled_add(design_row.dot(&block_vector), loading);
        }
        out
    }
}

/// The channels of one row's design ψ motion; see [`PsiRowChannels`].
///
/// On a slope tensored against a follow-up margin the ψ row is covariate-width,
/// and it lifts onto that row's entry, exit and exit-rate channel designs.
pub(crate) fn psi_row_channels(
    family: &SurvivalMarginalSlopeFamily,
    flex_primary: Option<&FlexPrimarySlices>,
    row: usize,
    block_idx: usize,
    psi_row: Array1<f64>,
) -> Result<PsiRowChannels, String> {
    if flex_primary.is_none()
        && block_idx == 2
        && let Some(margin) = family.slope_layout.time_margin()
    {
        let dimension = family.core_primary_dimension();
        let unit = |primary: usize| {
            let mut loading = Array1::<f64>::zeros(dimension);
            loading[primary] = 1.0;
            loading
        };
        let [entry, exit, rate] = margin.lift_row(row, &psi_row);
        return Ok(PsiRowChannels(vec![
            (unit(PRIMARY_SLOPE), entry),
            (unit(PRIMARY_SLOPE_EXIT), exit),
            (unit(PRIMARY_SLOPE_RATE), rate),
        ]));
    }
    let loading = match flex_primary {
        Some(primary) => spatial_block_primary_loading_flex(primary, block_idx)?,
        None => spatial_block_primary_loading(family, block_idx)?,
    };
    Ok(PsiRowChannels(vec![(loading, psi_row)]))
}

pub(crate) fn spatial_block_primary_loading_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
) -> Result<Array1<f64>, String> {
    let mut out = Array1::<f64>::zeros(primary.total);
    match block_idx {
        1 => {
            out[primary.q0] = 1.0;
            out[primary.q1] = 1.0;
            Ok(out)
        }
        2 => {
            out[primary.g] = 1.0;
            Ok(out)
        }
        _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival marginal-slope spatial psi loading requested for unsupported flex block {block_idx}"
            ),
        }
        .into()),
    }
}

// ── Block-local Hessian accumulator ────────────────────────────────────
//
// Avoids O(n p²) per-row allocation of full p×p matrices by accumulating
// the 6 independent block matrices (3 diagonal + 3 off-diagonal) directly.
// Assembly to a dense p×p matrix or an implicit operator is a single O(p²)
// pass at the end, after the n-loop.
