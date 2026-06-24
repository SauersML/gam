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
    pub(crate) grad_logslope: Array1<f64>,
    pub(crate) hess_time: Array2<f64>,
    pub(crate) hess_marginal: Array2<f64>,
    pub(crate) hess_logslope: Array2<f64>,
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
            grad_logslope: Array1::zeros(slices.logslope.len()),
            hess_time: Array2::zeros((slices.time.len(), slices.time.len())),
            hess_marginal: Array2::zeros((slices.marginal.len(), slices.marginal.len())),
            hess_logslope: Array2::zeros((slices.logslope.len(), slices.logslope.len())),
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
        self.grad_logslope += &other.grad_logslope;
        self.hess_time += &other.hess_time;
        self.hess_marginal += &other.hess_marginal;
        self.hess_logslope += &other.hess_logslope;
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
                gradient: self.grad_logslope,
                hessian: SymmetricMatrix::Dense(self.hess_logslope),
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

impl DenestedCellPrimaryFixedPartials {
    /// Reconstruct the struct from the device-flat layout emitted by
    /// `crate::families::survival::marginal_slope::gpu_prep::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC`.
    ///
    /// Layout (per cell):
    ///
    /// ```text
    ///   dc_da[4], dc_daa[4], dc_daaa[4]                       // 12 doubles
    ///   coeff_u[r][4]                                          // 4r
    ///   coeff_au[r][4], coeff_bu[r][4]                         // 8r
    ///   coeff_aau[r][4], coeff_abu[r][4], coeff_bbu[r][4]      // 12r
    ///   coeff_aaau[r][4], coeff_aabu[r][4], coeff_abbu[r][4], coeff_bbbu[r][4]
    ///                                                          // 16r
    /// ```
    ///
    /// Total length: `12 + 40 * r`.
    pub(crate) fn from_flat_slice(flat: &[f64], r: usize) -> Result<Self, String> {
        let expected = 12 + 40 * r;
        if flat.len() != expected {
            return Err(format!(
                "DenestedCellPrimaryFixedPartials::from_flat_slice: expected {expected} doubles \
                 (12 + 40·r with r={r}), got {}",
                flat.len()
            ));
        }
        let read4 =
            |off: usize| -> [f64; 4] { [flat[off], flat[off + 1], flat[off + 2], flat[off + 3]] };
        let dc_da = read4(0);
        let dc_daa = read4(4);
        let dc_daaa = read4(8);
        let mut cursor = 12;
        let read_run = |start: usize| -> Vec<[f64; 4]> {
            let mut out = Vec::with_capacity(r);
            for slot in 0..r {
                let off = start + slot * 4;
                out.push([flat[off], flat[off + 1], flat[off + 2], flat[off + 3]]);
            }
            out
        };
        let coeff_u = read_run(cursor);
        cursor += 4 * r;
        let coeff_au = read_run(cursor);
        cursor += 4 * r;
        let coeff_bu = read_run(cursor);
        cursor += 4 * r;
        let coeff_aau = read_run(cursor);
        cursor += 4 * r;
        let coeff_abu = read_run(cursor);
        cursor += 4 * r;
        let coeff_bbu = read_run(cursor);
        cursor += 4 * r;
        let coeff_aaau = read_run(cursor);
        cursor += 4 * r;
        let coeff_aabu = read_run(cursor);
        cursor += 4 * r;
        let coeff_abbu = read_run(cursor);
        cursor += 4 * r;
        let coeff_bbbu = read_run(cursor);
        cursor += 4 * r;
        assert_eq!(cursor, expected);
        Ok(Self {
            dc_da,
            dc_daa,
            dc_daaa,
            coeff_u,
            coeff_au,
            coeff_bu,
            coeff_aau,
            coeff_abu,
            coeff_bbu,
            coeff_aaau,
            coeff_aabu,
            coeff_abbu,
            coeff_bbbu,
        })
    }
}

pub(crate) const COEFF_SUPPORT_GHW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: true,
    include_w: true,
};

pub(crate) const COEFF_SUPPORT_GW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: false,
    include_w: true,
};

/// Pre-computed partition cell data for a single timepoint evaluation.
/// Built once per (a, b, β_h, β_w) and reused across the three passes
/// (F, D, D_uv) that previously each rebuilt partition cells independently.
pub(crate) struct CachedPartitionCells {
    pub(crate) cells: Vec<CachedCellEntry>,
    pub(crate) calibration_f_a: f64,
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
    pub(crate) beta_h: Option<Array1<f64>>,
    pub(crate) beta_w: Option<Array1<f64>>,
    pub(crate) entry_cached: CachedPartitionCells,
    pub(crate) exit_cached: CachedPartitionCells,
    pub(crate) entry_base:
        crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBase,
    pub(crate) exit_base:
        crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBase,
}

pub(crate) struct CachedCellEntry {
    pub(crate) partition_cell: exact_kernel::DenestedPartitionCell,
    pub(crate) neg_cell: exact_kernel::DenestedCubicCell,
    pub(crate) state: exact_kernel::CellMomentState,
    pub(crate) fixed: DenestedCellPrimaryFixedPartials,
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

// #932-2 cutover: `SurvivalFlexTimepoint{Directional,BiDirectional}Exact` are the
// return shapes of the now test-only hand directional/bidirectional oracle
// producers; they moved to the test-masked `flex_oracle_structs_tests` module
// (consumed only by the `*_oracle_tests` hand oracle + the `tests.rs` FD witnesses),
// since the production contracted path reads the Block-10 packs straight from the
// `Jet3`/`Jet4` builders.

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

pub(crate) struct TimewiggleMarginalPsiRowLift {
    pub(crate) dir: Array1<f64>,
    pub(crate) u_q0_time: Array1<f64>,
    pub(crate) u_q1_time: Array1<f64>,
    pub(crate) u_qd1_time: Array1<f64>,
    pub(crate) u_q0_marginal: Array1<f64>,
    pub(crate) u_q1_marginal: Array1<f64>,
    pub(crate) u_qd1_marginal: Array1<f64>,
    pub(crate) x_entry_base: Array1<f64>,
    pub(crate) x_exit_base: Array1<f64>,
    pub(crate) x_deriv_base: Array1<f64>,
    pub(crate) marginal_row: Array1<f64>,
    pub(crate) entry_basis_d1: Array1<f64>,
    pub(crate) entry_basis_d2: Array1<f64>,
    pub(crate) exit_basis_d1: Array1<f64>,
    pub(crate) exit_basis_d2: Array1<f64>,
    pub(crate) exit_basis_d3: Array1<f64>,
    pub(crate) entry_m2: f64,
    pub(crate) entry_m3: f64,
    pub(crate) exit_m2: f64,
    pub(crate) exit_m3: f64,
    pub(crate) exit_m4: f64,
    pub(crate) d_raw: f64,
    pub(crate) mu: f64,
    pub(crate) psi_row: Array1<f64>,
}

/// Returns a reference to the static zero direction in primary space
/// (an `Array1::zeros(N_PRIMARY)`). Used by sigma-jet contractions to
/// avoid the per-call `Array1::zeros(primary_dim)` allocation storm in
/// `row_sigma_primary_terms`, which previously allocated 2-4 fresh zero
/// slots per kernel invocation and ~30 zero slots per row.
#[inline]
pub(crate) fn zero_primary_direction_ref() -> &'static Array1<f64> {
    use std::sync::OnceLock;
    static ZERO: OnceLock<Array1<f64>> = OnceLock::new();
    ZERO.get_or_init(|| Array1::<f64>::zeros(N_PRIMARY))
}

pub(crate) fn spatial_block_primary_loading(block_idx: usize) -> Result<Array1<f64>, String> {
    match block_idx {
        1 => Ok(Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0])),
        2 => Ok(Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0])),
        _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival marginal-slope spatial psi loading requested for unsupported block {block_idx}"
            ),
        }
        .into()),
    }
}

pub(crate) fn scalar_composite_bilinear(
    base: f64,
    da: f64,
    daa: f64,
    fixed_d1: f64,
    fixed_d2: f64,
    fixed_d12: f64,
    da_d1: f64,
    da_d2: f64,
    ad1: f64,
    ad2: f64,
    ad12: f64,
) -> MultiDirJet {
    MultiDirJet::bilinear(
        base,
        da * ad1 + fixed_d1,
        da * ad2 + fixed_d2,
        da * ad12 + daa * ad1 * ad2 + da_d1 * ad2 + da_d2 * ad1 + fixed_d12,
    )
}

pub(crate) fn coeff4_fixed_bilinear(
    base: &[f64; 4],
    d1: &[f64; 4],
    d2: &[f64; 4],
    d12: &[f64; 4],
) -> Vec<MultiDirJet> {
    (0..4)
        .map(|k| MultiDirJet::bilinear(base[k], d1[k], d2[k], d12[k]))
        .collect()
}

pub(crate) fn coeff4_composite_bilinear(
    base: &[f64; 4],
    da: &[f64; 4],
    daa: &[f64; 4],
    fixed_d1: &[f64; 4],
    fixed_d2: &[f64; 4],
    fixed_d12: &[f64; 4],
    da_d1: &[f64; 4],
    da_d2: &[f64; 4],
    ad1: f64,
    ad2: f64,
    ad12: f64,
) -> Vec<MultiDirJet> {
    (0..4)
        .map(|k| {
            scalar_composite_bilinear(
                base[k],
                da[k],
                daa[k],
                fixed_d1[k],
                fixed_d2[k],
                fixed_d12[k],
                da_d1[k],
                da_d2[k],
                ad1,
                ad2,
                ad12,
            )
        })
        .collect()
}

/// Derive a primary-space direction from a precomputed psi design row and beta,
/// avoiding a redundant psi design row build inside `row_primary_psi_direction`.
pub(crate) fn primary_direction_from_psi_row(
    block_idx: usize,
    psi_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    let value = psi_row.dot(beta_block);
    match block_idx {
        1 => {
            out[0] = value;
            out[1] = value;
        }
        2 => {
            out[3] = value;
        }
        _ => {}
    }
    out
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

pub(crate) fn primary_direction_from_psi_row_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
    psi_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(primary.total);
    let value = psi_row.dot(beta_block);
    match block_idx {
        1 => {
            out[primary.q0] = value;
            out[primary.q1] = value;
        }
        2 => {
            out[primary.g] = value;
        }
        _ => {}
    }
    out
}

/// Derive a primary-space psi action on a direction from a precomputed psi design row.
pub(crate) fn primary_psi_action_from_psi_row(
    block_idx: usize,
    psi_row: &Array1<f64>,
    d_beta_block: ndarray::ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    let value = psi_row.dot(&d_beta_block);
    match block_idx {
        1 => {
            out[0] = value;
            out[1] = value;
        }
        2 => {
            out[3] = value;
        }
        _ => {}
    }
    out
}

pub(crate) fn primary_psi_action_from_psi_row_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
    psi_row: &Array1<f64>,
    d_beta_block: ndarray::ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(primary.total);
    let value = psi_row.dot(&d_beta_block);
    match block_idx {
        1 => {
            out[primary.q0] = value;
            out[primary.q1] = value;
        }
        2 => {
            out[primary.g] = value;
        }
        _ => {}
    }
    out
}

/// Derive a primary-space second-order direction from a precomputed second psi design row.
pub(crate) fn primary_second_direction_from_psi_row(
    block_idx: usize,
    psi_second_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    primary_direction_from_psi_row(block_idx, psi_second_row, beta_block)
}

pub(crate) fn primary_second_direction_from_psi_row_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
    psi_second_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    primary_direction_from_psi_row_flex(primary, block_idx, psi_second_row, beta_block)
}

// ── Block-local Hessian accumulator ────────────────────────────────────
//
// Avoids O(n p²) per-row allocation of full p×p matrices by accumulating
// the 6 independent block matrices (3 diagonal + 3 off-diagonal) directly.
// Assembly to a dense p×p matrix or an implicit operator is a single O(p²)
// pass at the end, after the n-loop.
