use super::*;

use crate::outer_subsample::{ARROW_ROW_CHUNK, arrow_row_chunk_count};
use gam_math::jet_scalar::{
    DynamicJetArena, DynamicOneSeed, DynamicOrder2, DynamicTwoSeed, JetScalar,
    Order2AtomChannels, RuntimeJetScalar,
};
use gam_row_macros::{row_atom, row_program};

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurvivalExactRowKernel {
    pub(crate) w: f64,
    pub(crate) d: f64,
    pub(crate) log_s0: f64,
    pub(crate) r0: f64,
    pub(crate) dr0: f64,
    pub(crate) ddr0: f64,
    pub(crate) dddr0: f64,
    pub(crate) log_s1: f64,
    pub(crate) r1: f64,
    pub(crate) dr1: f64,
    pub(crate) ddr1: f64,
    pub(crate) dddr1: f64,
    pub(crate) logphi1: f64,
    pub(crate) dlogphi1: f64,
    pub(crate) d2logphi1: f64,
    pub(crate) d3logphi1: f64,
    pub(crate) d4logphi1: f64,
    /// Stable pair value `log f(u1) − log S(u0)` (exit event density against
    /// entry survival). In the far tail both log stacks are astronomically
    /// large (|log| up to ~1e300) while their true difference is moderate —
    /// for a left-truncated event row with entry == exit it is exactly
    /// `log hazard(u0)` — so the naked stack difference is pure roundoff
    /// (#2335). Built cancellation-free per link at kernel construction.
    pub(crate) log_pdf1_minus_log_s0: f64,
    /// Stable pair value `log S(u1) − log S(u0)` (censored exit against entry
    /// survival), same construction.
    pub(crate) log_s1_minus_log_s0: f64,
    pub(crate) log_g: f64,
    pub(crate) d_log_g: f64,
    pub(crate) d2_log_g: f64,
    pub(crate) d3_log_g: f64,
    pub(crate) d4_log_g: f64,
}

/// The fifth derivative of each residual-distribution stack the row kernel
/// carries through the fourth, at the same indices and under the same
/// derivative log-rescale: `−(log S)⁽⁵⁾` at the entry and exit indices, the
/// log-density's fifth derivative at the exit index, and the guarded `log g`'s
/// fifth derivative. Only the fifth-order row program reads them, so no
/// lower-order path evaluates them (#2677).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlsFifthEntries {
    pub(crate) ddddr0: f64,
    pub(crate) ddddr1: f64,
    pub(crate) d5logphi1: f64,
    pub(crate) d5_log_g: f64,
}

/// Render a shared barrier-step refusal into the location-scale error
/// vocabulary, preserving the distinction the shared rule draws: a width
/// disagreement is a dimension fault, everything else is a statement about the
/// constraint geometry at the current iterate.
fn map_barrier_step_error(
    block: &str,
    error: gam_problem::ContractFeasibleStepError,
) -> String {
    let reason = format!("survival location-scale {block} step: {error}");
    match error {
        gam_problem::ContractFeasibleStepError::Dimension { .. } => {
            SurvivalLocationScaleError::DimensionMismatch { reason }.into()
        }
        _ => SurvivalLocationScaleError::ConstraintViolation { reason }.into(),
    }
}

/// Mix event and censored contributions, avoiding `0 * Inf = NaN` when
/// `d ∈ {0, 1}` and one branch is non-finite.
#[inline]
pub(crate) fn event_mix(d: f64, event_val: f64, censored_val: f64) -> f64 {
    if d == 1.0 {
        event_val
    } else if d == 0.0 {
        censored_val
    } else {
        d * event_val + (1.0 - d) * censored_val
    }
}

/// Build the row predictor state with possibly distinct entry/exit
/// evaluations of threshold and sigma.
///
/// `h0` and `h1` are the time transform's scaled shares of the residuals
/// (`hs_*` in [`SurvivalDynamicGeometry`]). `d_raw` and `qdot1` are the time
/// transform's and the location channel's shares of the unscaled event
/// Jacobian `ĝ` (`time_rate_exit` and `qdot_exit`), and `log_rate_scale` is
/// `−eta_ls` at exit. `entry_active` is false for a row entering at the origin,
/// which carries no `S(entry)` factor. For time-invariant blocks, the caller
/// passes the same value for both entry and exit.
#[inline]
pub(crate) fn survival_predictor_state(
    h0: f64,
    h1: f64,
    d_raw: f64,
    q0: f64,
    q1: f64,
    qdot1: f64,
    log_rate_scale: f64,
    entry_active: bool,
) -> SurvivalPredictorState {
    let g_diff = compensated_difference(d_raw, -qdot1);
    SurvivalPredictorState {
        h0,
        h1,
        g: g_diff.value,
        q0,
        q1,
        g_roundoff_slack: g_diff.roundoff_slack,
        g_operand_scale: g_diff.operand_scale,
        log_rate_scale,
        entry_active,
    }
}

impl SurvivalExactRowKernel {
    #[inline]
    pub(crate) fn log_likelihood(self) -> f64 {
        // `mix(d, a, b) − c == mix(d, a − c, b − c)` exactly for d ∈ {0, 1}
        // (and to rounding for fractional d); the pre-paired differences keep
        // the far-tail value from cancelling (#2335).
        self.w
            * event_mix(
                self.d,
                self.log_pdf1_minus_log_s0 + self.log_g,
                self.log_s1_minus_log_s0,
            )
    }

    /// The row log-likelihood at `state`. The kernel's rate stack is taken at
    /// the unscaled `ĝ`, so the event log-density adds the scale's
    /// `log_rate_scale` here (#2695).
    #[inline]
    pub(crate) fn log_likelihood_at(self, state: &SurvivalPredictorState) -> f64 {
        self.log_likelihood() + self.w * self.d * state.log_rate_scale
    }
}

/// `softplus(a) − softplus(b)`, cancellation-free for large same-sign
/// arguments where the naked difference of two ~|a| softplus values would
/// round away the O(a−b) result.
#[inline]
fn softplus_diff(a: f64, b: f64) -> f64 {
    if a >= 0.0 && b >= 0.0 {
        // softplus(x) = x + ln(1 + e^{−x}) on x ≥ 0; the log terms are ≤ ln 2.
        (a - b) + ((-a).exp().ln_1p() - (-b).exp().ln_1p())
    } else if a <= 0.0 && b <= 0.0 {
        // softplus(x) = ln(1 + e^{x}) on x ≤ 0; both values are ≤ ln 2.
        a.exp().ln_1p() - b.exp().ln_1p()
    } else {
        // Mixed signs: both magnitudes are bounded by max(|a|,|b|) + 1 and the
        // difference is at least min(|a|,|b|)-scale, so the naked form is fine.
        softplus(a) - softplus(b)
    }
}

pub(crate) struct SurvivalJointPsiDirection {
    pub(crate) x_t_exit_psi: Option<Array2<f64>>,
    pub(crate) x_t_entry_psi: Option<Array2<f64>>,
    pub(crate) x_t_deriv_psi: Option<Array2<f64>>,
    pub(crate) x_ls_exit_psi: Option<Array2<f64>>,
    pub(crate) x_ls_entry_psi: Option<Array2<f64>>,
    pub(crate) x_ls_deriv_psi: Option<Array2<f64>>,
    pub(crate) z_t_exit_psi: Array1<f64>,
    pub(crate) z_t_entry_psi: Array1<f64>,
    pub(crate) z_t_deriv_psi: Array1<f64>,
    pub(crate) z_ls_exit_psi: Array1<f64>,
    pub(crate) z_ls_entry_psi: Array1<f64>,
    pub(crate) z_ls_deriv_psi: Array1<f64>,
    pub(crate) x_t_exit_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_t_entry_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_t_deriv_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_ls_exit_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_ls_entry_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_ls_deriv_action: Option<CustomFamilyPsiDesignAction>,
}

pub(crate) fn split_survival_psi_design(
    x_psi: &Array2<f64>,
    n: usize,
    time_varying: bool,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>, Option<Array2<f64>>), String> {
    if time_varying {
        if x_psi.nrows() != 2 * n && x_psi.nrows() != 3 * n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} stacked psi design row mismatch: got {}, expected {} or {}",
                    x_psi.nrows(),
                    2 * n,
                    3 * n,
                ),
            }
            .into());
        }
        Ok((
            x_psi.slice(s![0..n, ..]).to_owned(),
            x_psi.slice(s![n..2 * n, ..]).to_owned(),
            (x_psi.nrows() == 3 * n).then(|| x_psi.slice(s![2 * n..3 * n, ..]).to_owned()),
        ))
    } else {
        if x_psi.nrows() != n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} psi design row mismatch: got {}, expected {}",
                    x_psi.nrows(),
                    n
                ),
            }
            .into());
        }
        Ok((x_psi.clone(), x_psi.clone(), None))
    }
}

impl SurvivalJointPsiDirection {
    fn channel_dense(&self, channel: usize) -> Option<&Array2<f64>> {
        match channel {
            3 => self.x_t_exit_psi.as_ref(),
            4 => self.x_t_entry_psi.as_ref(),
            5 => self.x_t_deriv_psi.as_ref(),
            6 => self.x_ls_exit_psi.as_ref(),
            7 => self.x_ls_entry_psi.as_ref(),
            8 => self.x_ls_deriv_psi.as_ref(),
            _ => None,
        }
    }

    fn channel_action(&self, channel: usize) -> Option<&CustomFamilyPsiDesignAction> {
        match channel {
            3 => self.x_t_exit_action.as_ref(),
            4 => self.x_t_entry_action.as_ref(),
            5 => self.x_t_deriv_action.as_ref(),
            6 => self.x_ls_exit_action.as_ref(),
            7 => self.x_ls_entry_action.as_ref(),
            8 => self.x_ls_deriv_action.as_ref(),
            _ => None,
        }
    }

    fn channel_row(
        &self,
        channel: usize,
        row: usize,
    ) -> Result<Option<Array1<f64>>, gam_problem::CustomFamilyError> {
        if let Some(action) = self.channel_action(channel) {
            return action.row_vector(row).map(Some);
        }
        Ok(self
            .channel_dense(channel)
            .map(|design| design.row(row).to_owned()))
    }

    fn primary_direction(&self, row: usize) -> [f64; SLS_ROW_K] {
        [
            0.0,
            0.0,
            0.0,
            self.z_t_exit_psi[row],
            self.z_t_entry_psi[row],
            self.z_t_deriv_psi[row],
            self.z_ls_exit_psi[row],
            self.z_ls_entry_psi[row],
            self.z_ls_deriv_psi[row],
        ]
    }

    fn jacobian_action(
        &self,
        row: usize,
        d_beta: &[f64],
        offsets: &[usize],
    ) -> Result<[f64; SLS_ROW_K], String> {
        let mut out = [0.0; SLS_ROW_K];
        for channel in 3..SLS_ROW_K {
            let Some(design_row) = self.channel_row(channel, row).map_err(|error| error.to_string())? else {
                continue;
            };
            let block = if channel <= 5 { 1 } else { 2 };
            out[channel] = design_row.dot(&ArrayView1::from(
                &d_beta[offsets[block]..offsets[block + 1]],
            ));
        }
        Ok(out)
    }
}

/// Number of linear-predictor primary channels for the survival
/// location-scale row kernel (non-wiggle configurations).
///
/// The row likelihood `ell = w[d(log f(u1)+log g) + (1-d)log S(u1) - log S(u0)]`
/// depends on three indices: the standardized residuals
/// `u = (h − eta_t)·exp(-eta_ls)` at entry and exit, and their time derivative
/// `du1/dt = exp(-eta_ls)·g` at exit, with `g = ḣ − eta_t' − (h − eta_t)·eta_ls'`
/// the event Jacobian with the scale factored out. The scale divides the whole
/// residual, time transform included, so `log(du1/dt) = log g − eta_ls` carries
/// the `−eta_ls` Jacobian that identifies σ(x) (#2695). That term is linear in a
/// primary, so the index `g` itself never sees the scale and its log stack stays
/// representable however extreme `exp(-eta_ls)` is. We make the model's linear
/// predictors the primary space so the row Jacobian is fixed (the `RowKernel`
/// framework requires this), and fold the nonlinear scale map into the per-row
/// kernel. The nine channels are:
///
/// | idx | predictor       | design                              | feeds |
/// |-----|-----------------|-------------------------------------|-------|
/// | 0   | h0  (time entry)| `time_jac_entry`                    | u0    |
/// | 1   | h1  (time exit) | `time_jac_exit`                     | u1, g |
/// | 2   | d_raw (time dot)| `time_jac_deriv`                    | g     |
/// | 3   | eta_t_exit      | `x_threshold`                       | u1, g |
/// | 4   | eta_t_entry     | `x_threshold_entry` (or threshold)  | u0    |
/// | 5   | eta_t_deriv     | `x_threshold_deriv` (or none)       | g     |
/// | 6   | eta_ls_exit     | `x_log_sigma`                       | u1, log-scale term |
/// | 7   | eta_ls_entry    | `x_log_sigma_entry` (or log_sigma)  | u0    |
/// | 8   | eta_ls_deriv    | `x_log_sigma_deriv` (or none)       | g     |
///
/// `H[a][b] = -Σ_i (ell_ii·D_i[a]·D_i[b] + ell_i·D2_i[a][b])` is lowered by
/// [`SurvivalLocationScaleFamily::survival_ls_coefficient_hessian`] through the
/// 24 structurally live upper-triangle pairs; the linear log-scale term has no
/// curvature. Indices `i ∈ {u0,u1,g}` are functionally independent, so the
/// index-space derivative tensors are diagonal.
pub(crate) const SLS_ROW_K: usize = 9;
const SLS_U0_AXES: [usize; 3] = [0, 4, 7];
const SLS_U1_AXES: [usize; 3] = [1, 3, 6];
const SLS_G_AXES: [usize; 5] = [1, 2, 3, 5, 8];

/// `RowKernel<9>` adapter for the survival location-scale joint likelihood
/// (non-wiggle path). Holds the per-β quantities already computed by
/// [`SurvivalLocationScaleFamily::build_dynamic_geometry`]; every trait method
/// is a pure repackaging of those scalars into linear-predictor primary space,
/// so every coefficient-space target consumes the same row program by construction.
pub(crate) struct SurvivalLsRowKernel<'a> {
    pub(crate) family: &'a SurvivalLocationScaleFamily,
    pub(crate) dynamic: &'a SurvivalDynamicGeometry,
    pub(crate) deriv_log_scale: f64,
    /// Joint block offsets `[0, p_time, p_time+p_thr, p_total]` (3 blocks).
    pub(crate) offsets: Vec<usize>,
}

impl SurvivalLsRowKernel<'_> {
    /// Resolve the design for a threshold/log-sigma channel, falling back to the
    /// exit design when the entry/derivative variant is absent (time-invariant).
    #[inline]
    pub(crate) fn entry_design<'b>(
        opt: &'b Option<DesignMatrix>,
        fallback: &'b DesignMatrix,
    ) -> &'b DesignMatrix {
        opt.as_ref().unwrap_or(fallback)
    }

    /// Per-row dense design row for each channel within its coefficient block:
    /// returns `(block_index, row_vector)` for channels `0..9`. Used by the
    /// pullback / diagonal assembly. Channels with an absent derivative design
    /// (time-invariant derivative channels) return `None` and contribute
    /// nothing.
    pub(crate) fn channel_block(&self, ch: usize) -> Option<usize> {
        match ch {
            0 | 1 | 2 => Some(Self::THRESHOLD_BLOCK_TIME),
            3 | 4 | 5 => Some(Self::THRESHOLD_BLOCK_THR),
            6 | 7 | 8 => Some(Self::THRESHOLD_BLOCK_LS),
            _ => None,
        }
    }
    pub(crate) const THRESHOLD_BLOCK_TIME: usize = 0;
    pub(crate) const THRESHOLD_BLOCK_THR: usize = 1;
    pub(crate) const THRESHOLD_BLOCK_LS: usize = 2;

    /// Dense per-row design vector for `channel` (length = its block width), or
    /// `None` when the channel's design is absent (time-invariant deriv channel,
    /// which carries no coefficients of its own).
    pub(crate) fn channel_row(&self, ch: usize, row: usize) -> Option<Array1<f64>> {
        let fam = self.family;
        match ch {
            0 => Some(self.dynamic.time_jac_entry.row(row).to_owned()),
            1 => Some(self.dynamic.time_jac_exit.row(row).to_owned()),
            2 => Some(self.dynamic.time_jac_deriv.row(row).to_owned()),
            3 => Some(design_dense_row(&fam.x_threshold, row)),
            4 => Some(design_dense_row(
                Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold),
                row,
            )),
            5 => fam
                .x_threshold_deriv
                .as_ref()
                .map(|d| design_dense_row(d, row)),
            6 => Some(design_dense_row(&fam.x_log_sigma, row)),
            7 => Some(design_dense_row(
                Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma),
                row,
            )),
            8 => fam
                .x_log_sigma_deriv
                .as_ref()
                .map(|d| design_dense_row(d, row)),
            _ => None,
        }
    }

    /// Per-row cached `(coefficient_offset, dense_design_row)` for each of the
    /// nine primary channels, materialized ONCE so the batched directional
    /// override reuses it for every swept axis instead of re-running
    /// [`Self::channel_row`] for every `(row, axis)` pair. Channel `c`'s entry is
    /// `None` exactly when [`Self::channel_block`]`(c).zip(`[`Self::channel_row`]
    /// `(c, row))` is — i.e. the time-invariant derivative channels (5/8) whose
    /// design is absent — so the cached-pullback walk is structurally identical to
    /// [`Self::add_pullback_hessian`].
    fn cached_channel_rows(&self, row: usize) -> Vec<Option<(usize, Array1<f64>)>> {
        (0..SLS_ROW_K)
            .map(
                |ch| match (self.channel_block(ch), self.channel_row(ch, row)) {
                    (Some(blk), Some(r)) => Some((self.offsets[blk], r)),
                    _ => None,
                },
            )
            .collect()
    }

    /// `⟨H²dot[u, e_a], K_b⟩` for every direction `u`, axis `a` and symmetric kernel `K_b`,
    /// without forming any axis matrix: `consume(index, contractions)` receives the `p × p`
    /// matrix `contractions[[a, b]]` for `directions[index]`.
    ///
    /// Per row, `H²dot[u, e_a] = Σ_{x,y} T4[J·u, J·e_a]_{xy} r_x r_yᵀ` over the channel design
    /// rows `r_x`, so its kernel contraction is `Σ_{x,y} T4[J·u, J·e_a]_{xy} w^b_{xy}` with
    /// `w^b_{xy} = r_xᵀ K_b r_y`. The fourth contraction is linear in `J·e_a = Σ_c r_c[a]·e_c`,
    /// so each row forms the nine `T4[J·u, e_c]` once per direction and scatters
    /// `⟨T4[J·u, e_c], w^b⟩` through the channel rows `r_c`. The kernel products `w^b` do not
    /// depend on the direction and are formed once per row for the whole batch. Rows reduce in
    /// `ARROW_ROW_CHUNK` tiles combined in tile order; a row without an exact kernel
    /// (non-positive weight) contributes nothing, as in the per-axis fold.
    pub(crate) fn second_directional_axis_contractions_each(
        &self,
        directions: &[Array1<f64>],
        kernels: &[Array2<f64>],
        consume: &mut dyn FnMut(usize, Array2<f64>) -> Result<(), String>,
    ) -> Result<(), String> {
        let p = crate::row_kernel::RowKernel::<SLS_ROW_K>::n_coefficients(self);
        if kernels.len() != p || kernels.iter().any(|kernel| kernel.dim() != (p, p)) {
            return Err(format!(
                "survival location-scale axis contractions: expected {p} kernels of shape \
                 ({p}, {p}), got {}",
                kernels.len(),
            ));
        }
        let direction_slices = directions
            .iter()
            .map(|direction| {
                direction.as_slice().filter(|slice| slice.len() == p).ok_or_else(|| {
                    format!(
                        "survival location-scale axis contractions: a direction is not a \
                         contiguous vector of length {p}"
                    )
                })
            })
            .collect::<Result<Vec<&[f64]>, String>>()?;
        let n = gam_math::jet_tower::RowProgram::n_rows(self);
        let count = directions.len();
        let tiles = (0..arrow_row_chunk_count(n))
            .into_par_iter()
            .map(|chunk_idx| -> Result<Vec<Array2<f64>>, String> {
                let start = chunk_idx * ARROW_ROW_CHUNK;
                let end = (start + ARROW_ROW_CHUNK).min(n);
                let mut totals = vec![Array2::<f64>::zeros((p, p)); count];
                let mut kernel_times_row = vec![0.0_f64; p];
                for row in start..end {
                    let Some((primary, exact)) = self.row_nll_inputs_opt(row)? else {
                        continue;
                    };
                    let plan = sls_outer_plan::<5>(&exact);
                    let chans = self.cached_channel_rows(row);
                    let weights: Vec<[[f64; SLS_ROW_K]; SLS_ROW_K]> = kernels
                        .iter()
                        .map(|kernel| {
                            let mut w = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                            for (y, slot_y) in chans.iter().enumerate() {
                                let Some((off_y, row_y)) = slot_y.as_ref() else {
                                    continue;
                                };
                                for (i, value) in kernel_times_row.iter_mut().enumerate() {
                                    *value = row_y
                                        .iter()
                                        .enumerate()
                                        .map(|(ib, &entry)| kernel[[i, off_y + ib]] * entry)
                                        .sum();
                                }
                                for (x, slot_x) in chans.iter().enumerate() {
                                    let Some((off_x, row_x)) = slot_x.as_ref() else {
                                        continue;
                                    };
                                    w[x][y] = row_x
                                        .iter()
                                        .enumerate()
                                        .map(|(ia, &entry)| entry * kernel_times_row[off_x + ia])
                                        .sum();
                                }
                            }
                            w
                        })
                        .collect();
                    for (index, direction) in direction_slices.iter().enumerate() {
                        let direction_u = crate::row_kernel::RowKernel::<SLS_ROW_K>::jacobian_action(
                            self, row, direction,
                        );
                        let total = &mut totals[index];
                        for (c, slot_c) in chans.iter().enumerate() {
                            let Some((off_c, row_c)) = slot_c.as_ref() else {
                                continue;
                            };
                            let mut unit = [0.0_f64; SLS_ROW_K];
                            unit[c] = 1.0;
                            let fourth =
                                sls_row_fourth_generated_with_plan(&primary, &plan, &direction_u, &unit);
                            for (b, w) in weights.iter().enumerate() {
                                let mut value = 0.0_f64;
                                for x in 0..SLS_ROW_K {
                                    for y in 0..SLS_ROW_K {
                                        value += fourth[x][y] * w[x][y];
                                    }
                                }
                                if value == 0.0 {
                                    continue;
                                }
                                for (ia, &entry) in row_c.iter().enumerate() {
                                    total[[off_c + ia, b]] += entry * value;
                                }
                            }
                        }
                    }
                }
                Ok(totals)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut totals = vec![Array2::<f64>::zeros((p, p)); count];
        for tile in tiles {
            for (total, part) in totals.iter_mut().zip(tile) {
                *total += &part;
            }
        }
        for (index, total) in totals.into_iter().enumerate() {
            consume(index, total)?;
        }
        Ok(())
    }

    /// `vec(sym(Uᵀ I'[e_a] U))` for every coefficient axis `a` and a Jeffreys basis `U`
    /// (`p × r`), row-major, without forming any axis matrix.
    ///
    /// Per row, `I'[e_a] = Σ_{x,y} T3[J·e_a]_{xy} r_x r_yᵀ` over the channel design rows `r_x`,
    /// and `J·e_a = Σ_c r_c[a]·e_c`, so with `g_x = Uᵀ r_x`
    ///
    /// ```text
    ///   Uᵀ I'[e_a] U = Σ_row Σ_c r_c[a] · M_c,   M_c = Σ_{x,y} T3[e_c]_{xy} g_x g_yᵀ .
    /// ```
    ///
    /// Each row forms its nine `T3[e_c]` from one outer plan and projects its channel rows onto
    /// `U` once. Each channel then scatters its `M_c` rows through one design product per
    /// `ARROW_ROW_CHUNK` tile, and tiles combine in tile order. That is `K·p_b·r + K²·r + K·r²`
    /// row work per channel plus one `n × r²` product, against the dense route's `p` axis
    /// pullbacks of `K²·p_b²` per row (#2668). A row without an exact kernel (non-positive
    /// weight) contributes nothing, as in the per-axis fold.
    pub(crate) fn directional_derivative_rotated_all_axes(
        &self,
        basis: ndarray::ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let p = crate::row_kernel::RowKernel::<SLS_ROW_K>::n_coefficients(self);
        let (basis_rows, r) = basis.dim();
        if basis_rows != p {
            return Err(format!(
                "survival location-scale rotated axis rows: the basis has {basis_rows} rows for \
                 {p} coefficients"
            ));
        }
        let squared = r * r;
        let n = gam_math::jet_tower::RowProgram::n_rows(self);
        let tiles = (0..arrow_row_chunk_count(n))
            .into_par_iter()
            .map(|chunk_idx| -> Result<Array2<f64>, String> {
                let start = chunk_idx * ARROW_ROW_CHUNK;
                let end = (start + ARROW_ROW_CHUNK).min(n);
                let len = end - start;
                let stride = SLS_ROW_K * r;
                let mut chans = Vec::with_capacity(len);
                let mut thirds = vec![[[[0.0_f64; SLS_ROW_K]; SLS_ROW_K]; SLS_ROW_K]; len];
                let mut projected = vec![0.0_f64; len * stride];
                for row in start..end {
                    let local = row - start;
                    let row_chans = self.cached_channel_rows(row);
                    if let Some((primary, exact)) = self.row_nll_inputs_opt(row)? {
                        let plan = sls_outer_plan::<5>(&exact);
                        let g = &mut projected[local * stride..(local + 1) * stride];
                        for (c, slot) in row_chans.iter().enumerate() {
                            let Some((off_c, row_c)) = slot.as_ref() else {
                                continue;
                            };
                            let mut unit = [0.0_f64; SLS_ROW_K];
                            unit[c] = 1.0;
                            thirds[local][c] =
                                sls_row_third_generated_with_plan(&primary, &plan, &unit);
                            let g_c = &mut g[c * r..(c + 1) * r];
                            for (ib, &entry) in row_c.iter().enumerate() {
                                if entry == 0.0 {
                                    continue;
                                }
                                for (value, &u) in g_c.iter_mut().zip(basis.row(off_c + ib).iter())
                                {
                                    *value += entry * u;
                                }
                            }
                        }
                    }
                    chans.push(row_chans);
                }
                let mut total = Array2::<f64>::zeros((p, squared));
                let mut half_product = vec![0.0_f64; r];
                for c in 0..SLS_ROW_K {
                    let Some(block) = self.channel_block(c) else {
                        continue;
                    };
                    if chans.iter().all(|row_chans| row_chans[c].is_none()) {
                        continue;
                    }
                    let offset = self.offsets[block];
                    let width = self.offsets[block + 1] - offset;
                    let mut design = Array2::<f64>::zeros((len, width));
                    let mut weighted = Array2::<f64>::zeros((len, squared));
                    let weighted_values = weighted
                        .as_slice_mut()
                        .expect("row-weighted basis products are contiguous");
                    for (local, row_chans) in chans.iter().enumerate() {
                        let Some(row_c) = row_chans[c].as_ref().map(|slot| &slot.1) else {
                            continue;
                        };
                        if row_c.len() != width {
                            return Err(format!(
                                "survival location-scale rotated axis rows: channel {c} has a \
                                 design row of width {} in a block of width {width}",
                                row_c.len()
                            ));
                        }
                        design.row_mut(local).assign(row_c);
                        let third = &thirds[local][c];
                        let g = &projected[local * stride..(local + 1) * stride];
                        let output = &mut weighted_values[local * squared..(local + 1) * squared];
                        for x in 0..SLS_ROW_K {
                            half_product.fill(0.0);
                            for y in 0..SLS_ROW_K {
                                let entry = third[x][y];
                                if entry == 0.0 {
                                    continue;
                                }
                                for (value, &g_y) in half_product.iter_mut().zip(&g[y * r..(y + 1) * r])
                                {
                                    *value += entry * g_y;
                                }
                            }
                            for s in 0..r {
                                let g_xs = g[x * r + s];
                                if g_xs == 0.0 {
                                    continue;
                                }
                                for (value, &h_t) in
                                    output[s * r..(s + 1) * r].iter_mut().zip(&half_product)
                                {
                                    *value += g_xs * h_t;
                                }
                            }
                        }
                    }
                    let moments = fast_atb_with_parallelism(&design, &weighted, faer::Par::Seq);
                    let mut target = total.slice_mut(s![offset..offset + width, ..]);
                    target += &moments;
                }
                Ok(total)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut rows = Array2::<f64>::zeros((p, squared));
        for tile in tiles {
            rows += &tile;
        }
        if squared > 0 {
            let values = rows
                .as_slice_mut()
                .expect("rotated axis rows are contiguous");
            for axis in values.chunks_exact_mut(squared) {
                for s in 0..r {
                    for t in (s + 1)..r {
                        let average = 0.5 * (axis[s * r + t] + axis[t * r + s]);
                        axis[s * r + t] = average;
                        axis[t * r + s] = average;
                    }
                }
            }
        }
        Ok(rows)
    }

    pub(crate) fn row_primary_values(&self, row: usize) -> [f64; SLS_ROW_K] {
        let inv_sigma_exit = self.dynamic.inv_sigma_exit[row];
        let eta_t_exit = -self.dynamic.q_base_exit[row] / inv_sigma_exit;
        [
            self.dynamic.h_entry[row],
            self.dynamic.h_exit[row],
            self.dynamic.hdot_exit[row],
            eta_t_exit,
            -self.dynamic.q_base_entry[row] / self.dynamic.inv_sigma_entry[row],
            self.dynamic.eta_t_deriv_exit[row],
            self.dynamic.eta_ls_exit[row],
            self.dynamic.eta_ls_entry[row],
            self.dynamic.eta_ls_deriv_exit[row],
        ]
    }

    /// The row's exact f64 derivative-stack kernel and the nine primary values
    /// `p` — the scalar-independent inputs the generic row NLL
    /// ([`sls_row_nll`]) consumes. Computed once per row; reused across every
    /// `JetScalar` instantiation (value/grad/Hessian, contracted third/fourth).
    fn row_nll_inputs(
        &self,
        row: usize,
    ) -> Result<([f64; SLS_ROW_K], SurvivalExactRowKernel), String> {
        self.row_nll_inputs_opt(row)?
            .ok_or_else(|| format!("survival location-scale row {row} has no exact kernel"))
    }

    /// Like [`Self::row_nll_inputs`] but returns `Ok(None)` for rows whose
    /// observation weight is non-positive. A positive-weight row whose exact
    /// derivatives cannot be represented is an error, never a zero
    /// contribution.
    fn row_nll_inputs_opt(
        &self,
        row: usize,
    ) -> Result<Option<([f64; SLS_ROW_K], SurvivalExactRowKernel)>, String> {
        let p = self.row_primary_values(row);
        let state = self.family.row_predictor_state_at(self.dynamic, row);
        let kernel = self
            .family
            .exact_row_kernel_rescaled(row, state, self.deriv_log_scale)?;
        Ok(kernel.map(|k| (p, k)))
    }

    /// [`Self::row_nll_inputs_opt`] with each residual-distribution stack's
    /// fifth derivative at the same predictor state. A link without a
    /// closed-form fifth stack is an error: the family declares the fifth-order
    /// channel unavailable for it, so no consumer reaches here.
    fn row_nll_fifth_inputs_opt(
        &self,
        row: usize,
    ) -> Result<Option<([f64; SLS_ROW_K], SurvivalExactRowKernel, SlsFifthEntries)>, String> {
        let p = self.row_primary_values(row);
        let state = self.family.row_predictor_state_at(self.dynamic, row);
        let fifth = SurvivalLocationScaleFamily::exact_row_kernel_fifth_from_parts(
            &self.family.inverse_link,
            self.family.derivative_guard,
            &state,
            self.deriv_log_scale,
        )
        .ok_or_else(|| {
            format!(
                "survival location-scale row {row}: the inverse link has no closed-form \
                 fifth derivative stack"
            )
        })?;
        let kernel = self
            .family
            .exact_row_kernel_rescaled(row, state, self.deriv_log_scale)?;
        Ok(kernel.map(|k| (p, k, fifth)))
    }
}

/// The survival location-scale row negative log-likelihood, written ONCE over a
/// generic [`JetScalar<9>`] so a single expression yields every derivative
/// channel a consumer needs:
///
/// * `S = Tower4<9>` → full `(v, g, H, t3, t4)` (the all-channels oracle path,
///   [`SurvivalLsRowKernel::row_nll_tower`]),
/// * `S = OneSeed<9>` → the contracted third `Σ_c ℓ_{abc} dir_c` (1.46 KiB/row,
///   the `RowKernel::row_third_contracted` directional path),
/// * `S = TwoSeed<9>` → the contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d`
///   (2.8 KiB/row, the `RowKernel::row_fourth_contracted` path).
///
/// The value/gradient/Hessian consumer lowers the same three scalar indices
/// `(u0,u1,g)` through one axis-mapped order-two scatter. Each index is differentiated
/// in its natural 3/3/5-dimensional support and scattered through a literal
/// axis map, so no dense 9×9 intermediates or runtime dependency masks survive.
///
/// The nine primary channels are `(h_entry, h_exit, hdot_exit, eta_t_exit,
/// eta_t_entry, eta_t_deriv, eta_ls_exit, eta_ls_entry, eta_ls_deriv)` — see
/// [`SurvivalLsRowKernel::row_primary_values`]. From them the survival index
/// quantities are
///   `u0 = (h_entry − eta_t_entry)·e^{−eta_ls_entry}`  (entry / left-truncation),
///   `u1 = (h_exit  − eta_t_exit) ·e^{−eta_ls_exit}`   (exit),
///   `g  = e^{−eta_ls_exit}·(hdot_exit − eta_t_deriv − (h_exit − eta_t_exit)·eta_ls_deriv)`
/// (the event log-density's Jacobian factor, `du1/dt`), and the NLL is
///   `w[ logS0(u0) − (1−d)·logS1(u1) − d·(logφ1(u1) + log g(g)) ]`,
/// each residual-distribution stack `logS/logφ/log g` supplied as a hand-certified
/// `[f64; 5]` derivative stack on the kernel and entered through
/// [`JetScalar::compose_unary`]. There is exactly one source for value and every
/// derivative order (the #736/#932 single-source contract).
#[derive(Clone, Copy)]
struct SlsOuterPlan<const ORDER: usize> {
    u0: [f64; ORDER],
    u1: Option<[f64; ORDER]>,
    g: Option<[f64; ORDER]>,
    /// The NLL's coefficient on `eta_ls_exit` from the event log-density's
    /// `log(du1/dt) = log g − eta_ls` (#2695): the event weight wherever the rate
    /// stack is live, zero otherwise. It is linear, so only the value and the
    /// gradient see it.
    g_log_scale: f64,
}

/// Exactly the six diagonal index-space NLL channels consumed by the
/// inner-Newton update: orders one and two for `(u0, u1, g)`. The channel
/// count is encoded in the array widths, so no unconsumed higher order can be
/// materialized or accidentally consumed.
#[derive(Clone, Copy, Debug)]
struct SlsIndexDerivativeChannels {
    gradient: [f64; 3],
    hessian_diagonal: [f64; 3],
}

/// Project one derivative order from the canonical outer stacks. Both the
/// number of live indices and the derivative order are compile-time constants;
/// optimized code is a fixed set of scalar loads with no dense jet storage.
#[inline(always)]
fn project_index_diagonal<const CHANNELS: usize, const ORDER: usize>(
    stacks: &[[f64; 5]; 3],
) -> [f64; CHANNELS] {
    assert!(CHANNELS <= stacks.len());
    assert!(ORDER < stacks[0].len());
    std::array::from_fn(|index| stacks[index][ORDER])
}

impl SlsOuterPlan<5> {
    /// Mechanically lower the canonical `(u0, u1, g)` outer derivative stacks
    /// to the sparse diagonal channels read by the inner-Newton consumer.
    /// Inactive event/censoring branches are structural zero stacks, while the
    /// active `u1` stack retains [`sls_outer_plan`]'s censored-then-event
    /// accumulation order. No derivative formula exists in this lowering.
    #[inline(always)]
    fn lower_index_derivative_channels(self) -> SlsIndexDerivativeChannels {
        let stacks = [
            self.u0,
            self.u1.unwrap_or([0.0; 5]),
            self.g.unwrap_or([0.0; 5]),
        ];
        SlsIndexDerivativeChannels {
            gradient: project_index_diagonal::<3, 1>(&stacks),
            hessian_diagonal: project_index_diagonal::<3, 2>(&stacks),
        }
    }
}

#[inline(always)]
fn add_scaled_stack<const ORDER: usize>(
    target: &mut [f64; ORDER],
    stack: [f64; 6],
    scale: f64,
) {
    for i in 0..ORDER {
        target[i] += scale * stack[i];
    }
}

/// Collapse the row's additive unary terms by scalar index. The censoring and
/// event transforms of `u1` share the same inner index, so linearity lets the
/// compiler combine their derivative stacks before one Faà di Bruno pass.
#[inline(always)]
fn sls_outer_plan<const ORDER: usize>(
    kernel: &SurvivalExactRowKernel,
) -> SlsOuterPlan<ORDER> {
    assert!(ORDER <= 5);
    // Through the fourth derivative no stack's sixth entry is read, so the
    // accumulation is the fifth-order one with nothing in that entry.
    sls_outer_plan_with_fifth(
        kernel,
        &SlsFifthEntries {
            ddddr0: 0.0,
            ddddr1: 0.0,
            d5logphi1: 0.0,
            d5_log_g: 0.0,
        },
    )
}

/// [`sls_outer_plan`] through the fifth derivative: the same accumulation in
/// the same order, each stack's sixth entry read from `fifth`. The fifth-order
/// row program is its only consumer.
#[inline(always)]
fn sls_outer_plan_with_fifth<const ORDER: usize>(
    kernel: &SurvivalExactRowKernel,
    fifth: &SlsFifthEntries,
) -> SlsOuterPlan<ORDER> {
    assert!(ORDER <= 6);
    let mut u0 = [0.0; ORDER];
    add_scaled_stack(
        &mut u0,
        [
            kernel.log_s0,
            -kernel.r0,
            -kernel.dr0,
            -kernel.ddr0,
            -kernel.dddr0,
            -fifth.ddddr0,
        ],
        kernel.w,
    );

    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let mut u1 = [0.0; ORDER];
    if censored_weight != 0.0 {
        add_scaled_stack(
            &mut u1,
            [
                kernel.log_s1,
                -kernel.r1,
                -kernel.dr1,
                -kernel.ddr1,
                -kernel.dddr1,
                -fifth.ddddr1,
            ],
            -censored_weight,
        );
    }
    if event_weight != 0.0 {
        add_scaled_stack(
            &mut u1,
            [
                kernel.logphi1,
                kernel.dlogphi1,
                kernel.d2logphi1,
                kernel.d3logphi1,
                kernel.d4logphi1,
                fifth.d5logphi1,
            ],
            -event_weight,
        );
    }
    let g = (event_weight != 0.0).then(|| {
        let mut stack = [0.0; ORDER];
        add_scaled_stack(
            &mut stack,
            [
                kernel.log_g,
                kernel.d_log_g,
                kernel.d2_log_g,
                kernel.d3_log_g,
                kernel.d4_log_g,
                fifth.d5_log_g,
            ],
            -event_weight,
        );
        stack
    });
    // The log-scale term rides with the rate stack: it is the `−η_σ` of the same
    // `log(du1/dt)`, so a row whose rate stack is exactly zero (an inactive rate
    // term, or a stack of shape partials that `log g` does not have) carries none.
    let g_log_scale = if g.as_ref().is_some_and(|stack| !stack_is_exactly_zero(stack)) {
        event_weight
    } else {
        0.0
    };
    SlsOuterPlan {
        u0,
        u1: (censored_weight != 0.0 || event_weight != 0.0).then_some(u1),
        g,
        g_log_scale,
    }
}

row_atom! {
    fn sls_index [generic, order2](h, eta_t, eta_ls) {
        (h - eta_t) * exp(-eta_ls)
    }
}

row_atom! {
    fn sls_event_rate [generic, order2](
        h,
        hdot,
        eta_t,
        eta_t_deriv,
        eta_ls_deriv
    ) {
        hdot - eta_t_deriv - (h - eta_t) * eta_ls_deriv
    }
}

/// Whether a composition stack is exactly zero in every entry. Such a stack
/// contributes nothing to the row NLL at ANY derivative order, so it must be
/// SKIPPED rather than composed: at a clamped far-tail row (a censored
/// observation with `S ≈ 1` carries `q` at f64 extremes) the index jet's
/// direction channels are enormous, and Faà di Bruno would form jet-channel
/// products that overflow to `∞` BEFORE the zero outer derivative multiplies
/// them — manufacturing `0·∞ = NaN` out of an exactly-zero term (#2342
/// far-tail dH NaN, localized by `zz_measure_2342_far_tail_dh_nan_localization`).
#[inline(always)]
fn stack_is_exactly_zero<const ORDER: usize>(stack: &[f64; ORDER]) -> bool {
    stack.iter().all(|v| *v == 0.0)
}

#[inline(always)]
fn sls_program_exp_stack(value: f64) -> [f64; 5] {
    let exp = value.exp();
    [exp; 5]
}

#[inline(always)]
fn sls_program_exp_stack_fifth(value: f64) -> [f64; 6] {
    let exp = value.exp();
    [exp; 6]
}

row_program! {
    fn sls_row_program(
        h0,
        h1,
        hdot,
        eta_t_exit,
        eta_t_entry,
        eta_t_deriv,
        eta_ls_exit,
        eta_ls_entry,
        eta_ls_deriv;
        u0_value,
        u0_first,
        u0_second,
        u0_third,
        u0_fourth,
        u0_fifth,
        u1_value,
        u1_first,
        u1_second,
        u1_third,
        u1_fourth,
        u1_fifth,
        g_value,
        g_first,
        g_second,
        g_third,
        g_fourth,
        g_fifth,
        g_log_scale
    )
    emit [generic, order2, third, fourth, fifth_contracted];
    leaves {
        exponential => sls_program_exp_stack => sls_program_exp_stack_cuda => sls_program_exp_stack_fifth,
        // Each residual-distribution stack (`logS`, `logφ`, `log g`) is
        // supplied: the kernel builder evaluated it at the point the program
        // recomputes from the same parameters (`u0 = (h0 − η_t)·e^{−η_ls}`,
        // likewise `u1`, `g`), and an inactive slot never reaches the compose. The sixth
        // entry is read only by the fifth-order surface; every lower order
        // is called with it zero (#2677).
        outer => supplied,
    }
    witnesses [];
    {
        let neg_eta_ls_entry = neg(eta_ls_entry);
        let inv_sigma_entry = compose(exponential, neg_eta_ls_entry);
        let u0 = mul(add(h0, neg(eta_t_entry)), inv_sigma_entry);

        let neg_eta_ls_exit = neg(eta_ls_exit);
        let inv_sigma_exit = compose(exponential, neg_eta_ls_exit);
        let residual_exit = add(h1, neg(eta_t_exit));
        let u1 = mul(residual_exit, inv_sigma_exit);
        // `du1/dt = e^{−η_ls}·g`: the log stack is taken at `g`, and the scale's
        // `−η_ls` enters the event log-density as the linear `g_log_scale` term.
        let g = add(add(hdot, neg(eta_t_deriv)), neg(mul(residual_exit, eta_ls_deriv)));

        let mut nll = zero();
        if (u0_value != 0.0 || u0_first != 0.0 || u0_second != 0.0 || u0_third != 0.0 || u0_fourth != 0.0 || u0_fifth != 0.0) {
            nll = compose(
                outer,
                u0,
                u0_value,
                u0_first,
                u0_second,
                u0_third,
                u0_fourth,
                u0_fifth
            );
        }
        if (u1_value != 0.0 || u1_first != 0.0 || u1_second != 0.0 || u1_third != 0.0 || u1_fourth != 0.0 || u1_fifth != 0.0) {
            nll = add(
                nll,
                compose(
                    outer,
                    u1,
                    u1_value,
                    u1_first,
                    u1_second,
                    u1_third,
                    u1_fourth,
                    u1_fifth
                )
            );
        }
        if (g_value != 0.0 || g_first != 0.0 || g_second != 0.0 || g_third != 0.0 || g_fourth != 0.0 || g_fifth != 0.0) {
            nll = add(
                nll,
                compose(
                    outer,
                    g,
                    g_value,
                    g_first,
                    g_second,
                    g_third,
                    g_fourth,
                    g_fifth
                )
            );
        }
        if (g_log_scale != 0.0) {
            nll = add(nll, scale(eta_ls_exit, g_log_scale));
        }
        return nll;
    }
}

/// The stacks the row program takes from a plan: an absent slot is the
/// all-zero stack, which the program's own activity condition reads as
/// inactive (an all-zero stack is also how the kernel spells "this term is
/// not on this row", an untruncated row's entry stack). The program takes no
/// activity flag: a flag computed here ran before the inlined program could
/// issue its first leaf call, and held both `exp` calls behind the scans.
#[inline(always)]
fn sls_program_stacks<const ORDER: usize>(
    plan: &SlsOuterPlan<ORDER>,
) -> ([f64; ORDER], [f64; ORDER]) {
    (plan.u1.unwrap_or([0.0; ORDER]), plan.g.unwrap_or([0.0; ORDER]))
}

#[inline(always)]
pub(crate) fn sls_row_nll<S: JetScalar<SLS_ROW_K>>(
    vars: &[S; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
) -> Result<S, String> {
    let plan = sls_outer_plan::<5>(kernel);
    let (u1, g) = sls_program_stacks(&plan);
    let (nll, []) = sls_row_program(
        &vars[0], &vars[1], &vars[2], &vars[3], &vars[4], &vars[5], &vars[6], &vars[7], &vars[8],
        plan.u0[0], plan.u0[1], plan.u0[2], plan.u0[3], plan.u0[4], 0.0, u1[0],
        u1[1], u1[2], u1[3], u1[4], 0.0, g[0], g[1], g[2], g[3], g[4], 0.0, plan.g_log_scale,
    );
    Ok(nll)
}

#[inline(always)]
fn sls_row_vgh_generated(
    primary: &[f64; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
) -> (f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K]) {
    let plan = sls_outer_plan::<3>(kernel);
    let (u1, g) = sls_program_stacks(&plan);
    let (value, gradient, hessian, []) = sls_row_program_order2(
        primary[0], primary[1], primary[2], primary[3], primary[4], primary[5], primary[6],
        primary[7], primary[8], plan.u0[0], plan.u0[1], plan.u0[2], 0.0, 0.0, 0.0,
        u1[0], u1[1], u1[2], 0.0, 0.0, 0.0, g[0], g[1], g[2], 0.0, 0.0, 0.0, plan.g_log_scale,
    );
    (value, gradient, hessian)
}

#[inline(always)]
fn sls_row_third_generated(
    primary: &[f64; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
    direction: &[f64; SLS_ROW_K],
) -> [[f64; SLS_ROW_K]; SLS_ROW_K] {
    sls_row_third_generated_with_plan(primary, &sls_outer_plan::<5>(kernel), direction)
}

/// [`sls_row_third_generated`] from the row's outer derivative plan, for a sweep over many
/// directions of one row; the contraction has this one source either way.
#[inline(always)]
fn sls_row_third_generated_with_plan(
    primary: &[f64; SLS_ROW_K],
    plan: &SlsOuterPlan<5>,
    direction: &[f64; SLS_ROW_K],
) -> [[f64; SLS_ROW_K]; SLS_ROW_K] {
    let (u1, g) = sls_program_stacks(plan);
    sls_row_program_third_contracted(
        primary[0], primary[1], primary[2], primary[3], primary[4], primary[5], primary[6],
        primary[7], primary[8], plan.u0[0], plan.u0[1], plan.u0[2], plan.u0[3],
        plan.u0[4], 0.0, u1[0], u1[1], u1[2], u1[3], u1[4], 0.0, g[0], g[1], g[2], g[3],
        g[4], 0.0, plan.g_log_scale, direction,
    )
}

#[inline(always)]
fn sls_row_fourth_generated(
    primary: &[f64; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
    direction_u: &[f64; SLS_ROW_K],
    direction_v: &[f64; SLS_ROW_K],
) -> [[f64; SLS_ROW_K]; SLS_ROW_K] {
    sls_row_fourth_generated_with_plan(
        primary,
        &sls_outer_plan::<5>(kernel),
        direction_u,
        direction_v,
    )
}

/// [`sls_row_fourth_generated`] from the row's outer derivative plan. The plan is a
/// function of the row alone, so a sweep over many directions for one row builds it
/// once; the contraction has this one source either way.
#[inline(always)]
fn sls_row_fourth_generated_with_plan(
    primary: &[f64; SLS_ROW_K],
    plan: &SlsOuterPlan<5>,
    direction_u: &[f64; SLS_ROW_K],
    direction_v: &[f64; SLS_ROW_K],
) -> [[f64; SLS_ROW_K]; SLS_ROW_K] {
    let (u1, g) = sls_program_stacks(plan);
    sls_row_program_fourth_contracted(
        primary[0],
        primary[1],
        primary[2],
        primary[3],
        primary[4],
        primary[5],
        primary[6],
        primary[7],
        primary[8],
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        0.0,
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        0.0,
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        0.0,
        plan.g_log_scale,
        direction_u,
        direction_v,
    )
}

/// `Σ_{cde} ℓ_{abcde} u_c v_d w_e`: the fifth-order lowering of the same
/// canonical [`sls_row_program`], reading each residual-distribution stack
/// through its fifth derivative (#2677).
#[inline(always)]
fn sls_row_fifth_generated(
    primary: &[f64; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
    fifth: &SlsFifthEntries,
    direction_u: &[f64; SLS_ROW_K],
    direction_v: &[f64; SLS_ROW_K],
    direction_w: &[f64; SLS_ROW_K],
) -> [[f64; SLS_ROW_K]; SLS_ROW_K] {
    let plan = sls_outer_plan_with_fifth::<6>(kernel, fifth);
    let (u1, g) = sls_program_stacks(&plan);
    sls_row_program_fifth_contracted(
        primary[0],
        primary[1],
        primary[2],
        primary[3],
        primary[4],
        primary[5],
        primary[6],
        primary[7],
        primary[8],
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        plan.u0[5],
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        u1[5],
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        g[5],
        plan.g_log_scale,
        direction_u,
        direction_v,
        direction_w,
    )
}

/// Hessian-only lowering of the same build-time symbolic atoms used by
/// [`sls_row_vgh_compiled`]. Only the 24 structurally live upper-triangle
/// channels exist in the output; no 9×9 primary Hessian is materialized.
#[inline(always)]
fn add_composed_hessian_pairs<const N: usize, const H: usize, A: Order2AtomChannels<N>>(
    output: &mut [f64; SLS_HESSIAN_PAIRS.len()],
    atom: &A,
    axes: [usize; N],
    first: f64,
    second: f64,
    add: [bool; H],
) {
    assert!(H == N * (N + 1) / 2);
    let mut packed = 0;
    for local_row in 0..N {
        for local_column in local_row..N {
            let row = axes[local_row];
            let column = axes[local_column];
            let slot = SLS_HESSIAN_PAIR_SLOTS[row][column];
            let inner_live = A::HESSIAN_BITS & (1u128 << packed) != 0;
            let outer_live = A::GRADIENT_BITS & (1u128 << local_row) != 0
                && A::GRADIENT_BITS & (1u128 << local_column) != 0;
            let channel = if inner_live {
                let inner = first * atom.hessian_at(local_row, local_column);
                if outer_live {
                    inner + second * atom.gradient_at(local_row) * atom.gradient_at(local_column)
                } else {
                    inner
                }
            } else if outer_live {
                second * atom.gradient_at(local_row) * atom.gradient_at(local_column)
            } else {
                packed += 1;
                continue;
            };
            if add[packed] {
                output[slot] += channel;
            } else {
                output[slot] = channel;
            }
            packed += 1;
        }
    }
}

#[inline(always)]
fn sls_row_hessian_pairs_compiled(
    primary: &[f64; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
) -> [f64; SLS_HESSIAN_PAIRS.len()] {
    let u0 = sls_index_order2(
        primary[SLS_U0_AXES[0]],
        primary[SLS_U0_AXES[1]],
        primary[SLS_U0_AXES[2]],
    );
    let u1 = sls_index_order2(
        primary[SLS_U1_AXES[0]],
        primary[SLS_U1_AXES[1]],
        primary[SLS_U1_AXES[2]],
    );
    let g = sls_event_rate_order2(
        primary[SLS_G_AXES[0]],
        primary[SLS_G_AXES[1]],
        primary[SLS_G_AXES[2]],
        primary[SLS_G_AXES[3]],
        primary[SLS_G_AXES[4]],
    );
    let plan = sls_outer_plan::<5>(kernel);
    let mut output = [0.0; SLS_HESSIAN_PAIRS.len()];
    add_composed_hessian_pairs(
        &mut output,
        &u0,
        SLS_U0_AXES,
        plan.u0[1],
        plan.u0[2],
        [false; 6],
    );
    if let Some(stack) = plan.u1 {
        add_composed_hessian_pairs(
            &mut output,
            &u1,
            SLS_U1_AXES,
            stack[1],
            stack[2],
            [false; 6],
        );
    }
    if let Some(stack) = plan.g {
        add_composed_hessian_pairs(
            &mut output,
            &g,
            SLS_G_AXES,
            stack[1],
            stack[2],
            // Upper triangle of the event rate's axes (h1, hdot, eta_t, eta_t',
            // eta_ls'): the pairs among (h1, eta_t) were written by `u1` and
            // accumulate; every other pair is first written here.
            [
                true, false, true, false, false, false, false, false, false, true, false, false,
                false, false, false,
            ],
        );
    }
    output
}

/// Materialize `X[row, :]` as a dense length-`ncols` vector (no sparse-aware
/// fast path — used only by the dense-Hessian / diagonal assembly, never the
/// hot matvec inner loop).
pub(crate) fn design_dense_row(d: &DesignMatrix, row: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(d.ncols());
    d.axpy_row_into(row, 1.0, &mut out.view_mut())
        .expect("design_dense_row: ncols-sized buffer matches design width");
    out
}

/// Accumulate `alpha * jac[row, :]` into the coefficient slice `out` for a dense
/// time Jacobian (the survival time block is materialized densely as
/// `time_jac_*`, so it has no sparse axpy primitive).
#[inline]
pub(crate) fn axpy_dense_row_into(jac: &Array2<f64>, row: usize, alpha: f64, out: &mut [f64]) {
    if alpha == 0.0 {
        return;
    }
    let jr = jac.row(row);
    for (o, &j) in out.iter_mut().zip(jr.iter()) {
        *o += alpha * j;
    }
}

pub(crate) fn row_set_from_survival_mask(
    row_mask: Option<&Array1<f64>>,
    n: usize,
) -> crate::row_kernel::RowSet {
    let Some(mask) = row_mask else {
        return crate::row_kernel::RowSet::All;
    };
    let rows = mask
        .iter()
        .enumerate()
        .filter_map(|(index, &weight)| {
            (weight != 0.0).then_some(crate::outer_subsample::WeightedOuterRow {
                index,
                weight,
                stratum: 0,
            })
        })
        .collect::<Vec<_>>();
    crate::row_kernel::RowSet::Subsample {
        rows: Arc::new(rows),
        n_full: n,
    }
}

/// The composed warp basis and every derivative order the row program's jet
/// composes it at, at one row's entry and exit index. Each inner slice has one
/// entry per wiggle column (`pw` long); bundling the slices keeps
/// [`sls_row_nll_wiggle`] within the argument budget.
///
/// The jet is order 4, so a `compose_unary` stack has five slots, and the row
/// program reads the basis at two places whose stacks are offset by one:
///
/// ```text
///   u0w, u1w = index + Σ βw_j·I_j(index)    stack [I,  I′, I″, I‴, I⁗ ]
///   m₁       = 1 + Σ βw_j·I′_j(q1)          stack [I′, I″, I‴, I⁗, I⁗′]   <- SHIFTED BY ONE
/// ```
///
/// So the entry stack `b_u0` carries five orders (`I … I⁗`) and the exit stack
/// `b_u1` carries SIX (`I … I⁗′`): the exit index feeds both the warp `u1w` and
/// the event-Jacobian slope `m₁`. Every slot is the evaluated derivative of the
/// basis actually built — none is a literal. Slot 4 of the value stack was once
/// the literal `0.0` (exact only for degree ≤ 3), and slot 4 of `m₁`'s stack
/// was once the literal `0.0` (exact only for degree ≤ 4); the composed-warp
/// floor is `4` and is a floor, not a cap, so both literals differentiated a
/// different function than the value at a degree the path admits — the first
/// at the floor itself, the second at any requested degree above it (gam#2695).
/// The ramp evaluator returns exact zeros for orders above the degree, so the
/// evaluated table is bit-identical to the literal wherever the literal was
/// right. A slot that is short changes nothing in the order-2 Hessian and
/// leaves an O(1) error in `row_third_contracted` / `row_fourth_contracted`,
/// which the FD oracle `survival_ls_wiggle_third_and_fourth_directional_match_fd_932`
/// catches (#932).
pub(crate) struct SlsWiggleRowBasis<'b> {
    pub(crate) b_u0: [&'b [f64]; 5],
    pub(crate) b_u1: [&'b [f64]; 6],
}

/// #932 link-wiggle: the survival-LS row NLL extended with the link warp
/// `q = q0 + Σ_j βw_j·B_j(q0)` and the time-derivative coupling
/// `du1/dt = e^{−η_ls}·g`, `g = (hdot − h·η_ls') + m1·qdot0`,
/// `qdot0 = η_t·η_ls' − η_t'`, `m1 = 1 + Σ_j βw_j·B'_j(q0_exit)`, with the time
/// channel scaled like the location (`u = h·e^{−η_ls} + q`, #2695) and the
/// scale's `−η_ls` entering the event log-density linearly, written ONCE over
/// a generic jet scalar (`KW = SLS_ROW_K + pw`). `vars[0..9]` are the base
/// channels (exactly [`sls_row_nll`]); `vars[9..9+pw]` are the wiggle
/// amplitudes βw. The per-row basis stacks are evaluated at the BASE indices
/// (the warp composes the basis onto the index jet) — see
/// [`SlsWiggleRowBasis`] for which orders each stack carries. Bit-identical
/// (modulo association) to the nested witness in
/// `survival_ls_wiggle_joint_hessian_matches_assembler_932`.
pub(crate) fn sls_row_nll_wiggle<'arena, S: RuntimeJetScalar<'arena>>(
    vars: &[S],
    kernel: &SurvivalExactRowKernel,
    pw: usize,
    basis: &SlsWiggleRowBasis<'_>,
) -> S {
    assert_eq!(
        vars.len(),
        SLS_ROW_K + pw,
        "link-wiggle row primary layout mismatch"
    );
    let [b0e, b1e, b2e, b3e, b4e] = basis.b_u0;
    let [b0x, b1x, b2x, b3x, b4x, b5x] = basis.b_u1;
    let inv_sigma_entry = vars[7].neg().exp();
    let q0 = vars[4].mul(&inv_sigma_entry).neg();
    let inv_sigma_exit = vars[6].neg().exp();
    let q1 = vars[3].mul(&inv_sigma_exit).neg();
    let qdot0 = vars[3].mul(&vars[8]).sub(&vars[5]);
    let hs0 = vars[0].mul(&inv_sigma_entry);
    let hs1 = vars[1].mul(&inv_sigma_exit);
    let time_rate = vars[2].sub(&vars[1].mul(&vars[8]));
    let mut q0w = q0.clone();
    let mut q1w = q1.clone();
    let mut m1 = vars[0].constant_like(1.0);
    for j in 0..pw {
        let bw = &vars[SLS_ROW_K + j];
        // The composition stacks are the basis's OWN derivative tower at the
        // current index; `m₁` reads it shifted by one because it is built from
        // `I′` rather than `I`, so its top slot is the FIFTH derivative. That
        // slot is evaluated, not stated: the composed-warp floor of 4 is a
        // floor and not a cap, and at a requested degree ≥ 5 the fifth
        // derivative is not zero (gam#2695).
        q0w = q0w.add(&bw.mul(&q0.compose_unary([b0e[j], b1e[j], b2e[j], b3e[j], b4e[j]])));
        q1w = q1w.add(&bw.mul(&q1.compose_unary([b0x[j], b1x[j], b2x[j], b3x[j], b4x[j]])));
        m1 = m1.add(&bw.mul(&q1.compose_unary([b1x[j], b2x[j], b3x[j], b4x[j], b5x[j]])));
    }
    let u0w = hs0.add(&q0w);
    let u1w = hs1.add(&q1w);
    let g = time_rate.add(&m1.mul(&qdot0));
    // Skip exactly-zero stacks instead of composing them (same #2342 far-tail
    // 0·∞ = NaN guard as [`sls_row_nll`]); `vars[0]` is a plain seed jet with
    // finite channels, so scaling it by zero is a safe zero of the right type.
    let u0_stack = [
        kernel.log_s0,
        -kernel.r0,
        -kernel.dr0,
        -kernel.ddr0,
        -kernel.dddr0,
    ];
    let mut nll = if stack_is_exactly_zero(&u0_stack) {
        vars[0].scale(0.0)
    } else {
        u0w.compose_unary(u0_stack).scale(kernel.w)
    };
    let censored_weight = kernel.w * (1.0 - kernel.d);
    if censored_weight != 0.0 {
        let u1_stack = [
            kernel.log_s1,
            -kernel.r1,
            -kernel.dr1,
            -kernel.ddr1,
            -kernel.dddr1,
        ];
        if !stack_is_exactly_zero(&u1_stack) {
            nll = nll.add(&u1w.compose_unary(u1_stack).scale(-censored_weight));
        }
    }
    let event_weight = kernel.w * kernel.d;
    if event_weight != 0.0 {
        let pdf_stack = [
            kernel.logphi1,
            kernel.dlogphi1,
            kernel.d2logphi1,
            kernel.d3logphi1,
            kernel.d4logphi1,
        ];
        if !stack_is_exactly_zero(&pdf_stack) {
            nll = nll.add(&u1w.compose_unary(pdf_stack).scale(-event_weight));
        }
        let g_stack = [
            kernel.log_g,
            kernel.d_log_g,
            kernel.d2_log_g,
            kernel.d3_log_g,
            kernel.d4_log_g,
        ];
        if !stack_is_exactly_zero(&g_stack) {
            nll = nll.add(&g.compose_unary(g_stack).scale(-event_weight));
            // `log(du1/dt) = log g − η_ls` (#2695).
            nll = nll.add(&vars[6].scale(event_weight));
        }
    }
    nll
}

/// Saved monotone time-wiggle inputs for one survival location-scale ALO row.
///
/// The affine time channels remain the first three ALO coordinates. `beta`
/// owns the protected tail of the same raw time-coefficient block, and the row
/// program composes
/// `h = h_base + B(h_base) beta` and
/// `h_dot = (1 + B'(h_base) beta) h_dot_base` inside its order-two jet. This is
/// essential: a post-warp Jacobian alone omits score-times-map-Hessian
/// curvature.
pub struct SurvivalLocationScaleAloTimeWiggleInput<'a> {
    pub beta: &'a [f64],
    pub entry_basis: &'a [f64],
    pub entry_basis_d1: &'a [f64],
    pub entry_basis_d2: &'a [f64],
    pub entry_basis_d3: &'a [f64],
    pub exit_basis: &'a [f64],
    pub exit_basis_d1: &'a [f64],
    pub exit_basis_d2: &'a [f64],
    pub exit_basis_d3: &'a [f64],
}

/// Saved link-wiggle inputs for one survival location-scale ALO row.
///
/// Every slice is evaluated at the unwarped standardized residual for the
/// same row. The exit third derivative is required even for an order-two
/// observed Hessian because the event-rate multiplier contains `B'(q_exit)`.
pub struct SurvivalLocationScaleAloWiggleInput<'a> {
    pub beta: &'a [f64],
    pub entry_basis: &'a [f64],
    pub entry_basis_d1: &'a [f64],
    pub entry_basis_d2: &'a [f64],
    pub entry_basis_d3: &'a [f64],
    pub exit_basis: &'a [f64],
    pub exit_basis_d1: &'a [f64],
    pub exit_basis_d2: &'a [f64],
    pub exit_basis_d3: &'a [f64],
}

/// Exact affine local coordinates consumed by the fitted survival
/// location-scale row likelihood.
///
/// The output coordinate order is
///
/// `[h_entry, h_exit, hdot_exit, eta_t_exit, eta_t_entry, eta_t_dot_exit,
///   eta_log_sigma_exit, eta_log_sigma_entry, eta_log_sigma_dot_exit,
///   time_wiggle_beta..., link_wiggle_beta...]`.
///
/// These are deliberately the affine predictor channels, not the post-warp
/// indices `(u_entry, u_exit, g)`: the latter have a nonlinear parameter map
/// whose second derivative contributes to observed curvature.
pub struct SurvivalLocationScaleAloRowInput<'a> {
    pub inverse_link: &'a InverseLink,
    pub prior_weight: f64,
    pub event: f64,
    pub derivative_guard: f64,
    pub h_entry: f64,
    pub h_exit: f64,
    pub hdot_exit: f64,
    pub eta_threshold_exit: f64,
    pub eta_threshold_entry: f64,
    pub eta_threshold_derivative_exit: f64,
    pub eta_log_sigma_exit: f64,
    pub eta_log_sigma_entry: f64,
    pub eta_log_sigma_derivative_exit: f64,
    /// Whether the row entered after the origin, so its likelihood conditions on
    /// surviving to its entry (`SurvivalLocationScaleFamily::entry_active`).
    pub entry_active: bool,
    pub time_wiggle: Option<SurvivalLocationScaleAloTimeWiggleInput<'a>>,
    pub link_wiggle: Option<SurvivalLocationScaleAloWiggleInput<'a>>,
}

pub struct SurvivalLocationScaleAloRowGeometry {
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
    pub coordinate_values: Array1<f64>,
}

/// Replay one saved survival location-scale row through the exact fitting
/// program and return its NLL score and observed Hessian.
///
/// No coefficient-space numerical differentiation, Fisher substitution, or
/// simplified post-warp map is used. A zero-weight row returns exact zero
/// channels before evaluating predictors, matching the fitter's dropped-row
/// semantics.
pub fn survival_location_scale_alo_row_geometry(
    input: SurvivalLocationScaleAloRowInput<'_>,
) -> Result<SurvivalLocationScaleAloRowGeometry, String> {
    if !input.prior_weight.is_finite() || input.prior_weight < 0.0 {
        return Err(format!(
            "survival location-scale ALO prior weight must be finite and non-negative, got {}",
            input.prior_weight
        ));
    }
    let time_wiggle_dimension = input
        .time_wiggle
        .as_ref()
        .map_or(0, |wiggle| wiggle.beta.len());
    if let Some(wiggle) = input.time_wiggle.as_ref() {
        for (label, values) in [
            ("entry basis", wiggle.entry_basis),
            ("entry basis d1", wiggle.entry_basis_d1),
            ("entry basis d2", wiggle.entry_basis_d2),
            ("entry basis d3", wiggle.entry_basis_d3),
            ("exit basis", wiggle.exit_basis),
            ("exit basis d1", wiggle.exit_basis_d1),
            ("exit basis d2", wiggle.exit_basis_d2),
            ("exit basis d3", wiggle.exit_basis_d3),
        ] {
            if values.len() != time_wiggle_dimension {
                return Err(format!(
                    "survival location-scale ALO time-wiggle {label} has {} entries; beta has {time_wiggle_dimension}",
                    values.len()
                ));
            }
        }
        if time_wiggle_dimension == 0 {
            return Err(
                "survival location-scale ALO time-wiggle metadata has zero coefficients"
                    .to_string(),
            );
        }
    }
    let link_wiggle_dimension = input
        .link_wiggle
        .as_ref()
        .map_or(0, |wiggle| wiggle.beta.len());
    if let Some(wiggle) = input.link_wiggle.as_ref() {
        for (label, values) in [
            ("entry basis", wiggle.entry_basis),
            ("entry basis d1", wiggle.entry_basis_d1),
            ("entry basis d2", wiggle.entry_basis_d2),
            ("entry basis d3", wiggle.entry_basis_d3),
            ("exit basis", wiggle.exit_basis),
            ("exit basis d1", wiggle.exit_basis_d1),
            ("exit basis d2", wiggle.exit_basis_d2),
            ("exit basis d3", wiggle.exit_basis_d3),
        ] {
            if values.len() != link_wiggle_dimension {
                return Err(format!(
                    "survival location-scale ALO link-wiggle {label} has {} entries; beta has {link_wiggle_dimension}",
                    values.len()
                ));
            }
        }
        if link_wiggle_dimension == 0 {
            return Err(
                "survival location-scale ALO link-wiggle metadata has zero coefficients"
                    .to_string(),
            );
        }
    }
    let dimension = SLS_ROW_K + time_wiggle_dimension + link_wiggle_dimension;
    let mut coordinate_values = vec![
        input.h_entry,
        input.h_exit,
        input.hdot_exit,
        input.eta_threshold_exit,
        input.eta_threshold_entry,
        input.eta_threshold_derivative_exit,
        input.eta_log_sigma_exit,
        input.eta_log_sigma_entry,
        input.eta_log_sigma_derivative_exit,
    ];
    if let Some(wiggle) = input.time_wiggle.as_ref() {
        coordinate_values.extend_from_slice(wiggle.beta);
    }
    if let Some(wiggle) = input.link_wiggle.as_ref() {
        coordinate_values.extend_from_slice(wiggle.beta);
    }
    let coordinate_values = Array1::from_vec(coordinate_values);
    if input.prior_weight == 0.0 {
        return Ok(SurvivalLocationScaleAloRowGeometry {
            nll_score: Array1::zeros(dimension),
            observed_hessian: Array2::zeros((dimension, dimension)),
            coordinate_values,
        });
    }
    if !input.event.is_finite() || !(0.0..=1.0).contains(&input.event) {
        return Err(format!(
            "survival location-scale ALO event target must lie in [0,1], got {}",
            input.event
        ));
    }

    let (h_entry, h_exit, hdot_exit) = match input.time_wiggle.as_ref() {
        Some(wiggle) => {
            let mut h_entry = input.h_entry;
            let mut h_exit = input.h_exit;
            let mut derivative_multiplier = 1.0;
            for coefficient in 0..time_wiggle_dimension {
                let beta = wiggle.beta[coefficient];
                h_entry += beta * wiggle.entry_basis[coefficient];
                h_exit += beta * wiggle.exit_basis[coefficient];
                derivative_multiplier += beta * wiggle.exit_basis_d1[coefficient];
            }
            (h_entry, h_exit, derivative_multiplier * input.hdot_exit)
        }
        None => (input.h_entry, input.h_exit, input.hdot_exit),
    };
    let inv_sigma_entry = exp_sigma_inverse_from_eta_scalar(input.eta_log_sigma_entry);
    let inv_sigma_exit = exp_sigma_inverse_from_eta_scalar(input.eta_log_sigma_exit);
    let q_base_entry = -input.eta_threshold_entry * inv_sigma_entry;
    let q_base_exit = -input.eta_threshold_exit * inv_sigma_exit;
    let qdot_base_exit = input.eta_threshold_exit * input.eta_log_sigma_derivative_exit
        - input.eta_threshold_derivative_exit;
    let (q_entry, q_exit, qdot_exit) = match input.link_wiggle.as_ref() {
        Some(wiggle) => {
            let mut q_entry = q_base_entry;
            let mut q_exit = q_base_exit;
            let mut derivative_multiplier = 1.0;
            for coefficient in 0..link_wiggle_dimension {
                let beta = wiggle.beta[coefficient];
                q_entry += beta * wiggle.entry_basis[coefficient];
                q_exit += beta * wiggle.exit_basis[coefficient];
                derivative_multiplier += beta * wiggle.exit_basis_d1[coefficient];
            }
            (q_entry, q_exit, derivative_multiplier * qdot_base_exit)
        }
        None => (q_base_entry, q_base_exit, qdot_base_exit),
    };
    // The scale divides the time transform too (#2695): `u = h·e^{−η_ls} + q`
    // and `du1/dt = e^{−η_ls}·g` with `g = (hdot − h·η_ls') + qdot`.
    let state = survival_predictor_state(
        h_entry * inv_sigma_entry,
        h_exit * inv_sigma_exit,
        hdot_exit - h_exit * input.eta_log_sigma_derivative_exit,
        q_entry,
        q_exit,
        qdot_exit,
        -input.eta_log_sigma_exit,
        input.entry_active,
    );
    let kernel = SurvivalLocationScaleFamily::exact_row_kernel_from_parts(
        input.inverse_link,
        input.derivative_guard,
        input.prior_weight,
        input.event,
        0,
        state,
        0.0,
    )?
    .expect("positive-weight ALO row produces an exact kernel");

    let (score, hessian) = match (input.time_wiggle.as_ref(), input.link_wiggle.as_ref()) {
        (None, None) => {
            let primary: [f64; SLS_ROW_K] = coordinate_values
                .as_slice()
                .expect("owned ALO coordinates are contiguous")
                .try_into()
                .expect("non-wiggle coordinate count is nine");
            let (_, score, hessian) = sls_row_vgh_generated(&primary, &kernel);
            (
                Array1::from_vec(score.to_vec()),
                Array2::from_shape_vec(
                    (SLS_ROW_K, SLS_ROW_K),
                    hessian.into_iter().flatten().collect(),
                )
                .expect("fixed survival location-scale Hessian shape"),
            )
        }
        (time_wiggle, link_wiggle) => {
            let arena = DynamicJetArena::new();
            let variables = arena.alloc_slice_fill_with(dimension, |axis| {
                DynamicOrder2::variable(coordinate_values[axis], axis, dimension, &arena)
            });
            let (time_entry, time_exit, time_derivative) = match time_wiggle {
                None => (
                    variables[0].clone(),
                    variables[1].clone(),
                    variables[2].clone(),
                ),
                Some(wiggle) => {
                    let mut entry = variables[0].clone();
                    let mut exit = variables[1].clone();
                    let mut derivative_multiplier = variables[0].constant_like(1.0);
                    for coefficient in 0..time_wiggle_dimension {
                        let beta = &variables[SLS_ROW_K + coefficient];
                        entry = entry.add(&beta.mul(&variables[0].compose_unary([
                            wiggle.entry_basis[coefficient],
                            wiggle.entry_basis_d1[coefficient],
                            wiggle.entry_basis_d2[coefficient],
                            wiggle.entry_basis_d3[coefficient],
                            0.0,
                        ])));
                        exit = exit.add(&beta.mul(&variables[1].compose_unary([
                            wiggle.exit_basis[coefficient],
                            wiggle.exit_basis_d1[coefficient],
                            wiggle.exit_basis_d2[coefficient],
                            wiggle.exit_basis_d3[coefficient],
                            0.0,
                        ])));
                        derivative_multiplier =
                            derivative_multiplier.add(&beta.mul(&variables[1].compose_unary([
                                wiggle.exit_basis_d1[coefficient],
                                wiggle.exit_basis_d2[coefficient],
                                wiggle.exit_basis_d3[coefficient],
                                0.0,
                                0.0,
                            ])));
                    }
                    (entry, exit, derivative_multiplier.mul(&variables[2]))
                }
            };
            let mut likelihood_variables = Vec::with_capacity(SLS_ROW_K + link_wiggle_dimension);
            likelihood_variables.push(time_entry);
            likelihood_variables.push(time_exit);
            likelihood_variables.push(time_derivative);
            likelihood_variables.extend(variables[3..SLS_ROW_K].iter().cloned());
            likelihood_variables.extend(
                variables[SLS_ROW_K + time_wiggle_dimension..]
                    .iter()
                    .cloned(),
            );
            let empty: &[f64] = &[];
            let basis = match link_wiggle {
                None => SlsWiggleRowBasis {
                    b_u0: [empty, empty, empty, empty, empty],
                    b_u1: [empty, empty, empty, empty, empty, empty],
                },
                // The ALO lowering is an ORDER-2 jet, so `compose_unary` reads
                // slots 0..=2 only and the top slots are inert. They are filled
                // with the third derivative the ALO input carries rather than
                // with a fresh literal, so this site states no fourth or fifth
                // derivative it does not have (gam#2695).
                Some(wiggle) => SlsWiggleRowBasis {
                    b_u0: [
                        wiggle.entry_basis,
                        wiggle.entry_basis_d1,
                        wiggle.entry_basis_d2,
                        wiggle.entry_basis_d3,
                        wiggle.entry_basis_d3,
                    ],
                    b_u1: [
                        wiggle.exit_basis,
                        wiggle.exit_basis_d1,
                        wiggle.exit_basis_d2,
                        wiggle.exit_basis_d3,
                        wiggle.exit_basis_d3,
                        wiggle.exit_basis_d3,
                    ],
                },
            };
            let result = sls_row_nll_wiggle(
                &likelihood_variables,
                &kernel,
                link_wiggle_dimension,
                &basis,
            );
            (
                Array1::from_vec(result.g().to_vec()),
                Array2::from_shape_vec((dimension, dimension), result.h().to_vec())
                    .expect("dynamic survival location-scale Hessian shape"),
            )
        }
    };
    if score.iter().any(|value| !value.is_finite())
        || hessian.iter().any(|value| !value.is_finite())
        || coordinate_values.iter().any(|value| !value.is_finite())
    {
        return Err(
            "survival location-scale ALO row geometry contains a non-finite channel".to_string(),
        );
    }
    Ok(SurvivalLocationScaleAloRowGeometry {
        nll_score: score,
        observed_hessian: hessian,
        coordinate_values,
    })
}

/// #932 link-wiggle joint-Hessian production kernel: routes the survival-LS
/// joint Hessian for link-wiggle rows through the single-source §13 warp
/// ([`sls_row_nll_wiggle`]) instead of the bespoke `assemble_h_wiggle`. The base
/// 9 channels reuse the existing [`SurvivalLsRowKernel`] designs; the βw
/// amplitudes are an IDENTITY map into a wiggle coefficient block appended last.
/// `KW = SLS_ROW_K + pw`.
pub(crate) struct SurvivalLsWiggleRowKernel<'a> {
    base: SurvivalLsRowKernel<'a>,
    pw: usize,
    wiggle_off: usize,
    betaw: Vec<f64>,
    b_u0_0: Array2<f64>,
    b_u0_1: Array2<f64>,
    b_u0_2: Array2<f64>,
    b_u0_3: Array2<f64>,
    b_u0_4: Array2<f64>,
    b_u1_0: Array2<f64>,
    b_u1_1: Array2<f64>,
    b_u1_2: Array2<f64>,
    b_u1_3: Array2<f64>,
    b_u1_4: Array2<f64>,
    b_u1_5: Array2<f64>,
}

struct SurvivalLsDynamicFold {
    matrix: Array2<f64>,
    arena: DynamicJetArena,
}

impl SurvivalLsDynamicFold {
    fn new(n_coefficients: usize) -> Self {
        Self {
            matrix: Array2::zeros((n_coefficients, n_coefficients)),
            arena: DynamicJetArena::new(),
        }
    }
}

impl<'a> SurvivalLsWiggleRowKernel<'a> {
    pub(crate) fn new(
        family: &'a SurvivalLocationScaleFamily,
        dynamic: &'a SurvivalDynamicGeometry,
        deriv_log_scale: f64,
    ) -> Result<Self, String> {
        let base = SurvivalLsRowKernel {
            family,
            dynamic,
            deriv_log_scale,
            offsets: family.joint_block_offsets(),
        };
        let knots = family
            .wiggle_knots
            .as_ref()
            .ok_or("link-wiggle kernel: missing wiggle knots")?;
        let degree = family
            .wiggle_degree
            .ok_or("link-wiggle kernel: missing wiggle degree")?;
        // The link warp is defined on the unwarped AFT index
        // `q = q0 + Σ βw_j B_j(q0)`, then the scaled time transform is added:
        // `u = h·e^{−η_ls} + q`. `dynamic.q_*` already contains the warp, so
        // composing at either that value or `u` would apply βB a second time. The base
        // indices persisted by `build_dynamic_geometry` are the unique centers
        // shared by fit, prediction, and this derivative program.
        let q_exit = &dynamic.q_base_exit;
        let q_entry = &dynamic.q_base_entry;
        let b_u0_0 = survival_wiggle_basis_with_options(
            q_entry.view(),
            knots,
            degree,
            BasisOptions::value(),
        )?;
        let b_u0_1 = survival_wiggle_basis_with_options(
            q_entry.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let b_u0_2 = survival_wiggle_basis_with_options(
            q_entry.view(),
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let b_u0_3 = survival_wiggle_third_basis(q_entry.view(), knots, degree)?;
        let b_u0_4 = survival_wiggle_fourth_basis(q_entry.view(), knots, degree)?;
        let b_u1_0 = survival_wiggle_basis_with_options(
            q_exit.view(),
            knots,
            degree,
            BasisOptions::value(),
        )?;
        let b_u1_1 = survival_wiggle_basis_with_options(
            q_exit.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let b_u1_2 = survival_wiggle_basis_with_options(
            q_exit.view(),
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let b_u1_3 = survival_wiggle_third_basis(q_exit.view(), knots, degree)?;
        let b_u1_4 = survival_wiggle_fourth_basis(q_exit.view(), knots, degree)?;
        let b_u1_5 = survival_wiggle_fifth_basis(q_exit.view(), knots, degree)?;
        let pw = b_u1_0.ncols();
        let design_pw = family
            .x_link_wiggle
            .as_ref()
            .ok_or("link-wiggle kernel: missing wiggle design")?
            .ncols();
        if pw == 0 {
            return Err("link-wiggle kernel: wiggle basis has zero columns".to_string());
        }
        if pw != design_pw {
            return Err(format!(
                "link-wiggle kernel: basis width {pw} does not match design width {design_pw}"
            ));
        }
        // joint_block_offsets() appends the wiggle block last, so its start is
        // the second-to-last offset (offsets = [0, time, +thr, +ls, +wiggle]).
        let wiggle_off = base.offsets[base.offsets.len() - 2];
        // βw is carried on the dynamic geometry (populated from the wiggle block
        // in `build_dynamic_geometry`), so this kernel needs no `block_states` —
        // letting the workspace `_from_parts` directional entry points build it.
        let betaw = dynamic
            .wiggle_beta
            .as_ref()
            .map(|b| b.to_vec())
            .ok_or("link-wiggle kernel: missing wiggle_beta on dynamic geometry")?;
        if betaw.len() != pw {
            return Err(format!(
                "link-wiggle kernel: coefficient width {} does not match basis width {pw}",
                betaw.len()
            ));
        }
        Ok(Self {
            base,
            pw,
            wiggle_off,
            betaw,
            b_u0_0,
            b_u0_1,
            b_u0_2,
            b_u0_3,
            b_u0_4,
            b_u1_0,
            b_u1_1,
            b_u1_2,
            b_u1_3,
            b_u1_4,
            b_u1_5,
        })
    }

    #[inline]
    fn primary_dimension(&self) -> usize {
        SLS_ROW_K + self.pw
    }

    #[inline]
    fn row_vars<'arena, S: RuntimeJetScalar<'arena, Workspace = DynamicJetArena>>(
        &self,
        row: usize,
        arena: &'arena DynamicJetArena,
        seed: impl Fn(f64, usize, usize, &'arena DynamicJetArena) -> S,
    ) -> &'arena [S] {
        let p = self.base.row_primary_values(row);
        let dimension = self.primary_dimension();
        arena.alloc_slice_fill_with(dimension, |a| {
            if a < SLS_ROW_K {
                seed(p[a], a, dimension, arena)
            } else {
                seed(self.betaw[a - SLS_ROW_K], a, dimension, arena)
            }
        })
    }

    #[inline]
    fn eval<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        row: usize,
        vars: &[S],
    ) -> Result<S, String> {
        let kernel = self.base.row_nll_inputs(row)?.1;
        let r_u0_0 = self.b_u0_0.row(row);
        let r_u0_1 = self.b_u0_1.row(row);
        let r_u0_2 = self.b_u0_2.row(row);
        let r_u0_3 = self.b_u0_3.row(row);
        let r_u0_4 = self.b_u0_4.row(row);
        let r_u1_0 = self.b_u1_0.row(row);
        let r_u1_1 = self.b_u1_1.row(row);
        let r_u1_2 = self.b_u1_2.row(row);
        let r_u1_3 = self.b_u1_3.row(row);
        let r_u1_4 = self.b_u1_4.row(row);
        let r_u1_5 = self.b_u1_5.row(row);
        let basis = SlsWiggleRowBasis {
            b_u0: [
                r_u0_0.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u0_1.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u0_2.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u0_3.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u0_4.as_slice().ok_or("non-contiguous wiggle basis row")?,
            ],
            b_u1: [
                r_u1_0.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u1_1.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u1_2.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u1_3.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u1_4.as_slice().ok_or("non-contiguous wiggle basis row")?,
                r_u1_5.as_slice().ok_or("non-contiguous wiggle basis row")?,
            ],
        };
        Ok(sls_row_nll_wiggle(vars, &kernel, self.pw, &basis))
    }

    /// Per-(channel, row) coefficient-block + dense design row, length KW.
    /// Base channels delegate to [`SurvivalLsRowKernel`]; βw channels map to the
    /// wiggle block with a unit (identity) row `e_b`.
    fn jrow(&self, ch: usize, row: usize) -> Option<(usize, Array1<f64>)> {
        if ch < SLS_ROW_K {
            let blk = self.base.channel_block(ch)?;
            let r = self.base.channel_row(ch, row)?;
            Some((self.base.offsets[blk], r))
        } else {
            let b = ch - SLS_ROW_K;
            let mut e = Array1::<f64>::zeros(self.pw);
            e[b] = 1.0;
            Some((self.wiggle_off, e))
        }
    }

    fn n_rows(&self) -> usize {
        self.base.family.n
    }

    fn n_coefficients(&self) -> usize {
        self.wiggle_off + self.pw
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> Vec<f64> {
        let base = crate::row_kernel::RowKernel::jacobian_action(
            &self.base,
            row,
            &d_beta[..self.wiggle_off],
        );
        base.into_iter()
            .chain((0..self.pw).map(|b| d_beta[self.wiggle_off + b]))
            .collect()
    }

    fn add_pullback_hessian(
        &self,
        row: usize,
        h: &[f64],
        row_weight: f64,
        target: &mut Array2<f64>,
    ) {
        let dimension = self.primary_dimension();
        assert_eq!(h.len(), dimension * dimension);
        let rows: Vec<Option<(usize, Array1<f64>)>> =
            (0..dimension).map(|ch| self.jrow(ch, row)).collect();
        for a in 0..dimension {
            let Some((off_a, ra)) = rows[a].as_ref() else {
                continue;
            };
            for b in 0..dimension {
                let hab = row_weight * h[a * dimension + b];
                if hab == 0.0 {
                    continue;
                }
                let Some((off_b, rb)) = rows[b].as_ref() else {
                    continue;
                };
                for (ia, &va) in ra.iter().enumerate() {
                    if va == 0.0 {
                        continue;
                    }
                    let w = hab * va;
                    let mut trow = target.row_mut(off_a + ia);
                    for (ib, &vb) in rb.iter().enumerate() {
                        trow[off_b + ib] += w * vb;
                    }
                }
            }
        }
    }

    pub(crate) fn row_order2<'arena>(
        &self,
        row: usize,
        arena: &'arena DynamicJetArena,
    ) -> Result<DynamicOrder2<'arena>, String> {
        let vars = self.row_vars(row, arena, DynamicOrder2::variable);
        self.eval(row, &vars)
    }

    fn row_third_contracted<'arena>(
        &self,
        row: usize,
        dir: &[f64],
        arena: &'arena DynamicJetArena,
    ) -> Result<DynamicOneSeed<'arena>, String> {
        assert_eq!(dir.len(), self.primary_dimension());
        let vars = self.row_vars(row, arena, |x, a, dimension, workspace| {
            DynamicOneSeed::seed_direction(x, a, dir[a], dimension, workspace)
        });
        self.eval(row, &vars)
    }

    fn row_fourth_contracted<'arena>(
        &self,
        row: usize,
        dir_u: &[f64],
        dir_v: &[f64],
        arena: &'arena DynamicJetArena,
    ) -> Result<DynamicTwoSeed<'arena>, String> {
        assert_eq!(dir_u.len(), self.primary_dimension());
        assert_eq!(dir_v.len(), self.primary_dimension());
        let vars = self.row_vars(row, arena, |x, a, dimension, workspace| {
            DynamicTwoSeed::seed(x, a, dir_u[a], dir_v[a], dimension, workspace)
        });
        self.eval(row, &vars)
    }

    fn hessian_dense(&self, rows: &crate::row_kernel::RowSet) -> Result<Array2<f64>, String> {
        let p = self.n_coefficients();
        rows.par_try_reduce_fold(
            self.n_rows(),
            || SurvivalLsDynamicFold::new(p),
            |mut acc, row, weight| {
                acc.arena.reset();
                let out = self.row_order2(row, &acc.arena)?;
                self.add_pullback_hessian(row, out.h(), weight, &mut acc.matrix);
                Ok(acc)
            },
            |mut a, b| {
                a.matrix += &b.matrix;
                Ok(a)
            },
        )
        .map(|fold| fold.matrix)
    }

    fn directional_derivative_dense(
        &self,
        rows: &crate::row_kernel::RowSet,
        d_beta: &[f64],
    ) -> Result<Array2<f64>, String> {
        assert_eq!(d_beta.len(), self.n_coefficients());
        let p = self.n_coefficients();
        rows.par_try_reduce_fold(
            self.n_rows(),
            || SurvivalLsDynamicFold::new(p),
            |mut acc, row, weight| {
                acc.arena.reset();
                let direction = self.jacobian_action(row, d_beta);
                let out = self.row_third_contracted(row, &direction, &acc.arena)?;
                self.add_pullback_hessian(row, out.contracted_third(), weight, &mut acc.matrix);
                Ok(acc)
            },
            |mut a, b| {
                a.matrix += &b.matrix;
                Ok(a)
            },
        )
        .map(|fold| fold.matrix)
    }

    fn second_directional_derivative_dense(
        &self,
        rows: &crate::row_kernel::RowSet,
        d_beta_u: &[f64],
        d_beta_v: &[f64],
    ) -> Result<Array2<f64>, String> {
        assert_eq!(d_beta_u.len(), self.n_coefficients());
        assert_eq!(d_beta_v.len(), self.n_coefficients());
        let p = self.n_coefficients();
        rows.par_try_reduce_fold(
            self.n_rows(),
            || SurvivalLsDynamicFold::new(p),
            |mut acc, row, weight| {
                acc.arena.reset();
                let direction_u = self.jacobian_action(row, d_beta_u);
                let direction_v = self.jacobian_action(row, d_beta_v);
                let out =
                    self.row_fourth_contracted(row, &direction_u, &direction_v, &acc.arena)?;
                self.add_pullback_hessian(row, out.contracted_fourth(), weight, &mut acc.matrix);
                Ok(acc)
            },
            |mut a, b| {
                a.matrix += &b.matrix;
                Ok(a)
            },
        )
        .map(|fold| fold.matrix)
    }
}

/// Assemble the link-wiggle joint Hessian through the runtime-sized packed row
/// jet. The primary dimension is exactly `SLS_ROW_K + pw`; no arity dispatch is
/// involved.
pub(crate) fn survival_ls_wiggle_joint_hessian_dense(
    family: &SurvivalLocationScaleFamily,
    dynamic: &SurvivalDynamicGeometry,
    deriv_log_scale: f64,
) -> Result<Array2<f64>, String> {
    let kernel = SurvivalLsWiggleRowKernel::new(family, dynamic, deriv_log_scale)?;
    kernel.hessian_dense(&crate::row_kernel::RowSet::All)
}

/// Assemble the single-source link-wiggle FIRST directional derivative
/// `Σ_c ℓ_{abc} dir_c =
/// (D_dir H)[a][b]` — the ε-Hessian channel of the §13 warp row NLL at the
/// packed `OneSeed<KW>` directional scalar, pulled back into coefficient space
/// by the SAME `JᵀHJ` the joint-Hessian path uses. Replaces the bespoke hand
/// assembly the `_from_parts_masked` wiggle fall-through previously ran (the
/// #736/#932 hand-derivative genus the single-source contract removes). The
/// convention matches the non-wiggle base path, which routes its directional
/// through the identical `row_kernel_directional_derivative` free function.
pub(crate) fn survival_ls_wiggle_directional_derivative_dense(
    family: &SurvivalLocationScaleFamily,
    dynamic: &SurvivalDynamicGeometry,
    deriv_log_scale: f64,
    rows: &crate::row_kernel::RowSet,
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let kernel = SurvivalLsWiggleRowKernel::new(family, dynamic, deriv_log_scale)?;
    kernel.directional_derivative_dense(rows, d_beta)
}

/// Assemble the single-source link-wiggle SECOND directional derivative
/// `Σ_cd ℓ_{abcd} u_c
/// v_d` — the ε,δ-Hessian channel of the §13 warp row NLL at the packed
/// `TwoSeed<KW>` bidirectional scalar. Replaces the previous wiggle carve-out
/// that returned `None` (no second-directional curvature for wiggle rows).
pub(crate) fn survival_ls_wiggle_second_directional_derivative_dense(
    family: &SurvivalLocationScaleFamily,
    dynamic: &SurvivalDynamicGeometry,
    deriv_log_scale: f64,
    rows: &crate::row_kernel::RowSet,
    d_beta_u: &[f64],
    d_beta_v: &[f64],
) -> Result<Array2<f64>, String> {
    let kernel = SurvivalLsWiggleRowKernel::new(family, dynamic, deriv_log_scale)?;
    kernel.second_directional_derivative_dense(rows, d_beta_u, d_beta_v)
}

/// Extract the unit-axis primary direction `J·e_a` from the per-row channel
/// cache. For the canonical axis `e_a` (a unit vector at global coefficient `a`)
/// the survival-LS Jacobian action collapses to: channel `c` carries
/// `design_row_c[a − offset_c]` when `a` lies in channel `c`'s coefficient block,
/// and `0` otherwise. This is `to_bits`-identical to
/// [`SurvivalLsRowKernel::jacobian_action`]`(row, e_a)`: that path forms each
/// channel as `design_row · e_a_block`, a dot product whose only surviving term
/// is `design_row[a − offset]·1.0`, with every other summand `·0.0` (and
/// `x + 0.0 == x`, `x·1.0 == x` exactly for finite `x`). Reading the entry
/// directly avoids the per-axis dot product entirely.
#[inline]
fn axis_direction_from_channel_cache(
    chans: &[Option<(usize, Array1<f64>)>],
    a: usize,
) -> [f64; SLS_ROW_K] {
    let mut dir = [0.0_f64; SLS_ROW_K];
    for (c, slot) in chans.iter().enumerate() {
        if let Some((off, ra)) = slot.as_ref()
            && a >= *off
            && a - *off < ra.len()
        {
            dir[c] = ra[a - *off];
        }
    }
    dir
}

/// Accumulate `Σ_{x,y} (w·t[x][y]) · (row_x ⊗ row_y)` into the dense `p×p`
/// `target` using the per-row channel cache, with the float operations in the
/// EXACT order [`SurvivalLsRowKernel::add_pullback_hessian`] uses (outer `x`,
/// inner `y`, then `ia`, `ib`; `hab·va` formed before `·vb`). The weight is
/// folded as `hab = w·t[x][y]`, which is `to_bits`-identical to both branches of
/// the generic per-axis reducer: the unit-weight branch passes `t` unscaled
/// (`hab = 1.0·t[x][y] == t[x][y]`) and the Horvitz–Thompson branch first builds
/// `scaled[x][y] = w·t[x][y]` (`1.0·x == x`, `w·0.0 == ±0.0 == 0.0` so the
/// `hab == 0.0` skip fires identically).
#[inline]
fn pullback_from_channel_cache(
    chans: &[Option<(usize, Array1<f64>)>],
    t: &[[f64; SLS_ROW_K]; SLS_ROW_K],
    w: f64,
    target: &mut Array2<f64>,
) {
    for x in 0..SLS_ROW_K {
        let Some((off_a, ra)) = chans[x].as_ref() else {
            continue;
        };
        for y in 0..SLS_ROW_K {
            let hab = w * t[x][y];
            if hab == 0.0 {
                continue;
            }
            let Some((off_b, rb)) = chans[y].as_ref() else {
                continue;
            };
            for (ia, &va) in ra.iter().enumerate() {
                if va == 0.0 {
                    continue;
                }
                let wv = hab * va;
                let mut trow = target.row_mut(off_a + ia);
                for (ib, &vb) in rb.iter().enumerate() {
                    trow[off_b + ib] += wv * vb;
                }
            }
        }
    }
}

/// #932: the canonical single-source seam. The row NLL is written ONCE as
/// [`sls_row_nll`]; this exposes it through [`gam_math::jet_tower::RowProgram`]
/// so the `RowKernel` contraction channels derive mechanically from `eval` (via
/// the `program_*` helpers) rather than re-seeding the packed jet per method.
/// Non-positive-weight rows carry no exact kernel and evaluate to a structural
/// zero — the `None` arm the hand contraction methods short-circuited on.
impl gam_math::jet_tower::RowProgram<SLS_ROW_K> for SurvivalLsRowKernel<'_> {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn primaries(&self, row: usize) -> Result<[f64; SLS_ROW_K], String> {
        Ok(self.row_primary_values(row))
    }

    fn eval<S: JetScalar<SLS_ROW_K>>(&self, row: usize, p: &[S; SLS_ROW_K]) -> Result<S, String> {
        match self.row_nll_inputs_opt(row)? {
            Some((_, kernel)) => sls_row_nll(p, &kernel),
            None => Ok(S::constant(0.0)),
        }
    }
}

impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_> {
    fn n_coefficients(&self) -> usize {
        *self.offsets.last().expect("offsets has block bounds")
    }

    fn row_kernel(
        &self,
        row: usize,
    ) -> Result<(f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K]), String> {
        // #932: value, gradient and Hessian consume the SAME scalar `u0/u1/g`
        // index expressions and outer derivative plan as `sls_row_nll`,
        // through the build-time symbolic order-2 lowering emitted from the
        // canonical [`sls_row_program`] declaration. The release cell races the
        // generated full V/G/H tuple against the retired strongest-hand fused
        // schedule after exact and NaN-poisoned endpoint parity.
        match self.row_nll_inputs_opt(row)? {
            Some((p, kernel)) => Ok(sls_row_vgh_generated(&p, &kernel)),
            None => Ok((0.0, [0.0; SLS_ROW_K], [[0.0; SLS_ROW_K]; SLS_ROW_K])),
        }
    }

    fn row_third_contracted(
        &self,
        row: usize,
        direction: &[f64; SLS_ROW_K],
    ) -> Result<[[f64; SLS_ROW_K]; SLS_ROW_K], String> {
        match self.row_nll_inputs_opt(row)? {
            Some((primary, kernel)) => Ok(sls_row_third_generated(&primary, &kernel, direction)),
            None => Ok([[0.0; SLS_ROW_K]; SLS_ROW_K]),
        }
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        direction_u: &[f64; SLS_ROW_K],
        direction_v: &[f64; SLS_ROW_K],
    ) -> Result<[[f64; SLS_ROW_K]; SLS_ROW_K], String> {
        match self.row_nll_inputs_opt(row)? {
            Some((primary, kernel)) => Ok(sls_row_fourth_generated(
                &primary,
                &kernel,
                direction_u,
                direction_v,
            )),
            None => Ok([[0.0; SLS_ROW_K]; SLS_ROW_K]),
        }
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; SLS_ROW_K] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.offsets[0]..self.offsets[1]]);
        let d_thr = d_beta.slice(s![self.offsets[1]..self.offsets[2]]);
        let d_ls = d_beta.slice(s![self.offsets[2]..self.offsets[3]]);
        let fam = self.family;
        let t_entry = Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold);
        let ls_entry = Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma);
        let ch5 = fam
            .x_threshold_deriv
            .as_ref()
            .map_or(0.0, |d| d.dot_row_view(row, d_thr));
        let ch8 = fam
            .x_log_sigma_deriv
            .as_ref()
            .map_or(0.0, |d| d.dot_row_view(row, d_ls));
        [
            self.dynamic.time_jac_entry.row(row).dot(&d_time),
            self.dynamic.time_jac_exit.row(row).dot(&d_time),
            self.dynamic.time_jac_deriv.row(row).dot(&d_time),
            fam.x_threshold.dot_row_view(row, d_thr),
            t_entry.dot_row_view(row, d_thr),
            ch5,
            fam.x_log_sigma.dot_row_view(row, d_ls),
            ls_entry.dot_row_view(row, d_ls),
            ch8,
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; SLS_ROW_K], out: &mut [f64]) {
        let fam = self.family;
        // Time block: channels 0,1,2 via the dense time Jacobians.
        {
            let time = &mut out[self.offsets[0]..self.offsets[1]];
            axpy_dense_row_into(&self.dynamic.time_jac_entry, row, v[0], time);
            axpy_dense_row_into(&self.dynamic.time_jac_exit, row, v[1], time);
            axpy_dense_row_into(&self.dynamic.time_jac_deriv, row, v[2], time);
        }
        // Threshold block: channels 3 (exit), 4 (entry), 5 (deriv).
        {
            let mut thr = ndarray::ArrayViewMut1::from(&mut out[self.offsets[1]..self.offsets[2]]);
            fam.x_threshold
                .axpy_row_into(row, v[3], &mut thr)
                .expect("threshold exit axpy");
            Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold)
                .axpy_row_into(row, v[4], &mut thr)
                .expect("threshold entry axpy");
            if let Some(d) = fam.x_threshold_deriv.as_ref() {
                d.axpy_row_into(row, v[5], &mut thr)
                    .expect("threshold deriv axpy");
            }
        }
        // Log-sigma block: channels 6 (exit), 7 (entry), 8 (deriv).
        {
            let mut ls = ndarray::ArrayViewMut1::from(&mut out[self.offsets[2]..self.offsets[3]]);
            fam.x_log_sigma
                .axpy_row_into(row, v[6], &mut ls)
                .expect("log_sigma exit axpy");
            Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma)
                .axpy_row_into(row, v[7], &mut ls)
                .expect("log_sigma entry axpy");
            if let Some(d) = fam.x_log_sigma_deriv.as_ref() {
                d.axpy_row_into(row, v[8], &mut ls)
                    .expect("log_sigma deriv axpy");
            }
        }
    }

    fn add_pullback_hessian(
        &self,
        row: usize,
        h: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        target: &mut Array2<f64>,
    ) {
        // Materialize each channel's dense block row once, then accumulate
        // h[a][b]·(row_a ⊗ row_b) into the (block_a, block_b) sub-block.
        let rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
            .map(|ch| self.channel_block(ch).zip(self.channel_row(ch, row)))
            .collect();
        for a in 0..SLS_ROW_K {
            let Some((ba, ra)) = rows[a].as_ref() else {
                continue;
            };
            let off_a = self.offsets[*ba];
            for b in 0..SLS_ROW_K {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((bb, rb)) = rows[b].as_ref() else {
                    continue;
                };
                let off_b = self.offsets[*bb];
                for (ia, &va) in ra.iter().enumerate() {
                    if va == 0.0 {
                        continue;
                    }
                    let w = hab * va;
                    let mut trow = target.row_mut(off_a + ia);
                    for (ib, &vb) in rb.iter().enumerate() {
                        trow[off_b + ib] += w * vb;
                    }
                }
            }
        }
    }

    fn add_diagonal_quadratic(
        &self,
        row: usize,
        h: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        diag: &mut [f64],
    ) {
        // diag[c] += Σ_{a,b ∈ block(c)} h[a][b]·row_a[c]·row_b[c]. Only
        // same-block channel pairs touch a given coefficient's diagonal slot.
        let rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
            .map(|ch| self.channel_block(ch).zip(self.channel_row(ch, row)))
            .collect();
        for a in 0..SLS_ROW_K {
            let Some((ba, ra)) = rows[a].as_ref() else {
                continue;
            };
            for b in 0..SLS_ROW_K {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((bb, rb)) = rows[b].as_ref() else {
                    continue;
                };
                if ba != bb {
                    continue;
                }
                let off = self.offsets[*ba];
                for (k, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                    diag[off + k] += hab * va * vb;
                }
            }
        }
    }

    /// Batched all-axes first directional derivative with each row's primary
    /// values and outer derivative plan built ONCE and reused across every
    /// swept axis.
    ///
    /// The generic per-axis dispatcher computes the `p` matrices `{∂H/∂β[e_a]}`
    /// by running `p` independent single-direction sweeps. Each sweep, for each
    /// row, calls `row_third_contracted` → `row_nll_inputs_opt` →
    /// `exact_row_kernel_rescaled`, the special-function-heavy derivative ladder
    /// (`exp` / `log` / log-Φ derivatives). That ladder is INDEPENDENT of the
    /// swept axis, so the per-axis path rebuilds it `p` times per row — the
    /// dominant cost of the inner-Newton Jeffreys term and the outer-REML
    /// Jeffreys `H_Φ` drift, which probe this every joint evaluation. Here each
    /// row's primary values, plan and channel rows are materialized a single
    /// time, then every axis closes with the generated third contraction and the
    /// cached pullback.
    ///
    /// **Correctness contract.** Output `a` equals, bit-for-bit, the generic
    /// per-axis `row_kernel_directional_derivative(self, rows, e_a)`: the unit-axis
    /// direction read from the channel cache (`axis_direction_from_channel_cache`),
    /// the `sls_row_third_generated_with_plan` contraction the per-row hook
    /// reaches through `sls_row_third_generated`, the same Horvitz–Thompson row
    /// weight and pullback order, and the chunk-ordered reduction of
    /// `RowSet::par_try_reduce_fold` over the row set's walk positions. A row
    /// without an exact kernel (non-positive weight) is the per-row hook's
    /// all-zero matrix, whose pullback adds nothing. Every `RowSet` is handled.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if p != self.n_coefficients() {
            return Some(Err(format!(
                "directional_derivative_all_axes_dense_override: axis count {p} disagrees \
                 with n_coefficients() {}",
                self.n_coefficients(),
            )));
        }
        Some((|| {
            let m = rows.walk_len(gam_math::jet_tower::RowProgram::n_rows(self));
            // Per walk position, shared by every axis: the primary values and outer derivative plan.
            let fixed: Vec<Option<([f64; SLS_ROW_K], SlsOuterPlan<5>)>> = (0..m)
                .into_par_iter()
                .map(|position| {
                    Ok(self
                        .row_nll_inputs_opt(rows.row_at(position).0)?
                        .map(|(primary, kernel)| (primary, sls_outer_plan::<5>(&kernel))))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let chans: Vec<Vec<Option<(usize, Array1<f64>)>>> = (0..m)
                .into_par_iter()
                .map(|position| self.cached_channel_rows(rows.row_at(position).0))
                .collect();
            let n_chunks = arrow_row_chunk_count(m);
            (0..p)
                .into_par_iter()
                .map(|a| {
                    gam_problem::with_nested_parallel(|| -> Result<Array2<f64>, String> {
                        let chunk_accs: Vec<Array2<f64>> = (0..n_chunks)
                            .into_par_iter()
                            .map(|chunk_idx| {
                                let start = chunk_idx * ARROW_ROW_CHUNK;
                                let end = (start + ARROW_ROW_CHUNK).min(m);
                                let mut acc = Array2::<f64>::zeros((p, p));
                                for position in start..end {
                                    let Some((primary, plan)) = fixed[position].as_ref() else {
                                        continue;
                                    };
                                    let weight = rows.row_at(position).1;
                                    let direction =
                                        axis_direction_from_channel_cache(&chans[position], a);
                                    let third =
                                        sls_row_third_generated_with_plan(primary, plan, &direction);
                                    pullback_from_channel_cache(&chans[position], &third, weight, &mut acc);
                                }
                                acc
                            })
                            .collect();
                        let mut total = Array2::<f64>::zeros((p, p));
                        for acc in chunk_accs {
                            total = total + acc;
                        }
                        Ok(total)
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })())
    }

    /// Batched all-axes SECOND directional derivative `{H²dot[u, e_a]}_{a=0..p}`: the
    /// outer-REML Jeffreys `H_Φ` drift's all-axes object, and the second-order sibling of
    /// the override above.
    ///
    /// The generic per-axis fall-back runs `p` full-data sweeps, and each `(row, axis)`
    /// pair rebuilds the row's exact kernel (`row_nll_inputs_opt`: exp / log / log-Φ
    /// ladders) and its outer derivative plan, re-materializes the nine channel rows, and
    /// projects the fixed direction `u`. Only the axis direction depends on the axis. Each
    /// row's primary values and plan, its channel rows and `J·u` are therefore built once,
    /// and every axis closes with the generated fourth contraction and the cached pullback.
    ///
    /// **Correctness contract.** Output `a` equals, bit-for-bit, the generic per-axis
    /// `row_kernel_second_directional_derivative(self, rows, u, e_a)`. Both use the same
    /// `jacobian_action` for `u`, the unit-axis direction read from the channel cache
    /// (`axis_direction_from_channel_cache`, as the override above does), the
    /// `sls_row_fourth_generated_with_plan` contraction the per-row hook reaches through
    /// `sls_row_fourth_generated`, the same Horvitz–Thompson row weight and pullback order,
    /// and the chunk-ordered reduction of `RowSet::par_try_reduce_fold` over the row set's
    /// walk positions. A row without an exact kernel (non-positive weight) is the per-row
    /// hook's all-zero matrix, whose pullback adds nothing. Every `RowSet` is handled.
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::row_kernel::RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        let p = self.n_coefficients();
        if d_beta_u.len() != p {
            return Some(Err(format!(
                "second_directional_derivative_all_axes_dense_override: fixed direction has \
                 {} entries, expected {p}",
                d_beta_u.len(),
            )));
        }
        Some((|| {
            crate::row_kernel::RowKernel::<SLS_ROW_K>::warm_up_directional_caches(
                self,
                gam_problem::EvalMode::ValueGradientHessian,
            )?;
            let m = rows.walk_len(gam_math::jet_tower::RowProgram::n_rows(self));
            // Per walk position, shared by every axis: the primary values and outer derivative
            // plan, and the fixed direction's primary projection `J·u`.
            let fixed: Vec<Option<([f64; SLS_ROW_K], SlsOuterPlan<5>, [f64; SLS_ROW_K])>> = (0..m)
                .into_par_iter()
                .map(|position| {
                    let row = rows.row_at(position).0;
                    Ok(self.row_nll_inputs_opt(row)?.map(|(primary, kernel)| {
                        (
                            primary,
                            sls_outer_plan::<5>(&kernel),
                            crate::row_kernel::RowKernel::<SLS_ROW_K>::jacobian_action(
                                self, row, d_beta_u,
                            ),
                        )
                    }))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let chans: Vec<Vec<Option<(usize, Array1<f64>)>>> = (0..m)
                .into_par_iter()
                .map(|position| self.cached_channel_rows(rows.row_at(position).0))
                .collect();
            let n_chunks = arrow_row_chunk_count(m);
            (0..p)
                .into_par_iter()
                .map(|a| {
                    gam_problem::with_nested_parallel(|| -> Result<Array2<f64>, String> {
                        let chunk_accs: Vec<Array2<f64>> = (0..n_chunks)
                            .into_par_iter()
                            .map(|chunk_idx| {
                                let start = chunk_idx * ARROW_ROW_CHUNK;
                                let end = (start + ARROW_ROW_CHUNK).min(m);
                                let mut acc = Array2::<f64>::zeros((p, p));
                                for position in start..end {
                                    let Some((primary, plan, direction_u)) = fixed[position].as_ref()
                                    else {
                                        continue;
                                    };
                                    let weight = rows.row_at(position).1;
                                    let direction_a =
                                        axis_direction_from_channel_cache(&chans[position], a);
                                    let fourth = sls_row_fourth_generated_with_plan(
                                        primary,
                                        plan,
                                        direction_u,
                                        &direction_a,
                                    );
                                    pullback_from_channel_cache(
                                        &chans[position],
                                        &fourth,
                                        weight,
                                        &mut acc,
                                    );
                                }
                                acc
                            })
                            .collect();
                        let mut total = Array2::<f64>::zeros((p, p));
                        for acc in chunk_accs {
                            total = total + acc;
                        }
                        Ok(total)
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })())
    }
}

impl crate::row_kernel::RowKernelFifth<SLS_ROW_K> for SurvivalLsRowKernel<'_> {
    fn row_fifth_contracted_all_axes(
        &self,
        row: usize,
        direction_u: &[f64; SLS_ROW_K],
        direction_v: &[f64; SLS_ROW_K],
    ) -> Result<[[[f64; SLS_ROW_K]; SLS_ROW_K]; SLS_ROW_K], String> {
        match self.row_nll_fifth_inputs_opt(row)? {
            Some((primary, kernel, fifth)) => Ok(std::array::from_fn(|axis| {
                let mut direction_w = [0.0; SLS_ROW_K];
                direction_w[axis] = 1.0;
                sls_row_fifth_generated(
                    &primary,
                    &kernel,
                    &fifth,
                    direction_u,
                    direction_v,
                    &direction_w,
                )
            })),
            None => Ok([[[0.0; SLS_ROW_K]; SLS_ROW_K]; SLS_ROW_K]),
        }
    }
}

/// Exact mixed coefficient/design derivative `D_beta(D_psi H)[u]` for the
/// non-wiggle survival location-scale Hessian.
///
/// For one row, write the coefficient-space observed information as
/// `H = Jᵀ L₂ J`, where `J` is the predictor Jacobian and `L_k` is the kth
/// derivative of the single-sourced row NLL in predictor space. A design
/// hyperparameter moves both `J` and the predictors (`p_psi = J_psi beta`).
/// Differentiating first in psi and then along coefficient direction `u` gives
///
/// ```text
/// J_psiᵀ L₃[Ju] J
/// + Jᵀ (L₄[Ju, p_psi] + L₃[J_psi u]) J
/// + Jᵀ L₃[Ju] J_psi .
/// ```
///
/// The three contractions come from the same packed `OneSeed`/`TwoSeed` row
/// program as the production Hessian. This avoids a second hand-derived
/// survival formula and includes exit, entry, and time-derivative design
/// channels uniformly.
pub(crate) fn survival_ls_joint_psi_hessian_directional_derivative_dense(
    family: &SurvivalLocationScaleFamily,
    dynamic: &SurvivalDynamicGeometry,
    direction: &SurvivalJointPsiDirection,
    d_beta: &[f64],
    row_mask: Option<&Array1<f64>>,
) -> Result<Array2<f64>, String> {
    if family.x_link_wiggle.is_some() {
        return Err(
            "survival joint psi mixed Hessian uses the fixed-width row program; \
             link-wiggle geometry requires its dynamic-width analogue"
                .to_string(),
        );
    }
    let kernel = family.survival_ls_row_kernel_rescaled(dynamic, 0.0);
    let offsets = &kernel.offsets;
    let p = *offsets
        .last()
        .ok_or_else(|| "missing survival joint coefficient offset".to_string())?;
    if d_beta.len() != p {
        return Err(format!(
            "survival joint psi mixed Hessian direction length {} != coefficient dimension {p}",
            d_beta.len()
        ));
    }
    let rows = row_set_from_survival_mask(row_mask, family.n);
    rows.par_try_reduce_fold(
        family.n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, row_weight| -> Result<_, String> {
            let beta_direction =
                crate::row_kernel::RowKernel::jacobian_action(&kernel, row, d_beta);
            let psi_direction = direction.primary_direction(row);
            let mixed_primary = direction.jacobian_action(row, d_beta, offsets)?;
            let third_beta = crate::row_kernel::RowKernel::row_third_contracted(
                &kernel,
                row,
                &beta_direction,
            )?;
            let fourth_beta_psi = crate::row_kernel::RowKernel::row_fourth_contracted(
                &kernel,
                row,
                &beta_direction,
                &psi_direction,
            )?;
            let third_mixed = crate::row_kernel::RowKernel::row_third_contracted(
                &kernel,
                row,
                &mixed_primary,
            )?;

            let mut center = [[0.0; SLS_ROW_K]; SLS_ROW_K];
            for a in 0..SLS_ROW_K {
                for b in 0..SLS_ROW_K {
                    center[a][b] =
                        row_weight * (fourth_beta_psi[a][b] + third_mixed[a][b]);
                }
            }
            crate::row_kernel::RowKernel::add_pullback_hessian(
                &kernel,
                row,
                &center,
                &mut acc,
            );

            let base_rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
                .map(|channel| {
                    kernel
                        .channel_block(channel)
                        .zip(kernel.channel_row(channel, row))
                        .map(|(block, design_row)| (offsets[block], design_row))
                })
                .collect();
            let psi_rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
                .map(|channel| {
                    let Some(block) = kernel.channel_block(channel) else {
                        return Ok(None);
                    };
                    direction
                        .channel_row(channel, row)
                        .map(|row| row.map(|design_row| (offsets[block], design_row)))
                })
                .collect::<Result<_, gam_problem::CustomFamilyError>>().map_err(|error| error.to_string())?;

            for a in 0..SLS_ROW_K {
                if let Some((psi_offset, psi_row)) = psi_rows[a].as_ref() {
                    for b in 0..SLS_ROW_K {
                        let Some((base_offset, base_row)) = base_rows[b].as_ref() else {
                            continue;
                        };
                        let scale = row_weight * third_beta[a][b];
                        if scale == 0.0 {
                            continue;
                        }
                        for (ia, &va) in psi_row.iter().enumerate() {
                            if va == 0.0 {
                                continue;
                            }
                            let left = scale * va;
                            for (ib, &vb) in base_row.iter().enumerate() {
                                acc[[psi_offset + ia, base_offset + ib]] += left * vb;
                            }
                        }
                    }
                }
                let Some((base_offset, base_row)) = base_rows[a].as_ref() else {
                    continue;
                };
                for b in 0..SLS_ROW_K {
                    let Some((psi_offset, psi_row)) = psi_rows[b].as_ref() else {
                        continue;
                    };
                    let scale = row_weight * third_beta[a][b];
                    if scale == 0.0 {
                        continue;
                    }
                    for (ia, &va) in base_row.iter().enumerate() {
                        if va == 0.0 {
                            continue;
                        }
                        let left = scale * va;
                        for (ib, &vb) in psi_row.iter().enumerate() {
                            acc[[base_offset + ia, psi_offset + ib]] += left * vb;
                        }
                    }
                }
            }
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )
}

/// Build `D_psi H` from the same nine-primary row program as `H` itself.
///
/// For each row,
///
/// ```text
/// D_psi H = J_psiᵀ L₂ J + Jᵀ L₃[J_psi beta] J + Jᵀ L₂ J_psi .
/// ```
///
/// [`CustomFamilyJointPsiOperator`] is exactly this factorization: its channel
/// actions carry `J_psi`, `weights` carry `L₂`, and `drift_weights` carry
/// `L₃[J_psi beta]`. Building those arrays from the canonical row jets keeps
/// this first derivative definition identical to the mixed derivative above.
pub(crate) fn survival_ls_joint_psi_hessian_operator(
    family: &SurvivalLocationScaleFamily,
    dynamic: &SurvivalDynamicGeometry,
    direction: &SurvivalJointPsiDirection,
    row_mask: Option<&Array1<f64>>,
) -> Result<Arc<dyn HyperOperator>, String> {
    if family.x_link_wiggle.is_some() {
        return Err(
            "survival joint psi Hessian operator uses the fixed-width row program; \
             link-wiggle geometry requires its dynamic-width analogue"
                .to_string(),
        );
    }
    let kernel = family.survival_ls_row_kernel_rescaled(dynamic, 0.0);
    let offsets = &kernel.offsets;
    let p = *offsets
        .last()
        .ok_or_else(|| "missing survival joint coefficient offset".to_string())?;
    let mut channels = Vec::new();
    let mut channel_index = [None; SLS_ROW_K];

    channel_index[0] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[0]..offsets[1],
        shared_dense_arc(&dynamic.time_jac_entry),
        None,
    ));
    channel_index[1] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[0]..offsets[1],
        shared_dense_arc(&dynamic.time_jac_exit),
        None,
    ));
    channel_index[2] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[0]..offsets[1],
        shared_dense_arc(&dynamic.time_jac_deriv),
        None,
    ));
    channel_index[3] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[1]..offsets[2],
        family.x_threshold.clone(),
        direction.x_t_exit_action.clone(),
    ));
    channel_index[4] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[1]..offsets[2],
        SurvivalLsRowKernel::entry_design(&family.x_threshold_entry, &family.x_threshold).clone(),
        direction.x_t_entry_action.clone(),
    ));
    if let Some(design) = family.x_threshold_deriv.as_ref() {
        channel_index[5] = Some(channels.len());
        channels.push(CustomFamilyJointDesignChannel::new(
            offsets[1]..offsets[2],
            design.clone(),
            direction.x_t_deriv_action.clone(),
        ));
    }
    channel_index[6] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[2]..offsets[3],
        family.x_log_sigma.clone(),
        direction.x_ls_exit_action.clone(),
    ));
    channel_index[7] = Some(channels.len());
    channels.push(CustomFamilyJointDesignChannel::new(
        offsets[2]..offsets[3],
        SurvivalLsRowKernel::entry_design(&family.x_log_sigma_entry, &family.x_log_sigma).clone(),
        direction.x_ls_entry_action.clone(),
    ));
    if let Some(design) = family.x_log_sigma_deriv.as_ref() {
        channel_index[8] = Some(channels.len());
        channels.push(CustomFamilyJointDesignChannel::new(
            offsets[2]..offsets[3],
            design.clone(),
            direction.x_ls_deriv_action.clone(),
        ));
    }

    let active_channels: Vec<usize> = channel_index
        .iter()
        .enumerate()
        .filter_map(|(channel, index)| index.map(|_| channel))
        .collect();
    let m = active_channels.len();
    let mut weights = vec![Array1::<f64>::zeros(family.n); m * m];
    let mut drift_weights = vec![Array1::<f64>::zeros(family.n); m * m];
    for row in 0..family.n {
        let row_weight = row_mask.map_or(1.0, |mask| mask[row]);
        if row_weight == 0.0 {
            continue;
        }
        let (_, _, hessian) = crate::row_kernel::RowKernel::row_kernel(&kernel, row)?;
        let psi_direction = direction.primary_direction(row);
        let third = crate::row_kernel::RowKernel::row_third_contracted(
            &kernel,
            row,
            &psi_direction,
        )?;
        for (left, &a) in active_channels.iter().enumerate() {
            for (right, &b) in active_channels.iter().enumerate() {
                let slot = left * m + right;
                weights[slot][row] = row_weight * hessian[a][b];
                drift_weights[slot][row] = row_weight * third[a][b];
            }
        }
    }
    let mut pairs = Vec::with_capacity(m * m);
    for left in 0..m {
        for right in 0..m {
            let slot = left * m + right;
            pairs.push(CustomFamilyJointDesignPairContribution::new(
                left,
                right,
                std::mem::take(&mut weights[slot]),
                std::mem::take(&mut drift_weights[slot]),
            ));
        }
    }
    Ok(Arc::new(CustomFamilyJointPsiOperator::new(
        p, channels, pairs,
    )))
}

/// Accumulator of the explicit first-order ψ terms over rows.
struct SurvivalLsPsiFold {
    objective: f64,
    score: Array1<f64>,
    hessian: Option<Array2<f64>>,
    arena: DynamicJetArena,
}

/// One row's NLL jets in primary space, with the ψ motion of its primaries.
#[derive(Clone, Copy)]
struct SurvivalLsPsiRowJets<'a> {
    k: usize,
    gradient: &'a [f64],
    /// Row-major `k × k`.
    hessian: &'a [f64],
    /// Row-major `k × k` third derivative contracted with `psi_direction`;
    /// `None` when no dense `H_ψ` is being assembled.
    third: Option<&'a [f64]>,
    psi_direction: &'a [f64],
}

/// Add one row's explicit ψ terms from its NLL jets in primary space.
///
/// With primaries `p = J β` (base design rows `J`) and the ψ motion of the
/// primaries `p_ψ = J_ψ β` (ψ-derivative design rows `J_ψ`), the explicit
/// derivatives of the row NLL value, coefficient gradient and coefficient
/// Hessian are
///
/// ```text
/// V_ψ = ∇ℓ·p_ψ,   g_ψ = Jᵀ(H p_ψ) + J_ψᵀ∇ℓ,   H_ψ = Jᵀ T₃[p_ψ] J + J_ψᵀ H J + Jᵀ H J_ψ .
/// ```
fn add_survival_ls_psi_row(
    fold: &mut SurvivalLsPsiFold,
    row_weight: f64,
    jets: &SurvivalLsPsiRowJets<'_>,
    base_rows: &[Option<(usize, Array1<f64>)>],
    psi_rows: &[Option<(usize, Array1<f64>)>],
) {
    let SurvivalLsPsiRowJets {
        k,
        gradient,
        hessian,
        third,
        psi_direction,
    } = *jets;
    fold.objective += row_weight
        * gradient
            .iter()
            .zip(psi_direction)
            .map(|(&g, &d)| g * d)
            .sum::<f64>();
    for a in 0..k {
        let h_p: f64 = (0..k).map(|b| hessian[a * k + b] * psi_direction[b]).sum();
        if let Some((offset, design)) = base_rows[a].as_ref() {
            let scale = row_weight * h_p;
            if scale != 0.0 {
                for (i, &value) in design.iter().enumerate() {
                    fold.score[offset + i] += scale * value;
                }
            }
        }
        if let Some((offset, design)) = psi_rows[a].as_ref() {
            let scale = row_weight * gradient[a];
            if scale != 0.0 {
                for (i, &value) in design.iter().enumerate() {
                    fold.score[offset + i] += scale * value;
                }
            }
        }
    }
    let (Some(target), Some(third)) = (fold.hessian.as_mut(), third) else {
        return;
    };
    for a in 0..k {
        for b in 0..k {
            let Some((off_b, row_b)) = base_rows[b].as_ref() else {
                continue;
            };
            let drift = row_weight * third[a * k + b];
            let mixed = row_weight * hessian[a * k + b];
            if let Some((off_a, row_a)) = base_rows[a].as_ref()
                && drift != 0.0
            {
                for (ia, &va) in row_a.iter().enumerate() {
                    if va == 0.0 {
                        continue;
                    }
                    let left = drift * va;
                    for (ib, &vb) in row_b.iter().enumerate() {
                        target[[off_a + ia, off_b + ib]] += left * vb;
                    }
                }
            }
            if mixed == 0.0 {
                continue;
            }
            let Some((psi_off, psi_row)) = psi_rows[a].as_ref() else {
                continue;
            };
            for (ia, &va) in psi_row.iter().enumerate() {
                if va == 0.0 {
                    continue;
                }
                let left = mixed * va;
                for (ib, &vb) in row_b.iter().enumerate() {
                    target[[psi_off + ia, off_b + ib]] += left * vb;
                    target[[off_b + ib, psi_off + ia]] += left * vb;
                }
            }
        }
    }
}

/// The ψ-derivative design rows of one row, placed in their coefficient blocks.
/// The time channels have no ψ motion; a wiggle kernel's amplitude channels
/// (`k > SLS_ROW_K`) have none either.
fn survival_ls_psi_rows(
    direction: &SurvivalJointPsiDirection,
    offsets: &[usize],
    row: usize,
    k: usize,
) -> Result<Vec<Option<(usize, Array1<f64>)>>, String> {
    (0..k)
        .map(|channel| {
            if !(3..SLS_ROW_K).contains(&channel) {
                return Ok(None);
            }
            let block = if channel <= 5 { 1 } else { 2 };
            direction
                .channel_row(channel, row)
                .map(|design_row| design_row.map(|design_row| (offsets[block], design_row)))
                .map_err(|error| error.to_string())
        })
        .collect()
}

/// The explicit first-order ψ terms `(V_ψ, g_ψ, H_ψ)` of the survival
/// location-scale NLL, from the same row program as the value, gradient and
/// Hessian (the #736/#932 single-source contract). `H_ψ` is assembled dense only
/// when `dense_hessian` is set; the non-wiggle design-action path serves it as
/// [`survival_ls_joint_psi_hessian_operator`] instead. A link-wiggle family uses
/// its runtime-width row program, whose amplitude channels carry no ψ motion.
pub(crate) fn survival_ls_joint_psi_first_order_terms(
    family: &SurvivalLocationScaleFamily,
    dynamic: &SurvivalDynamicGeometry,
    direction: &SurvivalJointPsiDirection,
    row_mask: Option<&Array1<f64>>,
    dense_hessian: bool,
) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
    let rows = row_set_from_survival_mask(row_mask, family.n);
    let base_kernel = family.survival_ls_row_kernel_rescaled(dynamic, 0.0);
    let offsets = base_kernel.offsets.clone();
    let wiggle_kernel = if family.x_link_wiggle.is_some() {
        Some(SurvivalLsWiggleRowKernel::new(family, dynamic, 0.0)?)
    } else {
        None
    };
    let p = match wiggle_kernel.as_ref() {
        Some(kernel) => kernel.n_coefficients(),
        None => *offsets
            .last()
            .ok_or_else(|| "missing survival joint coefficient offset".to_string())?,
    };
    let init = || SurvivalLsPsiFold {
        objective: 0.0,
        score: Array1::zeros(p),
        hessian: dense_hessian.then(|| Array2::zeros((p, p))),
        arena: DynamicJetArena::new(),
    };
    let fold = rows.par_try_reduce_fold(
        family.n,
        init,
        |mut fold, row, row_weight| -> Result<SurvivalLsPsiFold, String> {
            let base_direction = direction.primary_direction(row);
            match wiggle_kernel.as_ref() {
                None => {
                    let (_, gradient, hessian) =
                        crate::row_kernel::RowKernel::row_kernel(&base_kernel, row)?;
                    let third = if dense_hessian {
                        Some(crate::row_kernel::RowKernel::row_third_contracted(
                            &base_kernel,
                            row,
                            &base_direction,
                        )?)
                    } else {
                        None
                    };
                    let hessian_flat: Vec<f64> = hessian.iter().flatten().copied().collect();
                    let third_flat: Option<Vec<f64>> =
                        third.map(|third| third.iter().flatten().copied().collect());
                    let base_rows = base_kernel.cached_channel_rows(row);
                    let psi_rows = survival_ls_psi_rows(direction, &offsets, row, SLS_ROW_K)?;
                    add_survival_ls_psi_row(
                        &mut fold,
                        row_weight,
                        &SurvivalLsPsiRowJets {
                            k: SLS_ROW_K,
                            gradient: &gradient,
                            hessian: &hessian_flat,
                            third: third_flat.as_deref(),
                            psi_direction: &base_direction,
                        },
                        &base_rows,
                        &psi_rows,
                    );
                }
                Some(kernel) => {
                    let k = kernel.primary_dimension();
                    let mut psi_direction = vec![0.0; k];
                    psi_direction[..SLS_ROW_K].copy_from_slice(&base_direction);
                    fold.arena.reset();
                    let order2 = kernel.row_order2(row, &fold.arena)?;
                    let gradient = order2.g().to_vec();
                    let hessian = order2.h().to_vec();
                    let third = if dense_hessian {
                        let out = kernel.row_third_contracted(row, &psi_direction, &fold.arena)?;
                        Some(out.contracted_third().to_vec())
                    } else {
                        None
                    };
                    let base_rows: Vec<Option<(usize, Array1<f64>)>> =
                        (0..k).map(|channel| kernel.jrow(channel, row)).collect();
                    let psi_rows = survival_ls_psi_rows(direction, &offsets, row, k)?;
                    add_survival_ls_psi_row(
                        &mut fold,
                        row_weight,
                        &SurvivalLsPsiRowJets {
                            k,
                            gradient: &gradient,
                            hessian: &hessian,
                            third: third.as_deref(),
                            psi_direction: &psi_direction,
                        },
                        &base_rows,
                        &psi_rows,
                    );
                }
            }
            Ok(fold)
        },
        |mut left, right| {
            left.objective += right.objective;
            left.score += &right.score;
            if let (Some(target), Some(part)) = (left.hessian.as_mut(), right.hessian.as_ref()) {
                *target += part;
            }
            Ok(left)
        },
    )?;
    Ok((fold.objective, fold.score, fold.hessian))
}

fn require_fitted_block_geometry(
    block_states: &[ParameterBlockState],
    context: &'static str,
) -> Result<(), SurvivalLocationScaleError> {
    if block_states.is_empty() {
        return Err(SurvivalLocationScaleError::InternalInvariant {
            reason: format!(
                "{context}: fitted block state is missing; likelihood residuals and curvature \
                 are undefined without the converged per-block mode"
            ),
        });
    }
    Ok(())
}

/// The three coefficient-space views lowered from the same packed survival-LS
/// row-Hessian coefficients. `DenseFull` is the coupled exact-Newton matrix
/// and `BlockDiagonal` is the per-block inner-Newton working set. Neither
/// target re-evaluates row calculus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SlsCoefficientHessianTarget {
    DenseFull,
    BlockDiagonal,
}

pub(crate) enum SlsCoefficientHessian {
    DenseFull(Array2<f64>),
    BlockDiagonal(Vec<Array2<f64>>),
}

impl SlsCoefficientHessian {
    /// Consume the coupled dense-full Newton matrix. The lowering is
    /// single-sourced from `survival_ls_coefficient_hessian`: the returned
    /// variant always matches the `SlsCoefficientHessianTarget` that was
    /// requested. A mismatch is therefore a solver-internal invariant break,
    /// surfaced as a real error rather than a panic.
    pub(crate) fn into_dense_full(self) -> Result<Array2<f64>, String> {
        match self {
            SlsCoefficientHessian::DenseFull(dense) => Ok(dense),
            other => Err(SlsCoefficientHessian::variant_mismatch("DenseFull", &other)),
        }
    }

    /// Consume the per-block inner-Newton working set. See `into_dense_full`
    /// for the single-source invariant that makes a mismatch an internal error.
    pub(crate) fn into_block_diagonal(self) -> Result<Vec<Array2<f64>>, String> {
        match self {
            SlsCoefficientHessian::BlockDiagonal(blocks) => Ok(blocks),
            other => Err(SlsCoefficientHessian::variant_mismatch(
                "BlockDiagonal",
                &other,
            )),
        }
    }

    fn variant_mismatch(expected: &str, got: &SlsCoefficientHessian) -> String {
        let got_name = match got {
            SlsCoefficientHessian::DenseFull(_) => "DenseFull",
            SlsCoefficientHessian::BlockDiagonal(_) => "BlockDiagonal",
        };
        SurvivalLocationScaleError::InternalInvariant {
            reason: format!(
                "survival-LS coefficient Hessian: requested {expected} view but lowered \
                 {got_name}; packed 24-pair plan target and returned shape are single-sourced"
            ),
        }
        .into()
    }
}

#[derive(Clone, Debug)]
struct SlsHessianPairGroup {
    left_channel: usize,
    right_channel: usize,
    left_design: usize,
    right_design: usize,
    pair_slots: Vec<usize>,
}

/// The only structurally live upper-triangle pairs of the canonical
/// nine-primary survival-LS row program. The ordering is block-major:
/// TT, TQ, TL, QQ, QL, LL. Symmetry supplies the omitted lower triangle. The
/// exit time value `h1` reaches the event rate through `−(h1 − eta_t)·eta_ls'`,
/// which is what makes `(1, 2)`, `(1, 5)` and `(1, 8)` live. The scale reaches
/// the event log-density only through its linear `−eta_ls` term, so no pair of
/// `eta_ls_exit` with a rate-only channel is live.
const SLS_HESSIAN_PAIRS: [(usize, usize); 24] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 4),
    (1, 3),
    (2, 3),
    (2, 5),
    (1, 5),
    (0, 7),
    (1, 6),
    (2, 8),
    (1, 8),
    (3, 3),
    (3, 5),
    (4, 4),
    (5, 5),
    (3, 6),
    (3, 8),
    (4, 7),
    (5, 8),
    (6, 6),
    (7, 7),
    (8, 8),
];

const fn sls_hessian_pair_slots() -> [[usize; SLS_ROW_K]; SLS_ROW_K] {
    let mut slots = [[SLS_HESSIAN_PAIRS.len(); SLS_ROW_K]; SLS_ROW_K];
    let mut slot = 0;
    while slot < SLS_HESSIAN_PAIRS.len() {
        let (row, column) = SLS_HESSIAN_PAIRS[slot];
        slots[row][column] = slot;
        slots[column][row] = slot;
        slot += 1;
    }
    slots
}

const SLS_HESSIAN_PAIR_SLOTS: [[usize; SLS_ROW_K]; SLS_ROW_K] = sls_hessian_pair_slots();

impl SurvivalLocationScaleFamily {
    pub(crate) const BLOCK_TIME: usize = 0;
    pub(crate) const BLOCK_THRESHOLD: usize = 1;
    pub(crate) const BLOCK_LOG_SIGMA: usize = 2;
    pub(crate) const BLOCK_LINK_WIGGLE: usize = 3;
    pub(crate) const EVALUATE_PARALLEL_ROW_THRESHOLD: usize = 1024;

    /// First directional derivatives require third qdot map derivatives when
    /// threshold/log-sigma derivative designs are present.
    #[inline]
    pub(crate) fn row_kernel_directional_supported(&self) -> bool {
        // #932: the directional path no longer builds the dense `Tower4<9>`. The
        // contracted third/fourth (`row_third_contracted` / `row_fourth_contracted`)
        // now evaluate the SAME single-sourced row NLL (`sls_row_nll`) through the
        // PACKED directional scalars `OneSeed<9>` (1.46 KiB) / `TwoSeed<9>`
        // (2.8 KiB) — the nilpotent ε/δ fold the contraction direction INTO the
        // differentiation, so only the contracted K×K matrix is carried, never the
        // full fourth-order tensor. That removes the ~50 KiB per-row tower whose
        // by-value copies overflowed the stack / timed out the fit (the exact
        // representation objection this gate was waiting on; module note in
        // `jet_scalar`). The packed contractions are bit-identical to the dense
        // `row_nll_tower(row)?.{third,fourth}_contracted(...)` (the
        // `survival_ls_packed_directional_matches_dense_tower_932` oracle pins this
        // ≤ 1e-9). Link-wiggle remains a separate runtime-width lowering
        // (`row_kernel_supported` gates on `x_link_wiggle.is_none()`):
        // [`SurvivalLsWiggleRowKernel`] evaluates the same row NLL with dynamic
        // packed jets because its beta-dependent Jacobian width is not const.
        self.x_link_wiggle.is_none()
    }

    pub(crate) fn survival_ls_row_kernel_rescaled<'a>(
        &'a self,
        dynamic: &'a SurvivalDynamicGeometry,
        deriv_log_scale: f64,
    ) -> SurvivalLsRowKernel<'a> {
        SurvivalLsRowKernel {
            family: self,
            dynamic,
            deriv_log_scale,
            offsets: self.joint_block_offsets(),
        }
    }

    /// Lower the canonical non-wiggle [`RowProgram`] Hessian into coefficient
    /// space through one packed 24-pair plan.
    ///
    /// Channels which resolve to the same physical design share a group before
    /// any cross product is run: entry falls back to exit, while absent
    /// derivative designs remove their pairs. Thus the 24 structural pairs
    /// become 12 cross products for invariant threshold/scale designs, 18 when
    /// one block is fully time-varying, and 24 when both are. Per-row coefficients
    /// live in one slot-major `groups × n` buffer. An HT mask, when present, is
    /// multiplied into each final grouped row coefficient exactly once.
    pub(crate) fn survival_ls_coefficient_hessian(
        &self,
        dynamic: &SurvivalDynamicGeometry,
        deriv_log_scale: f64,
        row_mask: Option<&Array1<f64>>,
        target: SlsCoefficientHessianTarget,
    ) -> Result<SlsCoefficientHessian, String> {
        if self.x_link_wiggle.is_some() {
            return Err(SurvivalLocationScaleError::InternalInvariant {
                reason: "the packed 24-pair survival-LS plan requires fixed non-wiggle geometry"
                    .to_string(),
            }
            .into());
        }
        if let Some(mask) = row_mask
            && mask.len() != self.n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival-LS coefficient Hessian mask length {} != row count {}",
                    mask.len(),
                    self.n
                ),
            }
            .into());
        }

        let threshold_exit = self.x_threshold.to_dense_cow();
        let threshold_entry = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let threshold_deriv = self
            .x_threshold_deriv
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let log_sigma_exit = self.x_log_sigma.to_dense_cow();
        let log_sigma_entry = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let log_sigma_deriv = self
            .x_log_sigma_deriv
            .as_ref()
            .map(DesignMatrix::to_dense_cow);

        let designs: [Option<&Array2<f64>>; SLS_ROW_K] = [
            Some(&dynamic.time_jac_entry),
            Some(&dynamic.time_jac_exit),
            Some(&dynamic.time_jac_deriv),
            Some(threshold_exit.as_ref()),
            Some(
                threshold_entry
                    .as_deref()
                    .unwrap_or(threshold_exit.as_ref()),
            ),
            threshold_deriv.as_deref(),
            Some(log_sigma_exit.as_ref()),
            Some(
                log_sigma_entry
                    .as_deref()
                    .unwrap_or(log_sigma_exit.as_ref()),
            ),
            log_sigma_deriv.as_deref(),
        ];

        // Stable design identities. A fallback channel deliberately receives
        // its exit channel's identity, which is what merges pair coefficients
        // before the single X'WX call. Optional derivative channels have no
        // identity and their structural pairs disappear.
        let design_ids: [Option<usize>; SLS_ROW_K] = [
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(if threshold_entry.is_some() { 4 } else { 3 }),
            threshold_deriv.as_ref().map(|_| 5),
            Some(6),
            Some(if log_sigma_entry.is_some() { 7 } else { 6 }),
            log_sigma_deriv.as_ref().map(|_| 8),
        ];

        let mut groups: Vec<SlsHessianPairGroup> = Vec::with_capacity(SLS_HESSIAN_PAIRS.len());
        for (pair_slot, (left_channel, right_channel)) in SLS_HESSIAN_PAIRS.into_iter().enumerate()
        {
            let (Some(left_design), Some(right_design)) =
                (design_ids[left_channel], design_ids[right_channel])
            else {
                continue;
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                group.left_design == left_design && group.right_design == right_design
            }) {
                group.pair_slots.push(pair_slot);
            } else {
                groups.push(SlsHessianPairGroup {
                    left_channel,
                    right_channel,
                    left_design,
                    right_design,
                    pair_slots: vec![pair_slot],
                });
            }
        }

        // Block and diagonal consumers cannot observe cross-block groups. Drop
        // those slots before allocating or evaluating the shared row buffer.
        if target != SlsCoefficientHessianTarget::DenseFull {
            groups.retain(|group| group.left_channel / 3 == group.right_channel / 3);
        }

        let kernel = self.survival_ls_row_kernel_rescaled(dynamic, deriv_log_scale);

        // #2342: per-row stable paired index-derivative sums S1, S2 for the
        // far-tail rows whose entry/exit hazard channels are each ~1e300 and
        // (near-)opposite. The compiled 24-pair Hessian merges them onto a
        // shared coefficient (H[6][6]+H[7][7] for log-sigma, H[3][3]+H[4][4] for
        // threshold) and cancels catastrophically — A''(u0) rounds to exactly
        // 1.0, losing ρ'(u0) = −1/u0², which the ∂²q/∂η_ls² ≈ q ~ 1e150 chain
        // amplifies back to a moderate O(1e149) information. The merged
        // coefficient is recomputed cancellation-free in the per-row loop below;
        // every non-far-tail row keeps the naive fold (moderate-regime bitwise).
        // Weighted by the family weight only — the HT mask is applied uniformly
        // inside the loop, exactly as it is to the compiled coefficients.
        let mut paired_s1 = Array1::<f64>::zeros(self.n);
        let mut paired_s2 = Array1::<f64>::zeros(self.n);
        let mut use_paired = vec![false; self.n];
        for row in 0..self.n {
            let u0 = dynamic.hs_entry[row] + dynamic.q_entry[row];
            if self.w[row] > 0.0
                && self.entry_active[row]
                && paired_stacks::paired_contraction_needs_regroup(&self.inverse_link, u0)
            {
                let u1 = dynamic.hs_exit[row] + dynamic.q_exit[row];
                let delta_u = (dynamic.hs_exit[row] - dynamic.hs_entry[row])
                    + (dynamic.q_exit[row] - dynamic.q_entry[row]);
                if let Some(sums) = paired_stacks::weighted_paired_index_sums(
                    &self.inverse_link,
                    u0,
                    u1,
                    delta_u,
                    self.y[row],
                    self.w[row],
                ) {
                    paired_s1[row] = sums[0];
                    paired_s2[row] = sums[1];
                    use_paired[row] = true;
                }
            }
        }

        let mut slots = Array2::<f64>::zeros((groups.len(), self.n));
        slots
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .enumerate()
            .try_for_each(|(row, mut row_slots)| -> Result<(), String> {
                let (coefficients, entry_stack) = match kernel.row_nll_inputs_opt(row)? {
                    Some((primary, exact)) => (
                        sls_row_hessian_pairs_compiled(&primary, &exact),
                        sls_outer_plan::<5>(&exact).u0,
                    ),
                    None => ([0.0; SLS_HESSIAN_PAIRS.len()], [0.0; 5]),
                };
                for (slot, group) in groups.iter().enumerate() {
                    let mut coefficient = group
                        .pair_slots
                        .iter()
                        .fold(0.0, |sum, &pair_slot| sum + coefficients[pair_slot]);
                    // #2342: on far-tail rows, replace the cancelling merged
                    // (entry+exit) block-diagonal coefficient with the stable
                    // paired form −(S1·D2 + S2·D²), D=∂q/∂η, D2=∂²q/∂η² (entry ==
                    // exit for a time-invariant channel). The merged group's
                    // representative pair is the diagonal (6,6)/(3,3), so the
                    // block contraction adds it once (no transpose doubling); the
                    // dropped shared-axis event-rate term is ~1e-150 relative.
                    if use_paired[row]
                        && group.left_design == group.right_design
                        && group.left_channel / 3 == group.right_channel / 3
                    {
                        let block = group.left_channel / 3;
                        if block == 2 && self.x_log_sigma_entry.is_none() {
                            // The scale divides the time transform too, so the entry
                            // and exit derivatives `∂u/∂η_ls = ∂q/∂η_ls − hs` differ by
                            // the scaled time gap. Anchor the stable pair sums at exit
                            // and carry the entry side's difference exactly: those
                            // terms are honestly huge, never a cancelling pair. The
                            // gaps are taken channel by channel, since `hs` rounds
                            // away inside `d1` and `d2` once `q ~ 1e150`.
                            let d1_exit = dynamic.dq_ls_exit[row] - dynamic.hs_exit[row];
                            let d1_entry = dynamic.dq_ls_entry[row] - dynamic.hs_entry[row];
                            let d2_exit = dynamic.d2q_ls_exit[row] + dynamic.hs_exit[row];
                            let hs_gap = dynamic.hs_entry[row] - dynamic.hs_exit[row];
                            let d1_gap =
                                (dynamic.dq_ls_entry[row] - dynamic.dq_ls_exit[row]) - hs_gap;
                            let d2_gap =
                                (dynamic.d2q_ls_entry[row] - dynamic.d2q_ls_exit[row]) + hs_gap;
                            coefficient = -(paired_s1[row] * d2_exit
                                + paired_s2[row] * d1_exit * d1_exit)
                                + entry_stack[1] * d2_gap
                                + entry_stack[2] * d1_gap * (d1_entry + d1_exit);
                        } else if block == 1 && self.x_threshold_entry.is_none() {
                            // Threshold index is LINEAR in η_t (∂²u/∂η_t² = 0).
                            let d1 = dynamic.dq_t_exit[row];
                            coefficient = -(paired_s2[row] * d1 * d1);
                        }
                    }
                    row_slots[slot] = match row_mask {
                        Some(mask) => coefficient * mask[row],
                        None => coefficient,
                    };
                }
                Ok(())
            })?;

        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing survival-LS joint block offsets".to_string())?;
        if offsets.len() != 4 {
            return Err(SurvivalLocationScaleError::InternalInvariant {
                reason: format!(
                    "packed survival-LS plan expected three coefficient blocks, got {}",
                    offsets.len().saturating_sub(1)
                ),
            }
            .into());
        }

        let selected = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                target == SlsCoefficientHessianTarget::DenseFull
                    || group.left_channel / 3 == group.right_channel / 3
            })
            .collect::<Vec<_>>();
        let products = selected
            .into_par_iter()
            .map(|(slot, group)| {
                let left =
                    designs[group.left_channel].expect("active survival-LS pair has a left design");
                let right = designs[group.right_channel]
                    .expect("active survival-LS pair has a right design");
                let weights = slots.row(slot).to_owned();
                weighted_crossprod_dense_with_parallelism(left, &weights, right, faer::Par::Seq)
                    .map(|product| (group, product))
            })
            .collect::<Result<Vec<_>, String>>()?;

        match target {
            SlsCoefficientHessianTarget::DenseFull => {
                let mut dense = Array2::<f64>::zeros((p_total, p_total));
                for (group, product) in products {
                    let left_block = group.left_channel / 3;
                    let right_block = group.right_channel / 3;
                    let (left_start, left_end) = (offsets[left_block], offsets[left_block + 1]);
                    let (right_start, right_end) = (offsets[right_block], offsets[right_block + 1]);
                    dense
                        .slice_mut(s![left_start..left_end, right_start..right_end])
                        .scaled_add(1.0, &product);
                    if left_block != right_block {
                        dense
                            .slice_mut(s![right_start..right_end, left_start..left_end])
                            .scaled_add(1.0, &product.t());
                    } else if group.left_channel != group.right_channel {
                        dense
                            .slice_mut(s![left_start..left_end, right_start..right_end])
                            .scaled_add(1.0, &product.t());
                    }
                }
                Ok(SlsCoefficientHessian::DenseFull(dense))
            }
            SlsCoefficientHessianTarget::BlockDiagonal => {
                let mut blocks = (0..3)
                    .map(|block| {
                        Array2::<f64>::zeros((
                            offsets[block + 1] - offsets[block],
                            offsets[block + 1] - offsets[block],
                        ))
                    })
                    .collect::<Vec<_>>();
                for (group, product) in products {
                    let block = group.left_channel / 3;
                    blocks[block].scaled_add(1.0, &product);
                    if group.left_channel != group.right_channel {
                        blocks[block].scaled_add(1.0, &product.t());
                    }
                }
                Ok(SlsCoefficientHessian::BlockDiagonal(blocks))
            }
        }
    }

    #[inline]
    pub(crate) fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.x_time_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        p_total - p_w..p_total
    }

    pub(crate) fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.time_linear_constraints.as_ref() else {
            // No time constraints. With the rank-1 unit-log-t warp pin (#892) the
            // time block has ZERO free coefficients and its monotone warp is a
            // fixed positive offset (X' z_norm = 1/t > 0), so there is no
            // derivative-guard half-space to cap against — the step is uncapped.
            // (Every constrained time block, reduced or flexible, carries ≥1
            // column and a guard, so this `None` arises only for the pinned
            // empty block.)
            return Ok(None);
        };
        crate::marginal_slope_shared::feasible_step_fraction(constraints, beta, delta)
            .map(Some)
            .map_err(|error| map_barrier_step_error("time block", error))
    }

    /// Largest feasible fraction of `delta` inside the link-wiggle block's
    /// `beta >= 0` cone.
    ///
    /// The cone is the block's DECLARED
    /// [`block_linear_constraints`](gam_model_api::CustomFamily::block_linear_constraints)
    /// — the very system the blockwise QP enforces — built from the one
    /// constructor so the barrier hook and the QP cannot disagree about which
    /// points are feasible. Until gam#2719 this was a hand-rolled coordinate
    /// loop that re-derived the cone with its own rule: it rejected the iterate
    /// at an ABSOLUTE `CONSTRAINT_NONNEGATIVITY_REL_TOL` (despite the `REL` in
    /// the name), and gave the step no tolerance at all. On the linkwiggle
    /// witness fit the seed sits at `beta ≡ 0`, exactly on every face, so a
    /// Newton drift of `-3.3e-18` produced `alpha = 0` — 379 refusals across
    /// six seeds, 314 of them of steps whose endpoint violated nothing at the
    /// solver's declared `1e-8`.
    pub(crate) fn max_feasible_link_wiggle_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = crate::wiggle::monotone_wiggle_nonnegative_system(beta.len())
        else {
            // A zero-width link-wiggle block declares no constraints either, so
            // there is no cone to step inside.
            return Ok(None);
        };
        crate::marginal_slope_shared::feasible_step_fraction(&constraints, beta, delta)
            .map(Some)
            .map_err(|error| map_barrier_step_error("linkwiggle block", error))
    }

    #[inline]
    pub(crate) fn expected_blocks(&self) -> usize {
        if self.x_link_wiggle.is_some() { 4 } else { 3 }
    }

    #[inline]
    pub(crate) fn joint_block_dims(&self) -> Vec<usize> {
        let mut dims = vec![
            self.x_time_entry.ncols(),
            self.x_threshold.ncols(),
            self.x_log_sigma.ncols(),
        ];
        if let Some(xw) = self.x_link_wiggle.as_ref() {
            dims.push(xw.ncols());
        }
        dims
    }

    pub(crate) fn validate_joint_specs(
        &self,
        specs: &[ParameterBlockSpec],
        context: &str,
    ) -> Result<(), String> {
        let dims = self.joint_block_dims();
        if specs.len() != dims.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{context} expects {} specs, got {}",
                    dims.len(),
                    specs.len()
                ),
            }
            .into());
        }
        for (block_idx, (spec, expected_width)) in specs.iter().zip(dims.iter()).enumerate() {
            let width = spec.design.ncols();
            if width != *expected_width {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "{context} spec {block_idx} width mismatch: got {width}, expected {expected_width}"
                    ),
                }
                .into());
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn joint_block_offsets(&self) -> Vec<usize> {
        let dims = self.joint_block_dims();
        let mut offsets = Vec::with_capacity(dims.len() + 1);
        offsets.push(0);
        let mut acc = 0usize;
        for dim in dims {
            acc += dim;
            offsets.push(acc);
        }
        offsets
    }

    pub(crate) fn wiggle_geometry(
        &self,
        q0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) = (self.wiggle_knots.as_ref(), self.wiggle_degree) else {
            return Ok(None);
        };
        let basis = survival_wiggle_basis_with_options(q0, knots, degree, BasisOptions::value())?;
        let basis_d1 = survival_wiggle_basis_with_options(
            q0,
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let basis_d2 = survival_wiggle_basis_with_options(
            q0,
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let basis_d3 = survival_wiggle_third_basis(q0, knots, degree)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
            || basis_d3.ncols() != beta_w.len()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival linkwiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    basis_d3.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
        }))
    }

    pub(crate) fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let stack = build_survival_location_scale_time_wiggle_basis_stack(
            h0,
            knots,
            degree,
            beta_w.len(),
            "fitting",
        )?;
        let dq = fast_av(&stack.d1, &beta_w) + 1.0;
        let d2 = fast_av(&stack.d2, &beta_w);
        let d3 = fast_av(&stack.d3, &beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis: stack.value,
            basis_d1: stack.d1,
            basis_d2: stack.d2,
            dq_dq0: dq,
            d2q_dq02: d2,
            d3q_dq03: d3,
        }))
    }

    /// Returns
    /// `(h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry,
    ///   eta_t_deriv_exit, eta_ls_deriv_exit, etaw)`.
    ///
    /// The time block eta is stored as `[exit; entry; derivative_exit]` to
    /// match the stacked design, but callers consume `(entry, exit, deriv)`.
    /// For time-invariant blocks, `eta_t_entry == eta_t_exit` and likewise for ls.
    /// For time-varying threshold/log-sigma blocks, the block eta is 3n long:
    /// `[exit; entry; derivative_exit]`.
    /// The solver's ParameterBlockSpec uses the EXIT value design first.
    pub(crate) fn validate_joint_states<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<
        (
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            Option<ndarray::ArrayView1<'a, f64>>,
            Option<ndarray::ArrayView1<'a, f64>>,
            Option<&'a Array1<f64>>,
        ),
        String,
    > {
        crate::block_layout::block_count::validate_block_count::<SurvivalLocationScaleError>(
            "SurvivalLocationScaleFamily",
            self.expected_blocks(),
            block_states.len(),
        )?;
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t_raw = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls_raw = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = self
            .x_link_wiggle
            .as_ref()
            .map(|_| &block_states[Self::BLOCK_LINK_WIGGLE].eta);
        if eta_time.len() != 3 * n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale time eta length mismatch: got {}, expected {}",
                    eta_time.len(),
                    3 * n
                ),
            }
            .into());
        }
        // For time-varying blocks the stacked design is
        // [exit_design; entry_design; derivative_exit_design], giving eta of
        // length 3n. For time-invariant blocks eta is length n.
        let (eta_t_exit, eta_t_entry, eta_t_deriv_exit) = if self.x_threshold_entry.is_some() {
            if eta_t_raw.len() != 3 * n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time-varying threshold eta length mismatch: got {}, expected {}",
                        eta_t_raw.len(),
                        3 * n
                    ),
                }
                .into());
            }
            (
                eta_t_raw.slice(s![0..n]),
                eta_t_raw.slice(s![n..2 * n]),
                Some(eta_t_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_t_raw.len() != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "threshold eta length mismatch: got {}, expected {n}",
                        eta_t_raw.len()
                    ),
                }
                .into());
            }
            (eta_t_raw.slice(s![0..n]), eta_t_raw.slice(s![0..n]), None)
        };
        let (eta_ls_exit, eta_ls_entry, eta_ls_deriv_exit) = if self.x_log_sigma_entry.is_some() {
            if eta_ls_raw.len() != 3 * n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time-varying log-sigma eta length mismatch: got {}, expected {}",
                        eta_ls_raw.len(),
                        3 * n
                    ),
                }
                .into());
            }
            (
                eta_ls_raw.slice(s![0..n]),
                eta_ls_raw.slice(s![n..2 * n]),
                Some(eta_ls_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_ls_raw.len() != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "log-sigma eta length mismatch: got {}, expected {n}",
                        eta_ls_raw.len()
                    ),
                }
                .into());
            }
            (eta_ls_raw.slice(s![0..n]), eta_ls_raw.slice(s![0..n]), None)
        };
        if let Some(w) = etaw
            && w.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale wiggle eta length mismatch: got {}, expected {n}",
                    w.len()
                ),
            }
            .into());
        }
        // The time block's solver design stacks `[entry; exit; derivative_exit]`
        // (see `prepare.rs`'s `MultiChannelOperator::new`), so the stacked time
        // eta is laid out `[entry(0..n); exit(n..2n); deriv(2n..3n)]`. The first
        // return slot is `h_entry`, the second is `h_exit` (gam#1396): a prior
        // revision read the entry channel from `eta_time[n..2*n]` and the exit
        // channel from `eta_time[0..n]`, transposing the two so the exit-time
        // index was evaluated at the entry predictor and vice versa. That swap
        // left the *value* path self-consistent (every consumer saw the same
        // transposed pair) but mis-paired each index with its design Jacobian
        // (`time_jac_entry`/`time_jac_exit`), so the time-block gradient/Hessian
        // disagreed with a finite-difference of the likelihood whenever the
        // entry and exit designs differ — and the structural-time monotonicity
        // guard saw the wrong exit derivative.
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
            eta_time.slice(s![2 * n..3 * n]),
            eta_t_exit,
            eta_ls_exit,
            eta_t_entry,
            eta_ls_entry,
            eta_t_deriv_exit,
            eta_ls_deriv_exit,
            etaw,
        ))
    }

    /// Per-row NLL gradient and curvature with respect to the three additive
    /// time-block offset channels `(o_E, o_X, o_D)` (entry / exit / derivative-
    /// at-exit). The baseline configuration enters the location-scale fit
    /// **only** through these three offsets, so contracting these residuals
    /// against `∂o/∂θ_baseline` gives the analytic θ-gradient of the
    /// unpenalized NLL at converged β (envelope theorem on the penalized
    /// objective; the penalty has no θ dependence).
    ///
    /// Algebra. With `ell_i = w_i[d(log f(u1) + log g − eta_ls) + (1-d) log S(u1) −
    /// log S(u0)]`, `u0 = s0·h0 + q0`, `u1 = s1·h1 + q1` and
    /// `g = (d_raw − h1·eta_ls') + qdot1` (`s = e^{−eta_ls}`: the scale divides the
    /// whole residual, and `du1/dt = s1·g`, #2695), each offset enters its time
    /// channel additively, so the map `J` from `(o_E, o_X, o_D)` to `(u0, u1, g)` is
    /// linear:
    ///
    ///   J = [[s0, 0, 0], [0, s1, 0], [0, −eta_ls', 1]].
    ///
    /// The row likelihood factors through the functionally independent `u0`,
    /// `u1`, `g`, whose NLL partials are
    ///
    ///   ∂(−ell_i)/∂u0 = − w_i r(u0)
    ///   ∂(−ell_i)/∂u1 = − w_i [d ψ(u1) − (1−d) r(u1)]
    ///   ∂(−ell_i)/∂g  = − w_i d / g                                (event-row only)
    ///
    /// with a diagonal index-space Hessian `C` (`− w_i r'(u0)`, `− w_i [d ψ'(u1) −
    /// (1−d) r'(u1)]`, `w_i d / g²`). The offset residual is `Jᵀ ∂(−ell)/∂(u0,u1,g)`
    /// and the offset curvature `Jᵀ C J`, which couples `o_X` and `o_D` exactly
    /// when the scale is time-varying.
    ///
    /// The fields `grad_time_eta_*` / `h_time_*` produced by
    /// [`Self::row_derivatives`] are log-likelihood (not NLL) partials in the
    /// index channels `(u0, u1, g)`, stored as `+∂ℓ`/`+∂²ℓ`, so the
    /// NLL gradient/curvature negates each **uniformly**. This site delegates
    /// that to [`SurvivalRowDerivatives::time_channel_nll_gradient`] /
    /// [`SurvivalRowDerivatives::time_channel_nll_curvature_diag`], which own
    /// the sign in one place (gam#1396 — a prior `+h_time_d` outlier here and
    /// in the joint assembler flipped the event-Jacobian self-term).
    pub(crate) fn offset_channel_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), SurvivalLocationScaleError> {
        let n = self.n;
        // Missing fitted state means the row likelihood geometry is
        // undefined. Returning zeros would assert a false stationary point to
        // the outer baseline optimizer and manufacture convergence.
        require_fitted_block_geometry(
            block_states,
            "SurvivalLocationScaleFamily::offset_channel_geometry",
        )?;
        let dynamic = self.build_dynamic_geometry(block_states)?;

        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        let mut derivative = Array1::<f64>::zeros(n);
        let mut curvatures = vec![[[0.0_f64; 3]; 3]; n];

        let rows = (0..n)
            .into_par_iter()
            .map(
                |i| -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    let state = self.row_predictor_state_at(&dynamic, i);
                    let Some(row) = self.row_derivatives(i, state)? else {
                        // `row_derivatives` returns `None` only for a
                        // non-positive-weight observation. Numerical geometry
                        // failures on positive-weight rows propagate as errors.
                        return Ok((i, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    };
                    // NLL gradient + curvature in the index channels (u0, u1, g).
                    // Both helpers own the `-∂ℓ`/`-∂²ℓ` sign so the channels are
                    // negated uniformly (gam#1396); the index-space curvature is
                    // diagonal because (u0, u1, g) are functionally independent.
                    let [g_u0, g_u1, g_g] = row.time_channel_nll_gradient();
                    let [c_u0, c_u1, c_g] = row.time_channel_nll_curvature_diag();
                    // Pull back through J (see the doc above).
                    let s0 = dynamic.inv_sigma_entry[i];
                    let s1 = dynamic.inv_sigma_exit[i];
                    let j_gx = -dynamic.eta_ls_deriv_exit[i];
                    let r_entry = s0 * g_u0;
                    let r_exit = s1 * g_u1 + j_gx * g_g;
                    let r_deriv = g_g;
                    let mut curv = [[0.0_f64; 3]; 3];
                    curv[0][0] = s0 * s0 * c_u0;
                    curv[1][1] = s1 * s1 * c_u1 + j_gx * j_gx * c_g;
                    curv[1][2] = j_gx * c_g;
                    curv[2][1] = curv[1][2];
                    curv[2][2] = c_g;
                    Ok((i, r_entry, r_exit, r_deriv, curv))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        for (i, r_entry, r_exit, r_deriv, curv) in rows {
            entry[i] = r_entry;
            exit[i] = r_exit;
            derivative[i] = r_deriv;
            curvatures[i] = curv;
        }

        Ok((
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
                // Location-scale has no interval upper-bound channel.
                right: Array1::<f64>::zeros(n),
            },
            OffsetChannelCurvatures { rows: curvatures },
        ))
    }

    /// Exact data-fit gradient `Σ_i ∂ℓ_i/∂θ_link` of the unpenalized
    /// log-likelihood with respect to the inverse-link parameters θ_link
    /// (SAS `(ε, log δ)`, BetaLogistic `(ε, log δ)`, or Mixture `ρ`), holding
    /// the fitted β and λ fixed.
    ///
    /// The per-row log-likelihood is
    ///   ℓ_i = w_i·( event_mix(d_i, logφ(u1_i) + log g_i, log S(u1_i)) − log S(u0_i) ),
    /// where `u0 = hs0 + q0` and `u1 = hs1 + q1` (`hs = h·e^{−eta_ls}`) are the standardized residuals
    /// the inverse link evaluates (entry/exit), `log g` is the time-derivative
    /// Jacobian (link-independent), and the link enters ONLY through the scalar
    /// `log S(u) = log(1 − μ(u;θ))` and `log φ(u) = log d1(u;θ)` terms. Hence
    ///   ∂(log S)/∂θ = −(∂μ/∂θ)/S,   ∂(log φ)/∂θ = (∂d1/∂θ)/d1,
    /// with `S` the stable survival complement of `μ = jet.mu` that the row kernel's
    /// generic arm uses, `d1 = jet.d1`, and the parameter partials
    /// `∂μ/∂θ`, `∂d1/∂θ` supplied analytically by
    /// [`InverseLinkKernel::param_partials`]. The higher-order ratio/pdf
    /// derivatives (r, dr, …, fppp) carry the inner-Newton curvature only and do
    /// NOT appear in the scalar ℓ, so the data-fit θ-gradient needs only the
    /// `(μ, d1)` jet components and their param partials — all exact.
    ///
    /// At the converged β̂ the envelope theorem makes this the exact θ-gradient
    /// of the profile penalized NLL `−ℓ + ½βᵀSβ` (β profiled out; the penalty
    /// has no θ_link dependence). Returns a length-`n_link_params` vector
    /// (`∂(−ℓ)/∂θ` so it matches the profile-cost sign), or `None` when the
    /// inverse link carries no free parameters.
    pub(crate) fn link_param_data_fit_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, SurvivalLocationScaleError> {
        use gam_solve::mixture_link::{InverseLinkKernel, LinkParamPartials};
        let n = self.n;
        require_fitted_block_geometry(
            block_states,
            "SurvivalLocationScaleFamily::link_param_data_fit_gradient",
        )?;
        // ∂(log S)/∂θ and ∂(log φ)/∂θ contributions per row are accumulated
        // into a θ-length vector. Probe the parameter count from the link's
        // partials at a finite argument; `None` ⇒ no free link parameters.
        let probe = self
            .inverse_link
            .param_partials(0.0)
            .map_err(|e| format!("inverse-link param partials probe failed: {e}"))?;
        let n_theta = match &probe {
            None => return Ok(None),
            Some(LinkParamPartials::Sas(_)) => 2,
            Some(LinkParamPartials::Mixture(m)) => m.djet_drho.len(),
        };
        if n_theta == 0 {
            return Ok(None);
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        // ∂(log S)/∂θ = −(∂μ/∂θ)/S at argument u (S the stable complement of μ);
        // ∂(log φ)/∂θ = (∂d1/∂θ)/d1 at argument u.
        let dlog_survival_dtheta = |u: f64| -> Result<Vec<f64>, String> {
            let partials = self
                .inverse_link
                .param_partials(u)
                .map_err(|e| format!("inverse-link survival param partials failed: {e}"))?
                .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
            // The complement the value path takes, not `1 − μ` floored: in the far
            // tail `1 − μ` rounds to zero while `S` stays representable.
            let s = inverse_link_survival_probvalue(&self.inverse_link, u);
            if !(s.is_finite() && s > 0.0 && s <= 1.0) {
                return Err(format!(
                    "inverse-link survival probability must lie in (0,1] for the θ-gradient, \
                     got {s} at u={u}"
                ));
            }
            let map = |dmu: f64| -dmu / s;
            Ok(match partials {
                LinkParamPartials::Sas(p) => {
                    vec![map(p.djet_depsilon.mu), map(p.djet_dlog_delta.mu)]
                }
                LinkParamPartials::Mixture(p) => p.djet_drho.iter().map(|j| map(j.mu)).collect(),
            })
        };
        let dlog_pdf_dtheta = |u: f64| -> Result<Vec<f64>, String> {
            let partials = self
                .inverse_link
                .param_partials(u)
                .map_err(|e| format!("inverse-link pdf param partials failed: {e}"))?
                .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
            let jet = self
                .inverse_link
                .jet(u)
                .map_err(|e| format!("inverse-link jet failed at u={u}: {e}"))?;
            let f = jet.d1;
            if !(f.is_finite() && f > 0.0) {
                return Err(format!(
                    "inverse-link pdf (d1) must be finite positive for θ-gradient, got {f} at u={u}"
                ));
            }
            let map = |dd1: f64| dd1 / f;
            Ok(match partials {
                LinkParamPartials::Sas(p) => {
                    vec![map(p.djet_depsilon.d1), map(p.djet_dlog_delta.d1)]
                }
                LinkParamPartials::Mixture(p) => p.djet_drho.iter().map(|j| map(j.d1)).collect(),
            })
        };
        // Accumulate ∂(−ℓ)/∂θ = −Σ_i w_i·( event_mix(d, ∂logφ(u1), ∂logS(u1))
        //                                    − ∂logS(u0) ).
        let mut grad = Array1::<f64>::zeros(n_theta);
        for i in 0..n {
            let w = self.w[i];
            if w <= 0.0 {
                continue;
            }
            let d = self.validated_event_target(i)?;
            let u0 = dynamic.hs_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.hs_exit[i] + dynamic.q_exit[i];
            let dls_u0 = dlog_survival_dtheta(u0)?;
            // Entry channel always contributes (left-truncation term −log S(u0)).
            for k in 0..n_theta {
                grad[k] += w * dls_u0[k];
            }
            if d <= 0.0 {
                // Censored: +log S(u1).
                let dls_u1 = dlog_survival_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * dls_u1[k];
                }
            } else if d >= 1.0 {
                // Event: +log φ(u1) (log g is link-independent).
                let dlp_u1 = dlog_pdf_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * dlp_u1[k];
                }
            } else {
                // Fractional event weight: mix both branches.
                let dls_u1 = dlog_survival_dtheta(u1)?;
                let dlp_u1 = dlog_pdf_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * (d * dlp_u1[k] + (1.0 - d) * dls_u1[k]);
                }
            }
        }
        Ok(Some(grad))
    }

    /// First-order terms of inverse-link shape axis `axis` as a family-owned
    /// hyper axis (#2904): `objective_psi = ∂(−ℓ)/∂θ`, `score_psi = ∂θ ∇_β(−ℓ)`
    /// and `hessian_psi = ∂θ ∇²_β(−ℓ)`, at fixed β.
    ///
    /// The link enters the row NLL only through the residual-distribution
    /// stacks `sls_row_program` composes, and a supplied compose is linear in
    /// its stack: value `s₀`, gradient `s₁·u_a`, Hessian `s₁·u_ab + s₂·u_a·u_b`.
    /// So the row gradient's and Hessian's θ-partials are the same order-2
    /// program evaluated on the θ-partials of each stack's first two derivative
    /// entries, pulled back through the row Jacobian. The value slots and the
    /// link-independent `log g` stack are zero in that evaluation, and the
    /// objective partial is [`Self::link_param_data_fit_gradient`]. `None` when
    /// the link has no free parameters.
    pub(crate) fn link_param_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let Some(objective) = self.link_param_data_fit_gradient(block_states)? else {
            return Ok(None);
        };
        if axis >= objective.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "inverse-link shape axis {axis} is out of range for a link with {} free \
                     parameters",
                    objective.len()
                ),
            }
            .into());
        }
        if self.x_link_wiggle.is_some() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "inverse-link shape hyper axes are not available with a link wiggle block"
                    .to_string(),
            }
            .into());
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        // Shape parameters exist only for links whose derivative log-rescale is
        // the identity; it departs from zero for CLogLog alone.
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let p_total = *kernel
            .offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        let mut score_psi = Array1::<f64>::zeros(p_total);
        let mut hessian_psi = Array2::<f64>::zeros((p_total, p_total));
        for row in 0..self.n {
            let Some((primary, row_kernel)) = kernel.row_nll_inputs_opt(row)? else {
                continue;
            };
            let partial_kernel = self.link_param_partial_row_kernel(
                &row_kernel,
                dynamic.hs_entry[row] + dynamic.q_entry[row],
                dynamic.hs_exit[row] + dynamic.q_exit[row],
                axis,
            )?;
            let (_, gradient, hessian) = sls_row_vgh_generated(&primary, &partial_kernel);
            crate::row_kernel::RowKernel::<SLS_ROW_K>::jacobian_transpose_action(
                &kernel,
                row,
                &gradient,
                score_psi
                    .as_slice_mut()
                    .ok_or_else(|| "link-shape score accumulator is not contiguous".to_string())?,
            );
            crate::row_kernel::RowKernel::<SLS_ROW_K>::add_pullback_hessian(
                &kernel,
                row,
                &hessian,
                &mut hessian_psi,
            );
        }
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: objective[axis],
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    /// Mixed coefficient drift `D_β H_θ[u]` of the observed information along
    /// inverse-link shape axis `axis` over `rows` (#2904): the β-directional
    /// derivative of `link_param_joint_psi_terms`' `hessian_psi`, which the
    /// explicit-ψ Jeffreys score correction and curvature drift read along every
    /// coefficient axis. The row program's contracted third derivative on the
    /// θ-partial row kernel is pulled back through the same row Jacobian that
    /// `row_kernel_directional_derivative` uses. `None` when the link has no free
    /// parameters.
    pub(crate) fn link_param_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta_flat: &[f64],
        rows: &crate::row_kernel::RowSet,
    ) -> Result<Option<Array2<f64>>, String> {
        use gam_solve::mixture_link::{InverseLinkKernel, LinkParamPartials};
        let axis_count = match self
            .inverse_link
            .param_partials(0.0)
            .map_err(|e| format!("inverse-link param partials probe failed: {e}"))?
        {
            None => return Ok(None),
            Some(LinkParamPartials::Sas(_)) => 2,
            Some(LinkParamPartials::Mixture(partials)) => partials.djet_drho.len(),
        };
        if axis_count == 0 {
            return Ok(None);
        }
        if axis >= axis_count {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "inverse-link shape axis {axis} is out of range for a link with {axis_count} \
                     free parameters"
                ),
            }
            .into());
        }
        if self.x_link_wiggle.is_some() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "inverse-link shape hyper axes are not available with a link wiggle block"
                    .to_string(),
            }
            .into());
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let p_total = *kernel
            .offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "inverse-link shape drift direction has length {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }
        let drift = rows.par_try_reduce_fold(
            self.n,
            || Array2::<f64>::zeros((p_total, p_total)),
            |mut acc, row, weight| -> Result<Array2<f64>, String> {
                let Some((primary, row_kernel)) = kernel.row_nll_inputs_opt(row)? else {
                    return Ok(acc);
                };
                let partial_kernel = self.link_param_partial_row_kernel(
                    &row_kernel,
                    dynamic.hs_entry[row] + dynamic.q_entry[row],
                    dynamic.hs_exit[row] + dynamic.q_exit[row],
                    axis,
                )?;
                let direction = crate::row_kernel::RowKernel::<SLS_ROW_K>::jacobian_action(
                    &kernel,
                    row,
                    d_beta_flat,
                );
                let mut third = sls_row_third_generated(&primary, &partial_kernel, &direction);
                if weight != 1.0 {
                    for entry in third.iter_mut().flatten() {
                        *entry *= weight;
                    }
                }
                crate::row_kernel::RowKernel::<SLS_ROW_K>::add_pullback_hessian(
                    &kernel,
                    row,
                    &third,
                    &mut acc,
                );
                Ok(acc)
            },
            |left, right| Ok(left + right),
        )?;
        Ok(Some(drift))
    }

    /// The θ-partial of one row's residual-distribution stacks along shape axis
    /// `axis`, as a row kernel carrying that row's weight and event target
    /// (#2904). A supplied compose is linear in its stack at every order, so the
    /// row program run on this kernel returns the θ-partials of the row gradient,
    /// Hessian and contracted third derivative. The value slots, the stable pair
    /// values and the link-independent `log g` stack are zero, and no program
    /// below fourth order reads the fourth entries.
    fn link_param_partial_row_kernel(
        &self,
        row_kernel: &SurvivalExactRowKernel,
        entry_index: f64,
        exit_index: f64,
        axis: usize,
    ) -> Result<SurvivalExactRowKernel, String> {
        let (entry_survival, _) = self.link_param_stack_partials(entry_index, axis)?;
        let (exit_survival, exit_log_pdf) = self.link_param_stack_partials(exit_index, axis)?;
        Ok(SurvivalExactRowKernel {
            w: row_kernel.w,
            d: row_kernel.d,
            log_s0: 0.0,
            r0: entry_survival[0],
            dr0: entry_survival[1],
            ddr0: entry_survival[2],
            dddr0: 0.0,
            log_s1: 0.0,
            r1: exit_survival[0],
            dr1: exit_survival[1],
            ddr1: exit_survival[2],
            dddr1: 0.0,
            logphi1: 0.0,
            dlogphi1: exit_log_pdf[0],
            d2logphi1: exit_log_pdf[1],
            d3logphi1: exit_log_pdf[2],
            d4logphi1: 0.0,
            log_pdf1_minus_log_s0: 0.0,
            log_s1_minus_log_s0: 0.0,
            log_g: 0.0,
            d_log_g: 0.0,
            d2_log_g: 0.0,
            d3_log_g: 0.0,
            d4_log_g: 0.0,
        })
    }

    /// θ-partials of the first three derivative entries of the survival stack
    /// (`r = −(ln S)′`, `dr = −(ln S)″`, `ddr = −(ln S)‴`) and of the log-density
    /// stack (`(ln f)′`, `(ln f)″`, `(ln f)‴`) at index `u`, for one inverse-link
    /// shape axis (#2904), in that order. Both stacks are `ln` composed on a jet
    /// `(v, x′, x″, x‴)`: `(S, −f, −f′, −f″)` and `(f, f′, f″, f‴)` with `f = F′`,
    /// whose θ-partials `(dv, dx′, dx″, dx‴)` are `(−∂θF, −∂θf, −∂θf′, −∂θf″)` and
    /// `(∂θf, ∂θf′, ∂θf″, ∂θf‴)`. `S` is the stable complement the kernel's generic
    /// arm uses.
    fn link_param_stack_partials(
        &self,
        u: f64,
        axis: usize,
    ) -> Result<([f64; 3], [f64; 3]), String> {
        use gam_solve::mixture_link::{InverseLinkKernel, LinkParamPartials};
        let jet = inverse_link_jet_for_inverse_link(&self.inverse_link, u)
            .map_err(|e| format!("inverse link evaluation failed at eta={u}: {e}"))?;
        let survival = inverse_link_survival_probvalue(&self.inverse_link, u);
        if !(survival.is_finite() && survival > 0.0 && survival <= 1.0) {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: format!(
                    "inverse-link survival probability must lie in (0,1], got {survival} at eta={u}"
                ),
            }
            .into());
        }
        let pdf = jet.d1;
        if !(pdf.is_finite() && pdf > 0.0) {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: format!("inverse-link pdf must be finite and positive, got {pdf} at eta={u}"),
            }
            .into());
        }
        let partials = self
            .inverse_link
            .param_partials(u)
            .map_err(|e| format!("inverse-link param partials failed at eta={u}: {e}"))?
            .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
        let djet = match partials {
            LinkParamPartials::Sas(p) => match axis {
                0 => p.djet_depsilon,
                1 => p.djet_dlog_delta,
                _ => {
                    return Err(format!(
                        "an SAS-family link has two shape axes, got axis {axis}"
                    ));
                }
            },
            LinkParamPartials::Mixture(p) => *p.djet_drho.get(axis).ok_or_else(|| {
                format!(
                    "a mixture link has {} shape axes, got axis {axis}",
                    p.djet_drho.len()
                )
            })?,
        };
        let pdf_third = inverse_link_pdfthird_derivative_for_inverse_link(&self.inverse_link, u)
            .map_err(|e| format!("inverse-link pdf third derivative failed at eta={u}: {e}"))?;
        let pdf_third_partial =
            gam_solve::mixture_link::inverse_link_pdfthird_derivative_param_partials(
                &self.inverse_link,
                u,
            )
            .map_err(|e| {
                format!("inverse-link pdf third derivative partials failed at eta={u}: {e}")
            })?
            .and_then(|third_partials| third_partials.get(axis).copied())
            .ok_or_else(|| {
                format!("inverse link has no pdf third derivative partial on shape axis {axis}")
            })?;
        let survival_partials = Self::log_jet_theta_partials(
            survival,
            [-pdf, -jet.d2, -jet.d3],
            -djet.mu,
            [-djet.d1, -djet.d2, -djet.d3],
        );
        let log_pdf_partials = Self::log_jet_theta_partials(
            pdf,
            [jet.d2, jet.d3, pdf_third],
            djet.d1,
            [djet.d2, djet.d3, pdf_third_partial],
        );
        Ok((survival_partials.map(|value| -value), log_pdf_partials))
    }

    /// θ-partials of the first three entries of `ln x` for a one-variable jet
    /// `(v, x′, x″, x‴)` whose θ-partial jet is `(dv, dx′, dx″, dx‴)` (#2904). With
    /// `ρₖ = xₖ/v`: `(ln x)′ = ρ₁`, `(ln x)″ = ρ₂ − ρ₁²`, `(ln x)‴ = ρ₃ − 3ρ₁ρ₂ + 2ρ₁³`,
    /// and `∂θρₖ = (dxₖ − ρₖ·dv)/v`.
    #[inline]
    fn log_jet_theta_partials(v: f64, x: [f64; 3], dv: f64, dx: [f64; 3]) -> [f64; 3] {
        let ratio = x.map(|value| value / v);
        let d_ratio = [
            (dx[0] - ratio[0] * dv) / v,
            (dx[1] - ratio[1] * dv) / v,
            (dx[2] - ratio[2] * dv) / v,
        ];
        [
            d_ratio[0],
            d_ratio[1] - 2.0 * ratio[0] * d_ratio[0],
            d_ratio[2] - 3.0 * (d_ratio[0] * ratio[1] + ratio[0] * d_ratio[1])
                + 6.0 * ratio[0] * ratio[0] * d_ratio[0],
        ]
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<SurvivalJointPsiDirection>, String> {
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi direction expects {} blocks and derivative lists, got {} and {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }

        let n = self.n;
        let pt = self.x_threshold.ncols();
        let pls = self.x_log_sigma.ncols();
        let beta_t = &block_states[Self::BLOCK_THRESHOLD].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    let mut x_t_exit_psi = None;
                    let mut x_t_entry_psi = None;
                    let mut x_t_deriv_psi = None;
                    let mut x_ls_exit_psi = None;
                    let mut x_ls_entry_psi = None;
                    let mut x_ls_deriv_psi = None;
                    let mut x_t_exit_action = None;
                    let mut x_t_entry_action = None;
                    let mut x_t_deriv_action = None;
                    let mut x_ls_exit_action = None;
                    let mut x_ls_entry_action = None;
                    let mut x_ls_deriv_action = None;
                    let mut z_t_exit_psi = Array1::<f64>::zeros(n);
                    let mut z_t_entry_psi = Array1::<f64>::zeros(n);
                    let mut z_t_deriv_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_exit_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_entry_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_deriv_psi = Array1::<f64>::zeros(n);
                    match block_idx {
                        Self::BLOCK_THRESHOLD => {
                            let total_rows = if t_time_varying { 3 * n } else { n };
                            match resolve_custom_family_x_psi_map(
                                deriv,
                                total_rows,
                                pt,
                                0..total_rows,
                                "SurvivalLocationScaleFamily threshold",
                                &self.policy,
                            ).map_err(|error| error.to_string())? {
                                PsiDesignMap::First { action } => {
                                    if t_time_varying {
                                        let exit_action = action.slice_rows(0..n).map_err(|error| error.to_string())?;
                                        let entry_action = action.slice_rows(n..2 * n).map_err(|error| error.to_string())?;
                                        let deriv_action = action.slice_rows(2 * n..3 * n).map_err(|error| error.to_string())?;
                                        z_t_exit_psi = exit_action.forward_mul(beta_t.view());
                                        z_t_entry_psi = entry_action.forward_mul(beta_t.view());
                                        z_t_deriv_psi = deriv_action.forward_mul(beta_t.view());
                                        x_t_exit_action = Some(exit_action);
                                        x_t_entry_action = Some(entry_action);
                                        x_t_deriv_action = Some(deriv_action);
                                    } else {
                                        z_t_exit_psi = action.forward_mul(beta_t.view());
                                        z_t_entry_psi = z_t_exit_psi.clone();
                                        x_t_exit_action = Some(action.clone());
                                        x_t_entry_action = Some(action);
                                    }
                                }
                                PsiDesignMap::Dense { matrix } => {
                                    let (exit, entry, deriv) = split_survival_psi_design(
                                        &matrix,
                                        n,
                                        t_time_varying,
                                        "SurvivalLocationScaleFamily threshold",
                                    )?;
                                    z_t_exit_psi = fast_av(&exit, beta_t);
                                    z_t_entry_psi = fast_av(&entry, beta_t);
                                    if let Some(d) = deriv.as_ref() {
                                        z_t_deriv_psi = fast_av(d, beta_t);
                                    }
                                    x_t_exit_psi = Some(exit);
                                    x_t_entry_psi = Some(entry);
                                    x_t_deriv_psi = deriv;
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: "SurvivalLocationScaleFamily threshold: unexpected Second variant from _psi_map"
                                            .to_string(), }.into());
                                }
                            }
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            let total_rows = if ls_time_varying { 3 * n } else { n };
                            match resolve_custom_family_x_psi_map(
                                deriv,
                                total_rows,
                                pls,
                                0..total_rows,
                                "SurvivalLocationScaleFamily log-sigma",
                                &self.policy,
                            ).map_err(|error| error.to_string())? {
                                PsiDesignMap::First { action } => {
                                    if ls_time_varying {
                                        let exit_action = action.slice_rows(0..n).map_err(|error| error.to_string())?;
                                        let entry_action = action.slice_rows(n..2 * n).map_err(|error| error.to_string())?;
                                        let deriv_action = action.slice_rows(2 * n..3 * n).map_err(|error| error.to_string())?;
                                        z_ls_exit_psi = exit_action.forward_mul(beta_ls.view());
                                        z_ls_entry_psi = entry_action.forward_mul(beta_ls.view());
                                        z_ls_deriv_psi = deriv_action.forward_mul(beta_ls.view());
                                        x_ls_exit_action = Some(exit_action);
                                        x_ls_entry_action = Some(entry_action);
                                        x_ls_deriv_action = Some(deriv_action);
                                    } else {
                                        z_ls_exit_psi = action.forward_mul(beta_ls.view());
                                        z_ls_entry_psi = z_ls_exit_psi.clone();
                                        x_ls_exit_action = Some(action.clone());
                                        x_ls_entry_action = Some(action);
                                    }
                                }
                                PsiDesignMap::Dense { matrix } => {
                                    let (exit, entry, deriv) = split_survival_psi_design(
                                        &matrix,
                                        n,
                                        ls_time_varying,
                                        "SurvivalLocationScaleFamily log-sigma",
                                    )?;
                                    z_ls_exit_psi = fast_av(&exit, beta_ls);
                                    z_ls_entry_psi = fast_av(&entry, beta_ls);
                                    if let Some(d) = deriv.as_ref() {
                                        z_ls_deriv_psi = fast_av(d, beta_ls);
                                    }
                                    x_ls_exit_psi = Some(exit);
                                    x_ls_entry_psi = Some(entry);
                                    x_ls_deriv_psi = deriv;
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: "SurvivalLocationScaleFamily log-sigma: unexpected Second variant from _psi_map"
                                            .to_string(), }.into());
                                }
                            }
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(SurvivalJointPsiDirection {
                        x_t_exit_psi,
                        x_t_entry_psi,
                        x_t_deriv_psi,
                        x_ls_exit_psi,
                        x_ls_entry_psi,
                        x_ls_deriv_psi,
                        z_t_exit_psi,
                        z_t_entry_psi,
                        z_t_deriv_psi,
                        z_ls_exit_psi,
                        z_ls_entry_psi,
                        z_ls_deriv_psi,
                        x_t_exit_action,
                        x_t_entry_action,
                        x_t_deriv_action,
                        x_ls_exit_action,
                        x_ls_entry_action,
                        x_ls_deriv_action,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    /// The `(ln S, r, r', r'', r''')` stack of the negative log-survival
    /// `-ln S(η)` from the survival value `S` and the pdf jet
    /// `f = -S', f', f'', f'''`, for any inverse link.
    ///
    /// This is a one-variable composition, so it comes from the jet algebra:
    /// `S` as a `Tower4<1>` (its derivatives are the negated pdf jet) composed
    /// with the `ln` unary stack, negated. The hand quotient-rule chain it
    /// replaces (`r = f/S`, `r' = r² + f'/S`, `r'' = 2rr' + f''/S + f'f/S²`,
    /// and a third derivative whose own derivation comment contained the word
    /// "wait") served every non-closed-form link — LogLog, Cauchit, SAS,
    /// beta-logistic, mixture — with no independent witness: exactly the
    /// desync genus of #736/#932. Combinatorics belongs to the algebra;
    /// humans own the primitive stacks (`logwith_derivatives_positive`).
    #[inline]
    pub(crate) fn neglog_survival_stack_from_pdf_jet(
        s: f64,
        f: f64,
        fp: f64,
        fpp: f64,
        fppp: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let (log_s, d1, d2, d3, d4) = Self::log_stack_from_jet(s, -f, -fp, -fpp, -fppp);
        (log_s, -d1, -d2, -d3, -d4)
    }

    /// The `(ln x, (ln x)', (ln x)'', (ln x)''', (ln x)'''')` stack of the
    /// logarithm of a positive one-variable jet `x = (v, x', x'', x''', x'''')`:
    /// the jet composed with the `ln` unary stack at `v`. Both generic-link
    /// scalar stacks of this kernel (`-ln S`, `ln f`) are this composition and
    /// share it, so their fourth-order towers are never spelled by hand.
    #[inline]
    pub(crate) fn log_stack_from_jet(
        v: f64,
        d1: f64,
        d2: f64,
        d3: f64,
        d4: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let mut argument = gam_math::jet_tower::Tower4::<1>::zero();
        argument.v = v;
        argument.g[0] = d1;
        argument.h[0][0] = d2;
        argument.t3[0][0][0] = d3;
        argument.t4[0][0][0][0] = d4;
        let (log_v, l1, l2, l3, l4) = Self::logwith_derivatives_positive(v);
        let log_argument = argument.compose_unary([log_v, l1, l2, l3, l4]);
        (
            log_argument.v,
            log_argument.g[0],
            log_argument.h[0][0],
            log_argument.t3[0][0][0],
            log_argument.t4[0][0][0][0],
        )
    }

    /// Like [`Self::exact_log_pdf_derivatives_rescaled`] but with a log-scale shift
    /// on the derivative magnitudes.  For CLogLog the `exp(eta)` terms in
    /// the derivatives become `exp(eta - deriv_log_scale)`, and the constant
    /// term in `d/deta log f = 1 - exp(eta)` is scaled by the same factor.
    /// The function value is returned unshifted.
    pub(crate) fn exact_log_pdf_derivatives_rescaled(
        inverse_link: &InverseLink,
        eta: f64,
        deriv_log_scale: f64,
    ) -> Result<(f64, f64, f64, f64, f64), String> {
        match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => Ok((
                -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln(),
                -eta,
                -1.0,
                0.0,
                0.0,
            )),
            InverseLink::Standard(StandardLink::Logit) => {
                let mu = gam_solve::mixture_link::component_inverse_link_jet(
                    gam_problem::LinkComponent::Logit,
                    eta,
                )
                .mu;
                let w = mu * (1.0 - mu);
                Ok((
                    -softplus(eta) - softplus(-eta),
                    1.0 - 2.0 * mu,
                    -2.0 * w,
                    -2.0 * w * (1.0 - 2.0 * mu),
                    -2.0 * w * (1.0 - 6.0 * w),
                ))
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let t_val = eta.exp(); // for function value (may be Inf)
                let t_deriv = (eta - deriv_log_scale).exp(); // for derivatives
                let deriv_scale = (-deriv_log_scale).exp();
                Ok((
                    eta - t_val,
                    deriv_scale - t_deriv,
                    -t_deriv,
                    -t_deriv,
                    -t_deriv,
                ))
            }
            InverseLink::Standard(StandardLink::Identity) => Ok((0.0, 0.0, 0.0, 0.0, 0.0)),
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let f = jet.d1;
                if !(f.is_finite() && f > 0.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!(
                            "inverse-link pdf must be finite and positive, got {f} at eta={eta}"
                        ),
                    }
                    .into());
                }
                let fp = jet.d2;
                let fpp = jet.d3;
                let fppp = inverse_link_pdfthird_derivative_for_inverse_link(inverse_link, eta)
                    .map_err(|e| {
                        format!("inverse link third-derivative evaluation failed at eta={eta}: {e}")
                    })?;
                let fpppp = inverse_link_pdffourth_derivative(inverse_link, eta)?;
                // `ln f` composed on the pdf jet; the hand Bell-polynomial
                // expansion this replaces was the same tower spelled out.
                Ok(Self::log_stack_from_jet(f, fp, fpp, fppp, fpppp))
            }
        }
    }

    /// Survival log value and ratio derivatives, with the same log-scale shift
    /// on the derivative magnitudes as [`Self::exact_log_pdf_derivatives_rescaled`].
    /// For CLogLog the ratio derivatives are all `exp(eta)`, which enter the
    /// joint Hessian side-by-side with the pdf stack's `exp(eta − L)` terms:
    /// scaling one stack but not the other breaks the documented
    /// `H_scaled = exp(−L)·H_exact` contract on every censored or
    /// left-truncated row (their curvature would carry an extra `exp(L)`),
    /// corrupting the `logdet(H_exact) = logdet(H_scaled) + p·L` correction —
    /// and lets an unscaled `exp(eta)` overflow drop censored rows the pdf
    /// path would have kept.  The function value (`-exp(eta)` = `log S`) is
    /// returned unshifted, exactly like the pdf value channel.
    /// `deriv_log_scale` is only ever nonzero for CLogLog
    /// (see `hessian_deriv_log_rescale`), so the other links ignore it,
    /// mirroring the pdf evaluator.
    pub(crate) fn exact_survival_neglog_derivatives_fourth_rescaled(
        inverse_link: &InverseLink,
        eta: f64,
        deriv_log_scale: f64,
    ) -> Result<(f64, f64, f64, f64, f64), String> {
        match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => {
                let (log_s, r, dr, ddr, dddr) = probit_log_survival_and_ratio_derivatives(eta);
                Ok((log_s, r, dr, ddr, dddr))
            }
            InverseLink::Standard(StandardLink::Logit) => {
                let mu = gam_solve::mixture_link::component_inverse_link_jet(
                    gam_problem::LinkComponent::Logit,
                    eta,
                )
                .mu;
                let w = mu * (1.0 - mu);
                Ok((
                    -softplus(eta),
                    mu,
                    w,
                    w * (1.0 - 2.0 * mu),
                    w * (1.0 - 6.0 * w),
                ))
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let t_val = eta.exp(); // for function value (may be Inf)
                let t_deriv = (eta - deriv_log_scale).exp(); // for derivatives
                Ok((-t_val, t_deriv, t_deriv, t_deriv, t_deriv))
            }
            InverseLink::Standard(StandardLink::Identity) => {
                let s = 1.0 - eta;
                if !(s.is_finite() && s > 0.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!("identity-link survival invalid at eta={eta}: S={s}"),
                    }
                    .into());
                }
                let inv = s.recip();
                Ok((s.ln(), inv, inv * inv, 2.0 * inv.powi(3), 6.0 * inv.powi(4)))
            }
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let s = inverse_link_survival_probvalue(inverse_link, eta);
                if !(s.is_finite() && s > 0.0 && s <= 1.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure { reason: format!(
                        "inverse-link survival probability must lie in (0,1], got {s} at eta={eta}"
                    ) }.into());
                }
                let fppp = inverse_link_pdfthird_derivative_for_inverse_link(inverse_link, eta)
                    .map_err(|e| {
                        format!("inverse link third-derivative evaluation failed at eta={eta}: {e}")
                    })?;
                Ok(Self::neglog_survival_stack_from_pdf_jet(
                    s, jet.d1, jet.d2, jet.d3, fppp,
                ))
            }
        }
    }

    /// Fused CLogLog evaluator for the exit-row pair: returns the
    /// `(log_s, r, dr, ddr, dddr)` survival tuple and the
    /// `(logphi, d1, d2, d3, d4)` log-pdf tuple while computing the two
    /// expensive `exp` calls once.  This duplicates the CLogLog branches of
    /// `exact_survival_neglog_derivatives_fourth_rescaled` and
    /// `exact_log_pdf_derivatives_rescaled` to share their work.
    #[inline]
    pub(crate) fn clglog_exit_pair(
        u1: f64,
        deriv_log_scale: f64,
    ) -> ((f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64)) {
        let t_val = u1.exp();
        let t_deriv = (u1 - deriv_log_scale).exp();
        let deriv_scale = (-deriv_log_scale).exp();
        // Survival ratio derivatives share the same rescale as the pdf stack so
        // event and censored rows stay on one uniform exp(-L) Hessian scaling.
        let surv = (-t_val, t_deriv, t_deriv, t_deriv, t_deriv);
        let logpdf = (
            u1 - t_val,
            deriv_scale - t_deriv,
            -t_deriv,
            -t_deriv,
            -t_deriv,
        );
        (surv, logpdf)
    }

    /// Exact `log(x)` value and first four derivatives on the positive domain.
    pub(crate) fn logwith_derivatives_positive(x: f64) -> (f64, f64, f64, f64, f64) {
        assert!(
            x.is_finite() && x > 0.0,
            "log derivative kernel requires finite positive x: x={x}"
        );
        let inv = 1.0 / x;
        (
            x.ln(),
            inv,
            -inv * inv,
            2.0 * inv * inv * inv,
            -6.0 * inv * inv * inv * inv,
        )
    }

    /// `log g` continued below the modelling floor by its own Taylor series.
    ///
    /// The event Jacobian `g = dη/dt` must be positive for the model to be a
    /// model at all, and `log g` is neither bounded nor resolved as `g → 0`:
    /// `g` is reconstructed by a compensated difference `d_raw + qdot` whose
    /// own resolution is `roundoff_slack`. `derivative_guard` is the declared
    /// modelling floor for that quantity — the same constant
    /// `build_time_derivative_guard_constraints` uses as the right-hand side of
    /// the time block's linear feasibility cone — so it is the natural knot.
    ///
    /// Before gam#2695 the floor was applied to `g` and the whole derivative
    /// tower was then read at the FLOORED value, so on a floored row `log g`
    /// was constant in β while `d_log_g = 1/guard` reported a slope of `1e6`
    /// and `d2_log_g = -1/guard²` a curvature of `-1e12`. That is a value on
    /// one surface and a derivative on another: a trust-region model built on
    /// it cannot have `actual/(rhs·δ) → 1` at any step size, which is the
    /// fault class gam#2714 established and gam#2695 measures. The sibling
    /// Royston–Parmar arm states the same contract explicitly —
    /// `survival/base.rs`'s `stabilized_structural_derivative` returns the
    /// clamp's own slope and says "every consumer that differentiates through
    /// the structural derivative MUST scale its derivative-channel terms by
    /// `slope`" — and resolves it with a zero-slope clamp.
    ///
    /// A continuation is that same contract with a differentiable branch:
    ///
    /// ```text
    ///     Λ(g) = ln g                                          for g ≥ guard
    ///     Λ(g) = ln(guard) + x − x²/2 + x³/3 − x⁴/4            for g < guard
    ///     x    = (g − guard)/guard   (so x ≤ 0 on that branch)
    /// ```
    ///
    /// with the returned tower being that polynomial's own derivatives. Four
    /// properties make it the right object here, and none of them needs a
    /// tolerance:
    ///
    /// * **Exact on the modelled feasible set.** `Λ ≡ ln` for `g ≥ guard`,
    ///   bit-identical, so no fit that never reaches the floor changes at all.
    /// * **Continuous to fourth order at the knot.** The polynomial is the
    ///   degree-4 Taylor expansion of `ln` about `guard`, so all five returned
    ///   quantities agree with `logwith_derivatives_positive(guard)` exactly.
    /// * **Strictly increasing and strictly concave below it.**
    ///   `Λ' = (1/guard)(1 − x + x² − x³) ≥ 1/guard > 0` and
    ///   `Λ'' = (1/guard²)(−1 + 2x − 3x²) ≤ −1/guard² < 0` for every `x ≤ 0`
    ///   (the quadratic has no real roots), so the Newton model stays a
    ///   descent model on a concave branch.
    /// * **It keeps pushing back.** A flat clamp makes the likelihood at
    ///   `g = 0` equal to its value at the guard, i.e. it PAYS the fit to sit
    ///   outside the feasible region; the continuation falls away from the knot
    ///   and the gradient there is the barrier's own.
    pub(crate) fn log_with_derivatives_guarded(
        g: f64,
        guard: f64,
    ) -> (f64, f64, f64, f64, f64) {
        if g >= guard {
            return Self::logwith_derivatives_positive(g);
        }
        let inv = 1.0 / guard;
        let x = (g - guard) * inv;
        let x2 = x * x;
        let x3 = x2 * x;
        let inv2 = inv * inv;
        (
            guard.ln() + x - 0.5 * x2 + x3 / 3.0 - 0.25 * x3 * x,
            inv * (1.0 - x + x2 - x3),
            inv2 * (-1.0 + 2.0 * x - 3.0 * x2),
            inv2 * inv * (2.0 - 6.0 * x),
            inv2 * inv2 * (-6.0),
        )
    }

    /// The predictor state of `row` in `dynamic`: the one construction every
    /// likelihood consumer shares, so the scaled time channels and the log
    /// rate scale cannot be paired with the wrong row or left out.
    #[inline]
    pub(crate) fn row_predictor_state_at(
        &self,
        dynamic: &SurvivalDynamicGeometry,
        row: usize,
    ) -> SurvivalPredictorState {
        survival_predictor_state(
            dynamic.hs_entry[row],
            dynamic.hs_exit[row],
            dynamic.time_rate_exit[row],
            dynamic.q_entry[row],
            dynamic.q_exit[row],
            dynamic.qdot_exit[row],
            -dynamic.eta_ls_exit[row],
            self.entry_active[row],
        )
    }

    #[inline]
    pub(crate) fn validated_event_target(&self, row: usize) -> Result<f64, String> {
        let d = self.y[row];
        if !(d.is_finite() && (0.0..=1.0).contains(&d)) {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale event target must lie in [0,1] at row {row}, got {d}"
                ),
            }
            .into());
        }
        Ok(d)
    }

    pub(crate) fn exact_row_kernel(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        self.exact_row_kernel_rescaled(row, state, 0.0)
    }

    /// Like [`Self::exact_row_kernel`] but with a log-scale shift on the
    /// derivative magnitudes, propagated to the survival/pdf derivative
    /// functions.  Used by the logdet Hessian path to avoid overflow.
    pub(crate) fn exact_row_kernel_rescaled(
        &self,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        Self::exact_row_kernel_from_parts(
            &self.inverse_link,
            self.derivative_guard,
            self.w[row],
            self.y[row],
            row,
            state,
            deriv_log_scale,
        )
    }

    /// Cancellation-free `(log f(u1) − log S(u0), log S(u1) − log S(u0))`.
    ///
    /// In the far tail the two log stacks are each astronomically large while
    /// their true difference is moderate — for a left-truncated event row with
    /// entry == exit the exact value is `log hazard(u0)` — so the naked
    /// difference of the rounded stacks is pure roundoff (observed ~1e285
    /// noise at |log S| ~ 6.7e300, #2335). Pair the terms analytically:
    /// `log f(u1) − log S(u0) = ln r(u0) + [log φ(u1) − log φ(u0)]` with the
    /// pdf-log increment in closed form per link, and `δu = u1 − u0` supplied
    /// from the channel differences so identical entry/exit channels give an
    /// exact zero. When `r(u0)` underflows, `log S(u0)` is itself ≈ 0 and the
    /// naked difference is cancellation-free, so it is used directly; the
    /// same holds for every link without a closed-form increment (their
    /// survival values are bounded into (0,1], keeping both logs moderate).
    /// Assumes the monotone contract `u1 ≥ u0` (exit index at or after the
    /// entry index) the row kernel's monotonicity guard enforces.
    fn stable_exit_entry_log_pairs(
        inverse_link: &InverseLink,
        u0: f64,
        u1: f64,
        delta_u: f64,
        log_s0: f64,
        log_s1: f64,
        logphi1: f64,
        r0: f64,
        r1: f64,
    ) -> (f64, f64) {
        if matches!(inverse_link, InverseLink::Standard(StandardLink::CLogLog)) {
            // log S = −e^u, log f = u − e^u: fully closed forms in the
            // indices. (The r stacks carry the CLogLog `exp(−L)` derivative
            // rescale, so they must not enter the value channel here.)
            let growth = if delta_u == 0.0 {
                0.0
            } else {
                u0.exp() * delta_u.exp_m1()
            };
            return (u1 - growth, -growth);
        }
        let dlogphi = match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => {
                if delta_u == 0.0 {
                    // Guard 0·(u0 + u1): the sum may round to ±inf.
                    Some(0.0)
                } else {
                    // log φ(u1) − log φ(u0) = −(u1² − u0²)/2.
                    Some(-(0.5 * delta_u) * (u0 + u1))
                }
            }
            InverseLink::Standard(StandardLink::Logit) => {
                // log φ(u) = −softplus(u) − softplus(−u).
                Some(softplus_diff(u0, u1) + softplus_diff(-u0, -u1))
            }
            // Identity-link pdf is constant 1: log φ ≡ 0.
            InverseLink::Standard(StandardLink::Identity) => Some(0.0),
            _ => None,
        };
        match dlogphi {
            Some(dphi) => {
                let r0_ok = r0.is_finite() && r0 >= f64::MIN_POSITIVE;
                let r1_ok = r1.is_finite() && r1 >= f64::MIN_POSITIVE;
                let event = if r0_ok {
                    r0.ln() + dphi
                } else {
                    logphi1 - log_s0
                };
                let censor = if r0_ok && r1_ok {
                    (r0.ln() - r1.ln()) + dphi
                } else {
                    log_s1 - log_s0
                };
                (event, censor)
            }
            None => (logphi1 - log_s0, log_s1 - log_s0),
        }
    }

    /// Whether every residual-distribution stack of this link has the closed
    /// form [`Self::exact_row_kernel_fifth_from_parts`] extends through the
    /// fifth derivative. The parameterized links served by the generic pdf-jet
    /// dispatch have no certified fifth pdf derivative.
    pub(crate) fn inverse_link_has_fifth_derivative_stacks(inverse_link: &InverseLink) -> bool {
        matches!(
            inverse_link,
            InverseLink::Standard(
                StandardLink::Probit
                    | StandardLink::Logit
                    | StandardLink::CLogLog
                    | StandardLink::LogLog
                    | StandardLink::Cauchit
                    | StandardLink::Identity
            )
        )
    }

    /// The fifth derivative of each residual-distribution stack at the indices
    /// [`Self::exact_row_kernel_from_parts`] evaluates through the fourth, under
    /// the same derivative log-rescale: `−(log S)⁽⁵⁾` at `u0` and `u1`, the
    /// log-density's fifth derivative at `u1`, and the fifth derivative of the
    /// guarded `log g`. `None` for a link without closed-form fifth stacks.
    ///
    /// Per link, continuing each fourth-order tower one step:
    /// probit `ddddr = d⁵ log Φ(−u)`, `(log φ)⁽⁵⁾ = 0`; logit
    /// `ddddr = w(1 − 2μ)(1 − 12w)`, `(log φ)⁽⁵⁾ = −2w(1 − 2μ)(1 − 12w)`;
    /// cloglog `ddddr = e^{u − L}`, `(log f)⁽⁵⁾ = −e^{u − L}`; identity
    /// `ddddr = 24/(1 − u)⁵`, `(log φ)⁽⁵⁾ = 0`; loglog and cauchit
    /// `ddddr = −(log(1 − F))⁽⁵⁾` from the Bernoulli tail kernels, with
    /// `(log f)⁽⁵⁾ = e^{−u}` and `−48u(u⁴ − 10u² + 5)/(1 + u²)⁵` (#2903). The guarded `log g` continues
    /// `ln` above the guard (`24/g⁵`) and a quartic below it (`0`).
    pub(crate) fn exact_row_kernel_fifth_from_parts(
        inverse_link: &InverseLink,
        derivative_guard: f64,
        state: &SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Option<SlsFifthEntries> {
        let u0 = state.h0 + state.q0;
        let u1 = state.h1 + state.q1;
        let (ddddr0, ddddr1, d5logphi1) = match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => (
                gam_math::probability::normal_logcdf_derivatives_through_fifth(-u0)[5],
                gam_math::probability::normal_logcdf_derivatives_through_fifth(-u1)[5],
                0.0,
            ),
            InverseLink::Standard(StandardLink::Logit) => {
                let fifth = |eta: f64| {
                    let mu = gam_solve::mixture_link::component_inverse_link_jet(
                        gam_problem::LinkComponent::Logit,
                        eta,
                    )
                    .mu;
                    let w = mu * (1.0 - mu);
                    w * (1.0 - 2.0 * mu) * (1.0 - 12.0 * w)
                };
                let exit = fifth(u1);
                (fifth(u0), exit, -2.0 * exit)
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let exit = (u1 - deriv_log_scale).exp();
                ((u0 - deriv_log_scale).exp(), exit, -exit)
            }
            InverseLink::Standard(StandardLink::Identity) => {
                let fifth = |eta: f64| {
                    let inv = (1.0 - eta).recip();
                    24.0 * inv.powi(5)
                };
                (fifth(u0), fifth(u1), 0.0)
            }
            InverseLink::Standard(link @ (StandardLink::LogLog | StandardLink::Cauchit)) => {
                // `(−log S)⁽⁵⁾ = −(log(1 − F))⁽⁵⁾`: the survival tower of the
                // Bernoulli tail kernel, continued one step.
                let neglog_survival_fifth = |eta: f64| {
                    match gam_model_kernels::bernoulli_link::bernoulli_natural_log_fifth_derivatives(
                        eta,
                        inverse_link,
                    ) {
                        Ok(Some([_, log_one_minus_mu])) => Some(-log_one_minus_mu),
                        _ => None,
                    }
                };
                let d5logphi1 = if matches!(link, StandardLink::LogLog) {
                    // log f = −u − e^{−u}.
                    (-u1).exp()
                } else {
                    // log f = −ln π − ln(1 + u²), written in the tail-stable ratios
                    // u/(1 + u²) and 1/(1 + u²).
                    let (ratio, inv) = if u1.abs() <= 1.0 {
                        let inv = (1.0 + u1 * u1).recip();
                        (u1 * inv, inv)
                    } else {
                        let r = u1.recip();
                        ((u1 + r).recip(), r * r / (1.0 + r * r))
                    };
                    let ratio_sq = ratio * ratio;
                    let inv_sq = inv * inv;
                    -48.0 * ratio * (ratio_sq * ratio_sq - 10.0 * ratio_sq * inv_sq + 5.0 * inv_sq * inv_sq)
                };
                (neglog_survival_fifth(u0)?, neglog_survival_fifth(u1)?, d5logphi1)
            }
            _ => return None,
        };
        // The same infinite-rate clamp the fourth-order builder applies before
        // its guarded logarithm.
        let g = if state.g == f64::INFINITY {
            f64::MAX
        } else if state.g == f64::NEG_INFINITY {
            f64::MIN
        } else {
            state.g
        };
        let d5_log_g = if g >= derivative_guard {
            let inv = g.recip();
            24.0 * inv * inv * inv * inv * inv
        } else {
            0.0
        };
        Some(SlsFifthEntries {
            ddddr0: if state.entry_active { ddddr0 } else { 0.0 },
            ddddr1,
            d5logphi1,
            d5_log_g,
        })
    }

    fn exact_row_kernel_from_parts(
        inverse_link: &InverseLink,
        derivative_guard: f64,
        w: f64,
        d: f64,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        if w <= 0.0 {
            return Ok(None);
        }
        if !(d.is_finite() && (0.0..=1.0).contains(&d)) {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale event target must lie in [0,1] at row {row}, got {d}"
                ),
            }
            .into());
        }
        let u0 = state.h0 + state.q0;
        let u1 = state.h1 + state.q1;

        // A row entering at the origin has no `S(entry)` factor: its entry stack
        // is the exact zero every consumer skips (#2695).
        let (log_s0, r0, dr0, ddr0, dddr0) = if state.entry_active {
            Self::exact_survival_neglog_derivatives_fourth_rescaled(
                inverse_link,
                u0,
                deriv_log_scale,
            )
            .map_err(|e| {
                format!("inverse-link survival evaluation failed at row {row} entry: {e}")
            })?
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

        // Fast path: for CLogLog the survival and log-pdf evaluators both need
        // `exp(u1)`, and the PDF derivatives also need
        // `exp(u1 - deriv_log_scale)`. Share that work when both are called
        // back-to-back on the exit row.
        let ((log_s1, r1, dr1, ddr1, dddr1), (logphi1, dlogphi1, d2logphi1, d3logphi1, d4logphi1)) =
            if matches!(inverse_link, InverseLink::Standard(StandardLink::CLogLog)) {
                Self::clglog_exit_pair(u1, deriv_log_scale)
            } else {
                let surv = Self::exact_survival_neglog_derivatives_fourth_rescaled(
                    inverse_link,
                    u1,
                    deriv_log_scale,
                )
                .map_err(|e| {
                    format!("inverse-link survival evaluation failed at row {row} exit: {e}")
                })?;

                let pdf =
                    Self::exact_log_pdf_derivatives_rescaled(inverse_link, u1, deriv_log_scale)
                        .map_err(|e| {
                            format!("inverse-link log-pdf evaluation failed at row {row} exit: {e}")
                        })?;
                (surv, pdf)
            };

        // A positive-weight row must contribute its exact likelihood
        // geometry. If a hazard/pdf derivative is not representable in f64
        // (for example after survival underflow), silently excluding the row
        // would change both the fitted objective and the outer gradient.
        if !(r0.is_finite()
            && dr0.is_finite()
            && ddr0.is_finite()
            && dddr0.is_finite()
            && r1.is_finite()
            && dr1.is_finite()
            && ddr1.is_finite()
            && dddr1.is_finite()
            && dlogphi1.is_finite()
            && d2logphi1.is_finite()
            && d3logphi1.is_finite()
            && d4logphi1.is_finite())
        {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: format!(
                    "survival location-scale derivatives are non-finite at positive-weight \
                     row {row} (weight={w}, u0={u0:.6e}, u1={u1:.6e}); exact row geometry \
                     is required"
                ),
            }
            .into());
        }

        if !(derivative_guard.is_finite() && derivative_guard > 0.0) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival location-scale derivative guard must be finite and positive, got {derivative_guard}"
                ),
            }
            .into());
        }
        let guard = derivative_guard;
        let mut g = state.g;
        // Layer 4: NaN is a hard error (genuinely bad data or upstream logic
        // bug).  ±inf is clamped to finite extremes so the guarded logarithm
        // below has a finite argument; `-MAX` then trips the monotonicity
        // refusal, which is the right verdict for an infinitely decreasing
        // event Jacobian.
        if g.is_nan() {
            return Err(SurvivalLocationScaleError::NumericalFailure { reason: format!(
                "survival location-scale time derivative is non-finite at row {row}: d_eta/dt={g}"
            ) }.into());
        }
        if g == f64::INFINITY {
            g = f64::MAX;
        } else if g == f64::NEG_INFINITY {
            g = f64::MIN;
        }
        // Adaptive roundoff slack for the monotonicity guard.
        //
        // `g` is now formed with a compensated subtraction, so the low-part
        // residual from that subtraction is the primary estimate of how much
        // rounding error the d_eta/dt reconstruction may have accumulated.
        // The older state-scale heuristic remains as a floor for moderate
        // inputs.
        let legacy_slack = MONOTONICITY_GUARD_SLACK_REL
            * (1.0
                + state
                    .h0
                    .abs()
                    .max(state.h1.abs())
                    .max(state.q0.abs())
                    .max(state.q1.abs()));
        let roundoff_slack = state.g_roundoff_slack.max(legacy_slack);
        // Monotonicity refusal. `d_raw` is structurally constrained, but the
        // full event Jacobian is `g = d_raw + qdot` and the additive `qdot`
        // channel from the threshold/log-σ time transform is unconstrained, so
        // `g` is a near-cancellation whose own resolution is `roundoff_slack`.
        // A `g` more negative than the modelling guard PLUS that resolution is
        // a genuinely non-monotone state and cannot be confused with the
        // feasible boundary the optimizer converged to; anything inside it is
        // handled by the guarded logarithm below rather than by a substitution.
        //
        // The predicate is exactly the one in force before gam#2695: the three
        // floors this replaces ran first and lifted every `g` above
        // `-(guard + roundoff_slack)` to `guard`, so `g <= 0` fired here iff
        // `g < -(guard + roundoff_slack)`. No state that was accepted becomes a
        // refusal, and none that was refused becomes accepted.
        let cancellation_floor = guard + roundoff_slack;
        if g < -cancellation_floor {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale monotonicity violated at row {row}: \
                 d_eta/dt={g:.3e} <= 0 (lower_bound={guard:.3e}) \
                 (operand_scale={:.3e}, roundoff_slack={roundoff_slack:.3e})",
                    state.g_operand_scale
                ),
            }
            .into());
        }
        let (log_g, d_log_g, d2_log_g, d3_log_g, d4_log_g) =
            Self::log_with_derivatives_guarded(g, guard);

        // δu from the channel differences, not from the rounded u's: identical
        // entry/exit channels (a truncation-instant event) give an exact zero
        // where `u1 − u0` at far-tail magnitudes cannot even represent the
        // physical difference.
        let delta_u = (state.h1 - state.h0) + (state.q1 - state.q0);
        let (log_pdf1_minus_log_s0, log_s1_minus_log_s0) = if state.entry_active {
            Self::stable_exit_entry_log_pairs(
                inverse_link,
                u0,
                u1,
                delta_u,
                log_s0,
                log_s1,
                logphi1,
                r0,
                r1,
            )
        } else {
            (logphi1, log_s1)
        };

        Ok(Some(SurvivalExactRowKernel {
            w,
            d,
            log_s0,
            r0,
            dr0,
            ddr0,
            dddr0,
            log_s1,
            r1,
            dr1,
            ddr1,
            dddr1,
            logphi1,
            dlogphi1,
            d2logphi1,
            d3logphi1,
            d4logphi1,
            log_pdf1_minus_log_s0,
            log_s1_minus_log_s0,
            log_g,
            d_log_g,
            d2_log_g,
            d3_log_g,
            d4_log_g,
        }))
    }

    pub(crate) fn row_derivatives(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        self.row_derivatives_rescaled(row, state, 0.0)
    }

    pub(crate) fn row_derivatives_rescaled(
        &self,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        let Some(kernel) = self.exact_row_kernel_rescaled(row, state, deriv_log_scale)? else {
            return Ok(None);
        };
        let channels = sls_outer_plan::<5>(&kernel).lower_index_derivative_channels();
        let [nll_d1_q0, nll_d1_q1, nll_d1_qdot1] = channels.gradient;
        let [nll_d2_q0, nll_d2_q1, nll_d2_qdot1] = channels.hessian_diagonal;
        let d1_q0 = -nll_d1_q0;
        let d2_q0 = -nll_d2_q0;
        let d1_q1 = -nll_d1_q1;
        let d2_q1 = -nll_d2_q1;
        let d1_qdot1 = -nll_d1_qdot1;
        let d2_qdot1 = -nll_d2_qdot1;
        Ok(Some(SurvivalRowDerivatives {
            ll: kernel.log_likelihood_at(&state),
            d1_q0,
            d1_q1,
            d1_qdot1,
            grad_time_eta_h0: d1_q0,
            grad_time_eta_h1: d1_q1,
            grad_time_eta_d: d1_qdot1,
            h_time_h0: d2_q0,
            h_time_h1: d2_q1,
            h_time_d: d2_qdot1,
        }))
    }
}

/// Scalar chain-rule derivatives of
/// q(eta_t, eta_ls) = -eta_t * exp(-eta_ls).
///
/// Returns (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) — the full set of
/// partials up to third order needed by both the survival and GAMLSS engines.
#[inline]
pub(crate) fn q_chain_derivs_scalar(eta_t: f64, eta_ls: f64) -> (f64, f64, f64, f64, f64, f64) {
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    let q = -safe_product(eta_t, inv_sigma);
    (-inv_sigma, -q, inv_sigma, q, -inv_sigma, -q)
}

#[cfg(test)]
mod fifth_order_lowering_tests {
    use super::*;

    // `log S(u) = −e^{A u}`, `log φ(u) = −e^{B u} − C u²`, `log g`: every stack
    // is closed-form at any index, so a kernel rebuilt at a moved primary point
    // is exact and a difference of the fourth order isolates the fifth.
    const A: f64 = 0.7;
    const B: f64 = 0.4;
    const C: f64 = 0.15;
    const W: f64 = 1.27;

    fn kernel_at(point: &[f64; SLS_ROW_K], d: f64) -> (SurvivalExactRowKernel, SlsFifthEntries) {
        let [h0, h1, hdot, eta_t_exit, eta_t_entry, eta_t_deriv, eta_ls_exit, eta_ls_entry, eta_ls_deriv] =
            *point;
        let u0 = (h0 - eta_t_entry) * (-eta_ls_entry).exp();
        let inv_sigma_exit = (-eta_ls_exit).exp();
        let u1 = (h1 - eta_t_exit) * inv_sigma_exit;
        let g = hdot - eta_t_deriv - (h1 - eta_t_exit) * eta_ls_deriv;
        let entry = (A * u0).exp();
        let exit = (A * u1).exp();
        let density = (B * u1).exp();
        let inv_g = g.recip();
        let kernel = SurvivalExactRowKernel {
            w: W,
            d,
            log_s0: -entry,
            r0: A * entry,
            dr0: A.powi(2) * entry,
            ddr0: A.powi(3) * entry,
            dddr0: A.powi(4) * entry,
            log_s1: -exit,
            r1: A * exit,
            dr1: A.powi(2) * exit,
            ddr1: A.powi(3) * exit,
            dddr1: A.powi(4) * exit,
            logphi1: -density - C * u1 * u1,
            dlogphi1: -B * density - 2.0 * C * u1,
            d2logphi1: -B.powi(2) * density - 2.0 * C,
            d3logphi1: -B.powi(3) * density,
            d4logphi1: -B.powi(4) * density,
            log_pdf1_minus_log_s0: -density - C * u1 * u1 + entry,
            log_s1_minus_log_s0: -exit + entry,
            log_g: g.ln(),
            d_log_g: inv_g,
            d2_log_g: -inv_g.powi(2),
            d3_log_g: 2.0 * inv_g.powi(3),
            d4_log_g: -6.0 * inv_g.powi(4),
        };
        let fifth = SlsFifthEntries {
            ddddr0: A.powi(5) * entry,
            ddddr1: A.powi(5) * exit,
            d5logphi1: -B.powi(5) * density,
            d5_log_g: 24.0 * inv_g.powi(5),
        };
        (kernel, fifth)
    }

    /// #2677: the generated fifth order is the directional derivative of the
    /// generated fourth. Along `w`, a five-point difference of
    /// `Σ ℓ_abcd u_c v_d` with the kernel rebuilt at each moved point reproduces
    /// `Σ ℓ_abcde u_c v_d w_e` on censored, event and fractional rows, and the
    /// lowering is symmetric in its three directions.
    #[test]
    fn sls_generated_fifth_is_the_directional_derivative_of_the_fourth_2677() {
        let point: [f64; SLS_ROW_K] = [-0.37, 0.41, 1.31, 0.23, -0.18, 0.12, 0.09, -0.06, 0.07];
        let dir_u: [f64; SLS_ROW_K] = [0.7, -1.3, 0.4, 0.6, -0.5, 0.9, -0.2, 0.3, -0.8];
        let dir_v: [f64; SLS_ROW_K] = [-0.4, 0.6, 1.1, -0.2, 0.8, -0.7, 0.5, -0.9, 0.1];
        let dir_w: [f64; SLS_ROW_K] = [0.3, 0.5, -0.6, -0.9, 0.2, 0.4, 0.8, -0.5, -0.3];
        for d in [0.0, 1.0, 0.37] {
            let (kernel, fifth) = kernel_at(&point, d);
            let exact = sls_row_fifth_generated(&point, &kernel, &fifth, &dir_u, &dir_v, &dir_w);
            let fourth_at = |t: f64| {
                let moved: [f64; SLS_ROW_K] =
                    std::array::from_fn(|axis| point[axis] + t * dir_w[axis]);
                let (moved_kernel, _) = kernel_at(&moved, d);
                sls_row_fourth_generated(&moved, &moved_kernel, &dir_u, &dir_v)
            };
            let h = 1.0e-3;
            let (plus2, plus, minus, minus2) =
                (fourth_at(2.0 * h), fourth_at(h), fourth_at(-h), fourth_at(-2.0 * h));
            let mut largest = 0.0_f64;
            for a in 0..SLS_ROW_K {
                for b in 0..SLS_ROW_K {
                    let difference = (-plus2[a][b] + 8.0 * plus[a][b] - 8.0 * minus[a][b]
                        + minus2[a][b])
                        / (12.0 * h);
                    largest = largest.max(exact[a][b].abs());
                    let band = 1.0e-6 * exact[a][b].abs().max(difference.abs()).max(1.0);
                    assert!(
                        (exact[a][b] - difference).abs() <= band,
                        "d={d} fifth[{a}][{b}]: generated {:+.15e}, difference {difference:+.15e}",
                        exact[a][b],
                    );
                }
            }
            assert!(
                largest > 1.0e-2,
                "d={d}: the fifth contraction is too small ({largest:.3e}) for the agreement \
                 to say anything"
            );
            let rotated = sls_row_fifth_generated(&point, &kernel, &fifth, &dir_w, &dir_u, &dir_v);
            for a in 0..SLS_ROW_K {
                for b in 0..SLS_ROW_K {
                    let band = 1.0e-11 * exact[a][b].abs().max(1.0);
                    assert!(
                        (exact[a][b] - rotated[a][b]).abs() <= band,
                        "d={d} fifth[{a}][{b}] depends on the direction order: {:+.15e} vs {:+.15e}",
                        exact[a][b],
                        rotated[a][b],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod index_derivative_lowering_tests {
    use super::*;
    use gam_math::jet_tower::Tower3;

    const U0: f64 = -0.37;
    const U1: f64 = 0.41;
    const G: f64 = 1.31;
    const W: f64 = 1.27;

    fn log_survival(u: f64) -> f64 {
        -(0.7 * u).exp()
    }

    fn log_density(u: f64) -> f64 {
        -(0.4 * u).exp() - 0.15 * u * u
    }

    fn analytic_kernel(d: f64) -> SurvivalExactRowKernel {
        let exp0 = (0.7 * U0).exp();
        let exp1 = (0.7 * U1).exp();
        let density_exp = (0.4 * U1).exp();
        SurvivalExactRowKernel {
            w: W,
            d,
            log_s0: -exp0,
            r0: 0.7 * exp0,
            dr0: 0.49 * exp0,
            ddr0: 0.343 * exp0,
            dddr0: 0.2401 * exp0,
            log_s1: -exp1,
            r1: 0.7 * exp1,
            dr1: 0.49 * exp1,
            ddr1: 0.343 * exp1,
            dddr1: 0.2401 * exp1,
            logphi1: log_density(U1),
            dlogphi1: -0.4 * density_exp - 0.3 * U1,
            d2logphi1: -0.16 * density_exp - 0.3,
            d3logphi1: -0.064 * density_exp,
            d4logphi1: -0.0256 * density_exp,
            log_pdf1_minus_log_s0: log_density(U1) - (-exp0),
            log_s1_minus_log_s0: -exp1 - (-exp0),
            log_g: G.ln(),
            d_log_g: G.recip(),
            d2_log_g: -G.recip().powi(2),
            d3_log_g: 2.0 * G.recip().powi(3),
            d4_log_g: -6.0 * G.recip().powi(4),
        }
    }

    fn generic_index_channels(kernel: &SurvivalExactRowKernel) -> SlsIndexDerivativeChannels {
        // Constants on the six nonlinear predictor inputs reduce the canonical
        // row program's three inner atoms to independent unit seeds on h0, h1,
        // and hdot. This is the former dense generic-jet oracle, retained only
        // in the parity test; production never materializes its 9³ tensor.
        let vars: [Tower3<SLS_ROW_K>; SLS_ROW_K] = std::array::from_fn(|axis| {
            if axis < 3 {
                Tower3::variable(0.0, axis)
            } else {
                Tower3::constant(0.0)
            }
        });
        let nll = sls_row_nll(&vars, kernel).expect("canonical survival row NLL");
        SlsIndexDerivativeChannels {
            gradient: [nll.g[0], nll.g[1], nll.g[2]],
            hessian_diagonal: [nll.h[0][0], nll.h[1][1], nll.h[2][2]],
        }
    }

    fn flatten(channels: SlsIndexDerivativeChannels) -> [f64; 6] {
        [
            channels.gradient[0],
            channels.gradient[1],
            channels.gradient[2],
            channels.hessian_diagonal[0],
            channels.hessian_diagonal[1],
            channels.hessian_diagonal[2],
        ]
    }

    #[test]
    fn sls_index_sparse_lowering_matches_generic_jet_all_branches_932() {
        for d in [0.0, 1.0, 0.37] {
            let kernel = analytic_kernel(d);
            let sparse = flatten(sls_outer_plan::<5>(&kernel).lower_index_derivative_channels());
            let generic = flatten(generic_index_channels(&kernel));
            for channel in 0..sparse.len() {
                assert_eq!(
                    sparse[channel].to_bits(),
                    generic[channel].to_bits(),
                    "d={d} channel={channel}: sparse={} generic={}",
                    sparse[channel],
                    generic[channel]
                );
            }
        }
    }

    fn analytic_index_nll(point: [f64; 3], d: f64) -> f64 {
        W * (log_survival(point[0])
            - (1.0 - d) * log_survival(point[1])
            - d * (log_density(point[1]) + point[2].ln()))
    }

    fn sample_shifted(point: [f64; 3], axis: usize, shift: f64, d: f64) -> f64 {
        let mut shifted = point;
        shifted[axis] += shift;
        analytic_index_nll(shifted, d)
    }

    fn finite_difference_first(point: [f64; 3], axis: usize, d: f64) -> f64 {
        let h = 1.0e-4;
        (-sample_shifted(point, axis, 2.0 * h, d) + 8.0 * sample_shifted(point, axis, h, d)
            - 8.0 * sample_shifted(point, axis, -h, d)
            + sample_shifted(point, axis, -2.0 * h, d))
            / (12.0 * h)
    }

    fn finite_difference_second(point: [f64; 3], axis: usize, d: f64) -> f64 {
        let h = 3.0e-4;
        (-sample_shifted(point, axis, 2.0 * h, d) + 16.0 * sample_shifted(point, axis, h, d)
            - 30.0 * analytic_index_nll(point, d)
            + 16.0 * sample_shifted(point, axis, -h, d)
            - sample_shifted(point, axis, -2.0 * h, d))
            / (12.0 * h * h)
    }

    fn assert_fd_close(d: f64, order: usize, axis: usize, exact: f64, fd: f64) {
        let tolerance = 2.0e-7;
        let error = (exact - fd).abs();
        assert!(
            error <= tolerance * exact.abs().max(1.0),
            "d={d} order={order} axis={axis}: exact={exact:.16e} fd={fd:.16e} error={error:.3e}"
        );
    }

    #[test]
    fn sls_index_sparse_lowering_matches_independent_fd_all_branches_932() {
        let point = [U0, U1, G];
        for d in [0.0, 1.0, 0.37] {
            let channels =
                sls_outer_plan::<5>(&analytic_kernel(d)).lower_index_derivative_channels();
            for axis in 0..3 {
                assert_fd_close(
                    d,
                    1,
                    axis,
                    channels.gradient[axis],
                    finite_difference_first(point, axis, d),
                );
                assert_fd_close(
                    d,
                    2,
                    axis,
                    channels.hessian_diagonal[axis],
                    finite_difference_second(point, axis, d),
                );
            }
        }
    }
}

#[cfg(test)]
mod patterned_order2_perf_tests {
    /// MOVED here from the crate body by the ban rule: `#[cfg(test)]` on a
    /// bare `fn` under `src/` is a dead_code-lint escape hatch and aborts the ROOT
    /// build.rs, which fails EVERY root-crate target. Its only caller is `hand_fused`
    /// below, so it belongs inside this already-`#[cfg(test)]` module.
    /// Retired strongest-hand V/G/H schedule for the location-scale row. It remains
    /// test-only as the non-abstracted performance opponent for the generated
    /// whole-row [`sls_row_program_order2`] lowering.
    #[inline(always)]
    fn sls_row_vgh_fused(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K]) {
        let entry_exp = (-p[7]).exp();
        let exit_exp = (-p[6]).exp();

        let u0_value = kernel.w * kernel.log_s0;
        let u0_first = -kernel.w * kernel.r0;
        let u0_second = -kernel.w * kernel.dr0;

        let censored_weight = kernel.w * (1.0 - kernel.d);
        let event_weight = kernel.w * kernel.d;

        // THE SAME CONTRACT AS PRODUCTION, which is what makes this a race and
        // not a comparison of two different jobs. `sls_row_nll` — and the
        // program emitted from it — gates each term on the term's OWN
        // coefficient stack (`stack_is_exactly_zero`), never on the row weight.
        // The two predicates differ on a real row: a censored row whose exit
        // residual channels are all zero carries a nonzero weight and an
        // exactly zero stack, and applying the chain rule to that zero stack
        // against a far-tail index jet forms `0 * inf`. Gating on the weight
        // there returns NaN where production and the generic tower both return
        // a finite zero, so a weight-gated schedule is not one production could
        // ship and its saving is the guard it is missing
        // (`the_hand_carries_productions_activity_contract_932`).
        //
        // The ENTRY term is on that contract too, and was the one term still
        // composed unconditionally: an UNTRUNCATED row carries an exactly zero
        // `u0` stack with a nonzero weight, and its entry index jet overflows
        // whenever `exp(-eta_ls_entry)` does. Production skips the term; a hand
        // that composes it forms the same `0 * inf`
        // (`every_inactive_sls_term_skips_overflowing_index_channels_932`).
        let u0_active = u0_value != 0.0 || u0_first != 0.0 || u0_second != 0.0;
        let mut value = 0.0;
        if u0_active {
            value += u0_value;
        }

        let mut u1_value = 0.0;
        let mut u1_first = 0.0;
        let mut u1_second = 0.0;
        if censored_weight != 0.0 {
            u1_value -= censored_weight * kernel.log_s1;
            u1_first += censored_weight * kernel.r1;
            u1_second += censored_weight * kernel.dr1;
        }

        let mut g_value = 0.0;
        let mut g_first = 0.0;
        let mut g_second = 0.0;
        if event_weight != 0.0 {
            u1_value -= event_weight * kernel.logphi1;
            u1_first -= event_weight * kernel.dlogphi1;
            u1_second -= event_weight * kernel.d2logphi1;
            g_value = -event_weight * kernel.log_g;
            g_first = -event_weight * kernel.d_log_g;
            g_second = -event_weight * kernel.d2_log_g;
        }
        let u1_active = u1_value != 0.0 || u1_first != 0.0 || u1_second != 0.0;
        let g_active = g_value != 0.0 || g_first != 0.0 || g_second != 0.0;
        if u1_active {
            value += u1_value;
        }
        if g_active {
            value += g_value;
        }

        // The scale divides the whole residual (#2695): `u = (h − eta_t)·e^{−eta_ls}`
        // and `du1/dt = e^{−eta_ls}·g` with `g = hdot − eta_t' − (h1 − eta_t)·eta_ls'`,
        // whose `−eta_ls` enters the event log-density linearly.
        let u0_value_index = (p[0] - p[4]) * entry_exp;
        let u0_g0 = entry_exp;
        let u0_g4 = -entry_exp;
        let u0_g7 = -u0_value_index;
        let residual_exit = p[1] - p[3];
        let u1_value_index = residual_exit * exit_exp;
        let u1_g1 = exit_exp;
        let u1_g3 = -exit_exp;
        let u1_g6 = -u1_value_index;
        let g1 = -p[8];
        let g3 = p[8];
        let g8 = -residual_exit;
        if g_active {
            value += event_weight * p[6];
        }

        let mut gradient = [0.0; SLS_ROW_K];
        if u0_active {
            gradient[0] = u0_first * u0_g0;
            gradient[4] = u0_first * u0_g4;
            gradient[7] = u0_first * u0_g7;
        }
        if u1_active {
            gradient[1] += u1_first * u1_g1;
            gradient[3] += u1_first * u1_g3;
            gradient[6] += u1_first * u1_g6;
        }
        if g_active {
            gradient[1] += g_first * g1;
            gradient[2] += g_first;
            gradient[3] += g_first * g3;
            gradient[5] -= g_first;
            gradient[8] += g_first * g8;
            gradient[6] += event_weight;
        }

        let mut hessian = [[0.0; SLS_ROW_K]; SLS_ROW_K];
        macro_rules! symmetric {
            ($i:expr, $j:expr, $value:expr) => {{
                let channel = $value;
                hessian[$i][$j] += channel;
                if $i != $j {
                    hessian[$j][$i] += channel;
                }
            }};
        }

        if u0_active {
            symmetric!(0, 0, u0_second * u0_g0 * u0_g0);
            symmetric!(0, 4, u0_second * u0_g0 * u0_g4);
            symmetric!(0, 7, u0_second * u0_g0 * u0_g7 - u0_first * entry_exp);
            symmetric!(4, 4, u0_second * u0_g4 * u0_g4);
            symmetric!(4, 7, u0_second * u0_g4 * u0_g7 + u0_first * entry_exp);
            symmetric!(7, 7, u0_second * u0_g7 * u0_g7 + u0_first * u0_value_index);
        }

        if u1_active {
            symmetric!(1, 1, u1_second * u1_g1 * u1_g1);
            symmetric!(1, 3, u1_second * u1_g1 * u1_g3);
            symmetric!(1, 6, u1_second * u1_g1 * u1_g6 - u1_first * exit_exp);
            symmetric!(3, 3, u1_second * u1_g3 * u1_g3);
            symmetric!(3, 6, u1_second * u1_g3 * u1_g6 + u1_first * exit_exp);
            symmetric!(6, 6, u1_second * u1_g6 * u1_g6 + u1_first * u1_value_index);
        }

        if g_active {
            symmetric!(1, 1, g_second * g1 * g1);
            symmetric!(1, 2, g_second * g1);
            symmetric!(1, 3, g_second * g1 * g3);
            symmetric!(1, 5, -g_second * g1);
            symmetric!(1, 8, g_second * g1 * g8 - g_first);
            symmetric!(2, 2, g_second);
            symmetric!(2, 3, g_second * g3);
            symmetric!(2, 5, -g_second);
            symmetric!(2, 8, g_second * g8);
            symmetric!(3, 3, g_second * g3 * g3);
            symmetric!(3, 5, -g_second * g3);
            symmetric!(3, 8, g_second * g3 * g8 + g_first);
            symmetric!(5, 5, g_second);
            symmetric!(5, 8, -g_second * g8);
            symmetric!(8, 8, g_second * g8 * g8);
        }

        (value, gradient, hessian)
    }
    use super::*;

    /// Test-local axis-mapped order-two scatter: the exact Faà di Bruno
    /// composition `g[a_i] += f' q_i`, `H[a_i,a_j] += f' q_ij + f'' q_i q_j` of one
    /// low-dimensional atom into the global `K` channels. [`sls_row_vgh_compiled`]
    /// lowers through this copy, not through the production Hessian-pair scatter,
    /// so its comparison against `dense` stays independent of production.
    #[derive(Clone, Copy, Debug)]
    struct MappedOrder2Accumulator<const K: usize> {
        value: f64,
        gradient: [f64; K],
        hessian: [[f64; K]; K],
    }

    impl<const K: usize> MappedOrder2Accumulator<K> {
        fn zero() -> Self {
            Self {
                value: 0.0,
                gradient: [0.0; K],
                hessian: [[0.0; K]; K],
            }
        }

        /// Scatter `f(atom)` through an injective local-to-global axis map.
        #[inline(always)]
        fn add_composed<const N: usize, const H: usize, A: Order2AtomChannels<N>>(
            &mut self,
            atom: &A,
            axes: [usize; N],
            derivatives: [f64; 3],
            value_add: bool,
            gradient_add: [bool; N],
            hessian_add: [bool; H],
        ) {
            assert!(H == N * (N + 1) / 2, "invalid mapped Hessian write shape");
            assert!(N <= 128 && H <= 128, "mapped atom sparsity mask overflow");
            assert!(
                axes.iter().all(|&axis| axis < K),
                "mapped atom axis must be within the global primary dimension"
            );
            assert!(
                axes.iter()
                    .enumerate()
                    .all(|(i, axis)| !axes[..i].contains(axis)),
                "mapped atom axes must be injective"
            );

            if value_add {
                self.value += derivatives[0];
            } else {
                self.value = derivatives[0];
            }
            let mut packed = 0;
            for local_i in 0..N {
                let global_i = axes[local_i];
                if A::GRADIENT_BITS & (1u128 << local_i) != 0 {
                    let channel = derivatives[1] * atom.gradient_at(local_i);
                    if gradient_add[local_i] {
                        self.gradient[global_i] += channel;
                    } else {
                        self.gradient[global_i] = channel;
                    }
                }
                for local_j in local_i..N {
                    let global_j = axes[local_j];
                    let inner_live = A::HESSIAN_BITS & (1u128 << packed) != 0;
                    let outer_live = A::GRADIENT_BITS & (1u128 << local_i) != 0
                        && A::GRADIENT_BITS & (1u128 << local_j) != 0;
                    let channel = if inner_live {
                        let inner = derivatives[1] * atom.hessian_at(local_i, local_j);
                        if outer_live {
                            inner
                                + derivatives[2]
                                    * atom.gradient_at(local_i)
                                    * atom.gradient_at(local_j)
                        } else {
                            inner
                        }
                    } else if outer_live {
                        derivatives[2] * atom.gradient_at(local_i) * atom.gradient_at(local_j)
                    } else {
                        packed += 1;
                        continue;
                    };
                    if hessian_add[packed] {
                        self.hessian[global_i][global_j] += channel;
                        if global_i != global_j {
                            self.hessian[global_j][global_i] += channel;
                        }
                    } else {
                        self.hessian[global_i][global_j] = channel;
                        if global_i != global_j {
                            self.hessian[global_j][global_i] = channel;
                        }
                    }
                    packed += 1;
                }
            }
        }

        fn into_channels(self) -> (f64, [f64; K], [[f64; K]; K]) {
            (self.value, self.gradient, self.hessian)
        }
    }

    // Ahead-of-time sparse jet lowering of `sls_row_nll` for the V/G/H
    // channels — the mechanical oracle/racer the release cell measures
    // production against, not a production consumer (test-only since the
    // SPEC-line-1 promotion of the fused schedule). Lives in the test module
    // that consumes it so no `#[cfg(test)]` sits on a src-level item (#780).
    /// Ahead-of-time sparse jet lowering of [`sls_row_nll`] for the V/G/H
    /// channels. The scalar index expressions and outer derivative plan above are
    /// shared with every higher-order jet; only the execution representation
    /// changes. Since the SPEC-line-1 promotion of [`sls_row_vgh_fused`] this
    /// lowering is the mechanical oracle/racer the release cell measures
    /// production against, not a production consumer — hence test-gated.
    #[inline(always)]
    fn sls_row_vgh_compiled(
        primary: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K]) {
        // These static atoms are symbolically differentiated and CSE'd at build
        // time from the exact same `row_atom!` expressions used by the generic
        // high-order jets. Only final live channels exist at runtime.
        let u0 = sls_index_order2(
            primary[SLS_U0_AXES[0]],
            primary[SLS_U0_AXES[1]],
            primary[SLS_U0_AXES[2]],
        );
        let u1 = sls_index_order2(
            primary[SLS_U1_AXES[0]],
            primary[SLS_U1_AXES[1]],
            primary[SLS_U1_AXES[2]],
        );
        let g = sls_event_rate_order2(
            primary[SLS_G_AXES[0]],
            primary[SLS_G_AXES[1]],
            primary[SLS_G_AXES[2]],
            primary[SLS_G_AXES[3]],
            primary[SLS_G_AXES[4]],
        );
        let plan = sls_outer_plan::<5>(kernel);
        let truncate = |stack: [f64; 5]| [stack[0], stack[1], stack[2]];
        let mut output = MappedOrder2Accumulator::zero();
        output.add_composed(
            &u0,
            SLS_U0_AXES,
            truncate(plan.u0),
            false,
            [false; 3],
            [false; 6],
        );
        if let Some(stack) = plan.u1 {
            output.add_composed(
                &u1,
                SLS_U1_AXES,
                truncate(stack),
                true,
                [false; 3],
                [false; 6],
            );
        }
        if let Some(stack) = plan.g {
            output.add_composed(
                &g,
                SLS_G_AXES,
                truncate(stack),
                true,
                [true, false, true, false, false],
                [
                    true, false, true, false, false, false, false, false, false, true, false, false,
                    false, false, false,
                ],
            );
        }
        // The event log-density's linear `−eta_ls` (#2695).
        if plan.g_log_scale != 0.0 {
            output.value += plan.g_log_scale * primary[6];
            output.gradient[6] += plan.g_log_scale;
        }
        output.into_channels()
    }

    use gam_math::jet_scalar::Order2;
    use gam_math::nested_dual::JetField;

    /// The exact structural Hessian support of [`sls_row_nll`]. The likelihood is a
    /// sum of three univariate outer functions of:
    ///
    /// - `u0`, depending on primaries `{0,4,7}`;
    /// - `u1`, depending on `{1,3,6}`;
    /// - `g`, depending on `{1,2,3,5,8}`, plus the linear `eta_ls_exit` term.
    ///
    /// Their symmetric pair union contains 24 channels. This pattern is an
    /// execution schedule for the same generic row expression, not a derivative
    /// formula.
    #[derive(Clone, Copy, Debug)]
    struct SlsHessianPattern;

    const SLS_HESSIAN_PAIRS: [(usize, usize); 24] = [
        (0, 0),
        (0, 4),
        (0, 7),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 5),
        (1, 6),
        (1, 8),
        (2, 2),
        (2, 3),
        (2, 5),
        (2, 8),
        (3, 3),
        (3, 5),
        (3, 6),
        (3, 8),
        (4, 4),
        (4, 7),
        (5, 5),
        (5, 8),
        (6, 6),
        (7, 7),
        (8, 8),
    ];

    impl gam_math::jet_scalar::HessianPattern<SLS_ROW_K, 24> for SlsHessianPattern {
        const PAIRS: [(usize, usize); 24] = SLS_HESSIAN_PAIRS;
        const PAIR_BITS: [[u128; SLS_ROW_K]; SLS_ROW_K] =
            gam_math::jet_scalar::hessian_pair_bits(Self::PAIRS);
    }

    type SlsOrder2 = gam_math::jet_scalar::PatternedOrder2<SlsHessianPattern, SLS_ROW_K, 24>;
    use gam_math::paired_timing::{SpeedGate, batched, paired_interleaved};

    /// A censored FAR-TAIL row whose `u1` coefficient stack is EXACTLY zero
    /// while its weight is not: `w > 0`, `d = 0`, the exit residual channels all
    /// zero, and the exit log-scale far enough out that `exp(-eta_ls_exit)`
    /// overflows. The batched lowering's own documentation names this row shape
    /// — "a row's `u1` stack can be all-zero even though the row weight is
    /// nonzero" — and it is the whole reason the program gates on the stack.
    fn far_tail_zero_u1_stack_row() -> ([f64; SLS_ROW_K], SurvivalExactRowKernel) {
        let (mut p, mut kernel) = fixture();
        p[6] = -1000.0;
        kernel.d = 0.0;
        kernel.log_s1 = 0.0;
        kernel.r1 = 0.0;
        kernel.dr1 = 0.0;
        kernel.ddr1 = 0.0;
        kernel.dddr1 = 0.0;
        (p, kernel)
    }

    /// THE OPPONENT IS ON PRODUCTION'S CONTRACT, and this is the row that says
    /// so. The timed cell below asserts production beats the strongest hand
    /// schedule *of the same contract*; a hand that gated on the row weight
    /// instead of the term's coefficient stack was not on that contract, and
    /// the difference is not academic — on this row it applies the chain rule
    /// to a zero stack against an overflowed index jet and returns `NaN` in
    /// `gradient[3]` and `gradient[6]` where production and the independent
    /// generic tower both return a finite zero. Measured before the fix:
    ///
    /// ```text
    /// production g[1]=+0e0 g[3]=+0e0 g[6]=+0e0
    /// hand       g[1]=+0e0 g[3]=NaN  g[6]=NaN
    /// tower      g[1]=+0e0 g[3]=+0e0 g[6]=+0e0
    /// ```
    ///
    /// So this pin is what keeps the race honest: if the opponent ever drops
    /// the guard again to look faster, it fails here rather than winning a cell.
    #[test]
    fn the_hand_carries_productions_activity_contract_932() {
        let (p, kernel) = far_tail_zero_u1_stack_row();
        let production = sls_row_vgh_generated(&p, &kernel);
        let opponent = hand_fused(&p, &kernel);
        let tower = dense(&p, &kernel);
        let all_finite = |channels: &(f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K])| {
            channels.0.is_finite()
                && channels.1.iter().all(|channel| channel.is_finite())
                && channels.2.iter().flatten().all(|channel| channel.is_finite())
        };

        // NON-VACUITY: the row must actually reach the regime the pin is about,
        // or every assertion below is about an ordinary row.
        assert!(
            (-p[6]).exp().is_infinite(),
            "the fixture must overflow the exit scale, or the 0*inf it guards cannot form"
        );
        assert!(
            kernel.w * (1.0 - kernel.d) != 0.0,
            "the row's censoring weight must be nonzero, or the two predicates agree here"
        );

        assert!(
            all_finite(&production),
            "production must not form 0*inf on a zero stack: {production:?}"
        );
        assert!(
            all_finite(&tower),
            "the generic tower is the independent oracle and must agree: {tower:?}"
        );
        assert!(
            all_finite(&opponent),
            "the timed opponent must carry production's activity contract, or the cell \
             is racing two different jobs: {opponent:?}"
        );
        for axis in 0..SLS_ROW_K {
            assert_eq!(
                opponent.1[axis].to_bits(),
                production.1[axis].to_bits(),
                "opponent gradient[{axis}] {} vs production {}",
                opponent.1[axis],
                production.1[axis]
            );
        }
    }

    /// #932 parity gate for the compiler-emitted contracted third/fourth
    /// schedules. The canonical specialized and dense jets remain independent
    /// correctness oracles; the focused `gam-row-macros` release racer carries
    /// the honest performance gate against the family-specific handwritten
    /// analytic schedule and these specialized jets.
    #[test]
    fn sls_generated_contracted_orders_match_specialized_and_dense_jets_932() {
        use gam_math::jet_scalar::{OneSeed, TwoSeed};
        use gam_math::jet_tower::{Tower3, Tower4};

        let (p, kernel) = fixture();
        let dir_u: [f64; SLS_ROW_K] = [0.7, -1.3, 0.4, 0.6, -0.5, 0.9, -0.2, 0.3, -0.8];
        let dir_v: [f64; SLS_ROW_K] = [-0.4, 0.6, 1.1, -0.2, 0.8, -0.7, 0.5, -0.9, 0.1];

        let one_vars: [OneSeed<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| OneSeed::seed_direction(p[a], a, dir_u[a]));
        let specialized_third = sls_row_nll(&one_vars, &kernel)
            .expect("specialized third")
            .contracted_third();
        let two_vars: [TwoSeed<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| TwoSeed::seed(p[a], a, dir_u[a], dir_v[a]));
        let specialized_fourth = sls_row_nll(&two_vars, &kernel)
            .expect("specialized fourth")
            .contracted_fourth();
        let generated_third = sls_row_third_generated(&p, &kernel, &dir_u);
        let generated_fourth = sls_row_fourth_generated(&p, &kernel, &dir_u, &dir_v);
        let t3_vars: [Tower3<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| Tower3::variable(p[a], a));
        let dense3 = sls_row_nll(&t3_vars, &kernel).expect("dense Tower3");
        let t4_vars: [Tower4<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| Tower4::variable(p[a], a));
        let dense4 = sls_row_nll(&t4_vars, &kernel).expect("dense Tower4");
        let mut dense_fourth = [[0.0; SLS_ROW_K]; SLS_ROW_K];
        for a in 0..SLS_ROW_K {
            for b in 0..SLS_ROW_K {
                for c in 0..SLS_ROW_K {
                    for d in 0..SLS_ROW_K {
                        dense_fourth[a][b] += dense4.t4[a][b][c][d] * dir_u[c] * dir_v[d];
                    }
                }
            }
        }
        for a in 0..SLS_ROW_K {
            for b in 0..SLS_ROW_K {
                let mut dense_third_ab = 0.0;
                for c in 0..SLS_ROW_K {
                    dense_third_ab += dense3.t3[a][b][c] * dir_u[c];
                }
                let third_band = 1e-11
                    * specialized_third[a][b]
                        .abs()
                        .max(generated_third[a][b].abs())
                        .max(dense_third_ab.abs())
                        .max(1.0);
                assert!(
                    (specialized_third[a][b] - dense_third_ab).abs() <= third_band
                        && (generated_third[a][b] - dense_third_ab).abs() <= third_band,
                    "third[{a}][{b}]: specialized {:+.15e}, generated {:+.15e}, dense {dense_third_ab:+.15e}",
                    specialized_third[a][b],
                    generated_third[a][b],
                );
                let fourth_band = 1e-11
                    * specialized_fourth[a][b]
                        .abs()
                        .max(generated_fourth[a][b].abs())
                        .max(dense_fourth[a][b].abs())
                        .max(1.0);
                assert!(
                    (specialized_fourth[a][b] - dense_fourth[a][b]).abs() <= fourth_band
                        && (generated_fourth[a][b] - dense_fourth[a][b]).abs() <= fourth_band,
                    "fourth[{a}][{b}]: specialized {:+.15e}, generated {:+.15e}, dense {:+.15e}",
                    specialized_fourth[a][b],
                    generated_fourth[a][b],
                    dense_fourth[a][b],
                );
            }
        }

        // The speed contract for these contracted orders lives in
        // `gam-row-macros/tests/sls_codegen_perf.rs`, which races the generated
        // schedules against both the analytic hand schedule and these
        // specialized jets under the shared paired harness; this test is the
        // parity oracle only.
    }

    fn fixture() -> ([f64; SLS_ROW_K], SurvivalExactRowKernel) {
        (
            [0.4, -0.7, 0.2, 0.8, -0.35, 0.11, -0.25, 0.31, -0.17],
            SurvivalExactRowKernel {
                w: 1.3,
                d: 1.0,
                log_s0: -0.8,
                r0: 0.7,
                dr0: -0.3,
                ddr0: 0.12,
                dddr0: -0.05,
                log_s1: -1.1,
                r1: 0.9,
                dr1: -0.4,
                ddr1: 0.18,
                dddr1: -0.08,
                logphi1: -1.4,
                dlogphi1: -0.6,
                d2logphi1: -1.0,
                d3logphi1: 0.0,
                d4logphi1: 0.0,
                log_pdf1_minus_log_s0: -1.4 - (-0.8),
                log_s1_minus_log_s0: -1.1 - (-0.8),
                log_g: -0.2,
                d_log_g: 1.4,
                d2_log_g: -1.96,
                d3_log_g: 5.488,
                d4_log_g: -23.0496,
            },
        )
    }

    fn dense(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
        let vars: [Order2<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|axis| Order2::variable(p[axis], axis));
        let out = sls_row_nll(&vars, kernel).expect("dense row NLL");
        out.into_channels()
    }

    #[test]
    fn every_inactive_sls_term_skips_overflowing_index_channels_932() {
        for term in ["entry", "exit", "event"] {
            let (mut p, mut kernel) = fixture();
            assert!(kernel.w > 0.0);
            match term {
                "entry" => {
                    kernel.log_s0 = 0.0;
                    kernel.r0 = 0.0;
                    kernel.dr0 = 0.0;
                    kernel.ddr0 = 0.0;
                    kernel.dddr0 = 0.0;
                    p[4] = p[0].abs() + p[4].abs() + 1.0;
                    p[7] = -1000.0;
                    assert!((-p[7]).exp().is_infinite());
                    assert_eq!((p[0] - p[4]) * (-p[7]).exp(), f64::NEG_INFINITY);
                }
                "exit" => {
                    kernel.d = 0.0;
                    kernel.log_s1 = 0.0;
                    kernel.r1 = 0.0;
                    kernel.dr1 = 0.0;
                    kernel.ddr1 = 0.0;
                    kernel.dddr1 = 0.0;
                    p[6] = -1000.0;
                    assert!((-p[6]).exp().is_infinite());
                    assert!(sls_outer_plan::<3>(&kernel).u1.is_some());
                }
                "event" => {
                    kernel.log_g = 0.0;
                    kernel.d_log_g = 0.0;
                    kernel.d2_log_g = 0.0;
                    kernel.d3_log_g = 0.0;
                    kernel.d4_log_g = 0.0;
                    p[8] = f64::INFINITY;
                    assert!(sls_outer_plan::<3>(&kernel).g.is_some());
                }
                _ => unreachable!(),
            }
            let expected = dense(&fixture().0, &kernel);
            for (name, actual) in [
                ("generated", sls_row_vgh_generated(&p, &kernel)),
                ("hand", sls_row_vgh_fused(&p, &kernel)),
                ("tower", dense(&p, &kernel)),
            ] {
                let check = |actual: f64, expected: f64, channel: &str| {
                    let tolerance = 32.0 * f64::EPSILON * expected.abs().max(1.0);
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= tolerance,
                        "{term} {name} {channel}: {actual} != {expected}"
                    );
                };
                check(actual.0, expected.0, "value");
                for i in 0..SLS_ROW_K {
                    check(actual.1[i], expected.1[i], &format!("gradient[{i}]"));
                    for j in 0..SLS_ROW_K {
                        check(
                            actual.2[i][j],
                            expected.2[i][j],
                            &format!("hessian[{i}][{j}]"),
                        );
                    }
                }
            }
        }
    }

    fn patterned(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
        let vars: [SlsOrder2; SLS_ROW_K] =
            std::array::from_fn(|axis| SlsOrder2::variable(p[axis], axis));
        let out = sls_row_nll(&vars, kernel).expect("patterned row NLL");
        (out.value(), out.g(), out.h())
    }

    /// Same generic row program as [`patterned`], with literal identity seeds.
    /// This exposes every structural dependency mask as a compile-time constant
    /// after inlining, instead of asking LLVM to unroll `array::from_fn` before
    /// sparse-jet propagation. It is kept as a separate benchmark variant until
    /// the generated code and release timing establish whether that distinction
    /// matters on the production compiler profile.
    fn patterned_literal_seeds(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
        let vars: [SlsOrder2; SLS_ROW_K] = [
            SlsOrder2::variable(p[0], 0),
            SlsOrder2::variable(p[1], 1),
            SlsOrder2::variable(p[2], 2),
            SlsOrder2::variable(p[3], 3),
            SlsOrder2::variable(p[4], 4),
            SlsOrder2::variable(p[5], 5),
            SlsOrder2::variable(p[6], 6),
            SlsOrder2::variable(p[7], 7),
            SlsOrder2::variable(p[8], 8),
        ];
        let out = sls_row_nll(&vars, kernel).expect("literal-seeded patterned row NLL");
        (out.value(), out.g(), out.h())
    }

    fn compiled(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
        sls_row_vgh_compiled(p, kernel)
    }

    /// Direct sparse chain-rule schedule used only as the performance baseline.
    /// This deliberately duplicates the calculus in test code so the generic
    /// backend is compared with the strongest plausible hand implementation.
    fn hand(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
        struct Index {
            gradient: [f64; 9],
            hessian: [[f64; 9]; 9],
        }

        // u0 = (h0 − eta_t_entry)·e^{−eta_ls_entry} (#2695).
        let inv_entry = (-p[7]).exp();
        let u0_value = (p[0] - p[4]) * inv_entry;
        let mut u0 = Index {
            gradient: [0.0; 9],
            hessian: [[0.0; 9]; 9],
        };
        u0.gradient[0] = inv_entry;
        u0.gradient[4] = -inv_entry;
        u0.gradient[7] = -u0_value;
        for (i, j, value) in [(0, 7, -inv_entry), (4, 7, inv_entry), (7, 7, u0_value)] {
            u0.hessian[i][j] = value;
            u0.hessian[j][i] = value;
        }

        // u1 = (h1 − eta_t_exit)·e^{−eta_ls_exit}.
        let inv_exit = (-p[6]).exp();
        let residual_exit = p[1] - p[3];
        let u1_value = residual_exit * inv_exit;
        let mut u1 = Index {
            gradient: [0.0; 9],
            hessian: [[0.0; 9]; 9],
        };
        u1.gradient[1] = inv_exit;
        u1.gradient[3] = -inv_exit;
        u1.gradient[6] = -u1_value;
        for (i, j, value) in [(1, 6, -inv_exit), (3, 6, inv_exit), (6, 6, u1_value)] {
            u1.hessian[i][j] = value;
            u1.hessian[j][i] = value;
        }

        // du1/dt = e^{−eta_ls_exit}·g, g = hdot − eta_t' − (h1 − eta_t_exit)·eta_ls';
        // the scale's −eta_ls_exit enters the event log-density linearly.
        let mut g = Index {
            gradient: [0.0; 9],
            hessian: [[0.0; 9]; 9],
        };
        g.gradient[1] = -p[8];
        g.gradient[2] = 1.0;
        g.gradient[3] = p[8];
        g.gradient[5] = -1.0;
        g.gradient[8] = -residual_exit;
        for (i, j, value) in [(1, 8, -1.0), (3, 8, 1.0)] {
            g.hessian[i][j] = value;
            g.hessian[j][i] = value;
        }

        let mut value = 0.0;
        let mut gradient = [0.0; 9];
        let mut hessian = [[0.0; 9]; 9];
        let mut add = |index: &Index, stack: [f64; 3], scale: f64| {
            value += stack[0] * scale;
            let first = stack[1] * scale;
            let second = stack[2] * scale;
            for i in 0..9 {
                gradient[i] += first * index.gradient[i];
            }
            for &(i, j) in &SLS_HESSIAN_PAIRS {
                let channel =
                    second * index.gradient[i] * index.gradient[j] + first * index.hessian[i][j];
                hessian[i][j] += channel;
                if i != j {
                    hessian[j][i] += channel;
                }
            }
        };
        add(&u0, [kernel.log_s0, -kernel.r0, -kernel.dr0], kernel.w);
        let censored_weight = kernel.w * (1.0 - kernel.d);
        if censored_weight != 0.0 {
            add(
                &u1,
                [kernel.log_s1, -kernel.r1, -kernel.dr1],
                -censored_weight,
            );
        }
        let event_weight = kernel.w * kernel.d;
        if event_weight != 0.0 {
            add(
                &u1,
                [kernel.logphi1, kernel.dlogphi1, kernel.d2logphi1],
                -event_weight,
            );
            let rate = [kernel.log_g, kernel.d_log_g, kernel.d2_log_g];
            add(&g, rate, -event_weight);
            if rate.iter().any(|value| *value != 0.0) {
                value += event_weight * p[6];
                gradient[6] += event_weight;
            }
        }
        (value, gradient, hessian)
    }

    /// Retired strongest-hand fused schedule, retained only as the
    /// non-abstracted exactness and performance opponent.
    fn hand_fused(
        p: &[f64; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
        sls_row_vgh_fused(p, kernel)
    }

    /// #932 release speed gate for the location-scale row. Production is the
    /// complete build-time symbolic lowering emitted from [`sls_row_program`];
    /// the opponents are the retired strongest manually fused schedule and the
    /// dense generic tower the lowering specialises. Both consume the same
    /// frozen kernel; each is a `faster` contract on the shared [`SpeedGate`].
    #[test]
    fn release_measure_sls_compiled_vs_strongest_hand_932() {
        let (p, kernel) = fixture();
        let want = dense(&p, &kernel);
        let got = patterned(&p, &kernel);
        let literal_seed_result = patterned_literal_seeds(&p, &kernel);
        let compiled_result = compiled(&p, &kernel);
        let generated_full_result = sls_row_vgh_generated(&p, &kernel);
        let hand_result = hand(&p, &kernel);
        let hand_fused_result = hand_fused(&p, &kernel);
        let close = |a: f64, b: f64, label: &str| {
            let tolerance = 1e-12 * a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= tolerance,
                "{label}: {a:+.16e} vs {b:+.16e}"
            );
        };
        close(got.0, want.0, "value");
        close(literal_seed_result.0, want.0, "literal-seed value");
        close(compiled_result.0, want.0, "compiled value");
        close(generated_full_result.0, want.0, "generated-full value");
        close(hand_result.0, want.0, "hand value");
        close(hand_fused_result.0, want.0, "fused-hand value");
        for i in 0..SLS_ROW_K {
            close(got.1[i], want.1[i], &format!("gradient[{i}]"));
            close(
                literal_seed_result.1[i],
                want.1[i],
                &format!("literal-seed gradient[{i}]"),
            );
            close(
                compiled_result.1[i],
                want.1[i],
                &format!("compiled gradient[{i}]"),
            );
            close(
                generated_full_result.1[i],
                want.1[i],
                &format!("generated-full gradient[{i}]"),
            );
            close(hand_result.1[i], want.1[i], &format!("hand gradient[{i}]"));
            close(
                hand_fused_result.1[i],
                want.1[i],
                &format!("fused-hand gradient[{i}]"),
            );
            for j in 0..SLS_ROW_K {
                close(got.2[i][j], want.2[i][j], &format!("Hessian[{i},{j}]"));
                close(
                    literal_seed_result.2[i][j],
                    want.2[i][j],
                    &format!("literal-seed Hessian[{i},{j}]"),
                );
                close(
                    compiled_result.2[i][j],
                    want.2[i][j],
                    &format!("compiled Hessian[{i},{j}]"),
                );
                close(
                    generated_full_result.2[i][j],
                    want.2[i][j],
                    &format!("generated-full Hessian[{i},{j}]"),
                );
                close(
                    hand_result.2[i][j],
                    want.2[i][j],
                    &format!("hand Hessian[{i},{j}]"),
                );
                close(
                    hand_fused_result.2[i][j],
                    want.2[i][j],
                    &format!("fused-hand Hessian[{i},{j}]"),
                );
            }
        }

        // Exact event/censor endpoints are semantically active branches, not
        // merely convenient benchmark inputs: the inactive derivative stack
        // may be non-finite, so a fused schedule must never manufacture
        // `0 * Inf`. Pin both endpoints to the generic program separately.
        for d in [0.0, 1.0] {
            let mut endpoint_kernel = kernel;
            endpoint_kernel.d = d;
            if d == 0.0 {
                endpoint_kernel.logphi1 = f64::NAN;
                endpoint_kernel.dlogphi1 = f64::NAN;
                endpoint_kernel.d2logphi1 = f64::NAN;
                endpoint_kernel.log_g = f64::NAN;
                endpoint_kernel.d_log_g = f64::NAN;
                endpoint_kernel.d2_log_g = f64::NAN;
            } else {
                endpoint_kernel.log_s1 = f64::NAN;
                endpoint_kernel.r1 = f64::NAN;
                endpoint_kernel.dr1 = f64::NAN;
            }
            let endpoint_want = dense(&p, &endpoint_kernel);
            let endpoint_got = hand_fused(&p, &endpoint_kernel);
            let endpoint_compiled = compiled(&p, &endpoint_kernel);
            let endpoint_generated_full = sls_row_vgh_generated(&p, &endpoint_kernel);
            close(endpoint_got.0, endpoint_want.0, "fused-hand endpoint value");
            close(
                endpoint_compiled.0,
                endpoint_want.0,
                "compiled endpoint value",
            );
            close(
                endpoint_generated_full.0,
                endpoint_want.0,
                "generated-full endpoint value",
            );
            for i in 0..SLS_ROW_K {
                close(
                    endpoint_got.1[i],
                    endpoint_want.1[i],
                    &format!("fused-hand endpoint gradient d={d} [{i}]"),
                );
                close(
                    endpoint_compiled.1[i],
                    endpoint_want.1[i],
                    &format!("compiled endpoint gradient d={d} [{i}]"),
                );
                close(
                    endpoint_generated_full.1[i],
                    endpoint_want.1[i],
                    &format!("generated-full endpoint gradient d={d} [{i}]"),
                );
                for j in 0..SLS_ROW_K {
                    close(
                        endpoint_got.2[i][j],
                        endpoint_want.2[i][j],
                        &format!("fused-hand endpoint Hessian d={d} [{i},{j}]"),
                    );
                    close(
                        endpoint_compiled.2[i][j],
                        endpoint_want.2[i][j],
                        &format!("compiled endpoint Hessian d={d} [{i},{j}]"),
                    );
                    close(
                        endpoint_generated_full.2[i][j],
                        endpoint_want.2[i][j],
                        &format!("generated-full endpoint Hessian d={d} [{i},{j}]"),
                    );
                }
            }
        }

        // Speed contract, release profile only (`SpeedGate::open` documents
        // why). One arm call evaluates a batch of independent rows through
        // `paired_timing::batched`: a single SLS row is ~40 ns, and the
        // harness's per-call cost must stay far below the arm; the rows are
        // independent, as production's are, so the batch measures throughput
        // (a fold-chained batch measured the latency of one input's path and
        // inverted verdicts). Each row perturbs entry log-scale `p[7]` by its
        // own nudge and folds channels that depend on it, so no row is hoisted
        // or merged. Both arms are outlined so they cross the same ABI.
        // Production must beat the strongest fused hand schedule it replaced,
        // and the dense generic tower it specialises.
        if cfg!(debug_assertions) {
            return;
        }
        let mut gate = SpeedGate::open("SLS-ROW-VGH-932");
        const ROWS_PER_ARM: usize = 64;
        #[inline(never)]
        fn production_arm(
            p: &[f64; SLS_ROW_K],
            kernel: &SurvivalExactRowKernel,
        ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
            sls_row_vgh_generated(p, kernel)
        }
        #[inline(never)]
        fn fused_arm(
            p: &[f64; SLS_ROW_K],
            kernel: &SurvivalExactRowKernel,
        ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
            hand_fused(p, kernel)
        }
        #[inline(never)]
        fn dense_arm(
            p: &[f64; SLS_ROW_K],
            kernel: &SurvivalExactRowKernel,
        ) -> (f64, [f64; 9], [[f64; 9]; 9]) {
            dense(p, kernel)
        }
        let batch = |evaluate: fn(&[f64; SLS_ROW_K], &SurvivalExactRowKernel) -> (f64, [f64; 9], [[f64; 9]; 9])| {
            batched(ROWS_PER_ARM, move |nudge: f64| {
                let mut perturbed = p;
                perturbed[7] += nudge;
                let (value, gradient, hessian) = evaluate(&perturbed, &kernel);
                value + gradient[4] + hessian[4][4] + hessian[4][7]
            })
        };
        let hand = paired_interleaved(
            15,
            5_000,
            0x9320_5150,
            batch(production_arm),
            batch(fused_arm),
        );
        gate.faster(
            &format!("rows_per_call={ROWS_PER_ARM} opponent=strongest_hand_fused"),
            &hand,
            "production",
            "strongest_hand_fused",
        );
        let generic = paired_interleaved(
            15,
            5_000,
            0x9320_5151,
            batch(production_arm),
            batch(dense_arm),
        );
        gate.faster(
            &format!("rows_per_call={ROWS_PER_ARM} opponent=dense_tower"),
            &generic,
            "production",
            "dense_tower",
        );
        gate.finish();
    }

    /// Assert that a warm reused arena does no allocator work. After one warm row
    /// and a `reset`, the arena's reserved size must not change at any later
    /// `reset` or over any of 64 later rows. The arena grows only by adding a
    /// chunk, its only allocator call, and the compaction in
    /// `DynamicJetArena::reset` fires once, on the first reset.
    fn assert_warm_arena_does_no_allocator_work(
        order: usize,
        row: impl Fn(&DynamicJetArena, f64) -> f64,
    ) {
        let mut arena = DynamicJetArena::new();
        let mut checksum = row(&arena, 0.0);
        arena.reset();
        checksum += row(&arena, 1e-18);
        let warm = arena.allocated_bytes();
        assert!(warm > 0, "order {order}: the warm arena reserved nothing");
        for index in 0..64usize {
            arena.reset();
            assert_eq!(
                arena.allocated_bytes(),
                warm,
                "order {order}: allocator work at reset, the warm arena's reserved size changed \
                 before row {index}"
            );
            checksum += row(&arena, (index + 2) as f64 * 1e-18);
            assert_eq!(
                arena.allocated_bytes(),
                warm,
                "order {order}: allocator work in row {index} of a warm arena"
            );
        }
        assert!(checksum.is_finite(), "order {order}: non-finite checksum");
    }

    /// Positive control for `assert_warm_arena_does_no_allocator_work`: a row
    /// whose allocation doubles on every call must be refused, so a passing count
    /// means the warm arena really did no allocator work.
    #[test]
    #[should_panic(expected = "allocator work")]
    fn warm_arena_count_refuses_a_row_that_grows_the_arena_932() {
        let calls = std::cell::Cell::new(0u32);
        assert_warm_arena_does_no_allocator_work(0, |arena, nudge| {
            let len = 4096usize << calls.get();
            calls.set(calls.get() + 1);
            let slots = arena.alloc_slice_fill_with(len, |index| index as f64 + nudge);
            slots[len - 1]
        });
    }

    /// #932 release speed gate for the runtime-width SLS link/time-wiggle kernel
    /// ([`SurvivalLsWiggleRowKernel`], primary width `KW = SLS_ROW_K + pw`). The
    /// three ungated production lowerings over `sls_row_nll_wiggle` — the order-2
    /// joint Hessian (`DynamicOrder2`, consumed by `hessian_dense`), the
    /// directional third (`DynamicOneSeed`, consumed by
    /// `directional_derivative_dense`), and the second-directional fourth
    /// (`DynamicTwoSeed`, consumed by `second_directional_derivative_dense`) —
    /// all run at RUNTIME width and reuse one arena across the row fold
    /// (`acc.arena.reset()` per row).
    ///
    /// RACER CHOICE (documented per the #932 release-cell contract). The wiggle
    /// family width is `pw`-runtime, so production STRUCTURALLY cannot lower into
    /// a compile-time tower — that impossibility is the very premise of the
    /// runtime packed jet. A padded fixed-width tower is only realizable here by
    /// pinning `pw`, and racing `static_ns / dynamic_ns` would certify a
    /// monomorphization production can never take (and whose inequality direction
    /// is not a production property). The genuine engineering win the runtime
    /// lowering carries over a naive dynamic implementation is arena
    /// AMORTIZATION: `DynamicJetArena::reset` retains a single high-water chunk
    /// so equal-or-smaller later rows need zero allocator traffic, whereas a
    /// straightforward implementation allocates a fresh multi-chunk order-2 tape
    /// every row. The `fresh_arena_over_reused = fresh_ns / reused_ns`
    /// diagnostic isolates the value of arena reuse. It is deliberately not a
    /// strongest-hand comparison or closure gate.
    ///
    /// The padded fixed-width tower is NOT discarded — it is the PARITY ORACLE.
    /// `FixedRuntimeJet<Order2/OneSeed/TwoSeed<KW>, KW>` instantiates the SAME
    /// generic `sls_row_nll_wiggle` at compile-time width `KW`, and the runtime
    /// packed jet (production) must reproduce it channel-for-channel to `1e-11`
    /// relative — proving the runtime-width lowering is numerically exact before
    /// its speed is certified.
    #[test]
    fn release_measure_sls_wiggle_dynamic_jets_vs_padded_static_tower_932() {
        use gam_math::jet_scalar::{FixedRuntimeJet, OneSeed, TwoSeed};

        // Small runtime wiggle width so the padded static parity tower is a
        // clean fixed `KW`; production runs this same expression at runtime `pw`.
        const PW: usize = 3;
        const KW: usize = SLS_ROW_K + PW;

        let (p9, kernel) = fixture();
        let betaw = [0.13_f64, -0.21, 0.07];
        let mut p: [f64; KW] = [0.0; KW];
        p[..SLS_ROW_K].copy_from_slice(&p9);
        p[SLS_ROW_K..].copy_from_slice(&betaw);

        // Per-row warp basis stacks `[B, B', B'', B''', B'''']` at the base entry
        // (`b_u0`) and `[B, …, B''''']` at the exit (`b_u1`) index, one entry per
        // wiggle column. Synthetic but finite; the parity oracle certifies the
        // lowering, not these numbers.
        let b_u0_0 = [0.20_f64, -0.15, 0.09];
        let b_u0_1 = [-0.11_f64, 0.22, -0.07];
        let b_u0_2 = [0.05_f64, -0.03, 0.14];
        let b_u0_3 = [-0.08_f64, 0.06, -0.02];
        let b_u0_4 = [0.03_f64, -0.05, 0.11];
        let b_u1_0 = [0.17_f64, 0.12, -0.19];
        let b_u1_1 = [-0.09_f64, 0.04, 0.13];
        let b_u1_2 = [0.06_f64, -0.11, 0.03];
        let b_u1_3 = [-0.04_f64, 0.08, -0.05];
        // The fourth- and fifth-derivative slots carry real, DISTINCT values
        // rather than a repeat or a zero: this oracle certifies that the dynamic
        // and the padded static lowering agree on the WHOLE tower, so a slot
        // holding the same number as its neighbour could hide a lowering reading
        // the wrong one (gam#2695, which put the true fourth derivative where a
        // literal was, and then the true fifth where `m₁`'s top slot was `0.0`).
        let b_u1_4 = [0.07_f64, -0.02, 0.10];
        let b_u1_5 = [-0.06_f64, 0.09, -0.03];
        let basis = SlsWiggleRowBasis {
            b_u0: [&b_u0_0, &b_u0_1, &b_u0_2, &b_u0_3, &b_u0_4],
            b_u1: [&b_u1_0, &b_u1_1, &b_u1_2, &b_u1_3, &b_u1_4, &b_u1_5],
        };

        let dir_u: [f64; KW] = [
            0.7, -1.3, 0.4, 0.6, -0.5, 0.9, -0.2, 0.3, -0.8, 0.5, -0.6, 0.2,
        ];
        let dir_v: [f64; KW] = [
            -0.4, 0.6, 1.1, -0.2, 0.8, -0.7, 0.5, -0.9, 0.1, -0.3, 0.4, -0.5,
        ];

        let band = |a: f64, b: f64| 1e-11 * a.abs().max(b.abs()).max(1.0);

        // --- Parity oracle, order 2: runtime packed Hessian == padded static. ---
        let arena2 = DynamicJetArena::new();
        let dyn2_vars =
            arena2.alloc_slice_fill_with(KW, |a| DynamicOrder2::variable(p[a], a, KW, &arena2));
        let dyn2 = sls_row_nll_wiggle(dyn2_vars, &kernel, PW, &basis);
        let fix2_vars: Vec<FixedRuntimeJet<Order2<KW>, KW>> = (0..KW)
            .map(|a| FixedRuntimeJet::from_inner(Order2::variable(p[a], a)))
            .collect();
        let fix2 = sls_row_nll_wiggle(&fix2_vars, &kernel, PW, &basis).into_inner();
        assert!(
            (dyn2.value() - fix2.value()).abs() <= band(dyn2.value(), fix2.value()),
            "wiggle order-2 value: dynamic {:+.15e} vs padded-static {:+.15e}",
            dyn2.value(),
            fix2.value(),
        );
        for a in 0..KW {
            for b in 0..KW {
                let d = dyn2.h()[a * KW + b];
                let s = fix2.h()[a][b];
                assert!(
                    (d - s).abs() <= band(d, s),
                    "wiggle order-2 H[{a}][{b}]: dynamic {d:+.15e} vs padded-static {s:+.15e}"
                );
            }
        }

        // --- Parity oracle, order 3: directional third contraction. ---
        let arena3 = DynamicJetArena::new();
        let dyn3_vars = arena3.alloc_slice_fill_with(KW, |a| {
            DynamicOneSeed::seed_direction(p[a], a, dir_u[a], KW, &arena3)
        });
        let dyn3 = sls_row_nll_wiggle(dyn3_vars, &kernel, PW, &basis);
        let dyn3_third = dyn3.contracted_third();
        let fix3_vars: Vec<FixedRuntimeJet<OneSeed<KW>, KW>> = (0..KW)
            .map(|a| FixedRuntimeJet::from_inner(OneSeed::seed_direction(p[a], a, dir_u[a])))
            .collect();
        let fix3_third = sls_row_nll_wiggle(&fix3_vars, &kernel, PW, &basis)
            .into_inner()
            .contracted_third();
        for a in 0..KW {
            for b in 0..KW {
                let d = dyn3_third[a * KW + b];
                let s = fix3_third[a][b];
                assert!(
                    (d - s).abs() <= band(d, s),
                    "wiggle order-3 T[{a}][{b}]: dynamic {d:+.15e} vs padded-static {s:+.15e}"
                );
            }
        }

        // --- Parity oracle, order 4: second-directional fourth contraction. ---
        let arena4 = DynamicJetArena::new();
        let dyn4_vars = arena4.alloc_slice_fill_with(KW, |a| {
            DynamicTwoSeed::seed(p[a], a, dir_u[a], dir_v[a], KW, &arena4)
        });
        let dyn4 = sls_row_nll_wiggle(dyn4_vars, &kernel, PW, &basis);
        let dyn4_fourth = dyn4.contracted_fourth();
        let fix4_vars: Vec<FixedRuntimeJet<TwoSeed<KW>, KW>> = (0..KW)
            .map(|a| FixedRuntimeJet::from_inner(TwoSeed::seed(p[a], a, dir_u[a], dir_v[a])))
            .collect();
        let fix4_fourth = sls_row_nll_wiggle(&fix4_vars, &kernel, PW, &basis)
            .into_inner()
            .contracted_fourth();
        for a in 0..KW {
            for b in 0..KW {
                let d = dyn4_fourth[a * KW + b];
                let s = fix4_fourth[a][b];
                assert!(
                    (d - s).abs() <= band(d, s),
                    "wiggle order-4 F[{a}][{b}]: dynamic {d:+.15e} vs padded-static {s:+.15e}"
                );
            }
        }

        // Arena reuse at orders 3 and 4 is asserted as an allocation count, in
        // every profile, not timed. Timed, each order was a coin flip decided by
        // per-process placement (the reused arena's one chunk lands at one
        // address per process, the fresh arm's chunks wherever malloc places them
        // each row), a larger effect than the saving itself: across fourteen runs
        // the order-4 reused arm ranged 0.978–1.047 against a fresh arena per row
        // with `wins` unanimous in both directions, and order 3 read 0.988, 1.005
        // and 1.021 on three Speed Gates hosts.
        assert_warm_arena_does_no_allocator_work(3, |arena, nudge| {
            let mut pp = p;
            pp[0] += nudge;
            let vars = arena.alloc_slice_fill_with(KW, |a| {
                DynamicOneSeed::seed_direction(pp[a], a, dir_u[a], KW, arena)
            });
            let out = sls_row_nll_wiggle(vars, &kernel, PW, &basis);
            out.value() + out.contracted_third()[0]
        });
        assert_warm_arena_does_no_allocator_work(4, |arena, nudge| {
            let mut pp = p;
            pp[0] += nudge;
            let vars = arena.alloc_slice_fill_with(KW, |a| {
                DynamicTwoSeed::seed(pp[a], a, dir_u[a], dir_v[a], KW, arena)
            });
            let out = sls_row_nll_wiggle(vars, &kernel, PW, &basis);
            out.value() + out.contracted_fourth()[0]
        });

        // Speed contract, release profile only (`SpeedGate::open` documents
        // why): at order 2 the amortised reused arena (production) must beat a
        // fresh arena per row, where the instrument resolves the saving. The
        // nudge perturbs the first primary.
        if cfg!(debug_assertions) {
            return;
        }
        let mut gate = SpeedGate::open("SLS-WIGGLE-DYN-932");
        let iterations = 5_000usize;

        let mut prod_arena2 = DynamicJetArena::new();
        let order2 = paired_interleaved(
            15,
            iterations,
            0x9320_D1_02,
            |nudge| {
                let mut pp = p;
                pp[0] += nudge;
                prod_arena2.reset();
                let vars = prod_arena2
                    .alloc_slice_fill_with(KW, |a| DynamicOrder2::variable(pp[a], a, KW, &prod_arena2));
                let out = sls_row_nll_wiggle(vars, &kernel, PW, &basis);
                out.value() + out.g()[0] + out.h()[0]
            },
            |nudge| {
                let mut pp = p;
                pp[0] += nudge;
                let fresh = DynamicJetArena::new();
                let vars =
                    fresh.alloc_slice_fill_with(KW, |a| DynamicOrder2::variable(pp[a], a, KW, &fresh));
                let out = sls_row_nll_wiggle(vars, &kernel, PW, &basis);
                out.value() + out.g()[0] + out.h()[0]
            },
        );
        gate.faster("order=2", &order2, "reused_arena", "fresh_arena");
        gate.finish();
    }
}
#[cfg(test)]
mod all_axes_third_tests {
    use super::*;
    use gam_math::jet_scalar::OneSeed;

    #[test]
    fn missing_fitted_state_is_a_typed_geometry_error() {
        let error = require_fitted_block_geometry(&[], "offset geometry")
            .expect_err("missing fitted state must not become zero geometry");
        match error {
            SurvivalLocationScaleError::InternalInvariant { reason } => {
                assert!(reason.contains("fitted block state is missing"));
            }
            other => panic!("missing fitted state must be an internal invariant, got {other:?}"),
        }
    }

    /// Tiny deterministic LCG (no external rng dep in the test).
    struct Lcg(u64);
    impl Lcg {
        fn step(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Finite value in roughly `[-2, 2]`, occasionally exact `0.0` (to provoke
        /// signed-zero channels under the negative event/censored weights).
        fn val(&mut self) -> f64 {
            let u = self.step();
            if u & 0x1F == 0 {
                return 0.0;
            }
            ((u >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 4.0
        }
        fn nonfinite(&mut self) -> f64 {
            match self.step() % 3 {
                0 => f64::INFINITY,
                1 => f64::NEG_INFINITY,
                _ => f64::NAN,
            }
        }
        fn range(&mut self, n: usize) -> usize {
            (self.step() % n as u64) as usize
        }
    }

    /// A residual-distribution stack entry: the true value when the branch is
    /// active, else a non-finite poison value the gated path must never touch.
    fn stack_entry(active: bool, rng: &mut Lcg) -> f64 {
        if active {
            rng.val()
        } else if rng.step() & 1 == 0 {
            rng.nonfinite()
        } else {
            rng.val()
        }
    }

    fn make_kernel(rng: &mut Lcg, sig: usize) -> SurvivalExactRowKernel {
        let (w, d) = match sig {
            0 => (rng.val().abs() + 0.2, 0.0), // pure censored
            1 => (rng.val().abs() + 0.2, 1.0), // pure event
            2 => (rng.val().abs() + 0.2, 0.25 + (rng.range(50) as f64) / 100.0), // fractional
            _ => (0.0, if rng.step() & 1 == 0 { 0.0 } else { 1.0 }), // null (w = 0)
        };
        let cens = w * (1.0 - d) != 0.0;
        let ev = w * d != 0.0;
        // Draw the stacks in the same sequence as the original struct literal
        // so the LCG stream (and thus every downstream expectation) is
        // unchanged; the pair fields are derived, not drawn.
        let log_s0 = rng.val();
        let r0 = rng.val();
        let dr0 = rng.val();
        let ddr0 = rng.val();
        let dddr0 = rng.val();
        let log_s1 = stack_entry(cens, rng);
        let r1 = stack_entry(cens, rng);
        let dr1 = stack_entry(cens, rng);
        let ddr1 = stack_entry(cens, rng);
        let dddr1 = stack_entry(cens, rng);
        let logphi1 = stack_entry(ev, rng);
        let dlogphi1 = stack_entry(ev, rng);
        let d2logphi1 = stack_entry(ev, rng);
        let d3logphi1 = stack_entry(ev, rng);
        let d4logphi1 = stack_entry(ev, rng);
        SurvivalExactRowKernel {
            w,
            d,
            log_s0,
            r0,
            dr0,
            ddr0,
            dddr0,
            log_s1,
            r1,
            dr1,
            ddr1,
            dddr1,
            logphi1,
            dlogphi1,
            d2logphi1,
            d3logphi1,
            d4logphi1,
            log_pdf1_minus_log_s0: logphi1 - log_s0,
            log_s1_minus_log_s0: log_s1 - log_s0,
            log_g: stack_entry(ev, rng),
            d_log_g: stack_entry(ev, rng),
            d2_log_g: stack_entry(ev, rng),
            d3_log_g: stack_entry(ev, rng),
            d4_log_g: stack_entry(ev, rng),
        }
    }

    /// Entrywise agreement of two contracted thirds: `NaN` exactly where the
    /// scalar is `NaN`, otherwise within the `1e-11` relative band the dense-tower
    /// oracle above uses (the two paths accumulate the same terms in different
    /// orders, so cancelling entries differ by more than a few ulps).
    fn assert_thirds_agree(
        generated: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        scalar: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        context: &str,
    ) {
        for x in 0..SLS_ROW_K {
            for y in 0..SLS_ROW_K {
                let g = generated[x][y];
                let s = scalar[x][y];
                if s.is_nan() {
                    assert!(g.is_nan(), "scalar NaN but generated {g} at {context} x={x} y={y}");
                } else {
                    assert!(
                        (g - s).abs() <= 1e-11 * g.abs().max(s.abs()).max(1.0),
                        "generated third {g} != scalar {s} at {context} x={x} y={y}"
                    );
                }
            }
        }
    }

    /// The all-axes override's per-row contraction, `sls_row_third_generated`, agrees
    /// with the scalar `sls_row_nll` at the one-seed jet across mixed gating regimes,
    /// signed-zero primary and design channels, null (`w = 0`) rows, and non-finite
    /// poisoned inactive residual-distribution stacks, which neither path may touch.
    #[test]
    fn generated_axis_third_matches_scalar_row_nll() {
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let block_of = [0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut compared = 0usize;
        for _ in 0..2500 {
            let widths = [1 + rng.range(4), 1 + rng.range(4), 1 + rng.range(4)];
            let offs = [0usize, widths[0], widths[0] + widths[1]];
            let p = widths[0] + widths[1] + widths[2];
            let m = 5 + rng.range(20);
            let mut rows: Vec<([f64; SLS_ROW_K], SurvivalExactRowKernel)> = Vec::with_capacity(m);
            let mut chans: Vec<Vec<Option<(usize, Array1<f64>)>>> = Vec::with_capacity(m);
            for _ in 0..m {
                let sig = rng.range(4);
                let kernel = make_kernel(&mut rng, sig);
                let primary: [f64; SLS_ROW_K] =
                    std::array::from_fn(|_| if rng.step() & 7 == 0 { 0.0 } else { rng.val() });
                rows.push((primary, kernel));
                let row_chans: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
                    .map(|c| {
                        let blk = block_of[c];
                        if (c == 5 || c == 8) && rng.step() & 1 == 0 {
                            None
                        } else {
                            let row = Array1::from_iter(
                                (0..widths[blk])
                                    .map(|_| if rng.step() & 3 == 0 { 0.0 } else { rng.val() }),
                            );
                            Some((offs[blk], row))
                        }
                    })
                    .collect();
                chans.push(row_chans);
            }
            let a = rng.range(p);
            for (row, (primary, kernel)) in rows.iter().enumerate() {
                let direction = axis_direction_from_channel_cache(&chans[row], a);
                let generated = sls_row_third_generated(primary, kernel, &direction);
                let vars: [OneSeed<SLS_ROW_K>; SLS_ROW_K] =
                    std::array::from_fn(|c| OneSeed::seed_direction(primary[c], c, direction[c]));
                let scalar = sls_row_nll(&vars, kernel)
                    .expect("scalar row NLL")
                    .contracted_third();
                assert_thirds_agree(&generated, &scalar, &format!("row={row} axis={a}"));
                compared += SLS_ROW_K * SLS_ROW_K;
            }
        }
        assert!(
            compared >= 100_000,
            "expected >=100k channel comparisons, got {compared}"
        );
    }

    /// Root-cause guard for the zero-stack skip. A weighted row with no left
    /// truncation carries an entry `u0` stack of exactly `[0,0,0,0,0]` (because
    /// `S(entry) = 1`). The row program skips that stack rather than composing it,
    /// which would form `0·(negative channel) = -0.0`, or `0·∞ = NaN` on a far-tail
    /// row. Rows alternate truncated and untruncated; the third row is far-tail.
    #[test]
    fn zero_entry_stack_skip_keeps_generated_third_clean() {
        // A pure-censored kernel: event terms inactive (poisoned non-finite so a
        // regression that composes them would surface), entry stack zeroed for
        // the no-left-truncation rows.
        fn censored_kernel(entry_zero: bool) -> SurvivalExactRowKernel {
            let (log_s0, r0, dr0, ddr0, dddr0) = if entry_zero {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            } else {
                (-0.73, 0.41, 0.29, -0.17, 0.11)
            };
            SurvivalExactRowKernel {
                w: 1.3,
                d: 0.0,
                log_s0,
                r0,
                dr0,
                ddr0,
                dddr0,
                log_s1: -1.07,
                r1: 0.88,
                dr1: 0.52,
                ddr1: -0.31,
                dddr1: 0.19,
                logphi1: f64::NAN,
                dlogphi1: f64::NAN,
                d2logphi1: f64::NAN,
                d3logphi1: f64::NAN,
                d4logphi1: f64::NAN,
                log_pdf1_minus_log_s0: f64::NAN,
                log_s1_minus_log_s0: -1.07 - log_s0,
                log_g: f64::NAN,
                d_log_g: f64::NAN,
                d2_log_g: f64::NAN,
                d3_log_g: f64::NAN,
                d4_log_g: f64::NAN,
            }
        }

        let entry_zero = [false, true, true, false];
        // Row 2 is far-tail: primary[7] = -720 makes `exp(-p7) = +inf`, so `u0`'s
        // jet channels blow up; composing its zero stack would form NaN.
        let primaries: [[f64; SLS_ROW_K]; 4] = [
            [0.3, -0.4, 0.5, 0.6, -0.2, 0.1, 0.25, -0.35, 0.15],
            [-0.2, 0.35, -0.15, 0.45, 0.3, -0.05, 0.2, 0.4, -0.1],
            [0.1, -0.25, 0.2, 0.55, 0.6, 0.05, -0.3, -720.0, 0.2],
            [0.4, 0.2, -0.3, -0.5, 0.15, -0.2, 0.35, 0.1, -0.4],
        ];
        let dirs: [[f64; SLS_ROW_K]; 4] = [
            [1.0, 0.5, -0.5, 0.3, 0.7, -0.2, 0.4, 0.6, -0.3],
            [-0.6, 0.4, 0.2, -0.3, 0.5, 0.1, -0.4, 0.35, 0.25],
            [0.3, -0.7, 0.6, 0.2, -0.4, 0.15, 0.5, 0.45, -0.2],
            [0.2, 0.3, -0.4, 0.6, 0.1, -0.5, 0.25, -0.35, 0.4],
        ];
        for row in 0..4 {
            let kernel = censored_kernel(entry_zero[row]);
            let generated = sls_row_third_generated(&primaries[row], &kernel, &dirs[row]);
            let vars: [OneSeed<SLS_ROW_K>; SLS_ROW_K] = std::array::from_fn(|c| {
                OneSeed::seed_direction(primaries[row][c], c, dirs[row][c])
            });
            let scalar = sls_row_nll(&vars, &kernel)
                .expect("scalar row NLL")
                .contracted_third();
            assert_thirds_agree(&generated, &scalar, &format!("row={row}"));
            if entry_zero[row] {
                // The `u0`-only cross channel (entry η_t × entry η_ls) is a clean +0.0,
                // and the far-tail row leaks no composed `0·∞ = NaN`.
                assert_eq!(
                    generated[4][7].to_bits(),
                    0.0f64.to_bits(),
                    "untruncated row {row} entry cross channel must be +0.0, got {}",
                    generated[4][7]
                );
                assert!(
                    generated.iter().flatten().all(|v| v.is_finite()),
                    "untruncated row {row} leaked a non-finite entry"
                );
            }
        }
    }
}

/// gam#2695 — the event Jacobian's floor and its derivative tower must be ONE
/// function of `g = dη/dt`.
///
/// `exact_row_kernel_from_parts` floors `g` to `derivative_guard` on three
/// branches and then reads `(log_g, d_log_g, d2_log_g, …)` from
/// [`SurvivalLocationScaleFamily::logwith_derivatives_positive`] at the FLOORED
/// `g`. Inside the floored band the row's log-likelihood is therefore CONSTANT
/// in `qdot1` while the tower reports a slope of `1/guard` — a value on one
/// surface and a derivative on another, of exactly the class #2714 established.
///
/// The consequence is #2695's headline: the joint-Newton RHS carries that slope
/// into `predicted_reduction`, the accept test measures the flat value, and
/// `actual/(rhs·δ)` cannot approach 1 under refinement no matter how small the
/// step is (0 of 75 linear-dominated attempts within 50% of 1). The refutation
/// this thread needed is a finite difference of the row's own value against the
/// row's own analytic derivative at a state INSIDE the band — which no
/// synthetic fixture built away from the boundary can reach.
#[cfg(test)]
mod event_jacobian_floor_consistency_2695_tests {
    use super::*;

    /// The production floor on this fixture's family (`gam-cli survival
    /// --survival-likelihood location-scale` prints
    /// "derivative floor 1.000e-6").
    const GUARD: f64 = 1.0e-6;

    /// One event row (`d = 1`), so the `log g` Jacobian term is live, with
    /// `d_raw = 0` so `g` IS the `qdot1` coordinate and the finite difference
    /// moves exactly the channel whose derivative is under test.
    fn kernel_at(g: f64) -> SurvivalExactRowKernel {
        let state = survival_predictor_state(-0.4, 0.3, 0.0, -0.2, 0.15, g, 0.0, true);
        SurvivalLocationScaleFamily::exact_row_kernel_from_parts(
            &InverseLink::Standard(StandardLink::Probit),
            GUARD,
            1.0,
            1.0,
            0,
            state,
            0.0,
        )
        .expect("probit row kernel builds at a finite interior state")
        .expect("a positive-weight row yields a kernel")
    }

    /// `∂ℓ/∂qdot1` as the solver reads it: the third gradient channel of the
    /// lowered row program, sign-flipped out of the NLL convention exactly as
    /// [`SurvivalLocationScaleFamily::row_derivatives_rescaled`] does.
    fn analytic_dll_dg(g: f64) -> f64 {
        let kernel = kernel_at(g);
        -sls_outer_plan::<5>(&kernel)
            .lower_index_derivative_channels()
            .gradient[2]
    }

    fn central_difference_dll_dg(g: f64, h: f64) -> f64 {
        (kernel_at(g + h).log_likelihood() - kernel_at(g - h).log_likelihood()) / (2.0 * h)
    }

    /// Positive control: well above the floor the two agree, so the harness
    /// measures what it claims to.
    #[test]
    fn above_the_floor_the_row_gradient_differentiates_the_row_value() {
        let g = 0.75;
        let h = 1.0e-5;
        let analytic = analytic_dll_dg(g);
        let fd = central_difference_dll_dg(g, h);
        assert!(
            (fd - analytic).abs() <= 1.0e-6 * (1.0 + analytic.abs()),
            "unfloored row: fd={fd:.9e} vs analytic={analytic:.9e}"
        );
    }

    /// The defect. `g = 0.4·guard` is strictly inside the second floor branch
    /// (`0 < g < guard`), five orders above the compensated-difference
    /// roundoff slack, so the band is entered on its own terms rather than on
    /// a rounding accident.
    #[test]
    fn inside_the_floored_band_the_row_gradient_differentiates_the_row_value() {
        let g = 0.4 * GUARD;
        // The step is `eps^(1/3)` of the BAND's own scale (`guard`), not of a
        // unit scale. The channel's third derivative there is `O(1/guard^3)`,
        // so a unit-scaled step carries `h^2/6 * f'''` of pure truncation —
        // measured at 0.1% of the derivative under test, which reads as a
        // defect and is not one.
        let cbrt_eps = f64::EPSILON.cbrt();
        let h = cbrt_eps * GUARD;
        let analytic = analytic_dll_dg(g);
        let fd = central_difference_dll_dg(g, h);
        // The matching central-difference floor: truncation plus roundoff at
        // that step, scaled by the analytic magnitude.
        let tol = 64.0 * cbrt_eps * cbrt_eps * (1.0 + analytic.abs());
        assert!(
            (fd - analytic).abs() <= tol,
            "floored row: the value moves by {fd:.9e} per unit qdot1 while the \
             derivative tower asserts {analytic:.9e} — a first-order \
             disagreement of {:.3e}x that no step size can refine away",
            if fd == 0.0 {
                f64::INFINITY
            } else {
                analytic / fd
            }
        );
    }

    /// The band is not a measure-zero curiosity, which is why the
    /// disagreement above is an O(1) error rather than a rounding one: it is
    /// `guard` wide in `qdot1`, and today the row value is BITWISE identical
    /// across the whole of it. A likelihood that cannot tell `g = 0.05·guard`
    /// from `g = 0.95·guard` is not the function its derivative describes.
    #[test]
    fn the_row_value_responds_to_the_event_jacobian_across_the_guard_band() {
        let low = kernel_at(0.05 * GUARD).log_likelihood();
        let high = kernel_at(0.95 * GUARD).log_likelihood();
        assert!(
            high > low,
            "a larger event Jacobian must raise the row log-likelihood; \
             got {low} at 0.05·guard and {high} at 0.95·guard \
             (bitwise equal = {})",
            low.to_bits() == high.to_bits()
        );
    }
}

/// gam#2695 — the guarded logarithm's own contract.
///
/// [`SurvivalLocationScaleFamily::log_with_derivatives_guarded`] is the single
/// source for the event-Jacobian log channel: the row value
/// (`kernel.log_likelihood`), the block gradients
/// (`evaluate_log_likelihood_and_block_gradients`) and the joint Hessian
/// (`sls_row_nll_wiggle`, which composes the same five-entry stack) all read
/// this one tower, so a defect here cannot be repaired at any one of them.
/// These tests pin the four properties the derivation rests on.
#[cfg(test)]
mod guarded_log_channel_2695_tests {
    use super::*;

    const GUARD: f64 = 1.0e-6;

    fn guarded(g: f64) -> (f64, f64, f64, f64, f64) {
        SurvivalLocationScaleFamily::log_with_derivatives_guarded(g, GUARD)
    }

    /// Exact on the modelled feasible set: no fit that never reaches the floor
    /// changes by a single bit.
    #[test]
    fn above_the_guard_the_channel_is_the_plain_logarithm() {
        for g in [GUARD, 1.0e-5, 1.0e-3, 0.5, 1.0, 4.7, 1.0e6] {
            let plain = SurvivalLocationScaleFamily::logwith_derivatives_positive(g);
            let guarded = guarded(g);
            assert_eq!(plain.0.to_bits(), guarded.0.to_bits(), "value at g={g}");
            assert_eq!(plain.1.to_bits(), guarded.1.to_bits(), "d1 at g={g}");
            assert_eq!(plain.2.to_bits(), guarded.2.to_bits(), "d2 at g={g}");
            assert_eq!(plain.3.to_bits(), guarded.3.to_bits(), "d3 at g={g}");
            assert_eq!(plain.4.to_bits(), guarded.4.to_bits(), "d4 at g={g}");
        }
    }

    /// Every entry of the tower is the derivative of the entry above it, on
    /// the continued branch. This is the property whose absence IS #2695.
    #[test]
    fn the_tower_differentiates_itself_below_the_guard() {
        let h = 1.0e-4 * GUARD;
        for g in [0.9 * GUARD, 0.5 * GUARD, 0.1 * GUARD, 0.0, -0.5 * GUARD] {
            let plus = guarded(g + h);
            let minus = guarded(g - h);
            let here = guarded(g);
            let entries = [
                ((plus.0 - minus.0) / (2.0 * h), here.1, "d/dg of value"),
                ((plus.1 - minus.1) / (2.0 * h), here.2, "d/dg of d1"),
                ((plus.2 - minus.2) / (2.0 * h), here.3, "d/dg of d2"),
                ((plus.3 - minus.3) / (2.0 * h), here.4, "d/dg of d3"),
            ];
            for (fd, analytic, what) in entries {
                assert!(
                    (fd - analytic).abs() <= 1.0e-6 * (1.0 + analytic.abs()),
                    "g={g:.3e}: {what} is {analytic:.9e} but a central difference gives \
                     {fd:.9e}"
                );
            }
        }
    }

    /// Strictly increasing and strictly concave below the knot, so the Newton
    /// model on that branch is still a concave-descent model.
    #[test]
    fn the_continued_branch_is_increasing_and_concave() {
        let mut previous = f64::NEG_INFINITY;
        // Walk `g` UPWARD, from well below zero to the knot, so "increasing"
        // is asserted in the direction it is claimed in.
        for step in (0..=40).rev() {
            let g = GUARD * (1.0 - 0.05 * f64::from(step));
            let (value, d1, d2, _, _) = guarded(g);
            assert!(
                value > previous,
                "the channel must increase with g; g={g:.3e} gave {value} after {previous}"
            );
            previous = value;
            assert!(d1 > 0.0, "d1 must be positive at g={g:.3e}, got {d1:.6e}");
            assert!(d2 < 0.0, "d2 must be negative at g={g:.3e}, got {d2:.6e}");
        }
    }

    /// It keeps pushing back: a flat clamp would pay the fit `ln(guard)` at
    /// `g = 0`, i.e. exactly what it gets at the feasible boundary. The
    /// continuation charges for leaving.
    #[test]
    fn leaving_the_feasible_region_costs_something() {
        let at_guard = guarded(GUARD).0;
        let at_zero = guarded(0.0).0;
        assert!(
            at_zero < at_guard - 1.0,
            "g=0 must be materially worse than g=guard: {at_zero} vs {at_guard}"
        );
    }
}

/// #2695: dividing the time transform by the scale leaves a constant-scale fit
/// unchanged. With `eta_ls ≡ 0` the scale is exactly `1.0`, and a constant-σ fit
/// has no free log-σ coefficient (its constant is the pinned gauge column), so
/// every coefficient it owns reaches the row program through the time and
/// threshold primaries alone. On that block the row program's value, gradient and
/// Hessian must equal the former ratio-only kernel's (`u = h − eta_t·e^{−eta_ls}`,
/// `g = hdot + e^{−eta_ls}·(eta_t·eta_ls' − eta_t')`) exactly, not to a tolerance.
#[cfg(test)]
mod constant_scale_identity_2695_tests {
    use super::*;
    use gam_math::jet_scalar::{JetScalar, Order2};
    use gam_math::nested_dual::JetField;

    /// The former kernel, composed with the same outer plan and the same
    /// exponential leaf as [`sls_row_program`], kept only as this reference.
    fn former_row_nll(
        vars: &[Order2<SLS_ROW_K>; SLS_ROW_K],
        kernel: &SurvivalExactRowKernel,
    ) -> Order2<SLS_ROW_K> {
        let inv_sigma_entry = vars[7]
            .neg()
            .compose_unary(sls_program_exp_stack(-vars[7].value()));
        let u0 = vars[0].sub(&vars[4].mul(&inv_sigma_entry));
        let inv_sigma_exit = vars[6]
            .neg()
            .compose_unary(sls_program_exp_stack(-vars[6].value()));
        let u1 = vars[1].sub(&vars[3].mul(&inv_sigma_exit));
        let g = vars[2].add(&inv_sigma_exit.mul(&vars[3].mul(&vars[8]).sub(&vars[5])));
        let plan = sls_outer_plan::<5>(kernel);
        let mut nll = u0.compose_unary(plan.u0);
        if let Some(stack) = plan.u1 {
            nll = nll.add(&u1.compose_unary(stack));
        }
        if let Some(stack) = plan.g {
            nll = nll.add(&g.compose_unary(stack));
        }
        nll
    }

    #[test]
    fn a_constant_scale_row_is_the_former_kernel_exactly_2695() {
        // (h0, h1, hdot, eta_t_exit, eta_t_entry, eta_t_deriv); the three scale
        // primaries are zero.
        let points: [[f64; 6]; 3] = [
            [0.3, 1.1, 0.8, 0.4, 0.2, 0.0],
            [-0.9, 0.6, 1.3, -0.5, -0.7, 0.1],
            [1.4, 2.2, 0.35, 1.9, 1.6, -0.05],
        ];
        let links = [
            residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            residual_distribution_inverse_link(ResidualDistribution::Logistic),
            residual_distribution_inverse_link(ResidualDistribution::Gumbel),
        ];
        let mut compared = 0usize;
        for link in &links {
            for point in &points {
                for d in [0.0, 1.0] {
                    let mut primary = [0.0; SLS_ROW_K];
                    primary[..6].copy_from_slice(point);
                    let state = survival_predictor_state(
                        primary[0],
                        primary[1],
                        primary[2],
                        -primary[4],
                        -primary[3],
                        -primary[5],
                        0.0,
                        true,
                    );
                    let kernel = SurvivalLocationScaleFamily::exact_row_kernel_from_parts(
                        link, 1.0e-8, 1.3, d, 0, state, 0.0,
                    )
                    .expect("row kernel")
                    .expect("positive-weight row");
                    let vars: [Order2<SLS_ROW_K>; SLS_ROW_K] =
                        std::array::from_fn(|axis| Order2::variable(primary[axis], axis));
                    let (value, gradient, hessian) = sls_row_nll(&vars, &kernel)
                        .expect("row program")
                        .into_channels();
                    let (former_value, former_gradient, former_hessian) =
                        former_row_nll(&vars, &kernel).into_channels();
                    assert_eq!(value, former_value, "{link:?} d={d} {point:?}: value");
                    for a in 0..6 {
                        assert_eq!(
                            gradient[a], former_gradient[a],
                            "{link:?} d={d} {point:?}: gradient[{a}]"
                        );
                        for b in 0..6 {
                            assert_eq!(
                                hessian[a][b], former_hessian[a][b],
                                "{link:?} d={d} {point:?}: hessian[{a}][{b}]"
                            );
                        }
                    }
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, links.len() * points.len() * 2);
    }
}
