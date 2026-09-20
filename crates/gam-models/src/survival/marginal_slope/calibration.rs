//! Denested survival calibration (externally consumed): partitioning the
//! observation window into integration cells and the closed-form denested
//! intercept calibration evaluated over them.

use super::*;
use crate::bms::moving_law_rule::MovingLawError;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        if self.score_dim() == 1 {
            return shared_denested_partition_cells(
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }
        let score_breaks = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let link_breaks = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| self.score_warp_local_cubic_at(beta_h, z),
            |u| {
                if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
                    runtime.local_cubic_at(beta_w.view(), u)
                } else {
                    Ok(Self::zero_score_warp_span())
                }
            },
        )?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    /// The closed-form flex program's certificate anchor at one survival anchor
    /// under a finite law (gam#2926). The row's intercept `a` at marginal index `q`
    /// is the program's own, solved under `N(0, 1)`; at each node `u_k` of `law` the
    /// program's de-nested index `η(u_k)` gives the survival probability
    /// `Φ(−η(u_k))`, and the anchor reads `r = Σ_k w_k Φ(−η(u_k)) − Φ(−q)` through
    /// [`crate::bms::estimated_latent_law::survival_certificate_anchor`].
    pub(crate) fn flex_survival_certificate_anchor(
        &self,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        law: &crate::bms::EmpiricalZGrid,
    ) -> Result<crate::bms::CertificateAnchor, String> {
        if self.score_dim() != 1 {
            return Err(
                "the flex survival anchoring residual is defined for one latent score".to_string(),
            );
        }
        let (a, _) = self.solve_row_survival_intercept_with_slot(q, slope, beta_h, beta_w, None)?;
        let mut probabilities = Vec::with_capacity(law.nodes.len());
        for &u in &law.nodes {
            let eta = self.flex_survival_denested_index(u, a, slope, beta_h, beta_w)?;
            probabilities.push(crate::bms::estimated_latent_law::survival_tail_probability(q, eta));
        }
        crate::bms::estimated_latent_law::survival_certificate_anchor(
            q,
            &law.weights,
            &probabilities,
        )
        .map_err(|reason| format!("flex survival anchor at slope={slope}: {reason}"))
    }

    /// The flex program's de-nested index `η(u)` at node `u` and intercept `a`.
    fn flex_survival_denested_index(
        &self,
        u: f64,
        a: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let partials = shared_observed_denested_cell_partials(
            u,
            a,
            slope,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )?;
        Ok(eval_coeff4_at(&partials.coeff, u))
    }

    /// The moving-law certificate's `(ln S, ln(1 − S))` of one flex survival anchor
    /// under `law` (gam#2926): `S = Σ_k w_k Φ(−η(u_k))` at the program's own
    /// intercept, through [`crate::bms::moving_law_rule::log_grid_anchor_probabilities`].
    fn flex_survival_anchor_log_probabilities(
        &self,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        law: &crate::bms::EmpiricalZGrid,
    ) -> Result<(f64, f64), MovingLawError> {
        if self.score_dim() != 1 {
            return Err(MovingLawError::Unsupported {
                what: "the flex survival anchor probabilities are defined for one latent score",
            });
        }
        let program = |reason: String| MovingLawError::AnchorProgram { reason };
        let (a, _) = self
            .solve_row_survival_intercept_with_slot(q, slope, beta_h, beta_w, None)
            .map_err(program)?;
        crate::bms::moving_law_rule::log_grid_anchor_probabilities(law, |u| {
            Ok(-self
                .flex_survival_denested_index(u, a, slope, beta_h, beta_w)
                .map_err(program)?)
        })
    }

    /// The moving-law certificate's two anchors of `row` (gam#2926): the exit and
    /// entry anchors' `(ln S, ln(1 − S))` under `law`, at `time_row`'s exit and entry
    /// times, with `row`'s marginal index and slope.
    ///
    /// The certificate reads a row's anchors at times its own outcome did not
    /// choose. Its own exit time depends on its score, so `S(t_i | x_i, z_i)` is not
    /// an unbiased observation of the anchor's expectation under the true law, and
    /// a loss taken there favours whichever law leans toward the event times; at
    /// another row's times it is. Reached only where the anchored frame serves the
    /// fit, which it does without a time-wiggle.
    pub(crate) fn moving_law_certificate_anchors(
        &self,
        row: usize,
        time_row: usize,
        block_states: &[ParameterBlockState],
        law: &crate::bms::EmpiricalZGrid,
    ) -> Result<[(f64, f64); 2], MovingLawError> {
        if self.flex_timewiggle_active() {
            return Err(MovingLawError::Unsupported {
                what: "a time-wiggle baseline is served only in closed form, so no moving law is \
                       certified on it",
            });
        }
        let program = |reason: String| MovingLawError::AnchorProgram { reason };
        let beta_time = &block_states[0].beta;
        let marginal = block_states[1].eta[row];
        let q1 = self.design_exit.dot_row(time_row, beta_time) + self.offset_exit[time_row] + marginal;
        let q0 =
            self.design_entry.dot_row(time_row, beta_time) + self.offset_entry[time_row] + marginal;
        let slopes = self.row_slope_channels(row, block_states).map_err(program)?;
        let beta_h = self.flex_score_beta(block_states).map_err(program)?;
        let beta_w = self.flex_link_beta(block_states).map_err(program)?;
        // Every arm is scored at the anchor the fit used: the intercept the fitted
        // family solves on its own law, the closed form on the Gaussian law and the
        // anchored root on a finite one (as the flex program's own solve does).
        let fitted_law = self.flex_law_grid(Some(row)).map_err(program)?;
        let anchor = |q: f64, slope: f64| -> Result<(f64, f64), MovingLawError> {
            if beta_h.is_some() || beta_w.is_some() {
                return self.flex_survival_anchor_log_probabilities(q, slope, beta_h, beta_w, law);
            }
            let observed_slope = self.probit_frailty_scale() * slope;
            let alpha = match fitted_law {
                Some(grid) => solve_anchor(q, observed_slope, grid).map_err(program)?,
                None => q * (1.0 + observed_slope * observed_slope).sqrt(),
            };
            Ok(crate::bms::estimated_latent_law::survival_anchor_log_probabilities(
                alpha,
                observed_slope,
                law,
            )?)
        };
        Ok([anchor(q1, slopes.exit)?, anchor(q0, slopes.entry)?])
    }

    /// The closed-form certificate's two anchors of one row (gam#2926): the exit and
    /// entry anchors under `law`, at the row's own
    /// marginal indices and on each anchor's own slope channel, through the flex
    /// program when a flex block is installed and the rigid closed form otherwise.
    ///
    /// An installed CTN Stage-1 influence absorber is not read. The certificate asks
    /// whether each closed-form anchor satisfies its own defining equation
    /// `Σ∫Φ(−η_raw(a, g, z))φ(z) dz = Φ(−q)` under the estimated law, and that equation
    /// is offset-free: the fit solves the anchor with the offset absent and adds
    /// `o_infl` to the observed index only afterwards, and prediction drops the
    /// absorber (#461). Scoring the offset here would score the Stage-1 correction,
    /// not the anchoring error.
    pub(crate) fn closed_form_certificate_anchors(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        law: &crate::bms::EmpiricalZGrid,
    ) -> Result<[crate::bms::CertificateAnchor; 2], String> {
        let values = self.row_dynamic_q_values(row, block_states)?;
        // A time-constant slope is the degenerate case entry = exit.
        let slopes = self.row_slope_channels(row, block_states)?;
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        // A flex block runs the row through its own de-nested program; without one
        // the anchor is the rigid closed form, the same number in O(nodes).
        let anchor = |q: f64, slope: f64| -> Result<crate::bms::CertificateAnchor, String> {
            if beta_h.is_some() || beta_w.is_some() {
                self.flex_survival_certificate_anchor(q, slope, beta_h, beta_w, law)
            } else {
                crate::bms::estimated_latent_law::closed_form_survival_certificate_anchor(
                    q,
                    self.probit_frailty_scale() * slope,
                    law,
                )
            }
        };
        Ok([anchor(values.q1, slopes.exit)?, anchor(values.q0, slopes.entry)?])
    }

    /// The closed-form certificate's two anchors of one row of a `K ≥ 2` fit
    /// (gam#2926): the exit and entry anchors of the closed form `q·√(1 + s²·rᵀΣ(a)r) + s·rᵀz` under the joint latent law the fit
    /// would re-solve on, transported to the row's context (gam#2929). `r` is the
    /// row's per-score slope vector, or the shared slope on every score, read at
    /// each anchor's own follow-up time. Like the scalar certificate it reads no
    /// influence absorber.
    pub(crate) fn closed_form_joint_certificate_anchors(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        law: &JointLatentLawRuntime,
        workspace: &mut JointCertificateWorkspace,
    ) -> Result<[crate::bms::CertificateAnchor; 2], String> {
        let k = law.score_dim();
        if k != self.score_dim() {
            return Err(format!(
                "survival marginal-slope joint certificate law carries K={k} scores for a family \
                 of K={}",
                self.score_dim()
            ));
        }
        let values = self.row_dynamic_q_values(row, block_states)?;
        let (exit_slopes, entry_slopes) = if self.per_z_slope_active() {
            self.fill_slope_values_for_row(row, block_states, &mut workspace.slopes)?;
            let slopes = workspace.slopes.values().to_vec();
            (slopes.clone(), slopes)
        } else {
            let channels = self.row_slope_channels(row, block_states)?;
            (vec![channels.exit; k], vec![channels.entry; k])
        };
        law.row_nodes_into(row, &mut workspace.nodes)?;
        let scale = self.probit_frailty_scale();
        let covariance = self.score_covariance.at_row(row);
        let mut anchor = |q: f64, slopes: &[f64]| -> Result<crate::bms::CertificateAnchor, String> {
            let variance = scale * scale * covariance.quadratic_form_unchecked(slopes);
            let alpha = q * (1.0 + variance).sqrt();
            workspace.probabilities.clear();
            for node in workspace.nodes.chunks_exact(k) {
                let drive = scale
                    * slopes
                        .iter()
                        .zip(node.iter())
                        .map(|(slope, value)| slope * value)
                        .sum::<f64>();
                workspace.probabilities.push(
                    crate::bms::estimated_latent_law::survival_tail_probability(q, alpha + drive),
                );
            }
            crate::bms::estimated_latent_law::survival_certificate_anchor(
                q,
                law.weights(),
                &workspace.probabilities,
            )
            .map_err(|reason| format!("survival marginal-slope joint anchor at row {row}: {reason}"))
        };
        Ok([anchor(values.q1, &exit_slopes)?, anchor(values.q0, &entry_slopes)?])
    }

    pub(crate) fn evaluate_denested_survival_calibration(
        &self,
        a: f64,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -crate::probability::normal_cdf(-q);
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let pos_cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: pos_cell.left,
                right: pos_cell.right,
                c0: -pos_cell.c0,
                c1: -pos_cell.c1,
                c2: -pos_cell.c2,
                c3: -pos_cell.c3,
            };
            let state = exact_kernel::evaluate_cell_moments(neg_cell, 9)?;
            f += state.value;
            let (dc_da_pos, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (dc_daa_pos, _, _) = exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_pos, -scale);
            let dc_daa = scale_coeff4(dc_daa_pos, -scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact_kernel::cell_second_derivative_from_moments(
                neg_cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
        }
        Ok((f, f_a, f_aa))
    }
}

/// Scratch for [`SurvivalMarginalSlopeFamily::closed_form_joint_certificate_anchors`],
/// one per rayon job.
pub(crate) struct JointCertificateWorkspace {
    slopes: SlopeRowWorkspace,
    nodes: Vec<f64>,
    probabilities: Vec<f64>,
}

impl JointCertificateWorkspace {
    pub(crate) fn new(
        family: &SurvivalMarginalSlopeFamily,
        law: &JointLatentLawRuntime,
    ) -> Result<Self, String> {
        Ok(Self {
            slopes: family.slope_row_workspace()?,
            nodes: vec![0.0; law.node_count() * law.score_dim()],
            probabilities: Vec::with_capacity(law.node_count()),
        })
    }
}
