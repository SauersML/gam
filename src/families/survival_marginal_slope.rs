use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyPsiDesignAction,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, build_block_spatial_psi_derivatives, custom_family_outer_capability,
    evaluate_custom_family_joint_hyper, first_psi_linear_map, fit_custom_family,
    second_psi_linear_map,
};
use crate::estimate::{FitOptions, UnifiedFitResult, fit_gam};
use crate::families::bernoulli_marginal_slope::{
    MultiDirJet, unary_derivatives_exp, unary_derivatives_log, unary_derivatives_log_normal_pdf,
    unary_derivatives_neglog_phi, unary_derivatives_sqrt,
};
use crate::families::survival_location_scale::{
    TimeBlockInput, structural_nonnegative_time_constraints,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::cell::RefCell;
use std::sync::Arc;

// ── Spec and result types ─────────────────────────────────────────────

#[derive(Clone)]
pub struct SurvivalMarginalSlopeTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub logslopespec: TermCollectionSpec,
}

pub struct SurvivalMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub logslopespec_resolved: TermCollectionSpec,
    pub logslope_design: TermCollectionDesign,
    pub baseline_logslope: f64,
    pub time_block_penalties_len: usize,
}

// ── Family struct ─────────────────────────────────────────────────────

/// The time block has one beta vector but THREE design matrices (entry, exit,
/// derivative-at-exit). The ParameterBlockSpec uses the exit design as its
/// "official" design, so block_states[0].eta = design_exit @ beta + offset_exit.
/// This eta is NOT used in the likelihood computation — row_neglog_directional
/// recomputes all 3 linear predictors from beta_time directly. The exit-design
/// eta exists only to satisfy the CustomFamily/PIRLS interface; ExactNewton
/// blocks do not use eta for working response/weights.
#[derive(Clone)]
struct SurvivalMarginalSlopeFamily {
    n: usize,
    event: Array1<f64>,
    weights: Array1<f64>,
    z: Array1<f64>,
    derivative_guard: f64,
    /// Time block: 3 designs sharing one beta vector.
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    constraint_design_derivative: Option<Array2<f64>>,
    offset_entry: Array1<f64>,
    offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    constraint_derivative_offset: Option<Array1<f64>>,
    /// Log-slope block: standard single design.
    logslope_design: Array2<f64>,
}

// ── Block layout ──────────────────────────────────────────────────────

#[derive(Clone)]
struct BlockSlices {
    time: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    total: usize,
}

fn block_slices(block_states: &[ParameterBlockState]) -> BlockSlices {
    let time = 0..block_states[0].beta.len();
    let logslope = time.end..time.end + block_states[1].beta.len();
    BlockSlices {
        total: logslope.end,
        time,
        logslope,
    }
}

// ── Primary-space helpers ─────────────────────────────────────────────

// Primary scalar indices: 0=q0, 1=q1, 2=qd1, 3=g
const N_PRIMARY: usize = 4;

fn unit_primary_direction(idx: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    out[idx] = 1.0;
    out
}

// ── Eval cache ────────────────────────────────────────────────────────

#[derive(Clone)]
struct RowPrimaryBase {
    nll: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

struct EvalCache {
    slices: BlockSlices,
    row_bases: Vec<RowPrimaryBase>,
}

// ── Row-level NLL computation ─────────────────────────────────────────

impl SurvivalMarginalSlopeFamily {
    /// Per-row NLL and its directional derivatives through 4 primary scalars.
    ///
    /// NLL_i = w_i * [ (1-d)·neglogΦ(-η₁) − neglogΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
    ///
    /// where η = a(t) + β·z, a(t) = q(t)·√(1+β²), β = exp(g).
    ///
    /// block_states[0].eta is from the exit design and is NOT used here;
    /// all 3 time-block linear predictors are recomputed from beta_time
    /// because the time block has 3 design matrices sharing one coefficient vector.
    fn row_neglog_directional(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[Array1<f64>],
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "survival marginal-slope row directional expects 0..=4 directions, got {k}"
            ));
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let zi = self.z[row];

        // Primary scalar jets: q0, q1, qd1, g
        let q0_first: Vec<f64> = dirs.iter().map(|dir| dir[0]).collect();
        let q1_first: Vec<f64> = dirs.iter().map(|dir| dir[1]).collect();
        let qd1_first: Vec<f64> = dirs.iter().map(|dir| dir[2]).collect();
        let g_first: Vec<f64> = dirs.iter().map(|dir| dir[3]).collect();

        // Compute q0, q1, qd1 from beta_time directly. The time block has a
        // single beta vector but 3 different design matrices (entry, exit, derivative).
        let beta_time = &block_states[0].beta;
        let q0_val = self.design_entry.row(row).dot(beta_time) + self.offset_entry[row];
        let q1_val = self.design_exit.row(row).dot(beta_time) + self.offset_exit[row];
        let qd1_val =
            self.design_derivative_exit.row(row).dot(beta_time) + self.derivative_offset_exit[row];
        let g_val = block_states[1].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, &q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, &q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, &qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, &g_first);

        // beta = exp(g)
        let beta_jet = g_jet.compose_unary(unary_derivatives_exp(g_val));
        // c = sqrt(1 + beta^2)
        let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&beta_jet.mul(&beta_jet));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        // a0 = q0 * c, a1 = q1 * c, ad1 = qd1 * c
        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);

        // eta0 = a0 + beta * z, eta1 = a1 + beta * z
        let z_jet = MultiDirJet::constant(k, zi);
        let eta0_jet = a0_jet.add(&beta_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&beta_jet.mul(&z_jet));

        // NLL_i = w_i * {
        //   (1-d_i) * neglogphi(-eta1)  [exit survival for censored]
        //   - neglogphi(-eta0)           [entry survival, subtracted]
        //   - d_i * log_phi(eta1)        [event log-density of normal]
        //   - d_i * log(ad1)             [event log time-derivative]
        // }

        // Entry survival term: -neglogphi(-eta0) = log Phi(-eta0) = log S(t0|z)
        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0); // note: -w * neglogphi(-eta0) = w * log Phi(-eta0)

        // Exit survival term: (1-d)*neglogphi(-eta1) = -(1-d)*log Phi(-eta1)
        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        // Event density: -d * log phi(eta1)
        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        // Time derivative: -d * log(ad1)
        // If ad1_val is tiny/negative (before monotonicity converges), derivatives
        // through log() are meaningless — return a constant penalty instead.
        let time_deriv_term = if di > 0.0 {
            let ad1_val = ad1_jet.coeff(0);
            if ad1_val > 1e-300 {
                ad1_jet
                    .compose_unary(unary_derivatives_log(ad1_val))
                    .scale(-wi * di)
            } else {
                // At the floor: log(1e-300) is a large negative constant, derivatives are zero
                // because the constraint should push us away from here
                MultiDirJet::constant(k, (-wi * di) * (1e-300_f64).ln())
            }
        } else {
            MultiDirJet::zero(k)
        };

        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        if k == 0 {
            Ok(total.coeff(0))
        } else {
            Ok(total.coeff(total.full_mask()))
        }
    }

    fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let nll = self.row_neglog_directional(row, block_states, &[])?;
        let mut grad = Array1::<f64>::zeros(N_PRIMARY);
        let mut hess = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            grad[a] = self.row_neglog_directional(row, block_states, &[da.clone()])?;
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value =
                    self.row_neglog_directional(row, block_states, &[da.clone(), db.clone()])?;
                hess[[a, b]] = value;
                hess[[b, a]] = value;
            }
        }
        Ok((nll, grad, hess))
    }

    fn build_eval_cache(&self, block_states: &[ParameterBlockState]) -> Result<EvalCache, String> {
        let slices = block_slices(block_states);
        let row_bases = (0..self.n)
            .map(|row| {
                let (nll, gradient, hessian) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                Ok(RowPrimaryBase {
                    nll,
                    gradient,
                    hessian,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(EvalCache { slices, row_bases })
    }

    fn row_primary_gradient_hessian<'a>(
        &self,
        row: usize,
        cache: &'a EvalCache,
    ) -> (&'a Array1<f64>, &'a Array2<f64>) {
        let base = &cache.row_bases[row];
        (&base.gradient, &base.hessian)
    }

    fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value = self.row_neglog_directional(
                    row,
                    block_states,
                    &[da.clone(), db.clone(), dir.clone()],
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value = self.row_neglog_directional(
                    row,
                    block_states,
                    &[da.clone(), db.clone(), dir_u.clone(), dir_v.clone()],
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    // ── Pullback through design matrices ──────────────────────────────

    /// Map a primary-space vector [f_q0, f_q1, f_qd1, f_g] to coefficient space.
    fn pullback_primary_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary_vec: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(slices.total);
        // Time block: 3 primary scalars (q0, q1, qd1) all map to the same beta_time
        let x_entry = self.design_entry.row(row).to_owned();
        let x_exit = self.design_exit.row(row).to_owned();
        let x_deriv = self.design_derivative_exit.row(row).to_owned();
        let time_contrib =
            &x_entry * primary_vec[0] + &x_exit * primary_vec[1] + &x_deriv * primary_vec[2];
        out.slice_mut(s![slices.time.clone()]).assign(&time_contrib);
        // Slope block: primary scalar g
        let g_row = self.logslope_design.row(row).to_owned();
        out.slice_mut(s![slices.logslope.clone()])
            .assign(&(&g_row * primary_vec[3]));
        out
    }

    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let x_entry = self.design_entry.row(row).to_owned();
        let x_exit = self.design_exit.row(row).to_owned();
        let x_deriv = self.design_derivative_exit.row(row).to_owned();
        let g_row = self.logslope_design.row(row).to_owned();

        // Time-time block (indices 0,1,2 × 0,1,2 in primary space)
        // We have 3 design vectors for the time block. The time-time Hessian is:
        // sum over (i,j) in {entry,exit,deriv}×{entry,exit,deriv} of
        //   X_i * X_j^T * H_primary[idx_i, idx_j]
        let time_designs = [&x_entry, &x_exit, &x_deriv];
        for (i, xi) in time_designs.iter().enumerate() {
            for (j, xj) in time_designs.iter().enumerate() {
                let h_val = primary_hessian[[i, j]];
                if h_val.abs() > 1e-30 {
                    let outer = xi
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xj.view().insert_axis(Axis(0)))
                        * h_val;
                    target
                        .slice_mut(s![slices.time.clone(), slices.time.clone()])
                        .scaled_add(1.0, &outer);
                }
            }
        }

        // Time-slope cross block (indices 0,1,2 × 3)
        for (i, xi) in time_designs.iter().enumerate() {
            let h_val = primary_hessian[[i, 3]];
            if h_val.abs() > 1e-30 {
                let outer = xi
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&g_row.view().insert_axis(Axis(0)))
                    * h_val;
                target
                    .slice_mut(s![slices.time.clone(), slices.logslope.clone()])
                    .scaled_add(1.0, &outer);
                target
                    .slice_mut(s![slices.logslope.clone(), slices.time.clone()])
                    .scaled_add(1.0, &outer.t().to_owned());
            }
        }

        // Slope-slope block (index 3 × 3)
        let gg = g_row
            .view()
            .insert_axis(Axis(1))
            .dot(&g_row.view().insert_axis(Axis(0)))
            * primary_hessian[[3, 3]];
        target
            .slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .scaled_add(1.0, &gg);
    }

    /// Map a coefficient-space direction to primary-space for a given row.
    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        let d_time = d_beta_flat.slice(s![slices.time.clone()]).to_owned();
        out[0] = self.design_entry.row(row).dot(&d_time);
        out[1] = self.design_exit.row(row).dot(&d_time);
        out[2] = self.design_derivative_exit.row(row).dot(&d_time);
        out[3] = self
            .logslope_design
            .row(row)
            .dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned());
        out
    }

    // ── Joint gradient + Hessian ──────────────────────────────────────

    fn joint_gradient_hessian(
        &self,
        block_states: &[ParameterBlockState],
        need_hessian: bool,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String> {
        let cache = self.build_eval_cache(block_states)?;
        let slices = &cache.slices;
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let mut hessian = need_hessian.then(|| Array2::<f64>::zeros((slices.total, slices.total)));
        let mut ll = 0.0;
        for i in 0..self.n {
            ll -= cache.row_bases[i].nll;

            let (f_pi, f_pipi) = self.row_primary_gradient_hessian(i, &cache);
            gradient -= &self.pullback_primary_vector(i, slices, f_pi);
            if let Some(ref mut hmat) = hessian {
                self.add_pullback_primary_hessian(hmat, i, slices, f_pipi);
            }
        }
        Ok((ll, gradient, hessian))
    }

    // ── Hessian directional derivatives ───────────────────────────────

    fn joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, &slices, d_beta_flat);
            let third = self.row_primary_third_contracted(row, block_states, &row_dir)?;
            self.add_pullback_primary_hessian(&mut out, row, &slices, &third);
        }
        Ok(Some(out))
    }

    fn joint_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let row_u = self.row_primary_direction_from_flat(row, &slices, d_beta_u);
            let row_v = self.row_primary_direction_from_flat(row, &slices, d_beta_v);
            let fourth = self.row_primary_fourth_contracted(row, block_states, &row_u, &row_v)?;
            self.add_pullback_primary_hessian(&mut out, row, &slices, &fourth);
        }
        Ok(Some(out))
    }

    fn joint_hessian_matvec_from_cache(
        &self,
        direction: &Array1<f64>,
        cache: &EvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let mut out = Array1::<f64>::zeros(slices.total);
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, slices, direction);
            let (_, row_hessian) = self.row_primary_gradient_hessian(row, cache);
            let row_action = row_hessian.dot(&row_dir);
            out += &self.pullback_primary_vector(row, slices, &row_action);
        }
        Ok(out)
    }

    fn joint_hessian_diagonal_from_cache(&self, cache: &EvalCache) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let mut diagonal = Array1::<f64>::zeros(slices.total);
        for row in 0..self.n {
            let (_, row_hessian) = self.row_primary_gradient_hessian(row, cache);

            // Time block contributions from entry, exit, derivative designs
            let designs = [
                (0, &self.design_entry),
                (1, &self.design_exit),
                (2, &self.design_derivative_exit),
            ];
            for &(pi, ref des) in &designs {
                for (local_idx, &value) in des.row(row).iter().enumerate() {
                    diagonal[slices.time.start + local_idx] +=
                        value * value * row_hessian[[pi, pi]];
                }
                for &(pj, ref des_j) in &designs {
                    if pj <= pi {
                        continue;
                    }
                    for local_idx in 0..des.ncols() {
                        diagonal[slices.time.start + local_idx] += 2.0
                            * des[[row, local_idx]]
                            * des_j[[row, local_idx]]
                            * row_hessian[[pi, pj]];
                    }
                }
            }

            for (local_idx, &value) in self.logslope_design.row(row).iter().enumerate() {
                diagonal[slices.logslope.start + local_idx] += value * value * row_hessian[[3, 3]];
            }
        }
        Ok(diagonal)
    }

    // ── Psi (spatial length-scale) derivatives ────────────────────────

    fn resolve_psi_location(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Option<(usize, usize)> {
        let mut cursor = 0usize;
        for (block_idx, block) in derivative_blocks.iter().enumerate() {
            if psi_index < cursor + block.len() {
                return Some((block_idx, psi_index - cursor));
            }
            cursor += block.len();
        }
        None
    }

    fn psi_design_row_vector(
        &self,
        row: usize,
        deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            total_rows,
            p,
            0..total_rows,
            label,
        )
        .ok();
        first_psi_linear_map(action.as_ref(), &deriv.x_psi, total_rows, p).row_vector(row)
    }

    fn psi_second_design_row_vector(
        &self,
        row: usize,
        deriv_i: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        deriv_j: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        local_j: usize,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            total_rows,
            p,
            0..total_rows,
            label,
        )?;
        let dense = deriv_i
            .x_psi_psi
            .as_ref()
            .and_then(|rows| rows.get(local_j));
        second_psi_linear_map(action.as_ref(), dense, total_rows, p).row_vector(row)
    }

    /// Map a psi derivative to a primary-space direction for a given row.
    ///
    /// Only the logslope block (block 1) has spatial length-scale parameters.
    /// The time block (block 0) is a pure B-spline on time with no spatial terms,
    /// so its derivative_blocks entry is always empty and resolve_psi_location
    /// never maps to block 0.
    fn row_primary_psi_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        match block_idx {
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi: only logslope block (1) has spatial terms, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_action_on_direction(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        match block_idx {
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned());
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi action: only logslope block (1) has spatial terms, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_second_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some(Array1::<f64>::zeros(N_PRIMARY)));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        match block_i {
            1 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?;
                out[3] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "survival marginal-slope psi second: only logslope block (1) has spatial terms, got block {block_i}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn embedded_psi_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, Array1<f64>)>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(slices.total);
        match block_idx {
            1 => out
                .slice_mut(s![slices.logslope.clone()])
                .assign(&self.psi_design_row_vector(
                    row,
                    deriv,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?),
            _ => {
                return Err(format!(
                    "survival marginal-slope psi embedding: only logslope block (1) has spatial terms, got block {block_idx}"
                ));
            }
        }
        Ok(Some((block_idx, out)))
    }

    fn embedded_psi_second_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<(usize, Array1<f64>)>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some((block_i, Array1::<f64>::zeros(slices.total))));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(slices.total);
        match block_i {
            1 => out.slice_mut(s![slices.logslope.clone()]).assign(
                &self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.n,
                    self.logslope_design.ncols(),
                    "SurvivalMarginalSlope logslope",
                )?,
            ),
            _ => {
                return Err(format!(
                    "survival marginal-slope psi second embedding: only logslope block (1) has spatial terms, got block {block_i}"
                ));
            }
        }
        Ok(Some((block_i, out)))
    }

    // ── Psi terms (first and second order) ────────────────────────────

    fn psi_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx, _)) =
            self.embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        if block_idx != 1 {
            return Err(format!(
                "survival marginal-slope psi_terms: only logslope block (1) expected, got {block_idx}"
            ));
        }
        let idx_primary = 3usize;
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let Some(dir) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_index)?
            else {
                continue;
            };
            let (f_pi, f_pipi) = if let Some(c) = cache {
                let (g, h) = self.row_primary_gradient_hessian(row, c);
                (g.clone(), h.clone())
            } else {
                let (_, g, h) = self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                (g, h)
            };
            let third = self.row_primary_third_contracted(row, block_states, &dir)?;
            let (_, left_vec) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_index)?
                .ok_or_else(|| "missing survival marginal-slope psi vector".to_string())?;
            objective_psi += f_pi.dot(&dir);
            score_psi += &(left_vec.clone() * f_pi[idx_primary]);
            score_psi += &self.pullback_primary_vector(row, &slices, &f_pipi.dot(&dir));

            let right_vec =
                self.pullback_primary_vector(row, &slices, &f_pipi.row(idx_primary).to_owned());
            hessian_psi += &left_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&right_vec.view().insert_axis(Axis(0)));
            hessian_psi += &right_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&left_vec.view().insert_axis(Axis(0)));
            self.add_pullback_primary_hessian(&mut hessian_psi, row, &slices, &third);
        }
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner(block_states, derivative_blocks, psi_index, None)
    }

    fn psi_second_order_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = block_slices(block_states);
        let Some((_, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_i)? else {
            return Ok(None);
        };
        let Some((_, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_j)? else {
            return Ok(None);
        };
        let idx_i = 3usize;
        let idx_j = 3usize;
        let mut objective_psi_psi = 0.0;
        let mut score_psi_psi = Array1::<f64>::zeros(slices.total);
        let mut hessian_psi_psi = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let Some(dir_i) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_i)?
            else {
                continue;
            };
            let Some(dir_j) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_j)?
            else {
                continue;
            };
            let dir_ij = self
                .row_primary_psi_second_direction(
                    row,
                    block_states,
                    derivative_blocks,
                    psi_i,
                    psi_j,
                )?
                .unwrap_or_else(|| Array1::<f64>::zeros(N_PRIMARY));
            let (f_pi, f_pipi) = if let Some(c) = cache {
                let (g, h) = self.row_primary_gradient_hessian(row, c);
                (g.clone(), h.clone())
            } else {
                let (_, g, h) = self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                (g, h)
            };
            let third_i = self.row_primary_third_contracted(row, block_states, &dir_i)?;
            let third_j = self.row_primary_third_contracted(row, block_states, &dir_j)?;
            let fourth = self.row_primary_fourth_contracted(row, block_states, &dir_i, &dir_j)?;
            let (_, left_i) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_i)?
                .ok_or_else(|| "missing psi_i vector".to_string())?;
            let (_, left_j) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_j)?
                .ok_or_else(|| "missing psi_j vector".to_string())?;
            let left_ij = self
                .embedded_psi_second_vector(row, &slices, derivative_blocks, psi_i, psi_j)?
                .map(|(_, v)| v)
                .unwrap_or_else(|| Array1::<f64>::zeros(slices.total));

            objective_psi_psi += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);

            if left_ij.iter().any(|v| v.abs() > 0.0) {
                let idx_ij = 3usize; // only logslope has spatial terms
                score_psi_psi += &(left_ij.clone() * f_pi[idx_ij]);
            }
            score_psi_psi += &(left_i.clone() * f_pipi.row(idx_i).dot(&dir_j));
            score_psi_psi += &(left_j.clone() * f_pipi.row(idx_j).dot(&dir_i));
            score_psi_psi += &self.pullback_primary_vector(row, &slices, &f_pipi.dot(&dir_ij));
            score_psi_psi += &self.pullback_primary_vector(row, &slices, &third_i.dot(&dir_j));

            if left_ij.iter().any(|v| v.abs() > 0.0) {
                let idx_ij = 3usize; // only logslope has spatial terms
                let right_ij =
                    self.pullback_primary_vector(row, &slices, &f_pipi.row(idx_ij).to_owned());
                hessian_psi_psi += &left_ij
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&right_ij.view().insert_axis(Axis(0)));
                hessian_psi_psi += &right_ij
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&left_ij.view().insert_axis(Axis(0)));
            }

            let scalar_ij = f_pipi[[idx_i, idx_j]];
            hessian_psi_psi += &(left_i
                .view()
                .insert_axis(Axis(1))
                .dot(&left_j.view().insert_axis(Axis(0)))
                * scalar_ij);
            hessian_psi_psi += &(left_j
                .view()
                .insert_axis(Axis(1))
                .dot(&left_i.view().insert_axis(Axis(0)))
                * scalar_ij);

            let right_i =
                self.pullback_primary_vector(row, &slices, &third_j.row(idx_i).to_owned());
            hessian_psi_psi += &left_i
                .view()
                .insert_axis(Axis(1))
                .dot(&right_i.view().insert_axis(Axis(0)));
            hessian_psi_psi += &right_i
                .view()
                .insert_axis(Axis(1))
                .dot(&left_i.view().insert_axis(Axis(0)));

            let right_j =
                self.pullback_primary_vector(row, &slices, &third_i.row(idx_j).to_owned());
            hessian_psi_psi += &left_j
                .view()
                .insert_axis(Axis(1))
                .dot(&right_j.view().insert_axis(Axis(0)));
            hessian_psi_psi += &right_j
                .view()
                .insert_axis(Axis(1))
                .dot(&left_j.view().insert_axis(Axis(0)));

            self.add_pullback_primary_hessian(&mut hessian_psi_psi, row, &slices, &fourth);
            let third_ij = self.row_primary_third_contracted(row, block_states, &dir_ij)?;
            self.add_pullback_primary_hessian(&mut hessian_psi_psi, row, &slices, &third_ij);
        }
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
        }))
    }

    fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner(block_states, derivative_blocks, psi_i, psi_j, None)
    }

    fn psi_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let Some((_, _)) = self.embedded_psi_vector(0, &slices, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let idx_primary = 3usize; // only logslope has spatial terms
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row in 0..self.n {
            let row_dir = self.row_primary_direction_from_flat(row, &slices, d_beta_flat);
            let Some(psi_dir) =
                self.row_primary_psi_direction(row, block_states, derivative_blocks, psi_index)?
            else {
                continue;
            };
            let psi_action = self
                .row_primary_psi_action_on_direction(
                    row,
                    &slices,
                    derivative_blocks,
                    psi_index,
                    d_beta_flat,
                )?
                .unwrap_or_else(|| Array1::<f64>::zeros(N_PRIMARY));
            let third_beta = self.row_primary_third_contracted(row, block_states, &row_dir)?;
            let fourth =
                self.row_primary_fourth_contracted(row, block_states, &row_dir, &psi_dir)?;
            let (_, left_vec) = self
                .embedded_psi_vector(row, &slices, derivative_blocks, psi_index)?
                .ok_or_else(|| "missing psi vector".to_string())?;
            let right_vec =
                self.pullback_primary_vector(row, &slices, &third_beta.row(idx_primary).to_owned());
            out += &left_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&right_vec.view().insert_axis(Axis(0)));
            out += &right_vec
                .view()
                .insert_axis(Axis(1))
                .dot(&left_vec.view().insert_axis(Axis(0)));
            self.add_pullback_primary_hessian(&mut out, row, &slices, &fourth);
            let third_action = self.row_primary_third_contracted(row, block_states, &psi_action)?;
            self.add_pullback_primary_hessian(&mut out, row, &slices, &third_action);
        }
        Ok(Some(out))
    }
}

// ── Workspace structs ─────────────────────────────────────────────────

struct SurvivalMarginalSlopeHessianWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    cache: EvalCache,
}

struct SurvivalMarginalSlopePsiWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: EvalCache,
}

impl SurvivalMarginalSlopeHessianWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let cache = family.build_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            cache,
        })
    }
}

impl SurvivalMarginalSlopePsiWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let cache = family.build_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            cache,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for SurvivalMarginalSlopeHessianWorkspace {
    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.family
            .joint_hessian_matvec_from_cache(beta_flat, &self.cache)
            .map(Some)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        self.family
            .joint_hessian_diagonal_from_cache(&self.cache)
            .map(Some)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.joint_hessian_second_directional_derivative(
            &self.block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalMarginalSlopePsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.psi_terms_inner(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            Some(&self.cache),
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family.psi_second_order_terms_inner(
            &self.block_states,
            &self.derivative_blocks,
            psi_i,
            psi_j,
            Some(&self.cache),
        )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.psi_hessian_directional_derivative(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }
}

// ── CustomFamily impl ─────────────────────────────────────────────────

const EXACT_OUTER_HESSIAN_MAX_ROW_PAIR_WORK: usize = 2_000_000;

impl CustomFamily for SurvivalMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_outer_derivative_order(
        &self,
        _: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        let primary_total = N_PRIMARY;
        let row_pair_work = self.n.saturating_mul(primary_total * primary_total);
        if row_pair_work > EXACT_OUTER_HESSIAN_MAX_ROW_PAIR_WORK {
            ExactOuterDerivativeOrder::First
        } else {
            ExactOuterDerivativeOrder::Second
        }
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, gradient, hessian) = self.joint_gradient_hessian(block_states, true)?;
        let hessian = hessian.ok_or_else(|| "joint hessian unavailable".to_string())?;
        let slices = block_slices(block_states);
        let blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![slices.time.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian
                        .slice(s![slices.time.clone(), slices.time.clone()])
                        .to_owned(),
                ),
            },
            BlockWorkingSet::ExactNewton {
                gradient: gradient.slice(s![slices.logslope.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(
                    hessian
                        .slice(s![slices.logslope.clone(), slices.logslope.clone()])
                        .to_owned(),
                ),
            },
        ];
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        // Fast path: just compute NLL without gradient/Hessian.
        // Avoids build_eval_cache which computes per-row gradient+Hessian
        // (14 jet evaluations per row) that log_likelihood_only never uses.
        let mut ll = 0.0;
        for i in 0..self.n {
            let row_neglog = self.row_neglog_directional(i, block_states, &[])?;
            ll -= row_neglog;
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_gradient_hessian(block_states, true)
            .map(|(_, _, h)| h)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(Some(Arc::new(SurvivalMarginalSlopeHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        )?)))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.joint_hessian_second_directional_derivative(block_states, d_beta_u_flat, d_beta_v_flat)
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms(block_states, derivative_blocks, psi_index)
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms(block_states, derivative_blocks, psi_i, psi_j)
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.psi_hessian_directional_derivative(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(SurvivalMarginalSlopePsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        )?)))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == 0 {
            let constraint_rows = self
                .constraint_design_derivative
                .as_ref()
                .unwrap_or(&self.design_derivative_exit);
            let constraint_offsets = self
                .constraint_derivative_offset
                .as_ref()
                .unwrap_or(&self.derivative_offset_exit);
            // Monotonicity constraint: design_derivative_exit @ beta_time + offset >= guard
            // i.e. design_derivative_exit @ beta_time >= guard - offset
            if let Some(structural) = structural_nonnegative_time_constraints(
                constraint_rows,
                constraint_offsets,
                self.derivative_guard,
            ) {
                return Ok(Some(structural));
            }
            Ok(Some(LinearInequalityConstraints {
                a: constraint_rows.clone(),
                b: Array1::from_iter(
                    constraint_offsets
                        .iter()
                        .map(|&o| self.derivative_guard - o),
                ),
            }))
        } else {
            Ok(None)
        }
    }
}

// ── Building block specs ──────────────────────────────────────────────

fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &Array2<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(Arc::new(design_exit.clone())),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_block.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_logslope_blockspec(
    design: &TermCollectionDesign,
    baseline: f64,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: design.design.clone(),
        offset: Array1::from_elem(design.design.nrows(), baseline),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn inner_fit(
    family: &SurvivalMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

fn joint_setup(
    time_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = time_penalties + logslope_penalties;
    let rho0vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    );
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(logslope_kappa.as_array().iter());
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(logslope_kappa.dims_per_term());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values.clone()), dims.clone());
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&dims, kappa_options);
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&dims, kappa_options);
    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
    let n = spec.age_entry.len();
    if spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.z.len() != n
    {
        return Err(format!(
            "survival-marginal-slope row mismatch: entry={}, exit={}, event={}, weights={}, z={}",
            n,
            spec.age_exit.len(),
            spec.event_target.len(),
            spec.weights.len(),
            spec.z.len()
        ));
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err("survival-marginal-slope requires finite non-negative weights".to_string());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err("survival-marginal-slope requires finite z values".to_string());
    }
    for i in 0..n {
        if spec.age_exit[i] < spec.age_entry[i] {
            return Err(format!(
                "survival-marginal-slope row {i}: exit time ({}) < entry time ({})",
                spec.age_exit[i], spec.age_entry[i]
            ));
        }
    }
    let p_entry = spec.time_block.design_entry.ncols();
    let p_exit = spec.time_block.design_exit.ncols();
    let p_deriv = spec.time_block.design_derivative_exit.ncols();
    if p_exit != p_entry || p_deriv != p_entry {
        return Err(format!(
            "survival-marginal-slope time block design column mismatch: entry={p_entry}, exit={p_exit}, deriv={p_deriv}"
        ));
    }
    if let Some(rows) = spec.time_block.constraint_design_derivative.as_ref() {
        if rows.ncols() != p_entry {
            return Err(format!(
                "survival-marginal-slope time monotonicity constraint width mismatch: got {}, expected {p_entry}",
                rows.ncols()
            ));
        }
        let offsets = spec
            .time_block
            .constraint_derivative_offset
            .as_ref()
            .ok_or_else(|| {
                "survival-marginal-slope monotonicity constraints are missing derivative offsets"
                    .to_string()
            })?;
        if offsets.len() != rows.nrows() {
            return Err(format!(
                "survival-marginal-slope monotonicity constraint row mismatch: rows={} offsets={}",
                rows.nrows(),
                offsets.len()
            ));
        }
    } else if spec.time_block.constraint_derivative_offset.is_some() {
        return Err(
            "survival-marginal-slope monotonicity derivative offsets were provided without constraint rows"
                .to_string(),
        );
    }
    Ok(())
}

/// Compute a simple baseline log-slope from a pooled probit survival fit.
fn pooled_survival_baseline(event: &Array1<f64>, z: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    // Simple: regress event ~ z with probit link to get initial slope
    let n = event.len();
    if n == 0 {
        return (-2.0_f64).ln(); // log(0.1) ~ small slope
    }
    let fit = fit_gam(
        {
            let mut x = Array2::<f64>::zeros((n, 2));
            x.column_mut(0).fill(1.0);
            x.column_mut(1).assign(z);
            x
        }
        .view(),
        event.view(),
        weights.view(),
        Array1::zeros(n).view(),
        &[],
        LikelihoodFamily::BinomialProbit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    );
    match fit {
        Ok(result) => {
            let b = result.beta.get(1).copied().unwrap_or(0.1).abs().max(1e-6);
            b.ln()
        }
        Err(_) => (-2.0_f64).ln(),
    }
}

// ── Public fitting function ───────────────────────────────────────────

pub fn fit_survival_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: SurvivalMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    validate_spec(&spec)?;
    let n = spec.age_entry.len();
    let baseline_logslope = pooled_survival_baseline(&spec.event_target, &spec.z, &spec.weights);

    let logslope_design =
        build_term_collection_design(data, &spec.logslopespec).map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_spatial_length_scale_terms_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let time_penalties_len = spec.time_block.penalties.len();
    let setup = joint_setup(
        time_penalties_len,
        &logslopespec_boot,
        logslope_design.penalties.len(),
        kappa_options,
    );

    let hints = RefCell::new((None::<Array1<f64>>, None::<Array1<f64>>));
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);

    let event = spec.event_target.clone();
    let weights = spec.weights.clone();
    let z = spec.z.clone();
    let derivative_guard = spec.derivative_guard;
    let design_entry = spec.time_block.design_entry.clone();
    let design_exit = spec.time_block.design_exit.clone();
    let design_derivative_exit = spec.time_block.design_derivative_exit.clone();
    let constraint_design_derivative = spec.time_block.constraint_design_derivative.clone();
    let offset_entry = spec.time_block.offset_entry.clone();
    let offset_exit = spec.time_block.offset_exit.clone();
    let derivative_offset_exit = spec.time_block.derivative_offset_exit.clone();
    let constraint_derivative_offset = spec.time_block.constraint_derivative_offset.clone();
    let time_block_ref = spec.time_block.clone();

    let make_family = |logslope_design: &TermCollectionDesign| -> SurvivalMarginalSlopeFamily {
        SurvivalMarginalSlopeFamily {
            n,
            event: event.clone(),
            weights: weights.clone(),
            z: z.clone(),
            derivative_guard,
            design_entry: design_entry.clone(),
            design_exit: design_exit.clone(),
            design_derivative_exit: design_derivative_exit.clone(),
            constraint_design_derivative: constraint_design_derivative.clone(),
            offset_entry: offset_entry.clone(),
            offset_exit: offset_exit.clone(),
            derivative_offset_exit: derivative_offset_exit.clone(),
            constraint_derivative_offset: constraint_derivative_offset.clone(),
            logslope_design: logslope_design.design.to_dense(),
        }
    };

    let build_blocks = |rho: &Array1<f64>,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_time = rho
            .slice(s![cursor..cursor + time_penalties_len])
            .to_owned();
        cursor += time_penalties_len;
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        Ok(vec![
            build_time_blockspec(&time_block_ref, &design_exit, rho_time, hints.0.clone()),
            build_logslope_blockspec(
                logslope_design,
                baseline_logslope,
                rho_logslope,
                hints.1.clone(),
            ),
        ])
    };

    // ── Pilot fit: rigid (zero-penalty) to seed coefficients ────────────
    {
        let rigid_rho = Array1::<f64>::zeros(time_penalties_len + logslope_design.penalties.len());
        let rigid_blocks = build_blocks(&rigid_rho, &logslope_design)?;
        let rigid_family = make_family(&logslope_design);
        if let Ok(rigid_fit) = inner_fit(&rigid_family, &rigid_blocks, options) {
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = rigid_fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = rigid_fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
        }
    }

    // Check analytic derivatives
    let analytic_joint_derivatives_available =
        build_block_spatial_psi_derivatives(data, &logslopespec_boot, &logslope_design)
            .and_then(|maybe| {
                maybe.ok_or_else(|| "missing logslope spatial psi derivatives".to_string())
            })
            .is_ok();

    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact survival marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }

    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &logslope_design)?;
    let initial_family = make_family(&logslope_design);
    let joint_cap = custom_family_outer_capability(
        &initial_family,
        &initial_blocks,
        options,
        setup.theta0().len(),
        setup.log_kappa_dim() > 0,
    );
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_cap.gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_joint_hessian_available = analytic_joint_derivatives_available
        && matches!(
            joint_cap.hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );

    // Only logslope block has spatial terms (time block is pure B-spline)
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[logslopespec_boot.clone()],
        &[logslope_terms],
        kappa_options,
        &setup,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0])?;
            let family = make_family(&designs[0]);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let blocks = build_blocks(rho, &designs[0])?;
            let family = make_family(&designs[0]);
            // Time block has no spatial psi derivatives, so we insert an empty vec for it
            let mut derivative_blocks = vec![Vec::new()]; // time block
            derivative_blocks.push(
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?.ok_or_else(
                    || "missing survival logslope spatial psi derivatives".to_string(),
                )?,
            );
            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &blocks,
                options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian,
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            if need_hessian && eval.outer_hessian.is_none() {
                return Err(
                    "exact survival marginal-slope joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let designs = solved.designs;
    Ok(SurvivalMarginalSlopeFitResult {
        fit: solved.fit,
        logslopespec_resolved: resolved_specs.remove(0),
        logslope_design: designs.into_iter().next().unwrap(),
        baseline_logslope,
        time_block_penalties_len: time_penalties_len,
    })
}
