//! The latent-survival Jeffreys information derivatives in a Jeffreys drift basis (#2992).
//!
//! The outer Jeffreys drift reads the first, second and third information derivatives along
//! every coefficient axis only as rows `vec(sym(Uᵀ D[e_a] U))` in its drift basis `U` (`p × r`).
//! A data row contributes `X_iᵀ F_i X_i` to each derivative through at most six primaries, and
//! its kernel is linear in the axis seed, `F_i = Σ_γ (X_i e_a)_γ F_{i,γ}`. So with the projected
//! primaries `Y_i = X_i U`, every axis row closes in one row pass,
//!
//! ```text
//!   vec(Uᵀ D[e_a] U) = Σ_i Σ_γ (X_i)_{γa} vec(Y_iᵀ F_{i,γ} Y_i),
//! ```
//!
//! from the primary lifts `F_{i,γ}` the dense all-axes builders already form. No `p × p` axis
//! matrix is built, and no axis takes its own row pass. Without these hooks the trait default
//! ran one full row pass per axis for the second derivative, and the Jeffreys drift spent about
//! half of `tests_rho_domain_2902`'s fit there.

use super::*;
use ndarray::ArrayView2;

/// The kernel the dense pullback reads: its diagonal and upper triangle, mirrored.
///
/// `LatentSurvivalFamily::add_pullback_primary_hessian` reads `F[a, b]` only for `a ≤ b`, so
/// the congruence `Y_iᵀ F Y_i` is formed from the same entries.
fn upper_mirrored(kernel: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(kernel.dim(), |(a, b)| {
        if a <= b {
            kernel[[a, b]]
        } else {
            kernel[[b, a]]
        }
    })
}

/// Average each `r × r` block of `rows` with its transpose: `vec(sym(·))` of every axis row.
fn symmetrized_axis_rows(rows: &mut Array2<f64>, r: usize) {
    for mut row in rows.rows_mut() {
        for s in 0..r {
            for t in (s + 1)..r {
                let average = 0.5 * (row[s * r + t] + row[t * r + s]);
                row[s * r + t] = average;
                row[t * r + s] = average;
            }
        }
    }
}

impl LatentSurvivalFamily {
    /// The rows `X_i[γ, ·]` of a data row's primary Jacobian, each over the joint coefficients:
    /// the four time design rows on the time slice, the mean design row on the mean slice, and
    /// the log-σ unit where σ is estimated. `row_primary_direction_from_flat` is this matrix
    /// applied to a direction.
    pub(super) fn row_primary_jacobian(
        &self,
        row_idx: usize,
        slices: &LatentSurvivalJointSlices,
    ) -> Result<Array2<f64>, String> {
        let mut jacobian = Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, slices.total));
        for (primary, design) in [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, &self.x_time_entry),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, &self.x_time_exit),
            (LATENT_SURVIVAL_PRIMARY_QDOT_EXIT, &self.x_time_derivative_exit),
            (LATENT_SURVIVAL_PRIMARY_Q_RIGHT, &self.x_time_right),
        ] {
            jacobian
                .slice_mut(s![primary, slices.time.clone()])
                .assign(&design.row(row_idx));
        }
        let mean_row = self.x_mean.try_row_chunk(row_idx..row_idx + 1).map_err(|error| {
            format!(
                "latent survival rotated Jeffreys rows: mean design row {row_idx} is unavailable: {error}"
            )
        })?;
        jacobian
            .slice_mut(s![LATENT_SURVIVAL_PRIMARY_MU, slices.mean.clone()])
            .assign(&mean_row.row(0));
        if let Some(range) = &slices.log_sigma {
            jacobian[[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, range.start]] = 1.0;
        }
        Ok(jacobian)
    }

    /// Rows `vec(sym(Uᵀ D_k[e_a] U))` of `count` joint-Hessian derivatives `D_k`, each with a row
    /// kernel linear in one primary seed. `lift(k, ·)` returns `D_k`'s kernel `F_{i,γ}` along the
    /// primary `e_γ`; `channel` names it in refusals.
    ///
    /// Each row is built once for all `count` derivatives. A primary its design never touches
    /// contributes to no axis and is not lifted, as in the dense builders.
    fn rotated_axis_rows_from_primary_lifts(
        &self,
        block_states: &[ParameterBlockState],
        basis: ArrayView2<'_, f64>,
        count: usize,
        channel: &str,
        lift: impl Fn(usize, LatentSurvivalPrimaryLift<'_>) -> Result<Array2<f64>, String> + Sync,
    ) -> Result<Vec<Array2<f64>>, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        let (basis_rows, r) = basis.dim();
        if basis_rows != total {
            return Err(format!(
                "latent survival rotated {channel} rows: a Jeffreys basis of {basis_rows} rows for \
                 {total} joint coefficients"
            ));
        }
        let live_primaries = if slices.log_sigma.is_some() {
            LATENT_SURVIVAL_PRIMARY_DIM
        } else {
            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
        };
        let seeds: Vec<Array1<f64>> = (0..LATENT_SURVIVAL_PRIMARY_DIM)
            .map(|gamma| {
                let mut seed = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
                seed[gamma] = 1.0;
                seed
            })
            .collect();
        let mut rows = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || vec![Array2::<f64>::zeros((total, r * r)); count],
            |row_idx, acc| {
                let wi = weights.at(row_idx);
                if wi == 0.0 {
                    return Ok(());
                }
                let jacobian = self.row_primary_jacobian(row_idx, &slices)?;
                let projected = jacobian.dot(&basis);
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let point = LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: qdot_exit[row_idx],
                    q_right: q_right[row_idx],
                    mu: mu[row_idx],
                    sigma,
                };
                for gamma in 0..live_primaries {
                    let loading = jacobian.row(gamma);
                    if loading.iter().all(|&value| value == 0.0) {
                        continue;
                    }
                    for (index, target) in acc.iter_mut().enumerate() {
                        let kernel = lift(
                            index,
                            LatentSurvivalPrimaryLift {
                                row_idx,
                                row: &row,
                                point,
                                primary: &seeds[gamma],
                            },
                        )?;
                        let weighted = checked_weighted_row_matrix(wi, &kernel, row_idx, channel)?;
                        let reduced = projected.t().dot(&upper_mirrored(&weighted)).dot(&projected);
                        for (axis, &value) in loading.iter().enumerate() {
                            if value == 0.0 {
                                continue;
                            }
                            let mut axis_row = target.row_mut(axis);
                            for s in 0..r {
                                for t in 0..r {
                                    axis_row[s * r + t] += value * reduced[[s, t]];
                                }
                            }
                        }
                    }
                }
                Ok(())
            },
            |total_acc, chunk_acc| {
                for (total_rows, chunk_rows) in total_acc.iter_mut().zip(chunk_acc) {
                    *total_rows += &chunk_rows;
                }
            },
        )?;
        let quantity = format!("{channel} rotated Hessian axis rows");
        for axis_rows in &mut rows {
            symmetrized_axis_rows(axis_rows, r);
            require_finite_likelihood_matrix(axis_rows, &quantity)?;
        }
        Ok(rows)
    }

    /// `{vec(sym(Uᵀ Hdot[e_a] U))}` from one row pass.
    pub(super) fn first_directional_rotated_axis_rows(
        &self,
        block_states: &[ParameterBlockState],
        basis: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let include_log_sigma = self.joint_slices().log_sigma.is_some();
        self.rotated_axis_rows_from_primary_lifts(block_states, basis, 1, "contracted third", |_, lift| {
            latent_survival_row_primary_third_contracted(
                &self.quadctx,
                lift.row,
                lift.point,
                lift.primary,
                include_log_sigma,
            )
            .map_err(String::from)
        })?
        .pop()
        .ok_or_else(|| "latent survival rotated dH: the row pass returned no rows".to_string())
    }

    /// `{vec(sym(Uᵀ H²[δ_k, e_a] U))}` for every direction `δ_k` of a batch, from one row pass
    /// with each row built once.
    pub(super) fn second_directional_rotated_axis_rows_each(
        &self,
        block_states: &[ParameterBlockState],
        directions: &[Array1<f64>],
        basis: ArrayView2<'_, f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        if let Some(direction) = directions.iter().find(|direction| direction.len() != slices.total) {
            return Err(format!(
                "latent survival rotated d2H direction length mismatch: got {}, expected {}",
                direction.len(),
                slices.total
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        self.rotated_axis_rows_from_primary_lifts(
            block_states,
            basis,
            directions.len(),
            "contracted fourth",
            |index, lift| {
                let direction =
                    self.row_primary_direction_from_flat(lift.row_idx, &slices, &directions[index]);
                latent_survival_row_primary_fourth_contracted(
                    &self.quadctx,
                    lift.row,
                    lift.point,
                    &direction,
                    lift.primary,
                    include_log_sigma,
                )
                .map_err(String::from)
            },
        )
    }

    /// `{vec(sym(Uᵀ D³H[u, v, e_a] U))}` from one row pass.
    pub(super) fn third_directional_rotated_axis_rows(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        basis: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = self.joint_slices();
        if d_beta_u_flat.len() != slices.total || d_beta_v_flat.len() != slices.total {
            return Err(format!(
                "latent survival rotated d3H direction length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                slices.total
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        self.rotated_axis_rows_from_primary_lifts(block_states, basis, 1, "contracted fifth", |_, lift| {
            let direction_u = self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_u_flat);
            let direction_v = self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_v_flat);
            latent_survival_row_primary_fifth_contracted(
                &self.quadctx,
                lift.row,
                lift.point,
                &direction_u,
                &direction_v,
                lift.primary,
                include_log_sigma,
            )
            .map_err(String::from)
        })?
        .pop()
        .ok_or_else(|| "latent survival rotated d3H: the row pass returned no rows".to_string())
    }
}

impl crate::custom_family::JeffreysRotatedFirstDerivative for LatentSurvivalFamily {
    /// The rows `vec(sym(Uᵀ Hdot[e_a] U))` from each row's third-order primary lifts, so the `p`
    /// dense axis matrices are never formed (#2992).
    fn first_directional_rotated_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        basis: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if specs.len() != block_states.len() {
            return Err(format!(
                "first_directional_rotated_all_axes: {} parameter-block specs for {} block states",
                specs.len(),
                block_states.len()
            ));
        }
        self.first_directional_rotated_axis_rows(block_states, basis)
    }
}
