//! Exact first-order timepoint evaluation (production) + the shared moving-boundary
//! flux helpers.
//!
//! `compute_survival_timepoint_first_order_exact` carries the calibration solve and
//! the first partials — the production grad-only flex path (`flex_sensitivity`). The
//! full second-order hand pack (`compute_survival_timepoint_exact_from_cached` and its
//! D-path builders) is now the test-only oracle (`first_full_exact_oracle_tests`):
//! the production value/grad/Hessian + contracted base/extensions are jet-sourced
//! (`flex_jet`). `FluxVelocity` / `moving_density_boundary_flux` are shared by both
//! and stay here.

use super::*;

pub(super) fn moving_density_boundary_flux(
    axis: usize,
    primary: &FlexPrimarySlices,
    a_u: &Array1<f64>,
    entry: &CachedCellEntry,
    poly: &[f64],
    b: f64,
    include_intercept: bool,
) -> f64 {
    // The link crossing `z = (τ − a)/b` moves with `θ` both directly (through
    // `b = g`) and through the intercept response `a(θ)` at velocity `a_u`. Two
    // distinct quantities need this flux with two distinct velocities:
    //
    // * `include_intercept = true` — `∂z/∂θ_axis = −(a_u[axis] + direct_g)/b`,
    //   the genuine TOTAL z-motion, used by the first-derivative `d_u` (which
    //   also carries the `chi·a_u` intercept chain in its interior, so it is a
    //   consistent total derivative and is FD-verified correct).
    // * `include_intercept = false` — `∂z/∂θ_axis|_a = −direct_g/b`, the PARTIAL
    //   z-motion at fixed intercept, used by the IFT partials `f_uv`/`f_au`
    //   (whose interiors use partial cell coefficients, a held fixed). The
    //   intercept-chain contribution to `a_uv` is supplied separately by the
    //   explicit `f_au·a_u + f_aa·a_u²` terms in the IFT recovery; feeding the
    //   total velocity here double-counts the intercept motion (gam#1454).
    if b == 0.0 {
        return 0.0;
    }
    let cell = entry.partition_cell.cell;
    let edge_velocity = |edge: crate::cubic_cell_kernel::PartitionEdge, z: f64| -> f64 {
        match edge {
            crate::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                let direct_g = if axis == primary.g { z } else { 0.0 };
                let intercept = if include_intercept { a_u[axis] } else { 0.0 };
                -(intercept + direct_g) / b
            }
            crate::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
        }
    };
    let v_r = edge_velocity(entry.partition_cell.right_edge, cell.right);
    let v_l = edge_velocity(entry.partition_cell.left_edge, cell.left);
    let right = if v_r != 0.0 {
        v_r * crate::cubic_cell_kernel::cell_density_boundary_integrand(
            cell, poly, cell.right,
        )
    } else {
        0.0
    };
    let left = if v_l != 0.0 {
        v_l * crate::cubic_cell_kernel::cell_density_boundary_integrand(
            cell, poly, cell.left,
        )
    } else {
        0.0
    };
    right - left
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn compute_survival_timepoint_first_order_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        d_calibration: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
    ) -> Result<SurvivalFlexTimepointFirstOrderExact, String> {
        let p = primary.total;
        let cached =
            self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 9)?;

        struct FirstOrderCellAccum {
            f_u: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<FirstOrderCellAccum, String> {
                let state = &entry.state;
                let fixed = &entry.fixed;
                let mut f_u = vec![0.0; p];
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                        &neg_coeff_u,
                        &state.moments,
                    )?;
                }
                Ok(FirstOrderCellAccum { f_u })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_u = Array1::<f64>::zeros(p);
        for acc in cell_accums {
            for u in 0..p {
                f_u[u] += acc.f_u[u];
            }
        }
        f_u[q_index] += crate::probability::normal_pdf(q);

        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        if !d_check.is_finite() || d_check <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D={d_check:.3e}"
                ),
            }
            .into());
        }
        let d_rel_err = (d_check - d_calibration).abs() / d_check.max(d_calibration.abs()).max(1.0);
        if !d_calibration.is_finite() || d_calibration <= 0.0 || d_rel_err > 1e-8 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced inconsistent calibration derivative: solve={d_calibration:.12e}, direct={d_check:.12e}"
                ),
            }
            .into());
        }

        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = f_u[u] / d_check;
        }

        let d_u_cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<Vec<f64>, String> {
                let cell = entry.partition_cell.cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();
                let mut d_u = vec![0.0; p];
                for u in 0..p {
                    let eta_u_poly = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                    let chi_u_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
                    let integrand = poly_sub(
                        &chi_u_poly,
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly),
                    );
                    d_u[u] = exact_kernel::cell_polynomial_integral_from_moments(
                        &integrand,
                        &state.moments,
                        "survival D_t first derivative",
                    )? + moving_density_boundary_flux(
                        u, primary, &a_u, entry, &chi_poly, b, true,
                    );
                }
                Ok(d_u)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u = Array1::<f64>::zeros(p);
        for cell_d_u in d_u_cell_accums {
            for u in 0..p {
                d_u[u] += cell_d_u[u];
            }
        }

        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        // Absorbed-influence offset: observed-η shift only (see
        // `compute_survival_timepoint_exact`). `eta += o_infl`, and the trailing
        // `infl` channel carries the direct partial `∂η₁/∂o_infl = 1` via
        // `rho[infl]`; calibration-side `a_u`/`chi`/`chi_u`/`d_u` are untouched.
        let eta = eval_coeff4_at(&obs.coeff, z_obs) + o_infl;
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);

        let mut rho = Array1::<f64>::zeros(p);
        let mut tau = Array1::<f64>::zeros(p);
        let scale = self.probit_frailty_scale();
        rho[primary.g] = eval_coeff4_at(&obs.dc_db, z_obs);
        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        if let Some(infl) = primary.infl {
            rho[infl] = 1.0;
        }

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                        scale,
                    ),
                    z_obs,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                        scale,
                    ),
                    z_obs,
                );
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
            }
        }

        let mut eta_u = Array1::<f64>::zeros(p);
        let mut chi_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            eta_u[u] = chi * a_u[u] + rho[u];
            chi_u[u] = eta_aa * a_u[u] + tau[u];
        }

        Ok(SurvivalFlexTimepointFirstOrderExact {
            eta,
            chi,
            d: d_check,
            eta_u,
            chi_u,
            d_u,
        })
    }
}
