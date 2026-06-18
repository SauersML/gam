//! Exact first-order and full timepoint evaluations.
//!
//! `compute_survival_timepoint_first_order_exact` carries the calibration
//! solve and first partials; `compute_survival_timepoint_exact` extends it to
//! the full second-order timepoint quantities (η, χ, D and their u/uv partials)
//! over the cached partition.

use super::*;

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
                    )?;
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

    pub(crate) fn compute_survival_timepoint_exact(
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
        need_d_uv: bool,
    ) -> Result<SurvivalFlexTimepointExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct ExactTimepointCellAccum {
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<ExactTimepointCellAccum, String> {
                let neg_cell = entry.neg_cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let neg_dc_da = fixed.dc_da.map(|value| -value);
                let neg_dc_daa = fixed.dc_daa.map(|value| -value);
                let mut f_u = vec![0.0; p];
                let mut f_au = vec![0.0; p];
                let mut f_uv = vec![0.0; p * p];
                let f_aa = exact_kernel::cell_second_derivative_from_moments(
                    neg_cell,
                    &neg_dc_da,
                    &neg_dc_da,
                    &neg_dc_daa,
                    &state.moments,
                )?;
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    let neg_coeff_au = fixed.coeff_au[u].map(|value| -value);
                    f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                        &neg_coeff_u,
                        &state.moments,
                    )?;
                    f_au[u] = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_u,
                        &neg_coeff_au,
                        &state.moments,
                    )?;
                }
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    for v in u..p {
                        let second_coeff = if u == primary.g {
                            fixed.coeff_bu[v]
                        } else if v == primary.g {
                            fixed.coeff_bu[u]
                        } else {
                            [0.0; 4]
                        };
                        let neg_coeff_v = fixed.coeff_u[v].map(|value| -value);
                        let neg_second_coeff = second_coeff.map(|value| -value);
                        let value = exact_kernel::cell_second_derivative_from_moments(
                            neg_cell,
                            &neg_coeff_u,
                            &neg_coeff_v,
                            &neg_second_coeff,
                            &state.moments,
                        )?;
                        f_uv[u * p + v] = value;
                        f_uv[v * p + u] = value;
                    }
                }
                Ok(ExactTimepointCellAccum {
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_aa = 0.0;
        for acc in cell_accums {
            f_aa += acc.f_aa;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;

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
                let mut d_u = vec![0.0; p];
                for u in 0..p {
                    let eta_u_poly = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                    let chi_u_poly =
                        poly_add(&poly_scale(&fixed.dc_daa, a_u[u]), &fixed.coeff_au[u]);
                    let integrand = poly_sub(
                        &chi_u_poly,
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly),
                    );
                    d_u[u] = exact_kernel::cell_polynomial_integral_from_moments(
                        &integrand,
                        &state.moments,
                        "survival D_t first derivative",
                    )?;
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

        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let value =
                    (f_uv[[u, v]] - d_u[u] * a_u[v] - d_u[v] * a_u[u] - f_aa * a_u[u] * a_u[v])
                        / d_check;
                a_uv[[u, v]] = value;
                a_uv[[v, u]] = value;
            }
        }

        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        // The absorbed-influence offset shifts the OBSERVED index η₁ additively
        // (#461). It is independent of the calibration intercept `a`, so it
        // touches only `eta` itself and the trailing `infl` primary partial
        // `∂η₁/∂o_infl = 1` (set via `rho[infl]` below); every calibration-side
        // quantity (`a_u`, `chi`, `chi_u`, `d_u`, all second partials) is
        // untouched because `o_infl` never enters the de-nested cells.
        let eta = eval_coeff4_at(&obs.coeff, z_obs) + o_infl;
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);

        let mut rho = Array1::<f64>::zeros(p);
        let mut tau = Array1::<f64>::zeros(p);
        let mut tau_a = Array1::<f64>::zeros(p);
        let scale = self.probit_frailty_scale();
        rho[primary.g] = eval_coeff4_at(&obs.dc_db, z_obs);
        // Direct observed partial of the absorber channel: `∂η₁/∂o_infl = 1`
        // (and `a_u[infl] = 0`), so `eta_u[infl] = chi·0 + rho[infl] = 1`. All
        // other `infl` entries (tau, tau_a, second partials) stay zero.
        if let Some(infl) = primary.infl {
            rho[infl] = 1.0;
        }
        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);

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
                let (dc_aaw, _, _) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        let mut eta_u = Array1::<f64>::zeros(p);
        let mut chi_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            eta_u[u] = chi * a_u[u] + rho[u];
            chi_u[u] = eta_aa * a_u[u] + tau[u];
        }

        let mut eta_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv = self.observed_fixed_eta_second_partial(
                    primary, &obs, row, u, v, z_obs, u_obs, a, b,
                )?;
                let chi_uv_fixed = self
                    .observed_fixed_chi_second_partial(primary, &obs, u, v, z_obs, u_obs, a, b)?;

                let eta_val = chi * a_uv[[u, v]]
                    + eta_aa * a_u[u] * a_u[v]
                    + tau[u] * a_u[v]
                    + tau[v] * a_u[u]
                    + r_uv;
                eta_uv[[u, v]] = eta_val;
                eta_uv[[v, u]] = eta_val;

                let chi_val = eta_aa * a_uv[[u, v]]
                    + eta_aaa * a_u[u] * a_u[v]
                    + tau_a[u] * a_u[v]
                    + tau_a[v] * a_u[u]
                    + chi_uv_fixed;
                chi_uv[[u, v]] = chi_val;
                chi_uv[[v, u]] = chi_val;
            }
        }

        let mut d_uv = Array2::<f64>::zeros((p, p));
        if need_d_uv {
            let d_uv_cell_accums = cached
                .cells
                .iter()
                .map(|entry| -> Result<Vec<f64>, String> {
                    let cell = entry.partition_cell.cell;
                    let state = &entry.state;
                    let fixed = &entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    let mut d_u_integrand_poly = vec![PolyVec::new(); p];
                    let mut d_uv = vec![0.0; p * p];
                    for u in 0..p {
                        eta_u_poly[u] = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                        chi_u_poly[u] =
                            poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
                        d_u_integrand_poly[u] = poly_sub(
                            &chi_u_poly[u],
                            &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly[u]),
                        );
                    }
                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };
                            let chi_uv_fixed = if u == primary.g {
                                fixed.coeff_abu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_abu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };
                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_au[u], a_u[v]),
                                ),
                                &poly_add(&poly_scale(&fixed.coeff_au[v], a_u[u]), &r_uv_fixed),
                            );
                            let chi_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_aau[u], a_u[v]),
                                ),
                                &poly_add(&poly_scale(&fixed.coeff_aau[v], a_u[u]), &chi_uv_fixed),
                            );
                            let term2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let term3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let term4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let term5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let integrand = poly_add(
                                &poly_add(&poly_add(&chi_uv_poly, &term2), &term3),
                                &poly_add(&term4, &term5),
                            );
                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &integrand,
                                &state.moments,
                                "survival D_t second derivative",
                            )?;
                            let boundary = if b != 0.0 {
                                let part = &entry.partition_cell;
                                let edge_vel =
                                    |axis: usize,
                                     edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                     z: f64|
                                     -> f64 {
                                        match edge {
                                            crate::families::cubic_cell_kernel::PartitionEdge::Crossing {
                                                ..
                                            } => {
                                                let direct_g =
                                                    if axis == primary.g { z } else { 0.0 };
                                                -(a_u[axis] + direct_g) / b
                                            }
                                            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(
                                                _,
                                            ) => 0.0,
                                        }
                                    };
                                let flux = |axis: usize, poly: &[f64]| -> f64 {
                                    let v_r = edge_vel(axis, part.right_edge, cell.right);
                                    let v_l = edge_vel(axis, part.left_edge, cell.left);
                                    let right = if v_r != 0.0 {
                                        v_r * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                            cell,
                                            poly,
                                            cell.right,
                                        )
                                    } else {
                                        0.0
                                    };
                                    let left = if v_l != 0.0 {
                                        v_l * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                            cell,
                                            poly,
                                            cell.left,
                                        )
                                    } else {
                                        0.0
                                    };
                                    right - left
                                };
                                0.5 * (flux(v, &d_u_integrand_poly[u])
                                    + flux(u, &d_u_integrand_poly[v]))
                            } else {
                                0.0
                            };
                            d_uv[u * p + v] = value + boundary;
                            d_uv[v * p + u] = value + boundary;
                        }
                    }
                    Ok(d_uv)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for cell_d_uv in d_uv_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv[[u, v]] += cell_d_uv[u * p + v];
                    }
                }
            }
        }

        Ok(SurvivalFlexTimepointExact {
            eta,
            chi,
            d: d_check,
            eta_u,
            eta_uv,
            chi_u,
            chi_uv,
            d_u,
            d_uv,
        })
    }
}
