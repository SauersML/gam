//! #932-2 increment 3: the hand flex contracted/base θ-derivative producers, gated
//! to the test build. The production flex paths are fully jet-sourced
//! (value/grad/Hessian via `flex_jet::compute_survival_timepoint_exact_jet`, the
//! contracted base + directional/bidirectional via the `Jet2`/`Jet3`/`Jet4`
//! builders). This module pins those against the hand math (the `flex_jet`
//! `flex_timepoint_inputs_jet{3,4}_*_matches_hand_932` gates + the `tests.rs` FD
//! witnesses). `compute_survival_timepoint_first_order_exact` (grad-only) +
//! `FluxVelocity` / `moving_density_boundary_flux` stay production in `first_full`.

use super::*;
use super::first_full::moving_density_boundary_flux;
use crate::families::survival::marginal_slope::flex_oracle_structs_tests::neg_cell_of;

/// Horner evaluation of `Σ_k coefficients[k]·zᵏ`.
#[inline]
fn poly_eval_slice(coefficients: &[f64], z: f64) -> f64 {
    let mut acc = 0.0;
    for &coefficient in coefficients.iter().rev() {
        acc = acc * z + coefficient;
    }
    acc
}

/// Horner evaluation of the z-derivative `Σ_k k·coefficients[k]·z^{k-1}`.
#[inline]
fn poly_eval_deriv_slice(coefficients: &[f64], z: f64) -> f64 {
    let mut acc = 0.0;
    for (power, &coefficient) in coefficients.iter().enumerate().skip(1).rev() {
        acc = acc * z + (power as f64) * coefficient;
    }
    acc
}

/// Moving-boundary Leibniz flux for differentiation w.r.t. the **intercept**
/// `a` directly (as opposed to a θ-axis, whose `a`-sensitivity is `a_u`).
///
/// The link crossing `z = (τ − a)/b` moves with the intercept at velocity
/// `∂z/∂a = −1/b` at every `Crossing` edge (fixed partition edges do not move).
/// `f_aa`/`f_au` are second derivatives of the same moving-boundary cell
/// integral as `f_uv`, but the base cell-moment kernels only carry the interior
/// term — the `a`-axis boundary flux was omitted (gam#932/#1454), the dominant
/// residual in the intercept-solve `a_uv` Hessian. This mirrors the θ-axis
/// [`moving_density_boundary_flux`] for the `z_a = −1/b` velocity.
pub(crate) fn moving_density_boundary_flux_a(
    entry: &CachedCellEntry,
    poly: &[f64],
    b: f64,
) -> f64 {
    if b == 0.0 {
        return 0.0;
    }
    let cell = entry.partition_cell.cell;
    let edge_velocity = |edge: crate::families::cubic_cell_kernel::PartitionEdge| -> f64 {
        match edge {
            crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => -1.0 / b,
            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
        }
    };
    let v_r = edge_velocity(entry.partition_cell.right_edge);
    let v_l = edge_velocity(entry.partition_cell.left_edge);
    let right = if v_r != 0.0 {
        v_r * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
            cell, poly, cell.right,
        )
    } else {
        0.0
    };
    let left = if v_l != 0.0 {
        v_l * crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
            cell, poly, cell.left,
        )
    } else {
        0.0
    };
    right - left
}

impl SurvivalMarginalSlopeFamily {
    /// Base first derivative `d_u = ∂_θ D` of the density normalization
    /// `D = ∫ G0 dz` (`G0 = chi·w`): the interior moment integral of
    /// `∂_u G0 = chi_u − chi·η·η_u` plus the §D `[G0·z_u]_edge` total-velocity
    /// moving-boundary flux on `chi_poly`. Single-sourced so the `directional`
    /// pass recovers its intercept Hessian `a_uv` from this FD-validated
    /// quantity (D-path) instead of the inconsistent F-path `f_au` (gam#1454).
    pub(crate) fn survival_flex_base_d_u(
        &self,
        primary: &FlexPrimarySlices,
        a_u: &Array1<f64>,
        cached: &CachedPartitionCells,
        b: f64,
        p: usize,
    ) -> Result<Array1<f64>, String> {
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
                    )? + moving_density_boundary_flux(
                        u,
                        primary,
                        a_u,
                        entry,
                        &chi_poly,
                        b,
                        true,
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
        Ok(d_u)
    }

    /// Base second derivative `d_uv = ∂_θ∂_θ D` (D-path), carrying the full §D
    /// moving-domain second-derivative boundary (PART A + PART B self-flux +
    /// `G0·z_uv`). Single-sourced from the FD-validated full evaluation so the
    /// `directional` pass can form `d_u_dir = d_uv·dir` and recover a base/
    /// directional intercept Hessian that agrees with `first_full` (gam#1454).
    pub(crate) fn survival_flex_base_d_uv(
        &self,
        primary: &FlexPrimarySlices,
        a_u: &Array1<f64>,
        a_uv: &Array2<f64>,
        cached: &CachedPartitionCells,
        b: f64,
        p: usize,
    ) -> Result<Array2<f64>, String> {
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
                        // Full §D moving-domain second-derivative boundary of the
                        // density integral `D = ∫ G0 dz`, `G0 = chi·w`. PART A
                        // (boundary of the d_u interior integral) `∂_u G0·z_v`;
                        // PART B (D_v of the d_u boundary [G0·z_u]) `∂_v G0·z_u +
                        // G0_z·z_u·z_v + G0·z_uv`. The ln-density integrand is
                        // per-cell (not shared), so no term telescopes (gam#1454).
                        let boundary = if b != 0.0 {
                            let part = &entry.partition_cell;
                            let z_vel = |axis: usize,
                                         edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                         z: f64|
                             -> f64 {
                                match edge {
                                    crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                                        let direct_g = if axis == primary.g { z } else { 0.0 };
                                        -(a_u[axis] + direct_g) / b
                                    }
                                    crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                                }
                            };
                            let z_cross = |edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                           z: f64|
                             -> f64 {
                                let zu = z_vel(u, edge, z);
                                let zv = z_vel(v, edge, z);
                                let ug = if u == primary.g { 1.0 } else { 0.0 };
                                let vg = if v == primary.g { 1.0 } else { 0.0 };
                                match edge {
                                    crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                                        -(a_uv[[u, v]] + zu * vg + zv * ug) / b
                                    }
                                    crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                                }
                            };
                            let g0_z = |z: f64| -> f64 {
                                let eta = cell.eta(z);
                                let eta_z = cell.c1 + 2.0 * cell.c2 * z + 3.0 * cell.c3 * z * z;
                                let amp = poly_eval_slice(&chi_poly, z);
                                let amp_z = poly_eval_deriv_slice(&chi_poly, z);
                                let q_z = z + eta * eta_z;
                                (amp_z - amp * q_z) * (-cell.q(z)).exp()
                                    / std::f64::consts::TAU
                            };
                            let edge_term = |edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                             z: f64|
                             -> f64 {
                                let zu = z_vel(u, edge, z);
                                let zv = z_vel(v, edge, z);
                                if zu == 0.0 && zv == 0.0 {
                                    return 0.0;
                                }
                                let zuv = z_cross(edge, z);
                                let g0 = crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                    cell, &chi_poly, z,
                                );
                                let part_a = crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                    cell, &d_u_integrand_poly[u], z,
                                ) * zv;
                                let part_b1 = crate::families::cubic_cell_kernel::cell_density_boundary_integrand(
                                    cell, &d_u_integrand_poly[v], z,
                                ) * zu;
                                part_a + part_b1 + g0_z(z) * zu * zv + g0 * zuv
                            };
                            edge_term(part.right_edge, cell.right)
                                - edge_term(part.left_edge, cell.left)
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
        let mut d_uv = Array2::<f64>::zeros((p, p));
        for cell_d_uv in d_uv_cell_accums {
            for u in 0..p {
                for v in 0..p {
                    d_uv[[u, v]] += cell_d_uv[u * p + v];
                }
            }
        }
        Ok(d_uv)
    }

    pub(crate) fn compute_survival_timepoint_exact_from_cached(
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
        cached: &CachedPartitionCells,
    ) -> Result<SurvivalFlexTimepointExact, String> {
        let p = primary.total;

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
                let neg_cell = neg_cell_of(entry);
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

        let d_u = self.survival_flex_base_d_u(primary, &a_u, cached, b, p)?;

        if b != 0.0 {
            for entry in &cached.cells {
                let fixed = &entry.fixed;
                let cell = neg_cell_of(entry);
                let part = &entry.partition_cell;
                // Crossing-edge velocities z_u = −(a_u[u] + δ_{u,g}·z)/b (§C),
                // the same edge kinematics `moving_density_boundary_flux` uses
                // internally; recomputed here so the self-flux can reuse them.
                // IFT-PARTIAL θ-axis crossing velocity `∂z/∂θ_axis|_a = −direct_g/b`
                // (a held fixed) for the base intercept-Hessian partials f_uv/f_au
                // and their self-flux. The intercept-chain z-motion is carried
                // separately by the explicit f_au·a_u + f_aa·a_u² terms in the
                // a_uv recovery, so it must NOT appear in the partial boundary
                // flux/self-flux (feeding the total velocity double-counts it —
                // gam#1454). `a_edge_vel` (z_a = −1/b) is the genuine a-motion
                // and is unchanged. Mirrors directional.rs's base `edge_vel`.
                let edge_vel = |axis: usize,
                                edge: crate::families::cubic_cell_kernel::PartitionEdge,
                                z: f64|
                 -> f64 {
                    match edge {
                        crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                            let direct_g = if axis == primary.g { z } else { 0.0 };
                            -direct_g / b
                        }
                        crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                    }
                };
                let a_edge_vel =
                    |edge: crate::families::cubic_cell_kernel::PartitionEdge| -> f64 {
                        match edge {
                            crate::families::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                                -1.0 / b
                            }
                            crate::families::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
                        }
                    };
                // ∂_z of the calibration F integrand `G(z) = Φ(−η(z))·φ(z)` (NOT
                // the bare weight `w = exp(−q)/2π`). The §D self-flux `G_z·z_x·z_y`
                // uses the BASE integrand whose endpoints the Leibniz boundary
                // evaluates; for the calibration F that is `Φ(−η)φ`, so
                //   G_z = −η_z·exp(−q)/2π − z·Φ(−η)·φ(z).
                // The previous `w_z = −q_z·w` equals `G_z` only for the
                // D-normalization path; on the F-path it dropped the `−z·Φ(−η)φ`
                // term. Since the self-flux is nonzero ONLY when both crossing
                // velocities are nonzero (i.e. the (g,g)/a-axis diagonals), this
                // corrupted exactly the never-gated `f_uv[g,g]`/`f_aa`/`f_au[g]`
                // base intercept-Hessian diagonal (gam#1454). `part.cell` is the
                // POSITIVE cell (Φ(−η_pos)).
                let f_int_z = |z: f64| -> f64 {
                    let eta_pos = part.cell.eta(z);
                    let eta_z = part.cell.c1 + 2.0 * part.cell.c2 * z + 3.0 * part.cell.c3 * z * z;
                    let exp_q = (-part.cell.q(z)).exp() / std::f64::consts::TAU;
                    let phi_z = crate::probability::normal_pdf(z);
                    -eta_z * exp_q - z * crate::probability::normal_cdf(-eta_pos) * phi_z
                };
                // §D self-flux `G_z·z_x·z_y`, per the full Leibniz second
                // derivative of `∫_{zL(θ)}^{zR(θ)} G dz`: each edge carries
                // `∂_xG·z_y + ∂_yG·z_x + G_z·z_x·z_y + G·z_xy`. The first two are
                // the asymmetric `moving_density_boundary_flux` pair (added below);
                // this is the symmetric self-flux. The `G·z_xy` term carries the
                // continuous base integrand and telescope-cancels across shared
                // interior crossings (gam#932), so it is dropped.
                let self_flux_xy = |zx_r: f64, zy_r: f64, zx_l: f64, zy_l: f64| -> f64 {
                    let right = if zx_r != 0.0 && zy_r != 0.0 {
                        zx_r * zy_r * f_int_z(cell.right)
                    } else {
                        0.0
                    };
                    let left = if zx_l != 0.0 && zy_l != 0.0 {
                        zx_l * zy_l * f_int_z(cell.left)
                    } else {
                        0.0
                    };
                    right - left
                };
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    let zu_r = edge_vel(u, part.right_edge, cell.right);
                    let zu_l = edge_vel(u, part.left_edge, cell.left);
                    for v in u..p {
                        // Asymmetric Leibniz flux pair `∂_uG·z_v + ∂_vG·z_u`; on the
                        // DIAGONAL (u==v) both orderings coincide → `2·flux` (the
                        // previous single-add halved the g-axis `f_uv[g,g]`,
                        // gam#1454). Off-diagonal [g,w0] is unchanged: its second
                        // term `flux(w0,·)` is 0 since z_w0=0.
                        let neg_coeff_v = fixed.coeff_u[v].map(|value| -value);
                        let boundary = moving_density_boundary_flux(
                            v,
                            primary,
                            &a_u,
                            entry,
                            &neg_coeff_u,
                            b,
                            false,
                        ) + moving_density_boundary_flux(
                            u,
                            primary,
                            &a_u,
                            entry,
                            &neg_coeff_v,
                            b,
                            false,
                        );
                        // `G_z·z_u·z_v` is a single symmetric term, added once
                        // (unlike the asymmetric `flux` pair) to both triangles.
                        let zv_r = edge_vel(v, part.right_edge, cell.right);
                        let zv_l = edge_vel(v, part.left_edge, cell.left);
                        let boundary = boundary + self_flux_xy(zu_r, zv_r, zu_l, zv_l);
                        f_uv[[u, v]] += boundary;
                        if u != v {
                            f_uv[[v, u]] += boundary;
                        }
                    }
                }

                // a-axis moving-boundary flux for the intercept second
                // derivatives. f_aa and f_au are second derivatives of the same
                // moving-boundary cell integral as f_uv (the crossing
                // z = (τ−a)/b moves with a at velocity z_a = −1/b), so they carry
                // the full §D boundary structure: the asymmetric flux pair AND the
                // symmetric `G_z·z_a·z_x` self-flux. Mirror the f_uv pair + self-
                // flux with one axis = a (gam#932/#1454).
                let neg_dc_da = fixed.dc_da.map(|value| -value);
                let za_r = a_edge_vel(part.right_edge);
                let za_l = a_edge_vel(part.left_edge);
                // Diagonal (a,a): the asymmetric flux pair `∂_aG·z_a + ∂_aG·z_a`
                // coincides → DOUBLED (the f_uv[g,g] diagonal fix on the a-axis),
                // gam#1454.
                f_aa += 2.0 * moving_density_boundary_flux_a(entry, &neg_dc_da, b)
                    + self_flux_xy(za_r, za_r, za_l, za_l);
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    let zu_r = edge_vel(u, part.right_edge, cell.right);
                    let zu_l = edge_vel(u, part.left_edge, cell.left);
                    f_au[u] += moving_density_boundary_flux_a(entry, &neg_coeff_u, b)
                        + moving_density_boundary_flux(
                            u,
                            primary,
                            &a_u,
                            entry,
                            &neg_dc_da,
                            b,
                            false,
                        )
                        + self_flux_xy(za_r, zu_r, za_l, zu_l);
                }
            }
        }

        // #932 Item 1 (doc §B): lift the calibration intercept Hessian `a_uv`
        // through `filtered_implicit_solve_scalar` over the calibration
        // constraint `F(a, θ) = 0` (channels = the partials d_check/f_u/f_uv/
        // f_aa/d_u assembled above), single-sourcing the hand IFT closed form.
        let a_uv = self.lift_flex_intercept_hessian(p, d_check, &f_u, &f_uv, f_aa, &d_u, a)?;
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

        let d_uv = if need_d_uv {
            self.survival_flex_base_d_uv(primary, &a_u, &a_uv, cached, b, p)?
        } else {
            Array2::<f64>::zeros((p, p))
        };

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
