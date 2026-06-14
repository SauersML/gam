//! Exact piecewise-cubic hazard timepoint integrals: building the cached
//! partition with the required moment order and the exact first-order /
//! full / directional / bidirectional timepoint evaluations, plus the flex
//! primary third/fourth contracted exact tensors built on them.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Build a cached partition: cells + moment states + fixed partials,
    /// computed once per (a, b, β_h, β_w) and reused across the three
    /// integration passes (F, D, D_uv).
    ///
    /// The cell-table assembly and the per-cell primary-fixed-partials
    /// assembly route through the GPU-shaped `try_device_*` seams in
    /// [`crate::families::survival_marginal_slope_gpu_prep`].  Until the matching NVRTC kernels
    /// land, both seams return `Ok(None)` and the call site falls back to
    /// the existing CPU implementation, so behavior is preserved.
    pub(crate) fn build_cached_partition_with_moment_order(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        moment_order: usize,
    ) -> Result<CachedPartitionCells, String> {
        // ── 1. partition cells via the device seam, CPU fallback on decline ──
        let raw_cells = {
            let row_input =
                crate::families::survival_marginal_slope_gpu_prep::PartitionCellsRowInputs {
                    a,
                    b,
                    beta_h: beta_h.and_then(|b| b.as_slice()),
                    beta_w: beta_w.and_then(|b| b.as_slice()),
                };
            let dev =
                crate::families::survival_marginal_slope_gpu_prep::try_device_partition_cells(
                    std::slice::from_ref(&row_input),
                )
                .map_err(|e| e.to_string())?;
            match dev {
                Some(mut by_row) if by_row.len() == 1 => by_row.remove(0),
                _ => self.denested_partition_cells(a, b, beta_h, beta_w)?,
            }
        };

        // ── 2. per-cell prelude (neg_cell, z_mid, u_mid, moment state) ──
        let n = raw_cells.len();
        let mut neg_cells = Vec::with_capacity(n);
        let mut z_mids = Vec::with_capacity(n);
        let mut u_mids = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        let mut fp_inputs = Vec::<
            crate::families::survival_marginal_slope_gpu_prep::CellPrimaryFixedPartialsCellInputs,
        >::with_capacity(n);
        for partition_cell in &raw_cells {
            let cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: cell.left,
                right: cell.right,
                c0: -cell.c0,
                c1: -cell.c1,
                c2: -cell.c2,
                c3: -cell.c3,
            };
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact_kernel::evaluate_cell_moments(cell, moment_order)?;
            neg_cells.push(neg_cell);
            z_mids.push(z_mid);
            u_mids.push(u_mid);
            states.push(state);
            fp_inputs.push(
                crate::families::survival_marginal_slope_gpu_prep::CellPrimaryFixedPartialsCellInputs {
                    score_span: partition_cell.score_span,
                    link_span: partition_cell.link_span,
                    z_basis: z_mid,
                    u_basis: u_mid,
                },
            );
        }

        // ── 3. per-cell fixed partials via the device seam, CPU fallback ──
        let layout = crate::families::survival_marginal_slope_gpu_prep::FlexPrimaryLayout {
            r: u32::try_from(primary.total).map_err(|_| {
                format!(
                    "build_cached_partition_with_moment_order: primary.total={} exceeds u32",
                    primary.total
                )
            })?,
            g_slot: u32::try_from(primary.g).map_err(|_| {
                format!(
                    "build_cached_partition_with_moment_order: primary.g={} exceeds u32",
                    primary.g
                )
            })?,
        };
        let row_fp_input =
            crate::families::survival_marginal_slope_gpu_prep::CellPrimaryFixedPartialsRowInputs {
                a,
                b,
                cells: &fp_inputs,
                layout,
            };
        let dev_fixed = crate::families::survival_marginal_slope_gpu_prep::try_device_cell_primary_fixed_partials(
            std::slice::from_ref(&row_fp_input),
        )
        .map_err(|e| e.to_string())?;
        // When the device path returns flat-packed partials, reconstruct
        // the per-cell `DenestedCellPrimaryFixedPartials` from the device
        // buffer via the `from_flat_slice` shim — byte-identical to what
        // the CPU per-cell helper would produce for the supported
        // (trivial-span) shape.  Any decline drops through to the CPU
        // per-cell loop below.
        if let Some(out) = dev_fixed.as_ref()
            && out.partials.len() == 1
            && out.partials[0].len() == n
        {
            let mut cells = Vec::with_capacity(n);
            for (idx, partition_cell) in raw_cells.into_iter().enumerate() {
                let flat = &out.partials[0][idx];
                let fixed = DenestedCellPrimaryFixedPartials::from_flat_slice(
                    flat.as_slice(),
                    primary.total,
                )?;
                cells.push(CachedCellEntry {
                    partition_cell,
                    neg_cell: neg_cells[idx],
                    state: states[idx].clone(),
                    fixed,
                });
            }
            return Ok(CachedPartitionCells { cells });
        }
        let mut cells = Vec::with_capacity(n);
        for (idx, partition_cell) in raw_cells.into_iter().enumerate() {
            let fixed = self.denested_cell_primary_fixed_partials(
                primary,
                a,
                b,
                partition_cell.score_span,
                partition_cell.link_span,
                z_mids[idx],
                u_mids[idx],
            )?;
            cells.push(CachedCellEntry {
                partition_cell,
                neg_cell: neg_cells[idx],
                state: states[idx].clone(),
                fixed,
            });
        }
        Ok(CachedPartitionCells { cells })
    }

    pub(crate) fn build_cached_partition(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<CachedPartitionCells, String> {
        self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 24)
    }

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
                    let mut d_uv = vec![0.0; p * p];
                    for u in 0..p {
                        eta_u_poly[u] = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                        chi_u_poly[u] =
                            poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
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
                            d_uv[u * p + v] = value;
                            d_uv[v * p + u] = value;
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

impl SurvivalMarginalSlopeFamily {
    /// Compute directional extensions of a timepoint's exact quantities.
    /// Given the base `SurvivalFlexTimepointExact`, returns the directional
    /// derivatives eta_uv_dir, chi_uv_dir, d_u_dir, d_uv_dir contracted
    /// with `dir`.
    pub(crate) fn compute_survival_timepoint_directional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir: &Array1<f64>,
        need_d_uv_dir: bool,
    ) -> Result<SurvivalFlexTimepointDirectionalExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct DirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
            f_a_dir: f64,
            f_aa_dir: f64,
            f_au_dir: Vec<f64>,
            f_uv_dir: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(
                |cell_entry| -> Result<DirectionalTimepointCellAccum, String> {
                    let neg_cell = cell_entry.neg_cell;
                    let state = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let neg_dc_da: [f64; 4] = fixed.dc_da.map(|v| -v);
                    let neg_dc_daa: [f64; 4] = fixed.dc_daa.map(|v| -v);

                    let f_a = exact_kernel::cell_first_derivative_from_moments(
                        &neg_dc_da,
                        &state.moments,
                    )?;
                    let f_aa = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_dc_daa,
                        &state.moments,
                    )?;

                    let mut neg_coeff_dir = [0.0; 4];
                    let mut neg_coeff_a_dir = [0.0; 4];
                    let mut neg_coeff_aa_dir = [0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            neg_coeff_dir[k] -= fixed.coeff_u[c][k] * dir[c];
                            neg_coeff_a_dir[k] -= fixed.coeff_au[c][k] * dir[c];
                            neg_coeff_aa_dir[k] -= fixed.coeff_aau[c][k] * dir[c];
                        }
                    }

                    let f_a_dir = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_coeff_a_dir,
                        &state.moments,
                    )?;
                    let f_aa_dir = exact_kernel::cell_third_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_dc_daa,
                        &neg_coeff_a_dir,
                        &neg_coeff_a_dir,
                        &neg_coeff_aa_dir,
                        &state.moments,
                    )?;

                    let mut f_u = vec![0.0; p];
                    let mut f_au = vec![0.0; p];
                    let mut f_uv = vec![0.0; p * p];
                    let mut f_au_dir = vec![0.0; p];
                    let mut f_uv_dir = vec![0.0; p * p];
                    for u in 0..p {
                        let neg_coeff_u = fixed.coeff_u[u].map(|v| -v);
                        let neg_coeff_au = fixed.coeff_au[u].map(|v| -v);

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

                        let mut neg_coeff_u_dir = [0.0; 4];
                        let mut neg_coeff_au_dir = [0.0; 4];
                        for c in 0..p {
                            if dir[c] == 0.0 {
                                continue;
                            }
                            let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                            let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                            for k in 0..4 {
                                neg_coeff_u_dir[k] -= sc[k] * dir[c];
                                neg_coeff_au_dir[k] -= sca[k] * dir[c];
                            }
                        }

                        f_au_dir[u] = exact_kernel::cell_third_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_dir,
                            &neg_coeff_au,
                            &neg_coeff_a_dir,
                            &neg_coeff_u_dir,
                            &neg_coeff_au_dir,
                            &state.moments,
                        )?;
                    }

                    for u in 0..p {
                        for v in u..p {
                            let neg_coeff_u = fixed.coeff_u[u].map(|val| -val);
                            let neg_coeff_v = fixed.coeff_u[v].map(|val| -val);
                            let sc_uv = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, v);
                            let neg_sc_uv = sc_uv.map(|val| -val);

                            let base_val = exact_kernel::cell_second_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_sc_uv,
                                &state.moments,
                            )?;
                            f_uv[u * p + v] = base_val;
                            f_uv[v * p + u] = base_val;

                            let mut neg_coeff_u_dir = [0.0; 4];
                            let mut neg_coeff_v_dir = [0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_uc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_vc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                for k in 0..4 {
                                    neg_coeff_u_dir[k] -= sc_uc[k] * dir[c];
                                    neg_coeff_v_dir[k] -= sc_vc[k] * dir[c];
                                }
                            }

                            let dir_val = exact_kernel::cell_third_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_coeff_dir,
                                &neg_sc_uv,
                                &neg_coeff_u_dir,
                                &neg_coeff_v_dir,
                                &[0.0; 4], // third cross vanishes for cubic cells
                                &state.moments,
                            )?;
                            f_uv_dir[u * p + v] = dir_val;
                            f_uv_dir[v * p + u] = dir_val;
                        }
                    }

                    Ok(DirectionalTimepointCellAccum {
                        f_a,
                        f_aa,
                        f_u,
                        f_au,
                        f_uv,
                        f_a_dir,
                        f_aa_dir,
                        f_au_dir,
                        f_uv_dir,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_au_dir = Array1::<f64>::zeros(p);
        let mut f_uv_dir = Array2::<f64>::zeros((p, p));
        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_dir += acc.f_a_dir;
            f_aa_dir += acc.f_aa_dir;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_dir[u] += acc.f_au_dir[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                    f_uv_dir[[u, v]] += acc.f_uv_dir[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        f_uv_dir[[q_index, q_index]] += dir[q_index] * (1.0 - q * q) * phi_q;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        let mut a_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        // Observed-point quantities and their dir-extensions
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_val = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut tau = Array1::<f64>::zeros(p);
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut tau_a = Array1::<f64>::zeros(p);
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b).1,
                    scale,
                );
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi_dir = eta_aa * a_dir + tau.dot(dir);
        let eta_aa_dir = eta_aaa * a_dir
            + eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_GW),
                z_obs,
            );
        let eta_aaa_dir = eval_coeff4_at(
            &g_jet.directional_family(g_jet.aaa_first, dir, COEFF_SUPPORT_GW),
            z_obs,
        );

        let mut tau_dir = Array1::<f64>::zeros(p);
        let mut tau_a_dir = Array1::<f64>::zeros(p);
        for u in 0..p {
            let fixed_tau_dir =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_GW);
            tau_dir[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_dir, z_obs);

            let fixed_tau_a_dir =
                g_jet.param_directional_from_b_family(g_jet.aab_first, u, dir, COEFF_SUPPORT_GW);
            tau_a_dir[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_a_dir, z_obs);
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((p, p));
        let mut chi_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv_dir = self.observed_fixed_eta_second_partial_dir(
                    primary, &obs, u, v, z_obs, u_obs, a, b, a_dir, dir, beta_w,
                )?;
                let chi_uv_fixed_dir = self.observed_fixed_chi_second_partial_dir(
                    primary, u, v, z_obs, u_obs, a_dir, dir,
                )?;

                let eta_val = chi_dir * a_uv[[u, v]]
                    + chi_val * a_uv_dir[[u, v]]
                    + eta_aa_dir * a_u[u] * a_u[v]
                    + eta_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_dir[u] * a_u[v]
                    + tau[u] * a_u_dir[v]
                    + tau_dir[v] * a_u[u]
                    + tau[v] * a_u_dir[u]
                    + r_uv_dir;
                eta_uv_dir[[u, v]] = eta_val;
                eta_uv_dir[[v, u]] = eta_val;

                let chi_v = eta_aa_dir * a_uv[[u, v]]
                    + eta_aa * a_uv_dir[[u, v]]
                    + eta_aaa_dir * a_u[u] * a_u[v]
                    + eta_aaa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_a_dir[u] * a_u[v]
                    + tau_a[u] * a_u_dir[v]
                    + tau_a_dir[v] * a_u[u]
                    + tau_a[v] * a_u_dir[u]
                    + chi_uv_fixed_dir;
                chi_uv_dir[[u, v]] = chi_v;
                chi_uv_dir[[v, u]] = chi_v;
            }
        }

        // D_u_dir: directional derivative of the density normalization first derivative.
        let d_u_dir_cell_accums = cached
            .cells
            .iter()
            .map(|cell_entry| -> Result<Array1<f64>, String> {
                let mut d_u_dir = Array1::<f64>::zeros(p);
                let cell = cell_entry.partition_cell.cell;
                let state_ref = &cell_entry.state;
                let fixed = &cell_entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();

                let mut eta_u_poly = vec![PolyVec::new(); p];
                let mut chi_u_poly = vec![PolyVec::new(); p];
                for u in 0..p {
                    eta_u_poly[u] =
                        poly_add(&poly_scale(&chi_poly, a_u[u]), fixed.coeff_u[u].as_ref());
                    chi_u_poly[u] = poly_add(
                        &poly_scale(&eta_aa_poly, a_u[u]),
                        fixed.coeff_au[u].as_ref(),
                    );
                }

                let mut coeff_dir_poly = vec![0.0; 4];
                let mut coeff_a_dir_poly = vec![0.0; 4];
                for c in 0..p {
                    if dir[c] == 0.0 {
                        continue;
                    }
                    for k in 0..4 {
                        coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                        coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                    }
                }
                let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);

                for u in 0..p {
                    let mut eta_u_dir_fixed = vec![0.0; 4];
                    let mut chi_u_dir_fixed = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                        for k in 0..4 {
                            eta_u_dir_fixed[k] += sc[k] * dir[c];
                            chi_u_dir_fixed[k] += sca[k] * dir[c];
                        }
                    }
                    let eta_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&chi_poly, a_u_dir[u]),
                            &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                        ),
                        &eta_u_dir_fixed,
                    );
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let chi_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&eta_aa_poly, a_u_dir[u]),
                            &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                        ),
                        &chi_u_dir_fixed,
                    );

                    // D_u integrand: chi_u - chi * eta * eta_u
                    let integrand_base = poly_sub(
                        &chi_u_poly[u],
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly[u]),
                    );
                    // Polynomial derivative of integrand w.r.t. dir
                    let integrand_dir = poly_sub(
                        &poly_sub(
                            &poly_sub(
                                &chi_u_dir_poly,
                                &poly_mul(&poly_mul(&coeff_a_dir_poly, &eta_poly), &eta_u_poly[u]),
                            ),
                            &poly_mul(&poly_mul(&chi_poly, &eta_dir_poly), &eta_u_poly[u]),
                        ),
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_dir_poly),
                    );
                    // Moment-weighting correction: -eta*eta_dir * integrand_base
                    let full_integrand = poly_sub(
                        &integrand_dir,
                        &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &integrand_base),
                    );

                    d_u_dir[u] += exact_kernel::cell_polynomial_integral_from_moments(
                        &full_integrand,
                        &state_ref.moments,
                        "survival D_t first derivative directional",
                    )?;
                }
                Ok(d_u_dir)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u_dir = Array1::<f64>::zeros(p);
        for cell_d_u_dir in d_u_dir_cell_accums {
            for u in 0..p {
                d_u_dir[u] += cell_d_u_dir[u];
            }
        }

        // D_uv_dir
        let mut d_uv_dir = Array2::<f64>::zeros((p, p));
        if need_d_uv_dir {
            let d_uv_dir_cell_accums = cached
                .cells
                .iter()
                .map(|cell_entry| -> Result<Array2<f64>, String> {
                    let mut d_uv_dir = Array2::<f64>::zeros((p, p));
                    let cell = cell_entry.partition_cell.cell;
                    let state_ref = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();

                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    for u in 0..p {
                        eta_u_poly[u] =
                            poly_add(&poly_scale(&chi_poly, a_u[u]), fixed.coeff_u[u].as_ref());
                        chi_u_poly[u] = poly_add(
                            &poly_scale(&eta_aa_poly, a_u[u]),
                            fixed.coeff_au[u].as_ref(),
                        );
                    }
                    let mut coeff_dir_poly = vec![0.0; 4];
                    let mut coeff_a_dir_poly = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                            coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                        }
                    }
                    let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);
                    let chi_dir_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_dir), &coeff_a_dir_poly);

                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };

                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(fixed.coeff_au[u].as_ref(), a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_au[v].as_ref(), a_u[u]),
                                    &r_uv_fixed,
                                ),
                            );

                            // D_uv integrand: 5 terms
                            let t1 = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_aau[u].as_ref(), a_u[v]),
                                    &poly_add(
                                        &poly_scale(fixed.coeff_aau[v].as_ref(), a_u[u]),
                                        &if u == primary.g {
                                            fixed.coeff_abu[v].to_vec()
                                        } else if v == primary.g {
                                            fixed.coeff_abu[u].to_vec()
                                        } else {
                                            vec![0.0; 4]
                                        },
                                    ),
                                ),
                            );
                            let t2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let t3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let t4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let i_base =
                                poly_add(&poly_add(&poly_add(&t1, &t2), &t3), &poly_add(&t4, &t5));

                            // Polynomial dir-derivatives of per-u quantities
                            let mut eu_dir_fixed_u = vec![0.0; 4];
                            let mut eu_dir_fixed_v = vec![0.0; 4];
                            let mut cu_dir_fixed_u = vec![0.0; 4];
                            let mut cu_dir_fixed_v = vec![0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_u =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_v =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                let sca_u =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                                let sca_v =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, v, c);
                                for k in 0..4 {
                                    eu_dir_fixed_u[k] += sc_u[k] * dir[c];
                                    eu_dir_fixed_v[k] += sc_v[k] * dir[c];
                                    cu_dir_fixed_u[k] += sca_u[k] * dir[c];
                                    cu_dir_fixed_v[k] += sca_v[k] * dir[c];
                                }
                            }
                            let eta_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                                ),
                                &eu_dir_fixed_u,
                            );
                            let eta_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aa_poly, a_u[v] * a_dir),
                                ),
                                &eu_dir_fixed_v,
                            );
                            let chi_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                                ),
                                &cu_dir_fixed_u,
                            );
                            let chi_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aaa_poly, a_u[v] * a_dir),
                                ),
                                &cu_dir_fixed_v,
                            );
                            let eta_uv_dir_poly = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_uv_dir[[u, v]]),
                                    &poly_scale(
                                        &eta_aa_poly,
                                        a_u_dir[u] * a_u[v]
                                            + a_u[u] * a_u_dir[v]
                                            + a_uv[[u, v]] * a_dir,
                                    ),
                                ),
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(fixed.coeff_au[u].as_ref(), a_u_dir[v]),
                                        &poly_scale(fixed.coeff_au[v].as_ref(), a_u_dir[u]),
                                    ),
                                    &{
                                        let mut fp = vec![0.0; 4];
                                        for c in 0..p {
                                            if dir[c] == 0.0 {
                                                continue;
                                            }
                                            let sca_u = self.cell_pair_third_coeff_a(
                                                primary,
                                                &fixed.coeff_abu,
                                                u,
                                                c,
                                            );
                                            let sca_v = self.cell_pair_third_coeff_a(
                                                primary,
                                                &fixed.coeff_abu,
                                                v,
                                                c,
                                            );
                                            for k in 0..4 {
                                                fp[k] += sca_u[k] * dir[c] * a_u[v]
                                                    + sca_v[k] * dir[c] * a_u[u];
                                            }
                                        }
                                        fp
                                    },
                                ),
                            );

                            // Differentiate each of the 5 integrand terms
                            let t1_dir = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv_dir[[u, v]]),
                                    &poly_scale(
                                        &eta_aaa_poly,
                                        a_u_dir[u] * a_u[v]
                                            + a_u[u] * a_u_dir[v]
                                            + a_uv[[u, v]] * a_dir,
                                    ),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_aau[u].as_ref(), a_u_dir[v]),
                                    &poly_scale(fixed.coeff_aau[v].as_ref(), a_u_dir[u]),
                                ),
                            );
                            let t2_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_v, &eta_poly),
                                            &eta_u_poly[u],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[v], &eta_dir_poly),
                                            &eta_u_poly[u],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[v], &eta_poly),
                                        &eta_u_dir_poly_u,
                                    ),
                                ),
                                -1.0,
                            );
                            let t3_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_u, &eta_poly),
                                            &eta_u_poly[v],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[u], &eta_dir_poly),
                                            &eta_u_poly[v],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[u], &eta_poly),
                                        &eta_u_dir_poly_v,
                                    ),
                                ),
                                -1.0,
                            );
                            let t4_dir = poly_scale(
                                &poly_add(
                                    &poly_mul(
                                        &chi_dir_poly,
                                        &poly_add(
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                            &poly_mul(&eta_poly, &eta_uv_poly),
                                        ),
                                    ),
                                    &poly_mul(
                                        &chi_poly,
                                        &poly_add(
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                            &poly_add(
                                                &poly_mul(&eta_dir_poly, &eta_uv_poly),
                                                &poly_mul(&eta_poly, &eta_uv_dir_poly),
                                            ),
                                        ),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5_dir = poly_add(
                                &poly_mul(
                                    &chi_dir_poly,
                                    &poly_mul(
                                        &poly_mul(&eta_poly, &eta_poly),
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                    ),
                                ),
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(
                                            &poly_scale(&poly_mul(&eta_dir_poly, &eta_poly), 2.0),
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        ),
                                        &poly_mul(
                                            &poly_mul(&eta_poly, &eta_poly),
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                        ),
                                    ),
                                ),
                            );

                            let i_base_dir = poly_add(
                                &poly_add(&poly_add(&t1_dir, &t2_dir), &t3_dir),
                                &poly_add(&t4_dir, &t5_dir),
                            );
                            let full_integrand = poly_sub(
                                &i_base_dir,
                                &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &i_base),
                            );

                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &full_integrand,
                                &state_ref.moments,
                                "survival D_t second derivative directional",
                            )?;
                            d_uv_dir[[u, v]] += value;
                            d_uv_dir[[v, u]] = d_uv_dir[[u, v]];
                        }
                    }
                    Ok(d_uv_dir)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for cell_d_uv_dir in d_uv_dir_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv_dir[[u, v]] += cell_d_uv_dir[[u, v]];
                    }
                }
            }
        }

        Ok(SurvivalFlexTimepointDirectionalExact {
            eta_uv_dir,
            chi_uv_dir,
            d_u_dir,
            d_uv_dir,
        })
    }

}

impl SurvivalMarginalSlopeFamily {
    /// Exact mixed bidirectional extension D_{d1} D_{d2} of the timepoint
    /// quantities. This carries the calibration solve, observed eta/chi
    /// transport, and density-normalization transport analytically.
    pub(crate) fn compute_survival_timepoint_bidirectional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        self.compute_survival_timepoint_bidirectional_exact_full(
            row, primary, q, q_index, a, b, beta_h, beta_w, dir1, dir2,
        )
    }

    pub(crate) fn compute_survival_timepoint_bidirectional_exact_full(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        let p = primary.total;
        let zero4 = [0.0; 4];
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct BiDirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Array1<f64>,
            f_au: Array1<f64>,
            f_uv: Array2<f64>,
            f_a_d1: f64,
            f_aa_d1: f64,
            f_au_d1: Array1<f64>,
            f_uv_d1: Array2<f64>,
            f_a_d2: f64,
            f_aa_d2: f64,
            f_au_d2: Array1<f64>,
            f_uv_d2: Array2<f64>,
            f_a_d12: f64,
            f_aa_d12: f64,
            f_au_d12: Array1<f64>,
            f_uv_d12: Array2<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<BiDirectionalTimepointCellAccum, String> {
                let mut f_a = 0.0f64;
                let mut f_aa = 0.0f64;
                let mut f_u = Array1::<f64>::zeros(p);
                let mut f_au = Array1::<f64>::zeros(p);
                let mut f_uv = Array2::<f64>::zeros((p, p));
                let mut f_a_d1 = 0.0f64;
                let mut f_aa_d1 = 0.0f64;
                let mut f_au_d1 = Array1::<f64>::zeros(p);
                let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
                let mut f_a_d2 = 0.0f64;
                let mut f_aa_d2 = 0.0f64;
                let mut f_au_d2 = Array1::<f64>::zeros(p);
                let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
                let mut f_a_d12 = 0.0f64;
                let mut f_aa_d12 = 0.0f64;
                let mut f_au_d12 = Array1::<f64>::zeros(p);
                let mut f_uv_d12 = Array2::<f64>::zeros((p, p));
                let nc = ce.neg_cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let da = fx.dc_da.map(|v| -v);
                let daa = fx.dc_daa.map(|v| -v);

                f_a += exact_kernel::cell_first_derivative_from_moments(&da, &st.moments)?;
                f_aa += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &daa,
                    &st.moments,
                )?;

                let mut cd1 = [0.0; 4];
                let mut ca1 = [0.0; 4];
                let mut caa1 = [0.0; 4];
                let mut cd2 = [0.0; 4];
                let mut ca2 = [0.0; 4];
                let mut caa2 = [0.0; 4];
                let mut cd12 = [0.0; 4];
                let mut ca12 = [0.0; 4];
                for c in 0..p {
                    for k in 0..4 {
                        if dir1[c] != 0.0 {
                            cd1[k] -= fx.coeff_u[c][k] * dir1[c];
                            ca1[k] -= fx.coeff_au[c][k] * dir1[c];
                            caa1[k] -= fx.coeff_aau[c][k] * dir1[c];
                        }
                        if dir2[c] != 0.0 {
                            cd2[k] -= fx.coeff_u[c][k] * dir2[c];
                            ca2[k] -= fx.coeff_au[c][k] * dir2[c];
                            caa2[k] -= fx.coeff_aau[c][k] * dir2[c];
                        }
                    }
                }
                for c1 in 0..p {
                    if dir1[c1] == 0.0 {
                        continue;
                    }
                    for c2 in 0..p {
                        if dir2[c2] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, c1, c2);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, c1, c2);
                        for k in 0..4 {
                            cd12[k] -= sc[k] * dir1[c1] * dir2[c2];
                            ca12[k] -= sca[k] * dir1[c1] * dir2[c2];
                        }
                    }
                }

                f_a_d1 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &ca1,
                    &st.moments,
                )?;
                f_a_d2 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd2,
                    &ca2,
                    &st.moments,
                )?;
                f_a_d12 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &cd2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &ca12,
                    &st.moments,
                )?;
                f_aa_d1 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &daa,
                    &ca1,
                    &ca1,
                    &caa1,
                    &st.moments,
                )?;
                f_aa_d2 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd2,
                    &daa,
                    &ca2,
                    &ca2,
                    &caa2,
                    &st.moments,
                )?;
                f_aa_d12 += exact_kernel::cell_fourth_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &cd2,
                    &daa,
                    &ca1,
                    &ca2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &caa1,
                    &caa2,
                    &ca12,
                    &ca12,
                    &[0.0; 4],
                    &st.moments,
                )?;

                for u in 0..p {
                    let cu = fx.coeff_u[u].map(|v| -v);
                    let cau = fx.coeff_au[u].map(|v| -v);
                    f_u[u] += exact_kernel::cell_first_derivative_from_moments(&cu, &st.moments)?;
                    f_au[u] += exact_kernel::cell_second_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cau,
                        &st.moments,
                    )?;
                    let mut cu1 = [0.0; 4];
                    let mut cau1 = [0.0; 4];
                    let mut cu2 = [0.0; 4];
                    let mut cau2 = [0.0; 4];
                    for c in 0..p {
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, c);
                        for k in 0..4 {
                            if dir1[c] != 0.0 {
                                cu1[k] -= sc[k] * dir1[c];
                                cau1[k] -= sca[k] * dir1[c];
                            }
                            if dir2[c] != 0.0 {
                                cu2[k] -= sc[k] * dir2[c];
                                cau2[k] -= sca[k] * dir2[c];
                            }
                        }
                    }
                    f_au_d1[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cau,
                        &ca1,
                        &cu1,
                        &cau1,
                        &st.moments,
                    )?;
                    f_au_d2[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd2,
                        &cau,
                        &ca2,
                        &cu2,
                        &cau2,
                        &st.moments,
                    )?;
                    f_au_d12[u] += exact_kernel::cell_fourth_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cd2,
                        &cau,
                        &ca1,
                        &ca2,
                        &cu1,
                        &cu2,
                        &cd12,
                        &cau1,
                        &cau2,
                        &ca12,
                        &[0.0; 4],
                        &[0.0; 4],
                        &st.moments,
                    )?;
                }
                for u in 0..p {
                    for v in u..p {
                        let cu = fx.coeff_u[u].map(|x| -x);
                        let cv = fx.coeff_u[v].map(|x| -x);
                        let sc = self
                            .cell_pair_second_coeff(primary, &fx.coeff_bu, u, v)
                            .map(|x| -x);
                        let bv = exact_kernel::cell_second_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &sc,
                            &st.moments,
                        )?;
                        f_uv[[u, v]] += bv;
                        if u != v {
                            f_uv[[v, u]] += bv;
                        }
                        let mut cu1 = [0.0; 4];
                        let mut cv1 = [0.0; 4];
                        let mut cu2 = [0.0; 4];
                        let mut cv2 = [0.0; 4];
                        for c in 0..p {
                            let suc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                            let svc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, v, c);
                            for k in 0..4 {
                                if dir1[c] != 0.0 {
                                    cu1[k] -= suc[k] * dir1[c];
                                    cv1[k] -= svc[k] * dir1[c];
                                }
                                if dir2[c] != 0.0 {
                                    cu2[k] -= suc[k] * dir2[c];
                                    cv2[k] -= svc[k] * dir2[c];
                                }
                            }
                        }
                        let d1v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &sc,
                            &cu1,
                            &cv1,
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d1[[u, v]] += d1v;
                        if u != v {
                            f_uv_d1[[v, u]] += d1v;
                        }
                        let d2v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd2,
                            &sc,
                            &cu2,
                            &cv2,
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d2[[u, v]] += d2v;
                        if u != v {
                            f_uv_d2[[v, u]] += d2v;
                        }
                        let d12v = exact_kernel::cell_fourth_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &cd2,
                            &sc,
                            &cu1,
                            &cu2,
                            &cv1,
                            &cv2,
                            &cd12,
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d12[[u, v]] += d12v;
                        if u != v {
                            f_uv_d12[[v, u]] += d12v;
                        }
                    }
                }

                Ok(BiDirectionalTimepointCellAccum {
                    f_a,
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                    f_a_d1,
                    f_aa_d1,
                    f_au_d1,
                    f_uv_d1,
                    f_a_d2,
                    f_aa_d2,
                    f_au_d2,
                    f_uv_d2,
                    f_a_d12,
                    f_aa_d12,
                    f_au_d12,
                    f_uv_d12,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0f64;
        let mut f_aa = 0.0f64;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_d1 = 0.0f64;
        let mut f_aa_d1 = 0.0f64;
        let mut f_au_d1 = Array1::<f64>::zeros(p);
        let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
        let mut f_a_d2 = 0.0f64;
        let mut f_aa_d2 = 0.0f64;
        let mut f_au_d2 = Array1::<f64>::zeros(p);
        let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
        let mut f_a_d12 = 0.0f64;
        let mut f_aa_d12 = 0.0f64;
        let mut f_au_d12 = Array1::<f64>::zeros(p);
        let mut f_uv_d12 = Array2::<f64>::zeros((p, p));

        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_d1 += acc.f_a_d1;
            f_aa_d1 += acc.f_aa_d1;
            f_a_d2 += acc.f_a_d2;
            f_aa_d2 += acc.f_aa_d2;
            f_a_d12 += acc.f_a_d12;
            f_aa_d12 += acc.f_aa_d12;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_d1[u] += acc.f_au_d1[u];
                f_au_d2[u] += acc.f_au_d2[u];
                f_au_d12[u] += acc.f_au_d12[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[[u, v]];
                    f_uv_d1[[u, v]] += acc.f_uv_d1[[u, v]];
                    f_uv_d2[[u, v]] += acc.f_uv_d2[[u, v]];
                    f_uv_d12[[u, v]] += acc.f_uv_d12[[u, v]];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        f_uv_d1[[q_index, q_index]] += dir1[q_index] * (1.0 - q * q) * phi_q;
        f_uv_d2[[q_index, q_index]] += dir2[q_index] * (1.0 - q * q) * phi_q;
        f_uv_d12[[q_index, q_index]] += dir1[q_index] * dir2[q_index] * q * (q * q - 3.0) * phi_q;

        let inv = 1.0 / f_a;
        let mut au = Array1::<f64>::zeros(p);
        for u in 0..p {
            au[u] = -f_u[u] * inv;
        }
        let mut auv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * au[v] + f_au[v] * au[u] + f_aa * au[u] * au[v])
                        * inv;
                auv[[u, v]] = val;
                auv[[v, u]] = val;
            }
        }
        let ad1 = au.dot(dir1);
        let ad2 = au.dot(dir2);
        let aud1 = auv.dot(dir1);
        let aud2 = auv.dot(dir2);

        let mut auvd1 = Array2::<f64>::zeros((p, p));
        let mut auvd2 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n1 = f_uv_d1[[u, v]]
                    + f_au_d1[u] * au[v]
                    + f_au[u] * aud1[v]
                    + f_au_d1[v] * au[u]
                    + f_au[v] * aud1[u]
                    + f_aa_d1 * au[u] * au[v]
                    + f_aa * (aud1[u] * au[v] + au[u] * aud1[v]);
                let v1 = -(n1 + f_a_d1 * auv[[u, v]]) * inv;
                auvd1[[u, v]] = v1;
                auvd1[[v, u]] = v1;

                let n2 = f_uv_d2[[u, v]]
                    + f_au_d2[u] * au[v]
                    + f_au[u] * aud2[v]
                    + f_au_d2[v] * au[u]
                    + f_au[v] * aud2[u]
                    + f_aa_d2 * au[u] * au[v]
                    + f_aa * (aud2[u] * au[v] + au[u] * aud2[v]);
                let v2 = -(n2 + f_a_d2 * auv[[u, v]]) * inv;
                auvd2[[u, v]] = v2;
                auvd2[[v, u]] = v2;
            }
        }

        let ad12 = aud2.dot(dir1);
        let aud12 = auvd2.dot(dir1);
        let mut auvd12 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n = f_uv_d12[[u, v]]
                    + f_au_d12[u] * au[v]
                    + f_au_d1[u] * aud2[v]
                    + f_au_d2[u] * aud1[v]
                    + f_au[u] * aud12[v]
                    + f_au_d12[v] * au[u]
                    + f_au_d1[v] * aud2[u]
                    + f_au_d2[v] * aud1[u]
                    + f_au[v] * aud12[u]
                    + f_aa_d12 * au[u] * au[v]
                    + f_aa_d1 * (aud2[u] * au[v] + au[u] * aud2[v])
                    + f_aa_d2 * (aud1[u] * au[v] + au[u] * aud1[v])
                    + f_aa
                        * (aud12[u] * au[v]
                            + aud1[u] * aud2[v]
                            + aud2[u] * aud1[v]
                            + au[u] * aud12[v]);
                let val =
                    -(n + f_a_d12 * auv[[u, v]] + f_a_d1 * auvd2[[u, v]] + f_a_d2 * auvd1[[u, v]])
                        * inv;
                auvd12[[u, v]] = val;
                auvd12[[v, u]] = val;
            }
        }

        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, dc_bw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(dc_bw, scale);
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let chi_jet = scalar_composite_bilinear(
            chi,
            eta_aa,
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.ab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aa_jet = scalar_composite_bilinear(
            eta_aa,
            eta_aaa,
            0.0,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aaa_jet = MultiDirJet::bilinear(
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
        );

        let mut a_u_jets = Vec::with_capacity(p);
        let mut tau_jets = Vec::with_capacity(p);
        let mut tau_a_jets = Vec::with_capacity(p);
        for u in 0..p {
            a_u_jets.push(MultiDirJet::bilinear(au[u], aud1[u], aud2[u], aud12[u]));
            tau_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_au_fixed[u], z_obs),
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                ad1,
                ad2,
                ad12,
            ));
            tau_a_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                0.0,
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                0.0,
                0.0,
                ad1,
                ad2,
                ad12,
            ));
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv_uv = Array2::<f64>::zeros((p, p));
        let mut d_uv_uv = Array2::<f64>::zeros((p, p));

        for u in 0..p {
            for v in u..p {
                let a_uv_jet = MultiDirJet::bilinear(
                    auv[[u, v]],
                    auvd1[[u, v]],
                    auvd2[[u, v]],
                    auvd12[[u, v]],
                );
                let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                let r_uv_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_GHW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_uv_fixed_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    0.0,
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    0.0,
                    0.0,
                    ad1,
                    ad2,
                    ad12,
                );

                let eta_uv_jet = chi_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aa_jet.mul(&a_u_prod))
                    .add(&tau_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_jets[v].mul(&a_u_jets[u]))
                    .add(&r_uv_jet);
                let chi_uv_jet = eta_aa_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aaa_jet.mul(&a_u_prod))
                    .add(&tau_a_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_a_jets[v].mul(&a_u_jets[u]))
                    .add(&chi_uv_fixed_jet);

                eta_uv_uv[[u, v]] = eta_uv_jet.coeff(3);
                eta_uv_uv[[v, u]] = eta_uv_uv[[u, v]];
                chi_uv_uv[[u, v]] = chi_uv_jet.coeff(3);
                chi_uv_uv[[v, u]] = chi_uv_uv[[u, v]];
            }
        }

        let primary_view = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let d_uv_uv_cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<Array2<f64>, String> {
                let mut d_uv_uv = Array2::<f64>::zeros((p, p));
                let cell = ce.partition_cell.cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let eta_base = [cell.c0, cell.c1, cell.c2, cell.c3];

                let coeff_dir1 =
                    primary_view.directional_family(&fx.coeff_u, dir1, COEFF_SUPPORT_GHW);
                let coeff_dir2 =
                    primary_view.directional_family(&fx.coeff_u, dir2, COEFF_SUPPORT_GHW);
                let coeff_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_bu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_a_dir1 =
                    primary_view.directional_family(&fx.coeff_au, dir1, COEFF_SUPPORT_GHW);
                let coeff_a_dir2 =
                    primary_view.directional_family(&fx.coeff_au, dir2, COEFF_SUPPORT_GHW);
                let coeff_a_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_abu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_aa_dir1 =
                    primary_view.directional_family(&fx.coeff_aau, dir1, COEFF_SUPPORT_GHW);
                let coeff_aa_dir2 =
                    primary_view.directional_family(&fx.coeff_aau, dir2, COEFF_SUPPORT_GHW);
                let coeff_aa_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_aabu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_aaa_dir1 =
                    primary_view.directional_family(&fx.coeff_aaau, dir1, COEFF_SUPPORT_GHW);
                let coeff_aaa_dir2 =
                    primary_view.directional_family(&fx.coeff_aaau, dir2, COEFF_SUPPORT_GHW);
                let coeff_aaa_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_aabu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );

                let eta_poly_jet = coeff4_composite_bilinear(
                    &eta_base,
                    &fx.dc_da,
                    &fx.dc_daa,
                    &coeff_dir1,
                    &coeff_dir2,
                    &coeff_dir12,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_da,
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    &coeff_a_dir12,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aa_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &zero4,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    &coeff_aa_dir12,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aaa_poly_jet = coeff4_fixed_bilinear(
                    &fx.dc_daaa,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    &coeff_aaa_dir12,
                );

                let mut eta_u_poly_jets = Vec::with_capacity(p);
                let mut chi_u_poly_jets = Vec::with_capacity(p);
                let mut coeff_au_fixed_jets = Vec::with_capacity(p);
                let mut coeff_aau_fixed_jets = Vec::with_capacity(p);
                for u in 0..p {
                    let coeff_u_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_bu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_u_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_bu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_u_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_bbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_abu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_abu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_aabu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_aabu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );

                    let coeff_u_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_u[u],
                        &coeff_u_dir1,
                        &coeff_u_dir2,
                        &coeff_u_dir12,
                    );
                    let coeff_au_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_au[u],
                        &coeff_au_dir1,
                        &coeff_au_dir2,
                        &coeff_au_dir12,
                    );
                    let coeff_aau_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_aau[u],
                        &coeff_aau_dir1,
                        &coeff_aau_dir2,
                        &coeff_aau_dir12,
                    );

                    eta_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&chi_poly_jet, &a_u_jets[u]),
                        &coeff_u_fixed_jet,
                    ));
                    chi_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&eta_aa_poly_jet, &a_u_jets[u]),
                        &coeff_au_fixed_jet,
                    ));
                    coeff_au_fixed_jets.push(coeff_au_fixed_jet);
                    coeff_aau_fixed_jets.push(coeff_aau_fixed_jet);
                }

                for u in 0..p {
                    for v in u..p {
                        let a_uv_jet = MultiDirJet::bilinear(
                            auv[[u, v]],
                            auvd1[[u, v]],
                            auvd2[[u, v]],
                            auvd12[[u, v]],
                        );
                        let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                        let r_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, v),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                        );
                        let chi_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, v),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                        );

                        let eta_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&chi_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_au_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_au_fixed_jets[v], &a_u_jets[u]),
                                    &r_uv_fixed_jet,
                                ),
                            ),
                        );
                        let chi_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&eta_aa_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aaa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_aau_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_aau_fixed_jets[v], &a_u_jets[u]),
                                    &chi_uv_fixed_jet,
                                ),
                            ),
                        );

                        let t1 = chi_uv_poly_jet.clone();
                        let t2 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[v], &eta_poly_jet),
                                &eta_u_poly_jets[u],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t3 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[u], &eta_poly_jet),
                                &eta_u_poly_jets[v],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t4 = poly_scale_jets(
                            &poly_mul_jets(
                                &chi_poly_jet,
                                &poly_add_jets(
                                    &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                                    &poly_mul_jets(&eta_poly_jet, &eta_uv_poly_jet),
                                ),
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t5 = poly_mul_jets(
                            &chi_poly_jet,
                            &poly_mul_jets(
                                &poly_mul_jets(&eta_poly_jet, &eta_poly_jet),
                                &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                            ),
                        );
                        let i_base_jet = poly_add_jets(
                            &poly_add_jets(&poly_add_jets(&t1, &t2), &t3),
                            &poly_add_jets(&t4, &t5),
                        );

                        let i_base = poly_coeff_mask(&i_base_jet, 0);
                        let i_base_d2 = poly_coeff_mask(&i_base_jet, 2);
                        let i_base_d12 = poly_coeff_mask(&i_base_jet, 3);
                        let eta_poly = poly_coeff_mask(&eta_poly_jet, 0);
                        let eta_d1_poly = poly_coeff_mask(&eta_poly_jet, 1);
                        let eta_d2_poly = poly_coeff_mask(&eta_poly_jet, 2);
                        let eta_d12_poly = poly_coeff_mask(&eta_poly_jet, 3);

                        let correction = poly_add(
                            &poly_mul(
                                &poly_add(
                                    &poly_mul(&eta_d2_poly, &eta_d1_poly),
                                    &poly_mul(&eta_poly, &eta_d12_poly),
                                ),
                                &i_base,
                            ),
                            &poly_mul(&poly_mul(&eta_poly, &eta_d1_poly), &i_base_d2),
                        );
                        let full_integrand = poly_sub(&i_base_d12, &correction);
                        let value = exact_kernel::cell_polynomial_integral_from_moments(
                            &full_integrand,
                            &st.moments,
                            "survival D_t second derivative bidirectional",
                        )?;
                        d_uv_uv[[u, v]] += value;
                        d_uv_uv[[v, u]] = d_uv_uv[[u, v]];
                    }
                }
                Ok(d_uv_uv)
            })
            .collect::<Result<Vec<_>, String>>()?;
        for cell_d_uv_uv in d_uv_uv_cell_accums {
            for u in 0..p {
                for v in 0..p {
                    d_uv_uv[[u, v]] += cell_d_uv_uv[[u, v]];
                }
            }
        }

        Ok(SurvivalFlexTimepointBiDirectionalExact {
            eta_uv_uv,
            chi_uv_uv,
            d_uv_uv,
        })
    }

}

impl SurvivalMarginalSlopeFamily {
    /// Exact third-order directional contraction for the flexible survival
    /// path.  Returns D_dir H[u,v] where H is the primary-space NLL Hessian.
    pub(crate) fn row_flex_primary_third_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_third_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival third contracted: dir length {} != primary dimension {p}",
                    dir.len()
                ),
            }
            .into());
        }
        if dir.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival third contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry = self.compute_survival_timepoint_exact(
            row, &primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl, false,
        )?;
        let exit = self.compute_survival_timepoint_exact(
            row, &primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl, true,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival third contracted row {row}: non-positive chi1={:.3e}",
                    exit.chi,
                ),
            }
            .into());
        }

        let entry_ext = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir, false,
        )?;
        let exit_ext = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir, true,
        )?;

        // Delegate the per-(u, v) assembly to the Block 10 GPU-substrate
        // pure assembler in `crate::families::survival_marginal_slope_gpu`.  This is the
        // single source of truth for the third-contraction inner loop —
        // shared with the GPU dispatch path so CPU/GPU cannot drift.
        let entry_b = block10_pack_base(&entry);
        let exit_b = block10_pack_base(&exit);
        let entry_d = block10_pack_dir(&entry_ext);
        let exit_d = block10_pack_dir(&exit_ext);
        let dir_vec: Vec<f64> = dir.to_vec();
        let inputs = crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10ThirdInputs {
            p,
            qd1_index: primary.qd1,
            qd1,
            w: self.weights[row],
            d: self.event[row],
            dir: &dir_vec,
            entry_base: &entry_b,
            exit_base: &exit_b,
            entry_ext: &entry_d,
            exit_ext: &exit_d,
        };
        let flat =
            crate::families::survival_marginal_slope_gpu::cpu_oracle_third_contraction(&inputs)?;
        Ok(Array2::<f64>::from_shape_vec((p, p), flat).map_err(|e| e.to_string())?)
    }

    /// Fourth-order directional contraction for the flexible survival path.
    ///
    /// The mixed second-directional timepoint transport is carried exactly
    /// through the implicit intercept solve, the observed-point eta/chi jets,
    /// and the cellwise density-normalization integrand.
    pub(crate) fn row_flex_primary_fourth_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_fourth_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir_u.len() != p || dir_v.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival fourth contracted: dir lengths ({},{}) != {p}",
                    dir_u.len(),
                    dir_v.len(),
                ),
            }
            .into());
        }
        if dir_u.iter().all(|v| v.abs() == 0.0) || dir_v.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival fourth contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry_base = self.compute_survival_timepoint_exact(
            row, &primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl, false,
        )?;
        let exit_base = self.compute_survival_timepoint_exact(
            row, &primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl, true,
        )?;

        if !exit_base.chi.is_finite() || exit_base.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival fourth contracted row {row}: non-positive chi1={:.3e}",
                    exit_base.chi,
                ),
            }
            .into());
        }

        let entry_ext_u = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_u, false,
        )?;
        let entry_ext_v = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_v, false,
        )?;
        let exit_ext_u = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_u, true,
        )?;
        let exit_ext_v = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_v, true,
        )?;

        // Bidirectional extensions D_{d1} D_{d2} (η_uv, χ_uv, D_uv) via exact
        // IFT second-order recursion through the cell kernel.
        let entry_bi = self.compute_survival_timepoint_bidirectional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_u, dir_v,
        )?;
        let exit_bi = self.compute_survival_timepoint_bidirectional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_u, dir_v,
        )?;

        // Delegate the per-(u, v) assembly + averaged-ordered
        // symmetrization to the Block 10 GPU-substrate pure assembler.
        // Single source of truth shared with the GPU dispatch path.
        let entry_b = block10_pack_base(&entry_base);
        let exit_b = block10_pack_base(&exit_base);
        let entry_d1 = block10_pack_dir(&entry_ext_u);
        let entry_d2 = block10_pack_dir(&entry_ext_v);
        let exit_d1 = block10_pack_dir(&exit_ext_u);
        let exit_d2 = block10_pack_dir(&exit_ext_v);
        let entry_bi_p = block10_pack_bi(&entry_bi);
        let exit_bi_p = block10_pack_bi(&exit_bi);
        let dir_u_vec: Vec<f64> = dir_u.to_vec();
        let dir_v_vec: Vec<f64> = dir_v.to_vec();
        let inputs =
            crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10FourthInputs {
                p,
                qd1_index: primary.qd1,
                qd1,
                w: self.weights[row],
                d: self.event[row],
                dir_u: &dir_u_vec,
                dir_v: &dir_v_vec,
                entry_base: &entry_b,
                exit_base: &exit_b,
                entry_ext_u: &entry_d1,
                entry_ext_v: &entry_d2,
                exit_ext_u: &exit_d1,
                exit_ext_v: &exit_d2,
                entry_bi: &entry_bi_p,
                exit_bi: &exit_bi_p,
            };
        let flat =
            crate::families::survival_marginal_slope_gpu::cpu_oracle_fourth_contraction(&inputs)
                .map_err(|e| format!("block10 fourth contraction: {e}"))?;
        let mut out = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in 0..p {
                out[[u, v]] = flat[u * p + v];
            }
        }
        Ok(out)
    }

    pub(crate) fn row_primary_third_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_third_contracted_exact(row, block_states, dir)
        } else {
            self.row_primary_third_contracted(row, block_states, dir.view())
        }
    }

    pub(crate) fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=6 jet helper.
        let r = self.row_primary_fourth_contracted_batched(row, block_states, dir_u, dir_v)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    pub(crate) fn row_primary_fourth_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_fourth_contracted_exact(row, block_states, dir_u, dir_v)
        } else {
            self.row_primary_fourth_contracted(row, block_states, dir_u.view(), dir_v.view())
        }
    }

    // ── Pullback through design matrices ──────────────────────────────

}

