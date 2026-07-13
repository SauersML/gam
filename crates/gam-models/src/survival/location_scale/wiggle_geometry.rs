use super::*;

/// Exact monotone time-wiggle basis stack at one set of unwarped time indices.
///
/// The derivatives are with respect to the affine base index `h_base`, not
/// clock time.  Keeping all four tables together prevents saved replay from
/// silently substituting a zero third derivative in the event-rate map
/// `h_dot = (1 + B'(h_base) beta_w) h_dot_base`.
#[derive(Clone, Debug)]
pub struct SurvivalLocationScaleTimeWiggleBasisStack {
    pub value: Array2<f64>,
    pub d1: Array2<f64>,
    pub d2: Array2<f64>,
    pub d3: Array2<f64>,
}

/// Row-aligned entry/exit basis authority for exact saved location-scale
/// replay.
#[derive(Clone, Debug)]
pub struct SurvivalLocationScaleTimeWiggleBasisAuthority {
    pub entry: SurvivalLocationScaleTimeWiggleBasisStack,
    pub exit: SurvivalLocationScaleTimeWiggleBasisStack,
}

pub(crate) fn build_survival_location_scale_time_wiggle_basis_stack(
    h_base: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    coefficient_width: usize,
    label: &str,
) -> Result<SurvivalLocationScaleTimeWiggleBasisStack, String> {
    if coefficient_width == 0 {
        return Err(format!(
            "survival location-scale {label} time-wiggle basis requires at least one coefficient"
        ));
    }
    if h_base.is_empty() || h_base.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "survival location-scale {label} time-wiggle base indices must be non-empty and finite"
        ));
    }
    if knots.is_empty() || knots.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "survival location-scale {label} time-wiggle knots must be non-empty and finite"
        ));
    }
    let value = monotone_wiggle_basis_with_derivative_order(h_base, knots, degree, 0)?;
    let d1 = monotone_wiggle_basis_with_derivative_order(h_base, knots, degree, 1)?;
    let d2 = monotone_wiggle_basis_with_derivative_order(h_base, knots, degree, 2)?;
    let d3 = monotone_wiggle_basis_with_derivative_order(h_base, knots, degree, 3)?;
    for (channel, matrix) in [("B", &value), ("B'", &d1), ("B''", &d2), ("B'''", &d3)] {
        if matrix.ncols() != coefficient_width {
            return Err(format!(
                "survival location-scale {label} time-wiggle {channel} has {} columns; fitted beta tail has {coefficient_width}",
                matrix.ncols(),
            ));
        }
        if matrix.nrows() != h_base.len() || matrix.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "survival location-scale {label} time-wiggle {channel} is not a finite {}x{coefficient_width} table",
                h_base.len(),
            ));
        }
    }
    Ok(SurvivalLocationScaleTimeWiggleBasisStack { value, d1, d2, d3 })
}

/// Rebuild the exact fit-time monotone time-wiggle basis tables on saved rows.
///
/// This is basis authority only: callers must compose the affine base channels
/// and `beta_w` inside their order-two row program.  Returning a post-warp
/// Jacobian would omit the score-times-map-Hessian term from observed ALO
/// curvature.
pub fn survival_location_scale_time_wiggle_basis_authority(
    h_entry_base: ArrayView1<'_, f64>,
    h_exit_base: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    coefficient_width: usize,
) -> Result<SurvivalLocationScaleTimeWiggleBasisAuthority, String> {
    if h_entry_base.len() != h_exit_base.len() {
        return Err(format!(
            "survival location-scale saved time-wiggle entry/exit row mismatch: {}/{}",
            h_entry_base.len(),
            h_exit_base.len(),
        ));
    }
    Ok(SurvivalLocationScaleTimeWiggleBasisAuthority {
        entry: build_survival_location_scale_time_wiggle_basis_stack(
            h_entry_base,
            knots,
            degree,
            coefficient_width,
            "entry",
        )?,
        exit: build_survival_location_scale_time_wiggle_basis_stack(
            h_exit_base,
            knots,
            degree,
            coefficient_width,
            "exit",
        )?,
    })
}

#[derive(Clone)]
pub(crate) struct SurvivalWiggleGeometry {
    pub(crate) basis: Array2<f64>,
    pub(crate) basis_d1: Array2<f64>,
    pub(crate) basis_d2: Array2<f64>,
    pub(crate) dq_dq0: Array1<f64>,
    pub(crate) d2q_dq02: Array1<f64>,
    pub(crate) d3q_dq03: Array1<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct SurvivalBaseQScalars {
    pub(crate) eta_t: f64,
    pub(crate) inv_sigma: f64,
    pub(crate) q: f64,
    pub(crate) q_t: f64,
    pub(crate) q_ls: f64,
    pub(crate) q_tl: f64,
    pub(crate) q_ll: f64,
    pub(crate) q_tl_ls: f64,
    pub(crate) q_ll_ls: f64,
}

pub(crate) struct SurvivalDynamicGeometryRowsMut<'a> {
    pub(crate) q_exit: &'a mut [f64],
    pub(crate) q_entry: &'a mut [f64],
    pub(crate) qdot_exit: &'a mut [f64],
    pub(crate) dq_t_exit: &'a mut [f64],
    pub(crate) dq_t_entry: &'a mut [f64],
    pub(crate) dq_ls_exit: &'a mut [f64],
    pub(crate) dq_ls_entry: &'a mut [f64],
    pub(crate) d2q_tls_exit: &'a mut [f64],
    pub(crate) d2q_tls_entry: &'a mut [f64],
    pub(crate) d2q_ls_exit: &'a mut [f64],
    pub(crate) d2q_ls_entry: &'a mut [f64],
    pub(crate) d3q_tls_ls_exit: &'a mut [f64],
    pub(crate) d3q_tls_ls_entry: &'a mut [f64],
    pub(crate) d3q_ls_exit: &'a mut [f64],
    pub(crate) d3q_ls_entry: &'a mut [f64],
    pub(crate) dqdot_t: &'a mut [f64],
    pub(crate) dqdot_ls: &'a mut [f64],
    pub(crate) dqdot_td: &'a mut [f64],
    pub(crate) dqdot_lsd: &'a mut [f64],
}

impl<'a> SurvivalDynamicGeometryRowsMut<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.q_exit.len()
    }

    pub(crate) fn split_at_mut(self, mid: usize) -> (Self, Self) {
        let (q_exit_l, q_exit_r) = self.q_exit.split_at_mut(mid);
        let (q_entry_l, q_entry_r) = self.q_entry.split_at_mut(mid);
        let (qdot_exit_l, qdot_exit_r) = self.qdot_exit.split_at_mut(mid);
        let (dq_t_exit_l, dq_t_exit_r) = self.dq_t_exit.split_at_mut(mid);
        let (dq_t_entry_l, dq_t_entry_r) = self.dq_t_entry.split_at_mut(mid);
        let (dq_ls_exit_l, dq_ls_exit_r) = self.dq_ls_exit.split_at_mut(mid);
        let (dq_ls_entry_l, dq_ls_entry_r) = self.dq_ls_entry.split_at_mut(mid);
        let (d2q_tls_exit_l, d2q_tls_exit_r) = self.d2q_tls_exit.split_at_mut(mid);
        let (d2q_tls_entry_l, d2q_tls_entry_r) = self.d2q_tls_entry.split_at_mut(mid);
        let (d2q_ls_exit_l, d2q_ls_exit_r) = self.d2q_ls_exit.split_at_mut(mid);
        let (d2q_ls_entry_l, d2q_ls_entry_r) = self.d2q_ls_entry.split_at_mut(mid);
        let (d3q_tls_ls_exit_l, d3q_tls_ls_exit_r) = self.d3q_tls_ls_exit.split_at_mut(mid);
        let (d3q_tls_ls_entry_l, d3q_tls_ls_entry_r) = self.d3q_tls_ls_entry.split_at_mut(mid);
        let (d3q_ls_exit_l, d3q_ls_exit_r) = self.d3q_ls_exit.split_at_mut(mid);
        let (d3q_ls_entry_l, d3q_ls_entry_r) = self.d3q_ls_entry.split_at_mut(mid);
        let (dqdot_t_l, dqdot_t_r) = self.dqdot_t.split_at_mut(mid);
        let (dqdot_ls_l, dqdot_ls_r) = self.dqdot_ls.split_at_mut(mid);
        let (dqdot_td_l, dqdot_td_r) = self.dqdot_td.split_at_mut(mid);
        let (dqdot_lsd_l, dqdot_lsd_r) = self.dqdot_lsd.split_at_mut(mid);

        (
            Self {
                q_exit: q_exit_l,
                q_entry: q_entry_l,
                qdot_exit: qdot_exit_l,
                dq_t_exit: dq_t_exit_l,
                dq_t_entry: dq_t_entry_l,
                dq_ls_exit: dq_ls_exit_l,
                dq_ls_entry: dq_ls_entry_l,
                d2q_tls_exit: d2q_tls_exit_l,
                d2q_tls_entry: d2q_tls_entry_l,
                d2q_ls_exit: d2q_ls_exit_l,
                d2q_ls_entry: d2q_ls_entry_l,
                d3q_tls_ls_exit: d3q_tls_ls_exit_l,
                d3q_tls_ls_entry: d3q_tls_ls_entry_l,
                d3q_ls_exit: d3q_ls_exit_l,
                d3q_ls_entry: d3q_ls_entry_l,
                dqdot_t: dqdot_t_l,
                dqdot_ls: dqdot_ls_l,
                dqdot_td: dqdot_td_l,
                dqdot_lsd: dqdot_lsd_l,
            },
            Self {
                q_exit: q_exit_r,
                q_entry: q_entry_r,
                qdot_exit: qdot_exit_r,
                dq_t_exit: dq_t_exit_r,
                dq_t_entry: dq_t_entry_r,
                dq_ls_exit: dq_ls_exit_r,
                dq_ls_entry: dq_ls_entry_r,
                d2q_tls_exit: d2q_tls_exit_r,
                d2q_tls_entry: d2q_tls_entry_r,
                d2q_ls_exit: d2q_ls_exit_r,
                d2q_ls_entry: d2q_ls_entry_r,
                d3q_tls_ls_exit: d3q_tls_ls_exit_r,
                d3q_tls_ls_entry: d3q_tls_ls_entry_r,
                d3q_ls_exit: d3q_ls_exit_r,
                d3q_ls_entry: d3q_ls_entry_r,
                dqdot_t: dqdot_t_r,
                dqdot_ls: dqdot_ls_r,
                dqdot_td: dqdot_td_r,
                dqdot_lsd: dqdot_lsd_r,
            },
        )
    }
}

pub(crate) struct SurvivalDynamicGeometryRowInputs<'a> {
    pub(crate) eta_t_exit: ndarray::ArrayView1<'a, f64>,
    pub(crate) eta_ls_exit: ndarray::ArrayView1<'a, f64>,
    pub(crate) eta_t_entry: ndarray::ArrayView1<'a, f64>,
    pub(crate) eta_ls_entry: ndarray::ArrayView1<'a, f64>,
    pub(crate) eta_t_deriv_exit: &'a Array1<f64>,
    pub(crate) eta_ls_deriv_exit: &'a Array1<f64>,
    pub(crate) wiggle_exit: Option<&'a SurvivalWiggleGeometry>,
    pub(crate) wiggle_entry: Option<&'a SurvivalWiggleGeometry>,
    pub(crate) link_beta: Option<ndarray::ArrayView1<'a, f64>>,
}

pub(crate) const SURVIVAL_DYNAMIC_GEOMETRY_PAR_CHUNK: usize = 1024;

pub(crate) fn fill_survival_dynamic_geometry_rows(
    rows: SurvivalDynamicGeometryRowsMut<'_>,
    row_start: usize,
    inputs: &SurvivalDynamicGeometryRowInputs<'_>,
) {
    let len = rows.len();
    if len <= SURVIVAL_DYNAMIC_GEOMETRY_PAR_CHUNK {
        fill_survival_dynamic_geometry_rows_serial(rows, row_start, inputs);
    } else {
        let mid = len / 2;
        let (left, right) = rows.split_at_mut(mid);
        rayon::join(
            || fill_survival_dynamic_geometry_rows(left, row_start, inputs),
            || fill_survival_dynamic_geometry_rows(right, row_start + mid, inputs),
        );
    }
}

pub(crate) fn fill_survival_dynamic_geometry_rows_serial(
    rows: SurvivalDynamicGeometryRowsMut<'_>,
    row_start: usize,
    inputs: &SurvivalDynamicGeometryRowInputs<'_>,
) {
    for offset in 0..rows.len() {
        let i = row_start + offset;
        let base_exit = survival_base_q_scalars(inputs.eta_t_exit[i], inputs.eta_ls_exit[i]);
        let base_entry = survival_base_q_scalars(inputs.eta_t_entry[i], inputs.eta_ls_entry[i]);
        let exit_dyn = if let (Some(wig), Some(beta_w)) = (inputs.wiggle_exit, inputs.link_beta) {
            compose_survival_dynamic_q(
                base_exit,
                inputs.eta_t_deriv_exit[i],
                inputs.eta_ls_deriv_exit[i],
                wig.basis.row(i).dot(&beta_w),
                wig.dq_dq0[i],
                wig.d2q_dq02[i],
                wig.d3q_dq03[i],
            )
        } else {
            compose_survival_dynamic_q(
                base_exit,
                inputs.eta_t_deriv_exit[i],
                inputs.eta_ls_deriv_exit[i],
                0.0,
                1.0,
                0.0,
                0.0,
            )
        };
        let entry_dyn = if let (Some(wig), Some(beta_w)) = (inputs.wiggle_entry, inputs.link_beta) {
            compose_survival_dynamic_q(
                base_entry,
                0.0,
                0.0,
                wig.basis.row(i).dot(&beta_w),
                wig.dq_dq0[i],
                wig.d2q_dq02[i],
                wig.d3q_dq03[i],
            )
        } else {
            compose_survival_dynamic_q(base_entry, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        };
        rows.q_exit[offset] = exit_dyn.q;
        rows.q_entry[offset] = entry_dyn.q;
        rows.qdot_exit[offset] = exit_dyn.qdot;
        rows.dq_t_exit[offset] = exit_dyn.q_t;
        rows.dq_t_entry[offset] = entry_dyn.q_t;
        rows.dq_ls_exit[offset] = exit_dyn.q_ls;
        rows.dq_ls_entry[offset] = entry_dyn.q_ls;
        rows.d2q_tls_exit[offset] = exit_dyn.q_tl;
        rows.d2q_tls_entry[offset] = entry_dyn.q_tl;
        rows.d2q_ls_exit[offset] = exit_dyn.q_ll;
        rows.d2q_ls_entry[offset] = entry_dyn.q_ll;
        rows.d3q_tls_ls_exit[offset] = exit_dyn.q_tl_ls;
        rows.d3q_tls_ls_entry[offset] = entry_dyn.q_tl_ls;
        rows.d3q_ls_exit[offset] = exit_dyn.q_ll_ls;
        rows.d3q_ls_entry[offset] = entry_dyn.q_ll_ls;
        rows.dqdot_t[offset] = exit_dyn.qdot_t;
        rows.dqdot_ls[offset] = exit_dyn.qdot_ls;
        rows.dqdot_td[offset] = exit_dyn.qdot_td;
        rows.dqdot_lsd[offset] = exit_dyn.qdot_lsd;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SurvivalDynamicQScalars {
    pub(crate) q: f64,
    pub(crate) q_t: f64,
    pub(crate) q_ls: f64,
    pub(crate) q_tl: f64,
    pub(crate) q_ll: f64,
    pub(crate) q_tl_ls: f64,
    pub(crate) q_ll_ls: f64,
    pub(crate) qdot: f64,
    pub(crate) qdot_t: f64,
    pub(crate) qdot_ls: f64,
    pub(crate) qdot_td: f64,
    pub(crate) qdot_lsd: f64,
}

#[derive(Clone)]
pub(crate) struct SurvivalDynamicGeometry {
    pub(crate) h_exit: Array1<f64>,
    pub(crate) h_entry: Array1<f64>,
    pub(crate) hdot_exit: Array1<f64>,
    pub(crate) time_base_derivative_exit: Array1<f64>,
    pub(crate) time_jac_entry: Array2<f64>,
    pub(crate) time_jac_exit: Array2<f64>,
    pub(crate) time_jac_deriv: Array2<f64>,
    pub(crate) time_wiggle_basis_d1_entry: Option<Array2<f64>>,
    pub(crate) time_wiggle_basis_d1_exit: Option<Array2<f64>>,
    pub(crate) time_wiggle_basis_d2_exit: Option<Array2<f64>>,
    pub(crate) time_wiggle_d2_entry: Option<Array1<f64>>,
    pub(crate) time_wiggle_d2_exit: Option<Array1<f64>>,
    pub(crate) time_wiggle_d3_exit: Option<Array1<f64>>,
    pub(crate) eta_ls_exit: Array1<f64>,
    pub(crate) eta_ls_entry: Array1<f64>,
    /// Unwarped AFT index `q0 = -eta_t exp(-eta_ls)` at exit/entry. Link-wiggle
    /// row jets compose the warp from these bases; `q_exit`/`q_entry` below are
    /// the already-warped criterion values.
    pub(crate) q_base_exit: Array1<f64>,
    pub(crate) q_base_entry: Array1<f64>,
    /// Linear-predictor time derivatives before the link wiggle is composed.
    pub(crate) eta_t_deriv_exit: Array1<f64>,
    pub(crate) eta_ls_deriv_exit: Array1<f64>,
    pub(crate) q_exit: Array1<f64>,
    pub(crate) q_entry: Array1<f64>,
    pub(crate) qdot_exit: Array1<f64>,
    pub(crate) inv_sigma_exit: Array1<f64>,
    pub(crate) inv_sigma_entry: Array1<f64>,
    pub(crate) dq_t_exit: Array1<f64>,
    pub(crate) dq_t_entry: Array1<f64>,
    pub(crate) dq_ls_exit: Array1<f64>,
    pub(crate) dq_ls_entry: Array1<f64>,
    pub(crate) d2q_tls_exit: Array1<f64>,
    pub(crate) d2q_tls_entry: Array1<f64>,
    pub(crate) d2q_ls_exit: Array1<f64>,
    pub(crate) d2q_ls_entry: Array1<f64>,
    pub(crate) d3q_tls_ls_exit: Array1<f64>,
    pub(crate) d3q_tls_ls_entry: Array1<f64>,
    pub(crate) d3q_ls_exit: Array1<f64>,
    pub(crate) d3q_ls_entry: Array1<f64>,
    pub(crate) dqdot_t: Array1<f64>,
    pub(crate) dqdot_ls: Array1<f64>,
    pub(crate) dqdot_td: Array1<f64>,
    pub(crate) dqdot_lsd: Array1<f64>,
    pub(crate) wiggle_basis_exit: Option<Array2<f64>>,
    pub(crate) wiggle_basis_entry: Option<Array2<f64>>,
    pub(crate) wiggle_qdot_basis_exit: Option<Array2<f64>>,
    /// #932: the current link-wiggle coefficient vector βw (the last joint
    /// block's β), captured here so the single-source wiggle RowKernel
    /// (`SurvivalLsWiggleRowKernel`) can be built from `(q, dynamic)` alone —
    /// without re-threading `block_states` through the workspace-friendly
    /// `_from_parts` directional entry points that deliberately drop it.
    /// `Some(βw)` exactly when a link-wiggle design is present.
    pub(crate) wiggle_beta: Option<Array1<f64>>,
}

impl SurvivalDynamicGeometry {
    pub(crate) fn validate_precomputed_channels(&self) -> Result<(), String> {
        let n = self.h_exit.len();
        if self.time_base_derivative_exit.len() != n {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "survival dynamic geometry derivative length mismatch: base_derivative={}, rows={n}",
                self.time_base_derivative_exit.len()
            ) }.into());
        }
        for (channel, len) in [
            ("q_base_exit", self.q_base_exit.len()),
            ("q_base_entry", self.q_base_entry.len()),
            ("eta_t_deriv_exit", self.eta_t_deriv_exit.len()),
            ("eta_ls_deriv_exit", self.eta_ls_deriv_exit.len()),
        ] {
            if len != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "survival dynamic geometry {channel} length mismatch: len={len}, expected {n}"
                    ),
                }
                .into());
            }
        }
        if let Some(basis) = self.time_wiggle_basis_d1_entry.as_ref()
            && basis.nrows() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival dynamic geometry wiggle d1 entry row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ),
            }
            .into());
        }
        if let Some(basis) = self.time_wiggle_basis_d1_exit.as_ref()
            && basis.nrows() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival dynamic geometry wiggle d1 exit row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ),
            }
            .into());
        }
        if let Some(basis) = self.time_wiggle_basis_d2_exit.as_ref()
            && basis.nrows() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival dynamic geometry wiggle d2 exit row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ),
            }
            .into());
        }
        if let Some(values) = self.time_wiggle_d2_entry.as_ref()
            && values.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival dynamic geometry wiggle d2 entry length mismatch: len={}, expected {n}",
                    values.len()
                ) }.into());
        }
        if let Some(values) = self.time_wiggle_d2_exit.as_ref()
            && values.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival dynamic geometry wiggle d2 exit length mismatch: len={}, expected {n}",
                    values.len()
                ) }.into());
        }
        if let Some(values) = self.time_wiggle_d3_exit.as_ref()
            && values.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival dynamic geometry wiggle d3 exit length mismatch: len={}, expected {n}",
                    values.len()
                ) }.into());
        }
        Ok(())
    }
}

pub(crate) fn survival_wiggle_basis_with_options(
    q0: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    options: BasisOptions,
) -> Result<Array2<f64>, String> {
    monotone_wiggle_basis_with_derivative_order(q0, knots, degree, options.derivative_order)
}

pub(crate) fn survival_wiggle_third_basis(
    q0: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    monotone_wiggle_basis_with_derivative_order(q0, knots, degree, 3)
}

pub(crate) fn survival_base_q_scalars(eta_t: f64, eta_ls: f64) -> SurvivalBaseQScalars {
    let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    SurvivalBaseQScalars {
        eta_t,
        inv_sigma,
        q: survival_q0_from_eta(eta_t, eta_ls),
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
    }
}

#[inline]
pub(crate) fn survival_q0dot_from_base(
    base: SurvivalBaseQScalars,
    eta_t_deriv: f64,
    eta_ls_deriv: f64,
) -> f64 {
    let local_derivative = base.eta_t.mul_add(eta_ls_deriv, -eta_t_deriv);
    safe_product(base.inv_sigma, local_derivative)
}

pub(crate) fn compose_survival_dynamic_q(
    base: SurvivalBaseQScalars,
    eta_t_deriv: f64,
    eta_ls_deriv: f64,
    wiggle_value: f64,
    dq_dq0: f64,
    d2q_dq02: f64,
    d3q_dq03: f64,
) -> SurvivalDynamicQScalars {
    let a = base.q_t;
    let b = base.q_ls;
    let c = base.q_tl;
    let d = base.q_ll;
    let e = base.q_tl_ls;
    let f = base.q_ll_ls;
    let m1 = dq_dq0;
    let m2 = d2q_dq02;
    let m3 = d3q_dq03;
    let r = survival_q0dot_from_base(base, eta_t_deriv, eta_ls_deriv);
    let r_t = safe_product(c, eta_ls_deriv);
    let r_ls = safe_sum2(safe_product(c, eta_t_deriv), safe_product(d, eta_ls_deriv));
    let q_t = safe_product(m1, a);
    let q_ls = safe_product(m1, b);
    let q_tl = safe_sum2(safe_product(m2, safe_product(a, b)), safe_product(m1, c));
    let q_ll = safe_sum2(safe_product(m2, safe_product(b, b)), safe_product(m1, d));
    let q_tl_ls = safe_sum3(
        safe_product(m3, safe_product(a, safe_product(b, b))),
        safe_product(m2, safe_sum2(safe_product(a, d), 2.0 * safe_product(b, c))),
        safe_product(m1, e),
    );
    let q_ll_ls = safe_sum3(
        safe_product(m3, safe_product(b, safe_product(b, b))),
        safe_product(m2, 3.0 * safe_product(b, d)),
        safe_product(m1, f),
    );

    SurvivalDynamicQScalars {
        q: base.q + wiggle_value,
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
        qdot: safe_product(m1, r),
        qdot_t: safe_sum2(safe_product(m2, safe_product(a, r)), safe_product(m1, r_t)),
        qdot_ls: safe_sum2(safe_product(m2, safe_product(b, r)), safe_product(m1, r_ls)),
        qdot_td: q_t,
        qdot_lsd: q_ls,
    }
}

impl SurvivalLocationScaleFamily {
    pub(crate) fn build_dynamic_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalDynamicGeometry, String> {
        let n = self.n;
        let joint_states = self.validate_joint_states(block_states)?;
        let h_entry_base = joint_states.0.to_owned();
        let h_exit_base = joint_states.1.to_owned();
        let d_base = joint_states.2.to_owned();
        let eta_t_exit_view = joint_states.3;
        let eta_ls_exit = joint_states.4;
        let eta_t_entry_view = joint_states.5;
        let eta_ls_entry = joint_states.6;
        let eta_t_deriv_exit = joint_states.7;
        let eta_ls_deriv_exit = joint_states.8;
        let mut eta_t_deriv_exit = eta_t_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(n));
        let eta_ls_deriv_exit = eta_ls_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(n));
        // σ-scaled log-t AFT location baseline (issue #892). In the rank-1 reduced
        // parametric-AFT regime the time warp is removed (`h ≡ 0`); the `log t`
        // baseline instead shifts the effective location predictor on the σ-scaled
        // `q` channel — `η_t → η_t − log t` (value) with derivative `−1/t` — so the
        // standardized residual is `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`
        // and the event Jacobian gains `qdot = inv_sigma/t → log_g = −η_ls − log t`,
        // the `−log σ` term that identifies σ. Shifting the effective location here
        // (before q0 / the q-row kernel) routes the whole σ coupling through the
        // existing `q`-derivative/Hessian stack — no new time×log_sigma cross-terms.
        let (eta_t_exit, eta_t_entry) = if let Some(loc) = self.location_log_time.as_ref() {
            eta_t_deriv_exit += &loc.deriv_exit;
            (
                &eta_t_exit_view + &loc.value_exit,
                &eta_t_entry_view + &loc.value_entry,
            )
        } else {
            (eta_t_exit_view.to_owned(), eta_t_entry_view.to_owned())
        };
        let inv_sigma_exit = eta_ls_exit.mapv(exp_sigma_inverse_from_eta_scalar);
        let inv_sigma_entry = eta_ls_entry.mapv(exp_sigma_inverse_from_eta_scalar);
        let q0_exit = Array1::from_iter(
            eta_t_exit
                .iter()
                .zip(eta_ls_exit.iter())
                .map(|(&t, &ls)| survival_q0_from_eta(t, ls)),
        );
        let q0_entry = Array1::from_iter(
            eta_t_entry
                .iter()
                .zip(eta_ls_entry.iter())
                .map(|(&t, &ls)| survival_q0_from_eta(t, ls)),
        );
        let time_wiggle_range = self.time_wiggle_range();
        let beta_time_w = if self.time_wiggle_ncols > 0 {
            Some(
                block_states[Self::BLOCK_TIME]
                    .beta
                    .slice(s![time_wiggle_range.start..time_wiggle_range.end]),
            )
        } else {
            None
        };
        let time_wiggle_entry = if let Some(beta_w) = beta_time_w {
            self.time_wiggle_geometry(h_entry_base.view(), beta_w)?
        } else {
            None
        };
        let time_wiggle_exit = if let Some(beta_w) = beta_time_w {
            self.time_wiggle_geometry(h_exit_base.view(), beta_w)?
        } else {
            None
        };
        let beta_w = if self.x_link_wiggle.is_some() {
            Some(block_states[Self::BLOCK_LINK_WIGGLE].beta.view())
        } else {
            None
        };
        let wiggle_exit = if let Some(beta_w) = beta_w {
            self.wiggle_geometry(q0_exit.view(), beta_w)?
        } else {
            None
        };
        let wiggle_entry = if let Some(beta_w) = beta_w {
            self.wiggle_geometry(q0_entry.view(), beta_w)?
        } else {
            None
        };
        if self.x_link_wiggle.is_some() && (wiggle_exit.is_none() || wiggle_entry.is_none()) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival location-scale linkwiggle requires dynamic knot/degree metadata"
                    .to_string(),
            }
            .into());
        }
        if self.time_wiggle_ncols > 0 && (time_wiggle_exit.is_none() || time_wiggle_entry.is_none())
        {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival location-scale timewiggle requires dynamic knot/degree metadata"
                    .to_string(),
            }
            .into());
        }

        let mut h_entry = h_entry_base.clone();
        let mut h_exit = h_exit_base.clone();
        let mut hdot_exit = d_base.clone();
        let mut time_jac_entry = self.x_time_entry.as_ref().clone();
        let mut time_jac_exit = self.x_time_exit.as_ref().clone();
        let mut time_jac_deriv = self.x_time_deriv.as_ref().clone();
        let mut time_wiggle_basis_d1_entry = None;
        let mut time_wiggle_basis_d1_exit = None;
        let mut time_wiggle_basis_d2_exit = None;
        let mut time_wiggle_d2_entry = None;
        let mut time_wiggle_d2_exit = None;
        let mut time_wiggle_d3_exit = None;

        if let (Some(wig_entry), Some(wig_exit), Some(beta_w)) = (
            time_wiggle_entry.as_ref(),
            time_wiggle_exit.as_ref(),
            beta_time_w,
        ) {
            h_entry = &h_entry_base + &fast_av(&wig_entry.basis, &beta_w);
            h_exit = &h_exit_base + &fast_av(&wig_exit.basis, &beta_w);
            hdot_exit = &wig_exit.dq_dq0 * &d_base;
            time_jac_entry = scale_dense_rows(self.x_time_entry.as_ref(), &wig_entry.dq_dq0)?;
            time_jac_exit = scale_dense_rows(self.x_time_exit.as_ref(), &wig_exit.dq_dq0)?;
            time_jac_deriv = scale_dense_rows(
                self.x_time_exit.as_ref(),
                &safe_hadamard_product(&wig_exit.d2q_dq02, &d_base)?,
            )? + &scale_dense_rows(self.x_time_deriv.as_ref(), &wig_exit.dq_dq0)?;
            let wiggle_entry_full = embed_tail_columns(
                &wig_entry.basis,
                time_jac_entry.ncols(),
                time_wiggle_range.clone(),
            )?;
            let wiggle_exit_full = embed_tail_columns(
                &wig_exit.basis,
                time_jac_exit.ncols(),
                time_wiggle_range.clone(),
            )?;
            time_jac_entry
                .slice_mut(s![.., time_wiggle_range.start..time_wiggle_range.end])
                .assign(
                    &wiggle_entry_full
                        .slice(s![.., time_wiggle_range.start..time_wiggle_range.end]),
                );
            time_jac_exit
                .slice_mut(s![.., time_wiggle_range.start..time_wiggle_range.end])
                .assign(
                    &wiggle_exit_full.slice(s![.., time_wiggle_range.start..time_wiggle_range.end]),
                );
            let wiggle_qdot = scale_dense_rows(&wig_exit.basis_d1, &d_base)?;
            let wiggle_qdot_full = embed_tail_columns(
                &wiggle_qdot,
                time_jac_deriv.ncols(),
                time_wiggle_range.clone(),
            )?;
            time_jac_deriv
                .slice_mut(s![.., time_wiggle_range.start..time_wiggle_range.end])
                .assign(
                    &wiggle_qdot_full.slice(s![.., time_wiggle_range.start..time_wiggle_range.end]),
                );
            time_wiggle_basis_d1_entry = Some(wig_entry.basis_d1.clone());
            time_wiggle_basis_d1_exit = Some(wig_exit.basis_d1.clone());
            time_wiggle_basis_d2_exit = Some(wig_exit.basis_d2.clone());
            time_wiggle_d2_entry = Some(wig_entry.d2q_dq02.clone());
            time_wiggle_d2_exit = Some(wig_exit.d2q_dq02.clone());
            time_wiggle_d3_exit = Some(wig_exit.d3q_dq03.clone());
        }

        let mut q_exit = Array1::<f64>::zeros(n);
        let mut q_entry = Array1::<f64>::zeros(n);
        let mut qdot_exit = Array1::<f64>::zeros(n);
        let mut dq_t_exit = Array1::<f64>::zeros(n);
        let mut dq_t_entry = Array1::<f64>::zeros(n);
        let mut dq_ls_exit = Array1::<f64>::zeros(n);
        let mut dq_ls_entry = Array1::<f64>::zeros(n);
        let mut d2q_tls_exit = Array1::<f64>::zeros(n);
        let mut d2q_tls_entry = Array1::<f64>::zeros(n);
        let mut d2q_ls_exit = Array1::<f64>::zeros(n);
        let mut d2q_ls_entry = Array1::<f64>::zeros(n);
        let mut d3q_tls_ls_exit = Array1::<f64>::zeros(n);
        let mut d3q_tls_ls_entry = Array1::<f64>::zeros(n);
        let mut d3q_ls_exit = Array1::<f64>::zeros(n);
        let mut d3q_ls_entry = Array1::<f64>::zeros(n);
        let mut dqdot_t = Array1::<f64>::zeros(n);
        let mut dqdot_ls = Array1::<f64>::zeros(n);
        let mut dqdot_td = Array1::<f64>::zeros(n);
        let mut dqdot_lsd = Array1::<f64>::zeros(n);

        let dynamic_row_inputs = SurvivalDynamicGeometryRowInputs {
            eta_t_exit: eta_t_exit.view(),
            eta_ls_exit,
            eta_t_entry: eta_t_entry.view(),
            eta_ls_entry,
            eta_t_deriv_exit: &eta_t_deriv_exit,
            eta_ls_deriv_exit: &eta_ls_deriv_exit,
            wiggle_exit: wiggle_exit.as_ref(),
            wiggle_entry: wiggle_entry.as_ref(),
            link_beta: beta_w,
        };
        let dynamic_rows = SurvivalDynamicGeometryRowsMut {
            q_exit: q_exit.as_slice_mut().expect("q_exit must be contiguous"),
            q_entry: q_entry.as_slice_mut().expect("q_entry must be contiguous"),
            qdot_exit: qdot_exit
                .as_slice_mut()
                .expect("qdot_exit must be contiguous"),
            dq_t_exit: dq_t_exit
                .as_slice_mut()
                .expect("dq_t_exit must be contiguous"),
            dq_t_entry: dq_t_entry
                .as_slice_mut()
                .expect("dq_t_entry must be contiguous"),
            dq_ls_exit: dq_ls_exit
                .as_slice_mut()
                .expect("dq_ls_exit must be contiguous"),
            dq_ls_entry: dq_ls_entry
                .as_slice_mut()
                .expect("dq_ls_entry must be contiguous"),
            d2q_tls_exit: d2q_tls_exit
                .as_slice_mut()
                .expect("d2q_tls_exit must be contiguous"),
            d2q_tls_entry: d2q_tls_entry
                .as_slice_mut()
                .expect("d2q_tls_entry must be contiguous"),
            d2q_ls_exit: d2q_ls_exit
                .as_slice_mut()
                .expect("d2q_ls_exit must be contiguous"),
            d2q_ls_entry: d2q_ls_entry
                .as_slice_mut()
                .expect("d2q_ls_entry must be contiguous"),
            d3q_tls_ls_exit: d3q_tls_ls_exit
                .as_slice_mut()
                .expect("d3q_tls_ls_exit must be contiguous"),
            d3q_tls_ls_entry: d3q_tls_ls_entry
                .as_slice_mut()
                .expect("d3q_tls_ls_entry must be contiguous"),
            d3q_ls_exit: d3q_ls_exit
                .as_slice_mut()
                .expect("d3q_ls_exit must be contiguous"),
            d3q_ls_entry: d3q_ls_entry
                .as_slice_mut()
                .expect("d3q_ls_entry must be contiguous"),
            dqdot_t: dqdot_t.as_slice_mut().expect("dqdot_t must be contiguous"),
            dqdot_ls: dqdot_ls
                .as_slice_mut()
                .expect("dqdot_ls must be contiguous"),
            dqdot_td: dqdot_td
                .as_slice_mut()
                .expect("dqdot_td must be contiguous"),
            dqdot_lsd: dqdot_lsd
                .as_slice_mut()
                .expect("dqdot_lsd must be contiguous"),
        };
        fill_survival_dynamic_geometry_rows(dynamic_rows, 0, &dynamic_row_inputs);

        let wiggle_qdot_basis_exit = wiggle_exit.as_ref().map(|wig| {
            use rayon::prelude::*;

            let mut out = wig.basis_d1.clone();
            let r = Array1::from_vec(
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let base_exit = survival_base_q_scalars(eta_t_exit[i], eta_ls_exit[i]);
                        survival_q0dot_from_base(
                            base_exit,
                            eta_t_deriv_exit[i],
                            eta_ls_deriv_exit[i],
                        )
                    })
                    .collect(),
            );
            let ncols = out.ncols();
            out.as_slice_mut()
                .expect("wiggle qdot basis must be contiguous")
                .par_chunks_mut(ncols)
                .enumerate()
                .for_each(|(i, row)| {
                    for value in row {
                        *value *= r[i];
                    }
                });
            out
        });

        let dynamic = SurvivalDynamicGeometry {
            h_exit,
            h_entry,
            hdot_exit,
            time_base_derivative_exit: d_base,
            time_jac_entry,
            time_jac_exit,
            time_jac_deriv,
            time_wiggle_basis_d1_entry,
            time_wiggle_basis_d1_exit,
            time_wiggle_basis_d2_exit,
            time_wiggle_d2_entry,
            time_wiggle_d2_exit,
            time_wiggle_d3_exit,
            eta_ls_exit: eta_ls_exit.to_owned(),
            eta_ls_entry: eta_ls_entry.to_owned(),
            q_base_exit: q0_exit,
            q_base_entry: q0_entry,
            eta_t_deriv_exit,
            eta_ls_deriv_exit,
            q_exit,
            q_entry,
            qdot_exit,
            inv_sigma_exit,
            inv_sigma_entry,
            dq_t_exit,
            dq_t_entry,
            dq_ls_exit,
            dq_ls_entry,
            d2q_tls_exit,
            d2q_tls_entry,
            d2q_ls_exit,
            d2q_ls_entry,
            d3q_tls_ls_exit,
            d3q_tls_ls_entry,
            d3q_ls_exit,
            d3q_ls_entry,
            dqdot_t,
            dqdot_ls,
            dqdot_td,
            dqdot_lsd,
            wiggle_basis_exit: wiggle_exit.as_ref().map(|w| w.basis.clone()),
            wiggle_basis_entry: wiggle_entry.as_ref().map(|w| w.basis.clone()),
            wiggle_qdot_basis_exit,
            wiggle_beta: if self.x_link_wiggle.is_some() {
                Some(block_states[Self::BLOCK_LINK_WIGGLE].beta.to_owned())
            } else {
                None
            },
        };
        dynamic.validate_precomputed_channels()?;
        Ok(dynamic)
    }
}
