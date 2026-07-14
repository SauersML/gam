//! Time-wiggle dynamic-q geometry: the monotone wiggle basis geometry, the
//! first-/second-order dynamic-q row values and gradients, and the
//! marginal psi-row lift through the time-wiggle deformation.

use super::*;
use gam_math::nested_dual::JetField;

/// One row of each time-wiggle basis derivative from `B` through `B'''''`.
///
/// Keeping the six rows separate avoids allocating a per-row array of stacks
/// in the hot path. The generic scalar constructor consumes `[B..B'''']` for
/// `q` and the shifted `[B'..B''''']` stack for `dq/dh`.
pub(crate) struct TimewiggleBasisDerivativeRows<'a> {
    basis: ArrayView1<'a, f64>,
    basis_d1: ArrayView1<'a, f64>,
    basis_d2: ArrayView1<'a, f64>,
    basis_d3: ArrayView1<'a, f64>,
    basis_d4: ArrayView1<'a, f64>,
    basis_d5: ArrayView1<'a, f64>,
}

impl<'a> TimewiggleBasisDerivativeRows<'a> {
    pub(crate) fn new(
        basis: ArrayView1<'a, f64>,
        basis_d1: ArrayView1<'a, f64>,
        basis_d2: ArrayView1<'a, f64>,
        basis_d3: ArrayView1<'a, f64>,
        basis_d4: ArrayView1<'a, f64>,
        basis_d5: ArrayView1<'a, f64>,
    ) -> Self {
        Self {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            basis_d4,
            basis_d5,
        }
    }

    pub(crate) fn from_geometry(
        geometry: &'a SurvivalTimeWiggleGeometry,
        basis_d5: &'a Array2<f64>,
        row: usize,
    ) -> Self {
        Self::new(
            geometry.basis.row(row),
            geometry.basis_d1.row(row),
            geometry.basis_d2.row(row),
            geometry.basis_d3.row(row),
            geometry.basis_d4.row(row),
            basis_d5.row(row),
        )
    }

    fn validate_width(&self, width: usize, endpoint: &str) -> Result<(), String> {
        let widths = [
            self.basis.len(),
            self.basis_d1.len(),
            self.basis_d2.len(),
            self.basis_d3.len(),
            self.basis_d4.len(),
            self.basis_d5.len(),
        ];
        if widths.iter().any(|&actual| actual != width) {
            return Err(format!(
                "survival marginal-slope {endpoint} timewiggle B..B5 widths {widths:?} must all equal scalar coefficient width {width}"
            ));
        }
        Ok(())
    }

    #[inline]
    fn basis_stack(&self, coefficient: usize) -> [f64; 5] {
        [
            self.basis[coefficient],
            self.basis_d1[coefficient],
            self.basis_d2[coefficient],
            self.basis_d3[coefficient],
            self.basis_d4[coefficient],
        ]
    }

    #[inline]
    fn derivative_stack(&self, coefficient: usize) -> [f64; 5] {
        [
            self.basis_d1[coefficient],
            self.basis_d2[coefficient],
            self.basis_d3[coefficient],
            self.basis_d4[coefficient],
            self.basis_d5[coefficient],
        ]
    }
}

/// Bitwise current values anchoring the generic time-wiggle scalar program.
/// Centering every nonconstant term at its own value preserves the existing
/// f64 dot-product result exactly while leaving all derivative channels intact.
#[derive(Clone, Copy)]
pub(crate) struct TimewiggleQBaseValues {
    pub(crate) q0: f64,
    pub(crate) q1: f64,
    pub(crate) dq1_dh1: f64,
}

/// Generic time-wiggle row coordinates.
pub(crate) struct TimewiggleScalarQ<J> {
    pub(crate) q0: J,
    pub(crate) q1: J,
    pub(crate) qd1: J,
}

#[inline]
fn constant_like<J: JetField>(anchor: &J, value: f64) -> J {
    anchor.compose_unary([value, 0.0, 0.0, 0.0, 0.0])
}

#[inline]
fn centered<J: JetField>(value: &J) -> J {
    value.sub(&constant_like(value, value.value()))
}

fn timewiggle_q_at_endpoint<J: JetField>(
    h: &J,
    beta_w: &[J],
    basis: &TimewiggleBasisDerivativeRows<'_>,
    base_value: f64,
) -> J {
    let mut q = constant_like(h, base_value).add(&centered(h));
    for (coefficient, beta) in beta_w.iter().enumerate() {
        let basis_value = h.compose_unary(basis.basis_stack(coefficient));
        q = q.add(&centered(&beta.mul(&basis_value)));
    }
    q
}

/// Evaluate `q0 = h0 + B(h0) beta_w`, `q1 = h1 + B(h1) beta_w`, and
/// `qd1 = (1 + B'(h1) beta_w) d_raw` over one scalar algebra.
///
/// `h0`, `h1`, `d_raw`, and every wiggle coefficient share the same scalar, so
/// `Dual2<Order2<_>>` and `Dual2<OneSeed<_>>` carry baseline-family directions
/// and beta-wiggle primary channels through the identical program. The supplied
/// base values are the current f64 path's already-computed results; centering
/// makes those values bit-identical without changing any derivative.
pub(crate) fn timewiggle_q_from_basis_derivative_rows<J: JetField>(
    h0: &J,
    h1: &J,
    d_raw: &J,
    beta_w: &[J],
    entry_basis: &TimewiggleBasisDerivativeRows<'_>,
    exit_basis: &TimewiggleBasisDerivativeRows<'_>,
    base_values: TimewiggleQBaseValues,
) -> Result<TimewiggleScalarQ<J>, String> {
    entry_basis.validate_width(beta_w.len(), "entry")?;
    exit_basis.validate_width(beta_w.len(), "exit")?;

    let q0 = timewiggle_q_at_endpoint(h0, beta_w, entry_basis, base_values.q0);
    let q1 = timewiggle_q_at_endpoint(h1, beta_w, exit_basis, base_values.q1);
    let mut dq1_dh1 = constant_like(h1, base_values.dq1_dh1);
    for (coefficient, beta) in beta_w.iter().enumerate() {
        let basis_derivative = h1.compose_unary(exit_basis.derivative_stack(coefficient));
        dq1_dh1 = dq1_dh1.add(&centered(&beta.mul(&basis_derivative)));
    }

    Ok(TimewiggleScalarQ {
        q0,
        q1,
        qd1: dq1_dh1.mul(d_raw),
    })
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn flex_timewiggle_active(&self) -> bool {
        self.time_wiggle_ncols > 0
    }

    pub(crate) fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.design_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        (p_total - p_w)..p_total
    }

    pub(crate) fn time_wiggle_first_order_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleFirstOrderGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle basis/beta mismatch: B..B''={},{},{} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        Ok(Some(SurvivalTimeWiggleFirstOrderGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
        }))
    }

    pub(crate) fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleGeometry>, String> {
        Ok(self
            .time_wiggle_geometry_with_basis_d5(h0, beta_w)?
            .map(|(geometry, _basis_d5)| geometry))
    }

    /// Full geometry plus the per-coefficient fifth basis derivative retained
    /// for the generic scalar q program. The public current-geometry wrapper
    /// discards `basis_d5` after aggregating it, preserving its existing type.
    fn time_wiggle_geometry_with_basis_d5(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<(SurvivalTimeWiggleGeometry, Array2<f64>)>, String> {
        let first = self.time_wiggle_first_order_geometry(h0, beta_w)?;
        let Some(first) = first else {
            return Ok(None);
        };
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        let basis_d4 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 4)?;
        let basis_d5 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 5)?;
        if basis_d3.ncols() != beta_w.len()
            || basis_d4.ncols() != beta_w.len()
            || basis_d5.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle high-order basis/beta mismatch: B'''..B'''''={},{},{} betaw={}",
                    basis_d3.ncols(),
                    basis_d4.ncols(),
                    basis_d5.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        let d4q_dq04 = fast_av(&basis_d4, &beta_w);
        let d5q_dq05 = fast_av(&basis_d5, &beta_w);
        Ok(Some((
            SurvivalTimeWiggleGeometry {
                basis: first.basis,
                basis_d1: first.basis_d1,
                basis_d2: first.basis_d2,
                basis_d3,
                basis_d4,
                dq_dq0: first.dq_dq0,
                d2q_dq02: first.d2q_dq02,
                d3q_dq03,
                d4q_dq04,
                d5q_dq05,
            },
            basis_d5,
        )))
    }

    pub(crate) fn row_dynamic_q_values(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowValues, String> {
        let beta_time = &block_states[0].beta;
        if !self.flex_timewiggle_active() {
            return Ok(SurvivalMarginalSlopeDynamicRowValues {
                q0: self.design_entry.dot_row(row, beta_time)
                    + self.offset_entry[row]
                    + block_states[1].eta[row],
                q1: self.design_exit.dot_row(row, beta_time)
                    + self.offset_exit[row]
                    + block_states[1].eta[row],
                qd1: self.design_derivative_exit.dot_row(row, beta_time)
                    + self.derivative_offset_exit[row],
            });
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at exit"
                    .to_string()
            })?;

        Ok(SurvivalMarginalSlopeDynamicRowValues {
            q0: h0 + entry_geom.basis.row(0).dot(&beta_time_w),
            q1: h1 + exit_geom.basis.row(0).dot(&beta_time_w),
            qd1: exit_geom.dq_dq0[0] * d_raw,
        })
    }

    pub(crate) fn row_dynamic_q_gradient(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowGradient, String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        let mut out = SurvivalMarginalSlopeDynamicRowGradient {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(p_time),
            dq1_time: Array1::zeros(p_time),
            dqd1_time: Array1::zeros(p_time),
            dq0_marginal: Array1::zeros(p_marginal),
            dq1_marginal: Array1::zeros(p_marginal),
            dqd1_marginal: Array1::zeros(p_marginal),
        };

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(out);
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_row = if p_marginal > 0 {
            let marginal_chunk = self
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
            Some(marginal_chunk.row(0).to_owned())
        } else {
            None
        };

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
            }
        }

        Ok(out)
    }

    pub(crate) fn row_dynamic_q_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRow, String> {
        let mut out = SurvivalMarginalSlopeDynamicRow::empty_workspace();
        self.row_dynamic_q_geometry_into(row, block_states, &mut out)?;
        Ok(out)
    }

    /// Pooled-buffer variant of [`row_dynamic_q_geometry`]. Resizes and
    /// zero-fills `out` in place, then writes the same row geometry into the
    /// caller-owned workspace. Used by hot rayon `try_fold` accumulators that
    /// thread one `SurvivalMarginalSlopeDynamicRow` per worker thread to
    /// eliminate the ~250 KB-per-call allocator traffic the fresh-allocation
    /// path would incur at large-scale `n`.
    pub(crate) fn row_dynamic_q_geometry_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        out: &mut SurvivalMarginalSlopeDynamicRow,
    ) -> Result<(), String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        out.reset(p_time, p_marginal);

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
            // Perf (#large-scale): assign the design rows directly from the chunk
            // views into the reused `out` buffers. The previous `.to_owned()`
            // round-trip allocated three fresh `Array1<f64>` per row only to
            // immediately copy them in — removing it drops O(n·p_time) heap
            // traffic with bit-identical values.
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(());
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
        // Perf (#large-scale): hold the base/marginal design rows as borrowed views
        // into the chunk storage rather than `.to_owned()` copies. Every use
        // below is either a `.dot(...)` or scalar indexing, both of which work
        // directly on `ArrayView1`, so this is bit-identical while removing the
        // per-row Array1 allocations.
        let entry_row_view = entry_chunk.row(0);
        let exit_row_view = exit_chunk.row(0);
        let deriv_row_view = deriv_chunk.row(0);
        let x_entry_base = entry_row_view.slice(s![..p_base]);
        let x_exit_base = exit_row_view.slice(s![..p_base]);
        let x_deriv_base = deriv_row_view.slice(s![..p_base]);
        let marginal_chunk = if p_marginal > 0 {
            Some(
                self.marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?,
            )
        } else {
            None
        };
        let marginal_row = marginal_chunk.as_ref().map(|chunk| chunk.row(0));

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let (entry_geom, entry_basis_d5) = self
            .time_wiggle_geometry_with_basis_d5(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at entry"
                    .to_string()
            })?;
        let (exit_geom, exit_basis_d5) = self
            .time_wiggle_geometry_with_basis_d5(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at exit"
                    .to_string()
            })?;

        let beta_time_w_scalars = beta_time_w.as_slice().ok_or_else(|| {
            "survival marginal-slope timewiggle coefficient view must be contiguous".to_string()
        })?;
        let entry_basis =
            TimewiggleBasisDerivativeRows::from_geometry(&entry_geom, &entry_basis_d5, 0);
        let exit_basis =
            TimewiggleBasisDerivativeRows::from_geometry(&exit_geom, &exit_basis_d5, 0);
        let scalar_q = timewiggle_q_from_basis_derivative_rows(
            &h0,
            &h1,
            &d_raw,
            beta_time_w_scalars,
            &entry_basis,
            &exit_basis,
            TimewiggleQBaseValues {
                q0: h0 + entry_geom.basis.row(0).dot(&beta_time_w),
                q1: h1 + exit_geom.basis.row(0).dot(&beta_time_w),
                dq1_dh1: exit_geom.dq_dq0[0],
            },
        )?;
        out.q0 = scalar_q.q0;
        out.q1 = scalar_q.q1;
        out.qd1 = scalar_q.qd1;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
            for k in 0..p_base {
                out.d2q0_time_time[[j, k]] =
                    entry_geom.d2q_dq02[0] * x_entry_base[j] * x_entry_base[k];
                out.d2q1_time_time[[j, k]] =
                    exit_geom.d2q_dq02[0] * x_exit_base[j] * x_exit_base[k];
                out.d2qd1_time_time[[j, k]] =
                    exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j] * x_exit_base[k]
                        + exit_geom.d2q_dq02[0]
                            * (x_exit_base[j] * x_deriv_base[k] + x_deriv_base[j] * x_exit_base[k]);
            }
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
            for j in 0..p_base {
                let q0_tw = entry_geom.basis_d1[[0, local_idx]] * x_entry_base[j];
                let q1_tw = exit_geom.basis_d1[[0, local_idx]] * x_exit_base[j];
                out.d2q0_time_time[[j, coeff_idx]] = q0_tw;
                out.d2q0_time_time[[coeff_idx, j]] = q0_tw;
                out.d2q1_time_time[[j, coeff_idx]] = q1_tw;
                out.d2q1_time_time[[coeff_idx, j]] = q1_tw;
                let qd1_cross = exit_geom.basis_d2[[0, local_idx]] * d_raw * x_exit_base[j]
                    + exit_geom.basis_d1[[0, local_idx]] * x_deriv_base[j];
                out.d2qd1_time_time[[j, coeff_idx]] = qd1_cross;
                out.d2qd1_time_time[[coeff_idx, j]] = qd1_cross;
            }
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
                for k in 0..p_marginal {
                    out.d2q0_marginal_marginal[[j, k]] =
                        entry_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2q1_marginal_marginal[[j, k]] =
                        exit_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2qd1_marginal_marginal[[j, k]] =
                        exit_geom.d3q_dq03[0] * d_raw * marginal_row[j] * marginal_row[k];
                }
                for k in 0..p_base {
                    out.d2q0_time_marginal[[k, j]] =
                        entry_geom.d2q_dq02[0] * x_entry_base[k] * marginal_row[j];
                    out.d2q1_time_marginal[[k, j]] =
                        exit_geom.d2q_dq02[0] * x_exit_base[k] * marginal_row[j];
                    out.d2qd1_time_marginal[[k, j]] =
                        exit_geom.d3q_dq03[0] * d_raw * x_exit_base[k] * marginal_row[j]
                            + exit_geom.d2q_dq02[0] * x_deriv_base[k] * marginal_row[j];
                }
                for local_idx in 0..time_tail.len() {
                    let coeff_idx = time_tail.start + local_idx;
                    out.d2q0_time_marginal[[coeff_idx, j]] =
                        entry_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2q1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2qd1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d2[[0, local_idx]] * d_raw * marginal_row[j];
                }
            }
        }

        Ok(())
    }

    pub(crate) fn timewiggle_marginal_psi_row_lift(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_layout: Option<&FlexPrimarySlices>,
        psi_row: &Array1<f64>,
        beta_marginal: &Array1<f64>,
    ) -> Result<TimewiggleMarginalPsiRowLift, String> {
        let beta_time = &block_states[0].beta;
        let p_time = beta_time.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_chunk = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift marginal_design: {e}"))?;
        let marginal_row = marginal_chunk.row(0).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time.slice(s![..p_base]))
            + self.offset_entry[row]
            + base_marginal;
        let h1 =
            x_exit_base.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + base_marginal;
        let d_raw =
            x_deriv_base.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "missing entry timewiggle geometry for marginal psi lift".to_string())?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "missing exit timewiggle geometry for marginal psi lift".to_string())?;

        let mu = psi_row.dot(beta_marginal);
        let mut dir =
            Array1::<f64>::zeros(primary_layout.map_or(N_PRIMARY, |primary| primary.total));
        let q0_idx = primary_layout.map_or(0, |primary| primary.q0);
        let q1_idx = primary_layout.map_or(1, |primary| primary.q1);
        let qd1_idx = primary_layout.map_or(2, |primary| primary.qd1);
        dir[q0_idx] = entry_geom.dq_dq0[0] * mu;
        dir[q1_idx] = exit_geom.dq_dq0[0] * mu;
        dir[qd1_idx] = exit_geom.d2q_dq02[0] * d_raw * mu;

        let mut u_q0_time = Array1::<f64>::zeros(p_time);
        let mut u_q1_time = Array1::<f64>::zeros(p_time);
        let mut u_qd1_time = Array1::<f64>::zeros(p_time);
        for j in 0..p_base {
            u_q0_time[j] = entry_geom.d2q_dq02[0] * mu * x_entry_base[j];
            u_q1_time[j] = exit_geom.d2q_dq02[0] * mu * x_exit_base[j];
            u_qd1_time[j] = mu
                * (exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j]
                    + exit_geom.d2q_dq02[0] * x_deriv_base[j]);
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            u_q0_time[coeff_idx] = entry_geom.basis_d1[[0, local_idx]] * mu;
            u_q1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * mu;
            u_qd1_time[coeff_idx] = exit_geom.basis_d2[[0, local_idx]] * d_raw * mu;
        }

        let u_q0_marginal =
            psi_row * entry_geom.dq_dq0[0] + &marginal_row * (entry_geom.d2q_dq02[0] * mu);
        let u_q1_marginal =
            psi_row * exit_geom.dq_dq0[0] + &marginal_row * (exit_geom.d2q_dq02[0] * mu);
        let u_qd1_marginal = psi_row * (exit_geom.d2q_dq02[0] * d_raw)
            + &marginal_row * (exit_geom.d3q_dq03[0] * d_raw * mu);

        Ok(TimewiggleMarginalPsiRowLift {
            dir,
            u_q0_time,
            u_q1_time,
            u_qd1_time,
            u_q0_marginal,
            u_q1_marginal,
            u_qd1_marginal,
            x_entry_base,
            x_exit_base,
            x_deriv_base,
            marginal_row,
            entry_basis_d1: entry_geom.basis_d1.row(0).to_owned(),
            entry_basis_d2: entry_geom.basis_d2.row(0).to_owned(),
            exit_basis_d1: exit_geom.basis_d1.row(0).to_owned(),
            exit_basis_d2: exit_geom.basis_d2.row(0).to_owned(),
            exit_basis_d3: exit_geom.basis_d3.row(0).to_owned(),
            entry_m2: entry_geom.d2q_dq02[0],
            entry_m3: entry_geom.d3q_dq03[0],
            exit_m2: exit_geom.d2q_dq02[0],
            exit_m3: exit_geom.d3q_dq03[0],
            exit_m4: exit_geom.d4q_dq04[0],
            d_raw,
            mu,
            psi_row: psi_row.clone(),
        })
    }
}

#[cfg(test)]
mod generic_scalar_q_tests {
    use super::*;
    use gam_math::jet_scalar::{JetScalar, OneSeed, Order2};
    use gam_math::nested_dual::Dual2;
    use ndarray::Array1;

    // Two analytic basis functions, B0(h)=h^2 and B1(h)=exp(h), represented by
    // their exact B..B5 rows. This isolates the scalar algebra from spline
    // construction and contains no finite-difference oracle.
    fn analytic_basis_derivatives(h: f64) -> [Array1<f64>; 6] {
        let exponential = h.exp();
        [
            Array1::from_vec(vec![h * h, exponential]),
            Array1::from_vec(vec![2.0 * h, exponential]),
            Array1::from_vec(vec![2.0, exponential]),
            Array1::from_vec(vec![0.0, exponential]),
            Array1::from_vec(vec![0.0, exponential]),
            Array1::from_vec(vec![0.0, exponential]),
        ]
    }

    fn derivative_rows(derivatives: &[Array1<f64>; 6]) -> TimewiggleBasisDerivativeRows<'_> {
        TimewiggleBasisDerivativeRows::new(
            derivatives[0].view(),
            derivatives[1].view(),
            derivatives[2].view(),
            derivatives[3].view(),
            derivatives[4].view(),
            derivatives[5].view(),
        )
    }

    fn analytic_base_values(h0: f64, h1: f64, beta: [f64; 2]) -> TimewiggleQBaseValues {
        TimewiggleQBaseValues {
            q0: h0 + beta[0] * h0 * h0 + beta[1] * h0.exp(),
            q1: h1 + beta[0] * h1 * h1 + beta[1] * h1.exp(),
            dq1_dh1: 1.0 + 2.0 * beta[0] * h1 + beta[1] * h1.exp(),
        }
    }

    fn analytic_q<const K: usize, J: JetScalar<K>>(
        h0: &J,
        h1: &J,
        d_raw: &J,
        beta: &[J; 2],
    ) -> TimewiggleScalarQ<J> {
        let entry_exponential = h0.exp();
        let exit_exponential = h1.exp();
        let q0 = h0
            .add(&beta[0].mul(&h0.mul(h0)))
            .add(&beta[1].mul(&entry_exponential));
        let q1 = h1
            .add(&beta[0].mul(&h1.mul(h1)))
            .add(&beta[1].mul(&exit_exponential));
        let dq1_dh1 = J::constant(1.0)
            .add(&beta[0].mul(&h1.scale(2.0)))
            .add(&beta[1].mul(&exit_exponential));
        TimewiggleScalarQ {
            q0,
            q1,
            qd1: dq1_dh1.mul(d_raw),
        }
    }

    fn assert_close(actual: f64, expected: f64, channel: &str) {
        let tolerance = 1024.0 * f64::EPSILON * (1.0 + actual.abs().max(expected.abs()));
        assert!(
            (actual - expected).abs() <= tolerance,
            "{channel}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn assert_order2<const K: usize>(
        actual: &Order2<K>,
        expected: &Order2<K>,
        prefix: &str,
    ) {
        assert_close(actual.value(), expected.value(), &format!("{prefix}.value"));
        for a in 0..K {
            assert_close(actual.g()[a], expected.g()[a], &format!("{prefix}.g[{a}]"));
            for b in 0..K {
                assert_close(
                    actual.h()[a][b],
                    expected.h()[a][b],
                    &format!("{prefix}.h[{a}][{b}]"),
                );
            }
        }
    }

    fn assert_dual_order2<const K: usize>(
        actual: &Dual2<Order2<K>>,
        expected: &Dual2<Order2<K>>,
        prefix: &str,
    ) {
        assert_order2(&actual.v, &expected.v, &format!("{prefix}.v"));
        assert_order2(&actual.g, &expected.g, &format!("{prefix}.g"));
        assert_order2(&actual.h, &expected.h, &format!("{prefix}.h"));
    }

    fn assert_oneseed<const K: usize>(
        actual: &OneSeed<K>,
        expected: &OneSeed<K>,
        prefix: &str,
    ) {
        assert_order2(&actual.base, &expected.base, &format!("{prefix}.base"));
        assert_order2(&actual.eps, &expected.eps, &format!("{prefix}.eps"));
    }

    fn assert_dual_oneseed<const K: usize>(
        actual: &Dual2<OneSeed<K>>,
        expected: &Dual2<OneSeed<K>>,
        prefix: &str,
    ) {
        assert_oneseed(&actual.v, &expected.v, &format!("{prefix}.v"));
        assert_oneseed(&actual.g, &expected.g, &format!("{prefix}.g"));
        assert_oneseed(&actual.h, &expected.h, &format!("{prefix}.h"));
    }

    fn order2_family_variable<const K: usize>(
        value: f64,
        axis: usize,
        family_first: f64,
        family_second: f64,
    ) -> Dual2<Order2<K>> {
        Dual2 {
            v: Order2::variable(value, axis),
            g: Order2::constant(family_first),
            h: Order2::constant(family_second),
        }
    }

    fn oneseed_family_variable<const K: usize>(
        value: f64,
        axis: usize,
        direction: f64,
        family_first: f64,
        family_second: f64,
    ) -> Dual2<OneSeed<K>> {
        Dual2 {
            v: OneSeed::seed_direction(value, axis, direction),
            g: OneSeed::constant(family_first),
            h: OneSeed::constant(family_second),
        }
    }

    #[test]
    fn f64_timewiggle_q_preserves_current_value_bits() {
        let values = [0.35, -0.4, 1.2, 0.18, 0.07];
        let beta = [values[3], values[4]];
        let entry_derivatives = analytic_basis_derivatives(values[0]);
        let exit_derivatives = analytic_basis_derivatives(values[1]);
        let entry_rows = derivative_rows(&entry_derivatives);
        let exit_rows = derivative_rows(&exit_derivatives);
        let base_values = analytic_base_values(values[0], values[1], beta);

        let actual = timewiggle_q_from_basis_derivative_rows(
            &values[0],
            &values[1],
            &values[2],
            &beta,
            &entry_rows,
            &exit_rows,
            base_values,
        )
        .expect("analytic B..B5 rows have matching widths");

        assert_eq!(actual.q0.to_bits(), base_values.q0.to_bits());
        assert_eq!(actual.q1.to_bits(), base_values.q1.to_bits());
        assert_eq!(
            actual.qd1.to_bits(),
            (base_values.dq1_dh1 * values[2]).to_bits()
        );
    }

    #[test]
    fn dual2_order2_timewiggle_q_matches_analytic_scalar_program() {
        const K: usize = 5;
        let values = [0.35, -0.4, 1.2, 0.18, 0.07];
        let h0 = order2_family_variable(values[0], 0, 0.11, -0.03);
        let h1 = order2_family_variable(values[1], 1, -0.08, 0.02);
        let d_raw = order2_family_variable(values[2], 2, 0.05, -0.01);
        let beta = [
            <Dual2<Order2<K>> as JetScalar<K>>::variable(values[3], 3),
            <Dual2<Order2<K>> as JetScalar<K>>::variable(values[4], 4),
        ];
        let entry_derivatives = analytic_basis_derivatives(values[0]);
        let exit_derivatives = analytic_basis_derivatives(values[1]);
        let entry_rows = derivative_rows(&entry_derivatives);
        let exit_rows = derivative_rows(&exit_derivatives);

        let actual = timewiggle_q_from_basis_derivative_rows(
            &h0,
            &h1,
            &d_raw,
            &beta,
            &entry_rows,
            &exit_rows,
            analytic_base_values(values[0], values[1], [values[3], values[4]]),
        )
        .expect("analytic B..B5 rows have matching widths");
        let expected = analytic_q(&h0, &h1, &d_raw, &beta);

        assert_dual_order2(&actual.q0, &expected.q0, "q0");
        assert_dual_order2(&actual.q1, &expected.q1, "q1");
        assert_dual_order2(&actual.qd1, &expected.qd1, "qd1");
    }

    #[test]
    fn dual2_oneseed_timewiggle_q_matches_analytic_family_hessian_drift() {
        const K: usize = 5;
        let values = [0.35, -0.4, 1.2, 0.18, 0.07];
        let direction = [0.3, -0.2, 0.15, -0.4, 0.25];
        let h0 = oneseed_family_variable(values[0], 0, direction[0], 0.11, -0.03);
        let h1 = oneseed_family_variable(values[1], 1, direction[1], -0.08, 0.02);
        let d_raw = oneseed_family_variable(values[2], 2, direction[2], 0.05, -0.01);
        let beta = [
            oneseed_family_variable(values[3], 3, direction[3], 0.0, 0.0),
            oneseed_family_variable(values[4], 4, direction[4], 0.0, 0.0),
        ];
        let entry_derivatives = analytic_basis_derivatives(values[0]);
        let exit_derivatives = analytic_basis_derivatives(values[1]);
        let entry_rows = derivative_rows(&entry_derivatives);
        let exit_rows = derivative_rows(&exit_derivatives);

        let actual = timewiggle_q_from_basis_derivative_rows(
            &h0,
            &h1,
            &d_raw,
            &beta,
            &entry_rows,
            &exit_rows,
            analytic_base_values(values[0], values[1], [values[3], values[4]]),
        )
        .expect("analytic B..B5 rows have matching widths");
        let expected = analytic_q(&h0, &h1, &d_raw, &beta);

        assert_dual_oneseed(&actual.q0, &expected.q0, "q0");
        assert_dual_oneseed(&actual.q1, &expected.q1, "q1");
        assert_dual_oneseed(&actual.qd1, &expected.qd1, "qd1");
    }
}
