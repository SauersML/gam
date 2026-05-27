//! Host-resident substrate for the survival-flex per-row **partition-cell**
//! prep step.
//!
//! This module is the parity reference and host-fallback for the upcoming
//! `denested_partition_cells_kernel` NVRTC kernel.  Its job is to take a
//! per-row batch of `(a, b, β_h, β_w)` inputs together with shared
//! `DeviationRuntimeView` snapshots of the score-warp and link-deviation
//! runtimes and emit a **flat-packed device-shaped cell table** for the entire
//! batch:
//!
//! ```text
//!     cells     : Vec<GpuDenestedPartitionCell>            // flat, per-row contiguous
//!     offsets   : Vec<u32> of length n_rows + 1            // CSR-style row offsets
//!     status    : Vec<u8>  of length n_rows                // 0 = ok, non-zero = host fallback
//! ```
//!
//! The math is delegated to the existing CPU
//! `families::cubic_cell_kernel::build_denested_partition_cells_with_tails`
//! evaluated against the runtime view, so output is byte-identical to what
//! `SurvivalMarginalSlopeFamily::denested_partition_cells` produces on the same
//! inputs.  The kernel port in the sibling commit emits the same flat-packed
//! layout so the host and device paths share the same downstream consumers
//! (`denested_cell_primary_fixed_partials_kernel`, Layer A/B/C entries).
//!
//! Multi-coordinate score warps (`score_dim > 1`) sum the per-coordinate
//! `LocalSpanCubic` contributions exactly the way
//! `SurvivalMarginalSlopeFamily::score_warp_local_cubic_at` does on the CPU
//! family.

use crate::families::cubic_cell_kernel::{
    self, DenestedCubicCell, DenestedPartitionCell, LocalSpanCubic,
};

/// Borrowed snapshot of a `DeviationRuntime`'s per-span coefficient tables.
///
/// All slices are flattened in row-major order with stride `basis_dim`; row
/// `span_idx` of `span_c0` is `&span_c0[span_idx * basis_dim ..][..basis_dim]`.
/// The borrow lives as long as the source `DeviationRuntime`, so the view is
/// always coherent with the live runtime; uploading to device copies these
/// slices verbatim.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviationRuntimeView<'a> {
    pub basis_dim: usize,
    pub endpoint_points: &'a [f64],
    pub span_c0: &'a [f64],
    pub span_c1: &'a [f64],
    pub span_c2: &'a [f64],
    pub span_c3: &'a [f64],
    pub right_boundary_value_row: &'a [f64],
}

impl<'a> DeviationRuntimeView<'a> {
    #[inline]
    pub(crate) fn span_count(&self) -> usize {
        self.endpoint_points.len().saturating_sub(1)
    }

    #[inline]
    fn support_interval(&self) -> Result<(f64, f64), String> {
        if self.endpoint_points.len() < 2 {
            return Err(
                "deviation runtime view: need at least two endpoint points".to_string(),
            );
        }
        Ok((
            self.endpoint_points[0],
            self.endpoint_points[self.endpoint_points.len() - 1],
        ))
    }

    fn left_biased_span_index_for(&self, value: f64) -> Result<usize, String> {
        let n_spans = self.span_count();
        if n_spans == 0 {
            return Err("deviation runtime view: no spans".to_string());
        }
        let mut span_idx = if value <= self.endpoint_points[0] {
            0
        } else if value >= self.endpoint_points[n_spans] {
            n_spans - 1
        } else {
            let mut lo = 0usize;
            let mut hi = n_spans;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if value < self.endpoint_points[mid] {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            lo
        };
        if span_idx > 0 && value == self.endpoint_points[span_idx] {
            span_idx -= 1;
        }
        Ok(span_idx)
    }

    #[inline]
    fn span_row(&self, table: &'a [f64], span_idx: usize) -> &'a [f64] {
        &table[span_idx * self.basis_dim..(span_idx + 1) * self.basis_dim]
    }

    #[inline]
    fn dot(row: &[f64], beta: &[f64]) -> f64 {
        let mut acc = 0.0;
        for (a, b) in row.iter().zip(beta.iter()) {
            acc += *a * *b;
        }
        acc
    }

    /// Composite `LocalSpanCubic` for a single coordinate's β slice at the
    /// requested value. Mirrors `DeviationRuntime::local_cubic_at`
    /// (`src/families/bernoulli_marginal_slope/deviation_runtime.rs:1219`).
    fn local_cubic_at(&self, beta: &[f64], value: f64) -> Result<LocalSpanCubic, String> {
        if beta.len() != self.basis_dim {
            return Err(format!(
                "deviation runtime view: beta dimension {} != basis_dim {}",
                beta.len(),
                self.basis_dim
            ));
        }
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            let row = self.span_row(self.span_c0, 0);
            return Ok(LocalSpanCubic {
                left: left_ep,
                right: left_ep + 1.0,
                c0: Self::dot(row, beta),
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        if value > right_ep {
            return Ok(LocalSpanCubic {
                left: right_ep,
                right: right_ep + 1.0,
                c0: Self::dot(self.right_boundary_value_row, beta),
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        Ok(LocalSpanCubic {
            left: self.endpoint_points[span_idx],
            right: self.endpoint_points[span_idx + 1],
            c0: Self::dot(self.span_row(self.span_c0, span_idx), beta),
            c1: Self::dot(self.span_row(self.span_c1, span_idx), beta),
            c2: Self::dot(self.span_row(self.span_c2, span_idx), beta),
            c3: Self::dot(self.span_row(self.span_c3, span_idx), beta),
        })
    }
}

/// Borrowed view of a single survival-flex row's prep inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SurvivalPartitionRowView<'a> {
    pub a: f64,
    pub b: f64,
    /// Concatenated β vector for the (possibly multi-coordinate) score warp.
    /// Length is `score_warp.basis_dim * score_dim` when present.
    pub beta_h: Option<&'a [f64]>,
    /// β vector for the link deviation. Length is `link_dev.basis_dim` when
    /// present.
    pub beta_w: Option<&'a [f64]>,
}

/// Borrowed batch view consumed by [`build_host_partition_cells`].
///
/// `score_warp` / `link_dev` are `None` when the family does not have that
/// runtime installed; in that case the corresponding β is also `None` on every
/// row.  `score_dim` is the number of score-warp coordinates: 1 in the common
/// case, `> 1` for the multi-coordinate fast path (which sums per-coordinate
/// contributions exactly the way
/// `SurvivalMarginalSlopeFamily::score_warp_local_cubic_at` does).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SurvivalPartitionBatchView<'a, 'b> {
    pub rows: &'a [SurvivalPartitionRowView<'b>],
    pub score_warp: Option<DeviationRuntimeView<'b>>,
    pub link_dev: Option<DeviationRuntimeView<'b>>,
    pub score_dim: usize,
    pub probit_frailty_scale: f64,
}

/// Flat-packed `DenestedPartitionCell` for the device-shape output. Layout is
/// the catenation of `cell` (6 f64), `score_span` (6 f64), `link_span` (6 f64)
/// — 18 doubles per cell — so the upload to device is a single contiguous
/// `cudaMemcpy`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuDenestedPartitionCell {
    pub cell: DenestedCubicCell,
    pub score_span: LocalSpanCubic,
    pub link_span: LocalSpanCubic,
}

impl From<DenestedPartitionCell> for GpuDenestedPartitionCell {
    #[inline]
    fn from(p: DenestedPartitionCell) -> Self {
        GpuDenestedPartitionCell {
            cell: p.cell,
            score_span: p.score_span,
            link_span: p.link_span,
        }
    }
}

/// Per-row status emitted by [`build_host_partition_cells`].
///
/// Numeric values match the planned NVRTC kernel's status emission so the GPU
/// and host paths fill `Vec<u8>` with the same byte pattern.  Any non-zero
/// status signals that the row could not be built device-side and the caller
/// must fall back to the existing CPU per-row path for that row only.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PartitionRowStatus {
    Ok = 0,
    /// Build returned a hard CPU error (e.g. tail cell not affine, span
    /// lookup failed). The row's slice in `cells` is left at `[]`.
    HostError = 1,
}

/// CSR-style flat output of the partition substrate.
///
/// `offsets` has length `n_rows + 1`; row `r`'s cells are
/// `cells[offsets[r] as usize .. offsets[r + 1] as usize]`. Rows with non-zero
/// `status[r]` have an empty slice.  `status[r] == 0` means the row was built
/// successfully and the slice contains its `Vec<DenestedPartitionCell>`
/// equivalent in declaration order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PartitionCellBatch {
    pub cells: Vec<GpuDenestedPartitionCell>,
    pub offsets: Vec<u32>,
    pub status: Vec<u8>,
}

/// Host-resident reference implementation.
///
/// Per row, this mirrors `SurvivalMarginalSlopeFamily::denested_partition_cells`
/// (`src/families/survival_marginal_slope.rs:5718`) exactly:
///
/// 1. Build score / link breakpoint slices from the runtime views.
/// 2. Invoke `cubic_cell_kernel::build_denested_partition_cells_with_tails`
///    with closures that route through `DeviationRuntimeView::local_cubic_at`
///    (per-coord summation for multi-coordinate score warps).
/// 3. Apply the `probit_frailty_scale` post-pass on `(c0, c1, c2, c3)` when
///    the scale is not 1.0.
///
/// Any per-row `Err(_)` is recorded as `PartitionRowStatus::HostError` and the
/// row's cell slice is left empty; the batch as a whole still returns `Ok` so
/// the caller can selectively fall back per row.
pub(crate) fn build_host_partition_cells(
    view: &SurvivalPartitionBatchView<'_, '_>,
) -> PartitionCellBatch {
    let n_rows = view.rows.len();
    let mut cells = Vec::<GpuDenestedPartitionCell>::new();
    let mut offsets = Vec::<u32>::with_capacity(n_rows + 1);
    let mut status = vec![PartitionRowStatus::Ok as u8; n_rows];
    offsets.push(0);

    let scale = view.probit_frailty_scale;

    let score_breaks: Vec<f64> = view
        .score_warp
        .as_ref()
        .map(|w| w.endpoint_points.to_vec())
        .unwrap_or_default();
    let link_breaks: Vec<f64> = view
        .link_dev
        .as_ref()
        .map(|w| w.endpoint_points.to_vec())
        .unwrap_or_default();

    let zero_span = LocalSpanCubic {
        left: 0.0,
        right: 1.0,
        c0: 0.0,
        c1: 0.0,
        c2: 0.0,
        c3: 0.0,
    };

    for (row_idx, row) in view.rows.iter().enumerate() {
        let score_warp_view = view.score_warp.as_ref();
        let link_dev_view = view.link_dev.as_ref();
        let beta_h = row.beta_h;
        let beta_w = row.beta_w;
        let score_dim = view.score_dim.max(1);

        let score_span_at = |z: f64| -> Result<LocalSpanCubic, String> {
            let Some(runtime) = score_warp_view else {
                return Ok(zero_span);
            };
            let Some(beta) = beta_h else {
                return Ok(zero_span);
            };
            if score_dim == 1 {
                return runtime.local_cubic_at(beta, z);
            }
            // Multi-coord: sum per-coordinate contributions; first coord
            // contributes its (left, right) support to the composite span.
            let basis_dim = runtime.basis_dim;
            if beta.len() != basis_dim * score_dim {
                return Err(format!(
                    "partition substrate: multi-coord beta_h length {} != basis_dim {} * score_dim {}",
                    beta.len(),
                    basis_dim,
                    score_dim
                ));
            }
            let mut sum = zero_span;
            for coord in 0..score_dim {
                let local_beta =
                    &beta[coord * basis_dim..(coord + 1) * basis_dim];
                let span = runtime.local_cubic_at(local_beta, z)?;
                if coord == 0 {
                    sum.left = span.left;
                    sum.right = span.right;
                }
                sum.c0 += span.c0;
                sum.c1 += span.c1;
                sum.c2 += span.c2;
                sum.c3 += span.c3;
            }
            Ok(sum)
        };
        let link_span_at = |u: f64| -> Result<LocalSpanCubic, String> {
            let Some(runtime) = link_dev_view else {
                return Ok(zero_span);
            };
            let Some(beta) = beta_w else {
                return Ok(zero_span);
            };
            runtime.local_cubic_at(beta, u)
        };

        let built = cubic_cell_kernel::build_denested_partition_cells_with_tails(
            row.a,
            row.b,
            &score_breaks,
            &link_breaks,
            score_span_at,
            link_span_at,
        );

        match built {
            Ok(mut row_cells) => {
                if scale != 1.0 {
                    for partition_cell in &mut row_cells {
                        partition_cell.cell.c0 *= scale;
                        partition_cell.cell.c1 *= scale;
                        partition_cell.cell.c2 *= scale;
                        partition_cell.cell.c3 *= scale;
                    }
                }
                for pc in row_cells {
                    cells.push(pc.into());
                }
            }
            Err(_reason) => {
                status[row_idx] = PartitionRowStatus::HostError as u8;
            }
        }
        offsets.push(cells.len() as u32);
    }

    PartitionCellBatch {
        cells,
        offsets,
        status,
    }
}

/// Convenience: convert a flat row's cells back into a `Vec<DenestedPartitionCell>`
/// for direct comparison with the existing CPU path's output.
pub(crate) fn row_cells_as_partition_vec(
    batch: &PartitionCellBatch,
    row_idx: usize,
) -> Vec<DenestedPartitionCell> {
    let start = batch.offsets[row_idx] as usize;
    let end = batch.offsets[row_idx + 1] as usize;
    batch.cells[start..end]
        .iter()
        .map(|gc| DenestedPartitionCell {
            cell: gc.cell,
            score_span: gc.score_span,
            link_span: gc.link_span,
        })
        .collect()
}

/// Build a borrowed view of a `DeviationRuntime` for the substrate.
///
/// The CPU `DeviationRuntime` stores its per-span tables as `Array2<f64>`
/// (row-major); this helper exposes them as contiguous `&[f64]` slices.
/// Returns `None` when the runtime is absent or its tables are not contiguous
/// in standard row-major order (the contiguous-layout invariant always holds
/// for runtimes built via `DeviationRuntime::new`, so this is a defensive
/// fallthrough rather than an expected path).
pub(crate) fn deviation_runtime_view<'a>(
    runtime: &'a crate::families::bernoulli_marginal_slope::DeviationRuntime,
) -> Option<DeviationRuntimeView<'a>> {
    let endpoint_points = runtime.breakpoints().as_slice()?;
    let span_c0 = runtime.span_c0_slice()?;
    let span_c1 = runtime.span_c1_slice()?;
    let span_c2 = runtime.span_c2_slice()?;
    let span_c3 = runtime.span_c3_slice()?;
    let right_boundary_value_row = runtime.right_boundary_value_row_slice()?;
    Some(DeviationRuntimeView {
        basis_dim: runtime.basis_dim(),
        endpoint_points,
        span_c0,
        span_c1,
        span_c2,
        span_c3,
        right_boundary_value_row,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::cubic_cell_kernel as exact_kernel;

    /// A trivial runtime view built from raw arrays, for tests that don't
    /// want to go through the full `DeviationRuntime` constructor.
    fn make_view<'a>(
        endpoint_points: &'a [f64],
        span_c0: &'a [f64],
        span_c1: &'a [f64],
        span_c2: &'a [f64],
        span_c3: &'a [f64],
        right_boundary_value_row: &'a [f64],
        basis_dim: usize,
    ) -> DeviationRuntimeView<'a> {
        DeviationRuntimeView {
            basis_dim,
            endpoint_points,
            span_c0,
            span_c1,
            span_c2,
            span_c3,
            right_boundary_value_row,
        }
    }

    #[test]
    fn local_cubic_at_matches_left_tail_saturation() {
        // basis_dim = 1; one span [0, 1]; constant basis (c0 = 1, all other c = 0).
        let view = make_view(
            &[0.0, 1.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            1,
        );
        let beta = [0.5_f64];
        let span = view.local_cubic_at(&beta, -0.25).unwrap();
        assert_eq!(span.left, 0.0);
        assert_eq!(span.c0, 0.5);
        assert_eq!(span.c1, 0.0);
        assert_eq!(span.c2, 0.0);
        assert_eq!(span.c3, 0.0);
    }

    #[test]
    fn local_cubic_at_matches_right_tail_saturation() {
        let view = make_view(
            &[0.0, 1.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[2.0],
            1,
        );
        let beta = [0.5_f64];
        let span = view.local_cubic_at(&beta, 2.5).unwrap();
        assert_eq!(span.c0, 1.0); // 2.0 * 0.5
        assert_eq!(span.c1, 0.0);
    }

    #[test]
    fn build_host_partition_cells_empty_runtimes_single_cell() {
        // No score warp, no link dev: partition is a single (-inf, +inf) cell.
        let row = SurvivalPartitionRowView {
            a: 0.0,
            b: 1.0,
            beta_h: None,
            beta_w: None,
        };
        let batch = build_host_partition_cells(&SurvivalPartitionBatchView {
            rows: std::slice::from_ref(&row),
            score_warp: None,
            link_dev: None,
            score_dim: 1,
            probit_frailty_scale: 1.0,
        });
        assert_eq!(batch.offsets, vec![0, 1]);
        assert_eq!(batch.status, vec![0]);
        assert_eq!(batch.cells.len(), 1);
        let cell = batch.cells[0].cell;
        assert!(!cell.left.is_finite());
        assert!(!cell.right.is_finite());
        assert_eq!(cell.c2, 0.0);
        assert_eq!(cell.c3, 0.0);
    }

    #[test]
    fn build_host_partition_cells_applies_scale() {
        let row = SurvivalPartitionRowView {
            a: 0.5,
            b: 2.0,
            beta_h: None,
            beta_w: None,
        };
        let unscaled = build_host_partition_cells(&SurvivalPartitionBatchView {
            rows: std::slice::from_ref(&row),
            score_warp: None,
            link_dev: None,
            score_dim: 1,
            probit_frailty_scale: 1.0,
        });
        let scaled = build_host_partition_cells(&SurvivalPartitionBatchView {
            rows: std::slice::from_ref(&row),
            score_warp: None,
            link_dev: None,
            score_dim: 1,
            probit_frailty_scale: 0.5,
        });
        assert_eq!(unscaled.offsets, scaled.offsets);
        let u = unscaled.cells[0].cell;
        let s = scaled.cells[0].cell;
        assert!((s.c0 - 0.5 * u.c0).abs() < 1e-15);
        assert!((s.c1 - 0.5 * u.c1).abs() < 1e-15);
        assert!((s.c2 - 0.5 * u.c2).abs() < 1e-15);
        assert!((s.c3 - 0.5 * u.c3).abs() < 1e-15);
    }

    #[test]
    fn build_host_partition_cells_matches_direct_kernel_call_with_link_dev() {
        // Single span [-2, 2], basis_dim = 2: identity (1, x) basis on the span.
        let endpoint_points = vec![-2.0_f64, 2.0];
        // Per-span tables row-major [n_spans, basis_dim] = [1, 2].
        let span_c0 = vec![0.0, 0.0];
        let span_c1 = vec![1.0, 0.0];
        let span_c2 = vec![0.0, 0.0];
        let span_c3 = vec![0.0, 0.0];
        let right_boundary_value_row = vec![2.0, 0.0];
        let link_view = make_view(
            &endpoint_points,
            &span_c0,
            &span_c1,
            &span_c2,
            &span_c3,
            &right_boundary_value_row,
            2,
        );
        let beta_w = vec![0.3_f64, 0.4];
        let row = SurvivalPartitionRowView {
            a: 0.1,
            b: 1.5,
            beta_h: None,
            beta_w: Some(&beta_w),
        };
        let batch = build_host_partition_cells(&SurvivalPartitionBatchView {
            rows: std::slice::from_ref(&row),
            score_warp: None,
            link_dev: Some(link_view),
            score_dim: 1,
            probit_frailty_scale: 1.0,
        });
        assert_eq!(batch.status, vec![0]);
        let from_view = row_cells_as_partition_vec(&batch, 0);

        // Direct kernel call with the same span closures.
        let zero = LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let direct = exact_kernel::build_denested_partition_cells_with_tails(
            0.1,
            1.5,
            &[],
            &endpoint_points,
            |_z| Ok(zero),
            |u| link_view.local_cubic_at(&beta_w, u),
        )
        .expect("direct kernel build succeeds");

        assert_eq!(from_view.len(), direct.len());
        for (a, b) in from_view.iter().zip(direct.iter()) {
            assert_eq!(a.cell.left, b.cell.left);
            assert_eq!(a.cell.right, b.cell.right);
            assert_eq!(a.cell.c0, b.cell.c0);
            assert_eq!(a.cell.c1, b.cell.c1);
            assert_eq!(a.cell.c2, b.cell.c2);
            assert_eq!(a.cell.c3, b.cell.c3);
        }
    }

    #[test]
    fn status_codes_match_kernel_abi() {
        assert_eq!(PartitionRowStatus::Ok as u8, 0);
        assert_eq!(PartitionRowStatus::HostError as u8, 1);
    }
}
