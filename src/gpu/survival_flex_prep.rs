//! Survival-flex per-row **prep-step** dispatchers.
//!
//! These two `try_device_*` entries are the GPU-shaped seam for the per-row
//! prep work that currently dominates biobank survival-flex wall time:
//!
//! * [`try_device_partition_cells`] — batched version of
//!   `SurvivalMarginalSlopeFamily::denested_partition_cells`
//!   (`src/families/survival_marginal_slope.rs:5701`).
//! * [`try_device_cell_primary_fixed_partials`] — batched version of
//!   `SurvivalMarginalSlopeFamily::denested_cell_primary_fixed_partials`
//!   (`src/families/survival_marginal_slope.rs:6218`).
//!
//! Both entries return `Ok(None)` until the NVRTC kernel landings in
//! [`kernel_src`] are compiled and dispatched.  Until then, the family-side
//! call sites in `build_cached_partition_with_moment_order` (`:6522`) treat
//! `Ok(None)` as "fall through to the existing CPU path", so the seams are
//! inert wrt behavior but provide the stable boundary the NVRTC body fills
//! in next.
//!
//! Layout of the eventual device output mirrors the audit doc
//! `docs/survival_prep_port_audit.md`:
//!
//! ```text
//!   cells     : Vec<f64>  // flat 18·n_cells doubles (cell ⨁ score_span ⨁ link_span)
//!   offsets   : Vec<u32>  // CSR-style row offsets, length n_rows + 1
//!   status    : Vec<u8>   // 0 = ok, non-zero = host-fallback signal for that row
//! ```
//!
//! and the primary-fixed-partials kernel writes a parallel
//! `12 + 40·primary.total` doubles per cell into the same row/cell indexing.

use crate::families::cubic_cell_kernel::{DenestedPartitionCell, LocalSpanCubic};
use crate::gpu::error::GpuError;

/// CUDA C++ kernel source strings for the two NVRTC kernels.  Both bodies are
/// the literal translation of the CPU implementations cited above; sibling
/// commits compile them via NVRTC and route the device-residency branch of
/// the `try_device_*` entries through the compiled launches.
pub mod kernel_src {
    /// NVRTC source for `denested_partition_cells_kernel`.
    ///
    /// One thread per row.  Inputs:
    ///
    /// * `a[n_rows]`, `b[n_rows]`        — per-row affine intercept + slope
    /// * `score_breaks[n_score_breaks]`  — score-warp breakpoint vector
    /// * `link_breaks[n_link_breaks]`    — link-deviation breakpoint vector
    /// * `score_runtime` (knot table + per-span coefficient tables)
    /// * `link_runtime`  (knot table + per-span coefficient tables)
    /// * `beta_h_flat[n_rows * h_dim]`, `beta_w_flat[n_rows * w_dim]`
    ///   — concatenated per-row β slices
    /// * `score_dim`, `probit_frailty_scale` — scalar constants
    ///
    /// Outputs (flat-packed, CSR-style):
    ///
    /// * `cells_flat[total_cells * 18]`  — 18 doubles per cell:
    ///   `(cell.{left,right,c0,c1,c2,c3}, score_span.{left,right,c0,c1,c2,c3},
    ///   link_span.{left,right,c0,c1,c2,c3})`
    /// * `row_offsets[n_rows + 1]`       — first/last cell index per row
    /// * `status[n_rows]`                — 0 = ok, 1 = tail not affine,
    ///   2 = invalid interval, 3 = non-finite coefficient
    ///
    /// Mirrors `families::cubic_cell_kernel::build_denested_partition_cells_with_tails`
    /// (`src/families/cubic_cell_kernel.rs:2637`) plus the
    /// `SurvivalMarginalSlopeFamily::denested_partition_cells` post-scale
    /// (`src/families/survival_marginal_slope.rs:5701`).
    pub const DENESTED_PARTITION_CELLS_KERNEL_SRC: &str = r#"
// f64 throughout (no --use_fast_math).
// Compiled by NVRTC at runtime against the survival-flex shared module.

#include <cstdint>

extern "C" {

struct LocalSpanCubic {
    double left;
    double right;
    double c0;
    double c1;
    double c2;
    double c3;
};

struct GpuDenestedPartitionCell {
    // cell { left, right, c0, c1, c2, c3 }
    double cell_left;
    double cell_right;
    double cell_c0;
    double cell_c1;
    double cell_c2;
    double cell_c3;
    LocalSpanCubic score_span;
    LocalSpanCubic link_span;
};

__device__ inline double normalized_branch_tol() { return 1e-10; }

__device__ inline LocalSpanCubic zero_span() {
    LocalSpanCubic s;
    s.left = 0.0; s.right = 1.0;
    s.c0 = 0.0; s.c1 = 0.0; s.c2 = 0.0; s.c3 = 0.0;
    return s;
}

__device__ inline void global_cubic_from_local(
    const LocalSpanCubic span,
    double *q0, double *q1, double *q2, double *q3
) {
    const double L = span.left;
    *q0 = span.c0 - span.c1 * L + span.c2 * L * L - span.c3 * L * L * L;
    *q1 = span.c1 - 2.0 * span.c2 * L + 3.0 * span.c3 * L * L;
    *q2 = span.c2 - 3.0 * span.c3 * L;
    *q3 = span.c3;
}

__device__ inline void transformed_link_cubic(
    const LocalSpanCubic span, double a, double b,
    double *d0, double *d1, double *d2, double *d3
) {
    const double shift = a - span.left;
    *d0 = span.c0 + span.c1 * shift + span.c2 * shift * shift + span.c3 * shift * shift * shift;
    *d1 = b * (span.c1 + 2.0 * span.c2 * shift + 3.0 * span.c3 * shift * shift);
    *d2 = b * b * (span.c2 + 3.0 * span.c3 * shift);
    *d3 = span.c3 * b * b * b;
}

__device__ inline void denested_cell_coefficients(
    LocalSpanCubic score_span, LocalSpanCubic link_span, double a, double b,
    double *c0, double *c1, double *c2, double *c3
) {
    double h0, h1, h2, h3, d0, d1, d2, d3;
    global_cubic_from_local(score_span, &h0, &h1, &h2, &h3);
    transformed_link_cubic(link_span, a, b, &d0, &d1, &d2, &d3);
    *c0 = a + b * h0 + d0;
    *c1 = b + b * h1 + d1;
    *c2 = b * h2 + d2;
    *c3 = b * h3 + d3;
}

// NOTE: full body — split-point merge / dedup, left+interior+right tail
// cells, affine-tail validation, optional probit_frailty_scale post-pass —
// lands in the V100 wiring commit alongside the runtime upload helpers in
// `src/gpu/cubic_cell/`. The host fallback in the family already implements
// the identical math; this string is the kernel skeleton against which the
// V100 sibling will fill in the body and add the runtime closure dispatch.
__global__ void denested_partition_cells_kernel(
    int n_rows,
    const double *a_per_row,
    const double *b_per_row,
    GpuDenestedPartitionCell *out_cells_flat,
    unsigned int *out_row_offsets,
    unsigned char *out_status
);

}  // extern "C"
"#;

    /// NVRTC source for `denested_cell_primary_fixed_partials_kernel`.
    ///
    /// One thread per `(row, cell)`.  Consumes the cell table emitted by the
    /// partition kernel plus the `FlexPrimaryLayout` constant (q0/q1/qd1/g
    /// slot indices, h/w ranges, `r = total`) and produces the flat
    /// `12 + 40·r` doubles per cell that downstream Layer A/B/C entries
    /// consume.
    ///
    /// Output layout, per cell:
    ///
    /// ```text
    ///   dc_da[4], dc_daa[4], dc_daaa[4]                       // 12 doubles
    ///   coeff_u[r][4]                                          // 4r
    ///   coeff_au[r][4], coeff_bu[r][4]                         // 8r
    ///   coeff_aau[r][4], coeff_abu[r][4], coeff_bbu[r][4]      // 12r
    ///   coeff_aaau[r][4], coeff_aabu[r][4], coeff_abbu[r][4], coeff_bbbu[r][4]   // 16r
    /// ```
    ///
    /// Mirrors the CPU `denested_cell_primary_fixed_partials`
    /// (`src/families/survival_marginal_slope.rs:6218`).
    pub const DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC: &str = r#"
// f64 throughout (no --use_fast_math).

#include <cstdint>

extern "C" {

// Per-call constants describing the FlexPrimarySlices layout.
struct FlexPrimaryLayout {
    unsigned int q0;       // 0
    unsigned int q1;       // 1
    unsigned int qd1;      // 2
    unsigned int g;        // 3
    unsigned int h_start;  // 4
    unsigned int h_len;    // score_warp.basis_dim * score_dim (0 when absent)
    unsigned int w_start;  // h_start + h_len
    unsigned int w_len;    // link_dev.basis_dim (0 when absent)
    unsigned int r;        // 4 + h_len + w_len (= primary.total)
    unsigned int score_dim;            // >= 1
    unsigned int score_warp_basis_dim; // for score_warp_coord_basis_index
    double scale;                       // probit_frailty_scale
};

__global__ void denested_cell_primary_fixed_partials_kernel(
    FlexPrimaryLayout layout,
    int n_cells_total,
    const double *cells_flat,   // 18 doubles per cell
    const double *a_per_row,
    const double *b_per_row,
    const unsigned int *row_offsets,
    /* runtime tables (score_warp + link_dev) — wired in the V100 commit */
    double *out_partials_flat,  // (12 + 40·r) doubles per cell
    unsigned char *out_status
);

}  // extern "C"
"#;
}

/// Per-row inputs for [`try_device_partition_cells`].
#[derive(Clone, Copy, Debug)]
pub struct PartitionCellsRowInputs<'a> {
    pub a: f64,
    pub b: f64,
    pub beta_h: Option<&'a [f64]>,
    pub beta_w: Option<&'a [f64]>,
}

/// Output of [`try_device_partition_cells`]: per-row partition cells in the
/// existing `DenestedPartitionCell` shape, one inner `Vec` per row.
///
/// Kept as a `Vec<Vec<...>>` (not flat) so the family-side caller can drop
/// the GPU output straight into the existing per-row consumer loop without a
/// re-pack.
pub type PartitionCellsOutput = Vec<Vec<DenestedPartitionCell>>;

/// GPU-shaped seam for `SurvivalMarginalSlopeFamily::denested_partition_cells`.
///
/// Returns:
///
/// * `Ok(None)` when the GPU path is unsupported for this batch shape
///   (currently always — the NVRTC kernel lands in the sibling commit using
///   [`kernel_src::DENESTED_PARTITION_CELLS_KERNEL_SRC`]); callers fall back
///   to the existing CPU per-row path.
/// * `Ok(Some(out))` when the device-shaped output is materialized.  `out.len()
///   == rows.len()` and `out[i]` is byte-identical to what
///   `denested_partition_cells(rows[i].a, rows[i].b, rows[i].beta_h,
///   rows[i].beta_w)` would have produced on the CPU.
/// * `Err(_)` only when the request *is* supported but the driver failed.
pub fn try_device_partition_cells(
    rows: &[PartitionCellsRowInputs<'_>],
) -> Result<Option<PartitionCellsOutput>, GpuError> {
    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    // Always unsupported until the NVRTC kernel lands and the per-batch
    // runtime upload is wired.  The seam exists so call sites can route
    // through a single batched entry point without further rework.
    Ok(None)
}

/// Per-cell inputs for [`try_device_cell_primary_fixed_partials`].
#[derive(Clone, Copy, Debug)]
pub struct CellPrimaryFixedPartialsCellInputs {
    pub score_span: LocalSpanCubic,
    pub link_span: LocalSpanCubic,
    pub z_basis: f64,
    pub u_basis: f64,
}

/// Per-row inputs for [`try_device_cell_primary_fixed_partials`]: shared
/// `(a, b)` scalars and the per-cell slice from this row.
#[derive(Clone, Copy, Debug)]
pub struct CellPrimaryFixedPartialsRowInputs<'a> {
    pub a: f64,
    pub b: f64,
    pub cells: &'a [CellPrimaryFixedPartialsCellInputs],
}

/// Flat-packed output of [`try_device_cell_primary_fixed_partials`].
///
/// `partials[row_idx][cell_idx]` is a `Vec<f64>` of length `12 + 40·r` laid
/// out per the
/// [`kernel_src::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC`] schema.
#[derive(Clone, Debug, Default)]
pub struct CellPrimaryFixedPartialsOutput {
    pub partials: Vec<Vec<Vec<f64>>>,
}

/// GPU-shaped seam for
/// `SurvivalMarginalSlopeFamily::denested_cell_primary_fixed_partials`.
///
/// Returns `Ok(None)` until the NVRTC kernel and the runtime upload land
/// (see [`kernel_src::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC`]); the
/// family-side dispatcher then falls back to the existing CPU per-cell
/// `denested_cell_primary_fixed_partials` for every cell.
pub fn try_device_cell_primary_fixed_partials(
    rows: &[CellPrimaryFixedPartialsRowInputs<'_>],
) -> Result<Option<CellPrimaryFixedPartialsOutput>, GpuError> {
    if rows.is_empty() {
        return Ok(Some(CellPrimaryFixedPartialsOutput::default()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_partition_inputs_short_circuit() {
        let out = try_device_partition_cells(&[]).expect("ok");
        assert!(out.is_some());
        assert!(out.unwrap().is_empty());
    }

    #[test]
    fn nonempty_partition_inputs_decline_until_kernel_lands() {
        let inputs = [PartitionCellsRowInputs {
            a: 0.0,
            b: 1.0,
            beta_h: None,
            beta_w: None,
        }];
        let out = try_device_partition_cells(&inputs).expect("ok");
        assert!(out.is_none());
    }

    #[test]
    fn empty_fixed_partials_inputs_short_circuit() {
        let out = try_device_cell_primary_fixed_partials(&[]).expect("ok");
        assert!(out.is_some());
        assert!(out.unwrap().partials.is_empty());
    }

    #[test]
    fn nonempty_fixed_partials_inputs_decline_until_kernel_lands() {
        let inputs = [CellPrimaryFixedPartialsRowInputs {
            a: 0.0,
            b: 1.0,
            cells: &[],
        }];
        let out = try_device_cell_primary_fixed_partials(&inputs).expect("ok");
        assert!(out.is_none());
    }

    #[test]
    fn kernel_src_strings_are_nonempty() {
        assert!(!kernel_src::DENESTED_PARTITION_CELLS_KERNEL_SRC.is_empty());
        assert!(!kernel_src::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC.is_empty());
    }
}
