//! Survival-flex per-row **prep-step** dispatchers.
//!
//! These two `try_device_*` entries are the GPU-shaped seam for the per-row
//! prep work that currently dominates large-scale survival-flex wall time:
//!
//! * [`try_device_partition_cells`] — batched version of
//!   `SurvivalMarginalSlopeFamily::denested_partition_cells`
//!   (`src/families/survival_marginal_slope.rs:5701`).
//! * [`try_device_cell_primary_fixed_partials`] — batched version of
//!   `SurvivalMarginalSlopeFamily::denested_cell_primary_fixed_partials`
//!   (`src/families/survival_marginal_slope.rs:6218`).
//!
//! Layout of the device output:
//!
//! ```text
//!   cells     : Vec<f64>  // flat 18·n_cells doubles (cell ⨁ score_span ⨁ link_span)
//!   offsets   : Vec<u32>  // CSR-style row offsets, length n_rows + 1
//!   status    : Vec<u8>   // 0 = ok, non-zero = host-fallback signal for that row
//! ```
//!
//! and the primary-fixed-partials kernel writes a parallel
//! `12 + 40·primary.total` doubles per cell into the same row/cell indexing.
//!
//! ## Supported shapes
//!
//! The NVRTC bodies here cover the **no-runtime baseline** path: rows where
//! neither `beta_h` nor `beta_w` is provided.  In that regime
//! `build_denested_partition_cells_with_tails` returns a single trivial
//! affine cell `(c0=a·scale, c1=b·scale, c2=0, c3=0)` per row (no split
//! points), and the fixed-partials per cell reduces to just the `g`-slot
//! pieces (`coeff_u[g]=dc_db`, `coeff_au[g]=dc_dab`, ..., `dc_da=[1,0,0,0]·scale`).
//! Both kernels execute that closed-form path on-device with zero host
//! arithmetic and the dispatchers DtoH back to the caller's shape.
//!
//! Rows that need a non-trivial knot-table / B-spline runtime traversal
//! (i.e. any row carrying a `beta_h` or `beta_w` slice) cause the
//! dispatcher to return `Ok(None)` so the family-side path falls back to
//! the existing CPU per-row code.  The kernel surface, runtime upload
//! plumbing, and DtoH re-pack stay device-shaped so the eventual general
//! body lands behind the same call boundary.

use crate::families::cubic_cell_kernel::{
    DenestedCubicCell, DenestedPartitionCell, LocalSpanCubic,
};
use crate::gpu::error::GpuError;

/// CUDA C++ kernel source strings for the two NVRTC kernels.  Both bodies are
/// the literal translation of the CPU implementations cited above.
pub mod kernel_src {
    /// NVRTC source for `denested_partition_cells_kernel`.
    ///
    /// One thread per row.  Trivial no-runtime case: emits a single affine
    /// cell `(c0=a·scale, c1=b·scale, c2=0, c3=0)` with zero score/link
    /// spans, mirroring the CPU `build_denested_partition_cells_with_tails`
    /// empty-split-points branch followed by the
    /// `SurvivalMarginalSlopeFamily::denested_partition_cells` post-scale.
    pub const DENESTED_PARTITION_CELLS_KERNEL_SRC: &str = r#"
// f64 throughout (no --use_fast_math).

extern "C" {

__device__ __forceinline__ double pos_inf_f64() {
    // IEEE-754 +inf bit pattern: 0x7ff0000000000000.
    return __longlong_as_double((long long)0x7ff0000000000000LL);
}
__device__ __forceinline__ double neg_inf_f64() {
    // IEEE-754 -inf bit pattern: 0xfff0000000000000.
    return __longlong_as_double((long long)0xfff0000000000000LL);
}

__global__ void denested_partition_cells_kernel(
    int n_rows,
    double scale,
    const double *a_per_row,
    const double *b_per_row,
    double *out_cells_flat,        // 18 doubles per row (single cell)
    unsigned int *out_row_offsets, // length n_rows + 1
    unsigned char *out_status      // length n_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    double a = a_per_row[i];
    double b = b_per_row[i];
    double *cell = out_cells_flat + (long long)i * 18;
    // ── cell: (-inf, +inf, c0=a*scale, c1=b*scale, c2=0, c3=0) ──
    cell[0]  = neg_inf_f64();
    cell[1]  = pos_inf_f64();
    cell[2]  = a * scale;
    cell[3]  = b * scale;
    cell[4]  = 0.0;
    cell[5]  = 0.0;
    // ── score_span (zero cubic, left=0,right=1) ──
    cell[6]  = 0.0; cell[7]  = 1.0;
    cell[8]  = 0.0; cell[9]  = 0.0; cell[10] = 0.0; cell[11] = 0.0;
    // ── link_span (zero cubic, left=0,right=1) ──
    cell[12] = 0.0; cell[13] = 1.0;
    cell[14] = 0.0; cell[15] = 0.0; cell[16] = 0.0; cell[17] = 0.0;
    // ── row offset: one cell per row ──
    out_row_offsets[i] = (unsigned int)i;
    if (i == n_rows - 1) {
        out_row_offsets[n_rows] = (unsigned int)n_rows;
    }
    out_status[i] = 0;
}

}  // extern "C"
"#;

    /// NVRTC source for `denested_cell_primary_fixed_partials_kernel`.
    ///
    /// One thread per cell.  Trivial no-runtime case: only the `g` slot is
    /// populated because `primary.h` and `primary.w` are empty when both
    /// runtimes are absent.  Mirrors the closed-form arithmetic the CPU
    /// `denested_cell_primary_fixed_partials` runs when `h_len == 0 &&
    /// w_len == 0`.
    ///
    /// For the trivial cell `(c0=a·scale, c1=b·scale, c2=0, c3=0)` the
    /// partials evaluate to:
    /// * `dc_da   = [1, 0, 0, 0] · scale`
    /// * `dc_daa  = [0, 0, 0, 0]`
    /// * `dc_daaa = [0, 0, 0, 0]`
    /// * `dc_db = dc_dab = dc_dbb = dc_dabb = dc_dbbb = ...` reduce to
    ///   `[0, 1, 0, 0] · scale`, `[0, 0, 0, 0]`, ... per the
    ///   `denested_cell_*_partials` formulas with `score_span=zero`,
    ///   `link_span=zero`.
    pub const DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC: &str = r#"
// f64 throughout (no --use_fast_math).

extern "C" {

__global__ void denested_cell_primary_fixed_partials_kernel(
    int n_cells_total,
    unsigned int r,
    unsigned int g_slot,
    double scale,
    double *out_partials_flat,  // (12 + 40·r) doubles per cell
    unsigned char *out_status
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells_total) return;
    unsigned int per_cell = 12u + 40u * r;
    double *base = out_partials_flat + (long long)cell * (long long)per_cell;
    // Zero the whole block (cheap; r is small).
    for (unsigned int s = 0; s < per_cell; ++s) {
        base[s] = 0.0;
    }
    // dc_da = [1, 0, 0, 0] · scale
    base[0] = scale;
    // dc_daa, dc_daaa already zero.
    // g-slot fills (offset = 12 + 4·g_slot within each per-cell run).
    //   coeff_u   [g] = dc_db   = [0, 1, 0, 0] · scale
    //   coeff_au  [g] = dc_dab  = [0, 0, 0, 0]
    //   coeff_bu  [g] = dc_dbb  = [0, 0, 0, 0]
    //   coeff_aau [g] = dc_daab = [0, 0, 0, 0]
    //   coeff_abu [g] = dc_dabb = [0, 0, 0, 0]
    //   coeff_bbu [g] = dc_dbbb = [0, 0, 0, 0]
    //   (third partials all zero in the no-runtime case)
    unsigned int g_off = 12u + 4u * g_slot;
    base[g_off + 1] = scale;  // coeff_u[g][1] = scale
    out_status[cell] = 0;
}

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
pub type PartitionCellsOutput = Vec<Vec<DenestedPartitionCell>>;

/// GPU-shaped seam for `SurvivalMarginalSlopeFamily::denested_partition_cells`.
///
/// Returns:
///
/// * `Ok(None)` when the GPU path is unsupported (CUDA absent, or any row
///   carries a `beta_h`/`beta_w` slice that would require a B-spline
///   runtime traversal — those fall through to the existing CPU per-row
///   path).
/// * `Ok(Some(out))` when the device-shaped output is materialized.
/// * `Err(_)` only when the request *is* supported but the driver failed.
pub fn try_device_partition_cells(
    rows: &[PartitionCellsRowInputs<'_>],
) -> Result<Option<PartitionCellsOutput>, GpuError> {
    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    // Only the no-runtime baseline (no β slices on any row) is implemented
    // device-side today.  Any row carrying a beta vector needs the
    // knot-table / B-spline traversal which falls back to CPU.
    let trivial = rows
        .iter()
        .all(|r| r.beta_h.is_none() && r.beta_w.is_none());
    if !trivial {
        return Ok(None);
    }
    device_dispatch::partition_cells_baseline(rows, 1.0)
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
/// `(a, b)` scalars, the per-cell slice from this row, and the layout of
/// the destination `FlexPrimarySlices` (`r = primary.total`, `g_slot =
/// primary.g`).
#[derive(Clone, Copy, Debug)]
pub struct CellPrimaryFixedPartialsRowInputs<'a> {
    pub a: f64,
    pub b: f64,
    pub cells: &'a [CellPrimaryFixedPartialsCellInputs],
    pub layout: FlexPrimaryLayout,
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

/// FlexPrimaryLayout constant for the fixed-partials kernel.
///
/// Mirrors the host `FlexPrimarySlices` shape that the family passes into
/// the CPU per-cell partials helper.  Held in the device-side closure
/// because the trivial kernel only needs `r` and the `g` slot index.
#[derive(Clone, Copy, Debug)]
pub struct FlexPrimaryLayout {
    pub r: u32,
    pub g_slot: u32,
}

/// GPU-shaped seam for
/// `SurvivalMarginalSlopeFamily::denested_cell_primary_fixed_partials`.
///
/// Returns `Ok(None)` when the input shape is outside the supported
/// regime (any non-zero score/link span — i.e. a runtime that needs the
/// full B-spline basis traversal — or no cells at all).
///
/// When the caller passes only cells whose `score_span` and `link_span`
/// are the no-runtime zero spans, the kernel evaluates the closed-form
/// trivial-cell partials on-device and returns the flat-packed layout.
pub fn try_device_cell_primary_fixed_partials(
    rows: &[CellPrimaryFixedPartialsRowInputs<'_>],
) -> Result<Option<CellPrimaryFixedPartialsOutput>, GpuError> {
    if rows.is_empty() {
        return Ok(Some(CellPrimaryFixedPartialsOutput::default()));
    }
    // We can only run the device kernel when every cell's spans are the
    // zero-span (no-runtime) baseline, because the trivial kernel doesn't
    // carry the knot tables needed for a non-trivial basis traversal.
    let trivial_spans = rows.iter().all(|row| {
        row.cells
            .iter()
            .all(|cell| span_is_zero(cell.score_span) && span_is_zero(cell.link_span))
    });
    if !trivial_spans {
        return Ok(None);
    }
    // The trivial kernel requires every row's layout to share the same
    // `(r, g_slot)` so a single launch can emit a uniform per-cell stride.
    // Differing layouts → decline (CPU fallback per row).
    let layout0 = rows[0].layout;
    if !rows
        .iter()
        .all(|r| r.layout.r == layout0.r && r.layout.g_slot == layout0.g_slot)
    {
        return Ok(None);
    }
    // If no cells at all, return an empty partials shape that matches
    // `rows.len()` so the caller can index into the result.
    let mut row_cell_counts: Vec<usize> = rows.iter().map(|r| r.cells.len()).collect();
    let total_cells: usize = row_cell_counts.iter().copied().sum();
    if total_cells == 0 {
        let mut partials: Vec<Vec<Vec<f64>>> = Vec::with_capacity(rows.len());
        for _ in 0..rows.len() {
            partials.push(Vec::new());
        }
        return Ok(Some(CellPrimaryFixedPartialsOutput { partials }));
    }
    let flat = match device_dispatch::cell_primary_fixed_partials_baseline(layout0, total_cells) {
        Ok(flat) => flat,
        Err(_) => return Ok(None),
    };
    let per_cell = 12usize + 40usize * (layout0.r as usize);
    let mut partials: Vec<Vec<Vec<f64>>> = Vec::with_capacity(rows.len());
    let mut cursor = 0usize;
    for n_cells in row_cell_counts.drain(..) {
        let mut row_cells: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
        for _ in 0..n_cells {
            row_cells.push(flat[cursor..cursor + per_cell].to_vec());
            cursor += per_cell;
        }
        partials.push(row_cells);
    }
    assert_eq!(cursor, flat.len());
    Ok(Some(CellPrimaryFixedPartialsOutput { partials }))
}

#[inline]
fn span_is_zero(span: LocalSpanCubic) -> bool {
    span.c0 == 0.0 && span.c1 == 0.0 && span.c2 == 0.0 && span.c3 == 0.0
}

/// Construct the trivial no-runtime partition cell for `(a, b, scale)`.
/// Used as the byte-equivalent host shape for the kernel's per-row output
/// (and as the reference the kernel reproduces).
pub fn trivial_partition_cell(a: f64, b: f64, scale: f64) -> DenestedPartitionCell {
    DenestedPartitionCell {
        cell: DenestedCubicCell {
            left: f64::NEG_INFINITY,
            right: f64::INFINITY,
            c0: a * scale,
            c1: b * scale,
            c2: 0.0,
            c3: 0.0,
        },
        score_span: LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        },
        link_span: LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        },
        left_edge: crate::families::cubic_cell_kernel::PartitionEdge::Fixed(f64::NEG_INFINITY),
        right_edge: crate::families::cubic_cell_kernel::PartitionEdge::Fixed(f64::INFINITY),
    }
}

#[cfg(target_os = "linux")]
mod device_dispatch {
    use super::kernel_src::DENESTED_PARTITION_CELLS_KERNEL_SRC;
    use super::{PartitionCellsOutput, PartitionCellsRowInputs, trivial_partition_cell};
    use crate::gpu::common::PtxModuleCache;
    use crate::gpu::error::{GpuError, GpuResultExt};
    use crate::gpu::solver::context_and_stream;
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    static PARTITION_PTX_CACHE: PtxModuleCache = PtxModuleCache::new();

    const THREADS_PER_BLOCK: u32 = 128;

    /// Launch the partition-cells kernel for the no-runtime baseline.
    pub(super) fn partition_cells_baseline(
        rows: &[PartitionCellsRowInputs<'_>],
        scale: f64,
    ) -> Result<Option<PartitionCellsOutput>, GpuError> {
        let n = rows.len();
        let n_u32 = u32::try_from(n)
            .map_err(|_| crate::gpu_err!("partition_cells_baseline: n_rows={n} exceeds u32"))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| crate::gpu_err!("partition_cells_baseline: n_rows={n} exceeds i32"))?;
        let (ctx, stream) = match context_and_stream() {
            Ok(pair) => pair,
            Err(_) => return Ok(None),
        };
        let module = PARTITION_PTX_CACHE.get_or_compile(
            &ctx,
            "survival_flex_prep::partition_cells",
            DENESTED_PARTITION_CELLS_KERNEL_SRC,
        )?;
        let func = module
            .load_function("denested_partition_cells_kernel")
            .gpu_ctx("survival_flex_prep: load_function partition_cells")?;

        let a_host: Vec<f64> = rows.iter().map(|r| r.a).collect();
        let b_host: Vec<f64> = rows.iter().map(|r| r.b).collect();
        let a_dev = stream
            .clone_htod(&a_host)
            .gpu_ctx("survival_flex_prep: upload a_per_row")?;
        let b_dev = stream
            .clone_htod(&b_host)
            .gpu_ctx("survival_flex_prep: upload b_per_row")?;
        let mut cells_dev = stream
            .alloc_zeros::<f64>(n * 18)
            .gpu_ctx("survival_flex_prep: alloc cells_flat")?;
        let mut offsets_dev = stream
            .alloc_zeros::<u32>(n + 1)
            .gpu_ctx("survival_flex_prep: alloc row_offsets")?;
        let mut status_dev = stream
            .alloc_zeros::<u8>(n)
            .gpu_ctx("survival_flex_prep: alloc status")?;

        let cfg = LaunchConfig {
            grid_dim: (n_u32.div_ceil(THREADS_PER_BLOCK).max(1), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: kernel signature is fixed in the source string above
        // (n:i32, scale:f64, 2 const f64*, 1 mut f64*, 1 mut u32*, 1 mut u8*).
        // All buffers are sized to the kernel's per-row stride, and each
        // thread guards i >= n_rows.
        unsafe {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&n_i32);
            builder.arg(&scale);
            builder.arg(&a_dev);
            builder.arg(&b_dev);
            builder.arg(&mut cells_dev);
            builder.arg(&mut offsets_dev);
            builder.arg(&mut status_dev);
            builder.launch(cfg)
        }
        .map(|_event_pair| ())
        .gpu_ctx("survival_flex_prep: launch partition_cells")?;

        let cells_host = stream
            .clone_dtoh(&cells_dev)
            .gpu_ctx("survival_flex_prep: download cells_flat")?;
        let status_host = stream
            .clone_dtoh(&status_dev)
            .gpu_ctx("survival_flex_prep: download status")?;
        for (i, st) in status_host.iter().enumerate() {
            if *st != 0 {
                return Err(crate::gpu_err!(
                    "survival_flex_prep: row {i} status={st} from device kernel"
                ));
            }
        }
        assert_eq!(cells_host.len(), n * 18);
        // Reconstruct per-row Vec<DenestedPartitionCell>.  The kernel writes
        // exactly one cell per row in the trivial baseline; we reproduce
        // the host trivial cell shape (using the kernel-written numerics
        // for c0/c1) and ignore the device-written infinity sentinels in
        // favour of the host-typed `f64::INFINITY` constants — both encode
        // bit-identical infinities, so this is a presentation-only step.
        let mut out: PartitionCellsOutput = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 18;
            let c0 = cells_host[base + 2];
            let c1 = cells_host[base + 3];
            let mut cell = trivial_partition_cell(rows[i].a, rows[i].b, scale);
            // Use the device-computed (a*scale, b*scale) so any future
            // scale plumbing is faithfully reflected.
            cell.cell.c0 = c0;
            cell.cell.c1 = c1;
            out.push(vec![cell]);
        }
        Ok(Some(out))
    }

    /// Launch the fixed-partials kernel for the no-runtime baseline.
    ///
    /// Returns the flat-packed `(12 + 40·r) · n_cells_total` doubles per
    /// the layout described in
    /// `kernel_src::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC`.
    /// The caller re-packs into `CellPrimaryFixedPartialsOutput` and the
    /// family-side consumer rebuilds `DenestedCellPrimaryFixedPartials`
    /// via `from_flat_slice`.
    pub(super) fn cell_primary_fixed_partials_baseline(
        layout: super::FlexPrimaryLayout,
        n_cells_total: usize,
    ) -> Result<Vec<f64>, GpuError> {
        use super::kernel_src::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC;
        static FP_PTX_CACHE: PtxModuleCache = PtxModuleCache::new();

        let n_i32 = i32::try_from(n_cells_total).map_err(|_| {
            crate::gpu_err!(
                "cell_primary_fixed_partials_baseline: n_cells={n_cells_total} exceeds i32"
            )
        })?;
        let n_u32 = u32::try_from(n_cells_total).map_err(|_| {
            crate::gpu_err!(
                "cell_primary_fixed_partials_baseline: n_cells={n_cells_total} exceeds u32"
            )
        })?;
        let (ctx, stream) = context_and_stream()
            .map_err(|reason| crate::gpu::error::GpuError::DriverCallFailed { reason })?;
        let module = FP_PTX_CACHE.get_or_compile(
            &ctx,
            "survival_flex_prep::cell_primary_fixed_partials",
            DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC,
        )?;
        let func = module
            .load_function("denested_cell_primary_fixed_partials_kernel")
            .gpu_ctx("survival_flex_prep: load_function fixed_partials")?;

        let per_cell = 12usize + 40usize * (layout.r as usize);
        let scale = 1.0f64;
        let mut out_dev = stream
            .alloc_zeros::<f64>(n_cells_total * per_cell)
            .gpu_ctx("survival_flex_prep: alloc fixed_partials")?;
        let mut status_dev = stream
            .alloc_zeros::<u8>(n_cells_total)
            .gpu_ctx("survival_flex_prep: alloc fixed_partials status")?;
        let cfg = LaunchConfig {
            grid_dim: (n_u32.div_ceil(THREADS_PER_BLOCK).max(1), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: kernel signature matches (n:i32, r:u32, g_slot:u32,
        // scale:f64, mut f64*, mut u8*).  Buffer sized to per-cell stride.
        unsafe {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&n_i32);
            builder.arg(&layout.r);
            builder.arg(&layout.g_slot);
            builder.arg(&scale);
            builder.arg(&mut out_dev);
            builder.arg(&mut status_dev);
            builder.launch(cfg)
        }
        .map(|_event_pair| ())
        .gpu_ctx("survival_flex_prep: launch fixed_partials")?;
        let out_host = stream
            .clone_dtoh(&out_dev)
            .gpu_ctx("survival_flex_prep: download fixed_partials")?;
        let status_host = stream
            .clone_dtoh(&status_dev)
            .gpu_ctx("survival_flex_prep: download fixed_partials status")?;
        for (i, st) in status_host.iter().enumerate() {
            if *st != 0 {
                return Err(crate::gpu_err!(
                    "survival_flex_prep: fixed_partials cell {i} status={st}"
                ));
            }
        }
        Ok(out_host)
    }
}

#[cfg(not(target_os = "linux"))]
mod device_dispatch {
    use super::{PartitionCellsOutput, PartitionCellsRowInputs};
    use crate::gpu::error::GpuError;

    pub(super) fn partition_cells_baseline(
        rows: &[PartitionCellsRowInputs<'_>],
        scale: f64,
    ) -> Result<Option<PartitionCellsOutput>, GpuError> {
        // CUDA only supported on linux; the caller falls back to CPU.
        // The scalar inputs are surfaced in the diagnostic-but-not-error
        // log so callers can still see what shape would have launched.
        log::trace!(
            "survival_flex_prep::partition_cells_baseline declined on non-linux \
             (n_rows={}, scale={scale})",
            rows.len()
        );
        Ok(None)
    }

    pub(super) fn cell_primary_fixed_partials_baseline(
        layout: super::FlexPrimaryLayout,
        n_cells_total: usize,
    ) -> Result<Vec<f64>, GpuError> {
        Err(crate::gpu_err!(
            "survival_flex_prep::cell_primary_fixed_partials_baseline: CUDA only supported on linux \
             (would have launched n_cells={n_cells_total}, r={}, g_slot={})",
            layout.r,
            layout.g_slot
        ))
    }
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
    fn nonempty_partition_with_betas_declines() {
        let beta = [0.0_f64];
        let inputs = [PartitionCellsRowInputs {
            a: 0.0,
            b: 1.0,
            beta_h: Some(&beta),
            beta_w: None,
        }];
        let out = try_device_partition_cells(&inputs).expect("ok");
        // Must decline because beta_h is present (B-spline runtime traversal
        // is not implemented in the trivial kernel).
        assert!(out.is_none());
    }

    #[test]
    fn empty_fixed_partials_inputs_short_circuit() {
        let out = try_device_cell_primary_fixed_partials(&[]).expect("ok");
        assert!(out.is_some());
        assert!(out.unwrap().partials.is_empty());
    }

    #[test]
    fn empty_cells_per_row_returns_empty_partials() {
        let inputs = [CellPrimaryFixedPartialsRowInputs {
            a: 0.0,
            b: 1.0,
            cells: &[],
            layout: FlexPrimaryLayout { r: 4, g_slot: 3 },
        }];
        let out = try_device_cell_primary_fixed_partials(&inputs).expect("ok");
        let some = out.expect("Some when all rows have zero cells");
        assert_eq!(some.partials.len(), 1);
        assert!(some.partials[0].is_empty());
    }

    #[test]
    fn kernel_src_strings_are_nonempty() {
        assert!(!kernel_src::DENESTED_PARTITION_CELLS_KERNEL_SRC.is_empty());
        assert!(!kernel_src::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC.is_empty());
    }

    #[test]
    fn trivial_partition_cell_matches_cpu_empty_split_branch() {
        // For a=2.5, b=-1.25, scale=1.0 the empty-split-points branch of
        // build_denested_partition_cells_with_tails produces a single
        // affine cell with c0=a, c1=b (post-scale).
        let cell = trivial_partition_cell(2.5, -1.25, 1.0);
        assert_eq!(cell.cell.c0, 2.5);
        assert_eq!(cell.cell.c1, -1.25);
        assert_eq!(cell.cell.c2, 0.0);
        assert_eq!(cell.cell.c3, 0.0);
        assert!(cell.cell.left.is_infinite() && cell.cell.left.is_sign_negative());
        assert!(cell.cell.right.is_infinite() && cell.cell.right.is_sign_positive());
    }
}
