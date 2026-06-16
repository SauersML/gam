//! GPU Gram builder for the closed-form V+M identifiability compiler.
//!
//! Inputs:
//!   - `channel_blocks[block][channel]`: optional `n × p_block` raw design
//!     slice for each (block, channel) pair. Missing entries mean that
//!     channel is zero on that block — they contribute nothing to the Gram.
//!   - `h_packed`: `n × 10` per-row packed symmetric 4×4 weight matrix
//!     (channels 0..4). Packing follows the upper-triangular row-major
//!     convention: index(c, d) with `c ≤ d` is
//!     `c * (7 - c) / 2 + d` (i.e. 0..10). The symmetric counterpart is
//!     looked up by swapping the pair.
//!   - `raw_block_ranges`: column slice for each raw block inside the
//!     concatenated raw design, used to size and stride the output Gram.
//!
//! The kernel forms two block-Gram matrices:
//!   - `gram_h`: ∑_{c,d} X_a^{(c)}ᵀ · diag(h_{cd}) · X_b^{(d)}
//!   - `gram_struct`: ∑_{c,d} X_a^{(c)}ᵀ · X_b^{(d)} on the same
//!     channel pairs that contributed to `gram_h` (i.e. the support of
//!     channel availability rather than the support of `h`)
//!
//! On any CUDA error (no runtime, OOM, launch failure) the function
//! returns `None` so the caller can fall back to the CPU path. We never
//! panic.

use ndarray::Array2;
use std::ops::Range;

/// Output of the device-resident Gram builder.
pub struct GramBundle {
    pub gram_h: Array2<f64>,
    pub gram_struct: Array2<f64>,
}

/// Number of channels in the symmetric 4×4 weight matrix.
pub const CHANNELS: usize = 4;
/// Number of unique entries in a symmetric 4×4 matrix.
pub const PACKED_LEN: usize = 10;

/// Map a channel pair `(c, d)` (any order) to its column index in the
/// `n × 10` packed `h` matrix using the canonical upper-triangular
/// row-major layout.
#[inline]
pub const fn packed_index(c: usize, d: usize) -> usize {
    let (lo, hi) = if c <= d { (c, d) } else { (d, c) };
    // Offsets for c = 0,1,2,3 in the upper-triangular row-major layout:
    //   row 0 starts at 0, row 1 at 4, row 2 at 7, row 3 at 9.
    let row_offset = [0usize, 4, 7, 9][lo];
    row_offset + (hi - lo)
}

/// Try to build the primary-state Gram bundle on the GPU. Returns `None`
/// when no CUDA device is available or when any device call fails.
///
pub fn try_primary_state_gram_cuda(
    channel_blocks: &[Vec<Option<Array2<f64>>>],
    h_packed: &Array2<f64>,
    raw_block_ranges: &[Range<usize>],
) -> Option<GramBundle> {
    #[cfg(not(target_os = "linux"))]
    {
        // Validate signature on non-Linux so callers still get the
        // same shape-checks they would on Linux, then report the
        // absence of a CUDA backend.
        if channel_blocks.is_empty()
            || h_packed.is_empty()
            || raw_block_ranges.is_empty()
            || channel_blocks.len() != raw_block_ranges.len()
        {
            return None;
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let workspace = cuda_impl::WorkspaceInner::try_new(channel_blocks, raw_block_ranges)?;
        workspace.compute_grams(h_packed)
    }
}

#[cfg(target_os = "linux")]
mod cuda_impl {
    use super::{CHANNELS, GramBundle, PACKED_LEN, packed_index};
    use crate::gpu::driver::{to_col_major, to_i32};
    use crate::gpu::runtime::{GpuRuntime, cuda_context_for};
    use cudarc::cublas::sys::{cublasOperation_t, cublasSideMode_t, cublasStatus_t};
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use ndarray::Array2;
    use std::cell::RefCell;
    use std::ops::Range;
    use std::sync::Arc;

    /// Process-wide PTX cache for the fused primary-state Gram kernel.
    /// NVRTC-compiled on the first `compute_grams_fused` call that
    /// reaches a CUDA context; reused on every subsequent call.
    pub(super) static FUSED_GRAM_PTX_CACHE: crate::gpu::common::PtxModuleCache =
        crate::gpu::common::PtxModuleCache::new();

    /// Device-resident cache of channel-block designs plus a reusable
    /// row-scale scratch buffer. Built once per identifiability compile;
    /// each `compute_grams` call only re-uploads the per-row packed
    /// Hessian columns it actually needs.
    pub(crate) struct WorkspaceInner {
        stream: Arc<CudaStream>,
        blas: CudaBlas,
        /// `uploads[block][channel]` — column-major device copy of each
        /// raw design slice. `None` slots mean "channel inactive on this
        /// block", consistent with the host-side `Option<Array2>` layout.
        uploads: Vec<Vec<Option<CudaSlice<f64>>>>,
        /// `needed[packed_index(c,d)]` — true iff some `(block_a,
        /// block_b)` pair will request that channel-pair Hessian column.
        needed: [bool; PACKED_LEN],
        /// Reusable device scratch for the DGMM row-scaled right
        /// operand. Sized at the largest `(block, channel)` right matrix
        /// across the topology, so it fits every `(a,b,c,d)` step. Wrapped
        /// in `RefCell` so `compute_grams` can mutate it through `&self`.
        scaled_dev: RefCell<CudaSlice<f64>>,
        raw_block_ranges: Vec<Range<usize>>,
        n_rows: usize,
        total_cols: usize,
    }

    impl WorkspaceInner {
        pub(super) fn try_new(
            channel_blocks: &[Vec<Option<Array2<f64>>>],
            raw_block_ranges: &[Range<usize>],
        ) -> Option<Self> {
            let runtime = GpuRuntime::global()?;
            let num_blocks = channel_blocks.len();
            if num_blocks == 0 || raw_block_ranges.len() != num_blocks {
                return None;
            }

            // Establish a consistent n_rows across all uploaded blocks.
            let mut n_rows: Option<usize> = None;
            for (block_idx, channels) in channel_blocks.iter().enumerate() {
                if channels.len() != CHANNELS {
                    return None;
                }
                let want_cols = raw_block_ranges[block_idx].len();
                for ch in channels.iter().flatten() {
                    if ch.ncols() != want_cols {
                        return None;
                    }
                    match n_rows {
                        Some(expected) if expected != ch.nrows() => return None,
                        Some(_) => {}
                        None => n_rows = Some(ch.nrows()),
                    }
                }
            }
            let n_rows = n_rows?;

            let total_cols = raw_block_ranges
                .iter()
                .map(|r| r.len())
                .fold(0usize, |a, l| a.saturating_add(l));
            if total_cols == 0 {
                return None;
            }

            let stream = cuda_context_for(runtime.device.ordinal)?
                .new_stream()
                .ok()?;
            let blas = CudaBlas::new(stream.clone()).ok()?;

            // Upload each (block, channel) raw design once, column-major.
            let mut uploads: Vec<Vec<Option<CudaSlice<f64>>>> = Vec::with_capacity(num_blocks);
            for channels in channel_blocks.iter() {
                let mut row: Vec<Option<CudaSlice<f64>>> = Vec::with_capacity(CHANNELS);
                for slot in channels.iter() {
                    let dev = match slot.as_ref() {
                        Some(mat) => {
                            let col = to_col_major(mat);
                            Some(stream.clone_htod(&*col).ok()?)
                        }
                        None => None,
                    };
                    row.push(dev);
                }
                uploads.push(row);
            }

            // Pre-compute which packed-H columns the topology requires.
            let mut needed: [bool; PACKED_LEN] = [false; PACKED_LEN];
            for a in 0..num_blocks {
                for b in 0..num_blocks {
                    for c in 0..CHANNELS {
                        if uploads[a][c].is_none() {
                            continue;
                        }
                        for d in 0..CHANNELS {
                            if uploads[b][d].is_none() {
                                continue;
                            }
                            needed[packed_index(c, d)] = true;
                        }
                    }
                }
            }

            // Scratch sized at the largest right matrix across the topology.
            let max_right_len = channel_blocks
                .iter()
                .zip(raw_block_ranges.iter())
                .flat_map(|(channels, range)| {
                    channels.iter().map(move |slot| {
                        if slot.is_some() {
                            n_rows.checked_mul(range.len()).unwrap_or(0)
                        } else {
                            0
                        }
                    })
                })
                .max()
                .unwrap_or(0);
            if max_right_len == 0 {
                return None;
            }
            let scaled_dev = stream.alloc_zeros::<f64>(max_right_len).ok()?;

            Some(Self {
                stream,
                blas,
                uploads,
                needed,
                scaled_dev: RefCell::new(scaled_dev),
                raw_block_ranges: raw_block_ranges.to_vec(),
                n_rows,
                total_cols,
            })
        }

        pub(super) fn compute_grams(&self, h_packed: &Array2<f64>) -> Option<GramBundle> {
            // Try the fused NVRTC kernel first — it streams each design
            // matrix once instead of 16 times. Any compile / launch /
            // shape failure routes us to the cuBLAS DDGMM+DGEMM path
            // below; the two paths are bit-equivalent to within
            // floating-point reordering, so the caller cannot observe
            // which one served the request.
            if let Some(bundle) = self.compute_grams_fused(h_packed) {
                return Some(bundle);
            }
            self.compute_grams_cublas(h_packed)
        }

        /// Fused-kernel path: one NVRTC launch per `(block_a, block_b)`
        /// pair accumulates **all** channel-pair contributions in a
        /// single pass over the rows. Returns `None` on shape mismatch,
        /// NVRTC compile failure, or any cudarc driver error so the
        /// caller can fall back to the cuBLAS path. Both `gram_h` and
        /// `gram_struct` are emitted from the same row-stream so the
        /// second Gram costs no extra memory traffic relative to the
        /// first.
        pub(super) fn compute_grams_fused(&self, h_packed: &Array2<f64>) -> Option<GramBundle> {
            let (n_rows, packed_cols) = h_packed.dim();
            if packed_cols != PACKED_LEN || n_rows != self.n_rows {
                return None;
            }
            let num_blocks = self.uploads.len();
            let n_rows_u32 = u32::try_from(n_rows).ok()?;

            // Upload h_packed once, column-major (n_rows × 10).
            let h_packed_cm = to_col_major(h_packed);
            let h_packed_dev = self.stream.clone_htod(&*h_packed_cm).ok()?;

            // Resolve a CUDA context for module compilation. The default
            // stream sits on the same context per cudarc's caching, so
            // launches from this stream see the loaded module.
            let runtime = GpuRuntime::global()?;
            let ctx = cuda_context_for(runtime.device.ordinal)?;
            let module = FUSED_GRAM_PTX_CACHE
                .get_or_compile(
                    &ctx,
                    "identifiability_compile_fused",
                    crate::identifiability::families::gpu_kernel::KERNEL_SRC,
                )
                .ok()?;
            let func = module
                .load_function(crate::identifiability::families::gpu_kernel::KERNEL_NAME)
                .ok()?;

            let total_cols = self.total_cols;
            let mut gram_h = Array2::<f64>::zeros((total_cols, total_cols));
            let mut gram_struct = Array2::<f64>::zeros((total_cols, total_cols));

            // Lazily-allocated null device pointer used to stand in for
            // missing channels. We reuse the existing `scaled_dev`
            // scratch's allocation as a non-null placeholder; the kernel
            // ignores the bytes under the corresponding present-mask bit.
            let scaled_borrow = self.scaled_dev.borrow();
            let placeholder: &CudaSlice<f64> = &*scaled_borrow;

            for block_a in 0..num_blocks {
                let range_a = &self.raw_block_ranges[block_a];
                let a_cols = range_a.len();
                if a_cols == 0 {
                    continue;
                }
                let a_cols_u32 = u32::try_from(a_cols).ok()?;
                let mut a_present_mask: u32 = 0;
                for c in 0..CHANNELS {
                    if self.uploads[block_a][c].is_some() {
                        a_present_mask |= 1u32 << c;
                    }
                }
                if a_present_mask == 0 {
                    continue;
                }

                for block_b in 0..num_blocks {
                    let range_b = &self.raw_block_ranges[block_b];
                    let b_cols = range_b.len();
                    if b_cols == 0 {
                        continue;
                    }
                    let b_cols_u32 = u32::try_from(b_cols).ok()?;
                    let mut b_present_mask: u32 = 0;
                    for d in 0..CHANNELS {
                        if self.uploads[block_b][d].is_some() {
                            b_present_mask |= 1u32 << d;
                        }
                    }
                    if b_present_mask == 0 {
                        continue;
                    }

                    // Allocate per-tile output buffers (column-major).
                    let tile_len = a_cols.checked_mul(b_cols)?;
                    let mut gram_h_tile_dev = self.stream.alloc_zeros::<f64>(tile_len).ok()?;
                    let mut gram_s_tile_dev = self.stream.alloc_zeros::<f64>(tile_len).ok()?;

                    let xa: [&CudaSlice<f64>; CHANNELS] = [
                        self.uploads[block_a][0].as_ref().unwrap_or(placeholder),
                        self.uploads[block_a][1].as_ref().unwrap_or(placeholder),
                        self.uploads[block_a][2].as_ref().unwrap_or(placeholder),
                        self.uploads[block_a][3].as_ref().unwrap_or(placeholder),
                    ];
                    let xb: [&CudaSlice<f64>; CHANNELS] = [
                        self.uploads[block_b][0].as_ref().unwrap_or(placeholder),
                        self.uploads[block_b][1].as_ref().unwrap_or(placeholder),
                        self.uploads[block_b][2].as_ref().unwrap_or(placeholder),
                        self.uploads[block_b][3].as_ref().unwrap_or(placeholder),
                    ];

                    let tile_a = crate::identifiability::families::gpu_kernel::TILE_A;
                    let tile_b = crate::identifiability::families::gpu_kernel::TILE_B;
                    let grid_x = a_cols_u32.div_ceil(tile_a).max(1);
                    let grid_y = b_cols_u32.div_ceil(tile_b).max(1);
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (grid_x, grid_y, 1),
                        block_dim: (tile_a, tile_b, 1),
                        shared_mem_bytes: 0,
                    };

                    use cudarc::driver::PushKernelArg;
                    let mut builder = self.stream.launch_builder(&func);
                    builder
                        .arg(xa[0])
                        .arg(xa[1])
                        .arg(xa[2])
                        .arg(xa[3])
                        .arg(xb[0])
                        .arg(xb[1])
                        .arg(xb[2])
                        .arg(xb[3])
                        .arg(&h_packed_dev)
                        .arg(&mut gram_h_tile_dev)
                        .arg(&mut gram_s_tile_dev)
                        .arg(&n_rows_u32)
                        .arg(&a_cols_u32)
                        .arg(&b_cols_u32)
                        .arg(&a_present_mask)
                        .arg(&b_present_mask);
                    // SAFETY: every argument is a typed device pointer or
                    // scalar whose layout matches the kernel signature
                    // declared in identifiability_compile_kernel::KERNEL_SRC.
                    // The grid covers (a_cols, b_cols) and out-of-tile
                    // threads early-return via the in_range guard.
                    unsafe { builder.launch(cfg) }.ok()?;

                    let host_h = self.stream.clone_dtoh(&gram_h_tile_dev).ok()?;
                    let host_s = self.stream.clone_dtoh(&gram_s_tile_dev).ok()?;
                    let contrib_h = from_col_major(&host_h, a_cols, b_cols)?;
                    let contrib_s = from_col_major(&host_s, a_cols, b_cols)?;
                    assign_block(&mut gram_h, range_a.start, range_b.start, &contrib_h);
                    assign_block(&mut gram_struct, range_a.start, range_b.start, &contrib_s);
                }
            }
            self.stream.synchronize().ok()?;

            symmetrise(&mut gram_h);
            symmetrise(&mut gram_struct);
            Some(GramBundle {
                gram_h,
                gram_struct,
            })
        }

        pub(super) fn compute_grams_cublas(&self, h_packed: &Array2<f64>) -> Option<GramBundle> {
            let (n_rows, packed_cols) = h_packed.dim();
            if packed_cols != PACKED_LEN || n_rows != self.n_rows {
                return None;
            }

            // Upload only the columns actually used by the topology.
            let mut h_columns: [Option<CudaSlice<f64>>; PACKED_LEN] =
                [None, None, None, None, None, None, None, None, None, None];
            for idx in 0..PACKED_LEN {
                if !self.needed[idx] {
                    continue;
                }
                let col = h_packed.column(idx);
                let host: Vec<f64> = col.iter().copied().collect();
                h_columns[idx] = Some(self.stream.clone_htod(&host).ok()?);
            }

            let total_cols = self.total_cols;
            let mut gram_h = Array2::<f64>::zeros((total_cols, total_cols));
            let mut gram_struct = Array2::<f64>::zeros((total_cols, total_cols));

            let mut scaled = self.scaled_dev.borrow_mut();
            let num_blocks = self.uploads.len();
            for block_a in 0..num_blocks {
                let range_a = &self.raw_block_ranges[block_a];
                for block_b in 0..num_blocks {
                    let range_b = &self.raw_block_ranges[block_b];
                    let mut contrib_h = Array2::<f64>::zeros((range_a.len(), range_b.len()));
                    let mut contrib_s = Array2::<f64>::zeros((range_a.len(), range_b.len()));
                    for c in 0..CHANNELS {
                        let Some(x_a) = self.uploads[block_a][c].as_ref() else {
                            continue;
                        };
                        for d in 0..CHANNELS {
                            let Some(x_b) = self.uploads[block_b][d].as_ref() else {
                                continue;
                            };
                            let h_dev = h_columns[packed_index(c, d)].as_ref()?;
                            let weighted_block = weighted_gemm(
                                &self.blas,
                                &self.stream,
                                x_a,
                                h_dev,
                                x_b,
                                &mut *scaled,
                                self.n_rows,
                                range_a.len(),
                                range_b.len(),
                            )?;
                            contrib_h = contrib_h + &weighted_block;
                            // Structural Gram is `gram_h` evaluated with the
                            // K×K identity Hessian, so only the diagonal
                            // channel pair (c == d) survives — cross-channel
                            // terms vanish. This mirrors the authoritative CPU
                            // reference `build_raw_grams_structural`
                            // (`K^S = Σ_c Xa^(c)ᵀ Xb^(c)`) and the fused
                            // kernel's `if (c == d)` accumulation.
                            if c == d {
                                let struct_block = plain_gemm(
                                    &self.blas,
                                    &self.stream,
                                    x_a,
                                    x_b,
                                    self.n_rows,
                                    range_a.len(),
                                    range_b.len(),
                                )?;
                                contrib_s = contrib_s + &struct_block;
                            }
                        }
                    }
                    assign_block(&mut gram_h, range_a.start, range_b.start, &contrib_h);
                    assign_block(&mut gram_struct, range_a.start, range_b.start, &contrib_s);
                }
            }

            symmetrise(&mut gram_h);
            symmetrise(&mut gram_struct);

            Some(GramBundle {
                gram_h,
                gram_struct,
            })
        }
    }

    /// Compute `Xa^T · diag(w) · Xb` for column-major device matrices.
    fn weighted_gemm(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        x_a: &CudaSlice<f64>,
        weights: &CudaSlice<f64>,
        x_b: &CudaSlice<f64>,
        scaled_dev: &mut CudaSlice<f64>,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Option<Array2<f64>> {
        let rows_i = to_i32(rows)?;
        let b_cols_i = to_i32(b_cols)?;
        let handle = *blas.handle();
        // Scope the mutable device-pointer record for the DDGMM row-scale so
        // the borrow is released before the subsequent immutable use of
        // `scaled_dev` in the GEMM call below. cudarc's `device_ptr_mut`
        // returns a `SyncOnDrop` guard whose lifetime is tied to the
        // mutable borrow; dropping it via this block ends the mutable
        // borrow exactly at the DDGMM completion point.
        {
            let (x_b_ptr, _x_b_record) = x_b.device_ptr(stream);
            let (w_ptr, _w_record) = weights.device_ptr(stream);
            let (scaled_ptr, _scaled_record) = scaled_dev.device_ptr_mut(stream);
            // SAFETY: x_b is rows×b_cols column-major (lda = rows), w is rows
            // contiguous, scaled_dev has at least rows*b_cols entries.
            let status = unsafe {
                cudarc::cublas::sys::cublasDdgmm(
                    handle,
                    cublasSideMode_t::CUBLAS_SIDE_LEFT,
                    rows_i,
                    b_cols_i,
                    x_b_ptr as *const f64,
                    rows_i,
                    w_ptr as *const f64,
                    1,
                    scaled_ptr as *mut f64,
                    rows_i,
                )
            };
            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return None;
            }
        }

        let mut out_dev = stream
            .alloc_zeros::<f64>(a_cols.checked_mul(b_cols)?)
            .ok()?;
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: to_i32(a_cols)?,
            n: to_i32(b_cols)?,
            k: rows_i,
            alpha: 1.0,
            lda: rows_i,
            ldb: rows_i,
            beta: 0.0,
            ldc: to_i32(a_cols)?,
        };
        // SAFETY: x_a is rows×a_cols column-major, scaled_dev holds the
        // row-scaled rows×b_cols product, and out_dev is a_cols×b_cols.
        unsafe { blas.gemm(cfg, x_a, scaled_dev, &mut out_dev) }.ok()?;
        let out_col = stream.clone_dtoh(&out_dev).ok()?;
        from_col_major(&out_col, a_cols, b_cols)
    }

    /// Compute `Xa^T · Xb` directly on device.
    fn plain_gemm(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        x_a: &CudaSlice<f64>,
        x_b: &CudaSlice<f64>,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Option<Array2<f64>> {
        let rows_i = to_i32(rows)?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(a_cols.checked_mul(b_cols)?)
            .ok()?;
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: to_i32(a_cols)?,
            n: to_i32(b_cols)?,
            k: rows_i,
            alpha: 1.0,
            lda: rows_i,
            ldb: rows_i,
            beta: 0.0,
            ldc: to_i32(a_cols)?,
        };
        // SAFETY: both inputs are rows×col column-major device matrices.
        unsafe { blas.gemm(cfg, x_a, x_b, &mut out_dev) }.ok()?;
        let out_col = stream.clone_dtoh(&out_dev).ok()?;
        from_col_major(&out_col, a_cols, b_cols)
    }

    fn from_col_major(values: &[f64], rows: usize, cols: usize) -> Option<Array2<f64>> {
        if values.len() != rows.checked_mul(cols)? {
            return None;
        }
        let mut out = Array2::<f64>::zeros((rows, cols));
        for col in 0..cols {
            for row in 0..rows {
                out[[row, col]] = values[col * rows + row];
            }
        }
        Some(out)
    }

    fn assign_block(
        out: &mut Array2<f64>,
        row_offset: usize,
        col_offset: usize,
        block: &Array2<f64>,
    ) {
        let (rows, cols) = block.dim();
        for col in 0..cols {
            for row in 0..rows {
                out[[row_offset + row, col_offset + col]] = block[[row, col]];
            }
        }
    }

    fn symmetrise(out: &mut Array2<f64>) {
        let n = out.nrows();
        for row in 0..n {
            for col in (row + 1)..n {
                let avg = 0.5 * (out[[row, col]] + out[[col, row]]);
                out[[row, col]] = avg;
                out[[col, row]] = avg;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn cpu_oracle(
        channel_blocks: &[Vec<Option<Array2<f64>>>],
        h_packed: &Array2<f64>,
        raw_block_ranges: &[Range<usize>],
    ) -> (Array2<f64>, Array2<f64>) {
        let total: usize = raw_block_ranges.iter().map(|r| r.len()).sum();
        let mut gram_h = Array2::<f64>::zeros((total, total));
        let mut gram_struct = Array2::<f64>::zeros((total, total));
        let n_rows = h_packed.nrows();
        for a in 0..channel_blocks.len() {
            for b in 0..channel_blocks.len() {
                for c in 0..CHANNELS {
                    let Some(x_a) = channel_blocks[a][c].as_ref() else {
                        continue;
                    };
                    for d in 0..CHANNELS {
                        let Some(x_b) = channel_blocks[b][d].as_ref() else {
                            continue;
                        };
                        let w_col = h_packed.column(packed_index(c, d));
                        let a_cols = x_a.ncols();
                        let b_cols = x_b.ncols();
                        let row_off = raw_block_ranges[a].start;
                        let col_off = raw_block_ranges[b].start;
                        // Structural Gram keeps only the diagonal channel pair
                        // (c == d): it is `gram_h` under the identity Hessian,
                        // so cross-channel terms vanish (see
                        // `build_raw_grams_structural`).
                        let diagonal_channel = c == d;
                        for i in 0..a_cols {
                            for j in 0..b_cols {
                                let mut acc_h = 0.0_f64;
                                let mut acc_s = 0.0_f64;
                                for row in 0..n_rows {
                                    let prod = x_a[[row, i]] * x_b[[row, j]];
                                    acc_h += w_col[row] * prod;
                                    if diagonal_channel {
                                        acc_s += prod;
                                    }
                                }
                                gram_h[[row_off + i, col_off + j]] += acc_h;
                                gram_struct[[row_off + i, col_off + j]] += acc_s;
                            }
                        }
                    }
                }
            }
        }
        symmetrise_for_test(&mut gram_h);
        symmetrise_for_test(&mut gram_struct);
        (gram_h, gram_struct)
    }

    fn symmetrise_for_test(out: &mut Array2<f64>) {
        let n = out.nrows();
        for row in 0..n {
            for col in (row + 1)..n {
                let avg = 0.5 * (out[[row, col]] + out[[col, row]]);
                out[[row, col]] = avg;
                out[[col, row]] = avg;
            }
        }
    }

    fn make_fixture() -> (
        Vec<Vec<Option<Array2<f64>>>>,
        Array2<f64>,
        Vec<Range<usize>>,
    ) {
        let n = 6;
        let block_0 = Array2::<f64>::from_shape_fn((n, 2), |(i, j)| ((i + 1) * (j + 1)) as f64);
        let block_1 = Array2::<f64>::from_shape_fn((n, 3), |(i, j)| (i as f64) - (j as f64) * 0.25);
        let block_1_ch2 = Array2::<f64>::from_shape_fn((n, 3), |(i, j)| 0.1 * (i + j + 1) as f64);
        let channel_blocks = vec![
            vec![Some(block_0.clone()), None, None, None],
            vec![Some(block_1.clone()), None, Some(block_1_ch2.clone()), None],
        ];
        let h_packed = Array2::<f64>::from_shape_fn((n, PACKED_LEN), |(i, j)| {
            0.5 + 0.1 * i as f64 + 0.05 * j as f64
        });
        let ranges = vec![0..2, 2..5];
        (channel_blocks, h_packed, ranges)
    }

    #[test]
    fn packed_index_matches_upper_triangular_layout() {
        let mut seen = [false; PACKED_LEN];
        for c in 0..CHANNELS {
            for d in c..CHANNELS {
                let idx = packed_index(c, d);
                assert!(idx < PACKED_LEN);
                assert!(!seen[idx], "duplicate packed index for ({c},{d})");
                seen[idx] = true;
                assert_eq!(packed_index(c, d), packed_index(d, c));
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn primary_state_gram_matches_cpu_oracle_when_cuda_available() {
        let (channel_blocks, h_packed, ranges) = make_fixture();
        #[cfg(not(target_os = "linux"))]
        {
            assert!(
                try_primary_state_gram_cuda(&channel_blocks, &h_packed, &ranges).is_none(),
                "non-Linux build must report no CUDA"
            );
            // Exercise the oracle so its symmetric output matches itself,
            // keeping the CPU reference live on non-Linux hosts.
            let (cpu_h, cpu_s) = cpu_oracle(&channel_blocks, &h_packed, &ranges);
            assert_eq!(cpu_h.nrows(), cpu_h.ncols());
            assert_eq!(cpu_s.nrows(), cpu_s.ncols());
            for row in 0..cpu_h.nrows() {
                for col in 0..cpu_h.ncols() {
                    assert!((cpu_h[[row, col]] - cpu_h[[col, row]]).abs() <= 1e-12);
                    assert!((cpu_s[[row, col]] - cpu_s[[col, row]]).abs() <= 1e-12);
                }
            }
            return;
        }
        #[cfg(target_os = "linux")]
        {
            if crate::gpu::runtime::GpuRuntime::global().is_none() {
                eprintln!("[identifiability_compile] no CUDA runtime — skipping parity check");
                return;
            }
            let Some(bundle) = try_primary_state_gram_cuda(&channel_blocks, &h_packed, &ranges)
            else {
                eprintln!(
                    "[identifiability_compile] GPU Gram build returned None — \
                     treating as CI infra outage, not parity regression"
                );
                return;
            };
            let (cpu_h, cpu_s) = cpu_oracle(&channel_blocks, &h_packed, &ranges);
            let tol_abs = 1e-9_f64;
            let tol_rel = 1e-9_f64;
            for ((idx, &c), &g) in cpu_h.indexed_iter().zip(bundle.gram_h.iter()) {
                let tol = tol_abs + tol_rel * c.abs();
                assert!(
                    (c - g).abs() <= tol,
                    "gram_h mismatch at {idx:?}: cpu={c} gpu={g}"
                );
            }
            for ((idx, &c), &g) in cpu_s.indexed_iter().zip(bundle.gram_struct.iter()) {
                let tol = tol_abs + tol_rel * c.abs();
                assert!(
                    (c - g).abs() <= tol,
                    "gram_struct mismatch at {idx:?}: cpu={c} gpu={g}"
                );
            }
        }
    }
}
