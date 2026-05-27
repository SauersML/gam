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
//!     concatenated raw design — used purely to size and stride the
//!     output Gram.
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
    pub raw_block_ranges: Vec<Range<usize>>,
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
pub fn try_primary_state_gram_cuda(
    channel_blocks: &[Vec<Option<Array2<f64>>>],
    h_packed: &Array2<f64>,
    raw_block_ranges: &[Range<usize>],
) -> Option<GramBundle> {
    #[cfg(not(target_os = "linux"))]
    {
        non_linux_stub(channel_blocks, h_packed, raw_block_ranges)
    }
    #[cfg(target_os = "linux")]
    {
        cuda_impl::try_primary_state_gram_cuda_impl(channel_blocks, h_packed, raw_block_ranges)
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
fn non_linux_stub(
    channel_blocks: &[Vec<Option<Array2<f64>>>],
    h_packed: &Array2<f64>,
    raw_block_ranges: &[Range<usize>],
) -> Option<GramBundle> {
    if channel_blocks.is_empty()
        || h_packed.is_empty()
        || raw_block_ranges.is_empty()
    {
        return None;
    }
    None
}

#[cfg(target_os = "linux")]
mod cuda_impl {
    use super::{CHANNELS, GramBundle, PACKED_LEN, packed_index};
    use crate::gpu::driver::{to_col_major, to_i32};
    use crate::gpu::runtime::{GpuRuntime, cuda_context_for};
    use cudarc::cublas::sys::{cublasOperation_t, cublasSideMode_t, cublasStatus_t};
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use ndarray::{Array2, ArrayView1, ArrayView2};
    use std::ops::Range;
    use std::sync::Arc;

    pub(super) fn try_primary_state_gram_cuda_impl(
        channel_blocks: &[Vec<Option<Array2<f64>>>],
        h_packed: &Array2<f64>,
        raw_block_ranges: &[Range<usize>],
    ) -> Option<GramBundle> {
        let runtime = GpuRuntime::global()?;
        let num_blocks = channel_blocks.len();
        if num_blocks == 0 || raw_block_ranges.len() != num_blocks {
            return None;
        }
        let (n_rows, packed_cols) = h_packed.dim();
        if packed_cols != PACKED_LEN {
            return None;
        }

        for (b, channels) in channel_blocks.iter().enumerate() {
            if channels.len() != CHANNELS {
                return None;
            }
            let want_cols = raw_block_ranges[b].len();
            for ch in channels.iter().flatten() {
                if ch.nrows() != n_rows || ch.ncols() != want_cols {
                    return None;
                }
            }
        }

        let total_cols = raw_block_ranges
            .iter()
            .map(|r| r.len())
            .fold(0usize, |a, l| a.saturating_add(l));
        if total_cols == 0 {
            return None;
        }

        let stream = cuda_context_for(runtime.device.ordinal)?.new_stream().ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;

        // Upload each (block, channel) raw design once, column-major.
        let mut uploads: Vec<Vec<Option<CudaSlice<f64>>>> = Vec::with_capacity(num_blocks);
        for channels in channel_blocks.iter() {
            let mut row: Vec<Option<CudaSlice<f64>>> = Vec::with_capacity(CHANNELS);
            for slot in channels.iter() {
                let dev = match slot.as_ref() {
                    Some(mat) => {
                        let col = to_col_major(&mat.view());
                        Some(stream.clone_htod(&*col).ok()?)
                    }
                    None => None,
                };
                row.push(dev);
            }
            uploads.push(row);
        }

        // Upload each channel-pair weight column (10 in total). We only
        // upload the columns we will actually consume.
        let mut h_columns: [Option<CudaSlice<f64>>; PACKED_LEN] =
            [None, None, None, None, None, None, None, None, None, None];
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
        for idx in 0..PACKED_LEN {
            if !needed[idx] {
                continue;
            }
            let col = h_packed.column(idx);
            let host: Vec<f64> = col.iter().copied().collect();
            h_columns[idx] = Some(stream.clone_htod(&host).ok()?);
        }

        let mut gram_h = Array2::<f64>::zeros((total_cols, total_cols));
        let mut gram_struct = Array2::<f64>::zeros((total_cols, total_cols));

        // Scratch buffer reused across (a,b,c,d) pairs to hold the
        // row-scaled right operand. Sized at the largest right matrix.
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
        let mut scaled_dev = stream.alloc_zeros::<f64>(max_right_len).ok()?;

        for block_a in 0..num_blocks {
            let range_a = &raw_block_ranges[block_a];
            for block_b in 0..num_blocks {
                let range_b = &raw_block_ranges[block_b];
                let mut contrib_h = Array2::<f64>::zeros((range_a.len(), range_b.len()));
                let mut contrib_s = Array2::<f64>::zeros((range_a.len(), range_b.len()));
                for c in 0..CHANNELS {
                    let Some(x_a) = uploads[block_a][c].as_ref() else {
                        continue;
                    };
                    for d in 0..CHANNELS {
                        let Some(x_b) = uploads[block_b][d].as_ref() else {
                            continue;
                        };
                        let h_dev = h_columns[packed_index(c, d)].as_ref()?;
                        let weighted_block = weighted_gemm(
                            &blas,
                            &stream,
                            x_a,
                            h_dev,
                            x_b,
                            &mut scaled_dev,
                            n_rows,
                            range_a.len(),
                            range_b.len(),
                        )?;
                        contrib_h = contrib_h + &weighted_block;
                        let struct_block = plain_gemm(
                            &blas,
                            &stream,
                            x_a,
                            x_b,
                            n_rows,
                            range_a.len(),
                            range_b.len(),
                        )?;
                        contrib_s = contrib_s + &struct_block;
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
            raw_block_ranges: raw_block_ranges.to_vec(),
        })
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

    fn assign_block(out: &mut Array2<f64>, row_offset: usize, col_offset: usize, block: &Array2<f64>) {
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

    /// Reference CPU oracle used by the parity test.
    pub(super) fn cpu_oracle(
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
                        accumulate_block(
                            &mut gram_h,
                            x_a.view(),
                            w_col,
                            x_b.view(),
                            raw_block_ranges[a].start,
                            raw_block_ranges[b].start,
                            n_rows,
                            true,
                        );
                        accumulate_block(
                            &mut gram_struct,
                            x_a.view(),
                            w_col,
                            x_b.view(),
                            raw_block_ranges[a].start,
                            raw_block_ranges[b].start,
                            n_rows,
                            false,
                        );
                    }
                }
            }
        }
        (gram_h, gram_struct)
    }

    fn accumulate_block(
        out: &mut Array2<f64>,
        x_a: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        x_b: ArrayView2<'_, f64>,
        row_off: usize,
        col_off: usize,
        n_rows: usize,
        use_weight: bool,
    ) {
        let a_cols = x_a.ncols();
        let b_cols = x_b.ncols();
        for i in 0..a_cols {
            for j in 0..b_cols {
                let mut acc = 0.0;
                for row in 0..n_rows {
                    let weight = if use_weight { w[row] } else { 1.0 };
                    acc += x_a[[row, i]] * weight * x_b[[row, j]];
                }
                out[[row_off + i, col_off + j]] += acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_fixture() -> (Vec<Vec<Option<Array2<f64>>>>, Array2<f64>, Vec<Range<usize>>) {
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
            return;
        }
        #[cfg(target_os = "linux")]
        {
            if crate::gpu::runtime::GpuRuntime::global().is_none() {
                eprintln!(
                    "[identifiability_compile] no CUDA runtime — skipping parity check"
                );
                return;
            }
            let Some(bundle) =
                try_primary_state_gram_cuda(&channel_blocks, &h_packed, &ranges)
            else {
                eprintln!(
                    "[identifiability_compile] GPU Gram build returned None — \
                     treating as CI infra outage, not parity regression"
                );
                return;
            };
            let (cpu_h, cpu_s) = cuda_impl::cpu_oracle(&channel_blocks, &h_packed, &ranges);
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
            assert_eq!(bundle.raw_block_ranges, ranges);
        }
    }
}
