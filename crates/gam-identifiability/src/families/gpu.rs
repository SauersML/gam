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
//! Runtime absence is represented as `Ok(None)`. Runtime-probe and admitted
//! execution faults are preserved as [`gam_gpu::gpu_error::GpuError`] instead
//! of being collapsed into an apparent absence.

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

#[cfg(target_os = "linux")]
mod cuda_impl {
    use super::{CHANNELS, GramBundle, PACKED_LEN, packed_index};
    use cudarc::cublas::sys::{cublasOperation_t, cublasSideMode_t, cublasStatus_t};
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use gam_gpu::device_runtime::{GpuRuntime, cuda_context_for};
    use gam_gpu::driver::{to_col_major, to_i32};
    use ndarray::Array2;
    use std::cell::RefCell;
    use std::ops::Range;
    use std::sync::Arc;

    /// Process-wide PTX cache for the fused primary-state Gram kernel.
    /// NVRTC-compiled on the first `compute_grams_fused` call that
    /// reaches a CUDA context; reused on every subsequent call.
    pub(super) static FUSED_GRAM_PTX_CACHE: gam_gpu::device_cache::PtxModuleCache =
        gam_gpu::device_cache::PtxModuleCache::new();

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
        device_ordinal: usize,
        n_rows: usize,
        total_cols: usize,
    }

    impl WorkspaceInner {
        pub(super) fn try_new(
            runtime: &GpuRuntime,
            channel_blocks: &[Vec<Option<Array2<f64>>>],
            raw_block_ranges: &[Range<usize>],
        ) -> Option<Self> {
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
                device_ordinal: runtime.device.ordinal,
                n_rows,
                total_cols,
            })
        }

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
    fn primary_state_cpu_oracle_is_symmetric_and_nontrivial() {
        let (channel_blocks, h_packed, ranges) = make_fixture();
        let (cpu_h, cpu_s) = cpu_oracle(&channel_blocks, &h_packed, &ranges);
        assert!(cpu_h.iter().any(|value| value.abs() > 0.0));
        assert!(cpu_s.iter().any(|value| value.abs() > 0.0));
        for row in 0..cpu_h.nrows() {
            for col in 0..cpu_h.ncols() {
                assert!((cpu_h[[row, col]] - cpu_h[[col, row]]).abs() <= 1e-12);
                assert!((cpu_s[[row, col]] - cpu_s[[col, row]]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn primary_state_gram_matches_cpu_oracle_when_cuda_available() {
        let (channel_blocks, h_packed, ranges) = make_fixture();
        #[cfg(not(target_os = "linux"))]
        {
            assert!(
                try_primary_state_gram_cuda(&channel_blocks, &h_packed, &ranges)
                    .expect("non-Linux CUDA resolution must not fail")
                    .is_none(),
                "non-Linux build must report no CUDA"
            );
            return;
        }
        #[cfg(target_os = "linux")]
        {
            // #2422: the bare `resolve` announced an absent device with a wording
            // unique to this file and then returned before the first assertion,
            // so on every CPU-only runner this test printed `ok` having verified
            // nothing and left no trace a ledger could scrape. The shared gate
            // panics on a driver FAULT, panics under `GpuPolicy::Required`,
            // counts the genuine absence and prints the one greppable marker;
            // `assert_absent_device_was_counted` makes that count the assertion
            // this skip path executes.
            let skips_before = gam_gpu::test_gate::skipped_for_absent_device();
            let runtime = match gam_gpu::test_gate::gpu_for_test("identifiability Gram parity") {
                gam_gpu::test_gate::GpuTestGate::Ready(runtime) => runtime,
                gam_gpu::test_gate::GpuTestGate::AbsentDevice => {
                    gam_gpu::test_gate::assert_absent_device_was_counted(skips_before);
                    return;
                }
            };
            let bundle =
                primary_state_gram_cuda_with_runtime(runtime, &channel_blocks, &h_packed, &ranges)
                    .expect("CUDA primary-state Gram dispatch must succeed after Auto admission");
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
