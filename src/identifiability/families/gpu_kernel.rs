//! NVRTC source for the fused primary-state Gram kernel.
//!
//! One launch covers a single (block_a, block_b) tile pair and accumulates
//! contributions from ALL channel pairs (c, d) ∈ 0..4 × 0..4 in a single
//! pass over the rows. This collapses 16 separate cuBLAS DDGMM+DGEMM round
//! trips (and the corresponding 16 reads of each n×p channel matrix) into
//! one fused traversal of device memory.
//!
//! Per output coordinate (i, j) the kernel computes the two scalars
//!
//!   acc_h(i, j) = Σ_{c,d} Σ_r X_a^(c)[r, i] · H[r, c, d] · X_b^(d)[r, j]
//!   acc_s(i, j) = Σ_{c|c==d} Σ_r X_a^(c)[r, i] · X_b^(c)[r, j]
//!
//! `acc_s` mirrors the structural Gram and uses the identity H instead of
//! the data-adaptive H; we emit it from the same row-stream so the second
//! Gram is free relative to the cuBLAS path (which paid 16 extra DGEMMs).
//!
//! ## Layout contract (host ↔ device)
//!
//! * All design matrices are column-major with leading dimension `n_rows`,
//!   matching `to_col_major(&Array2)` and what cuBLAS DGEMM consumes.
//! * `h_packed_cm` is a column-major `n_rows × 10` matrix containing the
//!   upper-triangular packing produced by `super::packed_index(c, d)`.
//! * `x_a_ptrs[c]` / `x_b_ptrs[d]` is the device pointer for the channel
//!   `c`/`d` slice of `block_a`/`block_b`. Missing channels are passed as
//!   nullptr; the kernel treats them as zero contributions.
//! * `gram_h_tile_cm` and `gram_s_tile_cm` are `a_cols × b_cols`
//!   column-major output buffers, overwritten (not accumulated). The host
//!   wrapper allocates one launch per block-pair.
//!
//! ## Threading
//!
//! Each block owns one `(TILE_A × TILE_B)` output sub-tile. Each thread
//! accumulates exactly one (i, j) output. We stream the row index `r` in
//! chunks of `ROW_CHUNK`; per chunk we co-load row entries into registers
//! and the 4×4 H slice into shared memory once per row.

/// Kernel symbol exported by the NVRTC module.
pub const KERNEL_NAME: &str = "fused_primary_state_gram_kernel";

/// Tile constants — must match the `#define TILE_A` / `TILE_B` in `KERNEL_SRC`.
/// The host-side launch reads these to compute the launch grid; `ROW_CHUNK` is
/// only consumed by the kernel itself, so its `#define` in `KERNEL_SRC` is the
/// single source of truth and no Rust mirror is needed.
pub const TILE_A: u32 = 16;
pub const TILE_B: u32 = 16;

/// Compile-time NVRTC source for the fused Gram kernel.
///
/// The kernel signature (in order):
///   const double* x_a0, x_a1, x_a2, x_a3,
///   const double* x_b0, x_b1, x_b2, x_b3,
///   const double* h_packed_cm,    // n_rows × 10 column-major
///   double*       gram_h_tile_cm, // a_cols × b_cols column-major
///   double*       gram_s_tile_cm, // a_cols × b_cols column-major
///   unsigned int  n_rows,
///   unsigned int  a_cols,
///   unsigned int  b_cols,
///   unsigned int  a_present_mask, // bit c set ↔ x_a{c} != nullptr
///   unsigned int  b_present_mask  // bit d set ↔ x_b{d} != nullptr
pub const KERNEL_SRC: &str = r#"
// Fused primary-state Gram kernel — NVRTC source.
// One launch per (block_a, block_b) tile pair.
//
// Layout:
//   - All design matrices column-major, leading dimension n_rows.
//   - h_packed_cm column-major n_rows × 10 (upper-triangular packing).
//   - Output tiles column-major a_cols × b_cols.
// Threading:
//   - Block = TILE_A × TILE_B threads, each owning one (i, j) output.
//   - Rows streamed in chunks of ROW_CHUNK; 4×4 H slice cached in
//     shared memory per chunk.

#include <stdint.h>

#define TILE_A 16
#define TILE_B 16
#define ROW_CHUNK 32
#define CHANNELS 4
#define PACKED_LEN 10

// packed_index(c, d) mirrors the Rust const fn in
// `crate::identifiability::families::gpu::packed_index`. Symmetric in (c, d).
__device__ __forceinline__ int packed_index_dev(int c, int d) {
    int lo = c <= d ? c : d;
    int hi = c <= d ? d : c;
    // Offsets for lo = 0,1,2,3 in upper-triangular row-major: 0, 4, 7, 9.
    int row_offset;
    switch (lo) {
        case 0: row_offset = 0; break;
        case 1: row_offset = 4; break;
        case 2: row_offset = 7; break;
        default: row_offset = 9; break;
    }
    return row_offset + (hi - lo);
}

extern "C" __global__ void fused_primary_state_gram_kernel(
    const double* __restrict__ x_a0,
    const double* __restrict__ x_a1,
    const double* __restrict__ x_a2,
    const double* __restrict__ x_a3,
    const double* __restrict__ x_b0,
    const double* __restrict__ x_b1,
    const double* __restrict__ x_b2,
    const double* __restrict__ x_b3,
    const double* __restrict__ h_packed_cm,
    double* __restrict__ gram_h_tile_cm,
    double* __restrict__ gram_s_tile_cm,
    unsigned int n_rows,
    unsigned int a_cols,
    unsigned int b_cols,
    unsigned int a_present_mask,
    unsigned int b_present_mask)
{
    const unsigned int tile_i_base = blockIdx.x * TILE_A;
    const unsigned int tile_j_base = blockIdx.y * TILE_B;
    const unsigned int ti = threadIdx.x; // 0..TILE_A
    const unsigned int tj = threadIdx.y; // 0..TILE_B
    const unsigned int i = tile_i_base + ti;
    const unsigned int j = tile_j_base + tj;
    const bool in_range = (i < a_cols) && (j < b_cols);

    // Shared cache for the 4×4 symmetric H per row in the current chunk.
    // Stored as 10 packed entries per row, then expanded into a 4×4
    // double during the row loop via packed_index_dev.
    __shared__ double h_chunk[ROW_CHUNK][PACKED_LEN];

    // Per-thread accumulators.
    double acc_h = 0.0;
    double acc_s = 0.0;

    // Resolve the four X_a / X_b pointers into a small lookup the
    // unrolled (c, d) loop below indexes into. Missing channels keep
    // their default-null entry, which we guard with the present masks.
    const double* xa_ptrs[CHANNELS];
    xa_ptrs[0] = x_a0;
    xa_ptrs[1] = x_a1;
    xa_ptrs[2] = x_a2;
    xa_ptrs[3] = x_a3;
    const double* xb_ptrs[CHANNELS];
    xb_ptrs[0] = x_b0;
    xb_ptrs[1] = x_b1;
    xb_ptrs[2] = x_b2;
    xb_ptrs[3] = x_b3;

    for (unsigned int chunk_start = 0; chunk_start < n_rows; chunk_start += ROW_CHUNK) {
        const unsigned int chunk_end =
            (chunk_start + ROW_CHUNK <= n_rows) ? (chunk_start + ROW_CHUNK)
                                                : n_rows;
        const unsigned int chunk_len = chunk_end - chunk_start;

        // Cooperatively load the packed H slice for this chunk. Each
        // thread loads ceil(chunk_len * PACKED_LEN / blockDim) entries.
        const unsigned int n_entries = chunk_len * (unsigned int)PACKED_LEN;
        const unsigned int n_threads = TILE_A * TILE_B;
        const unsigned int my_linear = ti + tj * TILE_A;
        for (unsigned int e = my_linear; e < n_entries; e += n_threads) {
            const unsigned int local_row = e / (unsigned int)PACKED_LEN;
            const unsigned int packed_col = e % (unsigned int)PACKED_LEN;
            const unsigned int global_row = chunk_start + local_row;
            // h_packed_cm is n_rows × 10 column-major; column `packed_col`
            // starts at offset `packed_col * n_rows`.
            h_chunk[local_row][packed_col] =
                h_packed_cm[(size_t)packed_col * (size_t)n_rows + (size_t)global_row];
        }
        __syncthreads();

        if (in_range) {
            for (unsigned int local_row = 0; local_row < chunk_len; ++local_row) {
                const unsigned int r = chunk_start + local_row;

                // Load per-row entries from each present channel column.
                // For column-major design matrices, X^(c)[r, i] sits at
                // offset i * n_rows + r.
                double xa[CHANNELS];
                double xb[CHANNELS];
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    if ((a_present_mask >> c) & 1u) {
                        xa[c] = xa_ptrs[c][(size_t)i * (size_t)n_rows + (size_t)r];
                    } else {
                        xa[c] = 0.0;
                    }
                }
                #pragma unroll
                for (int d = 0; d < CHANNELS; ++d) {
                    if ((b_present_mask >> d) & 1u) {
                        xb[d] = xb_ptrs[d][(size_t)j * (size_t)n_rows + (size_t)r];
                    } else {
                        xb[d] = 0.0;
                    }
                }

                // 16-term fused accumulation. Structural-Gram uses the
                // identity H, which keeps only the diagonal channel
                // pairs (c == d).
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    double xa_c = xa[c];
                    #pragma unroll
                    for (int d = 0; d < CHANNELS; ++d) {
                        double xb_d = xb[d];
                        double prod = xa_c * xb_d;
                        double h_cd = h_chunk[local_row][packed_index_dev(c, d)];
                        acc_h += h_cd * prod;
                        if (c == d) {
                            acc_s += prod;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (in_range) {
        // Column-major writeback: tile entry (i, j) at offset j * a_cols + i.
        const size_t off = (size_t)j * (size_t)a_cols + (size_t)i;
        gram_h_tile_cm[off] = acc_h;
        gram_s_tile_cm[off] = acc_s;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_src_declares_expected_symbol() {
        assert!(KERNEL_SRC.contains(KERNEL_NAME));
        assert!(KERNEL_SRC.contains("extern \"C\" __global__"));
        assert!(KERNEL_SRC.contains("packed_index_dev"));
    }
}
