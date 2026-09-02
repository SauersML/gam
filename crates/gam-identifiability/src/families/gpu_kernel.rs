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

