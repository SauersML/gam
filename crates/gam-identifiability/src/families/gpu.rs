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

