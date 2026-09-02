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

use std::ops::Range;
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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

}
