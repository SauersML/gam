//! Shared helpers for the large-`K` scaling examples (`scale_k`, `tiered_gpu_scale`,
//! `tiered_k2000_measure`). Cargo does not build this directory as an example target
//! because it has no `main.rs`; each example pulls it in with `mod common;`.

/// splitmix64 mixing step — the self-contained deterministic RNG the scaling
/// examples use to synthesise seeds/coordinates without pulling in an RNG crate.
/// This is the same mixing function the library seeds its coordinate-partition
/// dictionary with (`sparse_dict::coordinate_partition_frames`); kept here as one
/// example-side copy instead of a paste in every scaling example.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
