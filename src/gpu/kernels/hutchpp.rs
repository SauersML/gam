pub const DEFAULT_RADEMACHER_SEED: u64 = 0xCAFE_BABE;

pub fn sketch_dim(dim: usize) -> usize {
    (dim / 32).clamp(4, 16)
}
