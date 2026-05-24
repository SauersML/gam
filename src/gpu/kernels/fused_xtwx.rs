#[derive(Clone, Debug)]
pub struct FusedXtWxConfig {
    pub tile_p: usize,
    pub chunk_rows: usize,
    pub signed_weights: bool,
}

impl Default for FusedXtWxConfig {
    fn default() -> Self {
        Self {
            tile_p: 32,
            chunk_rows: 32_768,
            signed_weights: true,
        }
    }
}
