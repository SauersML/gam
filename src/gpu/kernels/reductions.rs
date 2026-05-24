pub const DETERMINISTIC_TILE_SIZE: usize = 256;

pub fn pairwise_sum_host(values: &[f64]) -> f64 {
    if values.len() <= DETERMINISTIC_TILE_SIZE {
        values.iter().copied().sum()
    } else {
        values
            .chunks(DETERMINISTIC_TILE_SIZE)
            .map(pairwise_sum_host)
            .sum()
    }
}
