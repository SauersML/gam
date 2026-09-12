//! The e-BH procedure (Wang & Ramdas 2022) over a family of e-values.

/// e-BH (Wang & Ramdas 2022) at FDR level `alpha` over a family of e-values.
/// Returns the indices of the rejected hypotheses (the discoveries). Valid with
/// NO independence assumption across the e-values — the property that lets the
/// dependent pair statistics share one ledger. Sort descending, find the largest
/// `k` with the `k`-th largest e-value `≥ m/(α·k)`, reject those `k`.
pub fn ebh_reject(e_values: &[f64], alpha: f64) -> Vec<usize> {
    let m = e_values.len();
    if m == 0 || !(alpha > 0.0) {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        e_values[j]
            .partial_cmp(&e_values[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut k_star = 0usize;
    for rank in 1..=m {
        let e = e_values[order[rank - 1]];
        if e >= (m as f64) / (alpha * rank as f64) {
            k_star = rank;
        }
    }
    order.into_iter().take(k_star).collect()
}

#[cfg(test)]
mod tests {
    include!("pair_phase_tests.rs");
}
