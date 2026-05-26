use ndarray::Array1;

/// Sanitized memoization key for rho vectors.
///
/// - Rejects NaN entries (returns `None` so callers skip caching).
/// - Canonicalizes ±0.0 to +0.0 to avoid key drift.
pub(super) fn sanitized_rhokey(rho: &Array1<f64>) -> Option<Vec<u64>> {
    let mut key = Vec::with_capacity(rho.len());
    for &v in rho {
        if v.is_nan() {
            return None;
        }
        key.push(if v == 0.0 {
            0.0f64.to_bits()
        } else {
            v.to_bits()
        });
    }
    Some(key)
}
