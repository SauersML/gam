use ndarray::Array1;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sanitized memoization key for rho vectors.
///
/// - Rejects NaN entries (returns `None` so callers skip caching).
/// - Canonicalizes ±0.0 to +0.0 to avoid key drift.
pub(super) fn sanitized_rho_key(rho: &Array1<f64>) -> Option<Vec<u64>> {
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

/// Small RAII helper for temporary atomic-flag overrides.
pub(super) struct AtomicFlagGuard<'a> {
    flag: &'a AtomicBool,
    prev: bool,
    ordering: Ordering,
}

impl<'a> AtomicFlagGuard<'a> {
    pub(super) fn swap(flag: &'a AtomicBool, value: bool, ordering: Ordering) -> Self {
        let prev = flag.swap(value, ordering);
        Self {
            flag,
            prev,
            ordering,
        }
    }
}

impl Drop for AtomicFlagGuard<'_> {
    fn drop(&mut self) {
        self.flag.store(self.prev, self.ordering);
    }
}
