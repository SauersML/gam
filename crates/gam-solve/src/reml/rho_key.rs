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

/// Cache identity for an evaluation whose inner solve is controlled by the
/// screening and outer iteration caps.
///
/// The cap suffix is part of the mathematical input: a partial mode computed
/// under a search cap must not alias the uncapped stationary mode at identical
/// rho.  Keeping the construction here makes bundle, PIRLS, and outer-eval
/// caches share one identity convention.
pub(super) fn sanitized_eval_state_key(
    rho: &Array1<f64>,
    screening_cap: usize,
    outer_cap: usize,
) -> Option<Vec<u64>> {
    let mut key = sanitized_rhokey(rho)?;
    key.push(screening_cap as u64);
    key.push(outer_cap as u64);
    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn evaluation_cache_identity_includes_inner_fidelity_2309() {
        let rho = array![0.25, -1.5];
        let capped = sanitized_eval_state_key(&rho, 0, 3).expect("finite capped key");
        let finalized = sanitized_eval_state_key(&rho, 0, 0).expect("finite terminal key");
        let screened = sanitized_eval_state_key(&rho, 3, 0).expect("finite screening key");

        assert_ne!(capped, finalized);
        assert_ne!(screened, finalized);
        assert_eq!(&capped[..rho.len()], &finalized[..rho.len()]);
    }
}
